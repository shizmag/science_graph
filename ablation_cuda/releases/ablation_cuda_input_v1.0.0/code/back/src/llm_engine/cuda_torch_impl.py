"""PyTorch CUDA backend for the benchmark's pinned OCC-RAG checkpoint.

This backend intentionally uses the original Hugging Face BF16 safetensors
checkpoint. It does not quantize or convert model weights. Decoding is greedy
at temperature zero, matching the historical MLX benchmark setting.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from src.config import config
from src.llm_engine.base import BaseLLMEngine, strip_thinking_tokens, _local_request_lock


STOP_TOKEN_IDS = (151643, 151645, 151683)
_LN2 = 0.6931471805599453


def _token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return str(int(token_id))


def compact_logits_telemetry(logits: Any, token_id: int, tokenizer: Any,
                             top_k_values: Tuple[int, ...] = (20, 50, 100),
                             store_k: int = 5) -> Dict[str, Any]:
    """Compute one token's compact telemetry on the logits' current device.

    `logits` must be the raw, unprocessed model output, before generation
    processors. The function only retains compact scalars and selected IDs.
    """
    import torch

    vector = logits.reshape(-1).float()
    vocab_size = int(vector.numel())
    if vocab_size == 0:
        raise ValueError("Cannot compute telemetry for an empty vocabulary")
    log_z = torch.logsumexp(vector, dim=0)
    chosen_logprob = vector[int(token_id)] - log_z
    top_count = min(max(2, int(store_k)), vocab_size)
    top_values, top_indices = torch.topk(vector, k=top_count, largest=True, sorted=True)
    top_local = torch.softmax(top_values, dim=0)
    top_logprobs = {
        _token_text(tokenizer, int(index)): float((value - log_z).item())
        for value, index in zip(top_values[:store_k], top_indices[:store_k])
    }
    row: Dict[str, Any] = {
        "token_id": int(token_id),
        "token_text": _token_text(tokenizer, int(token_id)),
        "logprob": float(chosen_logprob.item()),
        "nll_nats": float((-chosen_logprob).item()),
        "nll_bits_per_token": float((-chosen_logprob / _LN2).item()),
        "logit_margin": float((top_values[0] - top_values[1]).item()) if vocab_size > 1 else 0.0,
        "top_logit_margins": {
            "top1_minus_top2": float((top_values[0] - top_values[1]).item()) if vocab_size > 1 else 0.0,
            "top1_minus_top5": float((top_values[0] - top_values[min(4, vocab_size - 1)]).item()),
        },
        "top1_prob_full": float(torch.exp(top_values[0] - log_z).item()),
        "msp": float(top_local[0].item()),
        "top_logprobs": top_logprobs,
        "telemetry_device": str(vector.device),
    }
    for requested_k in top_k_values:
        k = min(int(requested_k), vocab_size)
        values = torch.topk(vector, k=k, largest=True, sorted=False).values
        local_probabilities = torch.softmax(values, dim=0)
        entropy_bits = -(local_probabilities * torch.log2(local_probabilities.clamp_min(1e-12))).sum()
        retained_mass = torch.exp(values - log_z).sum().clamp(0.0, 1.0)
        suffix = str(requested_k)
        row[f"entropy_k{suffix}_bits"] = float(entropy_bits.item())
        row[f"retained_mass_k{suffix}"] = float(retained_mass.item())
        row[f"top1_prob_local_k{suffix}"] = float(local_probabilities.max().item())
    row["entropy"] = row.get("entropy_k50_bits", 0.0)
    row["stop_state"] = "continue"
    row["is_stop_token"] = False
    return row


class CudaTorchLLMEngine(BaseLLMEngine):
    """Original-checkpoint greedy generator with CUDA-resident per-step metrics."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or config.llm_local_rag_model_path
        self.model = None
        self.tokenizer = None
        self.events: List[Dict[str, Any]] = []
        self.device = "cuda"
        self.progress_callback = None

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_TORCH provider selected, but torch.cuda.is_available() is false")
        model_dir = Path(self.model_path).resolve(strict=True)
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Original model.safetensors missing from {model_dir}")
        if ((model_dir / "optiq_metadata.json").exists()
                or (model_dir / "mtp.safetensors").exists()):
            raise RuntimeError("Quantized model directories are not accepted by this benchmark backend")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=True,
            fix_mistral_regex=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, use_safetensors=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda")
        self.model.eval()
        generation_config = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
        checkpoint_stops = generation_config.get("eos_token_id")
        if checkpoint_stops != list(STOP_TOKEN_IDS):
            raise RuntimeError(f"Frozen OCC-RAG stop-token manifest changed: {checkpoint_stops}")
        self.stop_token_ids = tuple(int(value) for value in checkpoint_stops)
        devices = {str(parameter.device) for parameter in self.model.parameters()}
        if devices != {"cuda:0"} and not (devices and all(item.startswith("cuda:") for item in devices)):
            raise RuntimeError(f"OCC-RAG weights are not wholly resident on CUDA: {sorted(devices)}")
        if any(parameter.dtype != torch.bfloat16 for parameter in self.model.parameters()):
            raise RuntimeError("OCC-RAG checkpoint did not load in its original BF16 dtype")
        if bool(getattr(self.model.config, "use_mtp", False)):
            raise RuntimeError("OCC-RAG multi-token prediction path is not part of the frozen greedy protocol")

    def unload_model(self) -> None:
        if self.model is not None:
            import gc
            import torch
            self.model = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

    def model_device_manifest(self) -> Dict[str, Any]:
        self._ensure_model_loaded()
        import torch
        parameters = list(self.model.parameters())
        return {
            "model_device_set": sorted({str(parameter.device) for parameter in parameters}),
            "input_device": "cuda:0",
            "parameter_dtype_set": sorted({str(parameter.dtype) for parameter in parameters}),
            "cuda_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }

    def _format_prompt(self, prompt: str) -> str:
        formatted = prompt
        is_formatted = any(tag in prompt for tag in (
            "<|im_start|>", "<|start_header_id|>", "[INST]", "<start_of_turn>", "<|im_end|>"
        ))
        if not is_formatted and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                formatted = prompt
        return formatted

    def _input_ids(self, formatted_prompt: str):
        import torch
        add_special = not bool(getattr(self.tokenizer, "bos_token", None) and
                               formatted_prompt.startswith(self.tokenizer.bos_token))
        ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=add_special)
        if not ids:
            raise RuntimeError("Tokenizer produced an empty prompt")
        context_limit = int(config.llm_model_max_context or 12000)
        model_limit = int(getattr(self.model.config, "max_position_embeddings", context_limit))
        allowed = min(context_limit, model_limit)
        if len(ids) >= allowed:
            raise RuntimeError(
                f"Serialized prompt has {len(ids)} tokens; the configured context limit leaves no room for generation ({allowed})"
            )
        return torch.tensor([ids], dtype=torch.long, device="cuda"), allowed

    def _next_logits(self, input_ids: Any, past_key_values: Any = None,
                     attention_mask: Any = None):
        import torch
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        return output.logits[:, -1, :].float(), output.past_key_values

    def _generate(self, prompt: str, max_tokens: int | None, temp: float | None,
                  kind: str) -> Tuple[str, List[Dict[str, Any]]]:
        import torch
        self._ensure_model_loaded()
        requested = int(max_tokens or config.llm_max_tokens or 768)
        requested = min(requested, 768)
        temperature = config.llm_temp if temp is None else float(temp)
        formatted_prompt = self._format_prompt(prompt)
        input_ids, context_limit = self._input_ids(formatted_prompt)
        allowed = min(requested, context_limit - input_ids.shape[1])
        if allowed <= 0:
            raise RuntimeError("No output token budget remains after prompt serialization")
        started = time.perf_counter()
        attention_mask = torch.ones_like(input_ids, device="cuda")
        telemetry: List[Dict[str, Any]] = []
        generated_ids: List[int] = []
        full_prefix_text = ""
        with _local_request_lock:
            logits, cache = self._next_logits(input_ids, attention_mask=attention_mask)
            for _ in range(allowed):
                # Temperature zero is the locked benchmark protocol. Non-zero
                # temperatures are supported for application compatibility.
                next_logits = logits[0]
                if temperature <= 0.0:
                    next_id = int(torch.argmax(next_logits).item())
                else:
                    next_id = int(torch.multinomial(torch.softmax(next_logits / temperature, dim=-1), 1).item())
                row = compact_logits_telemetry(next_logits, next_id, self.tokenizer)
                generated_ids.append(next_id)
                decoded = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                token_text = decoded[len(full_prefix_text):] if decoded.startswith(full_prefix_text) else _token_text(self.tokenizer, next_id)
                row.update({
                    "token_text": token_text,
                    "token": token_text,
                    "char_start": len(full_prefix_text),
                    "char_end": len(full_prefix_text) + len(token_text),
                    "stop_state": "eos" if next_id in self.stop_token_ids else "continue",
                    "is_stop_token": next_id in self.stop_token_ids,
                })
                telemetry.append(row)
                full_prefix_text += token_text
                if callable(self.progress_callback):
                    self.progress_callback(kind=kind, token_count=len(generated_ids),
                                           elapsed_seconds=time.perf_counter() - started)
                if next_id in self.stop_token_ids:
                    break
                next_token = torch.tensor([[next_id]], dtype=torch.long, device="cuda")
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
                logits, cache = self._next_logits(next_token, past_key_values=cache,
                                                  attention_mask=attention_mask)

        if telemetry and telemetry[-1]["stop_state"] == "continue" and len(telemetry) >= allowed:
            telemetry[-1]["stop_state"] = "max_tokens" if allowed == requested else "context_limit"
        response = strip_thinking_tokens(self.tokenizer.decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ))
        torch.cuda.synchronize()
        event = {
            "kind": kind,
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "prompt_token_count": int(input_ids.shape[1]),
            "response": response,
            "token_count": len(generated_ids),
            "elapsed_seconds": time.perf_counter() - started,
            "temperature": temperature,
            "max_new_tokens": requested,
            "stop_state": telemetry[-1]["stop_state"] if telemetry else "empty",
            "input_device": str(input_ids.device),
            "logit_device": telemetry[-1].get("telemetry_device") if telemetry else None,
            "token_telemetry": telemetry,
        }
        self.events.append(event)
        return response, telemetry

    def generate_response(self, prompt: str, max_tokens: int = None,
                          temp: float = None, task: str = None,
                          model: Optional[str] = None) -> str:
        response, _ = self._generate(prompt, max_tokens, temp, "response")
        return response

    def generate_response_with_logits(self, prompt: str,
                                      max_tokens: Optional[int] = None,
                                      temp: Optional[float] = None,
                                      task: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        return self._generate(prompt, max_tokens, temp, "response_with_logits")

    def generate_json(self, prompt: str, schema_class: Type[BaseModel],
                      temp: float = 0.0, max_tokens: Optional[int] = None) -> str:
        self._ensure_model_loaded()
        formatted = prompt
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": "Output only valid JSON matching this schema: " + str(schema_class.model_json_schema())},
                     {"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                formatted = prompt
        response, _ = self._generate(formatted, max_tokens, temp, "json")
        return response

    def count_tokens(self, text: str) -> int:
        self._ensure_model_loaded()
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def score_text_logprobs(self, prompt: str, answer_text: str) -> List[Dict[str, Any]]:
        """Teacher force compact per-answer-token metrics without retaining logits."""
        import torch
        self._ensure_model_loaded()
        if not answer_text:
            return []
        formatted = self._format_prompt(prompt)
        prompt_ids, context_limit = self._input_ids(formatted)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        if prompt_ids.shape[1] + len(answer_ids) > context_limit:
            raise RuntimeError("Teacher-forced answer would exceed the frozen context limit")
        rows: List[Dict[str, Any]] = []
        prefix = ""
        attention_mask = torch.ones_like(prompt_ids, device="cuda")
        with _local_request_lock:
            logits, cache = self._next_logits(prompt_ids, attention_mask=attention_mask)
            for index, token_id in enumerate(answer_ids):
                row = compact_logits_telemetry(logits[0], int(token_id), self.tokenizer)
                decoded = self.tokenizer.decode(answer_ids[:index + 1], clean_up_tokenization_spaces=False)
                token_text = decoded[len(prefix):] if decoded.startswith(prefix) else _token_text(self.tokenizer, token_id)
                row.update({"token": token_text, "char_start": len(prefix), "char_end": len(prefix) + len(token_text),
                            "stop_state": "teacher_forced", "is_stop_token": int(token_id) in STOP_TOKEN_IDS})
                rows.append(row)
                prefix = decoded
                next_token = torch.tensor([[int(token_id)]], dtype=torch.long, device="cuda")
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
                logits, cache = self._next_logits(next_token, past_key_values=cache,
                                                  attention_mask=attention_mask)
        return rows

