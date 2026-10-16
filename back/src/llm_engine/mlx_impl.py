"""
MLX LLM Engine implementation for Apple Silicon devices.
"""

import os
from pathlib import Path
from typing import Optional, Type, List, Dict, Any, Tuple

from pydantic import BaseModel
try:
    import mlx.core as mx
except ImportError:
    mx = None

from src.config import config
from src import console as con
from src.llm_engine.base import BaseLLMEngine, strip_thinking_tokens, _local_request_lock


def build_mlx_tokenizer_data(tokenizer) -> Any:
    """Builds TokenEnforcerTokenizerData for mlx-lm tokenization wrapper."""
    from lmformatenforcer import TokenEnforcerTokenizerData
    
    hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
    vocab_size = len(hf_tokenizer)
    
    all_special_ids = set(getattr(hf_tokenizer, "all_special_ids", []))
    eos_token_id = getattr(hf_tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = []
    elif isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    else:
        eos_token_id = list(eos_token_id)
        
    try:
        token_0 = hf_tokenizer.encode("0", add_special_tokens=False)[-1]
    except Exception:
        token_0 = hf_tokenizer.encode("0")[-1]
        
    regular_tokens = []
    for token_idx in range(vocab_size):
        if token_idx in all_special_ids:
            continue
        try:
            decoded_after_0 = hf_tokenizer.decode([token_0, token_idx])[1:]
            decoded_regular = hf_tokenizer.decode([token_idx])
            is_word_start_token = len(decoded_after_0) > len(decoded_regular)
            regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))
        except Exception:
            continue
            
    def decode_fn(tokens: List[int]) -> str:
        return hf_tokenizer.decode(tokens).rstrip('')
        
    return TokenEnforcerTokenizerData(
        regular_tokens=regular_tokens,
        decoder=decode_fn,
        eos_token_id=eos_token_id,
        use_bitmask=False,
        vocab_size=vocab_size
    )


class ConstrainedLogitsProcessor:
    """Logits processor that filters tokens to match a given TokenEnforcer schema."""
    def __init__(self, token_enforcer: Any):
        self.token_enforcer = token_enforcer
        self.generated_tokens = []

    def __call__(self, tokens: Any, logits: Any) -> Any:
        num_tokens = tokens.shape[0]
        current_len = len(self.generated_tokens)
        
        if num_tokens > current_len + 1:
            for idx in range(current_len + 1, num_tokens):
                self.generated_tokens.append(tokens[idx].item())
                
        allowed_tokens = self.token_enforcer.get_allowed_tokens(self.generated_tokens).allowed_tokens
        
        if not allowed_tokens:
            return logits
            
        # Create logits mask directly in MLX to avoid CPU allocations and copies
        vocab_size = logits.shape[-1]
        mask = mx.full((vocab_size,), float("-inf"), dtype=mx.float32)
        mask[mx.array(allowed_tokens)] = 0.0
        
        return logits + mask



def _decode_token_key(tokenizer: Any, token_id: int) -> str:
    """Decode a token id for top_logprobs keys; fall back to str(id)."""
    if tokenizer is None:
        return str(token_id)
    try:
        text = tokenizer.decode([int(token_id)])
        if text is not None and text != "":
            return text
    except Exception:
        pass
    return str(int(token_id))


def _telemetry_defaults() -> Dict[str, Any]:
    defaults = {
        "logprob": 0.0,
        "entropy": 0.0,
        "msp": 0.0,
        "logit_margin": 0.0,
        "top1_prob_full": 0.0,
        "nll_nats": 0.0,
        "nll_bits_per_token": 0.0,
        "top_logprobs": {},
    }
    for k in (20, 50, 100):
        defaults[f"entropy_k{k}_bits"] = 0.0
        defaults[f"retained_mass_k{k}"] = 0.0
        defaults[f"top1_prob_local_k{k}"] = 0.0
    return defaults


def _telemetry_from_single_logprob(
    lp: Any,
    token_id: Any,
    tokenizer: Any = None,
    top_k_store: int = 5,
    top_k_entropy: int = 50,
) -> Dict[str, Any]:
    """Compute compact telemetry fields from a single 1-D logprob vector."""
    fields = _telemetry_defaults()
    if lp is None or mx is None:
        return fields
    try:
        vec = lp.astype(mx.float32).reshape(-1)
        V = int(vec.shape[0])
        if V == 0:
            return fields

        store_k = min(int(top_k_store), V)
        if token_id is not None:
            tid = int(token_id)
            if 0 <= tid < V:
                fields["logprob"] = float(vec[tid].item())

        if store_k < V:
            top_indices = mx.argpartition(vec, -store_k)[-store_k:]
        else:
            top_indices = mx.arange(V)
        top_lp = vec[top_indices]
        mx.eval(top_lp, top_indices)

        idx_list = top_indices.tolist()
        val_list = top_lp.tolist()
        if not isinstance(idx_list, list):
            idx_list = [idx_list]
            val_list = [val_list]
        pairs = sorted(zip(val_list, idx_list), key=lambda x: float(x[0]), reverse=True)

        top_logprobs: Dict[str, float] = {}
        for val, idx in pairs[:store_k]:
            top_logprobs[_decode_token_key(tokenizer, int(idx))] = float(val)
        fields["top_logprobs"] = top_logprobs

        if len(pairs) >= 2:
            fields["logit_margin"] = float(pairs[0][0]) - float(pairs[1][0])
        else:
            fields["logit_margin"] = 0.0

        p_store = mx.softmax(top_lp)
        fields["msp"] = float(mx.max(p_store).item())

        log_z = mx.logsumexp(vec)
        fields["top1_prob_full"] = float(mx.exp(pairs[0][0] - log_z).item())
        for requested_k in sorted({20, 50, 100, int(top_k_entropy)}):
            k = min(requested_k, V)
            if k < V:
                indices = mx.argpartition(vec, -k)[-k:]
                values = vec[indices]
            else:
                values = vec
            probabilities = mx.softmax(values)
            entropy = -mx.sum(probabilities * mx.log2(probabilities + 1e-12))
            mass = mx.sum(mx.exp(values - log_z))
            suffix = str(requested_k)
            fields[f"entropy_k{suffix}_bits"] = max(0.0, float(entropy.item()))
            fields[f"retained_mass_k{suffix}"] = min(1.0, max(0.0, float(mass.item())))
            fields[f"top1_prob_local_k{suffix}"] = float(mx.max(probabilities).item())
        fields["entropy"] = fields.get(f"entropy_k{int(top_k_entropy)}_bits", 0.0)
        fields["nll_nats"] = -fields["logprob"]
        fields["nll_bits_per_token"] = fields["nll_nats"] / 0.6931471805599453
        return fields
    except Exception:
        return fields


def build_compact_tokens_info(
    raw_logprobs_list: List[Any],
    tokens_meta: List[Dict[str, Any]],
    tokenizer: Any = None,
    top_k_store: int = 5,
    top_k_entropy: int = 50,
) -> List[Dict[str, Any]]:
    """Attach compact logprob telemetry to token metadata without storing full vocab.

    Each output dict includes: token_id, token_text, char_start, char_end,
    logprob, entropy, msp, logit_margin, top_logprobs (k=top_k_store).
    """
    out: List[Dict[str, Any]] = []
    for meta in tokens_meta:
        item = dict(meta)
        item.update(_telemetry_defaults())
        out.append(item)

    if not raw_logprobs_list or mx is None:
        return out

    valid_idx = [i for i, lp in enumerate(raw_logprobs_list) if lp is not None and i < len(out)]
    if not valid_idx:
        return out

    try:
        flats = [raw_logprobs_list[i].astype(mx.float32).reshape(-1) for i in valid_idx]
        stacked = mx.stack(flats)  # (N, V)
        N = int(stacked.shape[0])
        V = int(stacked.shape[1])
        if N == 0 or V == 0:
            return out

        store_k = min(int(top_k_store), V)
        if store_k < V:
            top_idx = mx.argpartition(stacked, -store_k, axis=-1)[:, -store_k:]
        else:
            top_idx = mx.broadcast_to(mx.arange(V)[None, :], (N, V))

        top_lp = mx.take_along_axis(stacked, top_idx, axis=-1)
        sort_order = mx.argsort(-top_lp, axis=-1)
        top_lp_sorted = mx.take_along_axis(top_lp, sort_order, axis=-1)
        top_idx_sorted = mx.take_along_axis(top_idx, sort_order, axis=-1)

        p_store = mx.softmax(top_lp, axis=-1)
        msp_arr = mx.max(p_store, axis=-1)
        if store_k >= 2:
            margin_arr = top_lp_sorted[:, 0] - top_lp_sorted[:, 1]
        else:
            margin_arr = mx.zeros((N,), dtype=mx.float32)

        log_z = mx.logsumexp(stacked, axis=-1, keepdims=True)
        full_top1_prob = mx.exp(top_lp_sorted[:, :1] - log_z).reshape(-1)
        topk_arrays = {}
        for requested_k in sorted({20, 50, 100, int(top_k_entropy)}):
            k = min(requested_k, V)
            if k < V:
                indices = mx.argpartition(stacked, -k, axis=-1)[:, -k:]
                values = mx.take_along_axis(stacked, indices, axis=-1)
            else:
                values = stacked
            probabilities = mx.softmax(values, axis=-1)
            entropy = -mx.sum(probabilities * mx.log2(probabilities + 1e-12), axis=-1)
            mass = mx.sum(mx.exp(values - log_z), axis=-1)
            local_top1 = mx.max(probabilities, axis=-1)
            topk_arrays[requested_k] = (entropy, mass, local_top1)
        ent_arr = topk_arrays[int(top_k_entropy)][0]

        tid_list: List[int] = []
        tid_valid: List[bool] = []
        for i in valid_idx:
            tid = tokens_meta[i].get("token_id") if i < len(tokens_meta) else None
            if tid is not None:
                ti = int(tid)
                if 0 <= ti < V:
                    tid_list.append(ti)
                    tid_valid.append(True)
                    continue
            tid_list.append(0)
            tid_valid.append(False)
        tid_arr = mx.array(tid_list)
        chosen = stacked[mx.arange(N), tid_arr]

        mx.eval(msp_arr, margin_arr, ent_arr, chosen, top_lp_sorted, top_idx_sorted, full_top1_prob)

        msp_list = msp_arr.tolist()
        margin_list = margin_arr.tolist()
        ent_list = ent_arr.tolist()
        chosen_list = chosen.tolist()
        full_top1_prob_list = full_top1_prob.tolist()
        top_lp_list = top_lp_sorted.tolist()
        top_idx_list = top_idx_sorted.tolist()
        topk_values = {}
        for requested_k, (entropy, mass, local_top1) in topk_arrays.items():
            mx.eval(entropy, mass, local_top1)
            topk_values[requested_k] = (entropy.tolist(), mass.tolist(), local_top1.tolist())

        for j, i in enumerate(valid_idx):
            out[i]["logprob"] = float(chosen_list[j]) if tid_valid[j] else 0.0
            out[i]["entropy"] = max(0.0, float(ent_list[j]))
            out[i]["msp"] = float(msp_list[j])
            out[i]["logit_margin"] = float(margin_list[j])
            out[i]["top1_prob_full"] = float(full_top1_prob_list[j])
            out[i]["nll_nats"] = -out[i]["logprob"]
            out[i]["nll_bits_per_token"] = out[i]["nll_nats"] / 0.6931471805599453
            for requested_k, (entropies, masses, local_top1s) in topk_values.items():
                suffix = str(requested_k)
                out[i][f"entropy_k{suffix}_bits"] = max(0.0, float(entropies[j]))
                out[i][f"retained_mass_k{suffix}"] = min(1.0, max(0.0, float(masses[j])))
                out[i][f"top1_prob_local_k{suffix}"] = float(local_top1s[j])

            row_lps = top_lp_list[j]
            row_ids = top_idx_list[j]
            if not isinstance(row_lps, list):
                row_lps = [row_lps]
                row_ids = [row_ids]
            top_dict: Dict[str, float] = {}
            for val, idx in zip(row_lps, row_ids):
                top_dict[_decode_token_key(tokenizer, int(idx))] = float(val)
            out[i]["top_logprobs"] = top_dict
        return out
    except Exception:
        for i in valid_idx:
            tid = tokens_meta[i].get("token_id") if i < len(tokens_meta) else None
            fields = _telemetry_from_single_logprob(
                raw_logprobs_list[i],
                tid,
                tokenizer=tokenizer,
                top_k_store=top_k_store,
                top_k_entropy=top_k_entropy,
            )
            out[i].update(fields)
        return out


def _clear_mlx_cache() -> None:
    """Best-effort GC + MLX/Metal cache purge after discarding heavy logprob tensors."""
    import gc
    gc.collect()
    if mx is None:
        return
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


class MlxLLMEngine(BaseLLMEngine):
    def __init__(self, model_path: str = None):
        if mx is None:
            raise ImportError(
                "MLX is not installed. MlxLLMEngine is only supported on Apple Silicon macOS with the 'mlx' package installed."
            )
        self.model_path = model_path or config.llm_local_model_path
        self._tokenizer_data = None
        self.model = None
        self.tokenizer = None
        self._sampler_cache = {}

        con.debug(f"MLX_INIT pid={os.getpid()} self_id={id(self)} model_path={self.model_path}")


        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"Local MLX model path not found: {self.model_path}\n"
                f"  Run: python3 main.py config  to see configured paths."
            )

        # Detect OptiQ quantized models
        optiq_meta = Path(self.model_path) / "optiq_metadata.json"
        if optiq_meta.exists():
            raise ValueError(
                f"Directory '{self.model_path}' is an OptiQ-quantized model.\n"
                "OptiQ models cannot be loaded directly in-process via provider='mlx'.\n"
                "To use this model, please run standard OptiQ server in your terminal:\n"
                "   optiq serve --model " + self.model_path + " --mtp\n"
                "And set your configuration in config.yaml:\n"
                "   llm:\n"
                "     provider: openai-compatible\n"
                "     local:\n"
                "       base_url: http://localhost:8080/v1"
            )

        # Force-import mlx_lm.generate on the current (main) thread so that
        # its module-level `generation_stream = mx.new_stream(...)` is bound
        # to the main thread.  Without this, the Marker PDF parser can trigger
        # the first import inside an asyncio.to_thread worker, which creates
        # the stream on a short-lived thread; subsequent generate() calls on
        # the main thread then crash with "There is no Stream(gpu, N) in
        # current thread" (MLX 0.31+ enforces thread-local streams).
        import mlx_lm.generate  # noqa: F401

    def _get_sampler(self, temp: float):
        temp_key = round(float(temp), 4)
        if temp_key not in self._sampler_cache:
            from mlx_lm.sample_utils import make_sampler
            self._sampler_cache[temp_key] = make_sampler(temp=temp_key)
        return self._sampler_cache[temp_key]

    def unload_model(self):
        con.debug(f"MLX_UNLOAD pid={os.getpid()} self_id={id(self)} model_path={self.model_path} model_was_none={self.model is None}")

        self._sampler_cache.clear()
        if self.model is not None:
            import gc
            self.model = None
            self.tokenizer = None
            self._tokenizer_data = None
            gc.collect()
            try:
                import mlx.core as mx
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except ImportError:
                pass
            con.success("MLX LLM model unloaded and GPU cache cleared")

    def _ensure_model_loaded(self):
        con.debug(f"MLX_ENSURE_BEFORE pid={os.getpid()} self_id={id(self)} model_path={self.model_path} model_is_none={self.model is None} tokenizer_is_none={self.tokenizer is None}")

        
        if self.model is None:
            model_name = Path(self.model_path).name
            con.model_msg(f"Loading MLX LLM [bold]{model_name}[/bold] …")

            con.debug(f"MLX_REAL_LOAD_START pid={os.getpid()} self_id={id(self)} model_path={self.model_path}")

            # Try to import optiq or mlx_optiq to register custom models at runtime
            optiq_loaded = False
            try:
                import optiq
                optiq_loaded = True
            except ImportError:
                try:
                    import mlx_optiq
                    optiq_loaded = True
                except ImportError:
                    pass

            # Detect Qwen3/3.5/OptiQ models to prevent incorrect fallback and raise a helpful error
            import json
            model_type = None
            config_json_path = os.path.join(self.model_path, "config.json")
            if os.path.isfile(config_json_path):
                try:
                    with open(config_json_path, "r", encoding="utf-8") as f:
                        model_cfg = json.load(f)
                        model_type = model_cfg.get("model_type")
                except Exception:
                    pass

            is_optiq_or_mtp = False
            # Check for optiq_metadata.json
            if os.path.exists(os.path.join(self.model_path, "optiq_metadata.json")):
                is_optiq_or_mtp = True
            # Check for mtp.safetensors
            if os.path.exists(os.path.join(self.model_path, "mtp.safetensors")):
                is_optiq_or_mtp = True

            if model_type:
                model_type_lower = str(model_type).lower()
                if "optiq" in model_type_lower or "mtp" in model_type_lower:
                    is_optiq_or_mtp = True
            
            model_path_lower = os.path.basename(self.model_path).lower()
            if "optiq" in model_path_lower or "mtp" in model_path_lower:
                is_optiq_or_mtp = True

            if is_optiq_or_mtp and not optiq_loaded:
                raise ImportError(
                    f"Model type '{model_type or 'Qwen3.5/OptiQ'}' requires the 'mlx-optiq' package to be installed to run locally.\n"
                    f"Please run 'pip install mlx-optiq' to run this model in-process, or configure provider='openai-compatible' "
                    f"in config.yaml and start 'optiq serve --model {self.model_path}'."
                )

            if not optiq_loaded:
                try:
                    import mlx_lm.utils
                    for qwen_type in ["qwen3", "qwen3_5"]:
                        if qwen_type not in mlx_lm.utils.MODEL_REMAPPING:
                            try:
                                import importlib
                                importlib.import_module(f"mlx_lm.models.{qwen_type}")
                            except ImportError:
                                # For non-OptiQ/non-MTP qwen3 models, we fall back to remapping to qwen2
                                # if the installed mlx_lm version doesn't support them natively.
                                mlx_lm.utils.MODEL_REMAPPING[qwen_type] = "qwen2"
                except Exception:
                    pass

            from mlx_lm import load
            with con.suppress_stderr(), con.suppress_stdout():
                self.model, self.tokenizer = load(self.model_path, tokenizer_config={"fix_mistral_regex": True})

            con.debug(f"MLX_REAL_LOAD_DONE pid={os.getpid()} self_id={id(self)} model_path={self.model_path}")


            con.success(f"MLX LLM ready: [bold]{model_name}[/bold]")

    def count_tokens(self, text: str) -> int:
        self._ensure_model_loaded()
        return super().count_tokens(text)

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: Optional[str] = None) -> str:
        self._ensure_model_loaded()
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            if task == "extraction":
                resolved_max_tokens = config.llm_extraction_output_limit
            elif task == "clustering":
                resolved_max_tokens = config.llm_clustering_output_limit
            elif task == "synthesis":
                resolved_max_tokens = config.llm_synthesis_output_limit
            
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp

        formatted_prompt = prompt
        is_formatted = any(
            tag in prompt
            for tag in [
                "<|im_start|>",
                "<|start_header_id|>",
                "[INST]",
                "<start_of_turn>",
                "<|im_end|>"
            ]
        )

        if not is_formatted and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        from mlx_lm import generate
        sampler = self._get_sampler(temp)

        with _local_request_lock:
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=resolved_max_tokens,
                sampler=sampler,
                verbose=False,
            )
        return strip_thinking_tokens(response)

    def generate_response_with_logits(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temp: Optional[float] = None,
        task: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self._ensure_model_loaded()
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            if task == "extraction":
                resolved_max_tokens = config.llm_extraction_output_limit
            elif task == "clustering":
                resolved_max_tokens = config.llm_clustering_output_limit
            elif task == "synthesis":
                resolved_max_tokens = config.llm_synthesis_output_limit
            
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp

        formatted_prompt = prompt
        is_formatted = any(
            tag in prompt
            for tag in [
                "<|im_start|>",
                "<|start_header_id|>",
                "[INST]",
                "<start_of_turn>",
                "<|im_end|>"
            ]
        )

        if not is_formatted and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        from mlx_lm import stream_generate
        sampler = self._get_sampler(temp)

        raw_logprobs_list = []
        tokens_meta = []
        full_text = ""

        with _local_request_lock:
            for response in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=resolved_max_tokens,
                sampler=sampler,
            ):
                token_id = getattr(response, "token", None)
                token_text = getattr(response, "text", "")
                logprobs = getattr(response, "logprobs", None)

                char_start = len(full_text)
                full_text += token_text
                char_end = len(full_text)

                tokens_meta.append({
                    "token_id": token_id,
                    "token_text": token_text,
                    "char_start": char_start,
                    "char_end": char_end,
                })
                raw_logprobs_list.append(logprobs)

        # Compact per-token telemetry (logprob/msp/margin/top-k/entropy) then drop full vocab tensors
        tokens_info = build_compact_tokens_info(
            raw_logprobs_list,
            tokens_meta,
            tokenizer=self.tokenizer,
            top_k_store=5,
            top_k_entropy=50,
        )

        del raw_logprobs_list
        _clear_mlx_cache()

        clean_text = strip_thinking_tokens(full_text)
        try:
            from core.shannon_estimator import align_tokens_info
            aligned_tokens = align_tokens_info(full_text, clean_text, tokens_info)
        except Exception:
            aligned_tokens = tokens_info

        return clean_text, aligned_tokens


    def _apply_chat_template(self, prompt: str) -> str:
        """Format a user prompt with the tokenizer chat template when not already formatted."""
        formatted_prompt = prompt
        is_formatted = any(
            tag in prompt
            for tag in [
                "<|im_start|>",
                "<|start_header_id|>",
                "[INST]",
                "<start_of_turn>",
                "<|im_end|>",
            ]
        )
        if not is_formatted and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return formatted_prompt

    def _encode_text(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids, tolerating tokenizer API differences."""
        try:
            ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            ids = self.tokenizer.encode(text)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [int(x) for x in list(ids)]

    def score_text_logprobs(self, prompt: str, answer_text: str) -> List[Dict[str, Any]]:
        """Teacher-force score answer_text under prompt; return compact tokens_info.

        Formats prompt like generate_response_with_logits, concatenates answer tokens,
        runs a full forward pass, and extracts logprobs for each answer token given
        previous context. Char spans are over the reconstructed answer token text.
        """
        self._ensure_model_loaded()
        if answer_text is None:
            return []
        answer_text = str(answer_text)
        if not answer_text:
            return []

        formatted_prompt = self._apply_chat_template(prompt)

        bos = getattr(self.tokenizer, "bos_token", None)
        add_special = bos is None or not str(formatted_prompt).startswith(str(bos))
        prompt_ids = self._encode_text(formatted_prompt, add_special_tokens=add_special)
        answer_ids = self._encode_text(answer_text, add_special_tokens=False)
        if not answer_ids:
            return []
        if not prompt_ids:
            # Need at least one context token for causal LM scoring positions
            return []

        full_ids = prompt_ids + answer_ids
        prompt_len = len(prompt_ids)
        answer_len = len(answer_ids)

        with _local_request_lock:
            input_arr = mx.array(full_ids)[None]
            logits = self.model(input_arr)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            # logits[t] predicts token at position t+1; answer token j uses index prompt_len+j-1
            start = prompt_len - 1
            end = prompt_len + answer_len - 1
            answer_logits = logits[0, start:end, :].astype(mx.float32)
            answer_logprobs = answer_logits - mx.logsumexp(answer_logits, axis=-1, keepdims=True)
            mx.eval(answer_logprobs)

        # Differential decode for token_text + char spans over reconstructed answer
        tokens_meta: List[Dict[str, Any]] = []
        raw_logprobs_list: List[Any] = []
        full_text = ""
        for j, tid in enumerate(answer_ids):
            try:
                if j == 0:
                    token_text = self.tokenizer.decode([tid])
                else:
                    prev = self.tokenizer.decode(answer_ids[:j])
                    curr = self.tokenizer.decode(answer_ids[: j + 1])
                    if curr.startswith(prev):
                        token_text = curr[len(prev) :]
                    else:
                        token_text = self.tokenizer.decode([tid])
            except Exception:
                token_text = ""

            char_start = len(full_text)
            full_text += token_text
            char_end = len(full_text)
            tokens_meta.append(
                {
                    "token_id": int(tid),
                    "token_text": token_text,
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
            raw_logprobs_list.append(answer_logprobs[j])

        tokens_info = build_compact_tokens_info(
            raw_logprobs_list,
            tokens_meta,
            tokenizer=self.tokenizer,
            top_k_store=5,
            top_k_entropy=50,
        )

        del raw_logprobs_list, answer_logprobs, answer_logits, logits
        _clear_mlx_cache()
        return tokens_info


    def generate_json(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        self._ensure_model_loaded()
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp

        if self._tokenizer_data is None:
            self._tokenizer_data = build_mlx_tokenizer_data(self.tokenizer)

        from lmformatenforcer import TokenEnforcer, JsonSchemaParser
        parser = JsonSchemaParser(schema_class.model_json_schema())
        enforcer = TokenEnforcer(self._tokenizer_data, parser)
        logits_processor = ConstrainedLogitsProcessor(enforcer)

        formatted_prompt = prompt
        is_formatted = any(
            tag in prompt
            for tag in [
                "<|im_start|>",
                "<|start_header_id|>",
                "[INST]",
                "<start_of_turn>",
                "<|im_end|>"
            ]
        )

        if not is_formatted and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a strict JSON extractor. Output ONLY valid JSON matching this schema: {schema_class.model_json_schema()}"
                    },
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        from mlx_lm import generate
        sampler = self._get_sampler(temp)

        with _local_request_lock:
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=resolved_max_tokens,
                sampler=sampler,
                verbose=False,
                logits_processors=[logits_processor],
            )
        return strip_thinking_tokens(response)

    async def generate_response_async(
        self,
        prompt: str,
        max_tokens: int = None,
        temp: float = None,
        task: str = None,
        model: Optional[str] = None,
    ) -> str:
        import asyncio
        return await asyncio.to_thread(self.generate_response, prompt, max_tokens, temp, task, model=model)

    async def generate_json_async(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        import asyncio
        return await asyncio.to_thread(self.generate_json, prompt, schema_class, temp, max_tokens)
