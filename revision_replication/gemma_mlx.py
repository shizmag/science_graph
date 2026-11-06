"""Lazy Gemma MLX adapter. Importing/constructing it does not load weights."""
from __future__ import annotations

from pathlib import Path

from revision_replication.entropy import topk_telemetry, summarize_token_telemetry
from revision_replication.provenance import GEMMA_PATH

AUTHORIZATION = "AUTHORIZE_LOCAL_MODEL_GENERATION"


class GemmaMLXAdapter:
    requires_real_execution_authorization = True

    def __init__(self, model_path: Path = GEMMA_PATH, *, context_limit: int = 12000, loader=None, streamer=None, sampler_factory=None):
        if context_limit < 1:
            raise ValueError("context_limit must be positive")
        self.model_path = Path(model_path)
        self.context_limit = context_limit
        self._loader = loader
        self._streamer = streamer
        self._sampler_factory = sampler_factory
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is None:
            if self._loader is None:
                from mlx_lm import load
                self._loader = load
            self.model, self.tokenizer = self._loader(str(self.model_path))
        return self.model, self.tokenizer

    def generate(self, prompt: str, *, max_tokens: int = 768, authorization: str | None = None) -> dict:
        if authorization != AUTHORIZATION:
            raise PermissionError("real local generation requires explicit authorization")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        try:
            model, tokenizer = self._load()
            if tokenizer.chat_template:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )
            else:
                raise ValueError("Gemma tokenizer must supply its configured chat template")
            input_ids = tokenizer.encode(rendered, add_special_tokens=False)
            if len(input_ids) + max_tokens > self.context_limit:
                raise OverflowError(f"prompt_too_long:{len(input_ids)}+{max_tokens}>{self.context_limit}")
            if self._streamer is None:
                from mlx_lm import stream_generate
                self._streamer = stream_generate
            if self._sampler_factory is None:
                from mlx_lm.sample_utils import make_sampler
                sampler = make_sampler(temp=0.0)
            else:
                sampler = self._sampler_factory()
            tokens, text, telemetry = [], [], []
            stopped = False
            eos_ids = getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id]
            for response in self._streamer(model=model, tokenizer=tokenizer, prompt=rendered, max_tokens=max_tokens, sampler=sampler):
                token_id = int(response.token)
                logits = getattr(response, "logprobs", None)
                if logits is None:
                    raise RuntimeError("MLX stream did not expose full pre-sampling token log-probabilities")
                tokens.append(token_id)
                text.append(getattr(response, "text", ""))
                logit_values = logits.tolist() if hasattr(logits, "tolist") else list(logits)
                telemetry.append(topk_telemetry(logit_values, token_id))
                if token_id in set(eos_ids):
                    stopped = True
                    break
            summary = summarize_token_telemetry(telemetry)
            return {"status": "success", "text": "".join(text), "token_ids": tokens, "token_telemetry": telemetry,
                    "stop_reason": "eos" if stopped else "max_tokens_or_stream_end", **summary}
        except OverflowError as exc:
            return {"status": "context_overflow", "error": str(exc), "text": "", "token_ids": [], "token_telemetry": []}
        except Exception as exc:
            return {"status": "model_error", "error_type": type(exc).__name__, "error": str(exc), "text": "", "token_ids": [], "token_telemetry": []}
