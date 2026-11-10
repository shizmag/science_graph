"""Optional direct MLX adapter; importing this module does not load a model."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import platform
import sys
import time
from typing import Any

from .identity import runtime_identity
from .protocol import (
    MLX_LM_VERSION,
    MLX_VERSION,
    entropy_windows,
)
from .telemetry import (
    STOP_TOKEN_IDS,
    TELEMETRY_IMPLEMENTATION,
    TELEMETRY_PROTOCOL_VERSION,
    TELEMETRY_SOURCE,
    TokenTelemetryAccumulator,
    validate_telemetry_payload,
)


class MLXUnavailable(RuntimeError):
    """Raised when the optional local MLX dependencies are not installed."""


class MLXAdapter:
    """One resident model, sequential generation, and explicit telemetry status."""

    def __init__(self, model_path: Path, *, output_limit: int = 768):
        self.model_path = Path(model_path).expanduser().resolve()
        if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit < 0:
            raise ValueError("output_limit must be a non-negative integer")
        self.output_limit = output_limit
        self.model: Any = None
        self.tokenizer: Any = None
        self._mlx_lm: Any = None

    @staticmethod
    def package_versions() -> dict[str, str | None]:
        versions: dict[str, str | None] = {}
        for name in ("mlx", "mlx-lm"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        return versions

    @classmethod
    def environment_identity(cls) -> dict[str, Any]:
        versions = cls.package_versions()
        return runtime_identity(
            mlx_version=versions["mlx"],
            mlx_lm_version=versions["mlx-lm"],
        )

    @classmethod
    def require_pinned_environment(cls) -> dict[str, Any]:
        """Return the current identity only when the locked runtime is present."""

        versions = cls.package_versions()
        if versions != {"mlx": MLX_VERSION, "mlx-lm": MLX_LM_VERSION}:
            raise MLXUnavailable(
                "Expected pinned MLX runtime "
                f"mlx=={MLX_VERSION}, mlx-lm=={MLX_LM_VERSION}; got {versions}"
            )
        return runtime_identity(
            mlx_version=versions["mlx"],
            mlx_lm_version=versions["mlx-lm"],
        )

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import mlx_lm
        except ImportError as exc:
            raise MLXUnavailable(
                "Local MLX is unavailable; install the pinned arm64 mlx/mlx-lm environment before smoke."
            ) from exc
        if not self.model_path.is_dir():
            raise MLXUnavailable(f"MLX model directory does not exist: {self.model_path}")
        self._mlx_lm = mlx_lm
        self.model, self.tokenizer = mlx_lm.load(
            str(self.model_path), tokenizer_config={"fix_mistral_regex": True}
        )

    def tokenize(self, prompt: str) -> tuple[int, ...]:
        self.load()
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        return tuple(int(item) for item in encoded)

    def generate(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> dict[str, Any]:
        """Greedily generate while reducing each exposed distribution once.

        ``mlx_lm.generate`` hides the predictive distribution.  MLX-LM 0.31.3's
        lower-level ``generate_step`` yields the sampled token together with
        the log-softmax vector used to select it.  The vector is reduced before
        EOS handling and is never retained beyond that one iteration.
        """
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)):
            raise ValueError("max_tokens must be an integer")
        resolved_limit = self.output_limit if max_tokens is None else min(max_tokens, self.output_limit)
        if resolved_limit < 0:
            raise ValueError("max_tokens cannot be negative")
        if temperature != 0.0:
            raise ValueError("The corrected R1 protocol permits greedy temperature=0 only")
        self.load()
        prompt_tokens = self.tokenize(prompt)
        if not prompt_tokens:
            raise ValueError("MLX-LM generate_step requires a non-empty prompt")

        started = time.perf_counter()
        # These imports stay inside generation.  CPU-only commands can import
        # this module for type checking without importing MLX or loading a model.
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        accumulator = TokenTelemetryAccumulator()
        response_token_ids: list[int] = []
        generated_token_count = 0
        stop_token_id: int | None = None
        finish_reason: str | None = None

        def scalar_int(value: Any) -> int:
            item = value.item() if hasattr(value, "item") else value
            return int(item)

        def vector_values(value: Any) -> list[float]:
            values = value.tolist() if hasattr(value, "tolist") else list(value)
            # MLX normally yields a one-dimensional vector.  Unwrap a batch
            # dimension defensively without retaining anything beyond this row.
            while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
                values = values[0]
            if not isinstance(values, list):
                raise ValueError("generate_step returned a non-vector logprob value")
            return [float(item) for item in values]

        def token_text(token_id: int) -> str:
            try:
                return str(self.tokenizer.decode([token_id]))
            except (AttributeError, TypeError):
                return ""

        # Omitting logits_processors is intentional: no temperature, repetition
        # penalty, or other transformation may change the historical distribution.
        for token, logprobs in generate_step(
            mx.array(prompt_tokens),
            self.model,
            max_tokens=resolved_limit,
        ):
            token_id = scalar_int(token)
            generated_token_count += 1
            is_stop = token_id in STOP_TOKEN_IDS
            stop_state = "eos" if is_stop else (
                "max_tokens" if generated_token_count >= resolved_limit else "continue"
            )
            accumulator.add(
                vector_values(logprobs),
                token_id,
                token_text=token_text(token_id),
                stop_state=stop_state,
            )
            if is_stop:
                stop_token_id = token_id
                finish_reason = "stop"
                break
            response_token_ids.append(token_id)
            if generated_token_count >= resolved_limit:
                finish_reason = "length"
                break

        if finish_reason is None:
            finish_reason = "length" if resolved_limit == 0 else "empty"
        telemetry = accumulator.finalize(
            finish_reason=finish_reason,
            stop_token_id=stop_token_id,
        )
        validate_telemetry_payload(telemetry)
        text = str(self.tokenizer.decode(response_token_ids))
        duration = time.perf_counter() - started
        return {
            "text": text,
            "finish_reason": finish_reason,
            # Match the high-level response surface: stop tokens are scored and
            # counted in telemetry, but are not part of returned response text.
            "output_token_count": len(response_token_ids),
            "output_token_ids": response_token_ids,
            "generation_token_count": generated_token_count,
            "duration_seconds": duration,
            # Keep the legacy field explicitly unavailable.  The new compact
            # protocol is carried separately and must not be mislabeled as the
            # old full-vocabulary EntropyTelemetry API.
            "entropy": entropy_windows(()).__dict__ | {"source": "unavailable"},
            "entropy_valid": False,
            "entropy_unavailable_reason": "legacy full-distribution entropy is not collected; see telemetry",
            "telemetry": telemetry,
            "telemetry_protocol_version": TELEMETRY_PROTOCOL_VERSION,
            "telemetry_implementation": TELEMETRY_IMPLEMENTATION,
            "telemetry_source": TELEMETRY_SOURCE,
            "telemetry_valid": bool(telemetry["valid"]),
            "runtime_identity": self.environment_identity(),
        }
