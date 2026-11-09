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
    EntropyTelemetry,
    entropy_windows,
)


class MLXUnavailable(RuntimeError):
    """Raised when the optional local MLX dependencies are not installed."""


class MLXAdapter:
    """One resident model, sequential generation, and explicit telemetry status."""

    def __init__(self, model_path: Path, *, output_limit: int = 768):
        self.model_path = Path(model_path).expanduser().resolve()
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
        self.load()
        resolved_limit = self.output_limit if max_tokens is None else min(max_tokens, self.output_limit)
        started = time.perf_counter()
        # The direct mlx_lm generator exposes text, not the complete predictive
        # distribution.  Do not label selected-token log-probs as H20/H50/H100.
        from mlx_lm.sample_utils import make_sampler

        text = self._mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=resolved_limit,
            # mlx-lm 0.31.x forwards keyword arguments to ``generate_step``;
            # that API accepts a sampler, not the CLI spelling ``temp``.
            # Supplying the sampler preserves temperature-zero greedy decoding
            # without passing an unsupported keyword downstream.
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        duration = time.perf_counter() - started
        return {
            "text": text,
            "finish_reason": "length_or_eos_unknown",
            "output_token_count": len(self.tokenizer.encode(text, add_special_tokens=False)),
            "duration_seconds": duration,
            # No predictive-distribution stream is exposed by the direct
            # mlx_lm.generate API.  Use the explicit unavailable schema rather
            # than entropy_windows(()), whose source label means that a full
            # distribution was actually observed.
            "entropy": entropy_windows(()).__dict__ | {"source": "unavailable"},
            "entropy_valid": False,
            "entropy_unavailable_reason": "mlx_lm.generate does not expose full predictive distributions",
            "runtime_identity": self.environment_identity(),
        }
