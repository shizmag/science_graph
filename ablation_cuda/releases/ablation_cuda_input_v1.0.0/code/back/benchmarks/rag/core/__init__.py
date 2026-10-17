"""Benchmark modules exposed lazily to keep query-only workers lightweight.

Historically this initializer imported every benchmark stage, including report
visualization dependencies. Importing ``core.config`` for a smoke run therefore
loaded pandas, matplotlib, and seaborn even though the query path never uses
them. Preserve the package-level module API without importing unrelated stages.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_MODULES = {
    "analytics", "clipboard", "config", "connector", "evaluator", "generation",
    "limiter", "metrics", "models", "pipelined", "reporting", "retrieval",
    "sanitization", "shannon_estimator", "statistics", "stats", "subprocess_runner",
    "traces", "visualization",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | _MODULES)


__all__ = sorted(_MODULES)
