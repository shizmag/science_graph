"""
Science Graph — centralized console output with rich formatting.

All modules should import from here instead of using print() directly.
Color semantics:
  info     → cyan    — progress steps, what's happening
  success  → green   — completed steps
  warning  → yellow  — non-fatal issues
  error    → red     — failures
  model    → magenta — model loading / AI activity
  search   → blue    — search / retrieval activity
  dim      → grey    — verbose / secondary info
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings
from typing import Generator

from rich.console import Console
from rich.theme import Theme

# ── Theme ─────────────────────────────────────────────────────────────────────

_THEME = Theme(
    {
        "info":    "bold cyan",
        "success": "bold green",
        "warning": "yellow",
        "error":   "bold red",
        "model":   "magenta",
        "search":  "bold blue",
        "muted":   "dim white",
        "accent":  "bold white",
        "label":   "bold cyan",
    },
    inherit=True,
)

# In MCP mode, redirect standard console output to stderr to prevent corrupting the stdio transport channel
if os.environ.get("SCIENCE_GRAPH_MCP_MODE") == "1":
    console = Console(theme=_THEME, highlight=False, stderr=True)
else:
    console = Console(theme=_THEME, highlight=False)
err_console = Console(theme=_THEME, highlight=False, stderr=True)

import time
from datetime import datetime

# Global flag that can be set externally or auto-detected from command-line arguments
SHOW_TIME: bool = "-t" in sys.argv or "--trace" in sys.argv

# Timing state
_start_time: float = time.time()
_last_time: float = time.time()


def _get_time_prefix() -> str:
    """Returns a formatted timestamp and elapsed time prefix if SHOW_TIME is active."""
    if not SHOW_TIME:
        return ""
    global _last_time
    now = time.time()
    current_time_str = datetime.now().strftime("%H:%M:%S")
    delta = now - _last_time
    _last_time = now
    return f"[muted][{current_time_str} (+{delta:.2f}s)][/] "


# ── Formatters ────────────────────────────────────────────────────────────────

def info(msg: str) -> None:
    """Cyan — progress step."""
    console.print(f"{_get_time_prefix()}  [info]→[/]  {msg}")


def success(msg: str) -> None:
    """Green — completed successfully."""
    console.print(f"{_get_time_prefix()}  [success]✓[/]  {msg}")


def warning(msg: str) -> None:
    """Yellow — non-fatal issue."""
    console.print(f"{_get_time_prefix()}  [warning]⚠[/]  {msg}")


def error(msg: str) -> None:
    """Red — error, shown on stderr."""
    err_console.print(f"{_get_time_prefix()}  [error]✗[/]  {msg}")


def model_msg(msg: str) -> None:
    """Magenta — model / AI activity."""
    console.print(f"{_get_time_prefix()}  [model]⚡[/]  {msg}")


def search_msg(msg: str) -> None:
    """Blue — search / retrieval."""
    console.print(f"{_get_time_prefix()}  [search]🔍[/]  {msg}")


def dim(msg: str) -> None:
    """Grey — secondary / verbose."""
    console.print(f"{_get_time_prefix()}  [muted]{msg}[/]")


def debug(msg: str) -> None:
    """Grey — debug / verbose."""
    console.print(f"{_get_time_prefix()}  [muted][DEBUG] {msg}[/]")


def section(title: str) -> None:
    """Horizontal rule with title."""
    prefix = _get_time_prefix()
    if prefix:
        console.print(prefix, end="")
    console.rule(f"[accent] {title} [/accent]")


def blank() -> None:
    console.print()


# ── Suppress noisy external library output ────────────────────────────────────

class _DevNull(io.TextIOBase):
    """Silent file-like object that discards all writes."""
    def write(self, s: str) -> int:
        return len(s)
    def flush(self) -> None:
        pass


@contextlib.contextmanager
def suppress_stderr() -> Generator[None, None, None]:
    """
    Temporarily redirect stderr to /dev/null to silence noisy library output
    (HuggingFace progress bars, tqdm loading bars from sentence-transformers, etc.)
    Errors from our own code are raised as exceptions, so this is safe.
    """
    old_stderr = sys.stderr
    try:
        sys.stderr = _DevNull()  # type: ignore[assignment]
        yield
    finally:
        sys.stderr = old_stderr


@contextlib.contextmanager
def suppress_stdout() -> Generator[None, None, None]:
    """Redirect stdout to /dev/null (for tqdm bars written to stdout)."""
    old_stdout = sys.stdout
    try:
        sys.stdout = _DevNull()  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = old_stdout


def setup_quiet_env() -> None:
    """
    Set environment variables to suppress noisy output from HuggingFace Hub,
    tokenizers, and transformers BEFORE those libraries are imported.
    Call this at the very start of main.py.
    """
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # Suppress Python warnings from third-party libs
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
