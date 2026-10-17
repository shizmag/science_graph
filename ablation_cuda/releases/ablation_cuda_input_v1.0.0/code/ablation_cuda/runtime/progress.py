from __future__ import annotations

import json
import sys
import time
from pathlib import Path


class Progress:
    def __init__(self, jsonl_path: Path, total_arms: int = 2):
        self.jsonl_path = jsonl_path
        self.total_arms = total_arms
        self.completed_arms = 0
        self.tokens = 0
        self.started = time.monotonic()
        self.stage = "starting"
        self._last_live_display = 0.0
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, stage: str, *, completed_arms: int | None = None,
               tokens: int | None = None, details: dict | None = None) -> None:
        if completed_arms is not None:
            self.completed_arms = completed_arms
        if tokens is not None:
            self.tokens = tokens
        self.stage = stage
        elapsed = round(time.monotonic() - self.started, 3)
        row = {"time_epoch": time.time(), "stage": stage,
               "completed_arms": self.completed_arms, "total_arms": self.total_arms,
               "tokens_generated": self.tokens, "elapsed_seconds": elapsed,
               "details": details or {}}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        display = (f"{stage} | arms {self.completed_arms}/{self.total_arms} | "
                   f"tokens {self.tokens} | {elapsed:.1f}s")
        if sys.stdout.isatty():
            print("\r" + display.ljust(100), end="", flush=True)
        else:
            print(display, flush=True)

    def token_progress(self, stage: str, *, completed_arms: int, tokens: int,
                       details: dict | None = None) -> None:
        """Log each token update while rate-limiting the terminal display."""
        self.completed_arms = completed_arms
        self.tokens = tokens
        self.stage = stage
        elapsed = round(time.monotonic() - self.started, 3)
        row = {"time_epoch": time.time(), "stage": stage,
               "completed_arms": self.completed_arms, "total_arms": self.total_arms,
               "tokens_generated": self.tokens, "elapsed_seconds": elapsed,
               "details": details or {}}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        now = time.monotonic()
        if (sys.stdout.isatty() and now - self._last_live_display >= 0.5) or (
            not sys.stdout.isatty() and tokens % 32 == 0
        ):
            display = (f"{stage} | arms {self.completed_arms}/{self.total_arms} | "
                       f"tokens {self.tokens} | {elapsed:.1f}s")
            if sys.stdout.isatty():
                print("\r" + display.ljust(100), end="", flush=True)
            else:
                print(display, flush=True)
            self._last_live_display = now

    def finish(self) -> None:
        if sys.stdout.isatty():
            print()
