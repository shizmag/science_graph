"""Persistent, rate-limited Rich progress for smoke and full matrix runs."""

from __future__ import annotations

import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Progress:
    def __init__(self, jsonl_path: Path, total_arms: int = 2, *, run_id: str = "smoke",
                 total_queries: int = 1, total_operations: int | None = None,
                 output_dir: str = ""):
        self.events_path = jsonl_path
        self.progress_path = jsonl_path.parent / "progress.json"
        self.errors_path = jsonl_path.parent / "errors.jsonl"
        self.run_id = run_id
        self.total_arms = total_arms
        self.total_queries = total_queries
        self.total_operations = total_operations if total_operations is not None else total_arms * total_queries
        self.output_dir = output_dir
        self.completed_operations = 0
        self.failed_operations = 0
        self.retried_operations = 0
        self.cache_hits = 0
        self.reused_generations = 0
        self.current_tokens = 0
        self.tokens = 0
        self.started = time.monotonic()
        self.stage_started = self.started
        self.stage = "starting"
        self.current_arm = ""
        self.arm_index = 0
        self.current_query = 0
        self.recent_tokens_per_second: float | None = None
        self._last_token_count = 0
        self._last_token_time = self.started
        self._last_persist = 0.0
        self._last_display = 0.0
        self._completion_samples: list[float] = []
        self.peak_gpu_memory_allocated_bytes = 0
        self.peak_gpu_memory_reserved_bytes = 0
        self._operation_started: float | None = None
        self.stage_seconds: dict[str, float] = {}
        self._last_details: dict[str, Any] = {}
        self._live = None
        self._console = None
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist(force=True)

    def start(self) -> "Progress":
        if sys.stdout.isatty():
            try:
                from rich.console import Console
                from rich.live import Live
                self._console = Console()
                self._live = Live(self._render(), console=self._console, refresh_per_second=4,
                                  transient=False, redirect_stdout=False, redirect_stderr=False)
                self._live.start()
            except Exception:
                self._live = None
        return self

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self._persist(force=True)

    def update(self, stage: str, *, completed_arms: int | None = None,
               tokens: int | None = None, details: dict | None = None,
               completed_operations: int | None = None, current_arm: str | None = None,
               arm_index: int | None = None, current_query: int | None = None,
               failed_operations: int | None = None, retried_operations: int | None = None,
               cache_hits: int | None = None, reused_generations: int | None = None,
               current_tokens: int | None = None) -> None:
        now = time.monotonic()
        if stage != self.stage:
            self.stage_seconds[self.stage] = self.stage_seconds.get(self.stage, 0.0) + now - self.stage_started
            self.stage_started = now
        self.stage = stage
        self._last_details = details or {}
        if completed_arms is not None and completed_operations is None:
            completed_operations = completed_arms
        if completed_operations is not None:
            self.completed_operations = completed_operations
        if tokens is not None:
            self.tokens = tokens
        if current_tokens is not None:
            self.current_tokens = current_tokens
        if failed_operations is not None:
            self.failed_operations = failed_operations
        if retried_operations is not None:
            self.retried_operations = retried_operations
        if cache_hits is not None:
            self.cache_hits = cache_hits
        if reused_generations is not None:
            self.reused_generations = reused_generations
        if current_arm is not None:
            self.current_arm = current_arm
        if arm_index is not None:
            self.arm_index = arm_index
        if current_query is not None:
            self.current_query = current_query
        event = self._snapshot(details or {})
        self._append_event(event)
        self._persist()
        self._display(force=True)

    def begin_operation(self, *, arm_id: str, arm_index: int, query_index: int) -> None:
        self.current_arm = arm_id
        self.arm_index = arm_index
        self.current_query = query_index
        self._operation_started = time.monotonic()
        self.update("retrieval", current_arm=arm_id, arm_index=arm_index, current_query=query_index,
                    details={"event": "operation_started"})

    def complete_operation(self, *, success: bool, elapsed_seconds: float,
                           generated_tokens: int = 0, details: dict | None = None) -> None:
        if elapsed_seconds >= 0:
            self._completion_samples.append(float(elapsed_seconds))
            if len(self._completion_samples) > 100:
                self._completion_samples = self._completion_samples[-100:]
        if success:
            self.completed_operations += 1
        else:
            self.failed_operations += 1
        self.tokens += int(generated_tokens)
        self.current_tokens = 0
        self.update("checkpoint", completed_operations=self.completed_operations,
                    failed_operations=self.failed_operations, tokens=self.tokens,
                    current_tokens=0, details=details or {})

    def observe_operation_duration(self, elapsed_seconds: float) -> None:
        """Add a real completed-operation timing sample without changing counters."""
        if elapsed_seconds < 0:
            return
        self._completion_samples.append(float(elapsed_seconds))
        if len(self._completion_samples) > 100:
            self._completion_samples = self._completion_samples[-100:]
        self._persist(force=True)
        self._display()

    def retry(self, *, details: dict[str, Any]) -> None:
        self.retried_operations += 1
        self.update("retrying failed operation", retried_operations=self.retried_operations,
                    details=details)

    def cache_hit(self, *, reused_generation: bool = False, details: dict[str, Any] | None = None) -> None:
        self.cache_hits += 1
        if reused_generation:
            self.reused_generations += 1
        self.update("checkpoint/cache hit", details=details or {})

    def error(self, message: str, *, arm_id: str = "", query_id: str = "",
              attempt: int | None = None, transient: bool | None = None) -> None:
        row = {"time_epoch": time.time(), "run_id": self.run_id,
               "arm_id": arm_id, "query_id": query_id, "attempt": attempt,
               "transient": transient, "error": message}
        with self.errors_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.update("operation error", details=row)

    def token_progress(self, stage: str, *, completed_arms: int, tokens: int,
                       details: dict | None = None) -> None:
        now = time.monotonic()
        delta = tokens - self._last_token_count
        elapsed = now - self._last_token_time
        if delta >= 0 and elapsed > 0:
            self.recent_tokens_per_second = delta / elapsed
        self._last_token_count = tokens
        self._last_token_time = now
        self.current_tokens = max(0, int((details or {}).get("current_generation_tokens", 0)))
        self.stage = stage
        self._last_details = details or {}
        self.tokens = max(self.tokens, tokens)
        self.completed_operations = max(self.completed_operations, completed_arms)
        self._persist()
        self._display()

    def _snapshot(self, details: dict[str, Any]) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - self.started)
        enough_samples = len(self._completion_samples) >= 3
        mean_seconds = statistics.median(self._completion_samples[-30:]) if enough_samples else None
        remaining = max(0, self.total_operations - self.completed_operations - self.failed_operations)
        eta_seconds = mean_seconds * remaining if mean_seconds is not None else None
        gpu = self._gpu_memory()
        if gpu:
            self.peak_gpu_memory_allocated_bytes = max(self.peak_gpu_memory_allocated_bytes,
                                                       int(gpu["allocated_bytes"]))
            self.peak_gpu_memory_reserved_bytes = max(self.peak_gpu_memory_reserved_bytes,
                                                      int(gpu["reserved_bytes"]))
        return {
            "schema_version": "1.0", "run_id": self.run_id,
            "stage": self.stage, "arm_id": self.current_arm,
            "arm_index": self.arm_index, "total_arms": self.total_arms,
            "query_index": self.current_query, "queries_per_arm": self.total_queries,
            "completed_operations": self.completed_operations,
            "failed_operations": self.failed_operations,
            "total_operations": self.total_operations,
            "retried_operations": self.retried_operations,
            "cache_hits": self.cache_hits,
            "reused_generations": self.reused_generations,
            "current_generated_tokens": self.current_tokens,
            "cumulative_generated_tokens": self.tokens,
            "recent_tokens_per_second": round(self.recent_tokens_per_second, 3)
            if self.recent_tokens_per_second is not None else None,
            "gpu_memory": gpu,
            "gpu_name": self._gpu_name(),
            "total_vram_bytes": gpu.get("total_bytes") if gpu else None,
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
            "eta_basis": "median completed query-arm duration" if eta_seconds is not None else None,
            "stage_seconds": {key: round(value, 3) for key, value in self.stage_seconds.items()},
            "recent_query_arm_seconds": [round(value, 3) for value in self._completion_samples[-30:]],
            "peak_gpu_memory_allocated_bytes": self.peak_gpu_memory_allocated_bytes,
            "peak_gpu_memory_reserved_bytes": self.peak_gpu_memory_reserved_bytes,
            "output_directory": self.output_dir,
            "updated_epoch": time.time(),
            "details": details,
        }

    @staticmethod
    def _gpu_memory() -> dict[str, Any] | None:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            return {"allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "total_bytes": int(props.total_memory)}
        except Exception:
            return None

    def _append_event(self, row: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _persist(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_persist < 1.0:
            return
        row = self._snapshot(self._last_details)
        temporary = self.progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(self.progress_path)
        self._last_persist = now

    def _render(self):
        from rich.panel import Panel
        from rich.table import Table
        elapsed = time.monotonic() - self.started
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column()
        grid.add_row("Run", self.run_id)
        grid.add_row("GPU", self._gpu_name() or "not detected")
        memory = self._gpu_memory()
        if memory:
            allocated = memory["allocated_bytes"] / (1024 ** 3)
            total = memory["total_bytes"] / (1024 ** 3)
            grid.add_row("GPU memory", f"{allocated:.2f}/{total:.2f} GiB allocated")
        grid.add_row("Stage", self.stage)
        grid.add_row("Arm", f"{self.current_arm or '—'} ({self.arm_index}/{self.total_arms})")
        grid.add_row("Query", f"{self.current_query}/{self.total_queries}")
        grid.add_row("Overall", f"{self.completed_operations + self.failed_operations}/{self.total_operations} operations; {self.failed_operations} failed")
        grid.add_row("Retries / cache", f"{self.retried_operations} / {self.cache_hits} hits ({self.reused_generations} generations reused)")
        grid.add_row("Tokens", f"{self.current_tokens} current, {self.tokens} cumulative; {self.recent_tokens_per_second or 0:.1f} tok/s")
        grid.add_row("Elapsed", f"{elapsed:,.0f}s")
        eta = self._snapshot({})["eta_seconds"]
        if eta is not None:
            grid.add_row("ETA", f"{eta:,.0f}s from observed query-arm timings")
        grid.add_row("Output", self.output_dir or "not selected")
        return Panel(grid, title="Biomedical RAG CUDA ablation", border_style="blue")

    @staticmethod
    def _gpu_name() -> str | None:
        try:
            import torch
            return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            return None

    def _display(self, force: bool = False) -> None:
        now = time.monotonic()
        if self._live is not None:
            self._live.update(self._render())
            return
        if now - self._last_display < 2.0 and not force:
            return
        self._last_display = now
        row = self._snapshot(self._last_details)
        message = (f"[{self.run_id}] {self.stage} | arm {self.arm_index}/{self.total_arms} "
                   f"{self.current_arm or '—'} | query {self.current_query}/{self.total_queries} | "
                   f"operations {self.completed_operations + self.failed_operations}/{self.total_operations} "
                   f"({self.failed_operations} failed) | tokens {self.tokens} | {row['elapsed_seconds']:.1f}s")
        print(message, flush=True)

    def finish(self) -> None:
        self.close()

    @contextmanager
    def live(self):
        self.start()
        try:
            yield self
        finally:
            self.close()
