import os
import re
import time
import subprocess
from rich.progress import ProgressColumn
from rich.text import Text
from src import console as con

PROGRESS_LINE_RE = re.compile(r"PROGRESS\s+(\w+)\s+(\d+)\s*/\s*(\d+)", re.I)


def format_progress_marker(stage: str, completed: int, total: int) -> str:
    """Return a machine-readable progress marker for subprocess stdout."""
    return f"PROGRESS {stage} {completed}/{total}"


def parse_progress_line(line: str, step_pattern: str) -> tuple[str, int] | None:
    """Return ('set', completed) or ('advance', 1) or None.

    Primary protocol: ``PROGRESS <stage> completed/total`` (stage should match
    step_pattern; unknown stages are accepted when they match step_pattern
    case-insensitively, otherwise ignored).

    Legacy fallbacks:
      retrieval: none — Stage-3 ``Query:`` logs are human-facing only. Counting
        them would race the bar to 100% before Stage 5 finishes and then snap
        backward when Stage-5 ``PROGRESS retrieval`` markers arrive. Units are
        Stage-5 completions only (or success-exit reconcile if no markers).
      generation: startswith ``Running `` with ``:`` and not ``Running command:``
                  also: ``Reusing previously generated`` (checkpoint counts as a unit)
      evaluation: ``Evaluated case N/M`` -> set completed=N
    """
    if not line:
        return None

    line_str = line.strip()
    m = PROGRESS_LINE_RE.search(line_str)
    if m:
        stage, completed_s, _total_s = m.group(1), m.group(2), m.group(3)
        if stage.lower() == step_pattern.lower() or step_pattern.lower() in stage.lower():
            try:
                return ("set", int(completed_s))
            except ValueError:
                return None
        return None

    if step_pattern == "retrieval":
        # Do not advance on Stage-3 Query lines — work unit is Stage-5 PROGRESS.
        return None
    if step_pattern == "generation":
        if line_str.startswith("Running ") and ":" in line_str and not line_str.startswith("Running command:"):
            return ("advance", 1)
        if "Reusing previously generated" in line_str:
            return ("advance", 1)
    elif step_pattern == "evaluation":
        if "Evaluated case " in line_str:
            try:
                parts = line_str.split("Evaluated case ")[1].split("/")
                current = int(parts[0])
                return ("set", current)
            except (IndexError, ValueError):
                return None
    return None


class IterationSpeedColumn(ProgressColumn):
    def render(self, task):
        speed = task.finished_speed or task.speed
        if speed is None or speed == 0:
            return Text("- sec/it", style="progress.data.speed")
        sec_per_it = 1.0 / speed
        return Text(f"{sec_per_it:.2f} sec/it", style="progress.data.speed")


def run_command_with_progress(cmd: list, title: str, total_steps: int, step_pattern: str) -> float:
    from rich.progress import (
        SpinnerColumn,
        Progress,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
    )

    t0 = time.perf_counter()
    total_steps = max(int(total_steps), 1)
    completed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, finished_style="green", complete_style="cyan"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        IterationSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=con.console,
        transient=False,
    ) as progress:
        # Caller passes unit info in title; use as-is.
        task = progress.add_task(title, total=total_steps)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        for line in iter(process.stdout.readline, ""):
            line_str = line.strip()
            line_lower = line_str.lower()

            parsed = parse_progress_line(line_str, step_pattern)
            is_pure_progress = bool(PROGRESS_LINE_RE.search(line_str)) and (
                line_str.upper().startswith("PROGRESS ") or line_str.upper().lstrip().startswith("PROGRESS ")
            )

            if parsed is not None:
                action, value = parsed
                if action == "set":
                    completed = min(max(value, 0), total_steps)
                elif action == "advance":
                    completed = min(completed + value, total_steps)
                progress.update(task, completed=completed)

            # Print important lines above the bar; skip pure PROGRESS markers.
            if is_pure_progress:
                continue
            if "✗" in line_str or "error" in line_lower or "warning" in line_lower or "failed" in line_lower:
                progress.console.print(f"  [red]✗[/] {line_str}")
            elif "Loaded" in line_str or "Successfully" in line_str:
                progress.console.print(f"  [green]✓[/] {line_str}")
            elif "Ready" in line_str.lower() or "initializing" in line_str.lower():
                progress.console.print(f"  [dim]{line_str}[/]")
            elif (
                "query" in line_lower
                or "running b" in line_lower
                or "reusing previously generated" in line_lower
                or "completed" in line_lower
                or "generating answer" in line_lower
                or "evaluated" in line_lower
                or "evaluation finished" in line_lower
                or "benchmarking complete" in line_lower
                or line_str.startswith("===")
                or line_str.endswith("===")
            ):
                progress.console.print(f"  {line_str}")

        process.stdout.close()
        return_code = process.wait()

        # Honest finish: successful exit reconciles bar to total.
        if return_code == 0 and completed < total_steps:
            completed = total_steps
            progress.update(task, completed=completed)

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

    return time.perf_counter() - t0
