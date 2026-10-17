"""
Science Graph — Clipboard & Prompt Export Utilities.
Provides cross-platform clipboard interaction, run discovery, and prompt template helpers.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import IntPrompt, Prompt
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False


def copy_to_clipboard(text: str) -> bool:
    """Copies the text to clipboard. Supports macOS pbcopy, Linux xclip/xsel, and Windows clip."""
    try:
        if sys.platform == "darwin":
            process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            process.communicate(input=text.encode("utf-8"))
            return True
        else:
            if subprocess.run(["which", "xclip"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
                process.communicate(input=text.encode("utf-8"))
                return True
            elif subprocess.run(["which", "xsel"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                process = subprocess.Popen(["xsel", "--clipboard", "--input"], stdin=subprocess.PIPE)
                process.communicate(input=text.encode("utf-8"))
                return True
            elif os.name == "nt":
                process = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
                process.communicate(input=text.encode("utf-8"))
                return True
    except Exception as e:
        print(f"Clipboard copy error: {e}")
    return False


def find_runs(reports_dir: Path) -> List[Path]:
    """Finds all run_* directories in reports_dir sorted by creation time (latest first)."""
    if not reports_dir.exists():
        print(f"Error: Reports directory {reports_dir} does not exist.")
        sys.exit(1)

    runs = [d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    runs.sort(key=lambda x: x.name, reverse=True)
    return runs


def get_metrics_summary(run_dir: Path, reports_dir: Path, script_dir: Path) -> str:
    """Reads or generates metrics_summary.md for the given run directory."""
    summary_md_path = run_dir / "metrics_summary.md"

    if summary_md_path.exists():
        with open(summary_md_path, "r", encoding="utf-8") as f:
            return f.read()

    result_metrics_path = run_dir / "result_metrics.yaml"
    if result_metrics_path.exists():
        parse_script = script_dir / "parse_metrics.py"
        if parse_script.exists():
            msg = f"[*] metrics_summary.md missing in {run_dir.name}. Auto-generating from result_metrics.yaml..."
            if HAS_RICH:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                print(msg)
            try:
                subprocess.run(
                    [sys.executable, str(parse_script), "--file", str(result_metrics_path), "--output-md", str(summary_md_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if summary_md_path.exists():
                    with open(summary_md_path, "r", encoding="utf-8") as f:
                        return f.read()
            except Exception as e:
                print(f"Warning: Failed to auto-generate summary: {e}")

    msg = f"[!] Warning: Run '{run_dir.name}' has not been evaluated yet (missing result_metrics.yaml)."
    if HAS_RICH:
        console.print(f"[bold red]{msg}[/bold red]")
        console.print("[dim]Falling back to reports/metrics_summary.md (global latest)[/dim]")
    else:
        print(msg)
        print("Falling back to reports/metrics_summary.md (global latest)")

    fallback_path = reports_dir / "metrics_summary.md"
    if fallback_path.exists():
        with open(fallback_path, "r", encoding="utf-8") as f:
            return f.read()

    print(f"Error: Could not find any metrics summary for run {run_dir.name}")
    sys.exit(1)


def load_templates(prompts_file: Path) -> Dict[str, Any]:
    """Loads prompt templates from the specified YAML file."""
    if not prompts_file.exists():
        print(f"Error: Prompt templates file not found at {prompts_file}")
        sys.exit(1)

    with open(prompts_file, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
            return data.get("templates", {})
        except Exception as e:
            print(f"Error parsing templates YAML: {e}")
            sys.exit(1)


def display_runs_table(runs: List[Path]):
    """Renders a beautiful table of available runs."""
    if HAS_RICH:
        table = Table(title="Available Benchmarking Runs", show_header=True, header_style="bold magenta")
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Run Directory Name", style="white")
        table.add_column("Created At (Estimate)", style="green")

        for idx, run in enumerate(runs):
            parts = run.name.split("_")
            time_str = "Unknown"
            if len(parts) >= 3:
                date_part = parts[1]
                time_part = parts[2]
                if len(date_part) == 8 and len(time_part) == 6:
                    time_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
            table.add_row(str(idx + 1), run.name, time_str)
        console.print(table)
    else:
        print("\nAvailable Benchmarking Runs:")
        print(f"{'Index':<6} | {'Run Directory Name':<50}")
        print("-" * 60)
        for idx, run in enumerate(runs):
            print(f"{idx + 1:<6} | {run.name:<50}")


def display_templates_table(templates: Dict[str, Any]):
    """Renders a table of available templates."""
    if HAS_RICH:
        table = Table(title="Prompt Templates", show_header=True, header_style="bold blue")
        table.add_column("Key", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Description", style="green")

        for key, template in templates.items():
            table.add_row(key, template.get("name", key), template.get("description", ""))
        console.print(table)
    else:
        print("\nAvailable Prompt Templates:")
        print(f"{'Key':<20} | {'Name':<40} | {'Description'}")
        print("-" * 80)
        for key, template in templates.items():
            print(f"{key:<20} | {template.get('name', key):<40} | {template.get('description', '')}")


def interactive_selection(runs: List[Path], templates: Dict[str, Any]) -> Tuple[Path, Optional[Path], str]:
    """Prompts the user interactively to select run(s) and template."""
    display_runs_table(runs)

    run1_idx = 1
    if len(runs) > 1:
        if HAS_RICH:
            run1_idx = IntPrompt.ask("Select run index (default: latest)", choices=[str(i) for i in range(1, len(runs) + 1)], default=1)
        else:
            try:
                val = input("Select run index (default: 1 [latest]): ").strip()
                run1_idx = int(val) if val else 1
            except ValueError:
                run1_idx = 1

    run1 = runs[run1_idx - 1]

    display_templates_table(templates)

    template_keys = list(templates.keys())
    selected_template_key = template_keys[0]
    if HAS_RICH:
        selected_template_key = Prompt.ask("Select prompt template key", choices=template_keys, default=template_keys[0])
    else:
        val = input(f"Select prompt template key (default: {template_keys[0]}): ").strip()
        if val in templates:
            selected_template_key = val

    run2 = None
    if "compare" in selected_template_key:
        if len(runs) < 2:
            print("Error: You selected a comparison template, but there is only one run available.")
            sys.exit(1)

        if HAS_RICH:
            run2_idx = IntPrompt.ask("Select second run index for comparison", choices=[str(i) for i in range(1, len(runs) + 1) if i != run1_idx], default=2 if run1_idx != 2 else 1)
        else:
            try:
                val = input("Select second run index for comparison: ").strip()
                run2_idx = int(val) if val else (2 if run1_idx != 2 else 1)
            except ValueError:
                run2_idx = 2 if run1_idx != 2 else 1
        run2 = runs[run2_idx - 1]

    return run1, run2, selected_template_key
