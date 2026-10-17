"""
Science Graph — RAG Scientific Visualization Utilities.
Provides academic theme setup, color palettes, figure saving, and dataset loading helpers.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Use Agg backend for headless execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from core.analytics import analyze_metrics, METRIC_LABELS
from core.models import load_report_file


# Standard quality metrics
METRICS = [
    "retrieval_recall",
    "context_precision",
    "faithfulness",
    "answer_relevance",
    "citation_fidelity",
    "semantic_accuracy"
]

# Baseline color palette for publication figures
COLOR_PALETTE: Dict[str, str] = {
    'B1': '#6C757D',  # Slate Gray (Pure Lexical)
    'B2': '#2B5C8F',  # Cool Blue (Pure Dense)
    'B3': '#2A9D8F',  # Emerald Green (Hybrid + Rerank)
    'B4': '#E76F51',  # Warm Coral (Graph + Rerank)
    'B5': '#E63946',  # Crimson Red (Full Pipeline)
    'Custom': '#D53F8C',  # Magenta
    'SciQ': '#2B6CB0'     # Standard Blue
}


def get_baseline_color(b: str) -> str:
    """Returns baseline color, with slate grey fallback."""
    return COLOR_PALETTE.get(b, '#718096')


def get_category_metric_value(stats: dict, category: str, baseline: str, metric: str) -> float:
    """Safely retrieves category metric value, defaulting to 0.0 if not found."""
    return stats.get("category_stats", {}).get(category, {}).get(baseline, {}).get(metric, 0.0)


def setup_academic_style():
    """Configures matplotlib and seaborn for publication-ready figures."""
    sns.set_theme(style="whitegrid")

    # Configure font settings to resemble Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times', 'serif']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    # Visual aesthetics
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#555555'
    plt.rcParams['ytick.color'] = '#555555'
    plt.rcParams['grid.color'] = '#EEEEEE'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


def create_output_directory(input_dir: Optional[Path] = None) -> Path:
    """
    Creates a figures directory. If input_dir is a directory, puts it there.
    Otherwise creates inside back/benchmarks/rag/figures.
    """
    if input_dir and input_dir.is_dir():
        run_dir = input_dir / "figures"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    base_path = Path(__file__).resolve().parents[1]
    figures_dir = base_path / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Append to .gitignore if not present
    gitignore_path = base_path / ".gitignore"
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        if "figures/" not in content:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\nfigures/\n")
    else:
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write("figures/\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = figures_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_report_data(yaml_path: Path) -> Tuple[pd.DataFrame, dict]:
    """Loads results YAML, computes all metrics, and returns (DataFrame, stats)."""
    report = load_report_file(yaml_path)
    data = report.model_dump()
    stats = analyze_metrics(data)

    found_baselines = set()
    metadata = data.get("metadata") or {}
    for b in metadata.get("baselines_evaluated", []):
        found_baselines.add(b)
    for r in data.get("results", []):
        for b in r.get("baselines", {}).keys():
            found_baselines.add(b)

    preferred_order = ['B1', 'B2', 'B3', 'B4', 'B5']
    baselines_list = [b for b in preferred_order if b in found_baselines]
    for b in sorted(found_baselines):
        if b not in baselines_list:
            baselines_list.append(b)

    if not baselines_list:
        baselines_list = ['B1', 'B2', 'B3', 'B4', 'B5']

    rows = []
    for r in data.get("results", []):
        q_id = r.get("id")
        category = r.get("category")
        for b in baselines_list:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
            eval_metrics = b_data.get("eval_metrics", {})
            row = {
                "query_id": q_id,
                "category": category,
                "baseline": b,
                "status": b_data.get("status"),
                "latency_sec": b_data.get("latency_sec"),
                "expected_papers": r.get("expected_papers", []),
                "retrieved_papers": b_data.get("retrieved_papers", []),
                "retrieval_recall": eval_metrics.get("retrieval_recall"),
                "context_precision": eval_metrics.get("context_precision"),
                "faithfulness": eval_metrics.get("faithfulness"),
                "answer_relevance": eval_metrics.get("answer_relevance"),
                "citation_fidelity": eval_metrics.get("citation_fidelity"),
                "semantic_accuracy": eval_metrics.get("semantic_accuracy"),
                "context_fillness": eval_metrics.get("context_fillness"),
                "token_output": eval_metrics.get("token_output"),
                "token_answer": eval_metrics.get("token_answer"),
                "token_reasoning": eval_metrics.get("token_reasoning")
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df, stats


def save_plot(fig, run_dir: Path, name: str):
    """Saves the figure in PNG, SVG, and PDF formats."""
    fig.tight_layout()
    fig.savefig(run_dir / f"{name}.png", bbox_inches='tight', dpi=300)
    fig.savefig(run_dir / f"{name}.svg", bbox_inches='tight')
    fig.savefig(run_dir / f"{name}.pdf", bbox_inches='tight')
    plt.close(fig)


def find_sciq_results(base_path: Path, input_dir: Optional[Path] = None) -> Optional[Path]:
    """Finds SciQ results inside reports directory or near input directory."""
    search_dirs = []
    if input_dir:
        search_dirs.append(input_dir.parent)
        search_dirs.append(input_dir)

    reports_dir = base_path / "reports"
    search_dirs.append(reports_dir)

    for r_dir in search_dirs:
        if not r_dir or not r_dir.exists() or not r_dir.is_dir():
            continue

        for item in r_dir.iterdir():
            if item.is_dir() and ("SciQ" in item.name or "sciq" in item.name.lower()):
                for yaml_name in ["result_metrics.yaml", "evaluation_results.yaml"]:
                    yaml_file = item / yaml_name
                    if yaml_file.exists():
                        return yaml_file

        for yaml_file in r_dir.glob("**/result_metrics.yaml"):
            if "sciq" in str(yaml_file).lower():
                return yaml_file
        for yaml_file in r_dir.glob("**/evaluation_results.yaml"):
            if "sciq" in str(yaml_file).lower():
                return yaml_file

    return None
