"""
Science Graph — RAG Statistical Integration Connector.
Orchestrates data preparation, statistical computation, and report generation pipeline stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.analytics import METRIC_LABELS
from core.statistics import (
    StatsConfig,
    compute_statistical_analysis,
    format_p_value,
    prepare_per_query_records,
    significance_stars,
)


def data_prep_agent(
    data: dict,
    joined_rows: list[dict[str, Any]] | None = None,
    baselines: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Normalize parsed benchmark data into per-query statistical records.

    Prefers structured YAML ``results`` when available; falls back to joined CSV rows.
    """
    if data and data.get("results"):
        return prepare_per_query_records(data, baselines)

    if not joined_rows:
        return [], baselines or []

    if baselines is None:
        baselines = sorted({str(r.get("baseline")) for r in joined_rows if r.get("baseline")})

    pseudo_data = {"results": []}
    by_query: dict[str, dict] = {}
    for row in joined_rows:
        qid = str(row.get("query_id", "UNKNOWN"))
        if qid not in by_query:
            is_ans = row.get("is_answerable", True)
            if isinstance(is_ans, str):
                is_ans = is_ans.lower() == "true"
            by_query[qid] = {
                "id": qid,
                "category": row.get("category", "general"),
                "is_answerable": is_ans,
                "baselines": {},
            }
        baseline = str(row.get("baseline"))
        by_query[qid]["baselines"][baseline] = {
            "status": "success",
            "latency_sec": row.get("latency_sec"),
            "predicted_abstained": row.get("predicted_abstained", False),
            "answerability_outcome": row.get("answerability_outcome"),
            "eval_metrics": {
                k: row.get(k)
                for k in [
                    "retrieval_recall",
                    "context_precision",
                    "faithfulness",
                    "answer_relevance",
                    "citation_fidelity",
                    "semantic_accuracy",
                    "context_fillness",
                    "ar_sa_f1",
                    "predicted_abstained",
                    "answerability_outcome",
                ]
            },
        }

    pseudo_data["results"] = list(by_query.values())
    return prepare_per_query_records(pseudo_data, baselines)


def stats_agent(
    data: dict,
    config: StatsConfig,
    joined_rows: list[dict[str, Any]] | None = None,
    baselines: list[str] | None = None,
) -> dict[str, Any]:
    """Run full statistical analysis using prepared records."""
    if not config.enable_stats:
        return {"enabled": False, "config": config.to_dict()}

    records, resolved_baselines = data_prep_agent(data, joined_rows, baselines)
    if records:
        synthetic = {"results": _records_to_results(records)}
        return compute_statistical_analysis(synthetic, config, resolved_baselines)

    return compute_statistical_analysis(data, config, baselines)


def _records_to_results(records: list[dict[str, Any]]) -> list[dict]:
    """Convert flat records back to nested results for compute_statistical_analysis."""
    by_query: dict[str, dict] = {}
    for rec in records:
        qid = rec["query_id"]
        if qid not in by_query:
            by_query[qid] = {
                "id": qid,
                "category": rec.get("category", "general"),
                "is_answerable": rec.get("is_answerable", True),
                "baselines": {},
            }
        baseline = rec["baseline"]
        by_query[qid]["baselines"][baseline] = {
            "status": "success",
            "latency_sec": rec.get("latency_sec"),
            "predicted_abstained": rec.get("predicted_abstained"),
            "answerability_outcome": rec.get("outcome"),
            "eval_metrics": {
                m: rec.get(m) for m in METRIC_LABELS if m in rec
            },
        }
    return list(by_query.values())


def reporting_agent(
    stats_analysis: dict[str, Any],
    output_dir: Path | None = None,
    *,
    enable_plots: bool = False,
) -> dict[str, Any]:
    """Generate markdown report sections and optional plot files."""
    if not stats_analysis.get("enabled"):
        return {"markdown_sections": [], "plot_paths": []}

    sections = [build_statistical_markdown(stats_analysis)]
    plot_paths: list[str] = []

    if enable_plots and output_dir is not None:
        plot_paths = generate_statistical_plots(stats_analysis, output_dir)

    return {
        "markdown_sections": sections,
        "plot_paths": plot_paths,
    }


def build_statistical_markdown(stats_analysis: dict[str, Any]) -> str:
    """Build markdown sections for statistical results."""
    if not stats_analysis.get("enabled"):
        return ""

    lines: list[str] = []
    config = stats_analysis.get("config", {})
    alpha = config.get("alpha", 0.05)
    ci_method = config.get("ci_method", "percentile")
    n_boot = config.get("n_bootstraps", 10000)

    lines.append("## 📐 Statistical Analysis")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append(f"> {stats_analysis.get('filtering_note', '')}")
    lines.append("")
    lines.append(
        f"Bootstrap CIs: **{n_boot:,}** resamples, **{100 * (1 - alpha):.0f}%** CI, "
        f"method: **{ci_method.upper()}**. "
        f"Pairwise tests: **Wilcoxon** (continuous metrics), **McNemar** (answerability). "
        f"Omnibus: **Friedman** (≥3 baselines). "
        f"Correction: **{config.get('correction_method', 'holm')}**."
    )
    lines.append("")

    lines.append("### Baseline Summary — Mean (95% CI)")
    lines.append("")
    baselines = stats_analysis.get("baselines", [])
    summary = stats_analysis.get("baseline_summary", {})

    header_metrics = ["semantic_accuracy", "faithfulness", "answer_relevance", "ar_sa_f1", "latency_sec"]
    header = ["Baseline"] + [METRIC_LABELS.get(m, m) for m in header_metrics] + ["MCC"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| :--- | " + " | ".join(["---:"] * (len(header) - 1)) + " |")

    for b in baselines:
        b_sum = summary.get(b, {})
        row = [f"**{b}**"]
        for m in header_metrics:
            m_data = b_sum.get(m, {})
            mean_v = m_data.get("mean")
            lo = m_data.get("ci_lower")
            hi = m_data.get("ci_upper")
            if mean_v is None:
                row.append("—")
            elif m == "latency_sec":
                row.append(f"{mean_v:.3f}s [{lo:.3f}, {hi:.3f}]")
            else:
                row.append(f"{mean_v:.3f} [{lo:.3f}, {hi:.3f}]")
        mcc = b_sum.get("classification", {}).get("mcc")
        row.append(f"{mcc:.4f}" if mcc is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("### Answerability Confusion Matrices (with MCC)")
    lines.append("")
    lines.append("| Baseline | TP | FP | TN | FN | MCC | Hallucination Rate |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in baselines:
        clas = summary.get(b, {}).get("classification", {})
        mcc = clas.get("mcc")
        hall = clas.get("hallucination_rate")
        mcc_str = f"{mcc:.4f}" if mcc is not None else "—"
        hall_str = f"{hall * 100:.1f}%" if hall is not None else "—"
        lines.append(
            f"| **{b}** | {clas.get('TP', 0)} | {clas.get('FP', 0)} | "
            f"{clas.get('TN', 0)} | {clas.get('FN', 0)} | {mcc_str} | {hall_str} |"
        )
    lines.append("")

    sig_improvements = stats_analysis.get("significant_improvements", [])
    lines.append("### ⭐ Significant Improvements")
    lines.append("")
    if not sig_improvements:
        lines.append("_No statistically significant improvements detected at the configured α level._")
    else:
        lines.append("| Comparison | Metric | Δ (A−B) | 95% CI | p-value | Effect size |")
        lines.append("| :--- | :--- | ---: | :--- | ---: | ---: |")
        for row in sig_improvements[:20]:
            metric_label = METRIC_LABELS.get(row["metric"], row["metric"])
            delta = row.get("delta")
            lo = row.get("ci_lower")
            hi = row.get("ci_upper")
            p = row.get("p_value")
            es = row.get("effect_size")
            stars = row.get("stars", "")
            comp = f"{row['baseline_a']} vs {row['baseline_b']}"
            ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "—"
            es_str = f"{es:.3f}" if es is not None else "—"
            lines.append(
                f"| {comp} | {metric_label} | "
                f"**{delta:+.4f}**{stars} | {ci_str} | {format_p_value(p)} | {es_str} |"
            )
    lines.append("")

    pairwise = stats_analysis.get("pairwise", [])
    if pairwise:
        lines.append("### Pairwise Comparisons")
        lines.append("")
        lines.append("| A | B | Metric | Test | Δ | 95% CI | p-value | Effect | Sig |")
        lines.append("| :--- | :--- | :--- | :--- | ---: | :--- | ---: | ---: | :--- |")
        for row in pairwise:
            if row.get("test") == "mcnemar":
                metric_label = "Answerability"
                delta_str = f"b={row.get('discordant_b', 0)}, c={row.get('discordant_c', 0)}"
                ci_str = "—"
            else:
                metric_label = METRIC_LABELS.get(row["metric"], row["metric"])
                delta = row.get("delta")
                lo = row.get("ci_lower")
                hi = row.get("ci_upper")
                delta_str = f"{delta:+.4f}" if delta is not None else "—"
                ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "—"

            sig = "**yes**" if row.get("significant") else "no"
            stars = row.get("stars", "")
            es = row.get("effect_size")
            es_str = f"{es:.3f}" if es is not None else "—"
            lines.append(
                f"| {row['baseline_a']} | {row['baseline_b']} | {metric_label} | "
                f"{row.get('test', 'wilcoxon')} | {delta_str}{stars} | {ci_str} | "
                f"{format_p_value(row.get('p_value'))} | {es_str} | {sig} |"
            )
        lines.append("")

    friedman = stats_analysis.get("friedman", {})
    if friedman:
        lines.append("### Friedman Omnibus Tests")
        lines.append("")
        lines.append("| Metric | χ² | p-value | n |")
        lines.append("| :--- | ---: | ---: | ---: |")
        for metric, res in friedman.items():
            if res.get("p_value") is None:
                continue
            label = METRIC_LABELS.get(metric, metric)
            stars = significance_stars(res["p_value"], alpha)
            lines.append(
                f"| {label} | {res.get('statistic', 0):.4f} | "
                f"{format_p_value(res['p_value'])}{stars} | {res.get('n', 0)} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_statistical_plots(stats_analysis: dict[str, Any], output_dir: Path) -> list[str]:
    """Optional boxplots, p-value heatmaps, and difference plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    records = stats_analysis.get("records", [])
    baselines = stats_analysis.get("baselines", [])
    if not records or not baselines:
        return paths

    # Presentation mapping for legacy baselines (exclude legacy B3, B4->B3, B5->B4, B6->B5)
    is_legacy = ("B6" in baselines) or ("B3" not in baselines and any(b in baselines for b in ["B4", "B5"]))
    legacy_map = {"B1": "B1", "B2": "B2", "B4": "B3", "B5": "B4", "B6": "B5"}

    plot_baselines = []
    baseline_map = {}
    for b in baselines:
        if is_legacy:
            if b == "B3":
                continue
            mapped = legacy_map.get(b)
            if mapped:
                plot_baselines.append(mapped)
                baseline_map[b] = mapped
        else:
            if b in ["B1", "B2", "B3", "B4", "B5"]:
                plot_baselines.append(b)
                baseline_map[b] = b

    metric = "semantic_accuracy"
    data_by_b = []
    labels = []
    for raw_b, mapped_b in baseline_map.items():
        vals = [
            float(r[metric])
            for r in records
            if r["baseline"] == raw_b
            and r["outcome"] in {"TP", "FP"}
            and r.get(metric) is not None
            and not (isinstance(r[metric], float) and np.isnan(r[metric]))
        ]
        if vals:
            data_by_b.append(vals)
            labels.append(mapped_b)

    if data_by_b:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data_by_b, labels=labels)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} (TP+FP only)")
        ax.set_ylabel("Score")
        fig.tight_layout()
        box_path = output_dir / "stats_boxplot_semantic_accuracy.png"
        fig.savefig(box_path, dpi=120)
        plt.close(fig)
        paths.append(str(box_path))

    active_raw_baselines = list(baseline_map.keys())
    pairwise = [
        p for p in stats_analysis.get("pairwise", [])
        if p.get("metric") == "semantic_accuracy"
        and p.get("test") == "wilcoxon"
        and p.get("baseline_a") in baseline_map
        and p.get("baseline_b") in baseline_map
    ]
    if pairwise and len(plot_baselines) >= 2:
        n = len(plot_baselines)
        idx = {b: i for i, b in enumerate(active_raw_baselines)}
        heat = np.ones((n, n))
        for row in pairwise:
            if row["baseline_a"] in idx and row["baseline_b"] in idx:
                i, j = idx[row["baseline_a"]], idx[row["baseline_b"]]
                p = row.get("p_value") or 1.0
                heat[i, j] = p
                heat[j, i] = p

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(heat, vmin=0, vmax=0.1, cmap="RdYlGn_r")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(plot_baselines)
        ax.set_yticklabels(plot_baselines)
        ax.set_title("p-values: Semantic Accuracy (Wilcoxon)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        heat_path = output_dir / "stats_pvalue_heatmap.png"
        fig.savefig(heat_path, dpi=120)
        plt.close(fig)
        paths.append(str(heat_path))

    return paths


def run_statistical_pipeline(
    data: dict,
    config: StatsConfig | None = None,
    joined_rows: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Single-call entry: data prep → stats → reporting.

    Returns combined analysis dict with markdown sections and plot paths.
    """
    config = config or StatsConfig()
    analysis = stats_agent(data, config, joined_rows)
    report = reporting_agent(
        analysis,
        output_dir=output_dir,
        enable_plots=config.enable_plots,
    )
    analysis["markdown_sections"] = report["markdown_sections"]
    analysis["plot_paths"] = report["plot_paths"]
    return analysis


def export_stats_json(stats_analysis: dict[str, Any], path: Path) -> None:
    """Serialize statistical results (excluding raw records) to JSON."""
    exportable = {k: v for k, v in stats_analysis.items() if k != "records"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exportable, f, ensure_ascii=False, indent=2, default=str)
