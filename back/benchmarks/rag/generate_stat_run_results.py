#!/usr/bin/env python3
"""
Science Graph — Benchmark Statistical Analysis Generator.
Performs full statistical evaluation for baselines B1, B2, B4, B5, B6:
- Bootstrap 95% Percentile CIs (10,000 resamples, seed 42)
- Friedman omnibus tests & Kendall's W
- Pairwise Wilcoxon signed-rank tests & rank-biserial effect size r_b
- Pairwise McNemar tests for Answerability correctness
- Holm-Bonferroni multiple-comparison corrections within each metric family
- Ceiling/floor effects (>40%) & heavy ties detection (>30%)
Saves results to reports/parsed/stat_run_results.json and prints a formatted summary.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, cast
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
import scipy.stats as scipy_stats

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.connector import data_prep_agent
from core.models import load_report_file
from core.statistics import (
    StatsConfig,
    bootstrap_ci,
    bootstrap_paired_difference_ci,
    rank_biserial_effect_size,
    wilcoxon_signed_rank_test,
    mcnemar_answerability_test,
    friedman_omnibus_test,
    holm_adjusted_p_values,
    compute_multiple_comparison_correction,
    compute_classification_metrics,
)

BASELINES = ["B1", "B2", "B4", "B5", "B6"]
CONTINUOUS_METRICS = [
    "retrieval_recall",
    "context_precision",
    "faithfulness",
    "answer_relevance",
    "citation_fidelity",
    "semantic_accuracy",
    "context_fillness",
    "ar_sa_f1",
    "latency_sec",
]
METRIC_NAMES = {
    "retrieval_recall": "Retrieval Recall",
    "context_precision": "Context Precision",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "citation_fidelity": "Citation Fidelity",
    "semantic_accuracy": "Semantic Accuracy",
    "context_fillness": "Context Fillness",
    "ar_sa_f1": "AR-SA F1",
    "latency_sec": "Latency (sec)",
}

SEED = 42
N_BOOTSTRAPS = 10000
ALPHA = 0.05


def run_full_statistical_analysis():
    project_root = Path(__file__).resolve().parent
    reports_dir = project_root / "reports"
    parsed_dir = reports_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data from result_metrics.yaml and metrics_details.parsed.csv
    yaml_path = reports_dir / "result_metrics.yaml"
    csv_path = parsed_dir / "metrics_details.parsed.csv"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing {yaml_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            rm_data = yaml.safe_load(f)
    except OSError as exc:
        raise RuntimeError(f"Unable to read statistical input {yaml_path}: {exc}") from exc

    records, _ = data_prep_agent(rm_data, baselines=BASELINES)
    df_rec: pd.DataFrame = pd.DataFrame(records)

    try:
        df_csv = pd.read_csv(csv_path)
    except OSError as exc:
        raise RuntimeError(f"Unable to read statistical input {csv_path}: {exc}") from exc
    df_csv = df_csv[df_csv["baseline"].isin(BASELINES)]

    # Merge context_fillness and ar_sa_f1 if missing in records
    merge_features: pd.DataFrame = df_csv.loc[:, ["query_id", "baseline", "context_fillness", "ar_sa_f1"]].copy()
    df_merged: pd.DataFrame = df_rec.merge(
        merge_features,
        on=["query_id", "baseline"],
        how="left",
        suffixes=("", "_csv"),
    )
    if "context_fillness_csv" in df_merged.columns:
        df_merged["context_fillness"] = df_merged["context_fillness"].fillna(df_merged["context_fillness_csv"])
        df_merged.drop(columns=["context_fillness_csv"], inplace=True)
    if "ar_sa_f1_csv" in df_merged.columns:
        df_merged["ar_sa_f1"] = df_merged["ar_sa_f1"].fillna(df_merged["ar_sa_f1_csv"])
        df_merged.drop(columns=["ar_sa_f1_csv"], inplace=True)

    records = df_merged.to_dict(orient="records")

    config = StatsConfig(
        n_bootstraps=N_BOOTSTRAPS,
        alpha=ALPHA,
        ci_method="percentile",
        correction_method="holm",
        random_seed=SEED,
    )

    # 2. Per-Baseline Summary & Ceiling/Floor Effects
    summary: dict[str, dict[str, Any]] = {}
    ceiling_floor_effects: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in CONTINUOUS_METRICS}

    for b in BASELINES:
        summary[b] = {}
        b_records = [r for r in records if r["baseline"] == b]
        n_b = len(b_records)

        # Classification & Confusion Matrix
        tp = sum(1 for r in b_records if r["outcome"] == "TP")
        fp = sum(1 for r in b_records if r["outcome"] == "FP")
        tn = sum(1 for r in b_records if r["outcome"] == "TN")
        fn = sum(1 for r in b_records if r["outcome"] == "FN")
        clas_metrics = compute_classification_metrics(tp, fp, tn, fn, n_b)
        summary[b]["classification"] = clas_metrics

        for m in CONTINUOUS_METRICS:
            vals = np.array([float(r[m]) for r in b_records if r.get(m) is not None and not math.isnan(float(r[m]))], dtype=float)
            ci_res = bootstrap_ci(vals, config)
            
            ceil_pct = float(np.mean(vals == 1.0)) if len(vals) > 0 else 0.0
            floor_pct = float(np.mean(vals == 0.0)) if len(vals) > 0 else 0.0
            ceil_flag = ceil_pct > 0.40
            floor_flag = floor_pct > 0.40

            summary[b][m] = {
                "mean": ci_res["mean"],
                "ci_lower": ci_res["ci_lower"],
                "ci_upper": ci_res["ci_upper"],
                "n": ci_res["n"],
                "ceiling_pct": round(ceil_pct, 4),
                "ceiling_effect": ceil_flag,
                "floor_pct": round(floor_pct, 4),
                "floor_effect": floor_flag,
            }
            ceiling_floor_effects[m][b] = {
                "ceiling_pct": round(ceil_pct, 4),
                "ceiling_effect": ceil_flag,
                "floor_pct": round(floor_pct, 4),
                "floor_effect": floor_flag,
            }

    # 3. Friedman Omnibus Tests & Kendall's W
    friedman_results: dict[str, dict[str, Any]] = {}
    for m in CONTINUOUS_METRICS:
        piv = df_merged.pivot(index="query_id", columns="baseline", values=m)
        piv = piv.fillna(0.0)
        arrays: list[np.ndarray] = [
            np.asarray(piv[b].to_numpy(dtype=float), dtype=float)
            for b in BASELINES if b in piv.columns
        ]

        # Check if all values identical across all baselines
        matrix = np.column_stack(cast(Any, arrays))
        row_ranges = np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)
        if not np.any(row_ranges > 0):
            stat, p_val, kendall_w = 0.0, 1.0, 0.0
        else:
            try:
                stat, p_val = scipy_stats.friedmanchisquare(*arrays)
                stat = float(stat)
                p_val = float(p_val)
                n_q = len(piv)
                k_b = len(BASELINES)
                kendall_w = stat / (n_q * (k_b - 1))
            except Exception:
                stat, p_val, kendall_w = 0.0, 1.0, 0.0

        friedman_results[m] = {
            "statistic": round(stat, 4) if stat is not None else None,
            "p_value": p_val,
            "n": len(piv),
            "kendall_w": round(kendall_w, 4) if kendall_w is not None else None,
        }

    # 4. Pairwise Continuous Metric Comparisons (Wilcoxon, Bootstrap Delta CI, Holm)
    pairwise_continuous: list[dict[str, Any]] = []
    heavy_ties_list: list[dict[str, Any]] = []

    for m in CONTINUOUS_METRICS:
        piv = df_merged.pivot(index="query_id", columns="baseline", values=m)
        metric_rows = []
        
        for b_a, b_b in combinations(BASELINES, 2):
            vec_a = np.asarray(piv[b_a].to_numpy(dtype=float), dtype=float)
            vec_b = np.asarray(piv[b_b].to_numpy(dtype=float), dtype=float)
            diffs = vec_a - vec_b
            n_pairs = len(diffs)
            
            zero_diffs_cnt = int(np.sum(diffs == 0.0))
            zero_diffs_pct = float(np.mean(diffs == 0.0))
            heavy_ties_flag = zero_diffs_pct > 0.30

            if heavy_ties_flag:
                heavy_ties_list.append({
                    "metric": m,
                    "baseline_a": b_a,
                    "baseline_b": b_b,
                    "zero_diffs_count": zero_diffs_cnt,
                    "zero_diffs_pct": round(zero_diffs_pct, 4),
                    "heavy_ties": True,
                })

            wilcox = wilcoxon_signed_rank_test(vec_a, vec_b)
            r_b = rank_biserial_effect_size(diffs)

            metric_rows.append({
                "metric": m,
                "baseline_a": b_a,
                "baseline_b": b_b,
                "mean_a": float(np.mean(vec_a)),
                "mean_b": float(np.mean(vec_b)),
                "p_uncorrected": wilcox["p_value"],
                "effect_size_r_b": r_b,
                "n_pairs": n_pairs,
                "zero_diffs_count": zero_diffs_cnt,
                "zero_diffs_pct": round(zero_diffs_pct, 4),
                "heavy_ties": heavy_ties_flag,
            })

        # Holm-Bonferroni correction within metric with FWER-synchronized CIs
        p_uncorrected_list = [r["p_uncorrected"] for r in metric_rows]
        sig_flags, p_holm_list, alpha_adj_list = compute_multiple_comparison_correction(p_uncorrected_list, config)

        for r, sig, p_h, alpha_adj in zip(metric_rows, sig_flags, p_holm_list, alpha_adj_list):
            r["p_holm"] = p_h
            r["significant"] = sig
            r["alpha_adjusted"] = alpha_adj
            vec_a = np.asarray(piv[r["baseline_a"]].to_numpy(dtype=float), dtype=float)
            vec_b = np.asarray(piv[r["baseline_b"]].to_numpy(dtype=float), dtype=float)
            delta_ci = bootstrap_paired_difference_ci(vec_a, vec_b, config, alpha_override=alpha_adj)
            r["delta"] = delta_ci["delta"]
            r["ci_lower"] = delta_ci["ci_lower"]
            r["ci_upper"] = delta_ci["ci_upper"]
            r["rank_delta"] = delta_ci["rank_delta"]
            r["rank_ci_lower"] = delta_ci["rank_ci_lower"]
            r["rank_ci_upper"] = delta_ci["rank_ci_upper"]
            pairwise_continuous.append(r)

    # 5. Pairwise Answerability Comparisons (McNemar, Holm)
    pairwise_answerability: list[dict[str, Any]] = []
    mc_rows = []

    for b_a, b_b in combinations(BASELINES, 2):
        sub_a = df_merged[df_merged["baseline"] == b_a].set_index("query_id")["answerability_correct"]
        sub_b = df_merged[df_merged["baseline"] == b_b].set_index("query_id")["answerability_correct"]
        
        shared_ids = sorted(set(sub_a.index) & set(sub_b.index))
        n_pairs = len(shared_ids)
        
        c_a = np.asarray(sub_a.loc[shared_ids].astype(bool).to_numpy(), dtype=bool)
        c_b = np.asarray(sub_b.loc[shared_ids].astype(bool).to_numpy(), dtype=bool)

        b_cnt = int(np.sum(c_a & ~c_b))
        c_cnt = int(np.sum(~c_a & c_b))
        n_disc = b_cnt + c_cnt

        if n_disc == 0:
            p_val = 1.0
        else:
            k = min(b_cnt, c_cnt)
            p_val = float(min(1.0, 2.0 * scipy_stats.binom.cdf(k, n_disc, 0.5)))

        odds_ratio = float(b_cnt / c_cnt) if c_cnt > 0 else (float(b_cnt) if b_cnt > 0 else 1.0)

        mc_rows.append({
            "metric": "answerability_correctness",
            "baseline_a": b_a,
            "baseline_b": b_b,
            "discordant_b": b_cnt,
            "discordant_c": c_cnt,
            "n_discordant": n_disc,
            "p_uncorrected": p_val,
            "effect_size_odds_ratio": round(odds_ratio, 4),
            "n_pairs": n_pairs,
        })

    p_mc_uncorrected = [r["p_uncorrected"] for r in mc_rows]
    p_mc_holm = holm_adjusted_p_values(p_mc_uncorrected)

    for r, p_h in zip(mc_rows, p_mc_holm):
        r["p_holm"] = p_h
        r["significant"] = bool(p_h <= ALPHA)
        pairwise_answerability.append(r)

    # 6. Assembly of JSON Output Structure
    output_data = {
        "baselines": BASELINES,
        "n_queries": 50,
        "config": {
            "n_bootstraps": N_BOOTSTRAPS,
            "alpha": ALPHA,
            "ci_method": "percentile",
            "correction_method": "holm",
            "random_seed": SEED,
        },
        "summary": summary,
        "friedman": friedman_results,
        "pairwise_continuous": pairwise_continuous,
        "pairwise_answerability": pairwise_answerability,
        "ceiling_floor_effects": ceiling_floor_effects,
        "heavy_ties": heavy_ties_list,
    }

    output_file = parsed_dir / "stat_run_results.json"
    try:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
    except OSError as exc:
        raise RuntimeError(f"Unable to write statistical results to {output_file}: {exc}") from exc

    print(f"Successfully generated statistical results: {output_file}\n")
    print_summary_tables(output_data)


def print_summary_tables(data: dict[str, Any]):
    baselines = data["baselines"]
    summary = data["summary"]
    friedman = data["friedman"]

    print("=" * 110)
    print(" 📊 RAG BENCHMARK STATISTICAL ANALYSIS SUMMARY (B1, B2, B4, B5, B6)")
    print("=" * 110)

    # 1. Baseline Summary Table
    print("\n### 1. CONTINUOUS METRICS SUMMARY (Mean [95% Bootstrap CI])")
    headers = ["Baseline"] + [METRIC_NAMES[m] for m in CONTINUOUS_METRICS]
    print(f"{'Baseline':<10} | " + " | ".join([f"{METRIC_NAMES[m]:<18}" for m in CONTINUOUS_METRICS]))
    print("-" * 180)
    for b in baselines:
        row = [f"{b:<10}"]
        for m in CONTINUOUS_METRICS:
            m_info = summary[b][m]
            mean_v = m_info["mean"]
            lo = m_info["ci_lower"]
            hi = m_info["ci_upper"]
            if mean_v is None:
                row.append(f"{'N/A':<18}")
            elif m == "latency_sec":
                row.append(f"{mean_v:.2f}s [{lo:.2f},{hi:.2f}]".ljust(18))
            else:
                row.append(f"{mean_v:.3f} [{lo:.3f},{hi:.3f}]".ljust(18))
        print(" | ".join(row))

    # 2. Answerability Confusion Matrix Table
    print("\n### 2. ANSWERABILITY CONFUSION MATRICES & QUALITY METRICS (n=50)")
    print(f"{'Baseline':<10} | {'TP':<5} | {'FP':<5} | {'TN':<5} | {'FN':<5} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'MCC':<8}")
    print("-" * 80)
    for b in baselines:
        c = summary[b]["classification"]
        acc = f"{c['accuracy']*100:.1f}%" if c['accuracy'] is not None else "N/A"
        prec = f"{c['precision']*100:.1f}%" if c['precision'] is not None else "N/A"
        rec = f"{c['recall']*100:.1f}%" if c['recall'] is not None else "N/A"
        mcc = f"{c['mcc']:.4f}" if c['mcc'] is not None else "N/A"
        print(f"{b:<10} | {c['TP']:<5} | {c['FP']:<5} | {c['TN']:<5} | {c['FN']:<5} | {acc:<10} | {prec:<10} | {rec:<10} | {mcc:<8}")

    # 3. Friedman Omnibus Tests
    print("\n### 3. FRIEDMAN OMNIBUS TESTS ACROSS B1, B2, B4, B5, B6")
    print(f"{'Metric':<22} | {'chi^2':<10} | {'p-value':<12} | {'n':<6} | {'Kendall W':<10}")
    print("-" * 70)
    for m in CONTINUOUS_METRICS:
        fr = friedman[m]
        p_str = f"{fr['p_value']:.4e}" if fr['p_value'] < 0.001 else f"{fr['p_value']:.4f}"
        print(f"{METRIC_NAMES[m]:<22} | {fr['statistic']:<10.4f} | {p_str:<12} | {fr['n']:<6} | {fr['kendall_w']:<10.4f}")

    # 4. Pairwise Significant Improvements
    sig_pairs = [p for p in data["pairwise_continuous"] if p["significant"]]
    print(f"\n### 4. SIGNIFICANT PAIRWISE COMPARISONS (Holm-corrected p <= 0.05, total = {len(sig_pairs)})")
    print(f"{'Metric':<22} | {'Pair (A vs B)':<15} | {'Delta (A-B)':<12} | {'95% CI':<18} | {'p_uncorr':<10} | {'p_holm':<10} | {'r_b':<8}")
    print("-" * 105)
    for p in sig_pairs:
        p_unc = f"{p['p_uncorrected']:.4e}" if p['p_uncorrected'] < 0.001 else f"{p['p_uncorrected']:.4f}"
        p_h = f"{p['p_holm']:.4e}" if p['p_holm'] < 0.001 else f"{p['p_holm']:.4f}"
        ci_str = f"[{p['ci_lower']:.3f}, {p['ci_upper']:.3f}]"
        r_b_str = f"{p['effect_size_r_b']:.3f}" if p['effect_size_r_b'] is not None else "N/A"
        print(f"{METRIC_NAMES[p['metric']]:<22} | {p['baseline_a'] + ' vs ' + p['baseline_b']:<15} | {p['delta']:<+12.4f} | {ci_str:<18} | {p_unc:<10} | {p_h:<10} | {r_b_str:<8}")

    # 5. Heavy Ties & Ceiling/Floor Effects
    heavy_ties = data["heavy_ties"]
    print(f"\n### 5. HEAVY TIES (>30% Zero Diffs, total = {len(heavy_ties)})")
    for ht in heavy_ties[:10]:
        print(f"  - {METRIC_NAMES[ht['metric']]}: {ht['baseline_a']} vs {ht['baseline_b']} -> {ht['zero_diffs_pct']*100:.1f}% zero diffs ({ht['zero_diffs_count']}/50)")
    if len(heavy_ties) > 10:
        print(f"  ... and {len(heavy_ties) - 10} more heavy tie pairs.")

    print("\n=" * 110)


if __name__ == "__main__":
    run_full_statistical_analysis()
