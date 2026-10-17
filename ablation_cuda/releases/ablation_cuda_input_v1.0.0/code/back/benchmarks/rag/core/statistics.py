"""
Pure statistical functions for RAG evaluation pipelines.

Quality metrics (faithfulness, relevance, etc.) are averaged only over queries
where the model produced an answer (outcome in TP, FP). TN and FP on
unanswerable queries carry NaN quality scores because faithfulness/relevance
cannot be meaningfully computed on abstentions or skipped judge calls.

Telemetry / logit confidence metrics (MSP, margins, H_gen, CLR, etc.) use the
same TP+FP answered filter: generation confidence is only meaningful when the
model produced an answer.

Safety / answerability metrics use the full query set.

Pairwise Holm-Bonferroni correction is applied within each metric family
(quality, telemetry, safety) and is not pooled across families.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any, Literal

import numpy as np
from scipy import stats as scipy_stats

from core.analytics import QUALITY_METRICS, TELEMETRY_METRICS, METRIC_LABELS
from core.metrics import classify_answerability, detect_abstention, get_is_answerable

try:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
except ImportError:  # pragma: no cover
    sm_mcnemar = None

ANSWERED_OUTCOMES = frozenset({"TP", "FP"})
QUALITY_METRIC_SET = frozenset(QUALITY_METRICS)
TELEMETRY_METRIC_SET = frozenset(TELEMETRY_METRICS)
# Quality + telemetry both require a model answer for valid per-query scores.
ANSWERED_METRIC_SET = QUALITY_METRIC_SET | TELEMETRY_METRIC_SET

SAFETY_METRICS = [
    "mcc",
    "hallucination_rate",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "fpr",
    "fnr",
    "answer_rate",
    "abstention_rate",
    "f1",
]

# Field aliases for shannon_diagnostics / top-level telemetry extraction.
# Primary name first, then common alternate keys from estimator / older dumps.
_TELEMETRY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "h_gen": ("h_gen", "generation_entropy"),
    "h_citation": ("h_citation", "citation_entropy"),
    "delta_h_gen": ("delta_h_gen", "entropy_reduction"),
    "avg_msp": ("avg_msp", "msp"),
    "avg_logit_margin": ("avg_logit_margin", "logit_margin"),
    "first_token_msp": ("first_token_msp",),
    "first_token_margin": ("first_token_margin",),
    "ll_rag": ("ll_rag",),
    "ll_base": ("ll_base",),
    "clr": ("clr",),
    "n_citation_tokens": ("n_citation_tokens", "citation_token_count"),
}


@dataclass
class StatsConfig:
    """Configurable parameters for statistical analysis."""

    enable_stats: bool = True
    n_bootstraps: int = 10000
    alpha: float = 0.05
    ci_method: Literal["percentile", "bca"] = "percentile"
    correction_method: Literal["holm", "bonferroni", "none"] = "holm"
    random_seed: int = 42
    enable_plots: bool = False
    plots_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def significance_stars(p_value: float | None, alpha: float = 0.05) -> str:
    if p_value is None or math.isnan(p_value):
        return ""
    if p_value <= 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < alpha:
        return "**"
    return ""


def format_p_value(p_value: float | None) -> str:
    if p_value is None or math.isnan(p_value):
        return "—"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.4f}"


def compute_mcc(tp: int, fp: int, tn: int, fn: int) -> float | None:
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return None
    return (tp * tn - fp * fn) / math.sqrt(denom)


def compute_classification_metrics(tp: int, fp: int, tn: int, fn: int, total_q: int) -> dict[str, Any]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    fnr = fn / (fn + tp) if (fn + tp) > 0 else None
    num_unans = fp + tn
    hallucination_rate = fp / num_unans if num_unans > 0 else None
    answer_rate = (tp + fp) / total_q if total_q > 0 else 0.0
    abstention_rate = (tn + fn) / total_q if total_q > 0 else 0.0
    mcc = compute_mcc(tp, fp, tn, fn)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "hallucination_rate": hallucination_rate,
        "answer_rate": answer_rate,
        "abstention_rate": abstention_rate,
        "mcc": mcc,
    }


def _resolve_outcome(case: dict, b_data: dict) -> str:
    outcome = b_data.get("answerability_outcome")
    if outcome not in ("TP", "FP", "TN", "FN"):
        eval_metrics = b_data.get("eval_metrics", {})
        if isinstance(eval_metrics, dict):
            outcome = eval_metrics.get("answerability_outcome")
    if outcome not in ("TP", "FP", "TN", "FN"):
        is_ans = get_is_answerable(case)
        pred_abst = b_data.get("predicted_abstained")
        if pred_abst is None and isinstance(b_data.get("eval_metrics"), dict):
            pred_abst = b_data["eval_metrics"].get("predicted_abstained")
        if pred_abst is None:
            try:
                pred_abst = detect_abstention(b_data.get("generated_answer", ""))
            except Exception:
                pred_abst = False
        outcome = classify_answerability(is_ans, bool(pred_abst))
    return outcome


def _quality_value_for_outcome(outcome: str, raw_value: Any) -> float | None:
    """
    TN / FP on unanswerable queries must be NaN for quality metrics.
    FN (abstained on answerable) is excluded via answered-outcome filter.
    """
    if outcome in ("TN", "FP"):
        return np.nan
    if raw_value is None or raw_value == "":
        return np.nan
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return np.nan


def _lookup_float_keys(source: dict | None, keys: tuple[str, ...]) -> float | None:
    """Return the first coercible float for any key in *keys* from *source*."""
    if not isinstance(source, dict):
        return None
    for key in keys:
        val = source.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _extract_telemetry_value(b_data: dict, metric: str) -> float:
    """
    Pull a telemetry scalar from shannon_diagnostics / metrics / top-level fields.

    Prefers primary names, then aliases (e.g. generation_entropy → h_gen).
    Returns NaN when the metric is absent or non-numeric.
    """
    aliases = _TELEMETRY_FIELD_ALIASES.get(metric, (metric,))
    shannon = b_data.get("shannon_diagnostics")
    if not isinstance(shannon, dict):
        metrics = b_data.get("metrics")
        if isinstance(metrics, dict):
            shannon = metrics.get("shannon_diagnostics")
        else:
            shannon = None

    for source in (
        shannon if isinstance(shannon, dict) else None,
        b_data,
        b_data.get("eval_metrics") if isinstance(b_data.get("eval_metrics"), dict) else None,
        b_data.get("metrics") if isinstance(b_data.get("metrics"), dict) else None,
    ):
        found = _lookup_float_keys(source, aliases)
        if found is not None:
            return found
    return np.nan


def prepare_per_query_records(
    data: dict,
    baselines: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Build per-(query, baseline) records with outcome-aware quality metric NaNs.

    Quality metrics are set to NaN for TN and FP (unanswerable cases where judge
    metrics are undefined). Telemetry values are extracted from shannon_diagnostics
    (and top-level aliases) for all outcomes; aggregation still uses TP+FP filtering.
    """
    results = data.get("results", [])
    if not results:
        return [], []

    if baselines is None:
        baselines = sorted(
            {
                b
                for r in results
                for b in r.get("baselines", {}).keys()
                if r.get("baselines", {}).get(b, {}).get("status") == "success"
            }
        )
    baselines = sorted(baselines)

    records: list[dict[str, Any]] = []
    for case in results:
        q_id = str(case.get("id", "UNKNOWN"))
        category = case.get("category", "general")
        is_answerable = get_is_answerable(case)

        for baseline in baselines:
            b_data = case.get("baselines", {}).get(baseline, {})
            if not b_data or b_data.get("status") != "success":
                continue

            outcome = _resolve_outcome(case, b_data)
            eval_metrics = b_data.get("eval_metrics", {}) or {}
            predicted_abstained = b_data.get("predicted_abstained")
            if predicted_abstained is None:
                predicted_abstained = eval_metrics.get("predicted_abstained", False)

            record: dict[str, Any] = {
                "query_id": q_id,
                "category": category,
                "baseline": baseline,
                "is_answerable": is_answerable,
                "outcome": outcome,
                "predicted_abstained": bool(predicted_abstained),
                "answerability_correct": outcome in ("TP", "TN"),
                "latency_sec": b_data.get("latency_sec"),
            }

            for metric in QUALITY_METRICS:
                record[metric] = _quality_value_for_outcome(outcome, eval_metrics.get(metric))

            for metric in TELEMETRY_METRICS:
                record[metric] = _extract_telemetry_value(b_data, metric)

            lat = b_data.get("latency_sec")
            record["latency_sec"] = float(lat) if lat is not None else np.nan
            records.append(record)

    return records, baselines


def filter_answered_quality_df(records: list[dict[str, Any]], metric: str) -> np.ndarray:
    """
    Return paired-quality values for metric over TP+FP outcomes only.

    Explicit filter: df[df['outcome'].isin(['TP', 'FP'])] — not skipna=True alone.
    """
    values = []
    for row in records:
        if row.get("outcome") not in ANSWERED_OUTCOMES:
            continue
        val = row.get(metric)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        values.append(float(val))
    return np.array(values, dtype=float)


def aggregate_metric_mean(
    records: list[dict[str, Any]],
    baseline: str,
    metric: str,
    *,
    use_answered_filter: bool = True,
) -> float | None:
    subset = [r for r in records if r["baseline"] == baseline]
    if use_answered_filter and metric in ANSWERED_METRIC_SET:
        subset = [r for r in subset if r["outcome"] in ANSWERED_OUTCOMES]
        vals = [
            float(r[metric])
            for r in subset
            if r.get(metric) is not None and not math.isnan(float(r[metric]))
        ]
    else:
        vals = [
            float(r[metric])
            for r in subset
            if r.get(metric) is not None and not (isinstance(r[metric], float) and math.isnan(r[metric]))
        ]
    if not vals:
        return None
    return float(np.mean(vals))


def paired_metric_vectors(
    records: list[dict[str, Any]],
    baseline_a: str,
    baseline_b: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align per-query paired values for two baselines."""
    by_query_a: dict[str, float] = {}
    by_query_b: dict[str, float] = {}
    answered_only = metric in ANSWERED_METRIC_SET

    for row in records:
        if row["baseline"] == baseline_a:
            if answered_only and row["outcome"] not in ANSWERED_OUTCOMES:
                continue
            val = row.get(metric)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            by_query_a[row["query_id"]] = float(val)
        elif row["baseline"] == baseline_b:
            if answered_only and row["outcome"] not in ANSWERED_OUTCOMES:
                continue
            val = row.get(metric)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            by_query_b[row["query_id"]] = float(val)

    shared_ids = sorted(set(by_query_a) & set(by_query_b))
    if not shared_ids:
        return np.array([]), np.array([]), []

    vec_a = np.array([by_query_a[q] for q in shared_ids], dtype=float)
    vec_b = np.array([by_query_b[q] for q in shared_ids], dtype=float)
    return vec_a, vec_b, shared_ids


def bootstrap_ci(
    values: np.ndarray,
    config: StatsConfig,
    statistic: str = "mean",
) -> dict[str, float | None]:
    """Bootstrap confidence interval for a 1-D sample."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0}

    rng = np.random.default_rng(config.random_seed)
    observed = float(np.mean(values)) if statistic == "mean" else float(np.median(values))

    boot_stats = np.empty(config.n_bootstraps, dtype=float)
    for i in range(config.n_bootstraps):
        sample = values[rng.integers(0, n, size=n)]
        boot_stats[i] = np.mean(sample) if statistic == "mean" else np.median(sample)

    alpha = config.alpha
    if config.ci_method == "bca":
        ci_lower, ci_upper = _bca_interval(values, boot_stats, observed, alpha, statistic)
    else:
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "mean": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
    }


def _bca_interval(
    values: np.ndarray,
    boot_stats: np.ndarray,
    observed: float,
    alpha: float,
    statistic: str,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap interval."""
    n = len(values)
    prop_less = float(np.mean(boot_stats < observed))
    z0 = scipy_stats.norm.ppf(max(min(prop_less, 1 - 1e-12), 1e-12))

    jack_vals = np.empty(n, dtype=float)
    for i in range(n):
        leave_one = np.delete(values, i)
        jack_vals[i] = np.mean(leave_one) if statistic == "mean" else np.median(leave_one)
    jack_mean = np.mean(jack_vals)
    num = np.sum((jack_mean - jack_vals) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)
    accel = num / den if den != 0 else 0.0

    def _adjusted_quantile(prob: float) -> float:
        z = scipy_stats.norm.ppf(prob)
        adj = z0 + z
        denom = 1 - accel * adj
        if abs(denom) < 1e-12:
            return z
        return z0 + (z0 + z) / denom

    lo_prob = scipy_stats.norm.cdf(_adjusted_quantile(alpha / 2))
    hi_prob = scipy_stats.norm.cdf(_adjusted_quantile(1 - alpha / 2))
    lo_prob = max(min(lo_prob, 1.0), 0.0)
    hi_prob = max(min(hi_prob, 1.0), 0.0)
    return float(np.percentile(boot_stats, 100 * lo_prob)), float(np.percentile(boot_stats, 100 * hi_prob))


def hodges_lehmann_delta(diffs: np.ndarray) -> float:
    """Computes Hodges-Lehmann pseudo-median difference for paired samples."""
    clean = np.asarray(diffs, dtype=float)
    clean = clean[~np.isnan(clean)]
    nonzero = clean[clean != 0]
    if len(nonzero) == 0:
        return 0.0
    iu = np.triu_indices(len(nonzero))
    walsh = (nonzero[iu[0]] + nonzero[iu[1]]) * 0.5
    return float(np.median(walsh))


def bootstrap_paired_difference_ci(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    config: StatsConfig,
    alpha_override: float | None = None,
) -> dict[str, float | None]:
    """
    Bootstrap CIs for paired differences (A - B) resampling queries.
    Computes both arithmetic mean delta CI and Hodges-Lehmann rank-based delta CI.
    Dynamically adjusts percentile bounds using alpha_override (alpha_adjusted).
    """
    if len(vec_a) == 0 or len(vec_b) == 0 or len(vec_a) != len(vec_b):
        return {
            "delta": None,
            "ci_lower": None,
            "ci_upper": None,
            "rank_delta": None,
            "rank_ci_lower": None,
            "rank_ci_upper": None,
            "n": 0,
        }

    diffs = np.asarray(vec_a, dtype=float) - np.asarray(vec_b, dtype=float)
    observed_mean = float(np.mean(diffs))
    observed_hl = hodges_lehmann_delta(diffs)
    n = len(diffs)

    alpha = alpha_override if alpha_override is not None else config.alpha
    rng = np.random.default_rng(config.random_seed + 1)

    boot_mean = np.empty(config.n_bootstraps, dtype=float)
    boot_rank = np.empty(config.n_bootstraps, dtype=float)

    for i in range(config.n_bootstraps):
        idx = rng.integers(0, n, size=n)
        sample = diffs[idx]
        boot_mean[i] = float(np.mean(sample))
        boot_rank[i] = hodges_lehmann_delta(sample)

    if config.ci_method == "bca":
        ci_lower, ci_upper = _bca_interval(diffs, boot_mean, observed_mean, alpha, "mean")
        rank_ci_lower, rank_ci_upper = _bca_interval(diffs, boot_rank, observed_hl, alpha, "median")
    else:
        ci_lower = float(np.percentile(boot_mean, 100.0 * (alpha / 2.0)))
        ci_upper = float(np.percentile(boot_mean, 100.0 * (1.0 - alpha / 2.0)))
        rank_ci_lower = float(np.percentile(boot_rank, 100.0 * (alpha / 2.0)))
        rank_ci_upper = float(np.percentile(boot_rank, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "delta": observed_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "rank_delta": observed_hl,
        "rank_ci_lower": rank_ci_lower,
        "rank_ci_upper": rank_ci_upper,
        "n": n,
    }


def rank_biserial_effect_size(differences: np.ndarray) -> float | None:
    """Rank-biserial correlation for Wilcoxon signed-rank effect size."""
    diffs = differences[~np.isnan(differences)]
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return None
    ranks = scipy_stats.rankdata(np.abs(diffs))
    w_plus = ranks[diffs > 0].sum()
    w_minus = ranks[diffs < 0].sum()
    denom = w_plus + w_minus
    if denom == 0:
        return None
    return float((w_plus - w_minus) / denom)


def _as_float64_array(values: np.ndarray) -> np.ndarray:
    """Coerce inputs to contiguous float64 arrays for scipy compatibility."""
    clean = [float(v) for v in np.asarray(values).ravel().tolist()]
    return np.ascontiguousarray(np.array(clean, dtype=np.float64))


def _wilcoxon_signed_rank_manual(vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test via normal approximation.

    Fallback when scipy.stats.wilcoxon fails on certain numpy/scipy builds.
    """
    diffs = vec_a - vec_b
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0

    abs_ranks = scipy_stats.rankdata(np.abs(nonzero), method="average")
    w_plus = float(abs_ranks[nonzero > 0].sum())
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return w_plus, 1.0
    z = (w_plus - mu) / sigma
    p_value = float(2 * scipy_stats.norm.cdf(-abs(z)))
    return w_plus, p_value


def wilcoxon_signed_rank_test(vec_a: np.ndarray, vec_b: np.ndarray) -> dict[str, Any]:
    vec_a = _as_float64_array(vec_a)
    vec_b = _as_float64_array(vec_b)
    if len(vec_a) < 2 or len(vec_b) < 2 or len(vec_a) != len(vec_b):
        return {"statistic": None, "p_value": None, "effect_size": None, "n": len(vec_a)}

    diffs = vec_a - vec_b
    if np.allclose(diffs, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "effect_size": 0.0, "n": len(diffs)}

    statistic: float | None = None
    p_value: float | None = None
    for kwargs in (
        {"alternative": "two-sided", "method": "approx"},
        {"alternative": "two-sided"},
    ):
        try:
            res = scipy_stats.wilcoxon(vec_a, vec_b, **kwargs)
            p_value = float(res.pvalue)
            statistic = float(res.statistic)
            break
        except (ValueError, TypeError, AttributeError):
            continue

    if p_value is None:
        try:
            statistic, p_value = _wilcoxon_signed_rank_manual(vec_a, vec_b)
        except (ValueError, TypeError):
            return {"statistic": None, "p_value": None, "effect_size": None, "n": len(diffs)}

    return {
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": rank_biserial_effect_size(diffs),
        "n": len(diffs),
    }


def mcnemar_answerability_test(records: list[dict[str, Any]], baseline_a: str, baseline_b: str) -> dict[str, Any]:
    """
    McNemar test on paired answerability correctness (TP/TN vs FP/FN).
    """
    correct_a: dict[str, bool] = {}
    correct_b: dict[str, bool] = {}
    for row in records:
        qid = row["query_id"]
        if row["baseline"] == baseline_a:
            correct_a[qid] = bool(row["answerability_correct"])
        elif row["baseline"] == baseline_b:
            correct_b[qid] = bool(row["answerability_correct"])

    shared = sorted(set(correct_a) & set(correct_b))
    if not shared:
        return {"p_value": None, "b": 0, "c": 0, "n": 0}

    b_count = sum(1 for q in shared if correct_a[q] and not correct_b[q])
    c_count = sum(1 for q in shared if not correct_a[q] and correct_b[q])

    if b_count + c_count == 0:
        return {"p_value": 1.0, "b": b_count, "c": c_count, "n": len(shared)}

    n_discordant = b_count + c_count
    k = min(b_count, c_count)
    p_value = float(2 * scipy_stats.binom.cdf(k, n_discordant, 0.5))
    p_value = min(p_value, 1.0)

    if sm_mcnemar is not None:
        try:
            result = sm_mcnemar([[0, b_count], [c_count, 0]], exact=True)
            p_value = float(result.pvalue)
        except (ValueError, TypeError):
            pass

    return {"p_value": p_value, "b": b_count, "c": c_count, "n": len(shared)}


def friedman_omnibus_test(
    records: list[dict[str, Any]],
    baselines: list[str],
    metric: str,
) -> dict[str, Any]:
    """Friedman test across >=3 baselines for a paired metric."""
    if len(baselines) < 3:
        return {"statistic": None, "p_value": None, "n": 0}

    by_baseline: dict[str, dict[str, float]] = {b: {} for b in baselines}
    if metric in QUALITY_METRIC_SET:
        target_records = [r for r in records if r.get("is_answerable", True)]
    else:
        target_records = records

    target_query_ids = sorted(list({str(r["query_id"]) for r in target_records if "query_id" in r}))
    if len(target_query_ids) < 2:
        return {"statistic": None, "p_value": None, "n": len(target_query_ids)}

    for row in target_records:
        b = row["baseline"]
        if b not in by_baseline:
            continue
        val = row.get(metric)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = 0.0
        by_baseline[b][str(row["query_id"])] = float(val)

    for b in baselines:
        for qid in target_query_ids:
            if qid not in by_baseline[b]:
                by_baseline[b][qid] = 0.0

    arrays = [
        _as_float64_array([by_baseline[b][q] for q in target_query_ids])
        for b in baselines
    ]
    # Stack as (n_queries, n_baselines). When every query has identical scores
    # across baselines, all ranks are fully tied and scipy's tie correction
    # factor c becomes 0 → RuntimeWarning + NaN (divide by zero).
    matrix = np.column_stack(arrays)
    row_ranges = np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)
    if not np.any(row_ranges > 0):
        return {"statistic": 0.0, "p_value": 1.0, "kendall_w": 0.0, "n": len(target_query_ids)}

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=RuntimeWarning,
                module=r"scipy\.stats\._stats_py",
            )
            stat, p_value = scipy_stats.friedmanchisquare(*arrays)
        if math.isnan(stat) or math.isnan(p_value):
            return {"statistic": 0.0, "p_value": 1.0, "kendall_w": 0.0, "n": len(target_query_ids)}
        n_queries = len(target_query_ids)
        k_baselines = len(baselines)
        kendall_w = float(stat) / (n_queries * (k_baselines - 1)) if n_queries > 0 and k_baselines > 1 else 0.0
        return {"statistic": float(stat), "p_value": float(p_value), "kendall_w": kendall_w, "n": n_queries}
    except (ValueError, TypeError, RuntimeWarning):
        return {"statistic": None, "p_value": None, "kendall_w": None, "n": len(target_query_ids)}


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return significance flags after Holm-Bonferroni correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * m
    for rank, (idx, p) in enumerate(indexed, start=1):
        threshold = alpha / (m - rank + 1)
        if p <= threshold:
            significant[idx] = True
        else:
            break
    return significant


def holm_adjusted_p_values(p_values: list[float]) -> list[float]:
    """Return adjusted p-values using Holm-Bonferroni method."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m
    cum_max = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        val = p * (m - rank + 1)
        cum_max = max(cum_max, val)
        adjusted[idx] = min(cum_max, 1.0)
    return adjusted


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(p_values)
    return [p <= alpha / m if m else False for p in p_values]


def compute_multiple_comparison_correction(
    p_values: list[float | None],
    config: StatsConfig,
) -> tuple[list[bool], list[float], list[float]]:
    """
    Computes FWER multiple-comparison correction returning:
    - sig_flags: list of bools
    - adj_p_vals: list of floats
    - adj_alphas: list of per-test adjusted alpha thresholds (alpha_adjusted floats)
    """
    m = len(p_values)
    if m == 0:
        return [], [], []

    clean = [p if p is not None and not math.isnan(p) else 1.0 for p in p_values]
    indexed = sorted(enumerate(clean), key=lambda x: x[1])

    sig_flags = [False] * m
    adj_p = [1.0] * m
    adj_alphas = [config.alpha] * m

    if config.correction_method == "holm":
        all_passed = True
        cum_max = 0.0
        for rank, (idx, p) in enumerate(indexed, start=1):
            step_alpha = config.alpha / (m - rank + 1)
            adj_alphas[idx] = step_alpha
            p_adj_val = min(max(cum_max, p * (m - rank + 1)), 1.0)
            cum_max = p_adj_val
            adj_p[idx] = p_adj_val

            if all_passed and p <= step_alpha:
                sig_flags[idx] = True
            else:
                all_passed = False
                sig_flags[idx] = False

    elif config.correction_method == "bonferroni":
        bonf_alpha = config.alpha / m
        adj_alphas = [bonf_alpha] * m
        for idx, p in enumerate(clean):
            adj_p[idx] = min(p * m, 1.0)
            sig_flags[idx] = p <= bonf_alpha
    else:
        adj_alphas = [config.alpha] * m
        for idx, p in enumerate(clean):
            adj_p[idx] = p
            sig_flags[idx] = p < config.alpha

    return sig_flags, adj_p, adj_alphas


def apply_multiple_comparison_correction(
    p_values: list[float | None],
    config: StatsConfig,
) -> list[bool]:
    sig_flags, _, _ = compute_multiple_comparison_correction(p_values, config)
    return sig_flags


def _pairwise_wilcoxon_family(
    records: list[dict[str, Any]],
    baselines: list[str],
    metrics: list[str],
    config: StatsConfig,
    family: str,
) -> list[dict[str, Any]]:
    """
    Wilcoxon pairwise tests for one metric family with per-metric Holm correction.

    Holm is applied within this family only (across baseline pairs for each metric).
    Families are never pooled together.
    """
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        metric_pairs: list[tuple[str, str, np.ndarray, np.ndarray, dict[str, Any]]] = []
        for b_a, b_b in combinations(baselines, 2):
            vec_a, vec_b, _ = paired_metric_vectors(records, b_a, b_b, metric)
            if len(vec_a) == 0:
                continue
            wilcox = wilcoxon_signed_rank_test(vec_a, vec_b)
            metric_pairs.append((b_a, b_b, vec_a, vec_b, wilcox))

        if not metric_pairs:
            continue

        p_vals = [w["p_value"] for _, _, _, _, w in metric_pairs]
        sig_flags, adj_p_vals, adj_alphas = compute_multiple_comparison_correction(p_vals, config)

        for (b_a, b_b, vec_a, vec_b, wilcox), sig, p_adj, alpha_adj in zip(
            metric_pairs, sig_flags, adj_p_vals, adj_alphas
        ):
            diff_ci = bootstrap_paired_difference_ci(
                vec_a, vec_b, config, alpha_override=alpha_adj
            )
            rows.append(
                {
                    "metric": metric,
                    "family": family,
                    "baseline_a": b_a,
                    "baseline_b": b_b,
                    "mean_a": float(np.mean(vec_a)),
                    "mean_b": float(np.mean(vec_b)),
                    "delta": diff_ci["delta"],
                    "ci_lower": diff_ci["ci_lower"],
                    "ci_upper": diff_ci["ci_upper"],
                    "rank_delta": diff_ci["rank_delta"],
                    "rank_ci_lower": diff_ci["rank_ci_lower"],
                    "rank_ci_upper": diff_ci["rank_ci_upper"],
                    "alpha_adjusted": alpha_adj,
                    "p_value": wilcox["p_value"],
                    "p_uncorrected": wilcox["p_value"],
                    "p_value_corrected": p_adj,
                    "significant": sig,
                    "effect_size": wilcox["effect_size"],
                    "test": "wilcoxon",
                    "n_pairs": wilcox["n"],
                    "stars": significance_stars(
                        p_adj if config.correction_method != "none" else wilcox["p_value"],
                        config.alpha,
                    )
                    if sig
                    else "",
                }
            )
    return rows


def compute_baseline_summary_with_ci(
    records: list[dict[str, Any]],
    baselines: list[str],
    config: StatsConfig,
) -> dict[str, dict[str, Any]]:
    """Per-baseline means with bootstrap CIs (quality, latency, telemetry)."""
    summary: dict[str, dict[str, Any]] = {}
    metrics = list(QUALITY_METRICS) + ["latency_sec"] + list(TELEMETRY_METRICS)

    for baseline in baselines:
        summary[baseline] = {}
        b_records = [r for r in records if r["baseline"] == baseline]

        tp = sum(1 for r in b_records if r["outcome"] == "TP")
        fp = sum(1 for r in b_records if r["outcome"] == "FP")
        tn = sum(1 for r in b_records if r["outcome"] == "TN")
        fn = sum(1 for r in b_records if r["outcome"] == "FN")
        total_q = len({r["query_id"] for r in b_records})
        summary[baseline]["classification"] = compute_classification_metrics(tp, fp, tn, fn, total_q)

        for metric in metrics:
            use_answered = metric in ANSWERED_METRIC_SET
            if use_answered:
                vals = np.array(
                    [
                        float(r[metric])
                        for r in b_records
                        if r["outcome"] in ANSWERED_OUTCOMES
                        and r.get(metric) is not None
                        and not math.isnan(float(r[metric]))
                    ],
                    dtype=float,
                )
            else:
                vals = np.array(
                    [
                        float(r[metric])
                        for r in b_records
                        if r.get(metric) is not None
                        and not (isinstance(r[metric], float) and math.isnan(r[metric]))
                    ],
                    dtype=float,
                )
            ci = bootstrap_ci(vals, config)
            summary[baseline][metric] = {
                "mean": ci["mean"],
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "n": ci["n"],
                "filter": "TP+FP" if use_answered else "all",
            }

    return summary


def compute_pairwise_comparisons(
    records: list[dict[str, Any]],
    baselines: list[str],
    config: StatsConfig,
) -> list[dict[str, Any]]:
    """
    Pairwise Wilcoxon + FWER-synchronized bootstrap CIs with rank-based delta.

    Quality and telemetry families are corrected separately (Holm within family);
    p-values are never pooled across families. Safety uses McNemar on answerability.
    """
    rows: list[dict[str, Any]] = []

    # Quality family (incl. latency) — existing behavior, tagged for clarity.
    rows.extend(
        _pairwise_wilcoxon_family(
            records,
            baselines,
            list(QUALITY_METRICS) + ["latency_sec"],
            config,
            family="quality",
        )
    )

    # Telemetry family — separate Holm; do not pool with quality p-values.
    rows.extend(
        _pairwise_wilcoxon_family(
            records,
            baselines,
            list(TELEMETRY_METRICS),
            config,
            family="telemetry",
        )
    )

    # McNemar for answerability correctness (safety family)
    mcnemar_pairs: list[tuple[str, str, dict[str, Any]]] = []
    for b_a, b_b in combinations(baselines, 2):
        mcnemar = mcnemar_answerability_test(records, b_a, b_b)
        if mcnemar["n"] == 0:
            continue
        mcnemar_pairs.append((b_a, b_b, mcnemar))

    if mcnemar_pairs:
        p_vals_mc = [m["p_value"] for _, _, m in mcnemar_pairs]
        sig_flags_mc, adj_p_mc, adj_alphas_mc = compute_multiple_comparison_correction(p_vals_mc, config)

        for (b_a, b_b, mcnemar), sig, p_adj, alpha_adj in zip(
            mcnemar_pairs, sig_flags_mc, adj_p_mc, adj_alphas_mc
        ):
            b_cnt = mcnemar["b"]
            c_cnt = mcnemar["c"]
            odds_ratio = (b_cnt / c_cnt) if c_cnt > 0 else (b_cnt if b_cnt > 0 else 1.0)
            rows.append(
                {
                    "metric": "answerability_correct",
                    "family": "safety",
                    "baseline_a": b_a,
                    "baseline_b": b_b,
                    "mean_a": None,
                    "mean_b": None,
                    "delta": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "rank_delta": None,
                    "rank_ci_lower": None,
                    "rank_ci_upper": None,
                    "alpha_adjusted": alpha_adj,
                    "p_value": mcnemar["p_value"],
                    "p_uncorrected": mcnemar["p_value"],
                    "p_value_corrected": p_adj,
                    "significant": sig,
                    "effect_size": odds_ratio,
                    "test": "mcnemar",
                    "n_pairs": mcnemar["n"],
                    "discordant_b": b_cnt,
                    "discordant_c": c_cnt,
                    "stars": significance_stars(
                        p_adj if config.correction_method != "none" else mcnemar["p_value"],
                        config.alpha,
                    )
                    if sig
                    else "",
                }
            )

    return rows


def compute_friedman_tests(
    records: list[dict[str, Any]],
    baselines: list[str],
) -> dict[str, dict[str, Any]]:
    if len(baselines) < 3:
        return {}
    results = {}
    for metric in list(QUALITY_METRICS) + ["latency_sec"] + list(TELEMETRY_METRICS):
        results[metric] = friedman_omnibus_test(records, baselines, metric)
    return results


def extract_significant_improvements(pairwise: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Highlight significant positive improvements (higher metric, lower latency)."""
    highlights = []
    for row in pairwise:
        if not row.get("significant") or row.get("p_value") is None:
            continue
        if row["test"] != "wilcoxon" or row.get("delta") is None:
            continue
        metric = row["metric"]
        delta = row["delta"]
        improved = delta < 0 if metric == "latency_sec" else delta > 0
        if improved:
            highlights.append(row)
    highlights.sort(key=lambda r: r["p_value"])
    return highlights


def compute_statistical_analysis(
    data: dict,
    config: StatsConfig | None = None,
    baselines: list[str] | None = None,
) -> dict[str, Any]:
    """Top-level entry: per-query prep, CIs, pairwise tests, Friedman."""
    config = config or StatsConfig()
    records, resolved_baselines = prepare_per_query_records(data, baselines)
    if not records or not resolved_baselines:
        return {
            "enabled": config.enable_stats,
            "config": config.to_dict(),
            "records": [],
            "baselines": [],
            "baseline_summary": {},
            "pairwise": [],
            "friedman": {},
            "significant_improvements": [],
            "filtering_note": (
                "Quality and telemetry metrics averaged over TP+FP (model answered). "
                "TN/FP unanswerable quality scores are NaN. "
                "Holm correction is within family (quality | telemetry | safety). "
                "Safety metrics use all queries."
            ),
        }

    baseline_summary = compute_baseline_summary_with_ci(records, resolved_baselines, config)
    pairwise = compute_pairwise_comparisons(records, resolved_baselines, config)
    friedman = compute_friedman_tests(records, resolved_baselines)
    significant = extract_significant_improvements(pairwise)

    return {
        "enabled": config.enable_stats,
        "config": config.to_dict(),
        "records": records,
        "baselines": resolved_baselines,
        "baseline_summary": baseline_summary,
        "pairwise": pairwise,
        "friedman": friedman,
        "significant_improvements": significant,
        "filtering_note": (
            "Quality and telemetry metrics averaged over TP+FP (model answered). "
            "TN/FP unanswerable quality scores are NaN. "
            "Holm correction is within family (quality | telemetry | safety). "
            "Safety metrics use all queries."
        ),
    }