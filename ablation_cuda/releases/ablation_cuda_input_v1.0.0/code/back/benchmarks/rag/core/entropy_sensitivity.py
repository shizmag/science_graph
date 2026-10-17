"""Numerically stable top-K telemetry with the full vocabulary as denominator."""

from __future__ import annotations

import math
from typing import Iterable


def compute_topk_distribution_metrics(logits: Iterable[float], top_ks=(20, 50, 100)) -> dict[str, float]:
    """Return local top-K entropy and retained full-softmax mass for each K.

    H@K is the Shannon entropy after renormalizing only the K largest values,
    matching the estimator used in the submitted paper for K=50. M@K is the
    probability mass those same K tokens carry under the full-vocabulary
    softmax. ``top1_prob_full`` uses the full vocabulary denominator.
    """
    values = [float(value) for value in logits]
    if not values:
        raise ValueError("logits must contain at least one value")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("logits must all be finite")
    ks = tuple(int(k) for k in top_ks)
    if any(k <= 0 for k in ks):
        raise ValueError("top-K values must be positive integers")

    max_value = max(values)
    exp_all = [math.exp(value - max_value) for value in values]
    denominator = math.fsum(exp_all)
    log_z = max_value + math.log(denominator)
    ordered = sorted(values, reverse=True)
    result = {
        "top1_prob_full": math.exp(ordered[0] - log_z),
        "logit_margin_top1_top2": ordered[0] - ordered[1] if len(ordered) > 1 else 0.0,
    }
    for requested_k in ks:
        k = min(requested_k, len(values))
        top_values = ordered[:k]
        local_max = top_values[0]
        local_exp = [math.exp(value - local_max) for value in top_values]
        local_denominator = math.fsum(local_exp)
        local_probabilities = [value / local_denominator for value in local_exp]
        entropy = -math.fsum(p * math.log2(p) for p in local_probabilities if p > 0)
        retained_mass = math.fsum(math.exp(value - log_z) for value in top_values)
        suffix = str(requested_k)
        result[f"entropy_k{suffix}_bits"] = entropy
        result[f"retained_mass_k{suffix}"] = min(1.0, retained_mass)
        result[f"top1_prob_local_k{suffix}"] = local_probabilities[0]
    return result


def summarize_emitted_logprobs(logprobs: Iterable[float]) -> dict[str, float | int]:
    """Summarize the likelihood of the actually emitted continuation tokens."""
    values = [float(value) for value in logprobs]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("log probabilities must all be finite")
    if not values:
        return {"n_tokens": 0, "mean_log_likelihood_nats_per_token": math.nan, "mean_nll_bits_per_token": math.nan}
    mean_logprob = math.fsum(values) / len(values)
    return {
        "n_tokens": len(values),
        "mean_log_likelihood_nats_per_token": mean_logprob,
        "mean_nll_bits_per_token": -mean_logprob / math.log(2.0),
    }
