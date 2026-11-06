"""Compact telemetry derived from raw pre-sampling vocabulary logits."""
from __future__ import annotations

import math
from typing import Iterable


def topk_telemetry(logits: Iterable[float], emitted_token_id: int, *, ks=(20, 50, 100)) -> dict:
    values = [float(x) for x in logits]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("logits must be a non-empty finite vector")
    if emitted_token_id < 0 or emitted_token_id >= len(values):
        raise ValueError("emitted token id is outside the vocabulary")
    peak = max(values)
    log_z = peak + math.log(math.fsum(math.exp(x - peak) for x in values))
    sorted_values = sorted(values, reverse=True)
    result = {
        "token_id": int(emitted_token_id),
        "logprob": values[emitted_token_id] - log_z,
        "nll_nats": log_z - values[emitted_token_id],
        "nll_bits_per_token": (log_z - values[emitted_token_id]) / math.log(2),
    }
    for requested in ks:
        if requested < 1:
            raise ValueError("K values must be positive")
        top = sorted_values[: min(requested, len(values))]
        top_peak = top[0]
        local_z = top_peak + math.log(math.fsum(math.exp(x - top_peak) for x in top))
        local_probs = [math.exp(x - local_z) for x in top]
        result[f"entropy_k{requested}_bits"] = -math.fsum(p * math.log2(p) for p in local_probs if p > 0)
        result[f"retained_mass_k{requested}"] = min(1.0, math.fsum(math.exp(x - log_z) for x in top))
    return result


def summarize_token_telemetry(tokens: list[dict]) -> dict:
    if not tokens:
        return {"token_count": 0, "mean_nll_nats": None, "mean_nll_bits_per_token": None}
    return {
        "token_count": len(tokens),
        "mean_nll_nats": math.fsum(t["nll_nats"] for t in tokens) / len(tokens),
        "mean_nll_bits_per_token": math.fsum(t["nll_bits_per_token"] for t in tokens) / len(tokens),
    }
