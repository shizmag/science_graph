"""Small pure-Python reference for CUDA logit metric tests."""

from __future__ import annotations

import math
from typing import Iterable


def reference_telemetry(logits: Iterable[float], token_id: int,
                        top_k_values=(20, 50, 100), store_k: int = 5) -> dict:
    values = [float(value) for value in logits]
    if not values:
        raise ValueError("logits must be nonempty")
    maximum = max(values)
    log_z = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    chosen_logprob = values[token_id] - log_z
    result = {
        "token_id": token_id,
        "logprob": chosen_logprob,
        "nll_nats": -chosen_logprob,
        "nll_bits_per_token": -chosen_logprob / math.log(2.0),
        "logit_margin": values[order[0]] - values[order[1]] if len(values) > 1 else 0.0,
        "top1_prob_full": math.exp(values[order[0]] - log_z),
    }
    for k_req in top_k_values:
        selected = [values[index] for index in order[:min(k_req, len(values))]]
        local_log_z = max(selected) + math.log(sum(math.exp(value - max(selected)) for value in selected))
        local = [math.exp(value - local_log_z) for value in selected]
        result[f"entropy_k{k_req}_bits"] = -sum(p * math.log2(p) for p in local if p > 0)
        result[f"retained_mass_k{k_req}"] = sum(math.exp(value - log_z) for value in selected)
        result[f"top1_prob_local_k{k_req}"] = max(local)
    return result
