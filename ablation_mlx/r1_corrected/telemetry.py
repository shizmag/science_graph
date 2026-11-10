"""Compact predictive-distribution telemetry for the corrected R1 run.

This module deliberately has no MLX dependency.  MLX-LM 0.31.3 exposes a
full-vocabulary *log-probability* vector from ``generate_step`` immediately
before greedy selection.  The reducer below consumes that one vector, emits
only compact per-token statistics, and discards the vector before the next
generation step.

The definitions are the historical CUDA definitions.  In particular,
``entropy_k20_bits`` is the Shannon entropy after renormalizing the selected
20 values, while ``retained_mass_k20`` is the selected mass under the full
vocabulary distribution.  These are intentionally not the legacy
``EntropyTelemetry`` fields in :mod:`protocol`, which describe a different
full-distribution entropy API.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


TELEMETRY_PROTOCOL_VERSION = "r1-mlx-predictive-distribution-v1"
TELEMETRY_IMPLEMENTATION = "mlx_lm.generate_step@0.31.3"
TELEMETRY_SOURCE = "mlx_lm.generate_step.full_vocabulary_logprobs"
TELEMETRY_TOP_KS = (20, 50, 100)
STOP_TOKEN_IDS = (151643, 151645, 151683)
TELEMETRY_UNITS = {
    "entropy": "bits",
    "logprob": "nats",
    "nll_nats": "nats",
    "nll_bits_per_token": "bits",
    "probabilities": "probability",
    "logit_margin": "logit_scale",
}
TELEMETRY_AGGREGATION = "arithmetic_mean_over_all_compact_telemetry_rows"

_STOP_STATES = frozenset({"continue", "eos", "max_tokens", "context_limit"})
_DENSE_KEYS = frozenset({"logits", "logprobs", "distribution", "vocabulary"})
_AGGREGATION = TELEMETRY_AGGREGATION


class TelemetryCompatibilityError(ValueError):
    """Raised when records use a different telemetry protocol."""


def _finite_values(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("predictive distribution must be a non-empty finite vector")
    return result


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _validated_ks(requested_ks: Sequence[int]) -> tuple[int, ...]:
    """Normalize K values while rejecting ambiguous or empty specifications."""

    ks = tuple(int(value) for value in requested_ks)
    if not ks or any(isinstance(value, bool) or value < 1 for value in ks):
        raise ValueError("K values must be a non-empty sequence of positive integers")
    if len(set(ks)) != len(ks):
        raise ValueError("K values must be unique")
    return ks


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _finite_number(value: Any, *, field: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise TelemetryCompatibilityError(f"Telemetry field {field!r} is not finite")
    return float(value)


def _assert_no_dense_values(value: Any) -> None:
    """Reject dense vocabulary payloads even when nested in an extension field."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _DENSE_KEYS:
                raise TelemetryCompatibilityError(
                    "Dense vocabulary telemetry cannot be persisted"
                )
            _assert_no_dense_values(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_dense_values(child)


def reduce_token_distribution(
    logprobs: Iterable[float],
    emitted_token_id: int,
    *,
    requested_ks: Sequence[int] = TELEMETRY_TOP_KS,
) -> dict[str, Any]:
    """Reduce one full predictive distribution to historical compact fields.

    ``generate_step`` returns log-softmax values rather than raw logits.  The
    stable normalization below is invariant to the additive constant between
    those values and raw logits, so it is mathematically identical to the
    historical CUDA calculation while also tolerating a tiny accumulated
    normalization error in a backend-produced log-probability vector.
    """

    values = _finite_values(logprobs)
    token_id = int(emitted_token_id)
    if token_id < 0 or token_id >= len(values):
        raise ValueError("emitted token id is outside the vocabulary")

    ks = _validated_ks(requested_ks)

    peak = max(values)
    # This is the exact stable max-shift denominator used by the historical
    # Python reference.  It also renormalizes MLX's already-normalized vector
    # without changing any probability ratios.
    log_z = peak + math.log(math.fsum(math.exp(value - peak) for value in values))
    sorted_values = sorted(values, reverse=True)
    second = sorted_values[1] if len(sorted_values) > 1 else None
    result: dict[str, Any] = {
        "token_id": token_id,
        "logprob": values[token_id] - log_z,
        "nll_nats": log_z - values[token_id],
        "nll_bits_per_token": (log_z - values[token_id]) / math.log(2),
        "logit_margin": (sorted_values[0] - second) if second is not None else None,
        "top1_prob_full": math.exp(sorted_values[0] - log_z),
    }

    for requested in ks:
        top = sorted_values[: min(requested, len(values))]
        top_peak = top[0]
        local_z = top_peak + math.log(
            math.fsum(math.exp(value - top_peak) for value in top)
        )
        local_probs = [math.exp(value - local_z) for value in top]
        result[f"entropy_k{requested}_bits"] = -math.fsum(
            probability * math.log2(probability)
            for probability in local_probs
            if probability > 0
        )
        result[f"retained_mass_k{requested}"] = min(
            1.0,
            math.fsum(math.exp(value - log_z) for value in top),
        )
        result[f"top1_prob_local_k{requested}"] = math.exp(top[0] - local_z)

    return result


class TokenTelemetryAccumulator:
    """Incrementally collect compact per-token telemetry.

    Only the compact rows and running sums are retained.  No full-vocabulary
    vector is stored by this object, and ``add`` returns after the supplied
    vector has been reduced.
    """

    def __init__(self, *, requested_ks: Sequence[int] = TELEMETRY_TOP_KS):
        self.requested_ks = _validated_ks(requested_ks)
        self._rows: list[dict[str, Any]] = []
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @property
    def token_count(self) -> int:
        return len(self._rows)

    def add(
        self,
        logprobs: Iterable[float],
        emitted_token_id: int,
        *,
        token_text: str | None = None,
        stop_state: str = "continue",
    ) -> dict[str, Any]:
        """Reduce one yielded distribution and append one compact row."""

        if stop_state not in {"continue", "eos", "max_tokens", "context_limit"}:
            raise ValueError(f"Unsupported telemetry stop state: {stop_state}")
        row = reduce_token_distribution(
            logprobs,
            emitted_token_id,
            requested_ks=self.requested_ks,
        )
        row["sequence_index"] = len(self._rows)
        row["stop_state"] = stop_state
        if token_text is not None:
            row["token_text"] = str(token_text)
        self._rows.append(row)
        for key, value in row.items():
            if key == "token_id" or key == "sequence_index" or key in {
                "stop_state",
                "token_text",
            }:
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self._sums[key] = self._sums.get(key, 0.0) + float(value)
                self._counts[key] = self._counts.get(key, 0) + 1
        return dict(row)

    def finalize(
        self,
        *,
        finish_reason: str | None = None,
        stop_token_id: int | None = None,
        invalid_reason: str | None = None,
    ) -> dict[str, Any]:
        """Return the compact versioned payload; EOS rows are included."""

        rows = [dict(row) for row in self._rows]
        result: dict[str, Any] = {
            "protocol_version": TELEMETRY_PROTOCOL_VERSION,
            "implementation": TELEMETRY_IMPLEMENTATION,
            "source": TELEMETRY_SOURCE,
            "top_k": list(self.requested_ks),
            "units": dict(TELEMETRY_UNITS),
            "aggregation": _AGGREGATION,
            "eos_included_in_aggregation": True,
            "token_count": len(rows),
            "tokens": rows,
            "valid": bool(rows) and invalid_reason is None,
            "invalid_reason": invalid_reason,
            "finish_reason": finish_reason,
            "stop_token_id": stop_token_id,
        }
        for requested in self.requested_ks:
            result[f"h{requested}_bits"] = _mean(
                [
                    self._sums[key] / self._counts[key]
                    for key in (f"entropy_k{requested}_bits",)
                    if key in self._counts
                ]
            )
            result[f"retained_mass_k{requested}"] = _mean(
                [
                    self._sums[key] / self._counts[key]
                    for key in (f"retained_mass_k{requested}",)
                    if key in self._counts
                ]
            )
            result[f"mean_top1_prob_local_k{requested}"] = (
                self._sums[f"top1_prob_local_k{requested}"]
                / self._counts[f"top1_prob_local_k{requested}"]
                if f"top1_prob_local_k{requested}" in self._counts
                else None
            )
        for key in (
            "nll_nats",
            "nll_bits_per_token",
            "logprob",
            "logit_margin",
            "top1_prob_full",
        ):
            result[f"mean_{key}"] = (
                self._sums[key] / self._counts[key]
                if key in self._counts
                else None
            )
        if not rows and invalid_reason is None:
            result["valid"] = False
            result["invalid_reason"] = "no_generated_tokens"
        return result


def validate_telemetry_payload(
    payload: Mapping[str, Any],
    *,
    expected_protocol_version: str = TELEMETRY_PROTOCOL_VERSION,
    expected_implementation: str = TELEMETRY_IMPLEMENTATION,
    expected_top_k: Sequence[int] = TELEMETRY_TOP_KS,
) -> None:
    """Reject absent or incompatible new telemetry provenance.

    Legacy ``entropy`` objects are intentionally not upgraded here.  A caller
    must either keep them explicitly marked unavailable/invalid or use a new
    telemetry-enabled namespace.
    """

    if not isinstance(payload, Mapping):
        raise TelemetryCompatibilityError("Telemetry payload must be a mapping")

    _assert_no_dense_values(payload)
    protocol_version = payload.get("protocol_version")
    implementation = payload.get("implementation")
    if protocol_version != expected_protocol_version:
        raise TelemetryCompatibilityError(
            "Telemetry protocol mismatch: "
            f"expected {expected_protocol_version}, got {protocol_version!r}"
        )
    if implementation != expected_implementation:
        raise TelemetryCompatibilityError(
            "Telemetry implementation mismatch: "
            f"expected {expected_implementation}, got {implementation!r}"
        )
    expected_ks = _validated_ks(expected_top_k)
    if payload.get("top_k") != list(expected_ks):
        raise TelemetryCompatibilityError(
            f"Telemetry top_k mismatch: expected {list(expected_ks)!r}, "
            f"got {payload.get('top_k')!r}"
        )
    if payload.get("units") != TELEMETRY_UNITS:
        raise TelemetryCompatibilityError("Telemetry units do not match the protocol")
    if payload.get("aggregation") != _AGGREGATION:
        raise TelemetryCompatibilityError("Telemetry aggregation does not match the protocol")
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        raise TelemetryCompatibilityError("Telemetry payload has no compact token rows")
    if payload.get("source") != TELEMETRY_SOURCE:
        raise TelemetryCompatibilityError("Telemetry payload has an unexpected source")
    if payload.get("eos_included_in_aggregation") is not True:
        raise TelemetryCompatibilityError("Telemetry payload does not declare EOS aggregation")
    token_count = payload.get("token_count")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count != len(tokens):
        raise TelemetryCompatibilityError("Telemetry token_count does not match compact rows")

    valid = payload.get("valid")
    if not isinstance(valid, bool):
        raise TelemetryCompatibilityError("Telemetry valid flag is missing or not boolean")
    invalid_reason = payload.get("invalid_reason")
    if token_count == 0:
        if valid or not isinstance(invalid_reason, str) or not invalid_reason:
            raise TelemetryCompatibilityError(
                "Empty telemetry must be explicitly marked invalid with a reason"
            )
    elif valid and invalid_reason is not None:
        raise TelemetryCompatibilityError(
            "Valid telemetry cannot carry an invalid_reason"
        )
    elif not valid and (not isinstance(invalid_reason, str) or not invalid_reason):
        raise TelemetryCompatibilityError(
            "Invalid telemetry must carry a non-empty invalid_reason"
        )

    stop_token_id = payload.get("stop_token_id")
    if stop_token_id is not None and (
        isinstance(stop_token_id, bool)
        or not isinstance(stop_token_id, int)
        or stop_token_id not in STOP_TOKEN_IDS
    ):
        raise TelemetryCompatibilityError("Telemetry stop_token_id is invalid")
    finish_reason = payload.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise TelemetryCompatibilityError("Telemetry finish_reason is invalid")

    required = {
        "token_id",
        "logprob",
        "nll_nats",
        "nll_bits_per_token",
        "logit_margin",
        "top1_prob_full",
    }
    for requested in expected_ks:
        required.update(
            {
                f"entropy_k{requested}_bits",
                f"retained_mass_k{requested}",
                f"top1_prob_local_k{requested}",
            }
        )
    allowed = required | {"sequence_index", "stop_state", "token_text"}
    means = {
        "nll_nats": "mean_nll_nats",
        "nll_bits_per_token": "mean_nll_bits_per_token",
        "logprob": "mean_logprob",
        "logit_margin": "mean_logit_margin",
        "top1_prob_full": "mean_top1_prob_full",
    }
    for requested in expected_ks:
        means[f"entropy_k{requested}_bits"] = f"h{requested}_bits"
        means[f"retained_mass_k{requested}"] = f"retained_mass_k{requested}"
        means[f"top1_prob_local_k{requested}"] = f"mean_top1_prob_local_k{requested}"

    def check_probability(value: Any, *, field: str) -> float:
        number = _finite_number(value, field=field)
        if number < -1e-9 or number > 1.0 + 1e-9:
            raise TelemetryCompatibilityError(
                f"Telemetry probability {field!r} is outside [0, 1]"
            )
        return number

    row_means: dict[str, list[float]] = {key: [] for key in means}
    for index, row in enumerate(tokens):
        if not isinstance(row, Mapping):
            raise TelemetryCompatibilityError("Telemetry compact rows must be mappings")
        unknown = set(row) - allowed
        missing = required - set(row)
        if missing or unknown:
            raise TelemetryCompatibilityError(
                f"Telemetry compact row schema mismatch; missing={sorted(missing)!r}, "
                f"unknown={sorted(unknown)!r}"
            )
        token_id = row["token_id"]
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise TelemetryCompatibilityError("Telemetry token_id is invalid")
        sequence_index = row.get("sequence_index")
        if sequence_index != index:
            raise TelemetryCompatibilityError("Telemetry sequence_index is not contiguous")
        if row.get("stop_state") not in _STOP_STATES:
            raise TelemetryCompatibilityError("Telemetry stop_state is invalid")
        if "token_text" in row and not isinstance(row["token_text"], str):
            raise TelemetryCompatibilityError("Telemetry token_text is invalid")

        logprob = _finite_number(row["logprob"], field="logprob")
        nll_nats = _finite_number(row["nll_nats"], field="nll_nats")
        nll_bits = _finite_number(row["nll_bits_per_token"], field="nll_bits_per_token")
        if nll_nats < -1e-9 or nll_bits < -1e-9 or logprob > 1e-9:
            raise TelemetryCompatibilityError("Telemetry log-probability/NLL range is invalid")
        row_means["nll_nats"].append(nll_nats)
        row_means["nll_bits_per_token"].append(nll_bits)
        row_means["logprob"].append(logprob)
        margin = row["logit_margin"]
        if margin is not None:
            margin = _finite_number(margin, field="logit_margin")
            if margin < -1e-9:
                raise TelemetryCompatibilityError("Telemetry logit_margin is negative")
            row_means["logit_margin"].append(margin)
        row_means["top1_prob_full"].append(
            check_probability(row["top1_prob_full"], field="top1_prob_full")
        )
        for requested in expected_ks:
            entropy = _finite_number(
                row[f"entropy_k{requested}_bits"],
                field=f"entropy_k{requested}_bits",
            )
            if entropy < -1e-9 or entropy > math.log2(requested) + 1e-8:
                raise TelemetryCompatibilityError("Telemetry entropy range is invalid")
            row_means[f"entropy_k{requested}_bits"].append(entropy)
            row_means[f"retained_mass_k{requested}"].append(
                check_probability(
                    row[f"retained_mass_k{requested}"],
                    field=f"retained_mass_k{requested}",
                )
            )
            row_means[f"top1_prob_local_k{requested}"].append(
                check_probability(
                    row[f"top1_prob_local_k{requested}"],
                    field=f"top1_prob_local_k{requested}",
                )
            )

    for source_key, aggregate_key in means.items():
        expected = _mean(row_means[source_key])
        actual = payload.get(aggregate_key)
        if expected is None:
            if actual is not None:
                raise TelemetryCompatibilityError(
                    f"Empty telemetry aggregate {aggregate_key!r} must be null"
                )
        else:
            actual_number = _finite_number(actual, field=aggregate_key)
            if not math.isclose(actual_number, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise TelemetryCompatibilityError(
                    f"Telemetry aggregate {aggregate_key!r} does not match compact rows"
                )
