"""Pure planning and fixture validation for structured-context ablations.

This module deliberately does not invoke retrieval, generation, or experiment
runners.  It creates a reproducible two-arm specification and validates paired
fixture observations before any downstream runner is considered.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class InterventionVariant(str, Enum):
    """The three distinct causal axes supported by this workstream."""

    STATIC_GRAPH_RELATIONS = "static_graph_relations"
    DYNAMIC_ENRICHMENT = "dynamic_enrichment"
    FORMAT_ONLY = "format_only"


@dataclass(frozen=True)
class VariantSpec:
    config_path: str
    control_value: Any
    intervention_value: Any
    evidence_must_match: bool
    description: str


VARIANT_SPECS: Mapping[InterventionVariant, VariantSpec] = {
    InterventionVariant.STATIC_GRAPH_RELATIONS: VariantSpec(
        "rag_components.static_graph_context", True, False, False,
        "Disable static graph-relation context while retaining dynamic enrichment settings.",
    ),
    InterventionVariant.DYNAMIC_ENRICHMENT: VariantSpec(
        "rag_components.graph_expansion", True, False, False,
        "Disable dynamic graph enrichment while retaining static graph-relation context.",
    ),
    InterventionVariant.FORMAT_ONLY: VariantSpec(
        "context_format", "structured", "plain", True,
        "Change context serialization only; selected evidence must remain identical.",
    ),
}


class FrozenMap(Mapping[str, Any]):
    """Small recursively immutable mapping with deterministic insertion order."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, values: Mapping[str, Any]):
        items = tuple((key, freeze(value)) for key, value in values.items())
        self._items = items
        self._lookup = dict(items)

    def __getitem__(self, key: str) -> Any:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._lookup)

    def __len__(self) -> int:
        return len(self._lookup)

    def __repr__(self) -> str:
        return f"FrozenMap({dict(self._items)!r})"


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=repr)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _at_path(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Required config path is missing: {path}")
        value = value[part]
    return value


def _with_path(config: Mapping[str, Any], path: str, value: Any) -> dict[str, Any]:
    result = _plain(config)
    cursor = result
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Required config path is missing: {path}")
        cursor = child
    cursor[parts[-1]] = value
    return result


@dataclass(frozen=True)
class ArmPlan:
    arm_id: str
    role: str
    config: FrozenMap

    def config_value(self, path: str) -> Any:
        return _at_path(self.config, path)


@dataclass(frozen=True)
class TwoArmPlan:
    variant: InterventionVariant
    description: str
    config_path: str
    evidence_must_match: bool
    arms: tuple[ArmPlan, ArmPlan]
    plan_hash: str


@dataclass(frozen=True)
class FixtureValidation:
    valid: bool
    errors: tuple[str, ...]
    fixture_hash: str
    plan_hash: str
    prompt_hashes: FrozenMap
    evidence_hashes: FrozenMap
    config_hashes: FrozenMap
    activation_counts: FrozenMap


def build_two_arm_plan(
    baseline_config: Mapping[str, Any], variant: InterventionVariant | str
) -> TwoArmPlan:
    """Build a frozen control/intervention plan without mutating input config."""
    try:
        variant = InterventionVariant(variant)
    except ValueError as exc:
        raise ValueError(f"Unknown intervention variant: {variant!r}") from exc
    spec = VARIANT_SPECS[variant]
    current = _at_path(baseline_config, spec.config_path)
    if current != spec.control_value:
        raise ValueError(
            f"Baseline {spec.config_path} must be {spec.control_value!r}; got {current!r}"
        )
    control = _plain(baseline_config)
    treatment = _with_path(baseline_config, spec.config_path, spec.intervention_value)
    arms = (
        ArmPlan("control", "control", freeze(control)),
        ArmPlan("intervention", "intervention", freeze(treatment)),
    )
    identity = {
        "variant": variant.value,
        "description": spec.description,
        "config_path": spec.config_path,
        "evidence_must_match": spec.evidence_must_match,
        "arms": [{"arm_id": arm.arm_id, "role": arm.role, "config": arm.config} for arm in arms],
    }
    return TwoArmPlan(
        variant=variant,
        description=spec.description,
        config_path=spec.config_path,
        evidence_must_match=spec.evidence_must_match,
        arms=arms,
        plan_hash=_sha256(identity),
    )


def validate_fixture_activation(
    plan: TwoArmPlan, fixture: Mapping[str, Any]
) -> FixtureValidation:
    """Validate paired fixture observations; report all detected violations."""
    errors: list[str] = []
    arm_rows = fixture.get("arms") if isinstance(fixture, Mapping) else None
    if not isinstance(arm_rows, Mapping):
        arm_rows = {}
        errors.append("Fixture must contain an arms mapping.")

    prompts: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    configs: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    for planned in plan.arms:
        observed = arm_rows.get(planned.arm_id)
        if not isinstance(observed, Mapping):
            errors.append(f"Missing fixture arm: {planned.arm_id}.")
            continue
        prompt = observed.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"Arm {planned.arm_id} must have a non-empty prompt.")
        else:
            prompts[planned.arm_id] = prompt
        if "evidence" not in observed:
            errors.append(f"Arm {planned.arm_id} is missing evidence.")
        else:
            evidence[planned.arm_id] = observed["evidence"]
        config = observed.get("config")
        if not isinstance(config, Mapping):
            errors.append(f"Arm {planned.arm_id} must provide its effective config.")
        else:
            configs[planned.arm_id] = config
            if _plain(config) != _plain(planned.config):
                errors.append(f"Arm {planned.arm_id} config does not match the immutable plan.")
        count = observed.get("activation_count")
        counts[planned.arm_id] = count
        expected_count = 0 if planned.role == "control" else 1
        if isinstance(count, bool) or count != expected_count:
            errors.append(
                f"Arm {planned.arm_id} activation_count must be {expected_count}; got {count!r}."
            )

    if set(prompts) == {"control", "intervention"}:
        if prompts["control"] == prompts["intervention"]:
            errors.append("Intervention must produce an intended prompt change.")
    if plan.evidence_must_match and set(evidence) == {"control", "intervention"}:
        if _canonical_bytes(evidence["control"]) != _canonical_bytes(evidence["intervention"]):
            errors.append("Evidence must remain invariant for this intervention variant.")

    return FixtureValidation(
        valid=not errors,
        errors=tuple(errors),
        fixture_hash=_sha256(fixture),
        plan_hash=plan.plan_hash,
        prompt_hashes=freeze({key: _sha256(value) for key, value in prompts.items()}),
        evidence_hashes=freeze({key: _sha256(value) for key, value in evidence.items()}),
        config_hashes=freeze({key: _sha256(value) for key, value in configs.items()}),
        activation_counts=freeze(counts),
    )
