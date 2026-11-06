from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ablation_cuda.structured_context_intervention import (
    InterventionVariant,
    build_two_arm_plan,
    validate_fixture_activation,
)


@pytest.fixture
def baseline_config():
    return {
        "rag_components": {
            "static_graph_context": True,
            "graph_expansion": True,
        },
        "context_format": "structured",
        "generation": {"temperature": 0},
    }


@pytest.mark.parametrize(
    ("variant", "path", "control", "treatment", "evidence_invariant"),
    [
        (InterventionVariant.STATIC_GRAPH_RELATIONS, "rag_components.static_graph_context", True, False, False),
        (InterventionVariant.DYNAMIC_ENRICHMENT, "rag_components.graph_expansion", True, False, False),
        (InterventionVariant.FORMAT_ONLY, "context_format", "structured", "plain", True),
    ],
)
def test_plan_is_two_arm_immutable_and_explicit(
    baseline_config, variant, path, control, treatment, evidence_invariant
):
    plan = build_two_arm_plan(baseline_config, variant)

    assert len(plan.arms) == 2
    assert plan.arms[0].arm_id == "control"
    assert plan.arms[1].arm_id == "intervention"
    assert plan.arms[0].config_value(path) == control
    assert plan.arms[1].config_value(path) == treatment
    assert plan.evidence_must_match is evidence_invariant
    assert plan.plan_hash == build_two_arm_plan(baseline_config, variant).plan_hash
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        plan.variant = InterventionVariant.FORMAT_ONLY
    assert baseline_config["rag_components"]["static_graph_context"] is True


def fixture_for(plan):
    return {
        "arms": {
            "control": {
                "prompt": "Use the structured context. Evidence: E1",
                "evidence": ["E1"],
                "config": plan.arms[0].config,
                "activation_count": 0,
            },
            "intervention": {
                "prompt": "Use the altered context. Evidence: E1",
                "evidence": ["E1"],
                "config": plan.arms[1].config,
                "activation_count": 1,
            },
        }
    }


@pytest.mark.parametrize("variant", list(InterventionVariant))
def test_fixture_validation_requires_prompt_change_and_checks_hashes_and_counts(
    baseline_config, variant
):
    plan = build_two_arm_plan(baseline_config, variant)
    report = validate_fixture_activation(plan, fixture_for(plan))

    assert report.valid
    assert report.prompt_hashes["control"] != report.prompt_hashes["intervention"]
    assert report.config_hashes["control"] != report.config_hashes["intervention"]
    assert report.activation_counts == {"control": 0, "intervention": 1}
    assert report.fixture_hash


def test_format_only_fixture_requires_evidence_invariance(baseline_config):
    plan = build_two_arm_plan(baseline_config, InterventionVariant.FORMAT_ONLY)
    fixture = fixture_for(plan)
    fixture["arms"]["intervention"]["evidence"] = ["E2"]

    report = validate_fixture_activation(plan, fixture)

    assert not report.valid
    assert "evidence" in " ".join(report.errors).lower()


def test_activation_validation_fails_closed_for_no_prompt_change_bad_config_and_counts(
    baseline_config,
):
    plan = build_two_arm_plan(baseline_config, InterventionVariant.DYNAMIC_ENRICHMENT)
    fixture = fixture_for(plan)
    fixture["arms"]["intervention"].update(
        prompt=fixture["arms"]["control"]["prompt"],
        config=fixture["arms"]["control"]["config"],
        activation_count=0,
    )

    report = validate_fixture_activation(plan, fixture)

    assert not report.valid
    errors = " ".join(report.errors).lower()
    assert "prompt" in errors
    assert "config" in errors
    assert "activation" in errors


def test_plan_rejects_missing_intervention_config_path(baseline_config):
    del baseline_config["context_format"]
    with pytest.raises(ValueError, match="context_format"):
        build_two_arm_plan(baseline_config, InterventionVariant.FORMAT_ONLY)


def test_plan_config_is_recursively_immutable(baseline_config):
    plan = build_two_arm_plan(baseline_config, InterventionVariant.FORMAT_ONLY)
    with pytest.raises(TypeError):
        plan.arms[0].config["generation"]["temperature"] = 1


def test_evidence_may_change_for_content_interventions(baseline_config):
    plan = build_two_arm_plan(baseline_config, InterventionVariant.STATIC_GRAPH_RELATIONS)
    fixture = fixture_for(plan)
    fixture["arms"]["intervention"]["evidence"] = ["E2"]

    assert validate_fixture_activation(plan, fixture).valid
