"""Frozen, explicit B6 ablation matrix and configuration identity helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


MATRIX_VERSION = "full-b6-matrix-v1"
PAPER_BASELINE = "B5"
INTERNAL_BASELINE = "B6"


@dataclass(frozen=True)
class ArmDefinition:
    arm_id: str
    label: str
    description: str
    overrides: tuple[tuple[str, Any], ...] = ()
    kind: str = "primary"
    dependencies: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    inclusion_reason: str = ""


ARM_DEFINITIONS: tuple[ArmDefinition, ...] = (
    ArmDefinition("B6_full", "B6 full reference", "Fresh internal B6 reference on the selected frozen snapshot.",
                  inclusion_reason="Required same-runtime reference arm."),
    ArmDefinition("B6_minus_score_blending", "B6 − score blending",
                  "Use raw Cross-Encoder scores for final candidate ordering while preserving Cross-Encoder and RRF inputs.",
                  (("rag_components.score_blending", False),),
                  validation=("ranking.score_blending_branch_executed", "ranking.raw_inputs_unchanged"),
                  inclusion_reason="The final ranking code has an isolated normalized CE/RRF blend branch."),
    ArmDefinition("B6_minus_final_reranking", "B6 − final Cross-Encoder reranking",
                  "Disable final candidate Cross-Encoder ranking; retain the shared checkpoint for graph-expander scoring.",
                  (("rag_components.reranker", False),), dependencies=("cross_encoder_checkpoint_shared",),
                  validation=("ranking.stage_status=reranker_disabled",),
                  inclusion_reason="Final ranking has a separate disabled branch; graph expansion remains unchanged."),
    ArmDefinition("B6_minus_dynamic_graph_expander", "B6 − dynamic graph expander",
                  "Disable ExperimentalGraphExpander scoring and prompt enrichment while retaining static graph evidence.",
                  (("rag_components.graph_expansion", False),),
                  validation=("context_and_prompt_assembly.dynamic_graph_expansion_executed=false",),
                  inclusion_reason="The dynamic expander is independently gated from static graph context in this runtime."),
    ArmDefinition("B6_minus_static_graph_context", "B6 − static graph context",
                  "Disable static graph relation text in the generator context while preserving retrieval and dynamic expansion.",
                  (("rag_components.static_graph_context", False),),
                  validation=("context_capture.static_graph_context_disabled",),
                  inclusion_reason="A dedicated static_graph_context gate isolates the build_context graph block."),
    ArmDefinition("B6_minus_intrapaper_graph_candidates", "B6 − intra-paper graph candidate augmentation",
                  "Disable intra-paper candidate augmentation while retaining the graph master path, concept retrieval, and bridge retrieval.",
                  (("graph_retrieval.intra_paper_candidate_augmentation_enabled", False),),
                  validation=("graph_retrieval.branch_execution.intra_paper_branch_executed=false",),
                  inclusion_reason="A dedicated gate isolates intra-paper candidate augmentation from concept and bridge sub-branches."),
    ArmDefinition("B6_minus_concept_retrieval", "B6 − concept retrieval",
                  "Disable ontology concept candidate lookup while preserving master graph and bridge retrieval.",
                  (("graph_retrieval.concept_retrieval_enabled", False), ("rag_components.graph_concept_retrieval", False)),
                  validation=("graph_retrieval.branch_execution.concept_branch_enabled=false",),
                  inclusion_reason="The concept sub-branch has an independent runtime switch."),
    ArmDefinition("B6_minus_bridge_retrieval", "B6 − bridge retrieval",
                  "Disable external bridge candidate lookup while preserving master graph and concept retrieval.",
                  (("graph_retrieval.bridge_retrieval_enabled", False), ("rag_components.graph_bridge_retrieval", False)),
                  validation=("graph_retrieval.branch_execution.bridge_branch_enabled=false",),
                  inclusion_reason="The bridge sub-branch has an independent runtime switch."),
    ArmDefinition("B6_minus_parent_deduplication", "B6 − parent deduplication",
                  "Retain duplicate parent chunks through ranking and context assembly.",
                  (("rag_components.deduplicate_parent_chunks", False),),
                  validation=("ranking.parent_deduplication_enabled=false",),
                  inclusion_reason="Deduplication is applied in independently gated ranking and context code paths."),
    ArmDefinition("B6_minus_lexical_retrieval", "B6 − lexical retrieval",
                  "Disable SQLite FTS5 retrieval while retaining dense E5 retrieval.",
                  (("rag_components.lexical_search", False),),
                  validation=("retrieval.lexical_enabled=false", "retrieval.dense_enabled=true"),
                  inclusion_reason="FTS5 and dense candidate generation are independent paths."),
    ArmDefinition("B6_minus_dense_retrieval", "B6 − dense retrieval",
                  "Disable dense vector retrieval while retaining FTS5; graph query embeddings remain available where the graph path requires them.",
                  (("rag_components.dense_search", False),),
                  validation=("retrieval.dense_enabled=false", "retrieval.lexical_enabled=true"),
                  inclusion_reason="Dense and lexical retrieval are independent candidate-generation paths; graph expansion may still embed the query."),
    ArmDefinition("B6_fixed_budget_context", "B6 fixed-budget context policy",
                  "Replace semantic context trimming with deterministic prefix truncation under the same 12,000-token context and 768-token output limits.",
                  (("rag_components.context_trimming", False),),
                  validation=("context_capture.fixed_budget_policy_executed=true",),
                  inclusion_reason="The replacement policy enforces the exact generator tokenizer budget without score-based chunk pruning."),
    ArmDefinition("B6_interaction_graph_candidate_x_dynamic_expander", "B6 interaction: graph candidates × dynamic expander",
                  "Disable intra-paper graph candidate augmentation and dynamic graph expansion together.",
                  (("graph_retrieval.intra_paper_candidate_augmentation_enabled", False), ("rag_components.graph_expansion", False)),
                  kind="interaction", validation=("both_intervention_paths_disabled",),
                  inclusion_reason="Tests the dependency between graph-derived candidate supply and downstream dynamic expansion."),
    ArmDefinition("B6_interaction_reranker_x_blending", "B6 interaction: final reranker × score blending",
                  "Disable final Cross-Encoder ranking and score blending together; graph-expander Cross-Encoder use is preserved.",
                  (("rag_components.reranker", False), ("rag_components.score_blending", False)),
                  kind="interaction", dependencies=("cross_encoder_checkpoint_shared",),
                  validation=("reranker_disabled", "score_blending_not_executed_without_reranker"),
                  inclusion_reason="Score blending consumes final Cross-Encoder scores, so their joint removal is a dependency interaction rather than two additive primary effects."),
)


def _set_path(value: dict[str, Any], path: str, item: Any) -> None:
    parts = path.split(".")
    current = value
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = item


def canonical_config_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_matrix(base_config: dict[str, Any], *, snapshot_sha256: str,
                   dataset_sha256: str, model_manifest_sha256: str,
                   paper_baseline: str = PAPER_BASELINE) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Resolve all arm configurations before full generation and freeze their hashes."""
    if paper_baseline != "B5":
        raise ValueError("The submitted-paper baseline mapping is frozen to B5 -> internal B6")
    arms: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}
    for definition in ARM_DEFINITIONS:
        effective = copy.deepcopy(base_config)
        for path, item in definition.overrides:
            _set_path(effective, path, item)
        component_config = effective.get("rag_components", {})
        effective_components = {key: component_config.get(key, True) for key in component_config}
        effective_components.update({
            "hyde": False,
            "reranker": bool(component_config.get("reranker", True)),
            "graph_neighbors_in_rrf": bool(component_config.get("graph_neighbors_in_rrf", True)),
            "intent_classifier": bool(component_config.get("intent_classifier", False)),
            "shannon_estimator_enabled": bool(component_config.get("shannon_estimator_enabled", True)),
            "deduplicate_parent_chunks": bool(component_config.get("deduplicate_parent_chunks", True)),
        })
        manifest_effective = {
            "baseline": INTERNAL_BASELINE,
            "paper_baseline": paper_baseline,
            "config": effective,
            "baseline_components": effective_components,
            "overrides": {path: item for path, item in definition.overrides},
        }
        config_hash = canonical_config_hash(manifest_effective)
        arms.append({
            "arm_id": definition.arm_id,
            "label": definition.label,
            "reference_baseline": INTERNAL_BASELINE,
            "paper_baseline_mapping": paper_baseline,
            "intervention": definition.description,
            "kind": definition.kind,
            "enabled_execution_paths": _enabled_paths(effective),
            "disabled_execution_paths": {path: item for path, item in definition.overrides if item is False},
            "effective_configuration_hash": config_hash,
            "expected_dependency_changes": list(definition.dependencies),
            "validation_requirements": list(definition.validation),
            "inclusion_reason": definition.inclusion_reason,
            "cache_compatibility": {
                "shared_query_embedding": True,
                "shared_cross_encoder_pair_score": True,
                "retrieval_cache": "only when query, corpus, model, and all retrieval-affecting settings match",
                "generation_cache": "only when exact serialized prompt, model, tokenizer, and decoding hashes match",
            },
            "status": "PLANNED",
        })
        configs[definition.arm_id] = effective
    if len({arm["arm_id"] for arm in arms}) != len(arms):
        raise ValueError("Arm IDs must be unique")
    return arms, configs


def _enabled_paths(config: dict[str, Any]) -> dict[str, Any]:
    components = config.get("rag_components", {})
    graph = config.get("graph_retrieval", {})
    return {
        "lexical_retrieval": bool(components.get("lexical_search", True)),
        "dense_retrieval": bool(components.get("dense_search", True)),
        "final_cross_encoder_reranking": bool(components.get("reranker", True)),
        "score_blending": bool(components.get("score_blending", True)),
        "dynamic_graph_expansion": bool(components.get("graph_expansion", True)),
        "static_graph_context": bool(components.get("static_graph_context", components.get("graph_expansion", True))),
        "graph_master_candidates": bool(graph.get("enabled", False)),
        "intra_paper_candidate_augmentation": bool(graph.get("intra_paper_candidate_augmentation_enabled", True)),
        "concept_retrieval": bool(graph.get("concept_retrieval_enabled", False)
                                   or components.get("graph_concept_retrieval", False)),
        "bridge_retrieval": bool(graph.get("bridge_retrieval_enabled", False)
                                  or components.get("graph_bridge_retrieval", False)),
        "graph_selected_sources_card": bool(graph.get("selected_sources_card_enabled", False)
                                             or components.get("graph_selected_sources_card", False)),
        "ontology_query_enrichment": bool(components.get("graph_ontology_lookup", True)),
        "parent_deduplication": bool(components.get("deduplicate_parent_chunks", True)),
        "semantic_context_trimming": bool(components.get("context_trimming", True)),
        "fixed_budget_context_guard": True,
    }


def manifest_yaml(arms: list[dict[str, Any]], *, dataset_sha256: str,
                  snapshot_sha256: str, model_manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "matrix_version": MATRIX_VERSION,
        "paper_baseline": PAPER_BASELINE,
        "internal_baseline": INTERNAL_BASELINE,
        "query_count": 375,
        "arm_count": len(arms),
        "expected_query_arm_operations": 375 * len(arms),
        "dataset_sha256": dataset_sha256,
        "external_db_snapshot_sha256": snapshot_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "arms": arms,
        "excluded_or_inactive_components": [
            {"component": "HyDE", "reason": "Explicitly disabled by B6 baseline protocol; not a B6 execution path."},
            {"component": "dynamic_alpha_blending", "reason": "B6 uses score fusion settings with the frozen trace; not a distinct independent causal component for this matrix."},
            {"component": "graph_neighbors_in_rrf", "reason": "Graph retrieval master path is enabled in the reference; the fallback branch is guarded by not graph_retrieval.enabled and is inactive."},
            {"component": "citation_repair", "reason": "Disabled in the frozen B6 configuration; inactive in the reference."},
            {"component": "ontology-derived query enrichment", "reason": "The frozen 375 questions all contain more than two whitespace-delimited words, while this branch is guarded by a <=2-word condition; the path is inactive for the complete population and is excluded from primary contrasts."},
            {"component": "llm_query_expansion", "reason": "Disabled in the frozen B6 configuration; inactive in the reference."},
            {"component": "graph_selected_sources_card", "reason": "Although its setting is enabled in the frozen config, the reference uses core.generation.run_query_on_baseline and its _build_prompt_from_chunks path; that path bypasses the RAGService.ask post-generation selected-sources-card branch."},
            {"component": "Shannon estimator", "reason": "Retained as fixed diagnostic instrumentation for historical entropy compatibility; it is not a retrieval/context treatment arm and all its generation/log-probability events are preserved."},
        ],
    }
