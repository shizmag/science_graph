from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import math
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ablation_portability.dataset.loader import load_frozen_dataset, select_frozen_query
from ablation_portability.scripts.rebuild_vector_index import rebuild_index
from ablation_portability.scripts.restore_db import restore_snapshot
from ablation_portability.scripts.validate_snapshot import validate_snapshot
from ablation_portability.storage import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERY_ID = "pubmedqa_21084567"
MODEL_LOCK = PROJECT_ROOT / "ablation_portability/models/model_assets.lock.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ablation_portability/smoke"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _redact(value: Any) -> Any:
    secret_keys = {"hf_token", "api_key", "token", "access_token", "secret", "password"}
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if str(key).lower() in secret_keys and item else _redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _model_assets(lock: dict) -> dict[str, dict]:
    resolved = {}
    local_paths = {
        "generator": lock["model_assets"]["generator"]["local_probe"]["path"],
        "embedder": lock["model_assets"]["embedder"]["local_probe"]["path"],
        "cross_encoder": lock["model_assets"]["cross_encoder"]["local_probe"]["path"],
    }
    for role, model in lock["model_assets"].items():
        location = Path(local_paths[role]).expanduser().resolve(strict=True)
        missing_or_mismatched = []
        for relpath, expected in model.get("files", {}).items():
            file_path = location / relpath
            if not file_path.is_file():
                missing_or_mismatched.append({"file": relpath, "reason": "missing"})
            elif sha256_file(file_path) != expected:
                missing_or_mismatched.append({"file": relpath, "reason": "sha256_mismatch"})
        if missing_or_mismatched:
            raise RuntimeError(f"Pinned {role} asset verification failed: {missing_or_mismatched}")
        resolved[role] = {
            "model_id": model["model_id"],
            "revision": model["revision"],
            "path": str(location),
            "files": model["files"],
            "verified": True,
        }
    return resolved


def _snapshot_preflight(snapshot_dir: Path) -> tuple[dict, Path, str]:
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    database_path = snapshot_dir / "database.db"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("snapshot_identity", {}).get("sha256")
    actual_hash = sha256_file(database_path)
    if expected_hash != actual_hash:
        raise RuntimeError(
            f"Snapshot database SHA-256 disagrees with its manifest: {expected_hash} != {actual_hash}"
        )
    selection = manifest.get("baseline_selection", {})
    if selection.get("status") != "author_selected":
        raise RuntimeError(
            "B6 smoke requires an author-selected database candidate; current state is "
            f"{selection.get('status', 'missing')}"
        )
    if selection.get("author_selection") != manifest.get("source_candidate"):
        raise RuntimeError("Author-selected database identity does not match the snapshot candidate")
    return manifest, database_path, actual_hash


def _validate_restored_snapshot(database_path: Path, snapshot_manifest: dict) -> dict:
    """Validate the restored database logically; SQLite backup need not preserve file bytes."""
    validation = validate_snapshot(database_path)
    if not validation.get("valid"):
        raise RuntimeError("Restored smoke database failed SQLite, FTS, graph, or vector validation")
    actual = validation["logical"]
    expected = snapshot_manifest.get("logical_corpus", {})
    actual_values = {
        "document_count": actual.get("document_count"),
        "chunk_count": actual.get("chunk_count"),
        "parent_count": actual.get("parent_count"),
        "vector_count": validation.get("vectors", {}).get("rows"),
        "vector_dimensions": validation.get("vectors", {}).get("dimensions"),
        "graph_node_count": actual.get("graph_node_count"),
        "graph_edge_count": actual.get("graph_edge_count"),
        "table_counts": actual.get("table_counts"),
        "node_labels": actual.get("node_labels"),
        "edge_types": actual.get("edge_types"),
        "fts_objects": actual.get("fts_objects"),
    }
    mismatches = {
        key: {"expected": expected.get(key), "actual": value}
        for key, value in actual_values.items()
        if expected.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Restored logical corpus differs from selected snapshot: {mismatches}")
    return {
        "valid": True,
        "method": "SQLite integrity, FTS5 fixture, vector validation, and frozen logical-summary comparison",
        "restored_database_sha256": validation["database_sha256"],
        "selected_snapshot_sha256": snapshot_manifest["snapshot_identity"]["sha256"],
        "physical_sha256_equal": (
            validation["database_sha256"] == snapshot_manifest["snapshot_identity"]["sha256"]
        ),
        "logical_summary": actual_values,
        "fts_fixture": validation["fts"],
        "vector_validation": validation["vectors"],
    }


def _one_query_yaml(row: dict) -> list[dict]:
    return [
        {
            "id": row["query_id"],
            "category": row.get("source_metadata", {}).get("historical_category") or "pubmedqa",
            "query": row["question"],
            "golden_answer": row["gold_reference_answer"],
            "is_answerable": row["answerability_label"] == "answerable",
            "expected_papers": row.get("original_paper_ids", []),
            "original_pmids": row.get("original_pmids", []),
        }
    ]


def _isolated_config(home: Path, conflicting_db: str, models: dict) -> Path:
    config_dir = home / ".config/pdf-graph-analyzer"
    data_dir = home / ".local/share/pdf-graph-analyzer"
    config_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "db_path": conflicting_db,
        "archive_dir": str(data_dir / "archive"),
        "hf_token": "",
        "llm": {
            "provider": "mlx",
            "local": {
                "model_path": models["generator"]["path"],
                "rag_model_path": models["generator"]["path"],
            },
            "cloud": {"api_key": ""},
        },
        "embedding": {"model_name": models["embedder"]["path"]},
        "reranker": {"model_name": models["cross_encoder"]["path"]},
    }
    path = config_dir / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8")
    return path


def _worker_main(args: argparse.Namespace) -> int:
    rag_root = PROJECT_ROOT / "back/benchmarks/rag"
    sys.path.insert(0, str(PROJECT_ROOT / "back"))
    sys.path.insert(0, str(rag_root))

    import runpy
    import core.generation as generation
    from src.services.rag_service import RAGService
    from src.vector_search import EmbeddingEngine
    from src.llm_engine.mlx_impl import MlxLLMEngine

    capture_path = Path(args.capture)
    stage_capture_path = Path(args.stage_capture)
    captured: list[dict] = []
    stage_events: list[dict] = []
    original = generation._generate_with_logits_safe
    original_get_reranker = RAGService._get_reranker
    original_ensure_embedding = EmbeddingEngine._ensure_model_loaded
    original_ensure_generator = MlxLLMEngine._ensure_model_loaded

    def capture_generation(engine, prompt):
        started = time.perf_counter()
        response, tokens = original(engine, prompt)
        captured.append(
            {
                "prompt": prompt,
                "response": response,
                "tokens_info": tokens,
                "elapsed_sec": time.perf_counter() - started,
            }
        )
        return response, tokens

    def capture_reranker_initialization(service):
        cold_start = service._reranker is None
        started = time.perf_counter()
        result = original_get_reranker(service)
        stage_events.append({
            "stage": "cross_encoder_initialization" if cold_start else "cross_encoder_reuse",
            "elapsed_sec": time.perf_counter() - started,
            "device": str(getattr(result, "device", "unknown")),
            "model_class": service._reranker.__class__.__name__,
        })
        return result

    def capture_embedding_initialization(engine):
        cold_start = engine.model is None
        started = time.perf_counter()
        result = original_ensure_embedding(engine)
        if cold_start:
            stage_events.append({
                "stage": "embedding_model_initialization",
                "elapsed_sec": time.perf_counter() - started,
                "device": engine.device,
                "model_name": engine.model_name,
            })
        return result

    def capture_generator_initialization(engine):
        cold_start = engine.model is None
        started = time.perf_counter()
        result = original_ensure_generator(engine)
        stage_events.append({
            "stage": "generator_initialization" if cold_start else "generator_reuse",
            "elapsed_sec": time.perf_counter() - started,
            "device": "mlx",
            "model_path": engine.model_path,
        })
        return result

    generation._generate_with_logits_safe = capture_generation
    RAGService._get_reranker = capture_reranker_initialization
    EmbeddingEngine._ensure_model_loaded = capture_embedding_initialization
    MlxLLMEngine._ensure_model_loaded = capture_generator_initialization
    run_pipeline = rag_root / "run_pipeline.py"
    sys.argv = [
        str(run_pipeline),
        "--dataset", args.dataset,
        "--baselines", "B6",
        "--skip-eval",
        "--in-process",
        "--logit-save",
        "--clear-checkpoint",
        "--output", args.pipeline_output,
        "--db-path", args.database,
        "--db-sha256", args.database_sha256,
        "--limit", "1",
    ]
    exit_code = 0
    try:
        runpy.run_path(str(run_pipeline), run_name="__main__")
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
        if exit_code:
            raise
    finally:
        generation._generate_with_logits_safe = original
        RAGService._get_reranker = original_get_reranker
        EmbeddingEngine._ensure_model_loaded = original_ensure_embedding
        MlxLLMEngine._ensure_model_loaded = original_ensure_generator
        try:
            from src.services.rag_service import parse_reasoning_response
            for event in captured:
                event["parser_status"], event["parsed_answer"] = parse_reasoning_response(
                    event.get("response", "")
                )
        except Exception as exc:
            for event in captured:
                event["parser_status"] = f"parser_exception:{type(exc).__name__}"
                event["parsed_answer"] = event.get("response")
        _write_json(capture_path, captured)
        _write_json(stage_capture_path, stage_events)
    return exit_code


def _make_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "AZURE_OPENAI_API_KEY",
    ):
        env.pop(key, None)
    env.update(
        {
            "HOME": str(home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    return env


def _parse_retrieval(retrieval_path: Path, query_id: str) -> tuple[dict, dict]:
    data = yaml.safe_load(retrieval_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Retrieval output must be a YAML list")
    case = next((row for row in data if str(row.get("id")) == query_id), None)
    if case is None:
        raise RuntimeError(f"Retrieval output did not persist query {query_id}")
    b6 = case.get("baselines", {}).get("B6")
    if not isinstance(b6, dict):
        raise RuntimeError("B6 retrieval result is missing from persisted retrieval output")
    return data, b6


def _collect_results(
    output: Path,
    query: dict,
    snapshot_manifest: dict,
    snapshot_hash: str,
    model_assets: dict,
    started_at: str,
    elapsed_sec: float,
    exit_code: int,
    log_path: Path,
    clone_db: Path,
    execution_database_hash: str | None = None,
    restoration_validation: dict | None = None,
    source_snapshot_unchanged: bool = False,
) -> dict:
    pipeline_dir = output / "pipeline"
    retrieval_path = pipeline_dir / "retrieved_contexts.yaml"
    results_path = pipeline_dir / "evaluation_results.yaml"
    capture_path = output / "_generation_capture.json"
    errors = []
    retrieval = {}
    b6_retrieval = {}
    if retrieval_path.is_file():
        try:
            retrieval, b6_retrieval = _parse_retrieval(retrieval_path, query["query_id"])
        except Exception as exc:
            errors.append(f"retrieval output parse failed: {type(exc).__name__}: {exc}")
    else:
        errors.append("retrieved_contexts.yaml was not persisted")

    result_data = {}
    result_case = {}
    b6_generation = {}
    if results_path.is_file():
        try:
            result_data = yaml.safe_load(results_path.read_text(encoding="utf-8")) or {}
            result_case = next(
                (row for row in result_data.get("results", []) if str(row.get("id")) == query["query_id"]),
                {},
            )
            b6_generation = result_case.get("baselines", {}).get("B6", {})
        except Exception as exc:
            errors.append(f"generation output parse failed: {type(exc).__name__}: {exc}")
    else:
        errors.append("evaluation_results.yaml was not persisted")

    events = json.loads(capture_path.read_text(encoding="utf-8")) if capture_path.is_file() else []
    stage_capture_path = output / "stage_timing_capture.json"
    stage_events = json.loads(stage_capture_path.read_text(encoding="utf-8")) if stage_capture_path.is_file() else []
    execution_database_hash = execution_database_hash or sha256_file(clone_db)
    post_run_database_hash = sha256_file(clone_db)
    try:
        post_run_validation = _validate_restored_snapshot(clone_db, snapshot_manifest)
    except Exception as exc:
        post_run_validation = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"Restored database logical validation failed after the worker: {post_run_validation['error']}")
    config_snapshot_path = pipeline_dir / "config_snapshot.yaml"
    config_snapshot = (
        yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8"))
        if config_snapshot_path.is_file() else {}
    ) or {}
    normalized_log = " ".join(log_path.read_text(encoding="utf-8", errors="replace").split()) if log_path.is_file() else ""
    # Rich wraps long paths at arbitrary columns, which can split a path in the
    # middle. Compare a whitespace-free form so formatting cannot invalidate an
    # otherwise exact path + pre-run database hash record.
    compact_log = "".join(normalized_log.split())
    compact_identity = "".join(
        f"Using database path: {clone_db.resolve()} (sha256={execution_database_hash})".split()
    )
    startup_database_identity_logged = compact_identity in compact_log
    configured_database_path = config_snapshot.get("db_path")
    config_database_path_matches = bool(
        configured_database_path
        and Path(configured_database_path).expanduser().resolve() == clone_db.resolve()
    )
    execution_database_identity_verified = bool(
        restoration_validation
        and restoration_validation.get("restored_database_sha256") == execution_database_hash
        and startup_database_identity_logged
        and config_database_path_matches
    )
    raw_answer = b6_generation.get("generated_answer")
    matching_events = [event for event in events if event.get("response") == raw_answer]
    final_event = matching_events[-1] if matching_events else (events[-1] if events else None)
    prompt = final_event.get("prompt", "") if final_event else ""
    metrics = b6_generation.get("metrics", {})
    component_metrics = metrics.get("components", {})
    effective_components = b6_generation.get("baseline_config", {})
    chunks = b6_generation.get("retrieved_chunks", [])
    graph_trace = b6_generation.get("trace") or b6_retrieval.get("trace") or {}
    seed_chunks = graph_trace.get("seed_chunks_from_lexical_dense", {}) if isinstance(graph_trace, dict) else {}
    lexical_seeds = seed_chunks.get("lexical", []) if isinstance(seed_chunks, dict) else []
    dense_seeds = seed_chunks.get("dense", []) if isinstance(seed_chunks, dict) else []
    retrieval_chunks = b6_retrieval.get("retrieved_chunks") or []
    source_ids = set(str(value) for value in query.get("original_paper_ids", []))
    source_pmids = {str(value) for value in query.get("original_pmids", [])}
    if query.get("source_pmid"):
        source_pmids.add(str(query["source_pmid"]))
    candidate_chunks = list(retrieval_chunks) + list(chunks)
    source_document_retrieved = any(
        str(item.get("paper_id") or item.get("id") or "").split("#", 1)[0] in source_ids
        or str(item.get("paper_id") or item.get("id") or "").split("#", 1)[0] in source_pmids
        for item in candidate_chunks if isinstance(item, dict)
    )
    parser_status = final_event.get("parser_status") if final_event else None
    parsed_answer = final_event.get("parsed_answer", raw_answer) if final_event else raw_answer
    cross_encoder_init = [
        event for event in stage_events if event.get("stage", "").startswith("cross_encoder_")
    ]
    embedding_init = [
        event for event in stage_events if event.get("stage") == "embedding_model_initialization"
    ]
    generator_init = [
        event for event in stage_events if event.get("stage") == "generator_initialization"
    ]
    generation_call_captured = bool(matching_events and final_event and prompt)

    gates = {
        "database_hash_matches_execution_identity": execution_database_identity_verified,
        "same_resolved_database_path_for_retrieval_and_generation": config_database_path_matches,
        "restored_logical_contents_match_selected_snapshot": bool(
            restoration_validation and restoration_validation.get("valid")
            and post_run_validation.get("valid")
        ),
        "selected_snapshot_file_unchanged": source_snapshot_unchanged,
        "retrieval_candidates_present": bool(chunks or b6_retrieval.get("retrieved_chunks")),
        "expected_source_document_retrieved": source_document_retrieved,
        "lexical_candidates_present": bool(lexical_seeds) if effective_components.get("lexical_search", True) else True,
        "dense_candidates_present": bool(dense_seeds) if effective_components.get("dense_search", True) else True,
        "embedding_model_execution_evidenced": (
            component_metrics.get("dense_retrieval", {}).get("calls", 0) > 0 and bool(embedding_init)
        ),
        "cross_encoder_scoring_executed": (
            component_metrics.get("reranking", {}).get("calls", 0) > 0 and bool(cross_encoder_init)
        ),
        "generator_model_execution_evidenced": (
            component_metrics.get("llm_generation", {}).get("calls", 0) > 0 and bool(generator_init)
        ),
        "final_prompt_captured": bool(prompt),
        "llm_generation_persisted": bool(raw_answer),
        "actual_generation_call_captured": generation_call_captured,
        "generation_status_success": b6_generation.get("status") == "success",
        "result_persisted": results_path.is_file(),
        "retrieval_output_persisted": retrieval_path.is_file(),
        "telemetry_preserved": isinstance(metrics, dict) and bool(metrics),
        "required_model_stages_succeeded": (
            (not effective_components.get("dense_search") or component_metrics.get("dense_retrieval", {}).get("calls", 0) > 0)
            and (not effective_components.get("reranker") or component_metrics.get("reranking", {}).get("calls", 0) > 0)
            and (not effective_components.get("shannon_estimator_enabled", True) or bool(generator_init))
            and component_metrics.get("llm_generation", {}).get("calls", 0) > 0
            and b6_generation.get("status") == "success"
        ),
        "external_evaluator_skipped": True,
        "process_exit_code_zero": exit_code == 0,
    }
    if not gates["retrieval_candidates_present"]:
        errors.append("B6 returned no persisted retrieval candidates")
    if not gates["expected_source_document_retrieved"]:
        errors.append("B6 did not retrieve the fixture's expected source document")
    if not gates["lexical_candidates_present"]:
        errors.append("B6 produced no lexical seed candidates although lexical search is enabled")
    if not gates["dense_candidates_present"]:
        errors.append("B6 produced no dense seed candidates although dense search is enabled")
    if not gates["cross_encoder_scoring_executed"]:
        errors.append("B6 Cross-Encoder initialization/scoring evidence is incomplete")
    if not gates["embedding_model_execution_evidenced"]:
        errors.append("B6 dense retrieval did not include captured embedding model initialization evidence")
    if not gates["final_prompt_captured"]:
        errors.append("The actual final prompt was not captured")
    if not gates["llm_generation_persisted"]:
        errors.append("No generated response was persisted")
    if not gates["actual_generation_call_captured"]:
        errors.append("The generation call could not be paired with its captured prompt and response")
    if not gates["database_hash_matches_execution_identity"]:
        errors.append("The worker did not log the restored database path and expected SHA-256")
    if not gates["same_resolved_database_path_for_retrieval_and_generation"]:
        errors.append("The pipeline config did not retain the restored database path")
    if not gates["generator_model_execution_evidenced"]:
        errors.append("B6 generation lacks captured local generator initialization/call evidence")
    if exit_code:
        errors.append(f"B6 worker exited with status {exit_code}; see execution.log for the traceback")
    status = "SUCCESS" if all(gates.values()) else "PARTIAL"
    failed_stage = None if status == "SUCCESS" else (
        "b6_pipeline_execution" if exit_code else
        "generation" if not (
            gates["llm_generation_persisted"]
            and gates["actual_generation_call_captured"]
            and gates["generator_model_execution_evidenced"]
            and gates["generation_status_success"]
        ) else "retrieval_or_database_validation"
    )

    retrieval_trace = {
        "query_id": query["query_id"],
        "question": query["question"],
        "database_sha256": execution_database_hash,
        "selected_snapshot_sha256": snapshot_hash,
        "persisted_retrieval_dataset": retrieval,
        "b6_retrieval": b6_retrieval,
        "final_evidence": chunks,
        "graph_path_counters": (
            b6_generation.get("trace")
            or b6_retrieval.get("trace")
            or b6_retrieval.get("diagnostics")
            or {}
        ),
    }
    generation = {
        "query_id": query["query_id"],
        "gold_reference_answer": query["gold_reference_answer"],
        "answerability_label": query["answerability_label"],
        "raw_generated_answer": raw_answer,
        "parsed_answer": parsed_answer,
        "parser_status": parser_status,
        "pipeline_status": b6_generation.get("status"),
        "generation_prompt_matches_captured_response": bool(matching_events),
        "retrieved_papers": b6_generation.get("retrieved_papers", []),
        "latency_sec": b6_generation.get("latency_sec"),
    }
    telemetry = {
        "telemetry_kind": "project compact per-token telemetry; no full-vocabulary logits fabricated",
        "metrics": metrics,
        "tokens_info": b6_generation.get("tokens_info", metrics.get("tokens_info", [])),
        "captured_generation_calls": [
            {
                "response": event.get("response"),
                "elapsed_sec": event.get("elapsed_sec"),
                "token_count": len(event.get("tokens_info", [])),
            }
            for event in events
        ],
    }
    component_execution = {
        name: {
            "enabled_by_b6_config": bool(enabled),
            "calls": component_metrics.get(name, {}).get("calls", 0),
            "time_sec": component_metrics.get(name, {}).get("time_sec", 0.0),
            "status": (
                "executed" if component_metrics.get(name, {}).get("calls", 0) > 0
                else "skipped_or_not_reported"
            ),
        }
        for name, enabled in effective_components.items()
    }
    graph_neighbors = graph_trace.get("graph_neighbor_paper_id_list", []) if isinstance(graph_trace, dict) else []
    graph_expansion_calls = component_metrics.get("graph_expansion", {}).get("calls", 0)
    lexical_seed_count = len(lexical_seeds)
    dense_seed_count = len(dense_seeds)
    final_context_paper_ids = graph_trace.get("final_context_paper_id_list", []) if isinstance(graph_trace, dict) else []
    final_graph_selected = sorted(set(graph_neighbors).intersection(final_context_paper_ids))
    graph_context = b6_retrieval.get("trimmed_graph") or b6_retrieval.get("context_graph") or ""
    graph_evidence = {
        "graph_candidates_produced_count": len(graph_neighbors),
        "graph_candidate_paper_ids": graph_neighbors,
        "graph_candidates_selected_in_final_context_count": len(final_graph_selected),
        "graph_candidates_selected_in_final_context": final_graph_selected,
        "graph_context_relationship_count": len(b6_retrieval.get("graph_relations") or []),
        "graph_context_characters": len(graph_context),
        "graph_information_inserted_in_generator_prompt": bool(graph_context.strip() and graph_context.strip() in prompt),
        "graph_expansion_component_calls": graph_expansion_calls,
        "graph_expansion_component_time_sec": component_metrics.get("graph_expansion", {}).get("time_sec", 0.0),
        "graph_context_source": "trimmed_graph" if b6_retrieval.get("trimmed_graph") else "context_graph",
    }
    candidate_counts = {
        "lexical_seed_chunks": lexical_seed_count,
        "dense_seed_chunks": dense_seed_count,
        "graph_candidate_papers": len(graph_neighbors),
        "candidates_before_reranker": graph_trace.get("candidate_count_before_reranker", 0) if isinstance(graph_trace, dict) else 0,
        "candidates_after_reranker": graph_trace.get("candidate_count_after_reranker", 0) if isinstance(graph_trace, dict) else 0,
        "retrieved_chunks": len(retrieval_chunks),
        "generator_visible_evidence_chunks": len(chunks),
    }
    retrieval_trace["candidate_counts"] = candidate_counts
    retrieval_trace["graph_evidence"] = graph_evidence
    _write_json(output / "retrieval_trace.json", retrieval_trace)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    _write_json(output / "generation.json", generation)
    _write_json(output / "telemetry.json", telemetry)
    _write_json(
        output / "database_identity.json",
        {
            "candidate": snapshot_manifest["source_candidate"],
            "snapshot_id": snapshot_manifest.get("snapshot_id"),
            "selected_snapshot_sha256": snapshot_hash,
            "restored_database_sha256_at_worker_start": execution_database_hash,
            "worker_startup_log_matches_database_identity": startup_database_identity_logged,
            "config_snapshot_database_path": configured_database_path,
            "config_snapshot_path_matches_restored_copy": config_database_path_matches,
            "restored_database_sha256_after_worker": post_run_database_hash,
            "restored_database_bytes_changed_during_run": post_run_database_hash != execution_database_hash,
            "logical_contents_match_selected_snapshot_after_worker": post_run_validation.get("valid", False),
            "selected_snapshot_file_unchanged": source_snapshot_unchanged,
            "same_resolved_path_for_retrieval_and_generation": config_database_path_matches,
        },
    )
    _write_json(
        output / "effective_config.json",
        {
            "resolved_database": {
                "path_during_run": str(clone_db.resolve()),
                "snapshot_id": snapshot_manifest.get("snapshot_id"),
                "sha256": execution_database_hash,
                "selected_snapshot_sha256": snapshot_hash,
                "post_run_sha256": post_run_database_hash,
                "post_run_bytes_changed": post_run_database_hash != execution_database_hash,
                "same_resolved_path_for_retrieval_and_generation": config_database_path_matches,
            },
            "baseline": "B6",
            "components": effective_components,
            "models": model_assets,
            "config_snapshot": _redact(
                yaml.safe_load((pipeline_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
            ) if (pipeline_dir / "config_snapshot.yaml").is_file() else None,
        },
    )
    run_manifest = {
        "schema_version": "1.0",
        "status": status,
        "started_at_utc": started_at,
        "elapsed_sec": elapsed_sec,
        "process_exit_code": exit_code,
        "failed_stage": failed_stage,
        "host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version},
        "query": {
            "query_id": query["query_id"],
            "question": query["question"],
            "gold_reference_answer": query["gold_reference_answer"],
            "answerability_label": query["answerability_label"],
        },
        "database": {
            "candidate": snapshot_manifest["source_candidate"],
            "snapshot_id": snapshot_manifest["snapshot_id"],
            "sha256": execution_database_hash,
            "selected_snapshot_sha256": snapshot_hash,
            "restored_database_sha256": execution_database_hash,
            "restoration_validation": restoration_validation,
            "post_run_database_sha256": post_run_database_hash,
            "post_run_logical_validation": post_run_validation,
            "post_run_bytes_changed": post_run_database_hash != execution_database_hash,
            "selected_snapshot_file_unchanged": source_snapshot_unchanged,
            "source_identity_historical_july_proven": False,
        },
        "models": model_assets,
        "stages": {
            "query_preprocessing": True,
            "lexical_retrieval": component_metrics.get("lexical_retrieval", {}).get("calls", 0) > 0,
            "e5_dense_retrieval": component_metrics.get("dense_retrieval", {}).get("calls", 0) > 0,
            "configured_graph_retrieval": {
                "configured": bool(effective_components.get("graph_expansion")),
                "executed": bool(graph_expansion_calls or graph_neighbors),
                "graph_neighbor_paper_count": len(graph_neighbors),
                "graph_candidates_selected_in_final_context_count": len(final_graph_selected),
                "graph_context_relationship_count": graph_evidence["graph_context_relationship_count"],
                "graph_information_inserted_in_generator_prompt": graph_evidence[
                    "graph_information_inserted_in_generator_prompt"
                ],
            },
            "candidate_fusion_and_ranking": bool(b6_retrieval),
            "cross_encoder": gates["cross_encoder_scoring_executed"],
            "cross_encoder_initialization_events": cross_encoder_init,
            "embedding_model_initialization_events": embedding_init,
            "generator_initialization_events": generator_init,
            "latency_breakdown": {
                "component_timings": component_metrics,
                "model_initialization_events": stage_events,
                "pipeline_process_wall_sec": elapsed_sec,
                "b6_reported_latency_sec": b6_generation.get("latency_sec"),
                "retrieval_reported_latency_sec": b6_retrieval.get("latency_sec"),
                "generation_call_elapsed_sec": final_event.get("elapsed_sec") if final_event else None,
            },
            "candidate_counts": candidate_counts,
            "graph_evidence": graph_evidence,
            "component_execution": component_execution,
            "prompt_assembly": gates["final_prompt_captured"],
            "local_generation": gates["llm_generation_persisted"],
            "generator_model_execution_evidenced": gates["generator_model_execution_evidenced"],
            "response_parsing": parser_status,
            "cloud_judge": False,
        },
        "validation_gates": gates,
        "errors": errors,
        "warning_log": str(log_path),
        "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
    }
    _write_json(output / "run_manifest.json", run_manifest)
    capture_path.unlink(missing_ok=True)
    return run_manifest


def run_smoke(snapshot_dir: Path, dataset_dir: Path, query_id: str, output: Path) -> dict:
    snapshot_dir = snapshot_dir.resolve(strict=True)
    dataset_dir = dataset_dir.resolve(strict=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Smoke output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_manifest, _, snapshot_hash = _snapshot_preflight(snapshot_dir)
        records = load_frozen_dataset(dataset_dir)
        if len(records) != 375:
            raise RuntimeError(f"Frozen dataset contains {len(records)} rows, expected 375")
        row = select_frozen_query(records, query_id)
        lock = yaml.safe_load(MODEL_LOCK.read_text(encoding="utf-8"))
        models = _model_assets(lock)
    except Exception as exc:
        started_at = datetime.now(timezone.utc).isoformat()
        blocker = {
            "schema_version": "1.0",
            "status": "BLOCKED",
            "started_at_utc": started_at,
            "process_exit_code": None,
            "query_id": query_id,
            "failed_stage": "preflight",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        log_path = output / "execution.log"
        log_path.write_text(f"Preflight blocked: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        _write_json(output / "preflight_blocker.json", blocker)
        blocker["log_sha256"] = sha256_file(log_path)
        _write_json(output / "run_manifest.json", blocker)
        return blocker

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    temp_context = tempfile.TemporaryDirectory(prefix="ablation-b6-smoke-")
    tmp = Path(temp_context.name)
    log_path = output / "execution.log"
    current_stage = "snapshot_restore"
    worker_started = False
    exit_code: int | None = None
    clone_db: Path | None = None
    clone_hash: str | None = None
    restoration_validation = None
    try:
        clone_dir = tmp / "restored_snapshot"
        clone_db = restore_snapshot(snapshot_dir, clone_dir)
        current_stage = "restored_snapshot_validation"
        restoration_validation = _validate_restored_snapshot(clone_db, snapshot_manifest)
        clone_hash = sha256_file(clone_db)
        current_stage = "usearch_index_rebuild"
        rebuild_index(clone_db)
        if sha256_file(clone_db) != clone_hash:
            raise RuntimeError("USEARCH rebuild changed the restored SQLite database bytes")

        current_stage = "worker_setup"
        dataset_file = tmp / "one_query.yaml"
        dataset_file.write_text(
            yaml.safe_dump(_one_query_yaml(row), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        home = tmp / "home"
        conflicting_default = str(tmp / "must_not_use_machine_default.db")
        _isolated_config(home, conflicting_default, models)
        pipeline_output = output / "pipeline"
        pipeline_output.mkdir(parents=True, exist_ok=True)
        worker_capture = tmp / "generation_capture.json"
        worker_stage_capture = tmp / "stage_capture.json"
        worker_argv = [
            sys.executable,
            "-m",
            "ablation_portability.scripts.smoke_one_query",
            "--worker",
            "--dataset", str(dataset_file),
            "--pipeline-output", str(pipeline_output),
            "--database", str(clone_db),
            "--database-sha256", clone_hash,
            "--capture", str(worker_capture),
            "--stage-capture", str(worker_stage_capture),
        ]
        current_stage = "b6_pipeline_execution"
        worker_started = True
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                worker_argv,
                cwd=PROJECT_ROOT,
                env=_make_env(home),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            exit_code = proc.returncode
        if worker_capture.is_file():
            shutil.copy2(worker_capture, output / "_generation_capture.json")
        else:
            (output / "_generation_capture.json").write_text("[]\n", encoding="utf-8")
        if worker_stage_capture.is_file():
            shutil.copy2(worker_stage_capture, output / "stage_timing_capture.json")
        else:
            (output / "stage_timing_capture.json").write_text("[]\n", encoding="utf-8")
        current_stage = "artifact_accounting"
        elapsed = time.perf_counter() - started
        manifest = _collect_results(
            output, row, snapshot_manifest, snapshot_hash, models,
            started_at, elapsed, exit_code or 0, log_path, clone_db,
            execution_database_hash=clone_hash,
            restoration_validation=restoration_validation,
            source_snapshot_unchanged=sha256_file(snapshot_dir / "database.db") == snapshot_hash,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        failure_line = f"Failure at {current_stage}: {type(exc).__name__}: {exc}\n"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(failure_line)
        manifest = {
            "schema_version": "1.0",
            "status": "PARTIAL" if worker_started else "BLOCKED",
            "started_at_utc": started_at,
            "elapsed_sec": elapsed,
            "process_exit_code": exit_code,
            "failed_stage": current_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "database": {
                "candidate": snapshot_manifest["source_candidate"],
                "selected_snapshot_sha256": snapshot_hash,
                "restored_database_sha256": clone_hash,
                "restoration_validation": restoration_validation,
                "source_identity_historical_july_proven": False,
            },
            "query": {
                "query_id": query_id,
                "question": row["question"],
                "gold_reference_answer": row["gold_reference_answer"],
                "answerability_label": row["answerability_label"],
            },
            "errors": [failure_line.strip()],
            "warning_log": str(log_path),
            "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
        }
        _write_json(output / "run_manifest.json", manifest)
    finally:
        temp_context.cleanup()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real B6 case from the frozen dataset and selected snapshot")
    parser.add_argument("snapshot", type=Path, nargs="?")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "ablation_portability/dataset")
    parser.add_argument("--query-id", default=DEFAULT_QUERY_ID)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", help=argparse.SUPPRESS)
    parser.add_argument("--pipeline-output", help=argparse.SUPPRESS)
    parser.add_argument("--database", help=argparse.SUPPRESS)
    parser.add_argument("--database-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--capture", help=argparse.SUPPRESS)
    parser.add_argument("--stage-capture", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(_worker_main(args))
    if args.snapshot is None:
        parser.error("snapshot directory is required")
    output = args.output or (DEFAULT_OUTPUT_ROOT / f"{args.query_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    manifest = run_smoke(args.snapshot, args.dataset_dir, args.query_id, output)
    print(json.dumps(
        {
            "status": manifest.get("status"),
            "query_id": manifest.get("query", {}).get("query_id") or manifest.get("query_id"),
            "database": manifest.get("database"),
            "process_exit_code": manifest.get("process_exit_code"),
            "failed_stage": manifest.get("failed_stage"),
            "validation_gates": manifest.get("validation_gates"),
            "candidate_counts": manifest.get("stages", {}).get("candidate_counts"),
            "errors": manifest.get("errors") or ([manifest.get("error")] if manifest.get("error") else []),
        },
        ensure_ascii=False,
        indent=2,
    ))
    if manifest.get("status") != "SUCCESS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
