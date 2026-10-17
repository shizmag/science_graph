from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from ablation_cuda.runtime.checkpoints import CheckpointStore, stable_input_hash
from ablation_cuda.runtime.matrix import ARM_DEFINITIONS, resolve_matrix
from ablation_cuda.runtime.progress import Progress
from ablation_cuda.verify import (
    EXPECTED_MODELS, FULL_MATRIX_ARM_IDS, _apply_overrides,
    _expected_arm_hash, _extract_full_archive, canonical_json,
    sha256_bytes, sha256_file, verify_result_archive,
)


def _config():
    return {
        "rag_components": {
            "dense_search": True, "lexical_search": True,
            "reranker": True, "score_blending": True,
            "graph_expansion": True, "static_graph_context": True,
            "context_trimming": True, "deduplicate_parent_chunks": True,
            "graph_concept_retrieval": False, "graph_bridge_retrieval": False,
            "graph_neighbors_in_rrf": False, "intent_classifier": False,
            "shannon_estimator_enabled": True,
        },
        "graph_retrieval": {
            "enabled": True, "concept_retrieval_enabled": True,
            "bridge_retrieval_enabled": True, "subbranch_policy": "explicit",
            "intra_paper_candidate_augmentation_enabled": True,
            "selected_sources_card_enabled": True,
        },
    }


def _models():
    asset_manifest = json.loads((Path(__file__).resolve().parents[1] / "models/asset_manifest.json")
                                .read_text(encoding="utf-8"))
    return {role: {"model_id": model_id, "revision": revision, "verified": True,
                   "files": asset_manifest["model_assets"][role]["files"]}
            for role, (model_id, revision) in EXPECTED_MODELS.items()}


def _question_rows():
    rows = []
    for index in range(375):
        label = "answerable" if index < 300 else "designated_negative"
        rows.append({
            "query_id": f"q_{index:03d}",
            "question": f"Frozen biomedical query {index}: what is the measured outcome?",
            "gold_reference_answer": "A reference answer." if label == "answerable" else "",
            "answerability_label": label,
            "original_pmids": [str(100000 + index)] if label == "answerable" else [],
            "original_paper_ids": [f"paper_{index}"] if label == "answerable" else [],
            "source_metadata": {}, "source_pmid": str(100000 + index) if label == "answerable" else None,
            "synthetic_negative_category": None if label == "answerable" else "synthetic_unanswerable",
            "row_index": index + 1,
        })
    return rows


def test_full_matrix_has_unique_hashes_and_isolates_declared_paths():
    arms, configs = resolve_matrix(_config(), snapshot_sha256="a" * 64,
                                   dataset_sha256="b" * 64,
                                   model_manifest_sha256="c" * 64)
    assert len(arms) == len(ARM_DEFINITIONS) == 14
    assert len({item["arm_id"] for item in arms}) == 14
    assert len({item["effective_configuration_hash"] for item in arms}) == 14
    assert len(arms) * 375 == 5250
    assert arms[0]["reference_baseline"] == "B6"
    assert arms[0]["paper_baseline_mapping"] == "B5"
    by_id = {item["arm_id"]: item for item in arms}
    baseline = configs["B6_full"]
    for arm_id, arm in by_id.items():
        definition = next(item for item in ARM_DEFINITIONS if item.arm_id == arm_id)
        assert configs[arm_id] == _apply_overrides(baseline, {"overrides": definition.overrides})
        record = dict(arm, effective_config=configs[arm_id], definition=definition.__dict__)
        assert arm["effective_configuration_hash"] == _expected_arm_hash(record)
    assert configs["B6_minus_concept_retrieval"]["graph_retrieval"]["bridge_retrieval_enabled"] is True
    assert configs["B6_minus_bridge_retrieval"]["graph_retrieval"]["concept_retrieval_enabled"] is True
    assert configs["B6_minus_intrapaper_graph_candidates"]["graph_retrieval"]["enabled"] is True
    assert configs["B6_minus_intrapaper_graph_candidates"]["graph_retrieval"]["intra_paper_candidate_augmentation_enabled"] is False
    assert "B6_minus_ontology_lookup" not in by_id


def test_author_verifier_identity_anchors_match_frozen_inputs():
    from ablation_cuda import verify

    repo_root = Path(__file__).resolve().parents[2]
    assert verify.CANONICAL_DATASET_SHA256 == sha256_file(
        repo_root / "ablation_portability/dataset/questions.jsonl")
    assert verify.CANONICAL_QUESTIONS_MANIFEST_SHA256 == sha256_file(
        repo_root / "ablation_portability/dataset/questions_manifest.json")
    assert verify.EXTERNAL_DB_SNAPSHOT_SHA256 == sha256_file(
        repo_root / "ablation_portability/snapshots/candidates/external_db/v1/database.db")
    assert verify.MODEL_ASSET_MANIFEST_SHA256 == sha256_file(
        repo_root / "ablation_cuda/models/asset_manifest.json")


def test_checkpoint_resume_hydrates_telemetry_and_rejects_stale_cache(tmp_path):
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir / "checkpoints")
    prompt = "serialized generation prompt"
    prompt_hash = sha256_bytes(prompt.encode())
    identity = "input-v1"
    token = {
        "token_id": 17, "token_text": "Yes", "token": "Yes", "logprob": -0.2,
        "nll_nats": 0.2, "nll_bits_per_token": 0.2885,
        "entropy_k20_bits": 1.0, "entropy_k50_bits": 1.1, "entropy_k100_bits": 1.2,
        "retained_mass_k20": 0.7, "retained_mass_k50": 0.8,
        "retained_mass_k100": 0.9, "top1_prob_full": 0.4,
        "logit_margin": 0.1, "top_logit_margins": [],
        "telemetry_device": "cuda:0", "stop_state": "length",
    }
    result = {
        "arm": "B6_full", "query_id": "q_000", "status": "SUCCESS",
        "input_identity": identity, "prompt": prompt, "prompt_sha256": prompt_hash,
        "stable_result_id": stable_input_hash({"input_identity": identity, "prompt_sha256": prompt_hash}),
        "raw_generated_response": "answer", "generation_events": [{
            "purpose": "final_answer_generation", "event": {"prompt": prompt},
            "token_telemetry": [token],
        }], "token_telemetry": [token], "parser_status": "UNKNOWN",
        "result_errors": [], "stage_status": {name: "complete" for name in (
            "preprocessing", "lexical_retrieval", "dense_retrieval", "cross_encoder",
            "graph_retrieval", "graph_expansion", "context_construction", "generation",
            "answerability_parser")},
        "retrieval_candidates_and_ranking_trace": {"trace_counters": {"final_evidence_chunk_count": 1}},
    }
    path = run_dir / "baselines/B6_full/results/q_000.json"
    store.put("B6_full", "q_000", identity, path, result)
    restored = CheckpointStore(run_dir / "checkpoints")
    reused = restored.get("B6_full", "q_000", identity)
    assert reused is not None
    assert reused["generation_events"][0]["token_telemetry"] == [token]
    assert reused["telemetry_sidecar"]["units"] == {
        "entropy": "bits", "log_probability": "nats", "probability": "fraction",
    }
    from ablation_cuda.runtime.full_run import _run_arm_checkpoint_valid
    assert _run_arm_checkpoint_valid(reused, "B6_full", identity) is True
    assert restored.get("B6_full", "q_000", "stale-input") is None
    sidecar = path.with_suffix(".telemetry.jsonl.gz")
    sidecar.write_bytes(b"corrupt")
    assert restored.get("B6_full", "q_000", identity) is None


def test_stage_cache_uses_full_dependency_identity(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints")
    first = {"snapshot": "a" * 64, "query_sha256": "b" * 64, "model": "e5@rev1"}
    store.cache_put("dense_candidates", first, [{"chunk_id": "c1", "score": 0.4}])
    assert store.cache_get("dense_candidates", first) == [{"chunk_id": "c1", "score": 0.4}]
    for mutation in ({"snapshot": "c" * 64}, {"query_sha256": "d" * 64}, {"model": "e5@rev2"}):
        stale = {**first, **mutation}
        assert store.cache_get("dense_candidates", stale) is None


def test_progress_uses_full_matrix_denominator_and_persists_interrupt_state(tmp_path):
    progress = Progress(tmp_path / "logs/events.jsonl", total_arms=14,
                        total_queries=375, total_operations=5250, run_id="cpu-fixture")
    progress.begin_operation(arm_id="B6_full", arm_index=1, query_index=1)
    progress.update("retrieval", details={"fixture": True})
    saved = json.loads((tmp_path / "logs/progress.json").read_text())
    assert saved["total_operations"] == 5250
    assert saved["total_arms"] == 14 and saved["queries_per_arm"] == 375
    assert saved["eta_seconds"] is None
    assert (tmp_path / "logs/events.jsonl").is_file()
    progress.close()


def test_blinded_judge_export_preserves_unknown_abstention_and_failed_states(tmp_path):
    from ablation_cuda.runtime.analysis import generate_judge_exports

    question = {"query_id": "q1", "question": "Is treatment effective?",
                "gold_reference_answer": "Yes", "answerability_label": "answerable",
                "original_pmids": ["123"]}
    arms = [{"arm_id": "B6_full", "effective_configuration_hash": "a" * 64},
            {"arm_id": "B6_abstain", "effective_configuration_hash": "b" * 64},
            {"arm_id": "B6_failed", "effective_configuration_hash": "c" * 64}]
    results = {
        ("B6_full", "q1"): {"status": "SUCCESS", "raw_generated_response": "Unclear.",
                              "parsed_answer": "Unclear.", "parser_status": "UNKNOWN", "prompt": "prompt",
                              "prompt_sha256": "d" * 64, "generation_events": [], "selected_evidence": []},
        ("B6_abstain", "q1"): {"status": "SUCCESS", "raw_generated_response": "Cannot be answered.",
                                 "parsed_answer": "", "parser_status": "UNANSWERABLE", "prompt": "prompt2",
                                 "prompt_sha256": "e" * 64, "generation_events": [], "selected_evidence": []},
        ("B6_failed", "q1"): {"status": "FAILED", "raw_generated_response": "", "parsed_answer": "",
                               "parser_status": None, "result_errors": ["synthetic failure"]},
    }
    generate_judge_exports(tmp_path, [question], arms, results,
                           dataset_sha256="f" * 64, model_manifest_sha256="1" * 64)
    rows = [json.loads(line) for line in (tmp_path / "judge/judge_inputs.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["parser_status"] == "UNKNOWN" and rows[0]["scoring_eligible"] is True
    assert rows[1]["abstention_decision"] is True
    assert rows[2]["operation_status"] == "FAILED" and rows[2]["scoring_eligible"] is False
    assert all("arm_id" not in row and "arm" not in row for row in rows)
    manifest = json.loads((tmp_path / "judge/judge_manifest.json").read_text())
    assert manifest["remote_requests_made"] is False and manifest["exported_records"] == 3


def test_full_run_gate_rejects_non_rtx5090_and_cuda_availability_only():
    from ablation_cuda.runtime.full_run import _require_real_rtx5090_cuda_probe

    with pytest.raises(RuntimeError, match="completed tensor operation"):
        _require_real_rtx5090_cuda_probe({"operation_device": "cpu", "device": "RTX 5090",
                                          "compute_capability": [12, 0], "result_sum": None})
    with pytest.raises(RuntimeError, match="requires an NVIDIA RTX 5090"):
        _require_real_rtx5090_cuda_probe({"operation_device": "cuda:0", "device": "RTX 4090",
                                          "compute_capability": [8, 9], "result_sum": 3.0})


def test_completed_resume_repackages_without_install_or_smoke(tmp_path, monkeypatch):
    from ablation_cuda.runtime import cli

    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "installed_models.json").write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "graphs/run_done_OCC-RAG-1.7B_CUDA_ABLATION"
    run_dir.mkdir(parents=True)
    manifest = {"run_id": "done", "status": "COMPLETE", "compatibility_identity": "compat",
                "matrix": {"planned_query_arm_operations": 5250,
                           "completed_query_arm_operations": 5250,
                           "failed_query_arm_operations": 0,
                           "missing_query_arm_operations": 0}}
    monkeypatch.setattr(cli, "MODEL_ROOT", model_root)
    monkeypatch.setattr(cli, "GRAPH_OUTPUT_ROOT", run_dir.parent)
    monkeypatch.setattr(cli, "validate_release", lambda: {"manifest": {"dataset": {}, "database": {}}})
    monkeypatch.setattr(cli, "_code_identity", lambda _release: "code")
    monkeypatch.setattr(cli, "_container_identity", lambda: {"image": "pinned"})
    monkeypatch.setattr(cli, "_experiment_identity", lambda *_args: "compat")
    monkeypatch.setattr(cli, "_find_compatible_run", lambda _identity: (run_dir, manifest))
    archive = tmp_path / "return/run_done.tar.gz"
    monkeypatch.setattr(cli, "pack_experiment", lambda _run, _out: (archive, "a" * 64))
    monkeypatch.setattr(cli, "_doctor", lambda: {"cuda_probe": {"device": "NVIDIA RTX 5090"}})
    monkeypatch.setattr(cli, "_install", lambda: pytest.fail("completed resume must not reinstall assets"))
    monkeypatch.setattr(cli, "run_smoke", lambda: pytest.fail("completed resume must not rerun smoke"))

    result = cli._run_full(resume_only=True, export_dir=archive.parent)
    assert result["already_complete"] is True
    assert result["status"] == "COMPLETE"
    assert result["archive_sha256"] == "a" * 64
    with pytest.raises(RuntimeError, match="Use --new-run"):
        cli._existing_run_action(new_run=False, requested_run_id=None, resume_only=False,
                                 device_report={"cuda_probe": {"device": "NVIDIA RTX 5090"}},
                                 export_dir=archive.parent)
    assert cli._existing_run_action(new_run=True, requested_run_id=None, resume_only=False,
                                    device_report={}, export_dir=archive.parent) is None


def test_failed_smoke_cannot_start_full_matrix():
    from ablation_cuda.runtime.full_run import run_full_matrix

    with pytest.raises(RuntimeError, match="mandatory real CUDA smoke"):
        run_full_matrix(smoke_result={"run_manifest": {"status": "FAILED"}},
                        smoke_archive=Path("missing-smoke.tar.gz"), smoke_archive_sha256="0" * 64)


def test_trace_noop_does_not_pass_component_intervention():
    from ablation_cuda.verify import _intervention_trace_state

    baseline = {"retrieval_candidates_and_ranking_trace": {
        "graph_retrieval": {"branch_execution": {
            "concept_branch_enabled": True, "concept_lookup_executed": True}},
        "trace_counters": {"candidate_selection_input_count": 2},
    }}
    no_op_off = {"retrieval_candidates_and_ranking_trace": {
        "graph_retrieval": {"branch_execution": {
            "concept_branch_enabled": True, "concept_lookup_executed": True}},
    }}
    real_off = {"retrieval_candidates_and_ranking_trace": {
        "graph_retrieval": {"branch_execution": {
            "concept_branch_enabled": False, "concept_lookup_executed": False}},
    }}
    assert _intervention_trace_state("B6_minus_concept_retrieval", baseline, off_arm=False)
    assert not _intervention_trace_state("B6_minus_concept_retrieval", no_op_off, off_arm=True)
    assert _intervention_trace_state("B6_minus_concept_retrieval", real_off, off_arm=True)


def _write_partial_run(root: Path) -> Path:
    run_id = "20261017T120000000000Z"
    run_dir = root / f"run_{run_id}_OCC-RAG-1.7B_CUDA_ABLATION"
    run_dir.mkdir(parents=True)
    questions = _question_rows()
    question_bytes = b"".join(canonical_json(row) + b"\n" for row in questions)
    question_path = run_dir / "questions.jsonl"
    question_path.write_bytes(question_bytes)
    questions_manifest = {"schema_version": "1.0", "row_count": 375,
                          "answerable_count": 300, "designated_negative_count": 75,
                          "unique_query_id_count": 375,
                          "dataset_sha256": sha256_bytes(question_bytes)}
    manifest_path = run_dir / "questions_manifest.json"
    manifest_path.write_text(json.dumps(questions_manifest, sort_keys=True, indent=2) + "\n")
    model_assets = _models()
    model_hash = sha256_bytes(canonical_json({"models": model_assets}))
    snapshot_sha = "d" * 64
    arms, configs = resolve_matrix(_config(), snapshot_sha256=snapshot_sha,
                                   dataset_sha256=sha256_bytes(question_bytes),
                                   model_manifest_sha256=model_hash)
    definitions = {item.arm_id: item.__dict__ for item in ARM_DEFINITIONS}
    for arm in arms:
        arm["effective_config"] = configs[arm["arm_id"]]
        arm["definition"] = definitions[arm["arm_id"]]
        arm["completion"] = {"planned": 375,
                              "complete": 1 if arm["arm_id"] == "B6_full" else 0,
                              "failed": 0,
                              "missing": 374 if arm["arm_id"] == "B6_full" else 375,
                              "status": "PARTIAL"}
        arm["intervention_integrity"] = {"status": "REFERENCE" if arm["arm_id"] == "B6_full" else "INCOMPLETE"}
    arm_manifest = {"schema_version": "1.0", "matrix_version": "full-b6-matrix-v1",
                    "paper_baseline": "B5", "internal_baseline": "B6",
                    "query_count": 375, "arm_count": 14,
                    "expected_query_arm_operations": 5250,
                    "dataset_sha256": sha256_bytes(question_bytes),
                    "external_db_snapshot_sha256": snapshot_sha,
                    "model_manifest_sha256": model_hash,
                    "arms": arms, "excluded_or_inactive_components": []}
    (run_dir / "arms_manifest.json").write_text(json.dumps(arm_manifest, sort_keys=True, indent=2) + "\n")
    database_identity = {"candidate": "external_db", "snapshot_id": "external_db-test",
                         "snapshot_sha256": snapshot_sha,
                         "logical_fingerprint": {"vector_count": 22068}}
    metadata = run_dir / "metadata"
    metadata.mkdir()
    (metadata / "database_identity.json").write_text(json.dumps(database_identity))
    (metadata / "token_telemetry_schema.json").write_text(json.dumps({
        "schema_version": "1.0", "format": "Apache Parquet 2.6",
        "units": {"entropy": "bits", "log_probability": "nats", "nll": "nats",
                  "probability": "fraction", "logit_margin": "raw logit units"},
        "columns": ["run_id", "arm_id", "query_id", "generation_purpose",
                    "token_index", "token_id", "token_text", "logprob", "nll_nats",
                    "nll_bits_per_token", "entropy_k20_bits", "entropy_k50_bits",
                    "entropy_k100_bits", "retained_mass_k20", "retained_mass_k50",
                    "retained_mass_k100", "top1_prob_full", "logit_margin",
                    "top_logit_margins_json", "telemetry_device", "stop_state"],
    }))
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow is needed for the local Parquet roundtrip fixture")
    empty_schema = pa.schema([("arm_id", pa.string()), ("query_id", pa.string()),
                              ("token_id", pa.int64())])
    pq.write_table(pa.Table.from_pylist([], schema=empty_schema), run_dir / "token_telemetry.parquet")
    for arm in arms:
        arm_dir = run_dir / "baselines" / arm["arm_id"]
        (arm_dir / "results").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([], schema=empty_schema), arm_dir / "token_telemetry.parquet")

    decoding = {"strategy": "greedy_argmax", "temperature": 0.0,
                "max_new_tokens": 768, "context_limit": 12000,
                "stop_token_ids": [151643, 151645, 151683],
                "tokenizer_revision": EXPECTED_MODELS["generator"][1]}
    image_identity = {"container_image_id": "sha256:image", "container_oci_digest": "sha256:oci",
                      "base_image": "python:3.12.13-slim-bookworm",
                      "base_image_manifest_digest": "sha256:base"}
    source_code_sha = "e" * 64
    telemetry_impl = "cuda-token-telemetry-v1"
    run_manifest = {
        "schema_version": "2.0", "experiment_type": "full_cuda_ablation",
        "status": "PARTIAL", "run_id": run_id,
        "internal_baseline": "B6", "paper_baseline": "B5",
        "input_package_manifest_sha256": "f" * 64,
        "judge": {"status": "PENDING_INDEPENDENT_AUTHOR_EVALUATION", "remote_requests_made": False},
        "smoke_gate": {"status": "PASSED", "run_id": "smoke"},
        "device_preflight": {"torch_cuda_operation": {
            "operation_device": "cuda:0", "device": "NVIDIA GeForce RTX 5090",
            "compute_capability": [12, 0], "result_sum": 3.0}},
        "dataset": {"row_count": 375, "answerable_count": 300,
                    "designated_negative_count": 75,
                    "sha256": sha256_bytes(question_bytes),
                    "questions_manifest_sha256": sha256_file(manifest_path),
                    "query_ids_sha256": sha256_bytes(canonical_json([row["query_id"] for row in questions]))},
        "database": {"candidate": "external_db", "snapshot_id": "external_db-test",
                     "snapshot_sha256": snapshot_sha, "logical_fingerprint": database_identity["logical_fingerprint"],
                     "identity_same_across_all_arms": True, "persisted_vectors": 22068,
                     "database_after_run_sha256": snapshot_sha,
                     "after_run_logical_fingerprint": database_identity["logical_fingerprint"]},
        "models": {"models": model_assets},
        "run_configuration_hashes": {"model_manifest_sha256": model_hash},
        "decoding": decoding, "actual_context_limit_tokens": 12000,
        "actual_output_token_limit": 768,
        "telemetry_implementation_version": telemetry_impl,
        "source_code_sha256": source_code_sha, "image_identity": image_identity,
        "matrix": {"version": "full-b6-matrix-v1", "arm_count": 14,
                   "query_count": 375, "planned_query_arm_operations": 5250,
                   "completed_query_arm_operations": 1,
                   "failed_query_arm_operations": 0,
                   "missing_query_arm_operations": 5249},
        "arms": [{"arm_id": arm["arm_id"], "status": "PARTIAL",
                  "completion": arm["completion"],
                  "intervention_integrity": arm["intervention_integrity"]} for arm in arms],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, sort_keys=True, indent=2) + "\n")

    result_base = run_dir / "baselines/B6_full/results/q_000.json"
    result_base.parent.mkdir(parents=True, exist_ok=True)
    query = questions[0]
    arm = arms[0]
    config_components = configs["B6_full"]["rag_components"]
    effective_components = {key: config_components.get(key, True) for key in config_components}
    effective_components.update({"hyde": False,
                                 "reranker": bool(config_components.get("reranker", True)),
                                 "graph_neighbors_in_rrf": bool(config_components.get("graph_neighbors_in_rrf", True)),
                                 "intent_classifier": bool(config_components.get("intent_classifier", False)),
                                 "shannon_estimator_enabled": bool(config_components.get("shannon_estimator_enabled", True)),
                                 "deduplicate_parent_chunks": bool(config_components.get("deduplicate_parent_chunks", True))})
    effective = {"config": configs["B6_full"], "baseline": "B6", "paper_baseline": "B5",
                 "baseline_components": effective_components, "intervention": None}
    base_input = sha256_bytes(canonical_json({
        "query_id": query["query_id"], "question_sha256": sha256_bytes(query["question"].encode()),
        "snapshot_sha256": snapshot_sha, "model_manifest_sha256": model_hash,
        "effective_config": effective, "decoding": decoding,
    }))
    input_identity = sha256_bytes(canonical_json({
        "input_id": base_input,
        "question_row_sha256": sha256_bytes(canonical_json(query)),
        "questions_manifest_sha256": sha256_file(manifest_path),
        "dataset_sha256": sha256_bytes(question_bytes),
        "code_sha256": source_code_sha, "image": image_identity,
        "telemetry_implementation": telemetry_impl,
    }))
    prompt = "<serialized prompt with retrieved evidence>"
    prompt_hash = sha256_bytes(prompt.encode())
    response = "The evidence is unclear."
    token = {"token_id": 17, "token_text": "unclear", "token": "unclear",
             "char_start": 0, "char_end": 7, "logprob": -0.4, "nll_nats": 0.4,
             "nll_bits_per_token": 0.576, "entropy_k20_bits": 1.1,
             "entropy_k50_bits": 1.2, "entropy_k100_bits": 1.3,
             "retained_mass_k20": 0.7, "retained_mass_k50": 0.8,
             "retained_mass_k100": 0.9, "top1_prob_full": 0.4,
             "logit_margin": 0.1, "top_logit_margins": [],
             "telemetry_device": "cuda:0", "stop_state": "length"}
    telemetry_path = result_base.with_suffix(".telemetry.jsonl.gz")
    telemetry_row = {"event_index": 0, "token_telemetry": [token]}
    telemetry_payload = (json.dumps(telemetry_row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    compressed = gzip.compress(telemetry_payload, mtime=0)
    telemetry_path.write_bytes(compressed)
    result = {
        "arm": "B6_full", "query_id": query["query_id"], "question": query["question"],
        "answerability_label": query["answerability_label"], "synthetic_negative": False,
        "source_pmids": query["original_pmids"], "status": "SUCCESS",
        "raw_generated_response": response, "parsed_answer": "UNKNOWN",
        "parser_status": "UNKNOWN", "input_prompt": "Question: query",
        "prompt": prompt, "prompt_sha256": prompt_hash,
        "stable_result_id": stable_input_hash({"input_identity": input_identity,
                                               "prompt_sha256": prompt_hash}),
        "input_identity": input_identity,
        "effective_configuration_hash": arm["effective_configuration_hash"],
        "effective_configuration": effective, "decoding": decoding,
        "model_manifest_sha256": model_hash,
        "snapshot_identity": {"snapshot_sha256": snapshot_sha},
        "selected_evidence": [], "retrieved_paper_ids": [],
        "retrieval_candidates_and_ranking_trace": {
            "trace_counters": {"lexical_seed_count": 1, "dense_seed_count": 1,
                               "final_evidence_chunk_count": 0},
            "current_trace": {"retrieval_paths": {"lexical_retrieval_enabled": True,
                                                    "dense_retrieval_enabled": True},
                              "seed_chunks_from_lexical_dense": {"lexical": ["l1"], "dense": ["d1"]}},
            "stage_cache_events": [],
        },
        "stage_status": {name: "complete" for name in (
            "preprocessing", "lexical_retrieval", "dense_retrieval", "cross_encoder",
            "graph_retrieval", "graph_expansion", "context_construction", "generation",
            "answerability_parser")},
        "generation_events": [{"purpose": "final_answer_generation",
                               "event": {"prompt": "Question: query", "formatted_prompt": prompt},
                               "token_telemetry": [token]}],
        "telemetry_sidecar": {"path": telemetry_path.relative_to(run_dir).as_posix(),
                              "sha256": sha256_file(telemetry_path),
                              "generation_event_count": 1, "token_count": 1,
                              "units": {"entropy": "bits", "log_probability": "nats", "probability": "fraction"}},
    }
    result_base.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")

    rows = []
    judge_rows = []
    mapping_rows = []
    for question in questions:
        for arm in arms:
            arm_id, query_id = arm["arm_id"], question["query_id"]
            success = arm_id == "B6_full" and query_id == "q_000"
            status = "SUCCESS" if success else "MISSING"
            raw = response if success else ""
            row = {"arm_id": arm_id, "query_id": query_id,
                   "canonical_result": f"baselines/{arm_id}/results/{query_id}.json",
                   "operation_status": status,
                   "response_sha256": sha256_bytes(raw.encode())}
            rows.append(row)
            response_id = "blind_" + sha256_bytes(f"{sha256_bytes(question_bytes)}:{query_id}:{arm_id}".encode())[:20]
            judge_rows.append({"blinded_response_id": response_id, "query_id": query_id,
                               "ground_truth_answerability_label": question["answerability_label"],
                               "raw_output": raw, "operation_status": status,
                               "response_sha256": sha256_bytes(raw.encode()),
                               "scoring_eligible": bool(success and raw.strip()),
                               "model_and_prompt_hashes": {"prompt_sha256": prompt_hash if success else None}})
            mapping_rows.append({"blinded_response_id": response_id, "query_id": query_id,
                                 "arm_id": arm_id,
                                 "effective_configuration_hash": arm["effective_configuration_hash"],
                                 "response_sha256": sha256_bytes(raw.encode())})
    (run_dir / "generation_results.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    judge_dir = run_dir / "judge"
    judge_dir.mkdir()
    (judge_dir / "judge_inputs.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in judge_rows))
    (judge_dir / "arm_mapping.json").write_text(json.dumps({"schema_version": "1.0", "records": mapping_rows}))
    (judge_dir / "judge_manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "judge_execution": "not_run", "remote_requests_made": False,
        "expected_records": 5250, "exported_records": 5250,
    }))
    (judge_dir / "judge_schema.json").write_text(json.dumps({"type": "object"}))
    (run_dir / "completion_report.md").write_text(
        "# Completion report\n\n- Status: **PARTIAL**\n- Planned query-arm operations: 5250\n")
    return run_dir


def test_partial_archive_roundtrip_imports_without_claiming_completion(tmp_path, monkeypatch):
    from ablation_cuda import verify

    from ablation_cuda.runtime.packaging import pack_experiment

    run_dir = _write_partial_run(tmp_path / "graphs")
    result_path = run_dir / "baselines/B6_full/results/q_000.json"
    result_record = json.loads(result_path.read_text())
    first_process = CheckpointStore(run_dir / "checkpoints")
    first_process.put("B6_full", "q_000", result_record["input_identity"], result_path, result_record)
    del first_process  # Simulated process interruption: all future access is through a fresh store.
    restarted = CheckpointStore(run_dir / "checkpoints")
    resumed = restarted.get("B6_full", "q_000", result_record["input_identity"])
    assert resumed is not None
    from ablation_cuda.runtime.full_run import _run_arm_checkpoint_valid
    assert _run_arm_checkpoint_valid(resumed, "B6_full", result_record["input_identity"])
    assert restarted.counters()["successful_operations"] == 1
    questions_manifest = json.loads((run_dir / "questions_manifest.json").read_text())
    dataset_sha = questions_manifest["dataset_sha256"]
    database_sha = json.loads((run_dir / "metadata/database_identity.json").read_text())["snapshot_sha256"]
    monkeypatch.setattr(verify, "CANONICAL_DATASET_SHA256", dataset_sha)
    monkeypatch.setattr(verify, "CANONICAL_QUESTIONS_MANIFEST_SHA256",
                        sha256_file(run_dir / "questions_manifest.json"))
    monkeypatch.setattr(verify, "EXTERNAL_DB_SNAPSHOT_SHA256", database_sha)
    archive, digest = pack_experiment(run_dir, tmp_path / "return")
    assert archive.name.endswith("_PARTIAL.tar.gz")
    assert Path(str(archive) + ".sha256").read_text().split()[0] == digest
    imported = tmp_path / "analysis/full_cuda_ablation/test-run"
    result = verify_result_archive(archive, destination=imported)
    assert result["verified"] is True
    assert result["status"] == "PARTIAL" and result["complete"] is False
    assert result["counts"] == {"successful": 1, "failed": 0, "missing": 5249}
    assert result["imported_to"] == str(imported.resolve())
    marker = json.loads((imported / "IMPORTED_ARCHIVE.json").read_text())
    assert marker["imported_status"] == "PARTIAL" and marker["complete"] is False
    damaged = tmp_path / "damaged.tar.gz"
    damaged.write_bytes(archive.read_bytes()[:-16] + b"corrupt-padding")
    Path(str(damaged) + ".sha256").write_text(Path(str(archive) + ".sha256").read_text())
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_result_archive(damaged, destination=tmp_path / "analysis/damaged")


def test_full_archive_rejects_path_traversal_and_corrupt_checksum(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"escape"
        import io
        info = tarfile.TarInfo("run_x/../../escape.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="Unsafe archive path"):
        _extract_full_archive(archive, tmp_path / "extract")


