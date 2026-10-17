from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ablation_cuda.identity import (
    arm_input_identity, score_blending_isolated, stable_result_identity,
)
from ablation_cuda.release import _asset_manifest, _redact_author_paths, _write_tar_gz
from ablation_cuda.runtime.assets import acquire_models, sha256_file
from ablation_cuda.runtime.smoke import (
    _preserve_stale_arm_checkpoint, _resumable_arm_checkpoint,
)
from ablation_cuda.telemetry_reference import reference_telemetry
from ablation_cuda.verify import verify_result_archive


REPO = Path(__file__).resolve().parents[2]


def test_runtime_lock_is_pinned_and_excludes_unrelated_application_stack():
    pyproject = (REPO / "ablation_cuda/runtime/pyproject.toml").read_text()
    lock = (REPO / "ablation_cuda/runtime/uv.lock").read_text()
    dockerfile = (REPO / "ablation_cuda/Dockerfile").read_text()
    assert '"torch==2.12.0"' in pyproject
    assert "https://download.pytorch.org/whl/cu130" in pyproject
    assert '"transformers==5.20.1"' in pyproject
    assert '"sentence-transformers==5.7.0"' in pyproject
    assert '"usearch==2.26.4"' in pyproject
    assert 'specifier = "==2.12.0"' in lock and 'version = "2.12.0+cu130"' in lock
    assert 'index = "https://download.pytorch.org/whl/cu130"' in lock
    for unused in ("mlx", "pymupdf", "marker-pdf", "pytesseract", "ocr",
                   "pandas", "matplotlib", "seaborn"):
        assert unused not in pyproject.lower()
    assert "@sha256:" in dockerfile and "--locked" in dockerfile
    assert "CREATE VIRTUAL TABLE fts_probe USING fts5" in dockerfile


def test_query_import_does_not_load_report_visualization_stack():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(REPO / "back/benchmarks/rag"), str(REPO / "back")))
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, core.generation; "
         "assert not {'pandas', 'matplotlib', 'seaborn'} & set(sys.modules)"],
        cwd=REPO, env=env, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_frozen_asset_manifest_matches_exact_checkpoint_revisions():
    import yaml

    lock = yaml.safe_load((REPO / "ablation_cuda/models/model_assets.lock.yaml").read_text())
    manifest = _asset_manifest(lock)
    phase1_lock = yaml.safe_load((REPO / "ablation_portability/models/model_assets.lock.yaml").read_text())
    assert manifest == _asset_manifest(phase1_lock)
    assert json.loads((REPO / "ablation_cuda/models/asset_manifest.json").read_text()) == manifest
    expected = {
        "generator": ("occ-ai/OCC-RAG-1.7B", "e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c"),
        "embedder": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"),
        "cross_encoder": ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825"),
    }
    for role, (model_id, revision) in expected.items():
        assert manifest["model_assets"][role]["model_id"] == model_id
        assert manifest["model_assets"][role]["revision"] == revision
        assert all(len(value) == 64 for value in manifest["model_assets"][role]["files"].values())
    wrong = dict(lock)
    wrong["model_assets"] = dict(lock["model_assets"])
    wrong["model_assets"]["generator"] = dict(lock["model_assets"]["generator"])
    wrong["model_assets"]["generator"]["revision"] = "mutable-main"
    with pytest.raises(ValueError, match="Model lock mismatch"):
        _asset_manifest(wrong)


def test_release_configuration_redacts_embedded_machine_paths():
    value = {
        "direct": "/private/var/folders/temporary/restore/database.db",
        "embedded": "archive=/Users/collaborator/Library/tmp/run/config.yaml",
        "ordinary": "B6 uses the frozen public model IDs",
    }
    sanitized = _redact_author_paths(value)
    assert sanitized["direct"] == "<author-local-path-redacted>"
    assert sanitized["embedded"] == "<author-local-path-redacted>"
    assert sanitized["ordinary"] == value["ordinary"]


def _fake_model_cache(root: Path, assets: dict) -> dict[str, Path]:
    paths = {}
    for role, item in assets.items():
        snapshot = root / f"models--{role}" / "snapshots" / item["revision"]
        snapshot.mkdir(parents=True)
        for name, digest in item["files"].items():
            payload = f"{role}:{name}".encode()
            actual = hashlib.sha256(payload).hexdigest()
            assert actual == digest
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        paths[role] = snapshot
    return paths


def test_model_download_cache_revision_and_asset_hash_validation(tmp_path, monkeypatch):
    import yaml

    model_ids = {
        "generator": ("occ-ai/OCC-RAG-1.7B", "e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c"),
        "embedder": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"),
        "cross_encoder": ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825"),
    }
    assets = {}
    for role, (model_id, revision) in model_ids.items():
        payload = f"{role}:model.safetensors".encode()
        assets[role] = {"model_id": model_id, "revision": revision,
                        "files": {"model.safetensors": hashlib.sha256(payload).hexdigest()}}
    release = tmp_path / "release"
    (release / "models").mkdir(parents=True)
    (release / "models/model_assets.lock.yaml").write_text(yaml.safe_dump({"model_assets": assets}))
    (release / "models/asset_manifest.json").write_text(json.dumps({
        "spacy_auxiliary_model": {"package": "en_core_web_md", "version": "3.8.0"}
    }))
    cache_root = tmp_path / "cache"
    cached = _fake_model_cache(cache_root, assets)
    spacy_module = tmp_path / "model_root/spacy_packages/en_core_web_md"
    spacy_module.mkdir(parents=True)
    (spacy_module / "__init__.py").write_text('__version__ = "3.8.0"\n')

    calls = []

    class FakeApi:
        def model_info(self, repo_id, revision):
            return SimpleNamespace(sha=revision)

    def fake_snapshot_download(repo_id, revision, cache_dir, local_files_only=False,
                               force_download=False):
        calls.append({"repo_id": repo_id, "revision": revision,
                      "local_files_only": local_files_only, "force_download": force_download})
        role = next(key for key, item in assets.items() if item["model_id"] == repo_id)
        return cached[role]

    fake_hub = SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)
    monkeypatch.syspath_prepend(str(tmp_path / "model_root/spacy_packages"))
    result = acquire_models(release, tmp_path / "model_root")
    assert result["models"]["generator"]["verified"] is True
    assert len(calls) == 3 and all(call["local_files_only"] for call in calls)
    assert {call["revision"] for call in calls} == {item["revision"] for item in assets.values()}


def test_sqlite_snapshot_restore_and_usearch_rebuild_use_persisted_vectors(tmp_path):
    from ablation_portability.scripts.rebuild_vector_index import rebuild_index
    from ablation_portability.scripts.restore_db import restore_snapshot
    from usearch.index import Index

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    database = snapshot / "database.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
    vectors = [([1.0, 0.0, 0.0]), ([0.0, 1.0, 0.0]), ([0.0, 0.0, 1.0])]
    for index, vector in enumerate(vectors):
        import numpy as np
        connection.execute("INSERT INTO chunks VALUES (?, ?)", (f"chunk-{index}", np.asarray(vector, dtype='<f4').tobytes()))
    connection.commit()
    connection.close()
    target = tmp_path / "restored"
    restored = restore_snapshot(snapshot, target)
    report = rebuild_index(restored)
    index = Index(ndim=3, metric="cos")
    index.load(str(restored.with_suffix(".usearch")))
    assert report["rows"] == 3 and report["dimensions"] == 3
    assert report["recomputed_embeddings"] is False
    assert len(index) == 3
    assert index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), 1).keys[0] == int(hashlib.md5(b"chunk-0").hexdigest()[:15], 16)


def test_selected_snapshot_retrieval_fixtures_replay_against_database():
    from ablation_portability.storage import validate_retrieval_fixtures

    snapshot = REPO / "ablation_portability/snapshots/candidates/external_db/v1"
    manifest = json.loads((snapshot / "snapshot_manifest.json").read_text())
    result = validate_retrieval_fixtures(
        snapshot / "database.db", snapshot / "validation_fixtures.json",
        manifest["validation_fixtures_sha256"],
    )
    assert result["valid"] is True
    assert result["chunk_fixtures_replayed"] == 5
    assert result["fts_fixture_replayed"] is True
    assert result["graph_nodes_replayed"] == 3


def test_input_archive_preserves_host_launcher_executable_mode(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    launcher = bundle / "ablation.sh"
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n")
    launcher.chmod(0o755)
    ordinary = bundle / "README.md"
    ordinary.write_text("instructions\n")
    archive = tmp_path / "bundle.tar.gz"
    _write_tar_gz(bundle, archive)
    with tarfile.open(archive, "r:gz") as tar:
        assert tar.getmember("ablation.sh").mode == 0o755
        assert tar.getmember("README.md").mode == 0o644


def test_linux_and_windows_bootstrap_scripts_gate_gpu_passthrough_and_share_commands():
    linux = (REPO / "ablation_cuda/ablation.sh").read_text()
    windows = (REPO / "ablation_cuda/ablation.ps1").read_text()
    for command in ("doctor", "install", "smoke", "status", "pack", "verify"):
        assert command in linux
        assert command in windows
    assert "--gpus" in linux and "all" in linux
    assert "--gpus" in windows and "all" in windows
    assert "nvidia-ctk" in linux and "NVIDIA Container Toolkit" in linux
    assert "wsl.exe" in windows and "--OSType" not in windows and "linux/amd64" in windows
    assert "Linux-container mode" in windows
    assert "--platform linux/amd64" in linux
    assert "type=bind,source=$VERIFY_ARCHIVE,target=/verify/archive.tar.gz,readonly" in linux
    assert "container verify --archive /verify/archive.tar.gz" in linux
    assert "RuntimeCommand 'verify'" in windows and "/verify/archive.tar.gz" in windows
    assert "models" in linux and "database" in linux and "logs" in linux and "outputs" in linux


def test_cuda_embedding_device_selection_fails_closed(monkeypatch):
    import torch
    from src.vector_search import EmbeddingEngine

    monkeypatch.setenv("ABLA_CUDA_REQUIRE_DEVICE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    engine = EmbeddingEngine("intfloat/multilingual-e5-base")
    with pytest.raises(RuntimeError, match="CUDA is required"):
        engine._get_device()


def test_cuda_logits_telemetry_matches_stable_reference_on_cpu():
    import torch
    from src.llm_engine.cuda_torch_impl import compact_logits_telemetry

    logits = torch.tensor([3.0, 1.25, -0.5, 0.0, 2.0, -2.0], dtype=torch.float64)
    token_id = 4

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return f"t{ids[0]}"

    actual = compact_logits_telemetry(logits, token_id, Tokenizer(), top_k_values=(2, 4, 6))
    expected = reference_telemetry(logits.tolist(), token_id, top_k_values=(2, 4, 6))
    for key in (
        "logprob", "nll_nats", "nll_bits_per_token", "logit_margin", "top1_prob_full",
        "entropy_k2_bits", "entropy_k4_bits", "entropy_k6_bits",
        "retained_mass_k2", "retained_mass_k4", "retained_mass_k6",
        "top1_prob_local_k2", "top1_prob_local_k4", "top1_prob_local_k6",
    ):
        assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-7)
    assert actual["telemetry_device"] == "cpu"
    assert actual["token_id"] == token_id


def test_mlx_reference_comparison_reports_prompt_tokens_and_aligned_metric_deltas(tmp_path):
    from ablation_cuda.runtime.smoke import _compare_mlx_reference

    reference = tmp_path / "mlx_reference"
    reference.mkdir()
    prompt = "<|im_start|>user\nquestion<|im_end|>"
    telemetry = [
        {"token_id": 4, "logprob": -0.1, "nll_nats": 0.1,
         "entropy_k20_bits": 0.2, "entropy_k50_bits": 0.3, "entropy_k100_bits": 0.4,
         "retained_mass_k20": 0.5, "retained_mass_k50": 0.6,
         "retained_mass_k100": 0.7, "top1_prob_full": 0.8, "logit_margin": 1.0},
        {"token_id": 5, "logprob": -0.2, "nll_nats": 0.2,
         "entropy_k20_bits": 0.25, "entropy_k50_bits": 0.35, "entropy_k100_bits": 0.45,
         "retained_mass_k20": 0.55, "retained_mass_k50": 0.65,
         "retained_mass_k100": 0.75, "top1_prob_full": 0.75, "logit_margin": 0.9},
    ]
    (reference / "prompt.txt").write_text(prompt)
    (reference / "generation.json").write_text(json.dumps({"raw_generated_answer": "ok"}))
    (reference / "telemetry.json").write_text(json.dumps({"tokens_info": telemetry}))
    arm = {"prompt": prompt, "raw_generated_response": "ok",
           "token_telemetry": [dict(row) for row in telemetry]}
    report = _compare_mlx_reference(reference, arm)
    assert report["prompt_byte_identical"] is True
    assert report["raw_response_identical"] is True
    assert report["token_id_sequence_identical"] is True
    assert report["shared_token_prefix_count"] == 2
    assert report["aligned_prefix_telemetry_absolute_differences"]["logprob"]["maximum_absolute_difference"] == 0


def test_generation_identity_and_score_blending_isolation():
    from ablation_cuda.identity import component_isolated

    question_hash = hashlib.sha256(b"question").hexdigest()
    common = {"rag_components": {"lexical_search": True, "score_blending": True}, "limit": 5}
    off = {"rag_components": {"lexical_search": True, "score_blending": False}, "limit": 5}
    full_arm = {"baseline": "B6", "config": common,
                "baseline_components": {"reranker": True, "score_blending": True}}
    off_arm = {"baseline": "B6", "config": off,
               "baseline_components": {"reranker": True, "score_blending": False}}
    assert score_blending_isolated(full_arm, off_arm)
    off_arm["config"]["limit"] = 4
    assert not score_blending_isolated(full_arm, off_arm)
    graph_full = {"baseline": "B6", "config": {"rag_components": {"graph_expansion": True}},
                  "baseline_components": {"graph_expansion": True, "score_blending": True}}
    graph_off = {"baseline": "B6", "config": {"rag_components": {"graph_expansion": False}},
                 "baseline_components": {"graph_expansion": False, "score_blending": True}}
    assert component_isolated(graph_full, graph_off, "graph_expansion")
    off_arm["config"]["limit"] = 5
    full_id = arm_input_identity(query_id="q", question_sha256=question_hash,
                                 snapshot_sha256="a" * 64, model_manifest_sha256="b" * 64,
                                 effective_config=full_arm, decoding={"temperature": 0.0})
    off_id = arm_input_identity(query_id="q", question_sha256=question_hash,
                                snapshot_sha256="a" * 64, model_manifest_sha256="b" * 64,
                                effective_config=off_arm, decoding={"temperature": 0.0})
    assert full_id != off_id
    assert stable_result_identity(full_id, "final prompt") != stable_result_identity(full_id, "changed prompt")


def test_completed_arm_checkpoint_resumes_only_for_matching_inputs_and_preserves_stale(tmp_path):
    input_identity = "a" * 64
    prompt = "<|im_start|>user\nFrozen prompt<|im_end|>"
    telemetry = [{"token_id": 17, "telemetry_device": "cuda:0"}]
    arm = {
        "arm": "B6", "status": "SUCCESS", "input_identity": input_identity,
        "stable_result_id": stable_result_identity(input_identity, prompt),
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "raw_generated_response": "Yes.", "parsed_answer": "Yes.",
        "parser_status": "ANSWERABLE", "result_errors": [],
        "selected_evidence": [{"chunk_id": "c1"}],
        "retrieval_candidates_and_ranking_trace": {"trace_counters": {"merged_candidates_before_reranking": 1}},
        "stage_status": {name: "complete" for name in (
            "preprocessing", "lexical_retrieval", "dense_retrieval", "cross_encoder",
            "graph_retrieval", "graph_expansion", "context_construction", "generation",
            "answerability_parser",
        )},
        "token_telemetry": telemetry,
        "generation_events": [{"token_telemetry": telemetry}],
    }
    assert _resumable_arm_checkpoint(arm, "B6", input_identity)
    assert not _resumable_arm_checkpoint(arm, "B6", "b" * 64)
    tampered = {**arm, "prompt": "different"}
    assert not _resumable_arm_checkpoint(tampered, "B6", input_identity)

    run_dir = tmp_path / "run"
    arm_dir = run_dir / "arms/B6"
    arm_dir.mkdir(parents=True)
    (run_dir / "arms/B6.json").write_text("partial")
    (arm_dir / "prompt.txt").write_text("partial prompt")
    _preserve_stale_arm_checkpoint(run_dir, "B6")
    preserved = list((run_dir / "superseded_checkpoints").rglob("*"))
    assert any(path.name == "B6.json" and path.read_text() == "partial" for path in preserved if path.is_file())
    assert any(path.name == "prompt.txt" and path.read_text() == "partial prompt" for path in preserved if path.is_file())


def _valid_result_payload(intervention_component: str = "score_blending") -> tuple[dict[str, bytes], dict]:
    from ablation_cuda.verify import canonical_json, sha256_bytes

    snapshot_sha = "a" * 64
    model_hash = "b" * 64
    model_items = {
        role: {"model_id": model_id, "revision": revision, "verified": True,
               "files": {"model.safetensors": hashlib.sha256(role.encode()).hexdigest()}}
        for role, (model_id, revision) in {
            "generator": ("occ-ai/OCC-RAG-1.7B", "e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c"),
            "embedder": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"),
            "cross_encoder": ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825"),
        }.items()
    }
    arms = []
    files: dict[str, bytes] = {}
    off_name = f"no_{intervention_component}"
    base_config = {"rag_components": {"lexical_search": True, "dense_search": True,
                                     "reranker": True, "score_blending": True,
                                     "graph_expansion": True}}
    for arm_name, enabled in (("B6", True), (off_name, False)):
        blending = intervention_component != "score_blending" or enabled
        graph_expansion = intervention_component != "graph_expansion" or enabled
        formula = "normalized_cross_encoder_rrf_blend" if blending else "raw_cross_encoder_score"
        prompt = f"prompt {arm_name}"
        config = json.loads(json.dumps(base_config))
        config["rag_components"]["score_blending"] = blending
        config["rag_components"]["graph_expansion"] = graph_expansion
        effective = {"baseline": "B6", "config": config,
                     "baseline_components": {"reranker": True, "score_blending": blending,
                                              "graph_expansion": graph_expansion},
                     "intervention": None if enabled else {"component": intervention_component, "enabled": False}}
        prompt_sha = sha256_bytes(prompt.encode())
        input_id = hashlib.sha256(arm_name.encode()).hexdigest()
        stable_id = hashlib.sha256(canonical_json({"input_identity": input_id, "prompt_sha256": prompt_sha})).hexdigest()
        telemetry = {"token_id": 1, "token_text": "ok", "logprob": -0.1, "nll_nats": 0.1,
                     "nll_bits_per_token": 0.144, "entropy_k20_bits": 0.1,
                     "entropy_k50_bits": 0.1, "entropy_k100_bits": 0.1,
                     "retained_mass_k20": 0.9, "retained_mass_k50": 0.95,
                     "retained_mass_k100": 0.99, "top1_prob_full": 0.8, "logit_margin": 2.0,
                     "top_logit_margins": {"top1_minus_top2": 2.0},
                     "telemetry_device": "cuda:0", "stop_state": "eos"}
        rank = {"ranking_formula": formula, "score_blending_enabled": blending,
                "score_blending_branch_executed": True,
                "candidate_ids_before_reranker": ["chunk-a", "chunk-b"],
                "rrf_scores_before_reranker": [0.02, 0.01],
                "cross_encoder_scores": [0.8, 0.7]}
        assembly = {"dynamic_graph_expansion_executed": graph_expansion,
                    "dynamic_graph_output_inserted": graph_expansion,
                    "prompt_template_branch": "ask_expander" if graph_expansion else "ask_no_expander"}
        trace = {"ranking": rank, "context_and_prompt_assembly": assembly,
                 "graph_expander": ({"hops": [{"rerank_time": 0.1}],
                                     "ingestion": {"rerank_time": 0.1}} if graph_expansion else None),
                 "trace_counters": {"lexical_seed_count": 1},
                 "context_capture": {"context_build": {"context_text_chars": 3},
                                     "context_trim": {"output_text_chars": 3}}}
        arm = {
            "arm": arm_name, "status": "SUCCESS", "query_id": "pubmedqa_21084567",
            "question": "question", "raw_generated_response": "raw", "parsed_answer": "answer",
            "parser_status": "ANSWERABLE", "prompt": prompt, "prompt_sha256": prompt_sha,
            "generation_event": {"formatted_prompt": prompt, "prompt": f"input {arm_name}"},
            "input_prompt": f"input {arm_name}",
            "generation_events": [{"purpose": "final_answer_generation",
                                   "event": {"formatted_prompt": prompt, "prompt": f"input {arm_name}"},
                                   "token_telemetry": [telemetry]}],
            "result_errors": [], "stage_status": {stage: "complete" for stage in (
                "preprocessing", "lexical_retrieval", "dense_retrieval", "cross_encoder",
                "graph_retrieval", "graph_expansion", "context_construction", "generation",
                "answerability_parser")},
            "retrieval_candidates_and_ranking_trace": trace,
            "token_telemetry": [telemetry],
            "device_diagnostics": {
                "cuda_inference_results": {"embedding_inference": True, "cross_encoder": True,
                                           "generation": True, "token_metrics": True},
                "e5_cuda_calls": [{"input_device_set": ["cuda:0"]}],
                "cross_encoder_cuda_calls": [
                    {"role": "final_candidate_reranking", "input_device_set": ["cuda:0"]},
                    *([{"role": "experimental_graph_expander", "input_device_set": ["cuda:0"]}]
                      if graph_expansion else []),
                ],
                "generator_model_device_set": ["cuda:0"],
                "cross_encoder_model_device_set": ["cuda:0"],
                "e5_query_embedding": {"model_device_set": ["cuda:0"]},
            },
            "snapshot_identity": {"snapshot_sha256": snapshot_sha},
            "model_manifest_sha256": model_hash, "decoding": {"temperature": 0.0},
            "effective_configuration": effective, "selected_evidence": [{"id": "evidence"}],
            "retrieved_paper_ids": ["pmid"], "input_identity": input_id,
            "stable_result_id": stable_id,
        }
        arms.append(arm)
        files[f"arms/{arm_name}.json"] = json.dumps(arm).encode()
    run = {
        "status": "SUCCESS", "query_id": "pubmedqa_21084567", "question": "question",
        "full_matrix_started": False, "judge": "not_run",
        "database": {"identity_same_across_arms": True, "candidate": "external_db",
                     "snapshot_sha256": snapshot_sha},
        "dataset": {"row_count": 375, "answerable_count": 300, "designated_negative_count": 75,
                    "sha256": "c" * 64},
        "models": {"models": model_items},
        "device_preflight": {"torch_cuda_operation": {"operation_device": "cuda:0",
                                                         "result_sum": 1.0,
                                                         "device": "NVIDIA GeForce RTX 5090",
                                                         "compute_capability": [12, 0]}},
        "arms": [{"arm": item["arm"], "status": item["status"]} for item in arms],
        "intervention": {"component": intervention_component, "trace_rule_passed": True,
                         "code_path_verified": True},
        "run_id": "test-run",
    }
    files["run_manifest.json"] = json.dumps(run).encode()
    inventory = [{"path": path, "size_bytes": len(content),
                  "sha256": sha256_bytes(content)} for path, content in sorted(files.items())]
    files["RESULTS_MANIFEST.json"] = json.dumps({
        "schema_version": "1.0", "package_type": "ablation_cuda_results",
        "run_manifest_sha256": sha256_bytes(files["run_manifest.json"]), "run_id": "test-run",
        "expected_files": inventory, "result_status": "SUCCESS",
    }).encode()
    return files, run


def _write_test_archive(path: Path, files: dict[str, bytes]):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, payload in sorted(files.items()):
                    import io
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
    Path(str(path) + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n")


def test_result_archive_verification_import_roundtrip_and_preview(tmp_path):
    files, _ = _valid_result_payload()
    archive = tmp_path / "results.tar.gz"
    _write_test_archive(archive, files)
    destination = tmp_path / "imported/run-1"
    result = verify_result_archive(archive, destination=destination)
    assert result["verified"] is True
    assert [arm["arm"] for arm in result["arms"]] == ["B6", "no_score_blending"]
    assert "## B6" in result["preview"] and "## no_score_blending" in result["preview"]
    assert (destination / "preview.md").is_file()
    repeated = verify_result_archive(archive, destination=destination)
    assert repeated["imported_to"] == str(destination.resolve())


def test_result_archive_verifies_trace_selected_graph_expansion_fallback(tmp_path):
    files, _ = _valid_result_payload("graph_expansion")
    archive = tmp_path / "graph-fallback.tar.gz"
    _write_test_archive(archive, files)
    result = verify_result_archive(archive, destination=tmp_path / "graph-fallback-import")
    assert [arm["arm"] for arm in result["arms"]] == ["B6", "no_graph_expansion"]


def test_verifier_requires_cuda_for_each_cross_encoder_scoring_role(tmp_path):
    files, _ = _valid_result_payload()
    arm = json.loads(files["arms/B6.json"])
    arm["device_diagnostics"]["cross_encoder_cuda_calls"] = [
        call for call in arm["device_diagnostics"]["cross_encoder_cuda_calls"]
        if call["role"] != "experimental_graph_expander"
    ]
    files["arms/B6.json"] = json.dumps(arm).encode()
    from ablation_cuda.verify import sha256_bytes
    manifest = json.loads(files["RESULTS_MANIFEST.json"])
    manifest["expected_files"] = [
        {"path": path, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for path, payload in sorted(files.items()) if path != "RESULTS_MANIFEST.json"
    ]
    files["RESULTS_MANIFEST.json"] = json.dumps(manifest).encode()
    archive = tmp_path / "missing-graph-reranker.tar.gz"
    _write_test_archive(archive, files)
    with pytest.raises(ValueError, match="graph-expander trace reports scoring"):
        verify_result_archive(archive, destination=tmp_path / "bad-graph-role-import")


def test_result_archive_rejects_corruption_and_missing_arm(tmp_path):
    files, _ = _valid_result_payload()
    archive = tmp_path / "results.tar.gz"
    _write_test_archive(archive, files)
    archive.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_result_archive(archive, destination=tmp_path / "bad-import")

    files, _ = _valid_result_payload()
    files.pop("arms/no_score_blending.json")
    incomplete = tmp_path / "incomplete.tar.gz"
    _write_test_archive(incomplete, files)
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_result_archive(incomplete, destination=tmp_path / "incomplete-import")
