from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ablation_portability.dataset.loader import FrozenDatasetError, load_frozen_dataset, select_frozen_query
from ablation_portability.scripts.freeze_dataset import extract_dataset
from ablation_portability.scripts.corpus_coverage import build_coverage_report, normalize_pmid
from ablation_portability.scripts.pack_assets import pack_candidate_set
from ablation_portability.scripts.record_baseline_decision import record_decision
from ablation_portability.scripts.rebuild_vector_index import rebuild_index
from ablation_portability.scripts.restore_db import restore_snapshot
from ablation_portability.scripts.snapshot_db import create_snapshot
from ablation_portability.scripts.validate_snapshot import validate_snapshot
from ablation_portability.scripts.verify_package import verify_package
from ablation_portability.storage import canonical_json_bytes, sha256_file
from ablation_portability.scripts.smoke_one_query import _collect_results, _snapshot_preflight


ROOT = Path(__file__).resolve().parents[2]


def _create_test_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE chunks(
            id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            text_content TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
        CREATE TABLE parent_chunks(
            id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            text_content TEXT NOT NULL
        );
        CREATE TABLE nodes(
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            properties TEXT
        );
        CREATE TABLE edges(
            source_id TEXT NOT NULL REFERENCES nodes(id),
            target_id TEXT NOT NULL REFERENCES nodes(id),
            type TEXT NOT NULL,
            properties TEXT,
            PRIMARY KEY(source_id, target_id, type)
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, text_content);
        """
    )
    conn.execute("INSERT INTO parent_chunks VALUES (?, ?, ?)", ("parent-1", "PMID:1", "parent text"))
    conn.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?)",
        [
            ("PMID:1", "Paper", '{"pmid":"1"}'),
            ("concept:1", "Concept", '{"name":"hypertension"}'),
        ],
    )
    conn.execute(
        "INSERT INTO edges VALUES (?, ?, ?, ?)",
        ("PMID:1", "concept:1", "MENTIONS_CONCEPT", "{}"),
    )
    for index, text in enumerate(
        [
            "Hypertension study measured clinical response.",
            "Cardiology cohort reported blood pressure outcomes.",
            "Clinical trial assessed treatment response markers.",
        ]
    ):
        vector = np.zeros(768, dtype="<f4")
        vector[index] = 1.0
        vector[3] = 0.25
        chunk_id = f"chunk-{index}"
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
            (chunk_id, "PMID:1", text, vector.tobytes()),
        )
        conn.execute("INSERT INTO chunks_fts VALUES (?, ?)", (chunk_id, text))
    conn.commit()
    conn.close()


def _write_dataset(root: Path) -> tuple[bytes, dict]:
    rows = [
        {
            "row_index": 1,
            "query_id": "pubmedqa_test_positive",
            "question": "What did the study measure?",
            "gold_reference_answer": "Clinical response.",
            "answerability_label": "answerable",
            "original_pmids": ["1"],
            "original_paper_ids": ["pubmed_1"],
            "synthetic_negative_category": None,
            "source_metadata": {"source_dataset": "test"},
        },
        {
            "row_index": 2,
            "query_id": "pubmedqa_test_negative",
            "question": "Does the missing intervention help?",
            "gold_reference_answer": "",
            "answerability_label": "designated_negative",
            "original_pmids": [],
            "original_paper_ids": [],
            "synthetic_negative_category": "unanswerable-synthetic-test",
            "source_metadata": {"source_dataset": "test"},
        },
    ]
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    manifest = {
        "schema_version": "1.0",
        "row_count": 2,
        "answerable_count": 1,
        "designated_negative_count": 1,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "questions.jsonl").write_bytes(raw)
    (root / "questions_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return raw, manifest


def test_canonical_json_and_frozen_loader_preserve_gold_and_negative_category(tmp_path: Path) -> None:
    first = canonical_json_bytes({"z": "é", "a": [1, True]})
    second = canonical_json_bytes({"a": [1, True], "z": "é"})
    assert first == second
    assert first == '{"a":[1,true],"z":"é"}'.encode("utf-8")

    dataset_dir = tmp_path / "dataset"
    raw, _ = _write_dataset(dataset_dir)
    rows = load_frozen_dataset(dataset_dir)
    assert rows[0]["gold_reference_answer"] == "Clinical response."
    assert rows[1]["synthetic_negative_category"] == "unanswerable-synthetic-test"
    assert hashlib.sha256(raw).hexdigest() == json.loads(
        (dataset_dir / "questions_manifest.json").read_text(encoding="utf-8")
    )["dataset_sha256"]

    (dataset_dir / "questions.jsonl").write_bytes(raw + b" ")
    with pytest.raises(FrozenDatasetError, match="SHA-256"):
        load_frozen_dataset(dataset_dir)


def test_authoritative_375_population_is_verified_when_source_is_restored() -> None:
    questions = ROOT / "ablation_portability/dataset/questions.jsonl"
    manifest_path = ROOT / "ablation_portability/dataset/questions_manifest.json"
    if not questions.is_file() or not manifest_path.is_file():
        pytest.skip("Canonical July retrieval population has not yet been restored")
    records = load_frozen_dataset(questions.parent)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(records) == manifest["row_count"] == 375
    assert len({row["query_id"] for row in records}) == 375
    assert sum(row["answerability_label"] == "answerable" for row in records) == 300
    assert sum(row["answerability_label"] == "designated_negative" for row in records) == 75
    assert "pubmedqa_19302863" in {row["query_id"] for row in records}


def test_canonical_extractor_is_byte_deterministic_when_source_is_restored(tmp_path: Path) -> None:
    historical = ROOT / "graphs/run_20260729_211324_OCC-RAG-1.7B/retrieved_contexts.yaml"
    golden = ROOT / "back/benchmarks/rag/dataset/pubmedqa/pubmedqa_golden.yaml"
    sample = ROOT / "back/benchmarks/rag/dataset/pubmedqa/sampled_300.json"
    negative_trace = ROOT / "revision_release/dataset_provenance/synthetic_negative_trace.csv"
    if not historical.is_file():
        pytest.skip("Canonical July retrieval population has not yet been restored")
    first_dir = tmp_path / "extract_a"
    second_dir = tmp_path / "extract_b"
    extract_dataset(historical, golden, sample, first_dir, negative_trace_path=negative_trace)
    extract_dataset(historical, golden, sample, second_dir, negative_trace_path=negative_trace)
    first = (first_dir / "questions.jsonl").read_bytes()
    second = (second_dir / "questions.jsonl").read_bytes()
    first_manifest = json.loads((first_dir / "questions_manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second_dir / "questions_manifest.json").read_text(encoding="utf-8"))
    assert first == second
    assert first_manifest["dataset_sha256"] == second_manifest["dataset_sha256"]
    assert (first_dir / "questions_manifest.json").read_bytes() == (second_dir / "questions_manifest.json").read_bytes()
    assert (first_dir / "questions_provenance.csv").read_bytes() == (second_dir / "questions_provenance.csv").read_bytes()
    assert (first_dir / "SHA256SUMS").read_bytes() == (second_dir / "SHA256SUMS").read_bytes()
    assert first_manifest["evidence_level"] == "EXACT_SOURCE_RECOVERED"
    assert first_manifest["duplicate_question_text_count"] == 0
    records = load_frozen_dataset(first_dir)
    assert len(records) == 375
    assert sum(row["answerability_label"] == "answerable" for row in records) == 300
    assert sum(row["answerability_label"] == "designated_negative" for row in records) == 75
    assert "pubmedqa_19302863" in {row["query_id"] for row in records}


def test_frozen_loader_rejects_missing_exact_question_text(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    raw, manifest = _write_dataset(dataset_dir)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    del rows[0]["question"]
    changed = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    manifest["dataset_sha256"] = hashlib.sha256(changed).hexdigest()
    (dataset_dir / "questions.jsonl").write_bytes(changed)
    (dataset_dir / "questions_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(FrozenDatasetError, match="Missing exact question text"):
        load_frozen_dataset(dataset_dir)


def test_frozen_query_selection_is_explicit_and_never_resamples(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir)
    records = load_frozen_dataset(dataset_dir)
    selected = select_frozen_query(records, "pubmedqa_test_positive")
    assert selected["row_index"] == 1
    assert selected["query_id"] == "pubmedqa_test_positive"
    with pytest.raises(FrozenDatasetError, match="absent from the frozen dataset"):
        select_frozen_query(records, "a-newly-sampled-query")


def test_local_model_asset_lock_is_parseable_and_pins_requested_models() -> None:
    import yaml

    lock_path = ROOT / "ablation_portability/models/model_assets.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assets = lock["model_assets"]
    assert assets["generator"]["model_id"] == "occ-ai/OCC-RAG-1.7B"
    assert assets["embedder"]["model_id"] == "intfloat/multilingual-e5-base"
    assert assets["cross_encoder"]["model_id"] == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert assets["embedder"]["preprocessing"]["documents"] == 'prepend "passage: " unless already present'
    assert assets["embedder"]["preprocessing"]["queries"] == 'prepend "query: " unless already present'


def test_generation_loop_is_reachable_from_run_benchmarking() -> None:
    import ast

    source = (ROOT / "back/benchmarks/rag/core/generation.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    functions = {
        node.name: node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    benchmarking_calls = {
        node.func.id for node in ast.walk(functions["run_benchmarking"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    checkpoint_loader_calls = {
        node.func.id for node in ast.walk(functions["_load_existing_checkpoint"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_query_on_baseline" in benchmarking_calls
    assert "run_query_on_baseline" not in checkpoint_loader_calls


def test_pmid_normalization_and_indexed_source_coverage(tmp_path: Path) -> None:
    database = tmp_path / "fixture.db"
    _create_test_database(database)
    assert normalize_pmid("pubmedqa_unans_17244100") == "17244100"
    assert normalize_pmid("pubmed_21084567_a_title") == "21084567"
    assert normalize_pmid("PMID:1") == "1"
    report = build_coverage_report(
        [
            {
                "query_id": "positive-1",
                "answerability_label": "answerable",
                "source_pmid": "1",
                "original_paper_ids": ["pubmed_1"],
            },
            {
                "query_id": "negative-2",
                "answerability_label": "designated_negative",
                "source_pmid": "2",
                "original_paper_ids": ["pubmed_2"],
            },
        ],
        database,
    )
    assert report["overall"] == {"query_count": 2, "source_present_count": 1, "source_absent_count": 1}
    assert report["per_query"][0]["source_pmid_present"] is True
    assert report["per_query"][1]["source_document_present"] is False


def test_retrieval_recall_uses_distinct_expected_document_denominator() -> None:
    from core.metrics import calculate_retrieval_recall

    assert calculate_retrieval_recall(["pubmed_1", "pubmed_2", "pubmed_2"], ["pubmed_2#chunk_0"]) == 0.5
    assert calculate_retrieval_recall(["pubmed_1"], []) == 0.0
    assert calculate_retrieval_recall([], []) == 1.0


def test_smoke_preflight_accepts_only_selected_snapshot_without_modifying_it(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    _create_test_database(source)
    snapshot_root = tmp_path / "snapshot_root"
    snapshot_dir = snapshot_root / "candidates" / "external_db" / "v1"
    create_snapshot(source, snapshot_dir, "external_db")
    record_decision(snapshot_root, "external_db", "New baseline for this smoke fixture.")
    before = sha256_file(snapshot_dir / "database.db")
    manifest, database_path, digest = _snapshot_preflight(snapshot_dir)
    assert manifest["baseline_selection"]["author_selection"] == "external_db"
    assert digest == before == sha256_file(database_path)


def test_snapshot_restore_preserves_vectors_fts_graph_and_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    _create_test_database(source)
    source_hash = sha256_file(source)
    snapshot_dir = tmp_path / "snapshot"
    manifest = create_snapshot(source, snapshot_dir, "fixture")
    assert manifest["source_identity"]["stable_during_backup"] is True
    assert sha256_file(source) == source_hash

    restored = []
    for name in ("restore_a", "restore_b"):
        target_dir = tmp_path / name
        database = restore_snapshot(snapshot_dir, target_dir)
        result = validate_snapshot(database)
        assert result["valid"]
        assert result["logical"]["chunk_count"] == 3
        assert result["logical"]["vector"]["dimensions"] == [768]
        assert result["logical"]["graph_node_count"] == 2
        assert result["logical"]["graph_edge_count"] == 1
        assert result["fts"]["matching_chunk_ids"]
        rebuilt = rebuild_index(database)
        assert rebuilt["recomputed_embeddings"] is False
        assert rebuilt["rows"] == 3
        restored.append(database)

    assert sha256_file(restored[0]) == sha256_file(restored[1])
    for database in restored:
        conn = sqlite3.connect(database)
        assert conn.execute("SELECT id FROM nodes ORDER BY id").fetchall() == [
            ("PMID:1",),
            ("concept:1",),
        ]
        assert conn.execute("SELECT source_id, target_id, type FROM edges").fetchall() == [
            ("PMID:1", "concept:1", "MENTIONS_CONCEPT")
        ]
        assert conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'hypertension'").fetchone()[0] == 1
        conn.close()


def test_explicit_database_path_wins_over_conflicting_home_default_and_hash_is_enforced(tmp_path: Path) -> None:
    home = tmp_path / "isolated-home"
    config_dir = home / ".config/pdf-graph-analyzer"
    config_dir.mkdir(parents=True)
    data_dir = home / ".local/share/pdf-graph-analyzer"
    data_dir.mkdir(parents=True)
    conflict = tmp_path / "machine-default.db"
    conflict.write_bytes(b"conflicting machine default")
    (config_dir / "config.yaml").write_text(
        f'db_path: "{conflict}"\narchive_dir: "{data_dir / "archive"}"\n',
        encoding="utf-8",
    )
    explicit = tmp_path / "frozen-snapshot.db"
    explicit.write_bytes(b"explicit immutable snapshot")
    digest = sha256_file(explicit)

    snippet = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3] + '/back')
sys.path.insert(0, sys.argv[3] + '/back/benchmarks/rag')
from core.config import resolve_project_db_path, database_file_sha256, database_path_cli_args
resolved = resolve_project_db_path(sys.argv[1], sys.argv[2])
print(json.dumps({'path': str(resolved), 'sha256': database_file_sha256(resolved), 'cli_args': database_path_cli_args(resolved, sys.argv[2])}))
"""
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("HF_TOKEN", None)
    env.pop("HUGGINGFACE_HUB_TOKEN", None)
    proc = subprocess.run(
        [sys.executable, "-c", snippet, str(explicit), digest, str(ROOT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(result["path"]) == explicit.resolve()
    assert result["path"] != str(conflict.resolve())
    assert result["sha256"] == digest
    assert result["cli_args"] == ["--db-path", str(explicit.resolve()), "--db-sha256", digest]

    mismatch = subprocess.run(
        [sys.executable, "-c", snippet, str(explicit), "0" * 64, str(ROOT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert mismatch.returncode != 0
    assert "SHA-256 does not match" in mismatch.stderr


def test_candidate_archive_verifies_third_restore_and_rejects_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    _create_test_database(source)
    snapshot_dir = tmp_path / "candidates" / "fixture_db" / "v1"
    create_snapshot(source, snapshot_dir, "fixture_db")
    comparison = tmp_path / "candidate_comparison.md"
    comparison.write_text("# Fixture candidate\n", encoding="utf-8")
    archive = tmp_path / "candidate_snapshot.tar.zst"

    pack_candidate_set([snapshot_dir], comparison, archive)
    second_archive = tmp_path / "candidate_snapshot_second.tar.zst"
    pack_candidate_set([snapshot_dir], comparison, second_archive)
    assert sha256_file(archive) == sha256_file(second_archive)
    result = verify_package(archive)
    assert result["valid"]
    assert result["snapshot_count"] == 1
    assert result["snapshots"][0]["third_clean_restore"]
    assert result["snapshots"][0]["rebuilt_vector_index"]["recomputed_embeddings"] is False

    corrupted = tmp_path / "candidate_snapshot_corrupt.tar.zst"
    payload = bytearray(archive.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    corrupted.write_bytes(payload)
    with pytest.raises(Exception):
        verify_package(corrupted)


def test_author_selected_candidate_decision_is_packaged_and_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    _create_test_database(source)
    snapshot_root = tmp_path / "snapshots"
    repo_snapshot = snapshot_root / "candidates" / "repo_db" / "v1"
    external_snapshot = snapshot_root / "candidates" / "external_db" / "v1"
    create_snapshot(source, repo_snapshot, "repo_db")
    create_snapshot(source, external_snapshot, "external_db")
    record = record_decision(
        snapshot_root,
        "external_db",
        "Author selected the graph-enabled candidate as a new baseline; July identity remains unproven.",
    )
    assert record["selected_candidate"] == "external_db"
    selected_manifest = json.loads((external_snapshot / "snapshot_manifest.json").read_text(encoding="utf-8"))
    rejected_manifest = json.loads((repo_snapshot / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert selected_manifest["baseline_selection"]["status"] == "author_selected"
    assert rejected_manifest["baseline_selection"]["status"] == "candidate_not_selected"
    assert sha256_file(snapshot_root / "baseline_decision.json") == selected_manifest["baseline_selection"]["decision_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        record_decision(snapshot_root, "repo_db", "Conflicting second choice")

    comparison = snapshot_root / "candidate_comparison.md"
    comparison.write_text("# Candidate comparison\n", encoding="utf-8")
    archive = tmp_path / "selected_candidate_set.tar.zst"
    pack_candidate_set([repo_snapshot, external_snapshot], comparison, archive)
    result = verify_package(archive)
    assert result["valid"]
    assert result["package_type"] == "candidate_snapshot_set"
    assert result["snapshot_count"] == 2


def test_smoke_package_requires_all_diagnostic_files(tmp_path: Path) -> None:
    from ablation_portability.scripts.pack_assets import pack_smoke_results

    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    (smoke_dir / "run_manifest.json").write_text(
        '{"status":"BLOCKED"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required diagnostic artifacts"):
        pack_smoke_results(smoke_dir, tmp_path / "smoke.tar.zst")


def test_one_query_artifact_accounting_marks_missing_inference_partial(tmp_path: Path) -> None:
    import yaml

    output = tmp_path / "smoke"
    pipeline_dir = output / "pipeline"
    pipeline_dir.mkdir(parents=True)
    clone_db = tmp_path / "restored.db"
    _create_test_database(clone_db)
    snapshot_hash = sha256_file(clone_db)
    retrieval = [
        {
            "id": "query-1",
            "baselines": {
                "B6": {
                    "retrieved_chunks": [{"id": "chunk-0", "paper_id": "pubmed_1"}],
                    "trace": {
                        "seed_chunks_from_lexical_dense": {
                            "lexical": ["chunk-0"], "dense": ["chunk-0"]
                        },
                        "graph_neighbor_paper_id_list": ["pubmed_2"],
                        "candidate_count_before_reranker": 4,
                        "candidate_count_after_reranker": 2,
                        "final_context_paper_id_list": ["pubmed_1"],
                    },
                }
            },
        }
    ]
    result = {
        "results": [
            {
                "id": "query-1",
                "baselines": {
                    "B6": {
                        "status": "partial",
                        "baseline_config": {"dense_search": True, "reranker": True},
                        "metrics": {},
                    }
                },
            }
        ]
    }
    (pipeline_dir / "retrieved_contexts.yaml").write_text(
        yaml.safe_dump(retrieval), encoding="utf-8"
    )
    (pipeline_dir / "evaluation_results.yaml").write_text(
        yaml.safe_dump(result), encoding="utf-8"
    )
    log_path = output / "execution.log"
    log_path.write_text("unit fixture: no inference was run\n", encoding="utf-8")
    query = {
        "query_id": "query-1",
        "question": "Test question?",
        "gold_reference_answer": "Reference answer.",
        "answerability_label": "answerable",
        "source_pmid": "1",
        "original_pmids": ["1"],
        "original_paper_ids": ["pubmed_1"],
    }
    snapshot_manifest = {"source_candidate": "fixture", "snapshot_id": "fixture-v1"}

    manifest = _collect_results(
        output,
        query,
        snapshot_manifest,
        snapshot_hash,
        {},
        "2026-10-16T00:00:00+00:00",
        0.01,
        0,
        log_path,
        clone_db,
    )

    assert manifest["status"] == "PARTIAL"
    assert manifest["validation_gates"]["database_hash_matches_execution_identity"] is False
    assert manifest["validation_gates"]["restored_logical_contents_match_selected_snapshot"] is False
    assert manifest["validation_gates"]["retrieval_candidates_present"] is True
    assert manifest["validation_gates"]["expected_source_document_retrieved"] is True
    assert manifest["validation_gates"]["lexical_candidates_present"] is True
    assert manifest["validation_gates"]["dense_candidates_present"] is True
    assert manifest["stages"]["candidate_counts"] == {
        "lexical_seed_chunks": 1,
        "dense_seed_chunks": 1,
        "graph_candidate_papers": 1,
        "candidates_before_reranker": 4,
        "candidates_after_reranker": 2,
        "retrieved_chunks": 1,
        "generator_visible_evidence_chunks": 0,
    }
    assert manifest["validation_gates"]["cross_encoder_scoring_executed"] is False
    assert manifest["validation_gates"]["final_prompt_captured"] is False
    assert manifest["validation_gates"]["llm_generation_persisted"] is False
    assert manifest["stages"]["cloud_judge"] is False
    for name in (
        "effective_config.json", "retrieval_trace.json", "prompt.txt",
        "generation.json", "telemetry.json", "run_manifest.json",
    ):
        assert (output / name).is_file()
    telemetry = json.loads((output / "telemetry.json").read_text(encoding="utf-8"))
    assert telemetry["tokens_info"] == []
    assert "no full-vocabulary logits fabricated" in telemetry["telemetry_kind"]


def test_database_identity_survives_rich_wrapped_startup_log(tmp_path: Path) -> None:
    import yaml

    output = tmp_path / "smoke"
    pipeline_dir = output / "pipeline"
    pipeline_dir.mkdir(parents=True)
    clone_db = tmp_path / "restored.db"
    _create_test_database(clone_db)
    database_hash = sha256_file(clone_db)
    resolved_path = str(clone_db.resolve())
    split = len(resolved_path) // 2
    log_path = output / "execution.log"
    log_path.write_text(
        f"Using database path: {resolved_path[:split]}\n"
        f"{resolved_path[split:]} (sha256={database_hash})\n",
        encoding="utf-8",
    )
    (pipeline_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump({"db_path": resolved_path}), encoding="utf-8"
    )

    manifest = _collect_results(
        output,
        {
            "query_id": "query-1",
            "question": "Test question?",
            "gold_reference_answer": "Reference answer.",
            "answerability_label": "answerable",
            "source_pmid": "1",
            "original_pmids": ["1"],
            "original_paper_ids": ["pubmed_1"],
        },
        {"source_candidate": "external_db", "snapshot_id": "external_db-test"},
        "selected-snapshot-hash",
        {},
        "2026-10-16T00:00:00+00:00",
        0.01,
        0,
        log_path,
        clone_db,
        execution_database_hash=database_hash,
        restoration_validation={"restored_database_sha256": database_hash},
    )

    assert manifest["validation_gates"]["database_hash_matches_execution_identity"] is True
    assert manifest["validation_gates"]["same_resolved_database_path_for_retrieval_and_generation"] is True
    identity = json.loads((output / "database_identity.json").read_text(encoding="utf-8"))
    assert identity["worker_startup_log_matches_database_identity"] is True


def test_smoke_result_archive_roundtrip_preserves_complete_diagnostics(tmp_path: Path) -> None:
    from ablation_portability.scripts.pack_assets import pack_smoke_results

    smoke = tmp_path / "smoke"
    pipeline = smoke / "pipeline"
    pipeline.mkdir(parents=True)
    files = {
        "run_manifest.json": '{"schema_version":"1.0","status":"PARTIAL","process_exit_code":2}\n',
        "effective_config.json": '{"baseline":"B6"}\n',
        "retrieval_trace.json": '{"candidate_counts":{"retrieved_chunks":1}}\n',
        "prompt.txt": "Question: fixture?\n",
        "generation.json": '{"raw_generated_answer":"answer"}\n',
        "telemetry.json": '{"tokens_info":[]}\n',
        "stage_timing_capture.json": '[{"stage":"embedding_model_initialization","elapsed_sec":1.0}]\n',
        "database_identity.json": '{"selected_snapshot_sha256":"abc"}\n',
        "execution.log": "captured process output\n",
    }
    for name, contents in files.items():
        (smoke / name).write_text(contents, encoding="utf-8")
    (pipeline / "retrieved_contexts.yaml").write_text("- id: fixture\n", encoding="utf-8")
    (pipeline / "evaluation_results.yaml").write_text("results: []\n", encoding="utf-8")
    archive = tmp_path / "smoke_results.tar.zst"
    pack_smoke_results(smoke, archive)
    extracted = tmp_path / "extracted"
    result = verify_package(archive, extracted)
    assert result["valid"] is True
    assert result["package_type"] == "smoke_results"
    assert result["smoke_completion_status"] == "PARTIAL"
    for source in smoke.rglob("*"):
        if source.is_file():
            assert (extracted / "smoke" / source.relative_to(smoke)).read_bytes() == source.read_bytes()


def test_recorded_real_b6_smoke_has_genuine_execution_evidence() -> None:
    smoke = ROOT / "ablation_unblock/smoke_run"
    if not smoke.is_dir():
        pytest.skip("The local real B6 smoke has not been run")
    manifest = json.loads((smoke / "run_manifest.json").read_text(encoding="utf-8"))
    trace = json.loads((smoke / "retrieval_trace.json").read_text(encoding="utf-8"))
    generation = json.loads((smoke / "generation.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "SUCCESS"
    assert manifest["process_exit_code"] == 0
    assert all(manifest["validation_gates"].values())
    assert manifest["query"]["query_id"] == "pubmedqa_21084567"
    assert trace["candidate_counts"]["lexical_seed_chunks"] > 0
    assert trace["candidate_counts"]["dense_seed_chunks"] > 0
    assert trace["candidate_counts"]["retrieved_chunks"] > 0
    assert generation["raw_generated_answer"]
    assert (smoke / "prompt.txt").read_text(encoding="utf-8").strip()
    assert manifest["stages"]["cross_encoder_initialization_events"]
    assert manifest["stages"]["embedding_model_initialization_events"]
