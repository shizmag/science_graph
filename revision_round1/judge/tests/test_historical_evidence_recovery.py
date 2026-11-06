from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from historical_evidence_recovery import (  # noqa: E402
    DEFAULT_RUN,
    EXPECTED_SOURCE_COMMIT,
    PAPER_TO_CODE,
    build_outputs,
    choose_stratified_cohort,
    evidence_for_baseline,
    load_historical_records,
    verify_source_contract,
)


def test_authoritative_july_run_and_paper_code_mapping():
    assert DEFAULT_RUN.name == "run_20260729_211324_OCC-RAG-1.7B"
    assert PAPER_TO_CODE == {"B3": "B4", "B5": "B6"}


def test_b4_preserves_post_trim_order_and_b6_uses_final_expansion_only():
    row = {"baselines": {
        "B4": {"status": "success", "trimmed_text": "TEXT ORDER", "trimmed_graph": "GRAPH ORDER",
               "enrichment_block": ""},
        "B6": {"status": "success", "trimmed_text": "DO NOT SUBSTITUTE", "trimmed_graph": "DO NOT APPEND",
               "enrichment_block": "EXPANSION FINAL"},
    }}
    b4, b4_components, b4_status = evidence_for_baseline(row, "B4")
    b6, b6_components, b6_status = evidence_for_baseline(row, "B6")
    assert b4 == "TEXT ORDER\n\n### KNOWLEDGE GRAPH CONNECTIONS:\nGRAPH ORDER"
    assert b4_components == ["trimmed_text", "trimmed_graph"]
    assert b6 == "EXPANSION FINAL"
    assert b6_components == ["enrichment_block"]
    assert b4_status == b6_status == "EXACT"
    assert "DO NOT SUBSTITUTE" not in b6 and "DO NOT APPEND" not in b6


def test_b6_empty_enrichment_uses_archived_fallback_and_missing_fields_stay_partial():
    fallback, components, status = evidence_for_baseline({"baselines": {"B6": {
        "status": "success", "trimmed_text": "T", "trimmed_graph": "G",
        "enrichment_block": "No essential knowledge graph enrichment found."}}}, "B6")
    assert fallback == "T\n\n### KNOWLEDGE GRAPH CONNECTIONS:\nG"
    assert components == ["trimmed_text", "trimmed_graph"]
    assert status == "EXACT"
    partial = evidence_for_baseline({"baselines": {"B4": {
        "status": "success", "retrieved_chunks": [{"text_content": "raw"}], "trimmed_text": "trimmed"}}}, "B4")
    assert partial == ("", [], "PARTIAL")
    assert evidence_for_baseline({"baselines": {"B6": {"status": "error"}}}, "B6")[2] == "UNAVAILABLE"


def test_source_commit_contract_is_verified_and_hashes_all_referenced_sources():
    contract = verify_source_contract(Path.cwd(), EXPECTED_SOURCE_COMMIT)
    assert contract["paper_to_code"] == PAPER_TO_CODE
    assert set(contract["source_sha256"]) >= {
        "back/benchmarks/rag/core/config.py",
        "back/benchmarks/rag/core/generation.py",
        "back/src/prompts/rag/ask_no_expander.txt",
        "back/src/prompts/rag/ask_expander.txt",
    }
    assert len(contract["source_sha256"]) == 8
    assert "exact tokenizer-rendered prompt reconstruction is not claimed" in " ".join(contract["limitations"])


def test_stratified_cohort_is_deterministic_and_paired():
    categories = ["answerable"] + [f"unanswerable-{name}" for name in
                                   ("missing-ablation", "missing-metric", "missing-relation", "missing-entity", "missing-context")]
    records = []
    for stratum in categories:
        for index in range(8):
            qid = f"{stratum}-{index}"
            for paper in ("B3", "B5"):
                records.append({"query_id": qid, "paper_baseline": paper,
                                "answerable": stratum == "answerable", "category": stratum,
                                "evidence_classification": "EXACT", "generated_answer_available": True,
                                "reference_answer_available": stratum == "answerable"})
    one = choose_stratified_cohort(records, seed=37, answerable_n=5, per_negative_category=2)
    assert one == choose_stratified_cohort(records, seed=37, answerable_n=5, per_negative_category=2)
    assert len(one) == 15
    assert len(set(one)) == len(one)
    assert sum(q.startswith("answerable-") for q in one) == 5
    for category in categories[1:]:
        assert sum(q.startswith(category + "-") for q in one) == 2


def test_authoritative_run_reconciles_metric_specific_eligibility_and_builds_packet(tmp_path):
    records, source_contract = load_historical_records(DEFAULT_RUN, Path.cwd())
    assert len(records) == 750
    assert source_contract["source_commit"] == EXPECTED_SOURCE_COMMIT
    report = build_outputs(DEFAULT_RUN, tmp_path, Path.cwd(), seed=20261106,
                           answerable_n=80, per_negative_category=4)
    counts = report["counts"]
    assert counts["query_pairs"] == 375
    assert counts["complete_evidence_pairs"] == 375
    assert counts["both_generated_answers"] == 323
    assert counts["semantic_accuracy_eligible_pairs"] == 252
    assert counts["faithfulness_eligible_pairs"] == 323
    assert counts["citation_fidelity_eligible_pairs"] == 323
    ids = json.loads((tmp_path / "paired_judge_query_ids.json").read_text())
    assert len(ids["query_ids"]) == 100
    inputs = [json.loads(line) for line in (tmp_path / "paired_judge_inputs.jsonl").read_text().splitlines()]
    assert len(inputs) == 100
    assert {item["query_id"] for item in inputs} == set(ids["query_ids"])
    assert all(set(item["judge_inputs"]) == {"B3", "B5"} for item in inputs)
    assert all(item["judge_inputs"]["B3"]["code_baseline"] == "B4" for item in inputs)
    assert all(item["judge_inputs"]["B5"]["code_baseline"] == "B6" for item in inputs)
