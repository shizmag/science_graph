import json
from pathlib import Path

import pytest

from ablation_cuda.recovery_analysis import (
    DEFAULT_ARCHIVE,
    DEFAULT_ARMS,
    analyze_archive,
    classify_failure,
    export_registry,
    main,
    mcnemar_exact,
    paired_bootstrap_ci,
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    arm_ids = ["reference", "treatment_a", "treatment_b", "treatment_c"]
    queries = ["q1", "q2", "q3", "q4"]
    _write_json(archive / "arms_manifest.json", {
        "arm_count": 4,
        "query_count": 4,
        "expected_query_arm_operations": 16,
        "arms": [{"arm_id": a, "completion": {"planned": 4}} for a in arm_ids],
    })
    for arm_i, arm in enumerate(arm_ids):
        for i, query_id in enumerate(queries):
            success = not (arm == "treatment_a" and query_id == "q4")
            row = {
                "query_id": query_id,
                "arm": arm,
                "answerability_label": "answerable",
                "status": "success" if success else "failed",
                "stage_status": {"generation": "success" if success else "failed"},
                "result_errors": [] if success else [{"stage": "generation", "type": "OOM", "message": "out of memory"}],
                "parsed_answer": f"answer-{arm}-{query_id}" if success else None,
                "timing": {"total_seconds": float(i + 1)},
                "total_pipeline_generated_tokens": i + 10,
            }
            _write_json(archive / "baselines" / arm / "results" / f"{query_id}.json", row)
    return archive


def test_default_archive_and_four_arm_selection_are_explicit():
    assert DEFAULT_ARCHIVE.as_posix().endswith(
        "graphs/run_20261020T065029857103Z_OCC-RAG-1.7B_CUDA_ABLATION"
    )
    assert len(DEFAULT_ARMS) == 4
    assert len(set(DEFAULT_ARMS)) == 4


def test_failure_classification_is_stable_and_specific():
    assert classify_failure({"status": "success", "result_errors": []}) == "success"
    assert classify_failure({"status": "failed", "result_errors": [{"stage": "generation", "type": "OOM"}]}) == "generation_oom"
    assert classify_failure({"status": "failed", "result_errors": [{"stage": "retrieval", "type": "TimeoutError"}]}) == "retrieval_failure"
    assert classify_failure(None) == "missing_result"


def test_mcnemar_exact_and_bootstrap_are_deterministic():
    result = mcnemar_exact([1, 1, 0, 0], [0, 1, 1, 0])
    assert result == {"both_success": 1, "both_failure": 1, "reference_only": 1,
                      "treatment_only": 1, "discordant": 2, "p_value": 1.0}
    ci1 = paired_bootstrap_ci([1.0, 2.0, 3.0], [0.0, 2.0, 2.0], seed=19, n_resamples=500)
    ci2 = paired_bootstrap_ci([1.0, 2.0, 3.0], [0.0, 2.0, 2.0], seed=19, n_resamples=500)
    assert ci1 == ci2
    assert ci1["estimate"] == pytest.approx(-2 / 3)
    assert ci1["n_pairs"] == 3


def test_archive_analysis_uses_answerable_population_and_does_not_zero_impute_quality(tmp_path):
    archive = _fixture(tmp_path)
    report = analyze_archive(archive, arms=("reference", "treatment_a", "treatment_b", "treatment_c"),
                             expected_answerable=4, bootstrap_resamples=200)
    assert report["population"]["answerable_ids"] == ["q1", "q2", "q3", "q4"]
    assert report["arms"]["treatment_a"]["availability"]["available"] == 3
    assert report["paired_comparisons"]["reference__vs__treatment_a"]["availability"]["reference_only"] == 1
    assert report["paired_comparisons"]["reference__vs__treatment_a"]["quality"]["n_pairs"] == 0
    assert report["judge_missingness"]["status"] == "not_run"
    assert report["operational_utility"]["treatment_a"]["quality_mean"] is None
    assert report["operational_utility"]["treatment_a"]["quality_observed_n"] == 0


def test_registry_is_sorted_deterministic_and_cli_is_read_only(tmp_path):
    archive = _fixture(tmp_path)
    out = tmp_path / "registry.jsonl"
    one = export_registry(archive, out, arms=("reference", "treatment_a", "treatment_b", "treatment_c"), expected_answerable=4)
    first = out.read_bytes()
    two = export_registry(archive, out, arms=("reference", "treatment_a", "treatment_b", "treatment_c"), expected_answerable=4)
    assert one == two
    assert first == out.read_bytes()
    assert out.stat().st_size > 0
    result = main(["--archive", str(archive), "--arms", "reference", "treatment_a", "treatment_b", "treatment_c",
                   "--expected-answerable", "4", "--bootstrap-resamples", "100"])
    assert result == 0


def test_archive_validation_rejects_wrong_population(tmp_path):
    archive = _fixture(tmp_path)
    with pytest.raises(ValueError, match="answerable"):
        analyze_archive(archive, arms=("reference", "treatment_a", "treatment_b", "treatment_c"),
                        expected_answerable=5)


def test_context_overflow_is_not_misclassified_as_oom():
    assert classify_failure({
        "status": "PARTIAL",
        "result_errors": ["RuntimeError: Serialized prompt has 51004 tokens; context limit leaves no room for generation (12000)"],
        "stage_status": {"generation": "error"},
    }) == "context_overflow"
