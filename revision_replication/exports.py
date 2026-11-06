"""Stable, provenance-preserving CSV exports for future result artifacts."""
from __future__ import annotations
import csv, json
from pathlib import Path

REQUIRED_RESULT_FIELDS = ("run_id", "query_id", "paper_baseline", "status", "text")


def export_operations_jsonl(input_path: Path, output_csv: Path, *, run_kind: str) -> int:
    rows = []
    for line in Path(input_path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            missing = [field for field in REQUIRED_RESULT_FIELDS if field not in row]
            if missing:
                raise ValueError(f"result row missing fields: {missing}")
            rows.append({
                "run_id": row["run_id"], "run_kind": run_kind,
                "query_id": row["query_id"], "paper_baseline": row["paper_baseline"],
                "status": row["status"], "text_present": bool(row["text"]),
                "token_count": row.get("token_count"), "mean_nll_bits_per_token": row.get("mean_nll_bits_per_token"),
                "error_type": row.get("error_type", ""), "input_sha256": row.get("input_sha256", ""),
            })
    rows.sort(key=lambda x: (x["run_id"], x["query_id"], x["paper_baseline"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["run_id", "run_kind", "query_id", "paper_baseline", "status", "text_present", "token_count", "mean_nll_bits_per_token", "error_type", "input_sha256"]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return len(rows)


def export_cuda_recovery_tables(analysis_json: Path, output_dir: Path) -> dict[str, int]:
    """Write separate arm and paired-availability CSVs from saved CUDA analysis.

    The export intentionally leaves quality metrics blank when historical judge
    scores are absent; failed generation and missing scores are never coerced to
    a zero-quality value.
    """
    analysis = json.loads(Path(analysis_json).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_rows = []
    for arm, data in sorted(analysis["arms"].items()):
        availability = data["availability"]
        quality = analysis.get("judge_missingness", {}).get("by_arm", {}).get(arm, {})
        arm_rows.append({
            "run_id": analysis.get("archive", ""),
            "run_kind": "historical_cuda_ablation",
            "arm": arm,
            "answerable_n": availability["answerable_n"],
            "generation_available_n": availability["available"],
            "generation_unavailable_n": availability["unavailable"],
            "availability_rate": availability["rate"],
            "failure_classifications_all_ids": json.dumps(data.get("failure_classifications_all_labeled_ids", {}), sort_keys=True),
            "judge_input_records_n": quality.get("input_records"),
            "judge_inputs_marked_eligible_n": quality.get("eligible_inputs"),
            "judge_scored_records_n": quality.get("scored_records", 0),
            "semantic_accuracy": "",  # absent scores are not zero
            "quality_status": quality.get("execution_status", "unknown"),
        })
    comparison_rows = []
    for contrast, data in sorted(analysis.get("paired_comparisons", {}).items()):
        avail = data["availability"]
        ci = data["availability_rate_difference_ci"]
        comparison_rows.append({
            "run_kind": "historical_cuda_ablation",
            "contrast": contrast,
            "paired_answerable_n": ci["n_pairs"],
            "reference_only_success_n": avail["reference_only"],
            "treatment_only_success_n": avail["treatment_only"],
            "both_success_n": avail["both_success"],
            "both_failure_n": avail["both_failure"],
            "availability_rate_difference": ci["estimate"],
            "availability_ci_low": ci["ci_low"],
            "availability_ci_high": ci["ci_high"],
            "mcnemar_exact_p": avail["p_value"],
            "joint_success_n": data["jointly_available_ids_n"],
            "quality_difference": "",
            "quality_status": data.get("quality", {}).get("status", "not_run"),
        })
    def write(filename: str, rows: list[dict]) -> None:
        path = output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as stream:
            fields = list(rows[0]) if rows else []
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write("cuda_arm_availability.csv", arm_rows)
    write("cuda_paired_availability.csv", comparison_rows)
    return {"arms": len(arm_rows), "comparisons": len(comparison_rows)}
