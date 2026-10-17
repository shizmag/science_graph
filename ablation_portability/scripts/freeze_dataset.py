from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from ablation_portability.dataset.loader import FrozenDatasetError, load_frozen_dataset
from ablation_portability.storage import canonical_json_bytes, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_RUN_ID = "run_20260729_211324_OCC-RAG-1.7B"
DEFAULT_HISTORICAL = Path("graphs") / HISTORICAL_RUN_ID / "retrieved_contexts.yaml"
LEGACY_HISTORICAL = Path("back/benchmarks/rag/graphs") / HISTORICAL_RUN_ID / "retrieved_contexts.yaml"
DEFAULT_GOLDEN = Path("back/benchmarks/rag/dataset/pubmedqa/pubmedqa_golden.yaml")
DEFAULT_SAMPLE = Path("back/benchmarks/rag/dataset/pubmedqa/sampled_300.json")
DEFAULT_NEGATIVE_TRACE = Path("revision_release/dataset_provenance/synthetic_negative_trace.csv")
DEFAULT_OUTPUT = Path("ablation_portability/dataset")


def resolve_default_historical() -> Path:
    """Find the July run in either known checkout layout, preferring its original location."""
    candidates = (PROJECT_ROOT / DEFAULT_HISTORICAL, PROJECT_ROOT / LEGACY_HISTORICAL)
    return next((path for path in candidates if path.is_file()), candidates[0])


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install it in the isolated portability environment") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _positive_sample_map(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in data:
        pmid = str(row.get("pubid", ""))
        if pmid:
            result[pmid] = row
    return result


def _pmids(row: dict[str, Any]) -> list[str]:
    values = row.get("original_pmids") or row.get("pmids") or []
    if isinstance(values, (str, int)):
        values = [values]
    result = [str(value) for value in values if str(value)]
    for paper_id in row.get("expected_papers", []) or []:
        match = re.search(r"(?:pubmed[_-])?(\d{5,})", str(paper_id), re.IGNORECASE)
        if match and match.group(1) not in result:
            result.append(match.group(1))
    match = re.match(r"pubmedqa_(\d+)", str(row.get("id", "")), re.IGNORECASE)
    if match and match.group(1) not in result:
        result.insert(0, match.group(1))
    return result


def _canonical_row(
    source: dict[str, Any],
    row_index: int,
    historical_run_id: str,
    golden_by_id: dict[str, dict[str, Any]],
    sampled_by_pmid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    query_id = str(source.get("id", ""))
    if not query_id:
        raise ValueError(f"Historical row {row_index} has no query ID")
    question = source.get("query", source.get("question"))
    if not isinstance(question, str) or not question:
        raise ValueError(f"Historical row {query_id} has no exact query text")
    raw_gold = source.get("golden_answer", source.get("gold_reference_answer", ""))
    if raw_gold is None:
        raw_gold = ""
    if not isinstance(raw_gold, str):
        raw_gold = str(raw_gold)
    answerable = source.get("is_answerable")
    if not isinstance(answerable, bool):
        raise ValueError(f"Historical row {query_id} has no boolean is_answerable label")

    golden = golden_by_id.get(query_id)
    if golden is None:
        raise ValueError(f"Historical ID is absent from golden source: {query_id}")
    if golden.get("query") != question:
        raise ValueError(f"Question text mismatch between archived and golden row: {query_id}")
    if isinstance(golden.get("is_answerable"), bool) and golden["is_answerable"] != answerable:
        raise ValueError(f"Answerability mismatch between archived and golden row: {query_id}")
    if source.get("category") != golden.get("category"):
        raise ValueError(f"Category mismatch between archived and golden row: {query_id}")
    archived_papers = source.get("expected_papers") or []
    if archived_papers != (golden.get("expected_papers") or []):
        raise ValueError(f"Source paper identity mismatch between archived and golden row: {query_id}")
    pmids = _pmids(source)
    if len(pmids) != 1:
        raise ValueError(f"Expected exactly one original source PMID for {query_id}; found {pmids}")

    if answerable:
        if raw_gold != str(golden.get("golden_answer") or ""):
            raise ValueError(f"Gold answer mismatch between archived and golden row: {query_id}")
        sample = sampled_by_pmid.get(pmids[0])
        if sample is None:
            raise ValueError(f"Positive PMID missing from sampled_300.json: {query_id}")
        if sample.get("question") != question:
            raise ValueError(f"Question text mismatch against sampled_300.json: {query_id}")
        source_decision = str(sample.get("final_decision", "")).strip()
        source_long_answer = str(sample.get("long_answer", ""))
        source_gold = f"{source_decision.capitalize()}. {source_long_answer}"
        if source_gold != raw_gold:
            raise ValueError(f"Gold answer mismatch against PubMedQA source fields: {query_id}")
    else:
        category = source.get("category") or golden.get("category")
        if not category or not str(category).startswith("unanswerable-"):
            raise ValueError(f"Negative row has no synthetic category: {query_id}")
        if raw_gold != str(golden.get("golden_answer") or ""):
            raise ValueError(f"Negative gold mismatch between archived and golden row: {query_id}")
        if pmids[0] not in sampled_by_pmid:
            raise ValueError(f"Negative source PMID missing from sampled_300.json: {query_id}")

    category = source.get("category", golden.get("category"))
    return {
        "row_index": row_index,
        "query_id": query_id,
        "question": question,
        "gold_reference_answer": raw_gold,
        "answerability_label": "answerable" if answerable else "designated_negative",
        "source_pmid": pmids[0],
        "original_pmids": pmids,
        "original_paper_ids": [str(value) for value in archived_papers],
        "historical_category": category,
        "synthetic_negative_category": (
            str(category) if not answerable else None
        ),
        "source_metadata": {
            "source_dataset": "PubMedQA",
            "source_dataset_revision": "unknown; original source revision was not recorded",
            "historical_run_id": historical_run_id,
            "historical_category": category,
            "description": golden.get("description"),
        },
    }


def extract_dataset(
    historical_path: Path,
    golden_path: Path,
    sample_path: Path,
    output_dir: Path,
    historical_run_id: str | None = None,
    negative_trace_path: Path | None = None,
) -> dict[str, Any]:
    for path in (historical_path, golden_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required extraction source is absent: {path}")
    archived = load_yaml(historical_path)
    golden = load_yaml(golden_path)
    if not isinstance(archived, list) or not isinstance(golden, list):
        raise ValueError("Historical retrieval artifact and golden source must be YAML lists")
    golden_by_id = {str(item.get("id")): item for item in golden if item.get("id")}
    if len(golden_by_id) != len(golden):
        raise ValueError("Golden source contains missing or duplicate IDs")
    sample_by_pmid = _positive_sample_map(sample_path)
    run_id = historical_run_id or historical_path.parent.name

    records = [
        _canonical_row(row, index, run_id, golden_by_id, sample_by_pmid)
        for index, row in enumerate(archived, start=1)
    ]
    ids = [row["query_id"] for row in records]
    if len(records) != 375 or len(set(ids)) != 375:
        raise ValueError(f"Expected exactly 375 unique historical rows; got {len(records)}")
    answerable_count = sum(row["answerability_label"] == "answerable" for row in records)
    negative_count = sum(row["answerability_label"] == "designated_negative" for row in records)
    if (answerable_count, negative_count) != (300, 75):
        raise ValueError(
            f"Expected 300 answerable and 75 designated negatives; got {answerable_count}/{negative_count}"
        )
    duplicate_question_count = len(records) - len({row["question"] for row in records})
    if duplicate_question_count:
        raise ValueError(
            f"Canonical historical population contains {duplicate_question_count} duplicate question texts"
        )

    historical_metrics_path = historical_path.parent / "metrics_details.csv"
    if historical_metrics_path.is_file():
        with historical_metrics_path.open("r", encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        expected_id_map = {row["query_id"]: row for row in records}
        actual_ids = {str(row.get("query_id") or "") for row in metric_rows}
        if len(metric_rows) != 1875 or actual_ids != set(expected_id_map):
            raise ValueError(
                "Historical metrics evidence does not contain the same 375 query IDs and 1,875 records: "
                f"rows={len(metric_rows)}, query_ids={len(actual_ids)}"
            )
        for metric_row in metric_rows:
            expected = expected_id_map[metric_row["query_id"]]
            expected_answerable = expected["answerability_label"] == "answerable"
            if (metric_row.get("is_answerable", "").lower() == "true") != expected_answerable:
                raise ValueError(f"Historical answerability mismatch in run metrics: {metric_row['query_id']}")
            if metric_row.get("category") != expected["historical_category"]:
                raise ValueError(f"Historical category mismatch in run metrics: {metric_row['query_id']}")

    negative_trace: dict[str, dict[str, str]] = {}
    if negative_trace_path is not None:
        if not negative_trace_path.is_file():
            raise FileNotFoundError(f"Negative provenance trace is absent: {negative_trace_path}")
        with negative_trace_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = str(row.get("synthetic_query_id") or "")
                if key:
                    if key in negative_trace:
                        raise ValueError(f"Duplicate negative provenance trace ID: {key}")
                    negative_trace[key] = row
        expected_negative_ids = {
            row["query_id"] for row in records if row["answerability_label"] == "designated_negative"
        }
        if set(negative_trace) != expected_negative_ids:
            raise ValueError(
                "Synthetic-negative audit IDs do not match the archived negative population: "
                f"missing={sorted(expected_negative_ids-set(negative_trace))[:5]}, "
                f"extra={sorted(set(negative_trace)-expected_negative_ids)[:5]}"
            )
        for row in records:
            if row["answerability_label"] != "designated_negative":
                continue
            audit = negative_trace[row["query_id"]]
            if audit.get("source_pmid") != row["source_pmid"]:
                raise ValueError(f"Source PMID mismatch against synthetic-negative audit: {row['query_id']}")
            if audit.get("construction_category") != row["synthetic_negative_category"]:
                raise ValueError(f"Negative category mismatch against synthetic-negative audit: {row['query_id']}")
            if audit.get("synthetic_modified_question") != row["question"]:
                raise ValueError(f"Negative question mismatch against synthetic-negative audit: {row['query_id']}")

    source_paths = {
        "historical_population_authority": historical_path,
        "gold_reference_and_category_crosscheck": golden_path,
        "source_pmid_crosscheck": sample_path,
    }
    if historical_metrics_path.is_file():
        source_paths["historical_run_membership_crosscheck"] = historical_metrics_path
    if negative_trace_path is not None:
        source_paths["synthetic_negative_provenance_audit"] = negative_trace_path
    source_hashes = {role: sha256_file(path) for role, path in source_paths.items()}

    jsonl = b"".join(canonical_json_bytes(row) + b"\n" for row in records)
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "questions.jsonl"
    questions_path.write_bytes(jsonl)

    field_sources = {
        "row_index": ("historical_population_authority", "array_position", "historical ordering"),
        "query_id": ("historical_population_authority", "id", "exact golden ID match"),
        "question": ("historical_population_authority", "query", "exact golden text match; positive PubMedQA sample match; negative audit match when supplied"),
        "gold_reference_answer": ("historical_population_authority", "golden_answer", "exact golden answer match; positive sample final_decision + long_answer serialization match"),
        "answerability_label": ("historical_population_authority", "is_answerable", "exact golden label match; historical metrics label match"),
        "source_pmid": ("historical_population_authority", "expected_papers", "exact golden paper ID match; PubMedQA sample pubid match"),
        "original_paper_ids": ("historical_population_authority", "expected_papers", "exact golden paper ID match"),
        "historical_category": ("historical_population_authority", "category", "exact golden category match; historical metrics category match"),
        "synthetic_negative_category": ("historical_population_authority", "category", "exact golden category match; negative audit match when supplied"),
        "historical_run_membership": ("historical_population_authority", "artifact membership", "archived July run path; query ID set cross-checked against 1,875 run metric rows"),
    }
    provenance_path = output_dir / "questions_provenance.csv"
    with provenance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index", "query_id", "field_name", "value_sha256", "value_preview",
                "source_path", "source_field", "source_sha256", "validated_against", "validation_status",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        provenance_fields = list(field_sources)
        for record in records:
            for field_name in provenance_fields:
                value = record.get(field_name)
                source_role, source_field, checks = field_sources[field_name]
                validation_status = "verified"
                if field_name == "synthetic_negative_category" and value is None:
                    checks = "answerable row; no synthetic-negative category applies"
                if (
                    field_name == "synthetic_negative_category"
                    and value is not None
                    and negative_trace_path is not None
                ):
                    checks += "; exact synthetic-negative audit match"
                if (
                    field_name == "historical_run_membership"
                    and not historical_metrics_path.is_file()
                ):
                    validation_status = "source_artifact_membership_only"
                if field_name == "historical_run_membership":
                    value = record.get("source_metadata", {}).get("historical_run_id")
                encoded = canonical_json_bytes(value)
                preview = str(value).replace("\n", " ")
                writer.writerow({
                    "row_index": record["row_index"],
                    "query_id": record["query_id"],
                    "field_name": field_name,
                    "value_sha256": hashlib.sha256(encoded).hexdigest(),
                    "value_preview": preview[:240],
                    "source_path": _display_path(source_paths[source_role]),
                    "source_field": source_field,
                    "source_sha256": source_hashes[source_role],
                    "validated_against": checks,
                    "validation_status": validation_status,
                })

    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "1.0",
        "evidence_level": "EXACT_SOURCE_RECOVERED",
        "row_count": len(records),
        "unique_query_id_count": len(set(ids)),
        "answerable_count": answerable_count,
        "designated_negative_count": negative_count,
        "dataset_sha256": hashlib.sha256(jsonl).hexdigest(),
        "questions_provenance_sha256": sha256_file(provenance_path),
        "duplicate_question_text_count": duplicate_question_count,
        "serialization": "UTF-8 JSONL; one compact sorted-key JSON object per line; ensure_ascii=false; LF terminator",
        "ordering": "exact row order of canonical archived retrieved_contexts.yaml; row_index is one-based",
        "generation_script": {
            "module": "ablation_portability.scripts.freeze_dataset",
            "sha256": sha256_file(script_path),
        },
        "canonical_sources": [
            {"role": role, "path": _display_path(path), "sha256": source_hashes[role]}
            for role, path in source_paths.items()
        ],
        "historical_run_id": run_id,
        "answerability_semantics": {
            "answerable": "source is_answerable=true",
            "designated_negative": "source is_answerable=false; negative category preserved from archive/golden source",
            "negative_gold_serialization": "preserve archived string; source null normalized to empty string only when archived field is null",
            "positive_gold_crosscheck": "archived and golden text preserved verbatim; sampled_300 cross-check is exact '<Final_decision>. <long_answer>' serialization using original source fields",
        },
    }

    manifest_path = output_dir / "questions_manifest.json"
    manifest_path.write_bytes(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (questions_path, manifest_path, provenance_path)
        ),
        encoding="utf-8",
    )
    load_frozen_dataset(output_dir)
    return manifest


def verify_dataset(output_dir: Path) -> dict[str, Any]:
    records = load_frozen_dataset(output_dir)
    manifest = json.loads((output_dir / "questions_manifest.json").read_text(encoding="utf-8"))
    expected = (375, 300, 75)
    actual = (
        len(records),
        sum(row["answerability_label"] == "answerable" for row in records),
        sum(row["answerability_label"] == "designated_negative" for row in records),
    )
    if actual != expected:
        raise FrozenDatasetError(f"Expected population {expected}, found {actual}")
    provenance_path = output_dir / "questions_provenance.csv"
    if not provenance_path.is_file() or sha256_file(provenance_path) != manifest.get("questions_provenance_sha256"):
        raise FrozenDatasetError("questions_provenance.csv SHA-256 does not match its manifest")
    checksum_path = output_dir / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FrozenDatasetError("Dataset SHA256SUMS file is missing")
    expected_checksums = {
        "questions.jsonl": manifest.get("dataset_sha256"),
        "questions_manifest.json": sha256_file(output_dir / "questions_manifest.json"),
        "questions_provenance.csv": manifest.get("questions_provenance_sha256"),
    }
    checksum_lines = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise FrozenDatasetError("Dataset SHA256SUMS contains a malformed line")
        checksum_lines[parts[1]] = parts[0]
    if checksum_lines != expected_checksums:
        raise FrozenDatasetError("Dataset SHA256SUMS does not match the frozen dataset artifacts")
    return {"valid": True, "row_count": actual[0], "answerable_count": actual[1], "designated_negative_count": actual[2], "sha256": manifest["dataset_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or verify the exact archived 375-query population")
    parser.add_argument("--historical", type=Path, default=None)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--negative-trace", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--historical-run-id")
    parser.add_argument("--verify", action="store_true", help="Verify the existing frozen JSONL and manifest")
    args = parser.parse_args()
    historical_path = args.historical or resolve_default_historical()
    negative_trace_path = args.negative_trace
    if negative_trace_path is None and DEFAULT_NEGATIVE_TRACE.is_file():
        negative_trace_path = DEFAULT_NEGATIVE_TRACE
    result = verify_dataset(args.output_dir) if args.verify else extract_dataset(
        historical_path,
        args.golden,
        args.sample,
        args.output_dir,
        args.historical_run_id,
        negative_trace_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
