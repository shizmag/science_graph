from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class FrozenDatasetError(ValueError):
    pass


def select_frozen_query(records: list[dict[str, Any]], query_id: str) -> dict[str, Any]:
    """Select one explicitly named record without sampling or mutating the frozen population."""
    matches = [record for record in records if record.get("query_id") == query_id]
    if len(matches) != 1:
        if not matches:
            raise FrozenDatasetError(f"Query ID is absent from the frozen dataset: {query_id}")
        raise FrozenDatasetError(f"Query ID is duplicated in the frozen dataset: {query_id}")
    return matches[0]


def load_frozen_dataset(dataset_dir: Path) -> list[dict[str, Any]]:
    questions_path = dataset_dir / "questions.jsonl"
    manifest_path = dataset_dir / "questions_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        raise FrozenDatasetError("Unsupported or malformed frozen dataset manifest schema")
    raw = questions_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest.get("dataset_sha256"):
        raise FrozenDatasetError("questions.jsonl SHA-256 does not match its manifest")

    records = []
    for row_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FrozenDatasetError(f"Invalid JSONL at row {row_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise FrozenDatasetError(f"Row {row_number} is not a JSON object")
        records.append(record)

    ids = [record.get("query_id") for record in records]
    if len(ids) != manifest.get("row_count"):
        raise FrozenDatasetError("Dataset row count does not match its manifest")
    if any(not isinstance(item, str) or not item for item in ids):
        raise FrozenDatasetError("Every question must have a nonempty string query_id")
    if len(set(ids)) != len(ids):
        raise FrozenDatasetError("Question IDs are not unique")
    if manifest.get("unique_query_id_count") is not None and manifest["unique_query_id_count"] != len(set(ids)):
        raise FrozenDatasetError("Unique query ID count does not match its manifest")
    answerable = sum(item.get("answerability_label") == "answerable" for item in records)
    negatives = sum(item.get("answerability_label") == "designated_negative" for item in records)
    if answerable != manifest.get("answerable_count"):
        raise FrozenDatasetError("Answerable population does not match its manifest")
    if negatives != manifest.get("designated_negative_count"):
        raise FrozenDatasetError("Designated-negative population does not match its manifest")
    if answerable + negatives != len(records):
        raise FrozenDatasetError("Every row must have a recognized answerability label")
    for expected_index, item in enumerate(records, start=1):
        if item.get("row_index") != expected_index:
            raise FrozenDatasetError("Question order does not match one-based row_index order")
        if not isinstance(item.get("question"), str) or not item["question"]:
            raise FrozenDatasetError(f"Missing exact question text: {item.get('query_id')}")
        if not isinstance(item.get("gold_reference_answer"), str):
            raise FrozenDatasetError(f"Gold answer must remain a string: {item.get('query_id')}")
        if item.get("answerability_label") not in {"answerable", "designated_negative"}:
            raise FrozenDatasetError(f"Invalid answerability label: {item.get('query_id')}")
        if not isinstance(item.get("original_pmids"), list) or not isinstance(item.get("original_paper_ids"), list):
            raise FrozenDatasetError(f"Source PMID and paper IDs must be lists: {item.get('query_id')}")
        if item.get("source_pmid") is not None and not isinstance(item.get("source_pmid"), str):
            raise FrozenDatasetError(f"Source PMID must remain a string: {item.get('query_id')}")
        if not isinstance(item.get("source_metadata"), dict):
            raise FrozenDatasetError(f"Source metadata must be an object: {item.get('query_id')}")
        if item.get("answerability_label") == "designated_negative" and not item.get(
            "synthetic_negative_category"
        ):
            raise FrozenDatasetError(f"Missing negative category: {item.get('query_id')}")
    if manifest.get("duplicate_question_text_count") is not None:
        duplicate_count = len(records) - len({item["question"] for item in records})
        if duplicate_count != manifest["duplicate_question_text_count"]:
            raise FrozenDatasetError("Duplicate question count does not match its manifest")
    return records
