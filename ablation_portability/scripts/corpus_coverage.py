from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from ablation_portability.dataset.loader import load_frozen_dataset
from ablation_portability.storage import sha256_file


_PMID_PATTERNS = (
    re.compile(r"(?i)^(?:pubmedqa[_-](?:unans[_-])?|pubmed[_-]|pmid[:_-])?(\d+)$"),
    re.compile(r"(?i)(?:pubmedqa[_-](?:unans[_-])?|pubmed[_-]|pmid[:_-])(\d+)"),
)


def normalize_pmid(value: Any) -> str | None:
    """Extract the source PMID from canonical PubMedQA or PubMed paper identifiers."""
    if value is None:
        return None
    raw = str(value).strip()
    for pattern in _PMID_PATTERNS:
        match = pattern.search(raw)
        if match:
            return match.group(1)
    return None


def _database_document_metadata(database_path: Path) -> tuple[set[str], set[str], dict[str, int]]:
    uri = database_path.resolve(strict=True).as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        documents = connection.execute(
            """
            SELECT n.id, n.label, n.properties, COUNT(c.id) AS chunk_count
            FROM nodes AS n
            JOIN chunks AS c ON c.paper_id = n.id
            GROUP BY n.id, n.label, n.properties
            """
        ).fetchall()
    finally:
        connection.close()

    ids: set[str] = set()
    pmids: set[str] = set()
    labels: Counter[str] = Counter()
    for node_id, label, properties_json, _chunk_count in documents:
        node_id = str(node_id)
        ids.add(node_id)
        labels[str(label)] += 1
        pmid = normalize_pmid(node_id)
        if pmid:
            pmids.add(pmid)
        try:
            properties = json.loads(properties_json or "{}")
        except (TypeError, json.JSONDecodeError):
            properties = {}
        if isinstance(properties, dict):
            for key in ("pmid", "pubmed_id"):
                pmid = normalize_pmid(properties.get(key))
                if pmid:
                    pmids.add(pmid)
    return ids, pmids, dict(sorted(labels.items()))


def build_coverage_report(records: list[dict[str, Any]], database_path: Path) -> dict[str, Any]:
    database_ids, database_pmids, document_labels = _database_document_metadata(database_path)
    per_query = []
    coverage_by_status: dict[str, Counter[str]] = {}
    for record in records:
        expected_ids = [str(value) for value in record.get("original_paper_ids", [])]
        source_pmid = normalize_pmid(record.get("source_pmid"))
        expected_pmids = {normalize_pmid(value) for value in expected_ids}
        expected_pmids.discard(None)
        if source_pmid:
            expected_pmids.add(source_pmid)
        exact_id_present = any(value in database_ids for value in expected_ids)
        pmid_present = bool(expected_pmids.intersection(database_pmids))
        covered = exact_id_present or pmid_present
        group = str(record.get("answerability_label", "unknown"))
        counter = coverage_by_status.setdefault(group, Counter())
        counter["query_count"] += 1
        counter["source_present_count"] += int(covered)
        counter["exact_paper_id_present_count"] += int(exact_id_present)
        counter["source_pmid_present_count"] += int(pmid_present)
        per_query.append(
            {
                "query_id": record.get("query_id"),
                "answerability_label": group,
                "source_pmid": source_pmid,
                "expected_paper_ids": expected_ids,
                "exact_paper_id_present": exact_id_present,
                "source_pmid_present": pmid_present,
                "source_document_present": covered,
            }
        )

    overall = Counter()
    overall["query_count"] = len(per_query)
    overall["source_present_count"] = sum(row["source_document_present"] for row in per_query)
    overall["source_absent_count"] = len(per_query) - overall["source_present_count"]
    all_pmids = {row["source_pmid"] for row in per_query if row["source_pmid"]}
    represented_pmids = all_pmids.intersection(database_pmids)
    missing = [row for row in per_query if not row["source_document_present"]]
    return {
        "database_path": str(database_path.resolve()),
        "database_sha256": sha256_file(database_path),
        "corpus_chunked_document_count": len(database_ids),
        "corpus_chunked_document_labels": document_labels,
        "unique_source_pmid_count": len(all_pmids),
        "unique_source_pmids_present": len(represented_pmids),
        "unique_source_pmids_absent": len(all_pmids - represented_pmids),
        "overall": dict(overall),
        "by_answerability": {key: dict(value) for key, value in sorted(coverage_by_status.items())},
        "missing_source_queries": missing,
        "per_query": per_query,
    }


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        "# Source article coverage in the selected frozen corpus",
        "",
        f"- Database: `{report['database_path']}`",
        f"- SHA-256: `{report['database_sha256']}`",
        f"- Chunked documents: {report['corpus_chunked_document_count']}",
        f"- Unique source PMIDs: {report['unique_source_pmid_count']}",
        f"- Unique source PMIDs represented: {report['unique_source_pmids_present']}",
        f"- Unique source PMIDs absent: {report['unique_source_pmids_absent']}",
        "",
        "## Per-query source availability",
        "",
        "| Population | Queries | Source documents present | Exact paper IDs present | PMID matches |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, row in report["by_answerability"].items():
        lines.append(
            f"| {group} | {row['query_count']} | {row['source_present_count']} | "
            f"{row['exact_paper_id_present_count']} | {row['source_pmid_present_count']} |"
        )
    lines.extend(
        [
            f"| **All** | **{overall['query_count']}** | **{overall['source_present_count']}** |  |  |",
            "",
            "Coverage uses indexed `nodes.id`/`nodes.properties` and requires a matching `chunks.paper_id`; it does not inspect raw article text.",
            "",
        ]
    )
    missing = report["missing_source_queries"]
    if missing:
        lines.extend(["## Missing sources", "", "| Query ID | Label | PMID | Expected paper IDs |", "|---|---|---|---|"])
        for row in missing:
            lines.append(
                f"| {row['query_id']} | {row['answerability_label']} | {row['source_pmid']} | "
                f"{', '.join(row['expected_paper_ids'])} |"
            )
        lines.append("")
    else:
        lines.extend(["## Missing sources", "", "None. All archived source PMIDs resolve to chunked documents in the selected snapshot.", ""])
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Check frozen query source coverage using indexed SQLite metadata")
    parser.add_argument("--dataset-dir", type=Path, default=project_root / "ablation_portability/dataset")
    parser.add_argument(
        "--database",
        type=Path,
        default=project_root / "ablation_portability/snapshots/candidates/external_db/v1/database.db",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_frozen_dataset(args.dataset_dir)
    report = build_coverage_report(records, args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in report if key != "per_query"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
