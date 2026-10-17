from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ablation_portability.scripts.freeze_dataset import load_yaml
from ablation_portability.storage import read_only_connection


def _baseline_chunks(row: dict[str, Any], baseline: str) -> list[dict[str, Any]]:
    baselines = row.get("baselines") or {}
    value = baselines.get(baseline) or baselines.get(baseline.upper()) or {}
    chunks = value.get("retrieved_chunks") or []
    return [chunk for chunk in chunks[:5] if isinstance(chunk, dict)]


def _pmid(value: str) -> str | None:
    match = re.search(r"(?:pubmed[_-])?(\d{5,})", value, re.IGNORECASE)
    return match.group(1) if match else None


def _candidate_details(database: Path, chunk: dict[str, Any]) -> dict[str, Any]:
    conn = read_only_connection(database)
    try:
        chunk_id = str(chunk.get("id", ""))
        row = conn.execute(
            "SELECT id, paper_id, text_content, parent_id FROM chunks WHERE id=?",
            (chunk_id,),
        ).fetchone()
        paper_id = str(chunk.get("paper_id", "")) or None
        paper_exists = False
        title_match = False
        title = None
        if paper_id:
            node = conn.execute(
                "SELECT id, properties FROM nodes WHERE id=?", (paper_id,)
            ).fetchone()
            paper_exists = node is not None
            if node is not None:
                try:
                    properties = json.loads(node["properties"] or "{}")
                    title = properties.get("title")
                    saved_title = chunk.get("title") or chunk.get("source_title")
                    title_match = (
                        saved_title.strip().casefold() == str(title).strip().casefold()
                        if saved_title and title
                        else None
                    )
                except Exception:
                    pass

        if row is None:
            return {
                "chunk_id": chunk_id,
                "exact_chunk_id_match": False,
                "parent_id_match": None if chunk.get("parent_id") is None else False,
                "paper_id_match": False,
                "pmid_match": bool(paper_id and _pmid(paper_id) and _pmid(paper_id) in chunk_id),
                "chunk_text_sha256_match": False,
                "archived_text_sha256": hashlib.sha256(
                    str(chunk.get("text_content", "")).encode("utf-8")
                ).hexdigest()
                if chunk.get("text_content") is not None
                else None,
                "database_text_sha256": None,
                "paper_node_exists": paper_exists,
                "title_match": title_match,
                "database_title": title,
            }

        archived_text = chunk.get("text_content")
        saved_text_hash = hashlib.sha256((row["text_content"] or "").encode("utf-8")).hexdigest()
        archived_text_hash = hashlib.sha256(str(archived_text or "").encode("utf-8")).hexdigest() if archived_text is not None else None
        expected_parent = chunk.get("parent_id")
        return {
            "chunk_id": chunk_id,
            "exact_chunk_id_match": row["id"] == chunk_id,
            "parent_id_match": (
                str(row["parent_id"]) == str(expected_parent)
                if expected_parent is not None
                else None
            ),
            "paper_id_match": paper_id is None or str(row["paper_id"]) == paper_id,
            "pmid_match": bool(paper_id and _pmid(paper_id) and _pmid(paper_id) in chunk_id),
            "chunk_text_sha256_match": archived_text_hash is not None and saved_text_hash == archived_text_hash,
            "archived_text_sha256": archived_text_hash,
            "database_text_sha256": saved_text_hash,
            "paper_node_exists": paper_exists,
            "title_match": title_match,
            "database_title": title,
        }
    finally:
        conn.close()


def compare_artifact(artifact: Path, candidates: dict[str, Path], baseline: str = "B6", sample_count: int = 12) -> dict:
    rows = load_yaml(artifact)
    if not isinstance(rows, list):
        raise ValueError("Retrieval artifact must be a YAML list")
    if not rows:
        raise ValueError("Retrieval artifact has no rows")
    count = min(max(sample_count, 1), len(rows))
    positions = sorted({round(i * (len(rows) - 1) / max(count - 1, 1)) for i in range(count)})
    chosen = [rows[position] for position in positions]
    output = {name: {"queries": [], "totals": {}} for name in candidates}
    for historical_row in chosen:
        chunks = _baseline_chunks(historical_row, baseline)
        per_db = {}
        for name, db_path in candidates.items():
            details = [_candidate_details(db_path, chunk) for chunk in chunks]
            per_db[name] = details
            output[name]["queries"].append(
                {
                    "query_id": historical_row.get("id"),
                    "baseline": baseline,
                    "saved_top_five_count": len(chunks),
                    "evidence": details,
                }
            )
        for name in candidates:
            totals = output[name]["totals"]
            details = per_db[name]
            for metric in (
                "exact_chunk_id_match", "parent_id_match", "paper_id_match",
                "chunk_text_sha256_match", "paper_node_exists", "title_match",
            ):
                totals[f"{metric}_matches"] = totals.get(f"{metric}_matches", 0) + sum(
                    item[metric] is True for item in details
                )
                totals[f"{metric}_known"] = totals.get(f"{metric}_known", 0) + sum(
                    item[metric] is not None for item in details
                )
            totals["saved_chunks_compared"] = totals.get("saved_chunks_compared", 0) + len(details)

    return {
        "schema_version": "1.0",
        "artifact": str(artifact),
        "baseline": baseline,
        "artifact_row_count": len(rows),
        "sampled_query_ids": [row.get("id") for row in chosen],
        "sampling_rule": "evenly spaced positions in the artifact's original row order, including endpoints",
        "candidates": output,
        "interpretation": "This comparison only measures saved artifacts present in the input file. Artifact provenance must be independently established before calling it July evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved retrieval evidence to candidate databases")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--repo-db", type=Path, required=True)
    parser.add_argument("--external-db", type=Path, required=True)
    parser.add_argument("--baseline", default="B6")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_artifact(
        args.artifact,
        {"repo_db": args.repo_db, "external_db": args.external_db},
        args.baseline,
        args.sample_count,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
