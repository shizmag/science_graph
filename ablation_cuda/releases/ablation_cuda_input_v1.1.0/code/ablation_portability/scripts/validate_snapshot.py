from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from ablation_portability.storage import (
    fts_fixture,
    inspect_database,
    read_only_connection,
    sha256_file,
    validate_vectors,
)


def validate_snapshot(snapshot: Path) -> dict:
    db_path = snapshot / "database.db" if snapshot.is_dir() else snapshot
    db_path = db_path.resolve(strict=True)
    conn = read_only_connection(db_path)
    try:
        integrity_check = [row[0] for row in conn.execute("PRAGMA integrity_check")]
        fk_check = [tuple(row) for row in conn.execute("PRAGMA foreign_key_check")]
    finally:
        conn.close()
    logical = inspect_database(db_path)
    vectors = validate_vectors(db_path)
    try:
        fts = fts_fixture(db_path)
        fts_ok = True
    except Exception as exc:
        fts = {"error": str(exc)}
        fts_ok = False

    manifest_path = db_path.parent / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    manifest_errors = []
    if manifest is not None:
        recorded_hash = manifest.get("snapshot_identity", {}).get("sha256")
        if recorded_hash != sha256_file(db_path):
            manifest_errors.append("snapshot database SHA-256 does not match manifest")
        recorded_counts = manifest.get("logical_corpus", {})
        actual_counts = {
            "document_count": logical.get("document_count"),
            "chunk_count": logical.get("chunk_count"),
            "parent_count": logical.get("parent_count"),
            "vector_count": logical.get("vector", {}).get("rows"),
            "graph_node_count": logical.get("graph_node_count"),
            "graph_edge_count": logical.get("graph_edge_count"),
        }
        for key, actual in actual_counts.items():
            if recorded_counts.get(key) != actual:
                manifest_errors.append(f"manifest logical count mismatch: {key}")

    result = {
        "database": str(db_path),
        "database_sha256": sha256_file(db_path),
        "integrity_check": integrity_check,
        "foreign_key_violation_count": len(fk_check),
        "logical": logical,
        "vectors": vectors,
        "fts": fts,
        "manifest_errors": manifest_errors,
        "valid": bool(
            integrity_check == ["ok"]
            and not fk_check
            and vectors["valid"]
            and fts_ok
            and not manifest_errors
        ),
    }
    if not result["valid"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a restored SQLite snapshot")
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_snapshot(args.snapshot), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
