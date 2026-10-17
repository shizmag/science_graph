from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ablation_portability.storage import (
    inspect_database,
    read_only_connection,
    sha256_file,
    source_file_identity,
)


def repository_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def working_tree_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(status.strip())
    except Exception:
        return None


def create_snapshot(source: Path, output_dir: Path, candidate_id: str) -> dict:
    source = source.resolve(strict=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Snapshot output must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_before = source_file_identity(source)
    source_probe = read_only_connection(source)
    try:
        source_journal_mode = str(
            source_probe.execute("PRAGMA journal_mode").fetchone()[0]
        )
    finally:
        source_probe.close()
    database_path = output_dir / "database.db"

    with tempfile.NamedTemporaryFile(
        prefix=".database.", suffix=".db.tmp", dir=output_dir, delete=False
    ) as handle:
        temp_path = Path(handle.name)

    try:
        src = read_only_connection(source)
        dst = sqlite3.connect(temp_path, timeout=30.0)
        try:
            dst.execute("PRAGMA journal_mode=DELETE")
            src.backup(dst, pages=2048, sleep=0.05)
            dst.commit()
            # The backup may copy the source database header's WAL mode.
            # Convert the self-contained destination back to DELETE mode so
            # the package has one SQLite file and no target-side WAL state.
            dst.execute("PRAGMA journal_mode=DELETE")
            integrity = dst.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite backup integrity check failed: {integrity}")
        finally:
            dst.close()
            src.close()

        os.replace(temp_path, database_path)
        source_after = source_file_identity(source)
        logical = inspect_database(database_path)
        source_sidecar = source.with_name(source.name.replace(".db", ".usearch"))
        sidecar_info = {
            "present_at_source": source_sidecar.is_file(),
            "copied": False,
            "rebuild_from_persisted_vectors": True,
            "rebuild_command": "python -m ablation_portability.scripts.rebuild_vector_index <restored database.db>",
        }
        if source_sidecar.is_file():
            sidecar_info.update(
                {
                    "source_sha256": sha256_file(source_sidecar),
                    "source_size_bytes": source_sidecar.stat().st_size,
                }
            )

        manifest = {
            "schema_version": "1.0",
            "snapshot_id": f"{candidate_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "source_candidate": candidate_id,
            "source_identity": {
                "capture_path": str(source),
                "before": source_before,
                "after": source_after,
                "stable_during_backup": source_before == source_after,
            },
            "snapshot_identity": {
                "database_file": "database.db",
                "sha256": sha256_file(database_path),
                "size_bytes": database_path.stat().st_size,
            },
            "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_method": "sqlite3.Connection.backup from mode=ro source connection; WAL-aware consistent SQLite backup",
            "sqlite_runtime": {
                "python_sqlite_version": sqlite3.sqlite_version,
                "source_journal_mode": source_journal_mode,
                "target_journal_mode": "delete",
                "quick_check": logical.get("quick_check"),
            },
            "logical_corpus": {
                "document_count": logical.get("document_count"),
                "chunk_count": logical.get("chunk_count"),
                "parent_count": logical.get("parent_count"),
                "vector_count": logical.get("vector", {}).get("rows"),
                "vector_dimensions": logical.get("vector", {}).get("dimensions"),
                "graph_node_count": logical.get("graph_node_count"),
                "graph_edge_count": logical.get("graph_edge_count"),
                "node_labels": logical.get("node_labels", {}),
                "edge_types": logical.get("edge_types", {}),
                "table_counts": logical.get("table_counts", {}),
                "fts_objects": logical.get("fts_objects", []),
            },
            "embedding_model": {
                "model_id": "intfloat/multilingual-e5-base",
                "requested_revision_prefix": "d128750597153bb5987e10b1c3493a34e5",
                "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
                "stored_dimensions": logical.get("vector", {}).get("dimensions"),
                "lineage_status": "unverified pending stored-vector comparison",
            },
            "usearch": sidecar_info,
            "validation_fixture_ids": [],
            "baseline_selection": {
                "status": "candidate_pending_author_decision",
                "author_selection": None,
                "classification": "NEW_BASELINE_CANDIDATE",
            },
            "code_revision": repository_revision(),
            "code_working_tree_dirty": working_tree_dirty(),
            "snapshot_script_sha256": sha256_file(Path(__file__).resolve()),
        }
        (output_dir / "snapshot_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "database_inspection.json").write_text(
            json.dumps(logical, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a WAL-aware read-only SQLite snapshot")
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()
    manifest = create_snapshot(args.source, args.output_dir, args.candidate_id)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
