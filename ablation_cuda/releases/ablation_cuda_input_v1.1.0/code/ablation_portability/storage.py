from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import struct
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def read_only_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"SQLite input is not a regular file: {resolved}")
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True, timeout=30.0)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def source_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    identity: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256_main_file": sha256_file(resolved),
    }
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{resolved}{suffix}")
        identity[f"{suffix[1:]}_present"] = sidecar.exists()
        if sidecar.is_file():
            identity[f"sha256_{suffix[1:]}"] = sha256_file(sidecar)
            identity[f"{suffix[1:]}_size_bytes"] = sidecar.stat().st_size
    return identity


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
        )
        if not row[0].startswith("sqlite_")
    ]


def _is_fts_shadow(name: str) -> bool:
    return any(
        name.endswith(suffix)
        for suffix in ("_data", "_idx", "_docsize", "_config", "_content")
    )


def inspect_database(path: Path, include_samples: bool = True) -> dict[str, Any]:
    conn = read_only_connection(path)
    try:
        tables = _tables(conn)
        table_counts: dict[str, int] = {}
        for table in tables:
            if _is_fts_shadow(table):
                continue
            quoted = '"' + table.replace('"', '""') + '"'
            table_counts[table] = int(
                conn.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
            )

        info: dict[str, Any] = {
            "sqlite_version": sqlite3.sqlite_version,
            "user_version": int(conn.execute("PRAGMA user_version").fetchone()[0]),
            "journal_mode": str(conn.execute("PRAGMA journal_mode").fetchone()[0]),
            "quick_check": str(conn.execute("PRAGMA quick_check").fetchone()[0]),
            "table_counts": table_counts,
            "fts_objects": [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%USING fts5%' ORDER BY name"
                )
            ],
            "views": [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
                )
            ],
        }

        if "chunks" in tables:
            vector_lengths = [
                {"blob_bytes": int(row[0]), "rows": int(row[1])}
                for row in conn.execute(
                    "SELECT length(embedding), count(*) FROM chunks "
                    "WHERE embedding IS NOT NULL GROUP BY length(embedding) ORDER BY length(embedding)"
                )
            ]
            dimensions = sorted(
                {
                    item["blob_bytes"] // 4
                    for item in vector_lengths
                    if item["blob_bytes"] % 4 == 0
                }
            )
            info["vector"] = {
                "rows": sum(item["rows"] for item in vector_lengths),
                "blob_lengths": vector_lengths,
                "dimensions": dimensions,
                "dtype": "float32",
            }
            if include_samples:
                samples = conn.execute(
                    "SELECT id, paper_id, text_content, embedding FROM chunks ORDER BY id LIMIT 16"
                ).fetchall()
                info["chunk_samples"] = [
                    {
                        "id": row["id"],
                        "paper_id": row["paper_id"],
                        "text_sha256": hashlib.sha256(
                            (row["text_content"] or "").encode("utf-8")
                        ).hexdigest(),
                        "vector_sha256": hashlib.sha256(row["embedding"] or b"").hexdigest(),
                    }
                    for row in samples
                ]

        if "nodes" in tables:
            info["node_labels"] = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT label, count(*) FROM nodes GROUP BY label ORDER BY label"
                )
            }
        if "edges" in tables:
            info["edge_types"] = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT type, count(*) FROM edges GROUP BY type ORDER BY type"
                )
            }
        if "chunks" in tables:
            info["document_count"] = int(
                conn.execute("SELECT count(DISTINCT paper_id) FROM chunks").fetchone()[0]
            )
            info["chunk_count"] = int(
                conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            )
        if "parent_chunks" in tables:
            info["parent_count"] = int(
                conn.execute("SELECT count(*) FROM parent_chunks").fetchone()[0]
            )
        if "nodes" in tables:
            info["graph_node_count"] = int(
                conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
            )
        if "edges" in tables:
            info["graph_edge_count"] = int(
                conn.execute("SELECT count(*) FROM edges").fetchone()[0]
            )

        fk_columns = conn.execute("PRAGMA foreign_key_check").fetchall()
        info["foreign_key_violation_count"] = len(fk_columns)
        return info
    finally:
        conn.close()


def validate_vectors(path: Path, expected_dimensions: int = 768) -> dict[str, Any]:
    conn = read_only_connection(path)
    try:
        rows = conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        dims: set[int] = set()
        bad_nonfinite: list[str] = []
        bad_lengths: list[str] = []
        zero_norm: list[str] = []
        for row in rows:
            blob = row["embedding"]
            if blob is None or len(blob) % 4:
                bad_lengths.append(str(row["id"]))
                continue
            dim = len(blob) // 4
            dims.add(dim)
            if dim != expected_dimensions:
                bad_lengths.append(str(row["id"]))
                continue
            values = struct.unpack(f"<{dim}f", blob)
            if not all(math.isfinite(value) for value in values):
                bad_nonfinite.append(str(row["id"]))
            if sum(value * value for value in values) <= 0.0:
                zero_norm.append(str(row["id"]))
        return {
            "rows": len(rows),
            "dimensions": sorted(dims),
            "expected_dimensions": expected_dimensions,
            "bad_lengths_or_dimensions": bad_lengths,
            "nonfinite_vectors": bad_nonfinite,
            "zero_norm_vectors": zero_norm,
            "valid": bool(rows) and dims == {expected_dimensions} and not (
                bad_lengths or bad_nonfinite or zero_norm
            ),
        }
    finally:
        conn.close()


def fts_fixture(path: Path) -> dict[str, Any]:
    conn = read_only_connection(path)
    try:
        row = conn.execute(
            "SELECT text_content FROM chunks WHERE trim(text_content) != '' ORDER BY id LIMIT 1"
        ).fetchone()
        import re

        terms = re.findall(r"[A-Za-z0-9_]{4,}", row[0] if row else "")
        for term in terms:
            escaped = '"' + term.replace('"', '""') + '"'
            try:
                found = conn.execute(
                    "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 5", (escaped,)
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            if found:
                return {
                    "query": term,
                    "match_count_at_most_five": len(found),
                    "matching_chunk_ids": [str(item[0]) for item in found],
                }
        raise RuntimeError("Could not find a working FTS5 term in the stored chunk text")
    finally:
        conn.close()


def validate_retrieval_fixtures(
    database_path: Path,
    fixtures_path: Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay frozen chunk, FTS5, and graph fixtures against a restored snapshot."""
    fixture_bytes = fixtures_path.read_bytes()
    actual_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Retrieval fixture SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    fixtures = json.loads(fixture_bytes)
    if fixtures.get("schema_version") != "1.0":
        raise RuntimeError("Unsupported retrieval fixture schema")
    chunks = fixtures.get("chunks")
    graph = fixtures.get("graph")
    fts = fixtures.get("fts")
    if not isinstance(chunks, list) or not chunks or not isinstance(graph, list) or not isinstance(fts, dict):
        raise RuntimeError("Retrieval fixture package is incomplete")

    conn = read_only_connection(database_path)
    try:
        checked_chunks = 0
        for fixture in chunks:
            row = conn.execute(
                "SELECT id, paper_id, text_content, embedding FROM chunks WHERE id=?",
                (fixture["chunk_id"],),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Retrieval fixture chunk is missing: {fixture['chunk_id']}")
            text_sha = hashlib.sha256((row["text_content"] or "").encode("utf-8")).hexdigest()
            vector_blob = row["embedding"] or b""
            if (str(row["paper_id"]) != fixture["paper_id"]
                    or text_sha != fixture["text_sha256"]
                    or hashlib.sha256(vector_blob).hexdigest() != fixture["vector_sha256"]
                    or len(vector_blob) // 4 != fixture["vector_dimension"]
                    or len(vector_blob) % 4):
                raise RuntimeError(f"Retrieval fixture chunk content changed: {fixture['chunk_id']}")
            checked_chunks += 1

        escaped = '"' + str(fts["query"]).replace('"', '""') + '"'
        fts_rows = conn.execute(
            "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY id LIMIT 5",
            (escaped,),
        ).fetchall()
        actual_fts_ids = [str(row[0]) for row in fts_rows]
        if actual_fts_ids != fts.get("matching_chunk_ids"):
            raise RuntimeError("Frozen FTS5 retrieval fixture did not replay exactly")

        checked_graph_nodes = 0
        for fixture in graph:
            node = conn.execute(
                "SELECT id, label FROM nodes WHERE id=?", (fixture["node_id"],)
            ).fetchone()
            if node is None or str(node["label"]) != fixture["label"]:
                raise RuntimeError(f"Frozen graph fixture node changed: {fixture['node_id']}")
            edges = conn.execute(
                "SELECT source_id, target_id, type FROM edges "
                "WHERE source_id=? OR target_id=? ORDER BY source_id,target_id,type",
                (fixture["node_id"], fixture["node_id"]),
            ).fetchall()
            actual_edges = [list(row) for row in edges]
            if actual_edges != fixture.get("incident_edges"):
                raise RuntimeError(f"Frozen graph fixture edges changed: {fixture['node_id']}")
            checked_graph_nodes += 1
    finally:
        conn.close()
    return {
        "fixtures_sha256": actual_sha256,
        "chunk_fixtures_replayed": checked_chunks,
        "fts_fixture_replayed": True,
        "fts_matching_chunk_ids": actual_fts_ids,
        "graph_nodes_replayed": checked_graph_nodes,
        "valid": True,
    }


def stable_id_hash(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:15], 16)
