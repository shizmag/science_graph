from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from ablation_portability.storage import read_only_connection


def build_fixtures(database: Path, output: Path | None = None) -> dict:
    database = database.resolve(strict=True)
    output = output or database.parent / "validation_fixtures.json"
    conn = read_only_connection(database)
    try:
        chunk_rows = conn.execute(
            "SELECT id, paper_id, text_content, embedding FROM chunks ORDER BY id LIMIT 5"
        ).fetchall()
        if not chunk_rows:
            raise RuntimeError("No chunks available for deterministic fixtures")
        term = None
        for row in chunk_rows:
            words = re.findall(r"[A-Za-z0-9_]{4,}", row["text_content"] or "")
            for word in words:
                escaped = '"' + word.replace('"', '""') + '"'
                try:
                    hits = conn.execute(
                        "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY id LIMIT 5",
                        (escaped,),
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                if hits:
                    term = word
                    fts_hits = [str(hit[0]) for hit in hits]
                    break
            if term:
                break
        if not term:
            raise RuntimeError("No deterministic FTS5 fixture could be built")

        graph_fixtures = []
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes'"
        ).fetchone():
            node_rows = conn.execute(
                "SELECT id, label FROM nodes ORDER BY id LIMIT 3"
            ).fetchall()
            for node in node_rows:
                neighbors = conn.execute(
                    "SELECT source_id, target_id, type FROM edges "
                    "WHERE source_id=? OR target_id=? ORDER BY source_id,target_id,type",
                    (node["id"], node["id"]),
                ).fetchall()
                graph_fixtures.append(
                    {
                        "node_id": str(node["id"]),
                        "label": str(node["label"]),
                        "incident_edges": [list(item) for item in neighbors],
                    }
                )

        document = {
            "schema_version": "1.0",
            "fixture_selection": "first five chunk IDs, first FTS5 term with matches, and first three graph node IDs in lexical order",
            "chunks": [
                {
                    "chunk_id": str(row["id"]),
                    "paper_id": str(row["paper_id"]),
                    "text_sha256": hashlib.sha256(
                        (row["text_content"] or "").encode("utf-8")
                    ).hexdigest(),
                    "vector_sha256": hashlib.sha256(row["embedding"] or b"").hexdigest(),
                    "vector_dimension": len(row["embedding"] or b"") // 4,
                }
                for row in chunk_rows
            ],
            "fts": {"query": term, "matching_chunk_ids": fts_hits},
            "graph": graph_fixtures,
        }
    finally:
        conn.close()
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stable storage validation fixtures")
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_fixtures(args.database, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
