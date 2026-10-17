from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from usearch.index import Index


def stable_id_hash(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:15], 16)


def rebuild_index(database: Path, output: Path | None = None) -> dict:
    database = database.resolve(strict=True)
    output = output or database.with_suffix(".usearch")
    uri = f"{database.as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise RuntimeError("No persisted chunk vectors found")
    dimensions = len(rows[0][1]) // 4
    if any(len(blob) % 4 or len(blob) // 4 != dimensions for _, blob in rows):
        raise RuntimeError("Stored vectors have inconsistent dimensions or byte lengths")

    ids = np.asarray([stable_id_hash(str(chunk_id)) for chunk_id, _ in rows], dtype=np.uint64)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("Stable USEARCH key collision detected")
    vectors = np.stack(
        [np.frombuffer(blob, dtype="<f4") for _, blob in rows], axis=0
    ).astype(np.float32, copy=False)
    if not np.isfinite(vectors).all():
        raise RuntimeError("Stored vectors contain NaN or infinite values")

    index = Index(ndim=dimensions, metric="cos")
    index.add(ids, vectors)
    output.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(output))

    check = Index(ndim=dimensions, metric="cos")
    check.load(str(output))
    if len(check) != len(rows) or check.ndim != dimensions:
        raise RuntimeError("Rebuilt USEARCH sidecar failed count or dimension validation")
    result = {
        "database": str(database),
        "index": str(output),
        "index_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "index_bytes": output.stat().st_size,
        "rows": len(rows),
        "dimensions": dimensions,
        "metric": "cos",
        "key_algorithm": "int(md5(chunk_id UTF-8).hexdigest()[:15], 16)",
        "vector_source": "SQLite chunks.embedding float32 blobs",
        "recomputed_embeddings": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild USEARCH from persisted float32 vectors in a restored database"
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(rebuild_index(args.database, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
