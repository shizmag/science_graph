from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
from usearch.index import Index

from ablation_portability.scripts.build_validation_fixtures import build_fixtures
from ablation_portability.scripts.rebuild_vector_index import rebuild_index, stable_id_hash
from ablation_portability.scripts.restore_db import restore_snapshot
from ablation_portability.scripts.validate_snapshot import validate_snapshot


def _index_results(database: Path, fixture: dict, k: int = 10, exact: bool = True) -> dict[str, list[dict]]:
    index_path = database.with_suffix(".usearch")
    index = Index(ndim=768, metric="cos")
    index.load(str(index_path))
    conn = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        id_by_hash = {
            stable_id_hash(str(row[0])): str(row[0])
            for row in conn.execute("SELECT id FROM chunks")
        }
        results = {}
        for row in fixture["chunks"]:
            blob = conn.execute(
                "SELECT embedding FROM chunks WHERE id=?", (row["chunk_id"],)
            ).fetchone()[0]
            vector = np.frombuffer(blob, dtype="<f4")
            matches = index.search(vector, count=min(k, len(index)), exact=exact)
            ids = [int(value) for value in matches.keys.tolist()]
            scores = [float(value) for value in matches.distances.tolist()]
            results[row["chunk_id"]] = [
                {"chunk_id": id_by_hash[key], "cosine_distance": score}
                for key, score in zip(ids, scores)
            ]
        return results
    finally:
        conn.close()


def roundtrip(snapshot: Path, work_root: Path | None = None) -> dict:
    snapshot = snapshot.resolve(strict=True)
    base_fixture = build_fixtures(snapshot / "database.db")
    with tempfile.TemporaryDirectory(prefix="ablation-roundtrip-", dir=work_root) as tmp:
        root = Path(tmp)
        restores = []
        for label in ("restore_a", "restore_b"):
            target_dir = root / label
            db_path = restore_snapshot(snapshot, target_dir)
            validation = validate_snapshot(db_path)
            rebuild = rebuild_index(db_path)
            retrieval = _index_results(db_path, base_fixture)
            restores.append(
                {
                    "label": label,
                    "database_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
                    "validation": validation,
                    "usearch": rebuild,
                    "retrieval_exact": retrieval,
                    "retrieval_approximate": _index_results(db_path, base_fixture, exact=False),
                }
            )

        if restores[0]["retrieval_exact"] != restores[1]["retrieval_exact"]:
            raise RuntimeError("Independent USEARCH rebuilds returned different exact fixture rankings")
        if restores[0]["validation"]["logical"] != restores[1]["validation"]["logical"]:
            raise RuntimeError("Independent restores have different logical database contents")

        approximate_overlaps = []
        for query_id, first in restores[0]["retrieval_approximate"].items():
            second = restores[1]["retrieval_approximate"][query_id]
            first_ids = {item["chunk_id"] for item in first}
            second_ids = {item["chunk_id"] for item in second}
            approximate_overlaps.append(
                {
                    "query_chunk_id": query_id,
                    "top_k": min(len(first), len(second)),
                    "overlap_count": len(first_ids & second_ids),
                    "overlap_fraction": len(first_ids & second_ids) / max(min(len(first), len(second)), 1),
                    "same_order": [item["chunk_id"] for item in first] == [item["chunk_id"] for item in second],
                }
            )

        fixture_path = snapshot / "validation_fixtures.json"
        fixture_path.write_text(
            json.dumps(base_fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path = snapshot / "snapshot_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_fixture_ids"] = [row["chunk_id"] for row in base_fixture["chunks"]]
        manifest["validation_fixtures_sha256"] = hashlib.sha256(
            fixture_path.read_bytes()
        ).hexdigest()
        manifest["roundtrip_validation"] = {
            "independent_restores": 2,
            "usearch_rebuilt_from_persisted_vectors": True,
            "exact_fixture_retrieval_order_identical": True,
            "approximate_top_k_overlap_mean": sum(item["overlap_fraction"] for item in approximate_overlaps) / max(len(approximate_overlaps), 1),
            "approximate_order_identical_for_all_fixtures": all(item["same_order"] for item in approximate_overlaps),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = {
            "schema_version": "1.0",
            "snapshot": str(snapshot),
            "snapshot_sha256": hashlib.sha256((snapshot / "database.db").read_bytes()).hexdigest(),
            "independent_restores": 2,
            "restores": restores,
            "exact_fixture_rankings_identical": True,
            "approximate_fixture_rankings": approximate_overlaps,
            "approximate_top_k_overlap_mean": sum(item["overlap_fraction"] for item in approximate_overlaps) / max(len(approximate_overlaps), 1),
            "approximate_order_identical_for_all_fixtures": all(item["same_order"] for item in approximate_overlaps),
            "fixture_file": str(fixture_path),
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "consistent_identifiers_and_exact_retrieval": True,
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore twice and rebuild USEARCH from saved vectors")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = roundtrip(args.snapshot, args.work_root)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
