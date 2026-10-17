from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ablation_portability.storage import read_only_connection, sha256_file


# Thresholds are fixed before model inference. They are intended to catch a
# changed model/preprocessor or vector corruption while allowing normal float32
# backend roundoff for a pinned checkpoint.
COSINE_AGREEMENT_MIN = 0.999
RELATIVE_NORM_DELTA_MAX = 1e-3
TOP10_OVERLAP_MEAN_MIN = 0.90
SAMPLE_COUNT = 16


def compatibility_plan(model_id: str, revision: str, local_model_path: Path) -> dict:
    return {
        "model_id": model_id,
        "revision": revision,
        "sample_count": SAMPLE_COUNT,
        "selection": "16 evenly spaced chunk IDs in lexical ID order, including endpoints",
        "checks": {
            "per_vector_cosine_min": COSINE_AGREEMENT_MIN,
            "relative_l2_norm_delta_max": RELATIVE_NORM_DELTA_MAX,
            "mean_top10_overlap_min": TOP10_OVERLAP_MEAN_MIN,
        },
        "logic": "Reuse is confident only if all dimension checks pass, every sampled vector meets cosine and norm tolerances, and mean top-10 ranking overlap meets its tolerance. Otherwise reuse remains conditional or is rejected based on the recorded failures.",
        "model_source": str(local_model_path),
        "source_checkpoint_sha256": sha256_file(local_model_path / "model.safetensors"),
    }


def compare_embeddings(database: Path, model_path: Path, output: Path, model_id: str, revision: str) -> dict:
    database = database.resolve(strict=True)
    model_path = model_path.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    plan = compatibility_plan(model_id, revision, model_path)
    (output / "embedding_comparison_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for token_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(token_name, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(f"Embedding comparison dependencies are missing: {exc}") from exc

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = SentenceTransformer(str(model_path), device=device)
    max_seq_length = int(getattr(model, "max_seq_length", 512))

    conn = read_only_connection(database)
    try:
        total = int(conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0])
        if total < SAMPLE_COUNT:
            raise RuntimeError(f"Need at least {SAMPLE_COUNT} stored vectors, found {total}")
        positions = sorted(
            {round(i * (total - 1) / (SAMPLE_COUNT - 1)) for i in range(SAMPLE_COUNT)}
        )
        samples = []
        for position in positions:
            row = conn.execute(
                "SELECT id, text_content, embedding FROM chunks WHERE embedding IS NOT NULL "
                "ORDER BY id LIMIT 1 OFFSET ?",
                (position,),
            ).fetchone()
            samples.append(row)
        all_rows = conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    stored_matrix = np.stack(
        [np.frombuffer(row["embedding"], dtype="<f4") for row in all_rows], axis=0
    ).astype(np.float32, copy=False)
    chunk_ids = [str(row["id"]) for row in all_rows]
    text_inputs = [
        row["text_content"] if str(row["text_content"]).startswith("passage: ")
        else "passage: " + str(row["text_content"])
        for row in samples
    ]
    fresh_matrix = model.encode(
        text_inputs,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)

    vectors = []
    overlaps = []
    for row, fresh in zip(samples, fresh_matrix):
        saved = np.frombuffer(row["embedding"], dtype="<f4")
        if saved.shape != fresh.shape:
            cosine = None
            rel_norm = None
            overlap = 0.0
            saved_top = []
            fresh_top = []
        else:
            dot = float(np.dot(saved.astype(np.float64), fresh.astype(np.float64)))
            saved_norm = float(np.linalg.norm(saved.astype(np.float64)))
            fresh_norm = float(np.linalg.norm(fresh.astype(np.float64)))
            cosine = dot / (saved_norm * fresh_norm) if saved_norm and fresh_norm else None
            rel_norm = abs(saved_norm - fresh_norm) / max(saved_norm, fresh_norm, 1e-12)
            saved_scores = stored_matrix @ saved
            fresh_scores = stored_matrix @ fresh
            # Stored E5 vectors are compared with cosine, matching repository logic.
            stored_norms = np.linalg.norm(stored_matrix, axis=1)
            saved_scores = saved_scores / np.maximum(stored_norms * saved_norm, 1e-12)
            fresh_scores = fresh_scores / np.maximum(stored_norms * fresh_norm, 1e-12)
            saved_idx = np.argsort(-saved_scores, kind="stable")[:10]
            fresh_idx = np.argsort(-fresh_scores, kind="stable")[:10]
            saved_top = [chunk_ids[index] for index in saved_idx]
            fresh_top = [chunk_ids[index] for index in fresh_idx]
            overlap = len(set(saved_top) & set(fresh_top)) / 10.0
            overlaps.append(overlap)
        vectors.append(
            {
                "chunk_id": str(row["id"]),
                "text_sha256": hashlib.sha256(str(row["text_content"]).encode("utf-8")).hexdigest(),
                "stored_dimension": int(len(saved)),
                "fresh_dimension": int(len(fresh)),
                "cosine_agreement": cosine,
                "stored_l2_norm": float(np.linalg.norm(saved.astype(np.float64))),
                "fresh_l2_norm": float(np.linalg.norm(fresh.astype(np.float64))),
                "relative_l2_norm_delta": rel_norm,
                "top10_overlap": overlap,
                "stored_top10_ids": saved_top,
                "fresh_top10_ids": fresh_top,
            }
        )

    dimensions_ok = all(row["stored_dimension"] == row["fresh_dimension"] == 768 for row in vectors)
    cosine_ok = all(
        row["cosine_agreement"] is not None and row["cosine_agreement"] >= COSINE_AGREEMENT_MIN
        for row in vectors
    )
    norm_ok = all(
        row["relative_l2_norm_delta"] is not None
        and row["relative_l2_norm_delta"] <= RELATIVE_NORM_DELTA_MAX
        for row in vectors
    )
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    rank_ok = mean_overlap >= TOP10_OVERLAP_MEAN_MIN
    report = {
        "schema_version": "1.0",
        "database": str(database),
        "database_sha256_before": sha256_file(database),
        "model_id": model_id,
        "model_revision": revision,
        "model_path": str(model_path),
        "model_weights_sha256": plan["source_checkpoint_sha256"],
        "device": device,
        "sample_size": len(samples),
        "stored_vector_count": len(all_rows),
        "dimensions_match": dimensions_ok,
        "preprocessing": {
            "document_prefix": "passage: ",
            "query_prefix_in_project": "query: ",
            "pooling": "SentenceTransformer configured mean-token pooling",
            "normalization": "encode(normalize_embeddings=False); no explicit project normalization",
            "max_sequence_length": max_seq_length,
            "truncation": "SentenceTransformer tokenizer/model truncation at max_seq_length",
            "project_cosine": "USEARCH cosine distance; repository returns similarity=1-distance",
        },
        "thresholds_predeclared": plan["checks"],
        "vectors": vectors,
        "ranking": {
            "top_k": 10,
            "mean_top10_overlap": mean_overlap,
            "passed": rank_ok,
        },
        "summary": {
            "cosine_min": min(row["cosine_agreement"] for row in vectors if row["cosine_agreement"] is not None),
            "cosine_median": float(np.median([row["cosine_agreement"] for row in vectors if row["cosine_agreement"] is not None])),
            "relative_norm_delta_max": max(row["relative_l2_norm_delta"] for row in vectors if row["relative_l2_norm_delta"] is not None),
            "dimensions_passed": dimensions_ok,
            "cosine_passed": cosine_ok,
            "norm_passed": norm_ok,
            "ranking_passed": rank_ok,
            "reuse_status": "CONFIDENT" if dimensions_ok and cosine_ok and norm_ok and rank_ok else "CONDITIONAL",
        },
        "database_sha256_after": sha256_file(database),
    }
    (output / "embedding_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved E5 vectors to the pinned local E5 checkpoint")
    parser.add_argument("database", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="intfloat/multilingual-e5-base")
    parser.add_argument("--revision", default="d128750597153bb5987e10b1c3493a34e5a4502a")
    args = parser.parse_args()
    result = compare_embeddings(args.database, args.model_path, args.output_dir, args.model_id, args.revision)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
