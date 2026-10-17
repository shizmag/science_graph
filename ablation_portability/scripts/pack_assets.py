from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from ablation_portability.dataset.loader import load_frozen_dataset
from ablation_portability.storage import inspect_database, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _add_tree(files: dict[str, Path], prefix: str, root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rel = path.relative_to(root).as_posix()
            files[f"{prefix}/{rel}" if prefix else rel] = path


def _code_files() -> dict[str, Path]:
    files = {
        "code/ablation_portability/__init__.py": PROJECT_ROOT / "ablation_portability/__init__.py",
        "code/ablation_portability/storage.py": PROJECT_ROOT / "ablation_portability/storage.py",
        "code/ablation_portability/dataset/__init__.py": PROJECT_ROOT / "ablation_portability/dataset/__init__.py",
        "code/ablation_portability/pyproject.toml": PROJECT_ROOT / "ablation_portability/pyproject.toml",
        "code/ablation_portability/uv.lock": PROJECT_ROOT / "ablation_portability/uv.lock",
        "code/back/benchmarks/rag/core/config.py": PROJECT_ROOT / "back/benchmarks/rag/core/config.py",
        "code/back/benchmarks/rag/run_pipeline.py": PROJECT_ROOT / "back/benchmarks/rag/run_pipeline.py",
        "code/back/benchmarks/rag/run_benchmarks.py": PROJECT_ROOT / "back/benchmarks/rag/run_benchmarks.py",
        "code/back/benchmarks/rag/run_custom_retrieve.py": PROJECT_ROOT / "back/benchmarks/rag/run_custom_retrieve.py",
    }
    for path in sorted((PROJECT_ROOT / "ablation_portability/scripts").glob("*.py")):
        files[f"code/ablation_portability/scripts/{path.name}"] = path
    for path in sorted((PROJECT_ROOT / "ablation_portability/dataset").glob("*.py")):
        files[f"code/ablation_portability/dataset/{path.name}"] = path
    for path in sorted((PROJECT_ROOT / "ablation_portability/tests").glob("*.py")):
        files[f"code/ablation_portability/tests/{path.name}"] = path
    model_lock = PROJECT_ROOT / "ablation_portability/models/model_assets.lock.yaml"
    if model_lock.is_file():
        files["models/model_assets.lock.yaml"] = model_lock
    return files


def _package_manifest(package_type: str, files: dict[str, Path], status: str | None = None) -> dict:
    inventory = [
        {"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(files.items())
    ]
    manifest: dict = {
        "schema_version": "1.0",
        "package_type": package_type,
        "expected_files": inventory,
        "optional_smoke_artifacts_present": False,
        "smoke_completion_status": status,
    }
    dataset_files = {key: val for key, val in files.items() if key.startswith("dataset/")}
    if {"dataset/questions.jsonl", "dataset/questions_manifest.json"}.issubset(dataset_files):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="ablation-package-dataset-") as tmp:
            dataset_root = Path(tmp)
            for name, source in dataset_files.items():
                relative = Path(name).relative_to("dataset")
                target = dataset_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            records = load_frozen_dataset(dataset_root)
        manifest["dataset_population_counts"] = {
            "row_count": len(records),
            "answerable_count": sum(row["answerability_label"] == "answerable" for row in records),
            "designated_negative_count": sum(row["answerability_label"] == "designated_negative" for row in records),
        }

    snapshot_db_entries = [
        (name, path)
        for name, path in files.items()
        if name.endswith("/database.db") or name == "snapshot/database.db"
    ]
    snapshot_summaries = []
    for name, path in snapshot_db_entries:
        logical = inspect_database(path, include_samples=False)
        snapshot_summaries.append(
            {
                "path": name,
                "database_sha256": sha256_file(path),
                "document_count": logical.get("document_count"),
                "chunk_count": logical.get("chunk_count"),
                "vector_count": logical.get("vector", {}).get("rows"),
                "vector_dimensions": logical.get("vector", {}).get("dimensions"),
                "graph_node_count": logical.get("graph_node_count"),
                "graph_edge_count": logical.get("graph_edge_count"),
                "fts_objects": logical.get("fts_objects"),
            }
        )
    manifest["database_snapshots"] = snapshot_summaries
    return manifest


def _write_deterministic_tar(files: dict[str, Path], manifest: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    all_files = dict(files)
    with tempfile.TemporaryDirectory(prefix="ablation-package-") as tmp:
        tar_path = Path(tmp) / "payload.tar"
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
            manifest_bytes = json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2
            ).encode("utf-8") + b"\n"
            info = tarfile.TarInfo("PACKAGE_MANIFEST.json")
            info.size = len(manifest_bytes)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            import io

            archive.addfile(info, io.BytesIO(manifest_bytes))
            for archive_name, source in sorted(all_files.items()):
                info = archive.gettarinfo(str(source), arcname=archive_name)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
        subprocess.run(
            ["zstd", "-q", "-19", "-T1", "-f", str(tar_path), "-o", str(output)],
            check=True,
        )
    digest = sha256_file(output)
    sums = output.parent / "SHA256SUMS"
    archive_name = output.name
    prior = []
    if sums.exists():
        prior = [line for line in sums.read_text(encoding="utf-8").splitlines() if line.split()[-1] != archive_name]
    prior.append(f"{digest}  {archive_name}")
    sums.write_text("\n".join(sorted(prior)) + "\n", encoding="utf-8")


def pack_candidate_set(snapshot_dirs: list[Path], comparison: Path, output: Path) -> None:
    files: dict[str, Path] = {"candidate_comparison.md": comparison.resolve(strict=True)}
    selected_candidates = []
    for snapshot_dir in snapshot_dirs:
        snapshot_dir = snapshot_dir.resolve(strict=True)
        label = snapshot_dir.parent.name
        _add_tree(files, f"snapshots/candidates/{label}/v1", snapshot_dir)
        snapshot_manifest = json.loads((snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8"))
        if snapshot_manifest.get("baseline_selection", {}).get("status") == "author_selected":
            selected_candidates.append(
                {
                    "candidate": label,
                    "database_sha256": snapshot_manifest.get("snapshot_identity", {}).get("sha256"),
                }
            )
    decision_file = comparison.resolve(strict=True).parent / "baseline_decision.json"
    if decision_file.is_file():
        files["snapshots/baseline_decision.json"] = decision_file
    if len(selected_candidates) > 1:
        raise ValueError("Candidate snapshot set has multiple author-selected baselines")
    if selected_candidates and "snapshots/baseline_decision.json" not in files:
        raise ValueError("Author-selected candidate is missing its baseline decision record")
    files.update(_code_files())
    manifest = _package_manifest("candidate_snapshot_set", files)
    manifest["baseline_selection_status"] = (
        "author_selected" if selected_candidates else "baseline_decision_pending"
    )
    manifest["selected_baseline"] = selected_candidates[0] if selected_candidates else None
    _write_deterministic_tar(files, manifest, output)


def pack_input_snapshot(snapshot_dir: Path, dataset_dir: Path, output: Path) -> None:
    snapshot_dir = snapshot_dir.resolve(strict=True)
    dataset_dir = dataset_dir.resolve(strict=True)
    snapshot_manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not snapshot_manifest_path.is_file():
        raise ValueError("Selected input requires a snapshot manifest")
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    if snapshot_manifest.get("baseline_selection", {}).get("status") != "author_selected":
        raise ValueError("Input snapshot packaging requires an explicit author-selected baseline")
    selection = snapshot_manifest["baseline_selection"]
    decision_path = (snapshot_dir / selection.get("decision_file", "")).resolve(strict=True)
    if sha256_file(decision_path) != selection.get("decision_sha256"):
        raise ValueError("Input snapshot's baseline decision checksum is invalid")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("selected_candidate") != snapshot_manifest.get("source_candidate")
        or decision.get("selected_snapshot_sha256") != snapshot_manifest.get("snapshot_identity", {}).get("sha256")
    ):
        raise ValueError("Input snapshot does not match its author decision record")
    records = load_frozen_dataset(dataset_dir)
    if len(records) != 375:
        raise ValueError("Portable input package requires exactly 375 frozen questions")
    answerable = sum(row["answerability_label"] == "answerable" for row in records)
    negatives = sum(row["answerability_label"] == "designated_negative" for row in records)
    if (answerable, negatives) != (300, 75):
        raise ValueError(
            "Portable input package requires exactly 300 answerable items and 75 designated negatives"
        )
    files: dict[str, Path] = {}
    _add_tree(files, "dataset", dataset_dir)
    _add_tree(files, "snapshot", snapshot_dir)
    files["baseline_decision.json"] = decision_path
    files.update(_code_files())
    with tempfile.TemporaryDirectory(prefix="ablation-input-manifest-") as tmp:
        packaged_snapshot_manifest = dict(snapshot_manifest)
        packaged_selection = dict(selection)
        packaged_selection["decision_file"] = "../baseline_decision.json"
        packaged_snapshot_manifest["baseline_selection"] = packaged_selection
        copied_manifest = Path(tmp) / "snapshot_manifest.json"
        copied_manifest.write_text(
            json.dumps(packaged_snapshot_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        files["snapshot/snapshot_manifest.json"] = copied_manifest
        manifest = _package_manifest("input_snapshot", files)
        manifest["baseline_selection_status"] = "author_selected"
        manifest["selected_baseline"] = {
            "candidate": snapshot_manifest.get("source_candidate"),
            "database_sha256": snapshot_manifest.get("snapshot_identity", {}).get("sha256"),
        }
        _write_deterministic_tar(files, manifest, output)


def pack_smoke_results(smoke_dir: Path, output: Path) -> None:
    smoke_dir = smoke_dir.resolve(strict=True)
    files: dict[str, Path] = {}
    _add_tree(files, "smoke", smoke_dir)
    required = {
        "smoke/run_manifest.json", "smoke/effective_config.json",
        "smoke/retrieval_trace.json", "smoke/prompt.txt",
        "smoke/generation.json", "smoke/telemetry.json", "smoke/execution.log",
        "smoke/stage_timing_capture.json",
        "smoke/database_identity.json",
        "smoke/pipeline/retrieved_contexts.yaml",
        "smoke/pipeline/evaluation_results.yaml",
    }
    missing = sorted(required - set(files))
    if missing:
        raise ValueError(f"Smoke package is missing required diagnostic artifacts: {missing}")
    run_manifest = json.loads(files["smoke/run_manifest.json"].read_text(encoding="utf-8"))
    status = run_manifest.get("status", "unknown")
    manifest = _package_manifest("smoke_results", files, status=status)
    manifest["optional_smoke_artifacts_present"] = True
    _write_deterministic_tar(files, manifest, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inspectable zstd tar packages")
    parser.add_argument("--type", choices=("candidate_snapshot_set", "input_snapshot", "smoke_results"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, action="append", default=[])
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    if args.type == "candidate_snapshot_set":
        if not args.snapshot or not args.comparison:
            parser.error("candidate_snapshot_set requires --snapshot (repeatable) and --comparison")
        pack_candidate_set(args.snapshot, args.comparison, args.output)
    elif args.type == "input_snapshot":
        if len(args.snapshot) != 1 or not args.dataset_dir:
            parser.error("input_snapshot requires one --snapshot and --dataset-dir")
        pack_input_snapshot(args.snapshot[0], args.dataset_dir, args.output)
    else:
        if not args.smoke_dir:
            parser.error("smoke_results requires --smoke-dir")
        pack_smoke_results(args.smoke_dir, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
