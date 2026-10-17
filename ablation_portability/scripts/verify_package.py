from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from ablation_portability.dataset.loader import load_frozen_dataset
from ablation_portability.scripts.rebuild_vector_index import rebuild_index
from ablation_portability.scripts.restore_db import restore_snapshot
from ablation_portability.scripts.validate_snapshot import validate_snapshot
from ablation_portability.storage import inspect_database, sha256_file


def _extract_archive(archive_path: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(prefix="ablation-", suffix=".tar", delete=False) as handle:
        tar_path = Path(handle.name)
    try:
        subprocess.run(
            ["zstd", "-q", "-d", "-f", str(archive_path), "-o", str(tar_path)],
            check=True,
        )
        with tarfile.open(tar_path, "r:") as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk() or not member.isfile():
                    raise ValueError(f"Unsafe or unsupported archive member: {member.name}")
                target = destination.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Could not read archive member: {member.name}")
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    finally:
        tar_path.unlink(missing_ok=True)


def _validate_baseline_decision(
    snapshot_manifest_path: Path,
    snapshot_manifest: dict,
    package_root: Path,
) -> dict | None:
    selection = snapshot_manifest.get("baseline_selection", {})
    status = selection.get("status")
    if status not in {"author_selected", "candidate_not_selected"}:
        return None
    decision_relpath = selection.get("decision_file")
    if not isinstance(decision_relpath, str) or not decision_relpath:
        raise ValueError(f"Snapshot decision file path is missing: {snapshot_manifest_path}")
    decision_path = (snapshot_manifest_path.parent / decision_relpath).resolve(strict=True)
    try:
        decision_path.relative_to(package_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Snapshot decision record escapes package root: {snapshot_manifest_path}") from exc
    if not decision_path.is_file():
        raise ValueError(f"Snapshot decision record is not a file: {decision_path}")
    if sha256_file(decision_path) != selection.get("decision_sha256"):
        raise ValueError(f"Snapshot decision record checksum mismatch: {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    candidate = snapshot_manifest.get("source_candidate")
    database_sha256 = snapshot_manifest.get("snapshot_identity", {}).get("sha256")
    if status == "author_selected":
        if decision.get("selected_candidate") != candidate:
            raise ValueError(f"Selected candidate disagrees with its decision record: {snapshot_manifest_path}")
        if decision.get("selected_snapshot_sha256") != database_sha256:
            raise ValueError(f"Selected database hash disagrees with its decision record: {snapshot_manifest_path}")
    elif decision.get("selected_candidate") == candidate:
        raise ValueError(f"Nonselected candidate is selected in the decision record: {snapshot_manifest_path}")
    return decision


def verify_package(archive_path: Path, extract_to: Path | None = None) -> dict:
    archive_path = archive_path.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="ablation-verify-") as tmp:
        root = Path(tmp) / "payload"
        root.mkdir()
        _extract_archive(archive_path, root)
        manifest_path = root / "PACKAGE_MANIFEST.json"
        if not manifest_path.is_file():
            raise ValueError("Package manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "1.0":
            raise ValueError(f"Unsupported package manifest schema: {manifest.get('schema_version')}")
        package_type = manifest.get("package_type")
        if package_type not in {"candidate_snapshot_set", "input_snapshot", "smoke_results"}:
            raise ValueError(f"Unknown package type: {package_type}")
        expected_items = manifest.get("expected_files", [])
        if not isinstance(expected_items, list) or not expected_items:
            raise ValueError("Package manifest expected_files must be a nonempty list")
        if len({item.get("path") for item in expected_items}) != len(expected_items):
            raise ValueError("Package manifest contains duplicate file paths")
        expected = {item["path"]: item for item in expected_items}
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "PACKAGE_MANIFEST.json"
        }
        if actual != set(expected):
            raise ValueError(
                f"Package inventory mismatch; missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}"
            )
        for relative, item in expected.items():
            payload = root / relative
            if not isinstance(item.get("size_bytes"), int) or not isinstance(item.get("sha256"), str):
                raise ValueError(f"Invalid file inventory record: {relative}")
            if len(item["sha256"]) != 64:
                raise ValueError(f"Invalid SHA-256 in file inventory: {relative}")
            if payload.stat().st_size != item["size_bytes"]:
                raise ValueError(f"File size mismatch: {relative}")
            if sha256_file(payload) != item["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {relative}")

        dataset_result = None
        dataset_dir = root / "dataset"
        if (dataset_dir / "questions.jsonl").is_file() or package_type == "input_snapshot":
            if not (dataset_dir / "questions.jsonl").is_file() or not (dataset_dir / "questions_manifest.json").is_file():
                raise ValueError("Dataset files are incomplete")
            records = load_frozen_dataset(dataset_dir)
            counts = {
                "row_count": len(records),
                "answerable_count": sum(row["answerability_label"] == "answerable" for row in records),
                "designated_negative_count": sum(row["answerability_label"] == "designated_negative" for row in records),
            }
            if counts != {"row_count": 375, "answerable_count": 300, "designated_negative_count": 75}:
                raise ValueError(f"Frozen dataset population is invalid: {counts}")
            if manifest.get("dataset_population_counts") != counts:
                raise ValueError("Package dataset population count disagrees with package manifest")
            dataset_result = counts
        elif package_type == "input_snapshot":
            raise ValueError("Input snapshot package is missing the frozen dataset")

        snapshot_dirs = []
        for db_file in root.rglob("database.db"):
            snapshot_dirs.append(db_file.parent)
        if not snapshot_dirs and package_type != "smoke_results":
            raise ValueError("Package has no SQLite snapshot")
        snapshot_results = []
        selected_snapshots = []
        for snapshot_dir in snapshot_dirs:
            snapshot_manifest_path = snapshot_dir / "snapshot_manifest.json"
            if not snapshot_manifest_path.is_file():
                raise ValueError(f"Snapshot manifest is missing: {snapshot_dir}")
            snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
            if snapshot_manifest.get("schema_version") != "1.0":
                raise ValueError(f"Unsupported snapshot manifest schema: {snapshot_dir}")
            if snapshot_manifest.get("source_candidate") is None:
                raise ValueError(f"Snapshot candidate identity is missing: {snapshot_dir}")
            decision = _validate_baseline_decision(snapshot_manifest_path, snapshot_manifest, root)
            if snapshot_manifest.get("baseline_selection", {}).get("status") == "author_selected":
                selected_snapshots.append(
                    {
                        "candidate": snapshot_manifest.get("source_candidate"),
                        "database_sha256": snapshot_manifest.get("snapshot_identity", {}).get("sha256"),
                    }
                )
            validation = validate_snapshot(snapshot_dir)
            logical = validation["logical"]
            if not logical.get("fts_objects"):
                raise ValueError(f"Snapshot lacks required FTS data: {snapshot_dir}")
            if logical.get("vector", {}).get("rows", 0) <= 0 or not logical.get("graph_node_count", 0):
                raise ValueError(f"Snapshot lacks vector or graph data: {snapshot_dir}")
            summary = next(
                (
                    item
                    for item in manifest.get("database_snapshots", [])
                    if PurePosixPath(item.get("path", "")).parent.as_posix()
                    == snapshot_dir.relative_to(root).as_posix()
                ),
                None,
            )
            if summary is None:
                raise ValueError(f"Package manifest has no logical summary for {snapshot_dir}")
            summary_counts = {
                "database_sha256": validation["database_sha256"],
                "document_count": logical.get("document_count"),
                "chunk_count": logical.get("chunk_count"),
                "vector_count": logical.get("vector", {}).get("rows"),
                "vector_dimensions": logical.get("vector", {}).get("dimensions"),
                "graph_node_count": logical.get("graph_node_count"),
                "graph_edge_count": logical.get("graph_edge_count"),
                "fts_objects": logical.get("fts_objects"),
            }
            if any(summary.get(key) != value for key, value in summary_counts.items()):
                raise ValueError(f"Package logical snapshot summary disagrees with database: {snapshot_dir}")
            restored_dir = Path(tmp) / "third_restore" / snapshot_dir.relative_to(root)
            restored = restore_snapshot(snapshot_dir, restored_dir)
            restored_validation = validate_snapshot(restored)
            rebuilt = rebuild_index(restored)
            snapshot_results.append(
                {
                    "path": snapshot_dir.relative_to(root).as_posix(),
                    "validation": validation["valid"],
                    "third_clean_restore": restored_validation["valid"],
                    "rebuilt_vector_index": rebuilt,
                }
            )

        smoke_files = sorted(path.relative_to(root).as_posix() for path in (root / "smoke").rglob("*") if path.is_file()) if (root / "smoke").exists() else []
        smoke_status = manifest.get("smoke_completion_status")
        if package_type == "smoke_results":
            required_smoke = {
                "smoke/run_manifest.json", "smoke/effective_config.json",
                "smoke/retrieval_trace.json", "smoke/prompt.txt", "smoke/generation.json",
                "smoke/telemetry.json", "smoke/execution.log", "smoke/stage_timing_capture.json",
                "smoke/database_identity.json",
                "smoke/pipeline/retrieved_contexts.yaml", "smoke/pipeline/evaluation_results.yaml",
            }
            if not required_smoke.issubset(actual):
                raise ValueError("Smoke package lacks required diagnostic files")
            run_manifest = json.loads((root / "smoke/run_manifest.json").read_text(encoding="utf-8"))
            if run_manifest.get("status") not in {"SUCCESS", "PARTIAL", "BLOCKED"}:
                raise ValueError("Smoke run manifest has invalid completion status")
            if run_manifest.get("status") != smoke_status:
                raise ValueError("Smoke completion status disagrees with package manifest")
            if run_manifest.get("status") == "SUCCESS":
                gates = run_manifest.get("validation_gates")
                if not isinstance(gates, dict) or not gates or not all(gates.values()):
                    raise ValueError("Successful smoke package has incomplete or failed validation gates")
                if run_manifest.get("process_exit_code") != 0:
                    raise ValueError("Successful smoke package has a nonzero process exit code")
        if package_type == "input_snapshot":
            if len(snapshot_results) != 1:
                raise ValueError("Input snapshot package must contain exactly one database snapshot")
            input_manifest_path = root / "snapshot/snapshot_manifest.json"
            input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
            selected = input_manifest.get("baseline_selection", {})
            if selected.get("status") != "author_selected":
                raise ValueError("Input snapshot is not marked as author-selected")
            _validate_baseline_decision(input_manifest_path, input_manifest, root)
            expected_selection = {
                "candidate": input_manifest.get("source_candidate"),
                "database_sha256": input_manifest.get("snapshot_identity", {}).get("sha256"),
            }
            if manifest.get("baseline_selection_status") != "author_selected":
                raise ValueError("Input package manifest does not record its selected baseline")
            if manifest.get("selected_baseline") != expected_selection:
                raise ValueError("Input package baseline identity disagrees with its snapshot")
        elif package_type == "candidate_snapshot_set":
            if len(selected_snapshots) > 1:
                raise ValueError("Candidate package contains multiple author-selected snapshots")
            package_selection = manifest.get("baseline_selection_status")
            if selected_snapshots:
                if package_selection != "author_selected" or manifest.get("selected_baseline") != selected_snapshots[0]:
                    raise ValueError("Candidate package baseline decision disagrees with its snapshot manifests")
                decision_path = root / "snapshots/baseline_decision.json"
                if not decision_path.is_file():
                    raise ValueError("Candidate package is missing snapshots/baseline_decision.json")
            elif package_selection != "baseline_decision_pending" or manifest.get("selected_baseline") is not None:
                raise ValueError("Candidate package incorrectly claims an author-selected baseline")

        result = {
            "valid": True,
            "package_type": package_type,
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "file_count": len(actual),
            "dataset_population": dataset_result,
            "snapshot_count": len(snapshot_results),
            "snapshots": snapshot_results,
            "optional_smoke_artifacts_present": bool(smoke_files),
            "smoke_completion_status": smoke_status,
        }
        if extract_to:
            extract_to.mkdir(parents=True, exist_ok=True)
            for path in root.rglob("*"):
                if path.is_file():
                    target = extract_to / path.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a portable experiment package")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_package(args.archive, args.extract_to), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
