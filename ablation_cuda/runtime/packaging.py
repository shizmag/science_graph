"""Portable, deterministic experiment archive creation and internal inventory."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from ablation_cuda.runtime.checkpoints import atomic_write_json, sha256_file


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    excluded = {"RESULTS_MANIFEST.json", "SHA256SUMS"}
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in excluded or path.name.endswith(".tmp"):
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        rows.append({"path": path.relative_to(run_dir).as_posix(),
                     "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def write_internal_integrity(run_dir: Path, *, result_status: str) -> dict[str, Any]:
    inventory = _inventory(run_dir)
    if not inventory:
        raise RuntimeError(f"Experiment directory is empty: {run_dir}")
    checksum_lines = [f"{item['sha256']}  {item['path']}" for item in inventory]
    (run_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    run_manifest_path = run_dir / "run_manifest.json"
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "2.0", "package_type": "ablation_cuda_results",
        "run_id": run.get("run_id"), "result_status": result_status,
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "expected_files": inventory,
        "schema": {
            "operation_record": "baselines/<arm_id>/results/<query_id>.json",
            "token_telemetry": "Apache Parquet 2.6, zstd, entropy in bits, log probabilities in nats, probabilities as fractions",
            "judge_export": "judge/judge_inputs.jsonl with separate blinded-ID mapping",
        },
    }
    atomic_write_json(run_dir / "RESULTS_MANIFEST.json", manifest)
    return manifest


def pack_experiment(run_dir: Path, export_dir: Path) -> tuple[Path, str]:
    run_dir = run_dir.resolve(strict=True)
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    result_status = run.get("status", "PARTIAL")
    write_internal_integrity(run_dir, result_status=result_status)
    export_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if result_status == "COMPLETE" else f"_{result_status}"
    stem = f"{run_dir.name}{suffix}"
    archive = export_dir / f"{stem}.tar.gz"
    if archive.exists():
        existing = sha256_file(archive)
        sidecar = Path(str(archive) + ".sha256")
        if sidecar.is_file() and sidecar.read_text(encoding="utf-8").split()[0] == existing:
            expected_manifest = (run_dir / "RESULTS_MANIFEST.json").read_bytes()
            try:
                with tarfile.open(archive, "r:gz") as tar:
                    archived_manifest = tar.extractfile(f"{run_dir.name}/RESULTS_MANIFEST.json")
                    if archived_manifest and archived_manifest.read() == expected_manifest:
                        return archive, existing
            except (tarfile.TarError, OSError):
                pass
        index = 2
        while archive.exists():
            archive = export_dir / f"{stem}_rev{index}.tar.gz"
            if not archive.exists():
                break
            index += 1
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           compresslevel=6, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(run_dir.rglob("*")):
                    if not path.is_file() or path.name.endswith(".tmp"):
                        continue
                    info = tar.gettarinfo(str(path), arcname=f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    digest = sha256_file(archive)
    Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, digest
