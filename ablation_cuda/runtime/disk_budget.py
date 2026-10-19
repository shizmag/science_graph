"""Disk-space preflight for the bundled CUDA ablation runtime."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


GIB = 1024 ** 3
DEFAULT_BUDGET = {
    "release_work_bytes": 1 * GIB,
    "model_cache_bytes": 8 * GIB,
    "database_bytes": 1 * GIB,
    "outputs_bytes": 16 * GIB,
    "return_archive_bytes": 16 * GIB,
}


def preflight_disk_space(
    release_root: Path,
    *,
    model_root: Path,
    database_root: Path,
    output_root: Path,
    return_root: Path,
) -> dict[str, Any]:
    """Check required free bytes, aggregating paths that share a filesystem."""
    manifest_path = release_root / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    budget = {**DEFAULT_BUDGET, **manifest.get("disk_budget", {})}
    targets = [
        ("release extraction and working space", release_root, int(budget["release_work_bytes"])),
        ("pinned model cache", model_root, int(budget["model_cache_bytes"])),
        ("restored SQLite and USEARCH index", database_root, int(budget["database_bytes"])),
        ("experiment outputs and checkpoints", output_root, int(budget["outputs_bytes"])),
        ("portable results archive", return_root, int(budget["return_archive_bytes"])),
    ]
    docker_bytes = int(budget.get("docker_build_cache_bytes", 12 * GIB))
    if os.environ.get("ABLA_REQUIRE_DOCKER_SPACE") == "1":
        targets.append(("Docker writable image/build storage", Path("/"), docker_bytes))

    by_device: dict[int, dict[str, Any]] = {}
    immutable_root = release_root.resolve()
    for label, path, required in targets:
        path.mkdir(parents=True, exist_ok=True)
        resolved_path = path.resolve()
        if label != "release extraction and working space":
            try:
                resolved_path.relative_to(immutable_root)
            except ValueError:
                pass
            else:
                raise RuntimeError(
                    f"Persistent path for {label} must be outside the immutable input release: "
                    f"{resolved_path}. Move ABLA_STATE_DIR/ABLA_RETURN_DIR to a separate directory."
                )
        device = path.stat().st_dev
        entry = by_device.setdefault(device, {
            "paths": [], "required_bytes": 0, "available_bytes": shutil.disk_usage(path).free,
        })
        entry["paths"].append({"label": label, "path": str(path), "required_bytes": required})
        entry["required_bytes"] += required
        entry["available_bytes"] = min(entry["available_bytes"], shutil.disk_usage(path).free)

    insufficient = []
    for entry in by_device.values():
        entry["available_gib"] = round(entry["available_bytes"] / GIB, 2)
        entry["required_gib"] = round(entry["required_bytes"] / GIB, 2)
        if entry["available_bytes"] < entry["required_bytes"]:
            insufficient.append(entry)

    result = {
        "status": "INSUFFICIENT" if insufficient else "PASSED",
        "filesystem_budgets": list(by_device.values()),
        "docker_build_cache_reserve_bytes": docker_bytes,
        "docker_storage_checked": os.environ.get("ABLA_REQUIRE_DOCKER_SPACE") == "1",
        "model_weight_floor_bytes": int(budget.get("measured_model_weight_floor_bytes", 0)),
    }
    if insufficient:
        detail = "; ".join(
            f"{item['available_gib']:.2f} GiB free, {item['required_gib']:.2f} GiB required "
            f"for {', '.join(path['label'] for path in item['paths'])}"
            for item in insufficient
        )
        docker_gib = result["docker_build_cache_reserve_bytes"] / GIB
        raise RuntimeError(
            f"Disk preflight failed: {detail}. Free space or move ABLA_STATE_DIR/ABLA_RETURN_DIR "
            f"to larger persistent storage. Also reserve at least {docker_gib:.0f} GiB in Docker's "
            "image/build storage before the first image build."
        )
    return result
