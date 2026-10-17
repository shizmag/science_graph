from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from ablation_cuda.runtime.assets import acquire_models
from ablation_cuda.runtime.smoke import (
    DB_ROOT, LOG_ROOT, MODEL_ROOT, OUTPUT_ROOT, RELEASE_ROOT,
    _logical_signature, _prepare_database, _version_manifest,
    run_smoke, validate_release,
)
from ablation_cuda.runtime.assets import sha256_file
from ablation_cuda.runtime.progress import Progress


def _install() -> dict:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    progress = Progress(LOG_ROOT / "install-progress.jsonl")
    progress.update("validating frozen input package", completed_arms=0, tokens=0)
    info = validate_release(RELEASE_ROOT)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_manifest_path = MODEL_ROOT / "installed_models.json"
    progress.update("acquiring pinned model assets", completed_arms=0, tokens=0)
    model_report = acquire_models(
        RELEASE_ROOT, MODEL_ROOT,
        progress_callback=lambda stage: progress.update(stage, completed_arms=0, tokens=0),
    )
    model_manifest_path.write_text(json.dumps(model_report, ensure_ascii=False,
                                              sort_keys=True, indent=2) + "\n",
                                   encoding="utf-8")
    progress.update("restoring external_db and rebuilding USEARCH from stored vectors",
                    completed_arms=0, tokens=0)
    db_path, database_report = _prepare_database(info)
    progress.update("recording locked software and installation state", completed_arms=0, tokens=0)
    versions = _version_manifest()
    install_report = {
        "schema_version": "1.0", "input_package_verified": True,
        "model_assets": model_report, "database": database_report,
        "versions": versions,
        "config_isolation": {"HOME": os.environ.get("HOME"),
                             "external_yaml_mount": False,
                             "effective_config_sha256": sha256_file(RELEASE_ROOT / "effective_config.yaml")},
    }
    (MODEL_ROOT / "install_manifest.json").write_text(
        json.dumps(install_report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    (LOG_ROOT / "install_manifest.json").write_text(
        json.dumps(install_report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    progress.update("installation complete", completed_arms=0, tokens=0)
    progress.finish()
    return install_report


def _doctor() -> dict:
    from ablation_cuda.runtime.smoke import _actual_cuda_probe
    result = _actual_cuda_probe()
    return {"status": "CUDA_VISIBLE_AND_EXECUTED", "cuda_probe": result}


def _status() -> dict:
    db_path = DB_ROOT / "database.db"
    model_manifest_path = MODEL_ROOT / "installed_models.json"
    runs_dir = OUTPUT_ROOT / "runs"
    runs = []
    if runs_dir.is_dir():
        for manifest_path in sorted(runs_dir.glob("*/run_manifest.json")):
            try:
                item = json.loads(manifest_path.read_text(encoding="utf-8"))
                runs.append({"run_id": item.get("run_id"), "status": item.get("status"),
                             "query_id": item.get("query_id"), "path": str(manifest_path.parent)})
            except Exception as exc:
                runs.append({"path": str(manifest_path), "error": str(exc)})
    return {
        "release_present": (RELEASE_ROOT / "PACKAGE_MANIFEST.json").is_file(),
        "database_present": db_path.is_file(),
        "database_sha256": sha256_file(db_path) if db_path.is_file() else None,
        "database_logical_fingerprint": _logical_signature(db_path) if db_path.is_file() else None,
        "model_manifest_present": model_manifest_path.is_file(),
        "runs": runs,
    }


def _latest_run() -> Path:
    manifests = list((OUTPUT_ROOT / "runs").glob("*/run_manifest.json"))
    if not manifests:
        raise RuntimeError("No smoke run exists to package")
    return max(manifests, key=lambda item: item.stat().st_mtime).parent


def _pack(run_dir: Path, export_dir: Path) -> Path:
    run_dir = run_dir.resolve(strict=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = run_dir / "logs"
    diagnostics.mkdir(exist_ok=True)
    for log_path in sorted(LOG_ROOT.rglob("*")):
        if log_path.is_file():
            destination = diagnostics / log_path.relative_to(LOG_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, destination)
    payload = [path for path in sorted(run_dir.rglob("*")) if path.is_file()
               and path.name != "RESULTS_MANIFEST.json"]
    if not payload:
        raise RuntimeError("Smoke result directory is empty")
    inventory = [{"path": path.relative_to(run_dir).as_posix(),
                  "size_bytes": path.stat().st_size,
                  "sha256": sha256_file(path)} for path in payload]
    manifest = {
        "schema_version": "1.0", "package_type": "ablation_cuda_results",
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "run_id": json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")).get("run_id"),
        "expected_files": inventory,
        "result_status": json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")).get("status"),
    }
    (run_dir / "RESULTS_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    run_id = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")).get("run_id", "unknown")
    archive = export_dir / f"ablation_cuda_results_v1_{run_id}.tar.gz"
    if archive.exists() or Path(str(archive) + ".sha256").exists():
        existing_sha = sha256_file(archive) if archive.is_file() else None
        expected_sha = (Path(str(archive) + ".sha256").read_text(encoding="utf-8").split()[0]
                        if Path(str(archive) + ".sha256").is_file() else None)
        if existing_sha and existing_sha == expected_sha:
            return archive
        raise FileExistsError(f"Results archive exists with different or unverified content: {archive}")
    with archive.open("wb") as raw:
        import gzip
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(run_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    info = tar.gettarinfo(str(path), arcname=path.relative_to(run_dir).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    digest = sha256_file(archive)
    Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(prog="ablation-cuda-runtime")
    parser.add_argument("command", choices=("doctor", "install", "smoke", "status", "pack", "verify"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--export-dir", type=Path, default=Path("/return"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = _doctor()
        elif args.command == "install":
            result = _install()
        elif args.command == "smoke":
            _install()
            run_result = run_smoke()
            archive = _pack(Path(run_result["run_directory"]), args.export_dir)
            result = {**run_result, "archive": str(archive),
                      "archive_sha256": sha256_file(archive)}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if run_result["run_manifest"]["status"] != "SUCCESS":
                raise RuntimeError("Two-arm CUDA smoke did not pass all gates; diagnostic archive was still exported")
            return
        elif args.command == "status":
            result = _status()
        elif args.command == "pack":
            archive = _pack(args.run_dir or _latest_run(), args.export_dir)
            result = {"archive": str(archive), "sha256": sha256_file(archive)}
        else:
            from ablation_cuda.verify import verify_result_archive
            if not args.archive:
                parser.error("verify requires --archive")
            result = verify_result_archive(
                args.archive,
                expected_sha256=args.expected_sha256,
                input_manifest_path=RELEASE_ROOT / "PACKAGE_MANIFEST.json",
                questions_path=RELEASE_ROOT / "dataset/questions.jsonl",
                destination=args.destination,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
