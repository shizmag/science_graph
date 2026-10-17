from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path, PurePosixPath

from ablation_cuda.runtime.assets import acquire_models
from ablation_cuda.runtime.smoke import (
    DB_ROOT, LOG_ROOT, MODEL_ROOT, OUTPUT_ROOT, RELEASE_ROOT,
    _logical_signature, _prepare_database, _version_manifest,
    run_smoke, validate_release,
)
from ablation_cuda.runtime.assets import sha256_file
from ablation_cuda.runtime.progress import Progress
from ablation_cuda.runtime.full_run import (
    GRAPH_OUTPUT_ROOT, _code_identity, _container_identity, _experiment_identity,
    _find_compatible_run, _model_identity_manifest, finalize_stale_run_for_pack,
    inspect_run_status, run_full_matrix,
)
from ablation_cuda.runtime.packaging import pack_experiment
from ablation_cuda.runtime.checkpoints import atomic_write_json
from ablation_cuda.verify import verify_result_archive


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
    full_runs = inspect_run_status()
    return {
        "release_present": (RELEASE_ROOT / "PACKAGE_MANIFEST.json").is_file(),
        "database_present": db_path.is_file(),
        "database_sha256": sha256_file(db_path) if db_path.is_file() else None,
        "database_logical_fingerprint": _logical_signature(db_path) if db_path.is_file() else None,
        "model_manifest_present": model_manifest_path.is_file(),
        "runs": runs,
        "full_experiments": full_runs.get("runs", []),
    }


def _latest_run() -> Path:
    full_runs = sorted(GRAPH_OUTPUT_ROOT.glob("run_*_OCC-RAG-1.7B_CUDA_ABLATION/run_manifest.json"))
    if full_runs:
        return max(full_runs, key=lambda item: item.stat().st_mtime).parent
    manifests = list((OUTPUT_ROOT / "runs").glob("*/run_manifest.json"))
    if not manifests:
        raise RuntimeError("No smoke run exists to package")
    return max(manifests, key=lambda item: item.stat().st_mtime).parent


def _pack(run_dir: Path, export_dir: Path) -> Path:
    run_dir = run_dir.resolve(strict=True)
    if run_dir.parent == GRAPH_OUTPUT_ROOT.resolve() or run_dir.name.endswith("_OCC-RAG-1.7B_CUDA_ABLATION"):
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "RUNNING":
            finalize_stale_run_for_pack(run_dir)
        archive, _ = pack_experiment(run_dir, export_dir)
        return archive
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


def _smoke_failure_artifacts(error: Exception, *, device_report: dict | None = None,
                             install_report: dict | None = None) -> dict:
    """Persist an early smoke exception as a portable diagnostic run."""
    run_id = f"smoke_failure_{time.time_ns()}"
    run_dir = OUTPUT_ROOT / "runs" / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    error_text = f"{type(error).__name__}: {error}"
    details = {
        "schema_version": "1.0", "run_id": run_id, "status": "BLOCKED",
        "stage": "mandatory_cuda_smoke", "error": error_text,
        "traceback": traceback.format_exc(), "device_preflight": device_report,
        "installation_status": "verified" if install_report else "not_completed",
        "input_release_manifest_sha256": (
            sha256_file(RELEASE_ROOT / "PACKAGE_MANIFEST.json")
            if (RELEASE_ROOT / "PACKAGE_MANIFEST.json").is_file() else None
        ),
        "full_matrix_started": False, "remote_judge_requests_made": False,
    }
    atomic_write_json(run_dir / "logs/smoke_failure.json", details)
    manifest = {
        "schema_version": "1.0", "status": "BLOCKED", "run_id": run_id,
        "full_matrix_started": False, "judge": "not_run",
        "errors": [error_text], "diagnostic_file": "logs/smoke_failure.json",
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    return {"run_directory": str(run_dir), "run_manifest": manifest}


def _existing_run_action(*, new_run: bool, requested_run_id: str | None,
                         resume_only: bool, device_report: dict,
                         export_dir: Path) -> dict | None:
    """Handle a completed compatible run without rerunning smoke or generation."""
    if new_run:
        return None
    model_manifest_path = MODEL_ROOT / "installed_models.json"
    if not model_manifest_path.is_file():
        if resume_only:
            raise RuntimeError("Pinned model identity is unavailable; run `./ablation.sh install` before resume.")
        return None
    release_info = validate_release()
    model_manifest = _model_identity_manifest(model_manifest_path)
    code_sha256 = _code_identity(release_info)
    compatibility_identity = _experiment_identity(
        release_info, model_manifest, _container_identity(), code_sha256,
        device_report["cuda_probe"],
    )
    existing_path, existing_manifest = _find_compatible_run(compatibility_identity)
    if requested_run_id:
        if Path(requested_run_id).name != requested_run_id:
            raise ValueError("--run-id must be a run directory name, without path separators")
        selected = GRAPH_OUTPUT_ROOT / requested_run_id
        manifest_path = selected / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Requested run ID does not exist: {requested_run_id}")
        existing_path, existing_manifest = selected, json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("compatibility_identity") != compatibility_identity:
            raise RuntimeError("Requested run is incompatible with this input package, database, models, source, or GPU")
    if existing_path is None:
        if resume_only:
            raise RuntimeError("No compatible prior run exists to resume. Use `ablation.sh run` to create the first run.")
        return None
    status = existing_manifest.get("status")
    if status == "SCIENTIFIC_VALIDATION_FAILED" and not requested_run_id:
        raise RuntimeError(f"The compatible run failed scientific validation at {existing_path}; inspect it and use --new-run for a separately identified rerun.")
    if status != "COMPLETE":
        return None
    if not (resume_only or requested_run_id):
        raise RuntimeError(f"A compatible completed run already exists at {existing_path}. Use --new-run to create a distinct experiment identity.")
    archive, archive_sha256 = pack_experiment(existing_path, export_dir)
    run_id = existing_manifest.get("run_id")
    return {
        "run_directory": str(existing_path), "run_id": run_id,
        "status": "COMPLETE", "manifest": existing_manifest,
        "archive": str(archive), "archive_sha256": archive_sha256,
        "planned_query_arm_operations": existing_manifest.get("matrix", {}).get("planned_query_arm_operations"),
        "successful_query_arm_operations": existing_manifest.get("matrix", {}).get("completed_query_arm_operations"),
        "failed_query_arm_operations": existing_manifest.get("matrix", {}).get("failed_query_arm_operations"),
        "missing_query_arm_operations": existing_manifest.get("matrix", {}).get("missing_query_arm_operations"),
        "interrupted": False, "already_complete": True,
        "host_run_directory": str(Path(os.environ["ABLA_HOST_OUTPUT_ROOT"]) / "graphs" / existing_path.name)
        if os.environ.get("ABLA_HOST_OUTPUT_ROOT") else None,
        "host_archive": str(Path(os.environ["ABLA_HOST_RETURN_DIR"]) / archive.name)
        if os.environ.get("ABLA_HOST_RETURN_DIR") else None,
        "device_preflight": device_report,
        "install_status": "previously verified; completed run reused",
    }


def _run_full(*, new_run: bool = False, requested_run_id: str | None = None,
              resume_only: bool = False, export_dir: Path = Path("/return")) -> dict:
    # This order makes the real RTX 5090 smoke an unconditional prerequisite
    # for all 5,250 planned query-arm operations.
    device_report = _doctor()
    existing_result = _existing_run_action(
        new_run=new_run, requested_run_id=requested_run_id,
        resume_only=resume_only, device_report=device_report,
        export_dir=export_dir,
    )
    if existing_result is not None:
        return existing_result
    install_report = _install()
    try:
        smoke_result = run_smoke()
    except Exception as exc:
        smoke_result = _smoke_failure_artifacts(
            exc, device_report=device_report, install_report=install_report,
        )
    smoke_archive = _pack(Path(smoke_result["run_directory"]), export_dir)
    smoke_archive_sha256 = sha256_file(smoke_archive)
    if smoke_result["run_manifest"].get("status") == "SUCCESS":
        with tempfile.TemporaryDirectory(prefix="ablation-smoke-verify-") as verify_root:
            smoke_verification = verify_result_archive(
                smoke_archive, expected_sha256=smoke_archive_sha256,
                input_manifest_path=RELEASE_ROOT / "PACKAGE_MANIFEST.json",
                questions_path=RELEASE_ROOT / "dataset/questions.jsonl",
                destination=Path(verify_root) / "verified-smoke",
            )
        if smoke_verification.get("verified") is not True:
            raise RuntimeError("The successful CUDA smoke archive did not pass portable result verification")
        smoke_result["archive_verification"] = {
            "status": "PASSED", "archive_sha256": smoke_archive_sha256,
            "verified_arms": [item["arm"] for item in smoke_verification.get("arms", [])],
        }
    else:
        smoke_result["archive_verification"] = {
            "status": "NOT_RUN_SMOKE_FAILED",
            "reason": "The smoke did not pass; its partial diagnostic archive is preserved and cannot qualify as a successful smoke result.",
        }
    if smoke_result["run_manifest"].get("status") != "SUCCESS":
        raise RuntimeError(
            "Mandatory real CUDA smoke failed; the full workload was not started. "
            f"Diagnostic run: {smoke_result['run_directory']}; diagnostic archive: {smoke_archive}"
        )
    result = run_full_matrix(
        smoke_result=smoke_result, smoke_archive=smoke_archive,
        smoke_archive_sha256=smoke_archive_sha256, new_run=new_run,
        requested_run_id=requested_run_id, resume_only=resume_only,
    )
    result["device_preflight"] = device_report
    result["install_status"] = "verified"
    result["host_run_directory"] = str(Path(os.environ.get("ABLA_HOST_OUTPUT_ROOT", "")) / "graphs" / Path(result["run_directory"]).name) if os.environ.get("ABLA_HOST_OUTPUT_ROOT") else None
    result["host_archive"] = str(Path(os.environ.get("ABLA_HOST_RETURN_DIR", "")) / Path(result["archive"]).name) if os.environ.get("ABLA_HOST_RETURN_DIR") else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="ablation-cuda-runtime")
    parser.add_argument("command", choices=("doctor", "install", "smoke", "run", "resume", "status", "pack", "verify"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--new-run", action="store_true",
                        help="Create a distinct run identity; required after a compatible run has completed")
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
            install_report = _install()
            try:
                run_result = run_smoke()
            except Exception as exc:
                run_result = _smoke_failure_artifacts(exc, install_report=install_report)
            archive = _pack(Path(run_result["run_directory"]), args.export_dir)
            archive_verification = None
            if run_result["run_manifest"].get("status") == "SUCCESS":
                with tempfile.TemporaryDirectory(prefix="ablation-smoke-verify-") as verify_root:
                    archive_verification = verify_result_archive(
                        archive, expected_sha256=sha256_file(archive),
                        input_manifest_path=RELEASE_ROOT / "PACKAGE_MANIFEST.json",
                        questions_path=RELEASE_ROOT / "dataset/questions.jsonl",
                        destination=Path(verify_root) / "verified-smoke",
                    )
            result = {**run_result, "archive": str(archive),
                      "archive_sha256": sha256_file(archive),
                      "archive_verification": ({"status": "PASSED",
                          "verified_arms": [item["arm"] for item in archive_verification.get("arms", [])]}
                          if archive_verification else {"status": "NOT_RUN_SMOKE_FAILED"})}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if run_result["run_manifest"]["status"] != "SUCCESS":
                raise RuntimeError("Two-arm CUDA smoke did not pass all gates; diagnostic archive was still exported")
            return
        elif args.command == "run":
            if args.run_id:
                parser.error("run accepts --new-run but not --run-id; use resume --run-id <directory-name>")
            result = _run_full(new_run=args.new_run, export_dir=args.export_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"Run status: {result['status']}")
            print(f"Experiment directory: {result.get('host_run_directory') or result['run_directory']}")
            print(f"Results archive: {result.get('host_archive') or result['archive']}")
            print(f"Archive SHA-256: {result['archive_sha256']}")
            print("Transfer the archive and its .sha256 sidecar to the author; no judge requests were made.")
            if result["status"] != "COMPLETE":
                raise RuntimeError(f"Full run ended as {result['status']}; diagnostic results were packaged and can be resumed.")
            return
        elif args.command == "resume":
            result = _run_full(requested_run_id=args.run_id, resume_only=True, export_dir=args.export_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"Run status: {result['status']}")
            print(f"Experiment directory: {result.get('host_run_directory') or result['run_directory']}")
            print(f"Results archive: {result.get('host_archive') or result['archive']}")
            print(f"Archive SHA-256: {result['archive_sha256']}")
            print("Transfer the archive and its .sha256 sidecar to the author; no judge requests were made.")
            if result["status"] != "COMPLETE":
                raise RuntimeError(f"Resume ended as {result['status']}; diagnostic results were packaged.")
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
