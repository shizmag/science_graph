from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from ablation_portability.dataset.loader import load_frozen_dataset
from ablation_portability.storage import (
    inspect_database,
    sha256_file,
    validate_retrieval_fixtures,
    validate_vectors,
)
from ablation_cuda import __version__


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_NAME = f"ablation_cuda_input_v{__version__}"
QUESTION_ID = "pubmedqa_21084567"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _redact_author_paths(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if normalized_key in {"api_key", "hf_token", "access_token", "token", "password", "secret"}:
                result[key] = ""
            else:
                result[key] = _redact_author_paths(item)
        return result
    if isinstance(value, list):
        return [_redact_author_paths(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        local_path_markers = ("/Users/", "/Volumes/", "/home/", "/var/folders/",
                              "/private/var/", "C:/Users/", "~/")
        if any(marker in normalized for marker in local_path_markers):
            return "<author-local-path-redacted>"
    return value


def _copy_tree(source: Path, destination: Path, *, suffixes: set[str] | None = None,
               exclude_paths: set[str] | None = None) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source)
        if relative.as_posix() in (exclude_paths or set()):
            continue
        if any(part in {".venv", "__pycache__", "releases", "imported", ".pytest_cache"} for part in relative.parts):
            continue
        if suffixes and path.suffix not in suffixes:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _source_revision() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    diff = b""
    try:
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    except Exception:
        pass
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_tree_policy": "packaged code files are checksummed independently; commit records their base revision",
    }


def _sanitized_effective_config(config_source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Historical effective configuration must be a YAML mapping")
    cfg = raw
    cfg["db_path"] = "/db/database.db"
    cfg["archive_dir"] = "/state/archive"
    cfg["hf_token"] = ""
    cfg.setdefault("llm", {})
    llm = cfg["llm"]
    llm["provider"] = "cuda_torch"
    llm["max_tokens"] = 768
    llm["model_max_context"] = 12000
    llm["temp"] = 0.0
    llm["local"] = {"model_path": "/models/generator", "rag_model_path": "/models/generator"}
    llm["cloud"] = {
        "provider": "disabled", "model_name": "", "cheap_model_name": "",
        "rag_model_name": "", "api_key": "", "base_url": "",
    }
    for secret_key in ("api_key", "hf_token", "token", "password", "secret"):
        if secret_key in llm:
            llm[secret_key] = ""
    cfg.setdefault("embedding", {})["model_name"] = "/models/embedder"
    cfg.setdefault("reranker", {})["model_name"] = "/models/cross_encoder"
    # The historical model folder reports core_web_md 3.8.0. The locked public
    # package is acquired into the persistent model volume at install time.
    cfg.setdefault("spacy", {})["model_name"] = "en_core_web_md"
    cfg.setdefault("ner", {})["model_name"] = "disabled_not_used_by_query_runtime"
    cfg = _redact_author_paths(cfg)
    differences = {
        "historical_source": str(config_source.relative_to(ROOT)),
        "historical_provider": raw.get("llm", {}).get("provider"),
        "runtime_provider": "cuda_torch",
        "generator_weights": "same public OCC-RAG-1.7B BF16 safetensors revision; no quantization",
        "generation_backend_difference": "MLX stream_generate replaced with Transformers/PyTorch SDPA and an explicit greedy token loop; backend output identity is not assumed",
        "query_context_tokens": {"historical": raw.get("llm", {}).get("model_max_context"), "runtime": 12000},
        "output_token_budget": {"historical": raw.get("llm", {}).get("max_tokens"), "runtime": 768},
        "temperature": {"historical": raw.get("llm", {}).get("temp"), "runtime": 0.0},
        "stop_token_ids": [151643, 151645, 151683],
        "configuration_isolation": "external machine YAML is not mounted; a private HOME receives this sanitized frozen file",
        "credential_policy": "all API and HF credentials are empty and omitted from logs",
        "intervention": "arm 2 disables only score_blending when the B6 trace proves a blend over at least two candidates; otherwise it may disable graph_expansion only when B6 inserted its dynamic output",
    }
    return cfg, differences


def _validate_author_inputs() -> tuple[Path, Path, dict, list[dict]]:
    dataset_dir = ROOT / "ablation_portability/dataset"
    records = load_frozen_dataset(dataset_dir)
    if len(records) != 375:
        raise ValueError(f"Expected 375 frozen queries, found {len(records)}")
    answerable = sum(row["answerability_label"] == "answerable" for row in records)
    negatives = sum(row["answerability_label"] == "designated_negative" for row in records)
    if answerable != 300 or negatives != 75:
        raise ValueError(f"Frozen population changed: expected 300 answerable and 75 negative, got {answerable}/{negatives}")
    if len([row for row in records if row["query_id"] == QUESTION_ID]) != 1:
        raise ValueError(f"Required smoke query is missing or duplicated: {QUESTION_ID}")
    snapshot_dir = ROOT / "ablation_portability/snapshots/candidates/external_db/v1"
    snapshot_manifest = json.loads((snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "ablation_portability/snapshots/baseline_decision.json").read_text(encoding="utf-8"))
    if snapshot_manifest.get("source_candidate") != "external_db":
        raise ValueError("The chosen snapshot folder is not external_db")
    if baseline.get("selected_candidate") != "external_db":
        raise ValueError("Author decision no longer selects external_db")
    database = snapshot_dir / "database.db"
    expected_sha = snapshot_manifest.get("snapshot_identity", {}).get("sha256")
    if sha256_file(database) != expected_sha or baseline.get("selected_snapshot_sha256") != expected_sha:
        raise ValueError("Selected external_db snapshot checksum does not match author decision")
    logical = inspect_database(database, include_samples=False)
    vectors = validate_vectors(database)
    fixture_path = snapshot_dir / "validation_fixtures.json"
    fixture_sha256 = sha256_file(fixture_path)
    if snapshot_manifest.get("validation_fixtures_sha256") != fixture_sha256:
        raise ValueError("Selected external_db retrieval fixtures do not match their snapshot manifest")
    fixture_validation = validate_retrieval_fixtures(database, fixture_path, fixture_sha256)
    if logical.get("quick_check") != "ok" or not logical.get("fts_objects"):
        raise ValueError("Selected external_db SQLite/FTS5 validation failed")
    if not vectors.get("valid") or logical.get("vector", {}).get("rows") != 22068:
        raise ValueError("Stored vectors failed validation; release is not allowed to re-embed")
    if logical.get("graph_node_count") != 9680 or logical.get("graph_edge_count") != 11712:
        raise ValueError("Selected external_db graph fingerprint changed")
    return dataset_dir, snapshot_dir, {
        "logical": logical, "vectors": vectors, "manifest": snapshot_manifest,
        "retrieval_fixtures": fixture_validation,
    }, records


def _asset_manifest(model_lock: dict[str, Any]) -> dict[str, Any]:
    assets = model_lock.get("model_assets", {})
    required = {
        "generator": ("occ-ai/OCC-RAG-1.7B", "e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c"),
        "embedder": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"),
        "cross_encoder": ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825"),
    }
    normalized: dict[str, Any] = {}
    for key, (model_id, revision) in required.items():
        item = assets.get(key)
        actual_revision = item.get("revision") if item else None
        if key == "embedder" and item:
            actual_revision = item.get("revision")
        if not item or item.get("model_id") != model_id or actual_revision != revision:
            raise ValueError(f"Model lock mismatch for {key}: expected {model_id}@{revision}, got {actual_revision}")
        files = item.get("files", {})
        if not files or any(len(str(digest)) != 64 for digest in files.values()):
            raise ValueError(f"Model file checksum manifest is missing/invalid for {key}")
        normalized[key] = {
            "model_id": model_id,
            "revision": revision,
            "files": files,
            "weights_format": item.get("weights", item.get("architecture")),
        }
    return {
        "schema_version": "1.0",
        "model_assets": normalized,
        "spacy_auxiliary_model": {
            "package": "en_core_web_md",
            "version": "3.8.0",
            "source_evidence": "recovered model metadata identified en_core_web_md 3.8.0; downloaded automatically with spaCy's versioned model resolver",
        },
    }


def _payload_files(bundle: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(bundle).as_posix(), "size_bytes": path.stat().st_size,
         "sha256": sha256_file(path)}
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json"
    ]


def _write_tar_gz(bundle: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(bundle.rglob("*")):
                    if not path.is_file():
                        continue
                    info = tar.gettarinfo(str(path), arcname=path.relative_to(bundle).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def _stage_artifact(archive_path: Path, store: str | None, authorize_remote: bool) -> str:
    if not store:
        return "No artifact store configured. Stage this exact archive in the author's private collaborator transfer location: " + str(archive_path.resolve())
    if store.startswith("s3://"):
        if not authorize_remote:
            raise ValueError("S3 upload requires --authorize-remote-upload; no remote upload was performed")
        subprocess.run(["aws", "s3", "cp", str(archive_path), store], check=True)
        subprocess.run(["aws", "s3", "cp", str(archive_path) + ".sha256", store.rstrip("/") + "/" + archive_path.name + ".sha256"], check=True)
        return f"Staged to configured S3 destination {store} using the author's AWS credentials. Confirm bucket ACL/policy is private."
    if store.startswith("https://") or store.startswith("http://"):
        raise ValueError("HTTP artifact upload is not implemented; configure a private filesystem path or S3 URI")
    destination = Path(store.removeprefix("file://")).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    copied = destination / archive_path.name
    shutil.copy2(archive_path, copied)
    shutil.copy2(Path(str(archive_path) + ".sha256"), Path(str(copied) + ".sha256"))
    return f"Copied package and SHA-256 file to configured local artifact directory: {destination.resolve()}"


def prepare_release(output_dir: Path | None = None, artifact_store: str | None = None,
                    authorize_remote_upload: bool = False) -> tuple[Path, Path, str]:
    output_dir = (output_dir or ROOT / "ablation_cuda/releases").resolve()
    release_dir = output_dir / DEFAULT_RELEASE_NAME
    if release_dir.exists():
        if (release_dir / "PACKAGE_MANIFEST.json").is_file():
            raise FileExistsError(
                f"Versioned release already exists at {release_dir}; choose a new --output-dir or bump ablation_cuda.__version__"
            )
        failed_dir = output_dir / f".{DEFAULT_RELEASE_NAME}.incomplete-{int(time.time())}"
        release_dir.rename(failed_dir)
    release_dir.mkdir(parents=True)

    dataset_dir, snapshot_dir, snapshot_evidence, records = _validate_author_inputs()
    for source, dest in ((dataset_dir / "questions.jsonl", release_dir / "dataset/questions.jsonl"),
                         (dataset_dir / "questions_manifest.json", release_dir / "dataset/questions_manifest.json"),
                         (dataset_dir / "questions_provenance.csv", release_dir / "dataset/questions_provenance.csv"),
                         (dataset_dir / "SHA256SUMS", release_dir / "dataset/SHA256SUMS")):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    _copy_tree(snapshot_dir, release_dir / "snapshot")
    for metadata_path in (release_dir / "snapshot").rglob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        metadata_path.write_bytes(_json_bytes(_redact_author_paths(metadata)))
    decision = ROOT / "ablation_portability/snapshots/baseline_decision.json"
    shutil.copy2(decision, release_dir / "baseline_decision.json")
    snapshot_json = json.loads((release_dir / "snapshot/snapshot_manifest.json").read_text(encoding="utf-8"))
    snapshot_json["baseline_selection"]["decision_file"] = "../baseline_decision.json"
    e5_report = release_dir / "snapshot/e5_compatibility/embedding_comparison.json"
    if e5_report.is_file():
        snapshot_json["embedding_compatibility"]["report_sha256"] = sha256_file(e5_report)
    (release_dir / "snapshot/snapshot_manifest.json").write_bytes(_json_bytes(snapshot_json))

    historical_config = ROOT / "graphs/run_20260729_211324_OCC-RAG-1.7B/config_snapshot.yaml"
    effective_config, config_differences = _sanitized_effective_config(historical_config)
    (release_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(effective_config, sort_keys=True, allow_unicode=True), encoding="utf-8")
    (release_dir / "configuration_differences.json").write_bytes(_json_bytes(config_differences))

    model_lock_path = ROOT / "ablation_cuda/models/model_assets.lock.yaml"
    model_lock = yaml.safe_load(model_lock_path.read_text(encoding="utf-8"))
    phase1_model_lock = yaml.safe_load(
        (ROOT / "ablation_portability/models/model_assets.lock.yaml").read_text(encoding="utf-8")
    )
    for role, item in model_lock.get("model_assets", {}).items():
        evidence_item = phase1_model_lock.get("model_assets", {}).get(role, {})
        for field in ("model_id", "revision", "architecture", "weights", "dimension",
                      "max_sequence_length", "tokenizer_identity", "chat_template", "preprocessing", "files", "license"):
            if field in item and item[field] != evidence_item.get(field):
                raise ValueError(f"Author-side {role}.{field} no longer matches frozen Phase 1 model evidence")
    sanitized_model_lock = _redact_author_paths(model_lock)
    (release_dir / "models").mkdir(parents=True, exist_ok=True)
    (release_dir / "models/model_assets.lock.yaml").write_text(
        yaml.safe_dump(sanitized_model_lock, sort_keys=False, allow_unicode=True), encoding="utf-8")
    asset_manifest = _asset_manifest(model_lock)
    frozen_asset_manifest_path = ROOT / "ablation_cuda/models/asset_manifest.json"
    frozen_asset_manifest = json.loads(frozen_asset_manifest_path.read_text(encoding="utf-8"))
    if frozen_asset_manifest != asset_manifest:
        raise ValueError("Author-side asset_manifest.json does not match the model lockfile")
    (release_dir / "models/asset_manifest.json").write_bytes(_json_bytes(frozen_asset_manifest))

    # Bundle source rather than requiring a collaborator checkout or a host-side
    # package installation. Only query/runtime code and its direct prompt assets
    # are included; ingestion/OCR source is not copied into the image.
    _copy_tree(ROOT / "back/src", release_dir / "code/back/src")
    portability_root = release_dir / "code/ablation_portability"
    for relative in ("__init__.py", "storage.py", "dataset/__init__.py", "dataset/loader.py",
                     "scripts/validate_snapshot.py", "scripts/restore_db.py",
                     "scripts/rebuild_vector_index.py", "scripts/__init__.py"):
        source = ROOT / "ablation_portability" / relative
        target = portability_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _copy_tree(ROOT / "back/benchmarks/rag/core", release_dir / "code/back/benchmarks/rag/core")
    _copy_tree(ROOT / "back/benchmarks/rag/prompts", release_dir / "code/back/benchmarks/rag/prompts")
    for name in ("run_benchmarks.py", "config_creator.py"):
        source = ROOT / "back/benchmarks/rag" / name
        if source.is_file():
            shutil.copy2(source, release_dir / "code/back/benchmarks/rag" / name)
    _copy_tree(ROOT / "ablation_cuda", release_dir / "code/ablation_cuda",
               exclude_paths={"final_report.md"})
    shutil.copy2(ROOT / "ablation_cuda/Dockerfile", release_dir / "Dockerfile")
    for name in ("requirements.lock", "uv.lock", "runtime-pyproject.toml", "ablation.sh", "ablation.ps1", "verify_results.py"):
        source = ROOT / "ablation_cuda" / name
        if source.is_file():
            shutil.copy2(source, release_dir / name)
    # Historical local smoke evidence is a parity diagnostic only.
    diag = ROOT / "ablation_unblock/smoke_run"
    (release_dir / "diagnostics/mlx_reference").mkdir(parents=True)
    for name in ("prompt.txt", "telemetry.json", "generation.json", "effective_config.json"):
        source = diag / name
        if source.is_file():
            destination = release_dir / "diagnostics/mlx_reference" / name
            if source.suffix == ".json":
                try:
                    metadata = json.loads(source.read_text(encoding="utf-8"))
                    destination.write_bytes(_json_bytes(_redact_author_paths(metadata)))
                    continue
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            shutil.copy2(source, destination)

    revision = _source_revision()
    (release_dir / "SOURCE_REVISION.json").write_bytes(_json_bytes(revision))
    db_identity = {
        "candidate": "external_db",
        "snapshot_id": snapshot_json.get("snapshot_id"),
        "snapshot_sha256": snapshot_json.get("snapshot_identity", {}).get("sha256"),
        "selected_baseline_only": True,
        "july_identity_proven": False,
        "logical_fingerprint": {
            "document_count": snapshot_evidence["logical"].get("document_count"),
            "chunk_count": snapshot_evidence["logical"].get("chunk_count"),
            "vector_count": snapshot_evidence["logical"].get("vector", {}).get("rows"),
            "vector_dimensions": snapshot_evidence["logical"].get("vector", {}).get("dimensions"),
            "graph_node_count": snapshot_evidence["logical"].get("graph_node_count"),
            "graph_edge_count": snapshot_evidence["logical"].get("graph_edge_count"),
            "fts_objects": snapshot_evidence["logical"].get("fts_objects"),
        },
        "index_rebuild": "USEARCH sidecar is reconstructed from the persisted SQLite float32 vectors; no re-embedding",
        "retrieval_fixtures": snapshot_evidence["retrieval_fixtures"],
    }
    (release_dir / "DATABASE_IDENTITY.json").write_bytes(_json_bytes(db_identity))
    (release_dir / "README_COLLABORATOR.md").write_text(
        "# Biomedical RAG CUDA release\n\n"
        "This frozen package contains the exact 375-question manifest and the author-selected `external_db` snapshot only. "
        "The original July database identity remains unproven.\n\n"
        "Run `./ablation.sh doctor`, then `./ablation.sh smoke`. `smoke` builds the pinned Linux image, checks NVIDIA access, "
        "restores SQLite, rebuilds USEARCH from stored vectors, downloads pinned public model revisions, and runs only "
        "`pubmedqa_21084567` under B6 and one trace-selected component-off arm. It prefers `score_blending=false` when the ranking trace proves an active blend over at least two candidates; otherwise it uses `graph_expansion=false` only if B6 inserted a dynamic graph block. It exports the result archive into `return/`.\n\n"
        "Windows users run `./ablation.ps1 doctor` and `./ablation.ps1 smoke` in PowerShell from this directory. "
        "Docker Desktop must use Linux containers and WSL2 GPU passthrough must already work. The scripts do not install or change host drivers.\n\n"
        "Return the generated `return/ablation_cuda_results_v1_<runid>.tar.gz` archive and its matching `.sha256` file. The author can verify/import it with "
        "`./ablation verify-results <archive>`. No cloud judge is called.\n",
        encoding="utf-8",
    )

    files = _payload_files(release_dir)
    manifest = {
        "schema_version": "1.0",
        "package_type": "ablation_cuda_input",
        "release_version": __version__,
        "expected_files": files,
        "dataset": {
            "query_count": len(records), "answerable_count": 300,
            "designated_negative_count": 75,
            "questions_sha256": sha256_file(release_dir / "dataset/questions.jsonl"),
            "manifest_sha256": sha256_file(release_dir / "dataset/questions_manifest.json"),
        },
        "database": db_identity,
        "model_asset_manifest_sha256": sha256_file(release_dir / "models/asset_manifest.json"),
        "effective_config_sha256": hashlib.sha256((release_dir / "effective_config.yaml").read_bytes()).hexdigest(),
        "source_revision": revision,
        "remote_artifact_store_configured": bool(artifact_store),
        "source_historical_july_db_identity_proven": False,
    }
    (release_dir / "PACKAGE_MANIFEST.json").write_bytes(_json_bytes(manifest))
    from ablation_cuda.runtime.smoke import validate_release
    validate_release(release_dir)
    archive_path = output_dir / f"{DEFAULT_RELEASE_NAME}.tar.gz"
    if archive_path.exists() or Path(str(archive_path) + ".sha256").exists():
        raise FileExistsError(f"Versioned release archive already exists at {archive_path}; choose a new --output-dir")
    _write_tar_gz(release_dir, archive_path)
    digest = sha256_file(archive_path)
    Path(str(archive_path) + ".sha256").write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    message = _stage_artifact(archive_path, artifact_store, authorize_remote_upload)
    return release_dir, archive_path, message


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the versioned Biomedical RAG CUDA collaborator bundle")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact-store", default=os.environ.get("ABLA_ARTIFACT_STORE"))
    parser.add_argument("--authorize-remote-upload", action="store_true",
                        help="Explicitly authorize staging to the configured remote S3 destination")
    args = parser.parse_args()
    bundle, archive, staging = prepare_release(args.output_dir, args.artifact_store,
                                               args.authorize_remote_upload)
    print(json.dumps({"bundle_directory": str(bundle), "archive": str(archive),
                      "archive_sha256": sha256_file(archive), "staging": staging},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
