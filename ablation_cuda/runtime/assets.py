from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(path: Path) -> list[dict[str, Any]]:
    result = []
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.name == ".DS_Store" or "__pycache__" in file.parts:
            continue
        result.append({"path": file.relative_to(path).as_posix(),
                       "size_bytes": file.stat().st_size,
                       "sha256": sha256_file(file)})
    return result


def tree_digest(entries: list[dict[str, Any]]) -> str:
    raw = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _verify_model_files(path: Path, model: dict[str, Any]) -> dict[str, Any]:
    missing = []
    mismatched = []
    actual = {}
    for relative, expected in model.get("files", {}).items():
        target = path / relative
        if not target.is_file():
            missing.append(relative)
            continue
        digest = sha256_file(target)
        actual[relative] = digest
        if digest != expected:
            mismatched.append({"path": relative, "expected": expected, "actual": digest})
    if missing or mismatched:
        raise RuntimeError(json.dumps({"model_id": model.get("model_id"),
                                       "missing": missing, "mismatched": mismatched}, indent=2))
    return {"files": actual, "verified_file_count": len(actual)}


def acquire_models(release_root: Path, model_root: Path,
                   progress_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    lock = yaml.safe_load((release_root / "models/model_assets.lock.yaml").read_text(encoding="utf-8"))
    assets = lock["model_assets"]
    aliases = {"generator": "generator", "embedder": "embedder", "cross_encoder": "cross_encoder"}
    report: dict[str, Any] = {"schema_version": "1.0", "models": {}, "spacy_model": {}}
    model_root.mkdir(parents=True, exist_ok=True)
    cache = model_root / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache / "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    api = HfApi()
    for role, key in aliases.items():
        if progress_callback:
            progress_callback(f"acquiring and validating {role} checkpoint")
        item = assets[key]
        repo_id, revision = item["model_id"], item["revision"]
        snapshot_path = None
        try:
            snapshot_path = Path(snapshot_download(
                repo_id=repo_id, revision=revision, cache_dir=str(cache / "hub"),
                local_files_only=True,
            ))
        except Exception:
            info = api.model_info(repo_id, revision=revision)
            if info.sha != revision:
                raise RuntimeError(f"Hub resolved {repo_id}@{revision} to unexpected commit {info.sha}")
            snapshot_path = Path(snapshot_download(
                repo_id=repo_id, revision=revision, cache_dir=str(cache / "hub"),
            ))
        actual_revision = snapshot_path.name
        if actual_revision != revision:
            raise RuntimeError(f"Cached model revision mismatch for {repo_id}: {actual_revision}")
        try:
            verified = _verify_model_files(snapshot_path, item)
        except RuntimeError:
            # Hugging Face downloads are resumable; force refresh only this
            # revision if the persistent cache contains damaged expected files.
            snapshot_path = Path(snapshot_download(
                repo_id=repo_id, revision=revision, cache_dir=str(cache / "hub"),
                force_download=True,
            ))
            if snapshot_path.name != revision:
                raise RuntimeError(f"Hub refreshed {repo_id} at an unexpected commit {snapshot_path.name}")
            verified = _verify_model_files(snapshot_path, item)
        report["models"][role] = {
            "model_id": repo_id, "revision": revision,
            "snapshot_path": str(snapshot_path), "verified": True, **verified,
        }
        if progress_callback:
            progress_callback(f"verified {role} checkpoint files")

    # The historical effective config points at an author-local core_web_md
    # directory. Its metadata identified the public en_core_web_md 3.8.0 model.
    # Fetch it automatically into the persistent volume, with no shell/network
    # URL embedded in the source bundle.
    spacy_root = model_root / "spacy_packages"
    spacy_module = spacy_root / "en_core_web_md"
    expected_spacy = json.loads((release_root / "models/asset_manifest.json").read_text(encoding="utf-8"))["spacy_auxiliary_model"]
    if not spacy_module.is_dir():
        if progress_callback:
            progress_callback("acquiring pinned spaCy query-normalization model")
        env = os.environ.copy()
        env["PIP_TARGET"] = str(spacy_root)
        env["PYTHONPATH"] = str(spacy_root) + os.pathsep + env.get("PYTHONPATH", "")
        spacy_root.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "spacy", "download",
                        f"{expected_spacy['package']}-{expected_spacy['version']}"],
                       check=True, env=env)
    sys.path.insert(0, str(spacy_root))
    import en_core_web_md
    installed_version = en_core_web_md.__version__
    if installed_version != expected_spacy["version"]:
        raise RuntimeError(f"spaCy model version mismatch: expected {expected_spacy['version']}, got {installed_version}")
    report["spacy_model"] = {
        "package": expected_spacy["package"], "version": installed_version,
        "path": str(spacy_module), "files": tree_manifest(spacy_module),
        "tree_sha256": tree_digest(tree_manifest(spacy_module)),
        "verified_version": True,
    }
    if progress_callback:
        progress_callback("verified spaCy query-normalization model")
    return report
