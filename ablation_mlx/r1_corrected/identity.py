"""Frozen identity and manifest guards for the separately fingerprinted MLX run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from ablation_cuda.runtime.checkpoints import atomic_write_json

from .protocol import EXPERIMENT_ID, PROTOCOL_VERSION, PROMPT_VERSION

HISTORICAL_MODEL_REVISION = "e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c"
HISTORICAL_WEIGHT_SHA256 = "913448d005d2af695a738da177120c9e741dc8a5b2328f717d7ed108c1fd2e4f"
DATABASE_SNAPSHOT_ID = "external_db-20261016T193200Z"
DATABASE_SHA256 = "c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0"
DATASET_SHA256 = "d2820ced7c2994218245887ad753700923262fb91ec983eeacc01b217ec43b6b"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(model_path: Path, *, model_revision: str = HISTORICAL_MODEL_REVISION) -> dict[str, Any]:
    model_path = Path(model_path).expanduser().resolve()
    weight_path = model_path / "model.safetensors"
    config_path = model_path / "config.json"
    generation_path = model_path / "generation_config.json"
    tokenizer_path = model_path / "tokenizer.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    weight_sha256 = file_sha256(weight_path)
    identity = {
        "model_id": "occ-ai/OCC-RAG-1.7B",
        "revision": model_revision,
        "architecture": "Qwen3ForCausalLM",
        "weight_sha256": weight_sha256,
        "config_sha256": file_sha256(config_path),
        "generation_config_sha256": file_sha256(generation_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "tokenizer_config_sha256": file_sha256(tokenizer_config_path),
        "weight_matches_historical": weight_sha256 == HISTORICAL_WEIGHT_SHA256,
        "historical_revision_source": "October CUDA model_assets.lock.yaml; local directory has no revision file",
        "historical_config_sha256": "cbfc589e89ed20d820ab698ca3b56f8d5914710f5b4019cb240eeb882376c9f0",
        "config_matches_historical": False,
    }
    identity["config_matches_historical"] = identity["config_sha256"] == identity["historical_config_sha256"]
    identity["identity_sha256"] = object_sha256(identity)
    return identity


def runtime_identity(*, mlx_version: str | None = None, mlx_lm_version: str | None = None) -> dict[str, Any]:
    try:
        physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        physical_memory = None
    return {
        "runtime": "MLX",
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0] or None,
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "physical_memory_bytes": physical_memory,
        "mlx_version": mlx_version,
        "mlx_lm_version": mlx_lm_version,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
    }


def query_input_identity(
    *,
    manifest_hash: str,
    query_id: str,
    question: str,
    evidence_hash: str,
    arm_id: str,
    prompt_sha256: str,
    token_ids_sha256: str | None = None,
    token_ids: Any | None = None,
    model_identity_hash: str | None = None,
    tokenizer_hash: str | None = None,
    model_identity: Any | None = None,
    tokenizer_identity: Any | None = None,
    decoding: dict[str, Any] | None = None,
) -> str:
    """Hash every identity-bearing input to one query/arm operation.

    The object-valued aliases are accepted so callers can pass the frozen
    identity records directly; their canonical hashes are stored in the
    operation identity rather than an unstable object representation.
    """
    if token_ids_sha256 is None and token_ids is not None:
        token_ids_sha256 = object_sha256([int(value) for value in token_ids])
    if model_identity_hash is None:
        model_identity_hash = object_sha256(model_identity or {})
    if tokenizer_hash is None:
        tokenizer_hash = object_sha256(tokenizer_identity or {})
    return object_sha256({
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "manifest_hash": manifest_hash,
        "query_id": query_id,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "evidence_hash": evidence_hash,
        "arm_id": arm_id,
        "prompt_sha256": prompt_sha256,
        "token_ids_sha256": token_ids_sha256,
        "model_identity_hash": model_identity_hash,
        "tokenizer_hash": tokenizer_hash,
        "decoding": decoding or {},
    })


def write_manifest(path: Path, manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.setdefault("experiment_id", EXPERIMENT_ID)
    payload.setdefault("protocol_version", PROTOCOL_VERSION)
    payload.setdefault("prompt_version", PROMPT_VERSION)
    payload["manifest_hash"] = object_sha256({key: value for key, value in payload.items() if key != "manifest_hash"})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(path), payload)
    if load_manifest(Path(path))["manifest_hash"] != payload["manifest_hash"]:
        raise ValueError("Manifest changed during atomic write")
    return payload["manifest_hash"]


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest.get("manifest_hash"):
        raise ValueError(f"Malformed frozen manifest: {path}")
    expected = object_sha256({key: value for key, value in manifest.items() if key != "manifest_hash"})
    if expected != manifest["manifest_hash"]:
        raise ValueError(f"Frozen manifest hash mismatch: {path}")
    return manifest


def require_manifest(path: Path, expected_hash: str) -> dict[str, Any]:
    manifest = load_manifest(path)
    if manifest["manifest_hash"] != expected_hash:
        raise ValueError(f"Experiment manifest mismatch: expected {expected_hash}, got {manifest['manifest_hash']}")
    return manifest
