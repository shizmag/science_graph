from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def arm_input_identity(*, query_id: str, question_sha256: str,
                       snapshot_sha256: str, model_manifest_sha256: str,
                       effective_config: dict[str, Any], decoding: dict[str, Any]) -> str:
    return object_sha256({
        "query_id": query_id,
        "question_sha256": question_sha256,
        "snapshot_sha256": snapshot_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "effective_config": effective_config,
        "decoding": decoding,
    })


def stable_result_identity(input_identity: str, prompt: str) -> str:
    return object_sha256({"input_identity": input_identity,
                          "prompt_sha256": sha256_bytes(prompt.encode("utf-8"))})


def score_blending_isolated(full: dict[str, Any], off: dict[str, Any]) -> bool:
    """Return true only when score_blending is the sole config/component change."""
    return component_isolated(full, off, "score_blending")


def component_isolated(full: dict[str, Any], off: dict[str, Any], component: str) -> bool:
    """Return true only when the named component is the sole config/bundle change."""
    full_config = json.loads(json.dumps(full.get("config", {})))
    off_config = json.loads(json.dumps(off.get("config", {})))
    full_components = full_config.get("rag_components", {})
    off_components = off_config.get("rag_components", {})
    full_value = full_components.pop(component, None)
    off_value = off_components.pop(component, None)
    config_isolated = full_value is True and off_value is False and full_config == off_config
    full_baseline = dict(full.get("baseline_components", {}))
    off_baseline = dict(off.get("baseline_components", {}))
    baseline_isolated = (full_baseline.pop(component, None) is True
                         and off_baseline.pop(component, None) is False
                         and full_baseline == off_baseline)
    return full.get("baseline") == off.get("baseline") == "B6" and config_isolated and baseline_isolated
