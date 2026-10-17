"""Resumable full B6 CUDA ablation matrix orchestrator."""

from __future__ import annotations

import copy
import datetime as dt
import gc
import hashlib
import json
import os
import random
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

from ablation_cuda.runtime.analysis import finalize_outputs
from ablation_cuda.runtime.checkpoints import CheckpointStore, atomic_write_json, hydrate_result, stable_input_hash
from ablation_cuda.runtime.matrix import ARM_DEFINITIONS, MATRIX_VERSION, manifest_yaml, resolve_matrix
from ablation_cuda.runtime.progress import Progress
from ablation_cuda.runtime.smoke import (
    DB_ROOT, LOG_ROOT, MODEL_ROOT, OUTPUT_ROOT, RELEASE_ROOT,
    TELEMETRY_IMPLEMENTATION_VERSION, _actual_cuda_probe, _instrument_service,
    _logical_signature, _prepare_database, _run_arm, _set_spacy_pythonpath,
    _version_manifest, _write_private_config, validate_release,
)
from ablation_cuda.runtime.stage_cache import StageCache
from ablation_cuda.identity import arm_input_identity, canonical_json, sha256_bytes
from ablation_cuda.runtime.assets import sha256_file
from ablation_portability.dataset.loader import load_frozen_dataset
from ablation_cuda.runtime.packaging import pack_experiment


GRAPH_OUTPUT_ROOT = OUTPUT_ROOT / "graphs"
SEED = 1701
MAX_ATTEMPTS = 3


class LazyResults:
    """Read one operation record at a time so 5,250 generations stay off-RAM."""

    def __init__(self, run_dir: Path, arm_ids: list[str], questions: list[dict[str, Any]]):
        self.run_dir = run_dir
        self.arm_ids = arm_ids
        self.query_ids = [item["query_id"] for item in questions]
        self.cache: OrderedDict[tuple[str, str], dict[str, Any] | None] = OrderedDict()
        self.cache_limit = 2

    def get(self, key: tuple[str, str], default=None):
        if key in self.cache:
            value = self.cache[key]
            self.cache.move_to_end(key)
            return value if value is not None else default
        arm_id, query_id = key
        path = self.run_dir / "baselines" / arm_id / "results" / f"{query_id}.json"
        if not path.is_file():
            self.cache[key] = None
            return default
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            value = None
        if isinstance(value, dict):
            try:
                value = hydrate_result(value, path, self.run_dir)
            except ValueError:
                value = None
        self.cache[key] = value
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return value if value is not None else default

    def iter_arm(self, arm_id: str):
        for query_id in self.query_ids:
            yield self.get((arm_id, query_id))

    def items(self):
        for arm_id in self.arm_ids:
            for query_id in self.query_ids:
                yield (arm_id, query_id), self.get((arm_id, query_id))


def _utc_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(path)


def _code_identity(release_info: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json({
        item["path"]: item["sha256"]
        for item in release_info["manifest"]["expected_files"]
        if item["path"].startswith("code/")
    }))


def _container_identity() -> dict[str, Any]:
    return {
        "container_image_id": os.environ.get("ABLA_IMAGE_ID", "unrecorded"),
        "container_oci_digest": os.environ.get("ABLA_IMAGE_DIGEST", "unrecorded"),
        "base_image": "python:3.12.13-slim-bookworm",
        "base_image_manifest_digest": os.environ.get("ABLA_BASE_IMAGE_DIGEST", "unrecorded"),
    }


def _require_real_rtx5090_cuda_probe(probe: dict[str, Any], *, expected_device: str | None = None) -> None:
    operation_device = probe.get("operation_device")
    result_sum = probe.get("result_sum")
    name = str(probe.get("device", ""))
    capability = probe.get("compute_capability")
    if (not isinstance(operation_device, str) or not operation_device.startswith("cuda")
            or not isinstance(result_sum, (int, float)) or not result_sum):
        raise RuntimeError("CUDA acceptance requires a completed tensor operation on CUDA with a recorded result")
    if "5090" not in name.lower() or capability != [12, 0]:
        raise RuntimeError(
            "The full-run hardware acceptance gate requires an NVIDIA RTX 5090 "
            f"(Blackwell, compute capability 12.0); detected {name!r} with compute capability {capability}. "
            "The smoke diagnostics were saved, and the 375-query workload was not started."
        )
    if expected_device is not None and name != expected_device:
        raise RuntimeError("GPU identity changed after the mandatory smoke gate; refusing to start the matrix")


def _experiment_identity(release_info: dict[str, Any], model_manifest: dict[str, Any],
                         image_identity: dict[str, Any], code_sha256: str,
                         device: dict[str, Any]) -> str:
    return stable_input_hash({
        "input_package_manifest_sha256": sha256_file(RELEASE_ROOT / "PACKAGE_MANIFEST.json"),
        "dataset_sha256": release_info["manifest"]["dataset"]["questions_sha256"],
        "snapshot_sha256": release_info["manifest"]["database"]["snapshot_sha256"],
        "model_manifest_sha256": sha256_bytes(canonical_json(model_manifest)),
        "source_code_sha256": code_sha256,
        "image_identity": image_identity,
        "gpu_name": device.get("device"),
        "gpu_compute_capability": device.get("compute_capability"),
        "matrix_version": MATRIX_VERSION,
        "seed": SEED,
    })


def _find_compatible_run(experiment_identity: str) -> tuple[Path | None, dict[str, Any] | None]:
    if not GRAPH_OUTPUT_ROOT.is_dir():
        return None, None
    candidates = []
    for manifest_path in GRAPH_OUTPUT_ROOT.glob("run_*_OCC-RAG-1.7B_CUDA_ABLATION/run_manifest.json"):
        try:
            manifest = _read_json(manifest_path)
        except Exception:
            continue
        if manifest.get("compatibility_identity") == experiment_identity:
            candidates.append((manifest_path.parent, manifest))
    candidates.sort(key=lambda pair: pair[0].name)
    incomplete = [(path, manifest) for path, manifest in candidates
                  if manifest.get("status") in {"RUNNING", "PARTIAL", "FAILED"}]
    if incomplete:
        return incomplete[-1]
    complete = [(path, manifest) for path, manifest in candidates
                if manifest.get("status") == "COMPLETE"]
    if complete:
        return complete[-1]
    return None, None


def _safe_retryable(error: str) -> bool:
    lowered = error.lower()
    return any(marker in lowered for marker in (
        "out of memory", "temporarily unavailable", "resource temporarily unavailable",
        "connection reset", "connection aborted", "timed out", "timeout",
        "cuda error: unspecified launch failure", "device-side assert triggered",
    ))


def _operation_identity(query: dict[str, Any], runtime_config: dict[str, Any],
                        model_manifest: dict[str, Any], source_identity: dict[str, Any],
                        image_identity: dict[str, Any], code_sha256: str) -> str:
    from core.config import get_baseline_config
    components = get_baseline_config("B6", runtime_config["rag_components"])
    effective = {"config": runtime_config, "baseline": "B6", "paper_baseline": "B5",
                 "baseline_components": components, "intervention": None}
    decoding = {
        "strategy": "greedy_argmax", "temperature": 0.0,
        "max_new_tokens": 768, "context_limit": 12000,
        "stop_token_ids": [151643, 151645, 151683],
        "tokenizer_revision": model_manifest["models"]["generator"]["revision"],
    }
    base = arm_input_identity(
        query_id=query["query_id"], question_sha256=sha256_bytes(query["question"].encode("utf-8")),
        snapshot_sha256=source_identity["snapshot_sha256"],
        model_manifest_sha256=sha256_bytes(canonical_json(model_manifest)),
        effective_config=effective, decoding=decoding,
    )
    return sha256_bytes(canonical_json({
        "input_id": base,
        "question_row_sha256": sha256_bytes(canonical_json(query)),
        "questions_manifest_sha256": source_identity["questions_manifest_sha256"],
        "dataset_sha256": source_identity["dataset_sha256"],
        "code_sha256": code_sha256, "image": image_identity,
        "telemetry_implementation": TELEMETRY_IMPLEMENTATION_VERSION,
    }))


def _operation_result_path(run_dir: Path, arm_id: str, query_id: str) -> Path:
    return run_dir / "baselines" / arm_id / "results" / f"{query_id}.json"


def _failure_result(*, arm_id: str, query: dict[str, Any], input_identity: str,
                    effective_configuration_hash: str, error: str, attempt_count: int,
                    generation_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "2.0", "arm": arm_id, "query_id": query["query_id"],
        "question": query["question"], "gold_reference_answer": query.get("gold_reference_answer", ""),
        "gold_paper_ids": query.get("original_paper_ids", []),
        "source_pmids": query.get("original_pmids", []),
        "answerability_label": query.get("answerability_label"),
        "synthetic_negative": query.get("answerability_label") == "designated_negative",
        "status": "FAILED", "raw_generated_response": "", "parsed_answer": "",
        "parser_status": None, "prompt": "", "prompt_sha256": None,
        "selected_evidence": [], "retrieved_paper_ids": [],
        "generation_events": generation_events or [], "token_telemetry": [],
        "input_identity": input_identity,
        "effective_configuration_hash": effective_configuration_hash,
        "result_errors": [error], "query_arm_attempt_count": attempt_count,
        "timing": {"elapsed_seconds": 0.0, "pipeline_component_metrics": {}},
        "stage_status": {"operation": "failed"},
        "retrieval_candidates_and_ranking_trace": {"trace_counters": {}, "stage_cache_events": []},
    }


def _preserve_result_and_telemetry(source: Path, target: Path, run_dir: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    sidecar = source.with_suffix(".telemetry.jsonl.gz")
    if not sidecar.is_file():
        return
    target_sidecar = target.with_suffix(".telemetry.jsonl.gz")
    shutil.copy2(sidecar, target_sidecar)
    try:
        value = _read_json(target)
        reference = value.get("telemetry_sidecar")
        if reference:
            reference["path"] = target_sidecar.relative_to(run_dir).as_posix()
            atomic_write_json(target, value)
    except Exception:
        pass


def _previous_errors(path: Path) -> list[str]:
    if not path.is_file():
        return []
    errors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            errors.append(str(value.get("error", "")))
        except json.JSONDecodeError:
            errors.append(line)
    return errors


def _cuda_settings(seed: int = SEED) -> dict[str, Any]:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not-set-before-process-start"),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
    }


def _manifest_template(*, run_id: str, experiment_identity: str,
                       compatibility_identity: str, status: str,
                       release_info: dict[str, Any], db_identity: dict[str, Any],
                       source_identity: dict[str, Any], model_manifest: dict[str, Any],
                       device_preflight: dict[str, Any], version_manifest: dict[str, Any],
                       image_identity: dict[str, Any], code_sha256: str,
                       arms: list[dict[str, Any]], matrix: dict[str, Any],
                       smoke_gate: dict[str, Any], run_dir: Path,
                       deterministic: dict[str, Any]) -> dict[str, Any]:
    dataset = release_info["manifest"]["dataset"]
    return {
        "schema_version": "2.0", "experiment_type": "full_cuda_ablation",
        "status": status, "run_id": run_id,
        "experiment_identity": experiment_identity,
        "compatibility_identity": compatibility_identity,
        "internal_baseline": "B6", "paper_baseline": "B5",
        "input_package_manifest_sha256": sha256_file(RELEASE_ROOT / "PACKAGE_MANIFEST.json"),
        "dataset": {"row_count": 375, "answerable_count": 300,
                    "designated_negative_count": 75,
                    "sha256": dataset["questions_sha256"],
                    "questions_manifest_sha256": dataset["manifest_sha256"]},
        "database": {
            "candidate": "external_db", "snapshot_id": source_identity["snapshot_id"],
            "snapshot_sha256": source_identity["snapshot_sha256"],
            "logical_fingerprint": source_identity["logical_fingerprint"],
            "selected_source_july_identity_proven": False,
            "author_original_database_accessed": False,
            "identity_same_across_all_arms": True,
            "snapshot_file_restored_byte_exactly": db_identity.get("snapshot_file_restored_byte_exactly"),
            "usearch_rebuilt_from_persisted_vectors": db_identity.get("usearch_rebuilt_from_persisted_vectors"),
            "persisted_vectors": db_identity.get("vector_validation", {}).get("rows"),
            "database_after_run_sha256": None,
        },
        "models": model_manifest,
        "model_revisions": {role: item.get("revision") for role, item in model_manifest.get("models", {}).items()},
        "device_preflight": device_preflight,
        "software_versions": version_manifest,
        "image_identity": image_identity,
        "source_code_sha256": code_sha256,
        "decoding": {"strategy": "greedy_argmax", "temperature": 0.0,
                     "max_new_tokens": 768, "context_limit": 12000,
                     "stop_token_ids": [151643, 151645, 151683],
                     "tokenizer_revision": model_manifest.get("models", {}).get("generator", {}).get("revision")},
        "telemetry_implementation_version": TELEMETRY_IMPLEMENTATION_VERSION,
        "actual_context_limit_tokens": 12000,
        "actual_output_token_limit": 768,
        "backend_differences": {
            "generator": "OCC-RAG-1.7B BF16 with Transformers/PyTorch CUDA SDPA; greedy token loop; no quantization",
            "embedding": "pinned E5 base model executed with SentenceTransformers on CUDA",
            "cross_encoder": "pinned multilingual Cross-Encoder executed with SentenceTransformers on CUDA",
            "database": "selected external_db frozen snapshot; July database identity remains unproven",
        },
        "deterministic_settings": deterministic,
        "memory_policy": {"runtime_batch_size_policy": "locked model semantics; repository and Cross-Encoder defaults; no dynamic precision or model-weight changes",
                          "automatic_quantization": False, "precision": "bfloat16 generator weights"},
        "matrix": {"version": matrix["matrix_version"],
                   "arm_count": len(arms), "query_count": 375,
                   "planned_query_arm_operations": 375 * len(arms),
                   "completed_query_arm_operations": 0, "failed_query_arm_operations": 0,
                   "missing_query_arm_operations": 375 * len(arms)},
        "arms": [{"arm_id": arm["arm_id"], "status": arm.get("status", "PLANNED"),
                  "completion": arm.get("completion"),
                  "intervention_integrity": arm.get("intervention_integrity")} for arm in arms],
        "smoke_gate": smoke_gate,
        "judge": {"status": "PENDING_INDEPENDENT_AUTHOR_EVALUATION", "remote_requests_made": False},
        "paths": {"run_directory": str(run_dir), "archive": None},
        "errors": [],
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": 0.0,
    }


def _write_manifests(run_dir: Path, manifest: dict[str, Any], arms_manifest: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    _atomic_yaml(run_dir / "run_manifest.yaml", manifest)
    atomic_write_json(run_dir / "arms_manifest.json", arms_manifest)
    _atomic_yaml(run_dir / "arms_manifest.yaml", arms_manifest)


def _write_static_run_inputs(run_dir: Path, release_info: dict[str, Any],
                             config_snapshot: dict[str, Any], matrix: dict[str, Any],
                             arm_configs: dict[str, dict[str, Any]],
                             arms: list[dict[str, Any]]) -> None:
    _atomic_yaml(run_dir / "config_snapshot.yaml", config_snapshot)
    (run_dir / "questions_manifest.json").write_bytes((RELEASE_ROOT / "dataset/questions_manifest.json").read_bytes())
    shutil.copy2(RELEASE_ROOT / "dataset/questions.jsonl", run_dir / "questions.jsonl")
    metadata = run_dir / "metadata"
    metadata.mkdir(exist_ok=True)
    shutil.copy2(RELEASE_ROOT / "DATABASE_IDENTITY.json", metadata / "database_identity.json")
    shutil.copy2(RELEASE_ROOT / "models/model_assets.lock.yaml", metadata / "model_assets.lock.yaml")
    shutil.copy2(RELEASE_ROOT / "configuration_differences.json", metadata / "backend_configuration_differences.json")
    for arm in arms:
        arm["effective_config"] = copy.deepcopy(arm_configs[arm["arm_id"]])
        arm["definition"] = next(item for item in ARM_DEFINITIONS if item.arm_id == arm["arm_id"]).__dict__
        arm_dir = run_dir / "baselines" / arm["arm_id"]
        (arm_dir / "results").mkdir(parents=True, exist_ok=True)


def _copy_smoke_evidence(run_dir: Path, smoke_gate: dict[str, Any]) -> None:
    source = Path(smoke_gate.get("run_directory", ""))
    if not source.is_dir():
        atomic_write_json(run_dir / "metadata/cuda_smoke_evidence.json", smoke_gate)
        return
    destination = run_dir / "metadata/cuda_smoke_evidence"
    destination.mkdir(parents=True, exist_ok=True)
    for relative in ("run_manifest.json", "model_index_manifest.json", "stage_status.json"):
        source_path = source / relative
        if source_path.is_file():
            shutil.copy2(source_path, destination / relative)
    for relative in ("arms/B6.json", "arms/no_score_blending.json", "arms/no_graph_expansion.json"):
        source_path = source / relative
        if source_path.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
    atomic_write_json(destination / "smoke_gate.json", smoke_gate)


def _model_identity_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("Pinned model assets are missing; run `./ablation.sh install` or `run` again")
    return _read_json(path)


def run_full_matrix(*, smoke_result: dict[str, Any], smoke_archive: Path,
                    smoke_archive_sha256: str, new_run: bool = False,
                    requested_run_id: str | None = None,
                    resume_only: bool = False) -> dict[str, Any]:
    """Execute the frozen full matrix, checkpointing every query-arm operation."""
    if smoke_result.get("run_manifest", {}).get("status") != "SUCCESS":
        raise RuntimeError("The mandatory real CUDA smoke did not pass; the 375-query matrix is not permitted to start")
    if smoke_result.get("archive_verification", {}).get("status") != "PASSED":
        raise RuntimeError("The mandatory CUDA smoke archive did not pass portable result verification")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    release_info = validate_release()
    records = load_frozen_dataset(RELEASE_ROOT / "dataset")
    if len(records) != 375:
        raise RuntimeError(f"Frozen dataset changed from 375 queries: found {len(records)}")
    device_operation = smoke_result["run_manifest"].get("device_preflight", {}).get("torch_cuda_operation", {})
    _require_real_rtx5090_cuda_probe(device_operation)
    cuda_probe = _actual_cuda_probe()
    _require_real_rtx5090_cuda_probe(cuda_probe, expected_device=device_operation.get("device"))
    os.environ["ABLA_CUDA_REQUIRE_DEVICE"] = "1"
    os.environ.pop("RAG_GRAPH_RETRIEVAL", None)
    os.environ.pop("SCIENCE_GRAPH_USE_CLOUD", None)

    model_manifest = _model_identity_manifest(MODEL_ROOT / "installed_models.json")
    db_path, db_identity = _prepare_database(release_info)
    runtime_config = _write_private_config(
        RELEASE_ROOT / "effective_config.yaml", db_path, model_manifest,
        Path(os.environ.get("HOME", "/state/home")),
    )
    _set_spacy_pythonpath(MODEL_ROOT)
    source_identity = {
        "candidate": "external_db", "snapshot_id": release_info["manifest"]["database"]["snapshot_id"],
        "snapshot_sha256": release_info["manifest"]["database"]["snapshot_sha256"],
        "dataset_sha256": release_info["manifest"]["dataset"]["questions_sha256"],
        "questions_manifest_sha256": release_info["manifest"]["dataset"]["manifest_sha256"],
        "logical_fingerprint": db_identity["logical_fingerprint"],
        "restored_database_sha256_at_start": db_identity["restored_database_sha256"],
        "selected_source_july_identity_proven": False,
    }
    code_sha256 = _code_identity(release_info)
    image_identity = _container_identity()
    model_manifest_sha256 = sha256_bytes(canonical_json(model_manifest))
    compatibility_identity = _experiment_identity(
        release_info, model_manifest, image_identity, code_sha256, cuda_probe,
    )
    existing_path, existing_manifest = (None, None) if new_run else _find_compatible_run(compatibility_identity)
    if requested_run_id:
        if Path(requested_run_id).name != requested_run_id:
            raise ValueError("--run-id must be a run directory name, without path separators")
        selected = GRAPH_OUTPUT_ROOT / requested_run_id
        manifest_path = selected / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Requested run ID does not exist: {requested_run_id}")
        existing_path, existing_manifest = selected, _read_json(manifest_path)
        if existing_manifest.get("compatibility_identity") != compatibility_identity:
            raise RuntimeError("Requested run is incompatible with this input package, database, models, source, or GPU")
    if (existing_manifest and existing_manifest.get("status") == "COMPLETE"
            and not new_run and not requested_run_id and not resume_only):
        raise RuntimeError(f"A compatible completed run already exists at {existing_path}. Use --new-run to create a distinct experiment identity.")
    if existing_manifest and existing_manifest.get("status") == "SCIENTIFIC_VALIDATION_FAILED" and not requested_run_id:
        raise RuntimeError(f"The compatible run failed scientific validation at {existing_path}; inspect it and use --new-run for a separately identified rerun.")
    if resume_only and existing_path is None:
        raise RuntimeError("No compatible prior run exists to resume. Use `ablation.sh run` to create the first run.")
    if (existing_manifest and existing_manifest.get("status") == "COMPLETE"
            and (requested_run_id or resume_only)):
        archive, archive_sha256 = pack_experiment(existing_path, OUTPUT_ROOT.parent / "return")
        return {"run_directory": str(existing_path), "run_id": existing_manifest.get("run_id"),
                "status": "COMPLETE", "manifest": existing_manifest,
                "archive": str(archive), "archive_sha256": archive_sha256,
                "planned_query_arm_operations": existing_manifest.get("matrix", {}).get("planned_query_arm_operations"),
                "successful_query_arm_operations": existing_manifest.get("matrix", {}).get("completed_query_arm_operations"),
                "failed_query_arm_operations": existing_manifest.get("matrix", {}).get("failed_query_arm_operations"),
                "missing_query_arm_operations": existing_manifest.get("matrix", {}).get("missing_query_arm_operations"),
                "interrupted": False, "already_complete": True}

    if existing_path is not None:
        run_dir = existing_path
        run_id = run_dir.name.removeprefix("run_").removesuffix("_OCC-RAG-1.7B_CUDA_ABLATION")
        experiment_identity = existing_manifest.get("experiment_identity", compatibility_identity)
    else:
        run_id = _utc_run_id()
        run_dir = GRAPH_OUTPUT_ROOT / f"run_{run_id}_OCC-RAG-1.7B_CUDA_ABLATION"
        run_dir.mkdir(parents=True, exist_ok=False)
        experiment_identity = stable_input_hash({"compatibility_identity": compatibility_identity,
                                                 "run_id": run_id})
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir_string = str(run_dir)

    matrix_arms, arm_configs = resolve_matrix(
        runtime_config,
        snapshot_sha256=source_identity["snapshot_sha256"],
        dataset_sha256=release_info["manifest"]["dataset"]["questions_sha256"],
        model_manifest_sha256=model_manifest_sha256,
    )
    for arm in matrix_arms:
        arm["effective_config"] = arm_configs[arm["arm_id"]]
        arm["definition"] = next(item for item in ARM_DEFINITIONS if item.arm_id == arm["arm_id"]).__dict__
    matrix = manifest_yaml(
        matrix_arms,
        dataset_sha256=release_info["manifest"]["dataset"]["questions_sha256"],
        snapshot_sha256=source_identity["snapshot_sha256"],
        model_manifest_sha256=model_manifest_sha256,
    )
    planned_operations = matrix["expected_query_arm_operations"]
    _write_static_run_inputs(run_dir, release_info, runtime_config, matrix, arm_configs, matrix_arms)
    _atomic_yaml(run_dir / "arms_manifest.yaml", matrix)
    atomic_write_json(run_dir / "metadata/cuda_hardware_acceptance.json", {
        "smoke_archive": str(smoke_archive), "smoke_archive_sha256": smoke_archive_sha256,
        "smoke_run_id": smoke_result["run_manifest"].get("run_id"),
        "smoke_status": smoke_result["run_manifest"].get("status"),
        "gpu": cuda_probe,
        "requirement": "real RTX 5090, pinned E5 and Cross-Encoder inference, OCC-RAG generation, CUDA token telemetry, frozen retrieval/context, and a trace-validated off path",
    })
    smoke_gate = {"status": "PASSED", "run_id": smoke_result["run_manifest"].get("run_id"),
                  "archive_sha256": smoke_archive_sha256,
                  "archive_verification": smoke_result["archive_verification"],
                  "run_directory": smoke_result.get("run_directory"),
                  "gpu_name": cuda_probe["device"], "operation_device": cuda_probe["operation_device"]}
    _copy_smoke_evidence(run_dir, smoke_gate)

    deterministic = _cuda_settings()
    version_manifest = _version_manifest()
    # Initialize all model stages on the selected CUDA device before measured
    # operations; this delay is surfaced as a separate progress stage.
    progress = Progress(run_dir / "logs/events.jsonl", total_arms=len(matrix_arms),
                        run_id=run_id, total_queries=len(records),
                        total_operations=planned_operations, output_dir=output_dir_string)
    prior_progress_path = run_dir / "logs/progress.json"
    if prior_progress_path.is_file():
        try:
            prior_progress = _read_json(prior_progress_path)
            progress.retried_operations = int(prior_progress.get("retried_operations", 0))
            progress.cache_hits = int(prior_progress.get("cache_hits", 0))
            progress.reused_generations = int(prior_progress.get("reused_generations", 0))
            progress._completion_samples = list(prior_progress.get("recent_query_arm_seconds", []))[-100:]
        except Exception:
            pass
    progress.start()
    start = time.monotonic()
    started_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    prior_started_iso = existing_manifest.get("started_utc") if existing_manifest else None
    elapsed_before_resume = float(existing_manifest.get("elapsed_seconds", 0.0)) if existing_manifest else 0.0
    manifest = _manifest_template(
        run_id=run_id, experiment_identity=experiment_identity,
        compatibility_identity=compatibility_identity, status="RUNNING",
        release_info=release_info, db_identity=db_identity,
        source_identity=source_identity, model_manifest=model_manifest,
        device_preflight={"torch_cuda_operation": cuda_probe},
        version_manifest=version_manifest, image_identity=image_identity,
        code_sha256=code_sha256, arms=matrix_arms, matrix=matrix,
        smoke_gate=smoke_gate, run_dir=run_dir, deterministic=deterministic,
    )
    manifest["started_utc"] = prior_started_iso or started_iso
    if existing_manifest:
        manifest["resume_history"] = list(existing_manifest.get("resume_history", [])) + [
            {"resumed_utc": started_iso, "prior_status": existing_manifest.get("status")}
        ]
    manifest["dataset"]["query_ids_sha256"] = sha256_bytes(
        canonical_json([item["query_id"] for item in records])
    )
    manifest["matrix"]["arm_manifest_sha256"] = sha256_file(run_dir / "arms_manifest.yaml")
    manifest["run_configuration_hashes"] = {
        "runtime_effective_config_sha256": sha256_bytes(canonical_json(runtime_config)),
        "model_manifest_sha256": model_manifest_sha256,
        "questions_manifest_sha256": release_info["manifest"]["dataset"]["manifest_sha256"],
    }
    _write_manifests(run_dir, manifest, matrix)

    checkpoint_store = CheckpointStore(run_dir / "checkpoints")
    initial_counters = checkpoint_store.counters()
    progress.completed_operations = initial_counters["successful_operations"]
    progress.failed_operations = initial_counters["failed_operations"]
    progress.tokens = initial_counters["new_generated_tokens"]
    progress.reused_generations = max(progress.reused_generations, initial_counters["reused_generations"])
    progress.cache_hits = max(progress.cache_hits, initial_counters["cache_hits"])
    progress.retried_operations = max(progress.retried_operations, initial_counters["retried_operations"])
    results = LazyResults(run_dir, [arm["arm_id"] for arm in matrix_arms], records)
    _update_run_counts(manifest, checkpoint_store, results, records, matrix_arms)
    _write_manifests(run_dir, manifest, matrix)
    stage_cache = StageCache(checkpoint_store, model_manifest=model_manifest,
                             snapshot_sha256=source_identity["snapshot_sha256"],
                             telemetry_version=f"{TELEMETRY_IMPLEMENTATION_VERSION}:{code_sha256}")
    service = None
    final_status = "PARTIAL"
    errors = _previous_errors(run_dir / "logs/errors.jsonl")
    operation_attempts: dict[tuple[str, str], int] = {}
    try:
        progress.update("CUDA model initialization", completed_operations=checkpoint_store.counters()["successful_operations"],
                        failed_operations=checkpoint_store.counters()["failed_operations"],
                        arm_index=0, current_query=0,
                        details={"initialization_is_excluded_from_query_throughput_eta": True})
        from src.config import config
        from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
        from src.vector_search import EmbeddingEngine
        from src.llm_engine import LLMEngine
        from src.services.rag_service import RAGService

        config.data = copy.deepcopy(runtime_config)
        if Path(config.db_path).resolve() != db_path.resolve():
            raise RuntimeError("Resolved retrieval database path conflicts with the frozen restored snapshot")
        graph_repo = SQLiteGraphRepository(str(db_path), read_only=True)
        vector_repo = SQLiteVectorRepository(str(db_path), read_only=True)
        embedder = EmbeddingEngine()
        generator = LLMEngine(use_cloud=False, purpose="rag")
        service = RAGService(graph_repo, vector_repo, embedder, generator,
                             warmup=False, trace_dir=run_dir / "logs/traces")
        probe_trace: dict[str, Any] = {"torch_cuda_operation": cuda_probe}
        _instrument_service(service, probe_trace, records[0]["question"])
        service.current_trace = None
        stage_cache.install(service)
        model_load_seconds = time.monotonic() - start
        manifest["model_initialization_seconds"] = round(model_load_seconds, 3)
        manifest["device_preflight"] = probe_trace
        _write_manifests(run_dir, manifest, matrix)
        progress.update("frozen dataset and restored database verified",
                        details={"questions": len(records), "snapshot_sha256": source_identity["snapshot_sha256"],
                                 "persisted_vectors": db_identity.get("vector_validation", {}).get("rows")})
        progress.update("full ablation execution", details={"planned_query_arm_operations": planned_operations})

        for query_index, query in enumerate(records, start=1):
            for arm_index, arm in enumerate(matrix_arms, start=1):
                arm_id = arm["arm_id"]
                input_identity = _operation_identity(
                    query, arm_configs[arm_id], model_manifest, source_identity,
                    image_identity, code_sha256,
                )
                result_path = _operation_result_path(run_dir, arm_id, query["query_id"])
                counters_before = checkpoint_store.counters()
                completed_count = counters_before["successful_operations"]
                failed_count = counters_before["failed_operations"]
                progress.begin_operation(arm_id=arm_id, arm_index=arm_index, query_index=query_index)
                cached_embedding = stage_cache.get_query_embedding(query["question"])
                if cached_embedding is not None:
                    service.emb_engine._query_cache[query["question"]] = cached_embedding
                    service._ablation_cache_events.append({"stage": "query_embedding", "hit": True, "items": 1})
                else:
                    service._ablation_cache_events.append({"stage": "query_embedding", "hit": False, "items": 1})
                saved = checkpoint_store.get(arm_id, query["query_id"], input_identity)
                if saved and _run_arm_checkpoint_valid(saved, arm_id, input_identity):
                    stage_cache.seed_generation_events(saved.get("generation_events", []))
                    progress.cache_hit(details={"arm_id": arm_id, "query_id": query["query_id"],
                                                "kind": "verified_completed_query_arm"})
                    progress.completed_operations = checkpoint_store.counters()["successful_operations"]
                    progress.tokens = checkpoint_store.counters()["new_generated_tokens"]
                    saved_duration = saved.get("timing", {}).get("elapsed_seconds")
                    if isinstance(saved_duration, (int, float)):
                        progress.observe_operation_duration(float(saved_duration))
                    continue
                if result_path.is_file():
                    superseded = run_dir / "checkpoints/superseded"
                    superseded.mkdir(parents=True, exist_ok=True)
                    target = superseded / f"{arm_id}__{query['query_id']}__{time.time_ns()}.json"
                    _preserve_result_and_telemetry(result_path, target, run_dir)
                result = None
                attempt_errors: list[str] = []
                operation_started = time.monotonic()
                total_attempt_seconds = 0.0
                for local_attempt in range(1, MAX_ATTEMPTS + 1):
                    attempt = checkpoint_store.next_attempt(arm_id, query["query_id"])
                    attempt_started = time.monotonic()
                    event_start = len(service.llm_engine.events)
                    try:
                        result = _run_arm(
                            arm_name=arm_id, component_override=None, query_row=query,
                            runtime_config=arm_configs[arm_id], service=service,
                            run_dir=run_dir, progress=progress,
                            completed=completed_count, model_manifest=model_manifest,
                            db_identity=db_identity, source_identity=source_identity,
                            image_identity=image_identity, code_sha256=code_sha256,
                            checkpoint_file=result_path,
                            preserve_query_embedding_cache=True,
                            require_query_embedding_inference=False,
                            require_cross_encoder_inference=False,
                            write_auxiliary_files=False,
                            input_identity_override=input_identity,
                        )
                    except Exception as exc:
                        error_text = f"{type(exc).__name__}: {exc}"
                        partial_events = []
                        for event in service.llm_engine.events[event_start:]:
                            partial_events.append({"purpose": "failed_operation_partial_generation",
                                                   "event": {key: value for key, value in event.items()
                                                             if key != "token_telemetry"},
                                                   "token_telemetry": event.get("token_telemetry", [])})
                        result = _failure_result(
                            arm_id=arm_id, query=query, input_identity=input_identity,
                            effective_configuration_hash=arm["effective_configuration_hash"],
                            error=error_text, attempt_count=attempt, generation_events=partial_events,
                        )
                    attempt_seconds = time.monotonic() - attempt_started
                    total_attempt_seconds += attempt_seconds
                    if result.get("status") != "SUCCESS":
                        result.setdefault("timing", {})["elapsed_seconds"] = total_attempt_seconds
                    operation_attempts[(arm_id, query["query_id"])] = attempt
                    result["effective_configuration_hash"] = arm["effective_configuration_hash"]
                    result["query_arm_attempt_count"] = attempt
                    fresh_tokens = sum(
                        len(event.get("token_telemetry", []))
                        for event in result.get("generation_events", [])
                        if not event.get("event", {}).get("cache_hit")
                    )
                    reused_tokens = sum(
                        len(event.get("token_telemetry", []))
                        for event in result.get("generation_events", [])
                        if event.get("event", {}).get("cache_hit")
                    )
                    result["cumulative_new_cuda_tokens"] = fresh_tokens
                    result["reused_generation_tokens"] = reused_tokens
                    checkpoint_store.put(arm_id, query["query_id"], input_identity,
                                         result_path, result, attempt_count=attempt)
                    if result.get("status") == "SUCCESS":
                        query_embedding = service.emb_engine._query_cache.get(query["question"])
                        if query_embedding is None:
                            query_embedding = service.emb_engine._query_cache.get("query: " + query["question"])
                        if query_embedding is not None:
                            stage_cache.put_query_embedding(query["question"], query_embedding)
                        break
                    result_error = "; ".join(result.get("result_errors", [])) or "query-arm validation returned PARTIAL"
                    attempt_errors.append(result_error)
                    transient = _safe_retryable(result_error)
                    checkpoint_store.record_attempt(arm_id, query["query_id"], error=result_error,
                                                    attempt=attempt, transient=transient)
                    progress.error(result_error, arm_id=arm_id, query_id=query["query_id"],
                                   attempt=attempt, transient=transient)
                    if not transient or local_attempt >= MAX_ATTEMPTS:
                        break
                    progress.retry(details={"arm_id": arm_id, "query_id": query["query_id"],
                                            "attempt": attempt + 1, "backoff_seconds": min(2 ** (attempt - 1), 4)})
                    time.sleep(min(2 ** (attempt - 1), 4))
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                if result is None:
                    continue
                elapsed = float(result.get("timing", {}).get("elapsed_seconds", time.monotonic() - operation_started))
                progress.observe_operation_duration(elapsed)
                counters_after = checkpoint_store.counters()
                actual_new_tokens = int(result.get("cumulative_new_cuda_tokens", 0))
                cumulative_new_tokens = counters_after["new_generated_tokens"]
                progress.tokens = cumulative_new_tokens
                progress.current_tokens = 0
                progress.completed_operations = counters_after["successful_operations"]
                progress.failed_operations = counters_after["failed_operations"]
                progress.cache_hits += max(0, counters_after["cache_hits"] - counters_before["cache_hits"])
                progress.reused_generations += max(
                    0, counters_after["reused_generations"] - counters_before["reused_generations"]
                )
                progress.retried_operations = counters_after["retried_operations"]
                progress.update("query-arm checkpoint written",
                                completed_operations=progress.completed_operations,
                                failed_operations=progress.failed_operations,
                                tokens=cumulative_new_tokens,
                                current_tokens=0,
                                cache_hits=progress.cache_hits,
                                reused_generations=progress.reused_generations,
                                details={"arm_id": arm_id, "query_id": query["query_id"],
                                         "attempt_count": operation_attempts[(arm_id, query["query_id"])],
                                         "operation_elapsed_seconds": elapsed,
                                         "stage_timings": result.get("timing", {}).get("pipeline_component_metrics", {}),
                                         "new_cuda_generated_tokens": actual_new_tokens,
                                         "reused_generation_tokens": result.get("reused_generation_tokens", 0),
                                         "cache_events": result.get("retrieval_candidates_and_ranking_trace", {}).get("stage_cache_events", [])})
                if result.get("status") != "SUCCESS":
                    errors.extend(attempt_errors[-1:])
                # Events carry full token telemetry; the checkpoint and bounded
                # generation cache are the durable copies, so release the
                # engine's append-only event list after every query-arm.
                service.llm_engine.events.clear()
                _update_run_counts(manifest, checkpoint_store, results, records, matrix_arms)
                manifest["errors"] = errors[-100:]
                manifest["elapsed_seconds"] = round(elapsed_before_resume + time.monotonic() - start, 3)
                _write_manifests(run_dir, manifest, matrix)

    except KeyboardInterrupt:
        errors.append("Interrupted by user; resumable checkpoints were kept.")
        progress.error(errors[-1])
        final_status = "PARTIAL"
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        progress.error(errors[-1])
        final_status = "FAILED"
    finally:
        if service is not None:
            try:
                if getattr(service.emb_engine, "_cuda_forward_hook", None) is not None:
                    service.emb_engine._cuda_forward_hook.remove()
                if getattr(service._reranker, "_cuda_forward_hook", None) is not None:
                    service._reranker._cuda_forward_hook.remove()
            except Exception:
                pass
        progress.close()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    current_counters = checkpoint_store.counters()
    if final_status != "FAILED" and current_counters["successful_operations"] == planned_operations \
            and current_counters["failed_operations"] == 0:
        analysis_status = "COMPLETE"
    else:
        analysis_status = final_status if final_status == "FAILED" else "PARTIAL"
    progress.update("local telemetry analysis and judge export",
                    completed_operations=current_counters["successful_operations"],
                    failed_operations=current_counters["failed_operations"],
                    tokens=current_counters["new_generated_tokens"],
                    details={"planned_query_arm_operations": planned_operations})
    summary = finalize_outputs(
        run_dir, run_id=run_id, status=analysis_status,
        questions=records, arms=matrix_arms, results=results,
        dataset_sha256=release_info["manifest"]["dataset"]["questions_sha256"],
        model_manifest_sha256=model_manifest_sha256,
        planned_operations=planned_operations, run_errors=errors,
    )
    final_status = summary["status"]
    completed_count = summary["completed"]
    failed_count = summary["failed"]
    missing_count = summary["missing"]
    post_signature = _logical_signature(db_path)
    post_database_sha256 = sha256_file(db_path)
    if (post_database_sha256 != source_identity["snapshot_sha256"]
            or post_signature != source_identity["logical_fingerprint"]):
        final_status = "SCIENTIFIC_VALIDATION_FAILED"
        errors.append("Read-only query execution changed the frozen SQLite snapshot identity.")
        (run_dir / "completion_report.md").write_text(
            (run_dir / "completion_report.md").read_text(encoding="utf-8").replace(
                f"Status: **{summary['status']}**", "Status: **SCIENTIFIC_VALIDATION_FAILED**"),
            encoding="utf-8")
        metrics_path = run_dir / "metadata/local_metrics_summary.json"
        metrics_summary = _read_json(metrics_path)
        metrics_summary["status"] = final_status
        atomic_write_json(metrics_path, metrics_summary)
    manifest["status"] = final_status
    manifest["database"]["database_after_run_sha256"] = post_database_sha256
    manifest["database"]["after_run_logical_fingerprint"] = post_signature
    manifest["matrix"].update({
        "completed_query_arm_operations": completed_count,
        "failed_query_arm_operations": failed_count,
        "missing_query_arm_operations": missing_count,
    })
    manifest["arms"] = [{"arm_id": arm["arm_id"], "status": arm.get("completion", {}).get("status", "PARTIAL"),
                         "completion": arm.get("completion"),
                         "intervention_integrity": arm.get("intervention_integrity")} for arm in matrix_arms]
    manifest["errors"] = errors[-100:]
    manifest["elapsed_seconds"] = round(elapsed_before_resume + time.monotonic() - start, 3)
    manifest["cache_hits"] = summary["cache_hits"]
    manifest["cache_stage_hit_rates"] = summary["cache_stage_hit_rates"]
    manifest["paired_entropy"] = summary["paired_entropy"]
    progress.update("final integrity verification and result packaging",
                    completed_operations=completed_count, failed_operations=failed_count,
                    tokens=current_counters["new_generated_tokens"],
                    details={"status": final_status, "run_directory": output_dir_string})
    progress.close()
    manifest["resource_usage"] = {
        "peak_gpu_memory_allocated_bytes": progress.peak_gpu_memory_allocated_bytes,
        "peak_gpu_memory_reserved_bytes": progress.peak_gpu_memory_reserved_bytes,
        "stage_seconds": progress.stage_seconds,
        "recent_tokens_per_second": progress.recent_tokens_per_second,
    }
    _write_manifests(run_dir, manifest, matrix)
    _write_trace_index(run_dir, matrix_arms, records)
    archive, archive_sha256 = pack_experiment(run_dir, OUTPUT_ROOT.parent / "return")
    return {"run_directory": str(run_dir), "run_id": run_id,
            "status": final_status, "manifest": manifest,
            "archive": str(archive), "archive_sha256": archive_sha256,
            "planned_query_arm_operations": planned_operations,
            "successful_query_arm_operations": manifest["matrix"]["completed_query_arm_operations"],
            "failed_query_arm_operations": manifest["matrix"]["failed_query_arm_operations"],
            "missing_query_arm_operations": manifest["matrix"]["missing_query_arm_operations"],
            "interrupted": any("Interrupted" in error for error in errors)}


def _run_arm_checkpoint_valid(result: dict[str, Any], arm_id: str, input_identity: str) -> bool:
    from ablation_cuda.runtime.smoke import _resumable_arm_checkpoint
    return _resumable_arm_checkpoint(result, arm_id, input_identity)


def _update_run_counts(manifest: dict[str, Any], checkpoint_store: CheckpointStore,
                       results: LazyResults, questions: list[dict[str, Any]],
                       arms: list[dict[str, Any]]) -> None:
    counters = checkpoint_store.counters()
    successes = counters["successful_operations"]
    failed = counters["failed_operations"]
    planned = len(questions) * len(arms)
    manifest["matrix"]["completed_query_arm_operations"] = successes
    manifest["matrix"]["failed_query_arm_operations"] = failed
    manifest["matrix"]["missing_query_arm_operations"] = max(0, planned - successes - failed)


def _write_trace_index(run_dir: Path, arms: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    traces = []
    for arm in arms:
        for question in questions:
            relative = f"baselines/{arm['arm_id']}/traces/{question['query_id']}.json"
            if (run_dir / relative).is_file():
                traces.append({"arm_id": arm["arm_id"], "query_id": question["query_id"],
                               "trace_path": relative})
    path = run_dir / "traces/index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def finalize_stale_run_for_pack(run_dir: Path) -> dict[str, Any]:
    """Turn a stopped RUNNING directory into a complete, explicitly partial package."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Full run manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("experiment_type") != "full_cuda_ablation":
        raise ValueError("Selected directory is not a full CUDA ablation run")
    if manifest.get("status") != "RUNNING":
        return manifest
    questions = [json.loads(line) for line in (run_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    arms_manifest = _read_json(run_dir / "arms_manifest.json")
    arms = arms_manifest.get("arms", [])
    if len(questions) != 375 or len(arms) != len(ARM_DEFINITIONS):
        raise ValueError("Cannot package the incomplete run: frozen question or arm manifest is incomplete")
    results = LazyResults(run_dir, [arm["arm_id"] for arm in arms], questions)
    summary = finalize_outputs(
        run_dir, run_id=manifest["run_id"], status="PARTIAL",
        questions=questions, arms=arms, results=results,
        dataset_sha256=manifest["dataset"]["sha256"],
        model_manifest_sha256=manifest["run_configuration_hashes"]["model_manifest_sha256"],
        planned_operations=375 * len(arms), run_errors=_previous_errors(run_dir / "logs/errors.jsonl"),
    )
    # An external pack of a stale RUNNING directory cannot certify the final
    # database integrity check performed by normal run finalization.
    if summary["status"] == "COMPLETE":
        summary["status"] = "PARTIAL"
        report = run_dir / "completion_report.md"
        report.write_text(report.read_text(encoding="utf-8").replace(
            "Status: **COMPLETE**", "Status: **PARTIAL**"), encoding="utf-8")
        metrics = _read_json(run_dir / "metadata/local_metrics_summary.json")
        metrics["status"] = "PARTIAL"
        atomic_write_json(run_dir / "metadata/local_metrics_summary.json", metrics)
    manifest["status"] = "PARTIAL"
    manifest["matrix"].update({
        "completed_query_arm_operations": summary["completed"],
        "failed_query_arm_operations": summary["failed"],
        "missing_query_arm_operations": summary["missing"],
    })
    manifest["arms"] = [{"arm_id": arm["arm_id"],
                         "status": arm.get("completion", {}).get("status", "PARTIAL"),
                         "completion": arm.get("completion"),
                         "intervention_integrity": arm.get("intervention_integrity")}
                        for arm in arms]
    manifest.setdefault("errors", []).append(
        "Archive packaged from a stopped RUNNING directory; classified PARTIAL because the final runtime integrity check was not completed."
    )
    atomic_write_json(manifest_path, manifest)
    _atomic_yaml(run_dir / "run_manifest.yaml", manifest)
    atomic_write_json(run_dir / "arms_manifest.json", arms_manifest)
    _atomic_yaml(run_dir / "arms_manifest.yaml", arms_manifest)
    _write_trace_index(run_dir, arms, questions)
    return manifest


def inspect_run_status() -> dict[str, Any]:
    rows = []
    if GRAPH_OUTPUT_ROOT.is_dir():
        for path in sorted(GRAPH_OUTPUT_ROOT.glob("run_*_OCC-RAG-1.7B_CUDA_ABLATION")):
            progress_path = path / "logs/progress.json"
            manifest_path = path / "run_manifest.json"
            progress = {}
            manifest = {}
            if progress_path.is_file():
                try:
                    progress = _read_json(progress_path)
                except Exception as exc:
                    progress = {"error": str(exc)}
            if manifest_path.is_file():
                try:
                    manifest = _read_json(manifest_path)
                except Exception as exc:
                    manifest = {"error": str(exc)}
            rows.append({"run_id": manifest.get("run_id", path.name),
                         "status": manifest.get("status", "INITIALIZING"),
                         "directory": str(path), "progress": progress})
    return {"runs": rows}
