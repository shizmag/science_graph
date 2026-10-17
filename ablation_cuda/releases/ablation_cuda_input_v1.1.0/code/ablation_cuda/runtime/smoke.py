from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from ablation_cuda.identity import arm_input_identity, canonical_json, component_isolated, sha256_bytes, stable_result_identity
from ablation_cuda.runtime.assets import acquire_models, sha256_file
from ablation_cuda.runtime.progress import Progress
from ablation_portability.dataset.loader import load_frozen_dataset, select_frozen_query
from ablation_portability.scripts.rebuild_vector_index import rebuild_index
from ablation_portability.storage import (
    inspect_database,
    validate_retrieval_fixtures,
    validate_vectors,
)


QUESTION_ID = "pubmedqa_21084567"
RELEASE_ROOT = Path(os.environ.get("ABLA_RELEASE_ROOT", "/app/release"))
DB_ROOT = Path(os.environ.get("ABLA_DB_ROOT", "/db"))
MODEL_ROOT = Path(os.environ.get("ABLA_MODEL_ROOT", "/models"))
OUTPUT_ROOT = Path(os.environ.get("ABLA_OUTPUT_ROOT", "/outputs"))
LOG_ROOT = Path(os.environ.get("ABLA_LOG_ROOT", "/logs"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")
    temporary.replace(path)


def _version_manifest() -> dict[str, Any]:
    import torch
    import transformers
    import sentence_transformers
    import tokenizers
    import usearch
    import numpy
    import huggingface_hub
    import yaml as pyyaml
    import rich
    import spacy
    from importlib.metadata import distributions, version

    package_names = [
        "torch", "transformers", "tokenizers", "sentence-transformers", "huggingface-hub",
        "usearch", "numpy", "PyYAML", "spacy", "rank-bm25", "networkx", "tiktoken",
    ]
    installed = {name: version(name) for name in package_names}
    all_installed = {
        distribution.metadata.get("Name", "unknown"): distribution.version
        for distribution in distributions()
    }
    sqlite_connection = sqlite3.connect(":memory:")
    try:
        sqlite_connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(text)")
        sqlite_fts5 = True
    finally:
        sqlite_connection.close()
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "architecture": platform.machine(), "packages": installed,
        "all_installed_distributions": dict(sorted(all_installed.items(), key=lambda item: item[0].lower())),
        "uv_build_version": os.environ.get("ABLA_UV_VERSION", "0.10.9"),
        "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__, "usearch": usearch.__version__,
        "numpy": numpy.__version__, "pyyaml": pyyaml.__version__,
        "rich": rich.__version__, "spacy": spacy.__version__,
        "sqlite_version": sqlite3.sqlite_version, "sqlite_fts5_available": sqlite_fts5,
    }


def validate_release(release_root: Path = RELEASE_ROOT) -> dict[str, Any]:
    manifest_path = release_root / "PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("Input PACKAGE_MANIFEST.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("package_type") != "ablation_cuda_input" or manifest.get("schema_version") != "1.0":
        raise RuntimeError("Unsupported input package manifest")
    expected = manifest.get("expected_files")
    if not isinstance(expected, list) or not expected:
        raise RuntimeError("Input package inventory is empty")
    expected_map = {item["path"]: item for item in expected}
    if len(expected_map) != len(expected):
        raise RuntimeError("Input package inventory contains duplicate paths")
    actual_paths = {path.relative_to(release_root).as_posix() for path in release_root.rglob("*")
                    if path.is_file() and path.name != "PACKAGE_MANIFEST.json"}
    if actual_paths != set(expected_map):
        raise RuntimeError(f"Input package inventory mismatch; missing={sorted(set(expected_map)-actual_paths)}, "
                           f"extra={sorted(actual_paths-set(expected_map))}")
    for relative, item in expected_map.items():
        path = release_root / relative
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Input package checksum mismatch: {relative}")
    records = load_frozen_dataset(release_root / "dataset")
    if len(records) != 375 or sum(row["answerability_label"] == "answerable" for row in records) != 300 or sum(
        row["answerability_label"] == "designated_negative" for row in records
    ) != 75:
        raise RuntimeError("Frozen canonical population does not match 300/75")
    if sha256_file(release_root / "dataset/questions.jsonl") != manifest.get("dataset", {}).get("questions_sha256"):
        raise RuntimeError("Frozen question file does not match the package-level dataset fingerprint")
    if sha256_file(release_root / "dataset/questions_manifest.json") != manifest.get("dataset", {}).get("manifest_sha256"):
        raise RuntimeError("Frozen question manifest does not match the package-level checksum")
    query = select_frozen_query(records, QUESTION_ID)
    db = release_root / "snapshot/database.db"
    db_sha = sha256_file(db)
    if db_sha != manifest["database"]["snapshot_sha256"]:
        raise RuntimeError("Selected external_db file fingerprint does not match package manifest")
    source_summary = inspect_database(db, include_samples=False)
    vectors = validate_vectors(db)
    if not vectors["valid"] or source_summary["quick_check"] != "ok" or not source_summary.get("fts_objects"):
        raise RuntimeError("Selected SQLite/FTS5/stored-vector validation failed")
    if source_summary.get("graph_node_count") != 9680 or source_summary.get("graph_edge_count") != 11712:
        raise RuntimeError("Selected external_db graph identity changed")
    snapshot_manifest = json.loads((release_root / "snapshot/snapshot_manifest.json").read_text(encoding="utf-8"))
    fixture_path = release_root / "snapshot/validation_fixtures.json"
    fixture_sha256 = sha256_file(fixture_path)
    if fixture_sha256 != snapshot_manifest.get("validation_fixtures_sha256"):
        raise RuntimeError("Frozen retrieval fixtures do not match the selected snapshot manifest")
    fixture_validation = validate_retrieval_fixtures(db, fixture_path, fixture_sha256)
    database_identity = json.loads((release_root / "DATABASE_IDENTITY.json").read_text(encoding="utf-8"))
    if database_identity.get("candidate") != "external_db" or database_identity.get("snapshot_sha256") != db_sha:
        raise RuntimeError("Input package does not identify the author-selected external_db baseline")
    model_asset_manifest = release_root / "models/asset_manifest.json"
    if sha256_file(model_asset_manifest) != manifest.get("model_asset_manifest_sha256"):
        raise RuntimeError("Frozen public checkpoint asset manifest checksum mismatch")
    config_path = release_root / "effective_config.yaml"
    if sha256_file(config_path) != manifest.get("effective_config_sha256"):
        raise RuntimeError("Frozen effective configuration checksum mismatch")
    effective = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if effective.get("hf_token") not in (None, "") or effective.get("llm", {}).get("cloud", {}).get("api_key") not in (None, ""):
        raise RuntimeError("Frozen effective configuration contains a credential")
    if effective.get("llm", {}).get("provider") != "cuda_torch" or effective.get("llm", {}).get("max_tokens") != 768:
        raise RuntimeError("Frozen effective configuration does not select the locked CUDA generation protocol")
    return {"manifest": manifest, "dataset_records": len(records), "selected_query": query,
            "source_database_sha256": db_sha, "logical": source_summary, "vectors": vectors,
            "retrieval_fixtures": fixture_validation}


def _logical_signature(path: Path) -> dict[str, Any]:
    summary = inspect_database(path, include_samples=False)
    return {
        "document_count": summary.get("document_count"), "chunk_count": summary.get("chunk_count"),
        "vector_count": summary.get("vector", {}).get("rows"),
        "vector_dimensions": summary.get("vector", {}).get("dimensions"),
        "graph_node_count": summary.get("graph_node_count"),
        "graph_edge_count": summary.get("graph_edge_count"),
        "fts_objects": summary.get("fts_objects"), "quick_check": summary.get("quick_check"),
    }


def _prepare_database(release_info: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    DB_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = DB_ROOT / "database.db"
    expected = release_info["manifest"]["database"]["logical_fingerprint"]
    expected_sha = release_info["manifest"]["database"]["snapshot_sha256"]
    expected_sig = {**expected, "quick_check": "ok"}
    if db_path.is_file():
        actual_sig = _logical_signature(db_path)
        if (sha256_file(db_path) != expected_sha or actual_sig != expected_sig
                or not validate_vectors(db_path)["valid"]):
            quarantine = DB_ROOT / "quarantine"
            quarantine.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            shutil.move(str(db_path), str(quarantine / f"database-{stamp}.db"))
            for suffix in ("-wal", "-shm", ".usearch"):
                sidecar = Path(str(db_path) + suffix) if suffix in ("-wal", "-shm") else db_path.with_suffix(suffix)
                if sidecar.exists():
                    shutil.move(str(sidecar), str(quarantine / f"database-{stamp}{suffix}"))
            _restore_fresh()
    else:
        _restore_fresh()
    if sha256_file(db_path) != expected_sha:
        raise RuntimeError("Restored database file SHA-256 differs from the frozen external_db snapshot")
    if _logical_signature(db_path) != expected_sig:
        raise RuntimeError("Restored database logical fingerprint differs from the frozen external_db snapshot")
    vector_report = validate_vectors(db_path)
    if not vector_report["valid"]:
        raise RuntimeError("Stored vectors failed validation; automatic re-embedding is disabled")
    index_path = db_path.with_suffix(".usearch")
    index_valid = False
    if index_path.is_file():
        try:
            from usearch.index import Index
            index = Index(ndim=768, metric="cos")
            index.load(str(index_path))
            index_valid = len(index) == vector_report["rows"] and index.ndim == 768
        except Exception:
            index_valid = False
    if not index_valid:
        if index_path.exists():
            quarantine = DB_ROOT / "quarantine"
            quarantine.mkdir(exist_ok=True)
            shutil.move(str(index_path), str(quarantine / f"database-{int(time.time())}.usearch"))
        rebuild_index(db_path, index_path)
    return db_path, {"snapshot_sha256": release_info["manifest"]["database"]["snapshot_sha256"],
                     "restored_database_sha256": sha256_file(db_path),
                     "logical_fingerprint": _logical_signature(db_path),
                     "vector_validation": vector_report,
                     "usearch_index_sha256": sha256_file(index_path),
                     "snapshot_file_restored_byte_exactly": True,
                     "usearch_rebuilt_from_persisted_vectors": not index_valid,
                     "recomputed_embeddings": False}


def _restore_fresh() -> None:
    source = RELEASE_ROOT / "snapshot/database.db"
    staging = DB_ROOT / f".database-{os.getpid()}-{time.time_ns()}.tmp"
    shutil.copy2(source, staging)
    os.replace(staging, DB_ROOT / "database.db")


def _write_private_config(base_config_path: Path, db_path: Path,
                          models: dict[str, Any], home: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    cfg["db_path"] = str(db_path)
    cfg["embedding"]["model_name"] = models["models"]["embedder"]["snapshot_path"]
    cfg["reranker"]["model_name"] = models["models"]["cross_encoder"]["snapshot_path"]
    cfg["llm"]["local"] = {
        "model_path": models["models"]["generator"]["snapshot_path"],
        "rag_model_path": models["models"]["generator"]["snapshot_path"],
    }
    cfg["spacy"]["model_name"] = "en_core_web_md"
    cfg["hf_token"] = ""
    cfg["llm"]["cloud"]["api_key"] = ""
    config_path = home / ".config/pdf-graph-analyzer/config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return cfg


def _actual_cuda_probe() -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke, but torch.cuda.is_available() is false")
    device = torch.device("cuda:0")
    left = torch.arange(4096, dtype=torch.float32, device=device).reshape(64, 64)
    right = left.T.contiguous()
    result = left @ right
    scalar = float(result.sum().item())
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item() or not scalar:
        raise RuntimeError("CUDA matrix operation produced invalid output")
    device_name = torch.cuda.get_device_name(0)
    compute_capability = list(torch.cuda.get_device_capability(0))
    if "5090" not in device_name.lower() or compute_capability != [12, 0]:
        raise RuntimeError(f"This frozen smoke targets an RTX 5090 (Blackwell sm_120); detected {device_name}, "
                           f"compute capability {compute_capability}")
    try:
        driver_version = torch._C._cuda_getDriverVersion()
    except (AttributeError, RuntimeError):
        driver_version = None
    return {"operation": "float32_matmul_64x64", "result_sum": scalar,
            "device": device_name,
            "device_index": torch.cuda.current_device(),
            "compute_capability": compute_capability,
            "torch_version": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
            "driver_version": driver_version,
            "operation_device": str(result.device)}


def _instrument_service(service: Any, device_trace: dict[str, Any], question_text: str) -> None:
    import torch

    def tensor_devices(value: Any) -> set[str]:
        if isinstance(value, torch.Tensor):
            return {str(value.device)}
        if isinstance(value, dict):
            return set().union(*(tensor_devices(item) for item in value.values())) if value else set()
        if isinstance(value, (tuple, list)):
            return set().union(*(tensor_devices(item) for item in value)) if value else set()
        return set()

    engine = service.emb_engine
    engine._ensure_model_loaded()
    embedding_devices = {str(parameter.device) for parameter in engine.model.parameters()}
    if not embedding_devices or not all(item.startswith("cuda:") for item in embedding_devices):
        raise RuntimeError(f"E5 model fell back from CUDA: {sorted(embedding_devices)}")
    engine._cuda_forward_input_devices = []
    engine._cuda_forward_hook = engine.model.register_forward_pre_hook(
        lambda module, args, kwargs: engine._cuda_forward_input_devices.extend(
            sorted(tensor_devices((args, kwargs)))
        ), with_kwargs=True,
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    embedding = engine.get_embedding(question_text)
    end.record(); torch.cuda.synchronize()
    if len(embedding) != 768:
        raise RuntimeError(f"E5 query embedding dimension changed: {len(embedding)}")
    device_trace["e5_query_embedding"] = {
        "model_device_set": sorted(embedding_devices),
        "input_device_set": sorted(set(engine._cuda_forward_input_devices)),
        "inference_executed": True, "output_dimension": len(embedding),
        "output_norm": sum(value * value for value in embedding) ** 0.5,
        "cuda_elapsed_seconds": start.elapsed_time(end) / 1000.0,
    }
    # The warm-up above is only a placement check. Record the actual E5 model
    # calls made by each arm separately, and clear the warm-up/cache value.
    engine._query_cache.clear()
    if not hasattr(engine, "_cuda_call_records"):
        engine._cuda_call_records = []
    original_encode = engine.model.encode

    def timed_encode(*args, **kwargs):
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        input_start = len(engine._cuda_forward_input_devices)
        before.record()
        result = original_encode(*args, **kwargs)
        after.record()
        torch.cuda.synchronize()
        engine._cuda_call_records.append({
            "inference_executed": True,
            "model_device_set": sorted(embedding_devices),
            "input_device_set": sorted(set(engine._cuda_forward_input_devices[input_start:])),
            "cuda_elapsed_seconds": before.elapsed_time(after) / 1000.0,
            "output_shape": list(getattr(result, "shape", ())),
        })
        return result

    engine.model.encode = timed_encode

    reranker = service._get_reranker()
    ce_devices = {str(parameter.device) for parameter in reranker.model.parameters()}
    if not ce_devices or not all(item.startswith("cuda:") for item in ce_devices):
        raise RuntimeError(f"Cross-Encoder fell back from CUDA: {sorted(ce_devices)}")
    ce_calls: list[dict[str, Any]] = []
    reranker._cuda_forward_input_devices = []
    reranker._cuda_forward_hook = reranker.model.register_forward_pre_hook(
        lambda module, args, kwargs: reranker._cuda_forward_input_devices.extend(
            sorted(tensor_devices((args, kwargs)))
        ), with_kwargs=True,
    )
    original_predict = reranker.predict

    def timed_predict(pairs, *args, **kwargs):
        import inspect
        graph_expander_call = any(
            frame.filename.replace("\\", "/").endswith("/src/services/graph_expander.py")
            for frame in inspect.stack(context=0)
        )
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        input_start = len(reranker._cuda_forward_input_devices)
        before.record()
        result = original_predict(pairs, *args, **kwargs)
        after.record(); torch.cuda.synchronize()
        ce_calls.append({"role": "experimental_graph_expander" if graph_expander_call else "final_candidate_reranking",
                         "pair_count": len(pairs), "model_device_set": sorted(ce_devices),
                         "input_device_set": sorted(set(reranker._cuda_forward_input_devices[input_start:])),
                         "inference_executed": True,
                         "cuda_elapsed_seconds": before.elapsed_time(after) / 1000.0})
        return result

    reranker.predict = timed_predict
    probe_scores = reranker.predict([(question_text, "CUDA placement verification passage")])
    if len(probe_scores) != 1:
        raise RuntimeError("Cross-Encoder CUDA placement probe returned an invalid result")
    device_trace["cross_encoder"] = {"model_device_set": sorted(ce_devices),
                                     "input_device": sorted(ce_devices)[0],
                                     "inference_executed": True,
                                     "probe_score": float(probe_scores[0]),
                                     "calls": list(ce_calls)}

    service.llm_engine._ensure_model_loaded()
    generator_manifest = service.llm_engine.model_device_manifest()
    if not all(device.startswith("cuda:") for device in generator_manifest["model_device_set"]):
        raise RuntimeError("OCC-RAG weights are not wholly resident on CUDA")
    device_trace["generator"] = {**generator_manifest, "weights_dtype": "bfloat16",
                                  "inference_executed": False,
                                  "loading_probe_only": True}
    service._cuda_cross_encoder_calls = ce_calls


QUESTION_TEXT = "Are stroke patients' reports of home blood pressure readings reliable?"
TELEMETRY_IMPLEMENTATION_VERSION = "cuda-token-telemetry-v1"


def _capture_context(service: Any, capture: dict[str, Any]) -> None:
    if not hasattr(service, "_cuda_original_build_context"):
        service._cuda_original_build_context = service.build_context
        service._cuda_original_trim_context = service.trim_context
    original_build = service._cuda_original_build_context
    original_trim = service._cuda_original_trim_context

    def build_context(*args, **kwargs):
        result = original_build(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            capture["context_build"] = {"context_text": result[0], "context_graph": result[1],
                                        "context_text_chars": len(result[0]),
                                        "context_graph_chars": len(result[1])}
        return result

    def trim_context(*args, **kwargs):
        result = original_trim(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            output_chunk_ids = [
                str(item[0].id if isinstance(item, (tuple, list)) else getattr(item, "id", None))
                for item in result[2]
            ]
            capture["context_trim"] = {
                "input_text_chars": len(kwargs.get("context_text", "")),
                "input_graph_chars": len(kwargs.get("context_graph", "")),
                "output_text_chars": len(result[0]), "output_graph_chars": len(result[1]),
                "output_text": result[0], "output_graph": result[1],
                "input_chunk_count": len(kwargs.get("final_chunks", [])),
                "output_chunk_count": len(result[2]),
                "output_chunk_ids": output_chunk_ids,
            }
        return result

    service.build_context = build_context
    service.trim_context = trim_context


def _set_spacy_pythonpath(model_root: Path) -> None:
    spacy_packages = model_root / "spacy_packages"
    if spacy_packages.is_dir():
        sys.path.insert(0, str(spacy_packages))
        current = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = str(spacy_packages) + (os.pathsep + current if current else "")


def _compare_mlx_reference(reference_root: Path, cuda_arm: dict[str, Any]) -> dict[str, Any]:
    prompt_path = reference_root / "prompt.txt"
    generation_path = reference_root / "generation.json"
    telemetry_path = reference_root / "telemetry.json"
    if not all(path.is_file() for path in (prompt_path, generation_path, telemetry_path)):
        return {"available": False, "reason": "The frozen Phase 1 prompt, generation, or telemetry file is missing"}

    mlx_prompt = prompt_path.read_bytes()
    cuda_prompt = str(cuda_arm.get("prompt", "")).encode("utf-8")
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    mlx_output = generation.get("raw_generated_answer")
    if not mlx_output:
        telemetry_json = json.loads(telemetry_path.read_text(encoding="utf-8"))
        calls = telemetry_json.get("captured_generation_calls", [])
        mlx_output = calls[-1].get("response") if calls else ""
    mlx_telemetry = json.loads(telemetry_path.read_text(encoding="utf-8")).get("tokens_info", [])
    cuda_telemetry = cuda_arm.get("token_telemetry", [])
    mlx_ids = [item.get("token_id") for item in mlx_telemetry]
    cuda_ids = [item.get("token_id") for item in cuda_telemetry]
    shared_prefix = 0
    for left, right in zip(mlx_ids, cuda_ids):
        if left != right:
            break
        shared_prefix += 1
    fields = (
        "logprob", "nll_nats", "entropy_k20_bits", "entropy_k50_bits", "entropy_k100_bits",
        "retained_mass_k20", "retained_mass_k50", "retained_mass_k100", "top1_prob_full",
        "logit_margin",
    )
    metric_deltas = {}
    for field in fields:
        pairs = [(float(mlx_telemetry[index][field]), float(cuda_telemetry[index][field]))
                 for index in range(shared_prefix)
                 if field in mlx_telemetry[index] and field in cuda_telemetry[index]]
        if pairs:
            differences = [abs(left - right) for left, right in pairs]
            metric_deltas[field] = {
                "aligned_tokens": len(pairs),
                "mean_absolute_difference": sum(differences) / len(differences),
                "maximum_absolute_difference": max(differences),
            }
    cuda_output = str(cuda_arm.get("raw_generated_response", ""))
    first_divergence = shared_prefix if shared_prefix < min(len(mlx_ids), len(cuda_ids)) else None
    return {
        "available": True,
        "prompt_sha256_mlx": hashlib.sha256(mlx_prompt).hexdigest(),
        "prompt_sha256_cuda": hashlib.sha256(cuda_prompt).hexdigest(),
        "prompt_byte_identical": mlx_prompt == cuda_prompt,
        "mlx_token_count": len(mlx_ids), "cuda_token_count": len(cuda_ids),
        "token_id_sequence_identical": mlx_ids == cuda_ids,
        "shared_token_prefix_count": shared_prefix,
        "first_token_divergence_index_zero_based": first_divergence,
        "raw_response_identical": mlx_output == cuda_output,
        "mlx_raw_response_sha256": hashlib.sha256(str(mlx_output).encode("utf-8")).hexdigest(),
        "cuda_raw_response_sha256": hashlib.sha256(cuda_output.encode("utf-8")).hexdigest(),
        "aligned_prefix_telemetry_absolute_differences": metric_deltas,
        "telemetry_comparison_scope": "Only matching token IDs before the first sequence divergence are compared",
        "backend_difference": "MLX and PyTorch/Transformers use different execution kernels and cache implementations; sequence identity is measured, not assumed",
    }


def _run_arm(*, arm_name: str, component_override: str | None, query_row: dict[str, Any],
             runtime_config: dict[str, Any], service: Any, run_dir: Path,
             progress: Progress, completed: int, model_manifest: dict[str, Any],
             db_identity: dict[str, Any], source_identity: dict[str, Any],
             image_identity: dict[str, Any], code_sha256: str,
             output_dir: Path | None = None, checkpoint_file: Path | None = None,
             preserve_query_embedding_cache: bool = False,
             require_query_embedding_inference: bool = True,
             require_cross_encoder_inference: bool = True,
             write_auxiliary_files: bool = True,
             input_identity_override: str | None = None) -> dict[str, Any]:
    from src.config import config
    from core.config import get_baseline_config
    from core.generation import run_query_on_baseline
    from src.services.rag_service import parse_reasoning_response
    import torch

    config.data = copy.deepcopy(runtime_config)
    if component_override:
        config.data["rag_components"][component_override] = False
    base_components = get_baseline_config("B6", config.rag_components)
    effective = {"config": copy.deepcopy(config.data), "baseline": "B6",
                 "paper_baseline": "B5",
                 "baseline_components": base_components,
                 "intervention": ({"component": component_override, "enabled": False}
                                  if component_override else None)}
    query_id = query_row["query_id"]
    question = query_row["question"]
    query_sha = sha256_bytes(question.encode("utf-8"))
    decoding = {"strategy": "greedy_argmax", "temperature": 0.0,
                "max_new_tokens": 768, "context_limit": 12000,
                "stop_token_ids": [151643, 151645, 151683],
                "tokenizer_revision": model_manifest["models"]["generator"]["revision"]}
    input_id = arm_input_identity(
        query_id=query_id, question_sha256=query_sha,
        snapshot_sha256=source_identity["snapshot_sha256"],
        model_manifest_sha256=sha256_bytes(canonical_json(model_manifest)),
        effective_config=effective, decoding=decoding,
    )
    input_id = sha256_bytes(canonical_json({"input_id": input_id, "code_sha256": code_sha256,
                                            "image": image_identity,
                                            "telemetry_implementation": TELEMETRY_IMPLEMENTATION_VERSION}))
    if input_identity_override is not None:
        input_id = input_identity_override
    arm_path = checkpoint_file or (run_dir / "arms" / f"{arm_name}.json")
    arm_output_dir = output_dir or (run_dir / "arms" / arm_name)
    if arm_path.is_file():
        try:
            saved = json.loads(arm_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            saved = None
        if saved and _resumable_arm_checkpoint(saved, arm_name, input_id):
            progress.update(f"resume {arm_name}", completed_arms=completed,
                            tokens=progress.tokens + int(saved.get("total_pipeline_generated_tokens", 0)),
                            details={"result_id": saved.get("stable_result_id")})
            return saved
        _preserve_stale_arm_checkpoint(run_dir, arm_name)

    service._cuda_cross_encoder_calls.clear()
    service._last_graph_expander = None
    service._last_graph_trace = {}
    service._last_ranking_trace = {}
    service.current_trace = {
        "query_id": query_id, "category": query_row.get("historical_category", "single-document"),
        "baseline": "B6", "arm": arm_name,
    }
    service._ablation_cache_events = []
    query_embedding_cache_hit = bool(preserve_query_embedding_cache and
                                     question in service.emb_engine._query_cache)
    if query_embedding_cache_hit:
        service._ablation_cache_events.append({"stage": "query_embedding", "hit": True, "items": 1})
    if not preserve_query_embedding_cache:
        service.emb_engine._query_cache.clear()
    service._last_fixed_context_budget = None
    embedding_call_start = len(service.emb_engine._cuda_call_records)
    token_generation_state = {"current_generation_tokens": 0, "completed_generation_tokens": 0}
    token_generation_offset = progress.tokens

    def generation_progress(*, kind: str, token_count: int, elapsed_seconds: float) -> None:
        if token_count < token_generation_state["current_generation_tokens"]:
            token_generation_state["completed_generation_tokens"] += token_generation_state["current_generation_tokens"]
        token_generation_state["current_generation_tokens"] = token_count
        progress.token_progress(
            f"{arm_name}: CUDA generation ({kind})", completed_arms=completed,
            tokens=token_generation_offset + token_generation_state["completed_generation_tokens"] + token_count,
            details={"generation_elapsed_seconds": round(elapsed_seconds, 3),
                     "current_generation_tokens": token_count},
        )

    service.llm_engine.progress_callback = generation_progress
    context_capture: dict[str, Any] = {}
    _capture_context(service, context_capture)
    if hasattr(service, "_cuda_original_expand_query"):
        original_expand_query = service._cuda_original_expand_query
    else:
        original_expand_query = service._expand_query
        service._cuda_original_expand_query = original_expand_query

    def traced_expand_query(query_text):
        enabled = bool(config.rag_components.get("graph_ontology_lookup", True))
        eligible = len(str(query_text).strip().split()) <= 2
        variants = original_expand_query(query_text)
        service.current_trace["query_expansion"] = {
            "ontology_lookup_enabled": enabled,
            "ontology_lookup_condition_eligible": eligible,
            "query_variants": list(variants),
            "ontology_derived_variants": [item for item in variants if item != query_text],
            "ontology_lookup_branch_executed": bool(enabled and eligible),
        }
        return variants

    service._expand_query = traced_expand_query
    event_count = len(service.llm_engine.events)
    cross_encoder_start = len(service._cuda_cross_encoder_calls)
    progress.update(f"{arm_name}: query preprocessing and retrieval", completed_arms=completed)
    started = time.perf_counter()
    error = None
    raw_answer = ""
    parsed_answer = ""
    parser_status = None
    retrieved = []
    metrics = {}
    chunks = []
    try:
        case = {
            "id": query_id, "category": query_row.get("historical_category", "single-document"),
            "query": question, "golden_answer": query_row["gold_reference_answer"],
            "expected_papers": query_row.get("original_paper_ids", []),
            "is_answerable": query_row["answerability_label"] == "answerable",
        }
        raw_answer, retrieved, metrics, chunks = run_query_on_baseline(
            service, question, "B6", use_cloud=False, config=config)
        raw_answer = getattr(service, "last_raw_response", None) or raw_answer
        parser_status, parsed_answer = parse_reasoning_response(raw_answer)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        service.llm_engine.progress_callback = None
        service._expand_query = original_expand_query
    elapsed = time.perf_counter() - started
    events = service.llm_engine.events[event_count:]
    answer_events = [event for event in events if event.get("kind") == "response_with_logits" and
                     event.get("response") == raw_answer]
    final_event = answer_events[-1] if answer_events else (events[-1] if events else {})
    input_prompt = final_event.get("prompt", "")
    prompt = final_event.get("formatted_prompt", "")
    telemetry = metrics.get("tokens_info", []) if isinstance(metrics, dict) else []
    generation_events = []
    for event in events:
        if event is final_event:
            purpose = "final_answer_generation"
        elif "Answer based on your general knowledge." in str(event.get("prompt", "")):
            purpose = "shannon_b0_entropy_reference"
        else:
            purpose = "pipeline_diagnostic_generation"
        generation_events.append({
            "purpose": purpose,
            "event": {key: value for key, value in event.items() if key != "token_telemetry"},
            "token_telemetry": event.get("token_telemetry", []),
        })
    total_pipeline_tokens = sum(len(event["token_telemetry"]) for event in generation_events)
    progress.update(f"{arm_name}: output and telemetry", completed_arms=completed,
                    tokens=token_generation_offset + total_pipeline_tokens)
    if raw_answer and (not prompt or not telemetry):
        error = error or "Final CUDA generation did not expose its exact prompt or per-token telemetry"

    trace = {
        "current_trace": copy.deepcopy(service.current_trace),
        "graph_retrieval": copy.deepcopy(getattr(service, "_last_graph_trace", {})),
        "ranking": copy.deepcopy(getattr(service, "_last_ranking_trace", {})),
        "graph_expander": copy.deepcopy(getattr(getattr(service, "_last_graph_expander", None), "last_telemetry", None)),
        "dynamic_graph_output": getattr(getattr(service, "_last_graph_expander", None), "last_output", None),
        "context_capture": context_capture,
        "stage_cache_events": copy.deepcopy(getattr(service, "_ablation_cache_events", [])),
        "component_metrics": metrics.get("components", {}) if isinstance(metrics, dict) else {},
        "graph_branch_execution": copy.deepcopy(getattr(service, "_last_graph_trace", {}).get("branch_execution", {})),
        "graph_branch_candidate_counts": copy.deepcopy(getattr(service, "_last_graph_trace", {}).get("branch_candidate_counts", {})),
        "trace_counters": {
            "lexical_seed_count": len(service.current_trace.get("seed_chunks_from_lexical_dense", {}).get("lexical", [])),
            "dense_seed_count": len(service.current_trace.get("seed_chunks_from_lexical_dense", {}).get("dense", [])),
            "graph_candidate_count": getattr(service, "_last_graph_trace", {}).get("graph_chunk_candidates_count", 0),
            "candidate_selection_input_count": len(getattr(service, "_last_ranking_trace", {}).get("candidate_ids_before_reranker", [])),
            "reranker_returned_count": len(getattr(service, "_last_ranking_trace", {}).get("returned_chunk_ids", [])),
            "final_evidence_chunk_count": len(chunks),
            "static_context_chars": context_capture.get("context_build", {}).get("context_text_chars", 0),
            "static_graph_context_chars": context_capture.get("context_build", {}).get("context_graph_chars", 0),
            "dynamic_graph_prompt_chars": len(getattr(getattr(service, "_last_graph_expander", None), "last_output", None) or ""),
            "serialized_prompt_chars": len(prompt),
            "serialized_prompt_tokens": final_event.get("prompt_token_count"),
        },
    }
    if service._last_fixed_context_budget:
        fixed = copy.deepcopy(service._last_fixed_context_budget)
        trace["context_capture"]["fixed_budget_policy"] = fixed
        trace["context_capture"]["context_trim"] = {
            "policy": fixed.get("policy"),
            "output_text": fixed.get("trimmed_text", ""),
            "output_graph": fixed.get("trimmed_graph", ""),
            "output_chunk_ids": [str(item[0].id if isinstance(item, (tuple, list))
                                       else getattr(item, "id", None)) for item in chunks],
            "output_chunk_count": len(chunks),
            "fixed_budget_policy": fixed,
        }
    context_build = context_capture.get("context_build", {})
    context_trim = context_capture.get("context_trim", {})
    graph_expander = getattr(service, "_last_graph_expander", None)
    dynamic_graph_output = getattr(graph_expander, "last_output", None)
    static_context_text = context_trim.get("output_text", "")
    static_context_graph = context_trim.get("output_graph", "")
    graph_candidate_ids = set(trace["graph_retrieval"].get("graph_chunk_candidate_chunk_ids", []))
    selected_context_ids = set(context_trim.get("output_chunk_ids", []))
    trace["context_and_prompt_assembly"] = {
        "static_context_text_inserted": bool(static_context_text and static_context_text in prompt),
        "static_graph_context_inserted": bool(static_context_graph and static_context_graph in prompt),
        "static_graph_context_enabled": bool(config.rag_components.get("static_graph_context", config.rag_components.get("graph_expansion", True))),
        "static_graph_context_disabled": not bool(config.rag_components.get("static_graph_context", config.rag_components.get("graph_expansion", True))),
        "dynamic_graph_expansion_executed": getattr(graph_expander, "last_telemetry", None) is not None,
        "dynamic_graph_output_inserted": bool(dynamic_graph_output and
                                               dynamic_graph_output != "No essential knowledge graph enrichment found."
                                               and dynamic_graph_output in prompt),
        "graph_candidate_chunks_surviving_context": sorted(graph_candidate_ids & selected_context_ids),
        "context_chunk_ids_after_trimming": context_trim.get("output_chunk_ids", []),
        "serialized_prompt_sha256": sha256_bytes(prompt.encode("utf-8")) if prompt else None,
        "serialized_prompt_characters": len(prompt),
        "formatted_model_input_characters": len(final_event.get("formatted_prompt", "")),
        "prompt_template_branch": "ask_expander" if dynamic_graph_output and dynamic_graph_output in prompt else "ask_no_expander",
        "fixed_budget_policy_executed": bool(service._last_fixed_context_budget),
    }
    trace["trace_counters"].update({
        "merged_candidates_before_reranking": len(trace["ranking"].get("candidate_ids_before_reranker", [])),
        "selected_candidate_count_after_reranking": len(trace["ranking"].get("returned_chunk_ids", [])),
        "context_chunks_after_trimming": len(context_trim.get("output_chunk_ids", [])),
        "static_context_characters_inserted": len(static_context_text) if static_context_text in prompt else 0,
        "static_graph_characters_inserted": len(static_context_graph) if static_context_graph in prompt else 0,
        "dynamic_graph_characters_inserted": len(dynamic_graph_output or "")
        if dynamic_graph_output and dynamic_graph_output in prompt else 0,
        "serialized_prompt_characters": len(prompt),
        "serialized_model_input_tokens": final_event.get("prompt_token_count"),
    })
    arm_devices = {
        "e5_query_embedding": {"model_device_set": [str(p.device) for p in service.emb_engine.model.parameters()]},
        "cross_encoder_model_device_set": sorted({str(p.device) for p in service._reranker.model.parameters()}),
        "cross_encoder_cuda_calls": service._cuda_cross_encoder_calls[cross_encoder_start:],
        "cross_encoder_role_call_counts": {
            role: sum(call.get("role") == role
                      for call in service._cuda_cross_encoder_calls[cross_encoder_start:])
            for role in ("final_candidate_reranking", "experimental_graph_expander")
        },
        "generator_model_device_set": sorted({str(p.device) for p in service.llm_engine.model.parameters()}),
        "generation_input_device": final_event.get("input_device"),
        "generation_logit_device": final_event.get("logit_device"),
        "token_telemetry_devices": sorted({item.get("telemetry_device") for item in telemetry if item.get("telemetry_device")}),
        "e5_cuda_calls": service.emb_engine._cuda_call_records[embedding_call_start:],
        "cuda_inference_results": {
            "embedding": bool(service.emb_engine.device and service.emb_engine.device.startswith("cuda")),
            "embedding_inference": any(item.get("inference_executed") for item in service.emb_engine._cuda_call_records[embedding_call_start:]),
            "cross_encoder": any(item.get("inference_executed") for item in service._cuda_cross_encoder_calls[cross_encoder_start:]),
            "cross_encoder_cached": any(item.get("cache_hit") for item in service._cuda_cross_encoder_calls[cross_encoder_start:]),
            "generation": bool(final_event and final_event.get("logit_device", "").startswith("cuda")),
            "token_metrics": bool(telemetry) and all(item.get("telemetry_device", "").startswith("cuda") for item in telemetry),
        },
    }
    stage_status = {
        "preprocessing": "complete" if service.current_trace else "error",
        "lexical_retrieval": "complete" if "lexical" in service.current_trace.get("seed_chunks_from_lexical_dense", {}) else "missing",
        "dense_retrieval": "complete" if "dense" in service.current_trace.get("seed_chunks_from_lexical_dense", {}) else "missing",
        "cross_encoder": "complete" if trace["ranking"].get("stage_status") in {"complete", "reranker_disabled"} else trace["ranking"].get("stage_status", "missing"),
        "graph_retrieval": "complete" if trace["graph_retrieval"] else "not_recorded",
        "graph_expansion": (
            "complete" if not effective["baseline_components"].get("graph_expansion", False)
            or (trace["graph_expander"] is not None and trace["dynamic_graph_output"] is not None)
            else "missing"
        ),
        "context_construction": "complete" if context_capture.get("context_build") and (context_capture.get("context_trim") or context_capture.get("fixed_budget_policy")) else "missing",
        "generation": "complete" if raw_answer and prompt else "error",
        "answerability_parser": "complete" if parser_status is not None else "error",
    }
    success = error is None and all(value == "complete" for key, value in stage_status.items()
                                    if key in {"preprocessing", "lexical_retrieval", "dense_retrieval", "cross_encoder", "context_construction", "generation", "answerability_parser"})
    if success and require_query_embedding_inference and not query_embedding_cache_hit and not arm_devices["cuda_inference_results"]["embedding_inference"]:
        success = False
        error = "No instrumented CUDA E5 encode call occurred inside the query arm"
    if success and require_cross_encoder_inference and not arm_devices["cuda_inference_results"]["cross_encoder"]:
        success = False
        error = "No instrumented CUDA Cross-Encoder call occurred inside the query arm"
    if success and effective["baseline_components"].get("reranker", True) and not arm_devices["cross_encoder_role_call_counts"]["final_candidate_reranking"]:
        success = False
        error = "No CUDA Cross-Encoder inference was traced at final candidate reranking"
    if success and any(
        not str(token.get("telemetry_device", "")).startswith("cuda")
        for event in generation_events for token in event["token_telemetry"]
    ):
        success = False
        error = "A pipeline generation call computed token telemetry outside CUDA"
    if success and any(
        not str(event["event"].get("input_device", "")).startswith("cuda")
        or not str(event["event"].get("logit_device", "")).startswith("cuda")
        for event in generation_events
    ):
        success = False
        error = "A pipeline generation call did not execute on CUDA"
    e5_calls = service.emb_engine._cuda_call_records[embedding_call_start:]
    if success and any(not call.get("cache_hit") and (not call.get("input_device_set") or any(
        not str(device).startswith("cuda") for device in call["input_device_set"]
    )) for call in e5_calls):
        success = False
        error = "An E5 model inference input was not observed on CUDA"
    ce_calls = service._cuda_cross_encoder_calls[cross_encoder_start:]
    if success and any(not call.get("cache_hit") and (not call.get("input_device_set") or any(
        not str(device).startswith("cuda") for device in call["input_device_set"]
    )) for call in ce_calls):
        success = False
        error = "A Cross-Encoder inference input was not observed on CUDA"
    if runtime_config.get("graph_retrieval", {}).get("enabled", False) and stage_status["graph_retrieval"] != "complete":
        success = False
        error = error or "Configured graph retrieval branch did not produce a trace"
    if runtime_config.get("rag_components", {}).get("graph_expansion", False) and stage_status["graph_expansion"] != "complete":
        success = False
        error = error or "Configured dynamic graph-expansion branch did not complete"
    graph_telemetry = trace.get("graph_expander") or {}
    graph_scoring_expected = (
        any(float(item.get("rerank_time", 0.0) or 0.0) > 0
            for item in graph_telemetry.get("hops", []))
        or float(graph_telemetry.get("ingestion", {}).get("rerank_time", 0.0) or 0.0) > 0
    )
    arm_devices["cross_encoder_role_scoring_expected"] = {
        "experimental_graph_expander": graph_scoring_expected,
    }
    if (success and graph_scoring_expected
            and not arm_devices["cross_encoder_role_call_counts"]["experimental_graph_expander"]):
        success = False
        error = "Graph-expander trace reports Cross-Encoder scoring but no CUDA graph-expander call was traced"
    expected_blending = bool(effective["baseline_components"].get("score_blending", False))
    expected_reranker = bool(effective["baseline_components"].get("reranker", True))
    if success and expected_reranker and trace["ranking"].get("score_blending_enabled") is not expected_blending:
        error = "Ranking trace does not match the frozen B6 score_blending setting"
        success = False
    if success and expected_reranker:
        expected_formula = "normalized_cross_encoder_rrf_blend" if expected_blending else "raw_cross_encoder_score"
        if trace["ranking"].get("ranking_formula") != expected_formula:
            error = "Ranking formula trace does not match the frozen B6 score_blending setting"
            success = False
    elif success and (trace["ranking"].get("stage_status") != "reranker_disabled"
                      or trace["ranking"].get("score_blending_branch_executed") is not False):
        error = "Final reranker-off arm did not take the disabled ranking branch"
        success = False
    if success and component_override:
        if effective["baseline_components"].get(component_override) is not False:
            error = f"Intervention did not disable {component_override} in the effective B6 component bundle"
            success = False
        elif component_override == "score_blending" and trace["ranking"].get("score_blending_branch_executed") is not True:
            error = "The score_blending intervention code path did not execute"
            success = False
        elif component_override == "graph_expansion" and trace["context_and_prompt_assembly"].get("dynamic_graph_expansion_executed") is not False:
            error = "The graph_expansion intervention code path did not disable dynamic expansion"
            success = False

    stable_id = stable_result_identity(input_id, prompt) if prompt else input_id
    result = {
        "schema_version": "1.0", "arm": arm_name,
        "status": "SUCCESS" if success else "PARTIAL",
        "query_id": query_id, "question": question,
        "gold_reference_answer": query_row["gold_reference_answer"],
        "gold_paper_ids": query_row.get("original_paper_ids", []),
        "gold_pmids": query_row.get("original_pmids", []),
        "answerability_label": query_row.get("answerability_label"),
        "synthetic_negative": query_row.get("answerability_label") == "designated_negative",
        "source_pmids": query_row.get("original_pmids", []),
        "parsed_answer": parsed_answer, "parser_status": parser_status,
        "raw_generated_response": raw_answer, "input_prompt": input_prompt, "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")) if prompt else None,
        "retrieved_paper_ids": retrieved, "selected_evidence": chunks,
        "retrieval_candidates_and_ranking_trace": trace,
        "effective_configuration": effective,
        "decoding": decoding,
        "input_identity": input_id, "stable_result_id": stable_id,
        "model_manifest_sha256": sha256_bytes(canonical_json(model_manifest)),
        "snapshot_identity": source_identity,
        "device_diagnostics": arm_devices,
        "query_embedding_cache_hit": query_embedding_cache_hit,
        "generation_event": {key: value for key, value in final_event.items() if key != "token_telemetry"},
        "generation_events": generation_events,
        "total_pipeline_generated_tokens": total_pipeline_tokens,
        "token_telemetry": telemetry,
        "stage_status": stage_status,
        "timing": {"elapsed_seconds": elapsed,
                   "pipeline_component_metrics": metrics.get("components", {}) if isinstance(metrics, dict) else {},
                   "cuda_cross_encoder_calls": service._cuda_cross_encoder_calls[cross_encoder_start:]},
        "result_errors": [error] if error else [],
        "image_identity": image_identity,
        "source_code_sha256": code_sha256,
    }
    _atomic_json(arm_path, result)
    if write_auxiliary_files:
        for filename, value in (("effective_config.json", effective),
                                 ("retrieval_trace.json", trace),
                                 ("generation.json", {"raw_generated_response": raw_answer,
                                                       "parsed_answer": parsed_answer,
                                                       "parser_status": parser_status,
                                                       "generation_event": {key: value for key, value in final_event.items()
                                                                            if key != "token_telemetry"},
                                                       "generation_events": generation_events}),
                                 ("token_telemetry.json", telemetry)):
            _atomic_json(arm_output_dir / filename, value)
        (arm_output_dir / "prompt.txt").parent.mkdir(parents=True, exist_ok=True)
        (arm_output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    progress.update(f"{arm_name}: {'complete' if success else 'partial'}",
                    completed_arms=completed + (1 if success else 0),
                    tokens=token_generation_offset + total_pipeline_tokens,
                    details={"result_id": stable_id, "error": error})
    return result


def _resumable_arm_checkpoint(saved: dict[str, Any], arm_name: str,
                              expected_input_identity: str) -> bool:
    """Reuse only a complete atomic arm result for the exact frozen inputs."""
    if not isinstance(saved, dict):
        return False
    prompt = saved.get("prompt")
    if (saved.get("status") != "SUCCESS" or saved.get("arm") != arm_name
            or saved.get("input_identity") != expected_input_identity
            or not isinstance(prompt, str) or not prompt
            or saved.get("prompt_sha256") != sha256_bytes(prompt.encode("utf-8"))
            or saved.get("stable_result_id") != stable_result_identity(expected_input_identity, prompt)
            or not isinstance(saved.get("raw_generated_response"), str)
            or not saved.get("raw_generated_response")
            or saved.get("parser_status") is None
            or saved.get("result_errors")
            or not saved.get("retrieval_candidates_and_ranking_trace", {}).get("trace_counters")):
        return False
    required_stages = {
        "preprocessing", "lexical_retrieval", "dense_retrieval", "cross_encoder",
        "graph_retrieval", "graph_expansion", "context_construction", "generation",
        "answerability_parser",
    }
    stages = saved.get("stage_status", {})
    if not required_stages.issubset(stages) or any(stages.get(key) != "complete" for key in required_stages):
        return False
    telemetry = saved.get("token_telemetry")
    generation_events = saved.get("generation_events")
    return bool(
        isinstance(telemetry, list) and telemetry
        and isinstance(generation_events, list) and generation_events
        and all(isinstance(item, dict) and str(item.get("telemetry_device", "")).startswith("cuda")
                for item in telemetry)
        and all(isinstance(item, dict) and isinstance(item.get("token_telemetry"), list)
                and item.get("token_telemetry")
                and all(isinstance(token, dict)
                        and str(token.get("telemetry_device", "")).startswith("cuda")
                        for token in item["token_telemetry"])
                for item in generation_events)
    )


def _preserve_stale_arm_checkpoint(run_dir: Path, arm_name: str) -> None:
    """Move incomplete or mismatched arm artifacts aside before a retry."""
    timestamp = time.time_ns()
    candidates = [run_dir / "arms" / f"{arm_name}.json", run_dir / "arms" / arm_name]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return
    destination = run_dir / "superseded_checkpoints" / str(timestamp)
    destination.mkdir(parents=True, exist_ok=False)
    for source in existing:
        shutil.move(str(source), str(destination / source.name))


def run_smoke(image_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    start_all = time.monotonic()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    release_info = validate_release()
    progress = Progress(LOG_ROOT / "smoke-progress.jsonl")
    progress.update("input package verified")
    cuda_probe = _actual_cuda_probe()
    os.environ["ABLA_CUDA_REQUIRE_DEVICE"] = "1"
    os.environ.pop("RAG_GRAPH_RETRIEVAL", None)
    os.environ.pop("SCIENCE_GRAPH_USE_CLOUD", None)
    model_manifest_path = MODEL_ROOT / "installed_models.json"
    if not model_manifest_path.is_file():
        raise RuntimeError("Pinned model assets are not installed; run `ablation install` or `smoke`")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    db_path, db_identity = _prepare_database(release_info)
    runtime_config = _write_private_config(RELEASE_ROOT / "effective_config.yaml", db_path,
                                          model_manifest, Path(os.environ.get("HOME", "/state/home")))
    _set_spacy_pythonpath(MODEL_ROOT)

    from src.config import config
    from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
    from src.vector_search import EmbeddingEngine
    from src.llm_engine import LLMEngine
    from src.services.rag_service import RAGService

    config.data = copy.deepcopy(runtime_config)
    # Avoid accidental environment or profile fallback to the developer's config.
    if Path(config.db_path).resolve() != db_path.resolve():
        raise RuntimeError("Resolved retrieval database path differs from the restored snapshot")
    graph_repo = SQLiteGraphRepository(str(db_path), read_only=True)
    vector_repo = SQLiteVectorRepository(str(db_path), read_only=True)
    embedder = EmbeddingEngine()
    generator = LLMEngine(use_cloud=False, purpose="rag")
    service = RAGService(graph_repo, vector_repo, embedder, generator,
                         warmup=False, trace_dir=LOG_ROOT / "traces")
    question = release_info["selected_query"]["question"]
    if question != QUESTION_TEXT:
        raise RuntimeError("Frozen question text disagrees with the smoke fixture")
    progress.update("CUDA model placement preflight")
    device_preflight: dict[str, Any] = {"torch_cuda_operation": cuda_probe}
    _instrument_service(service, device_preflight, question)
    service.current_trace = None

    version_manifest = _version_manifest()
    image_identity = image_identity or {
        "container_image_id": os.environ.get("ABLA_IMAGE_ID", "unrecorded"),
        "container_oci_digest": os.environ.get("ABLA_IMAGE_DIGEST", "unrecorded"),
        "base_image": "python:3.12.13-slim-bookworm",
        "base_image_manifest_digest": os.environ.get("ABLA_BASE_IMAGE_DIGEST", "unrecorded"),
    }
    source_identity = {
        "candidate": "external_db", "snapshot_id": release_info["manifest"]["database"]["snapshot_id"],
        "snapshot_sha256": release_info["manifest"]["database"]["snapshot_sha256"],
        "restored_database_sha256_at_start": db_identity["restored_database_sha256"],
        "logical_fingerprint": db_identity["logical_fingerprint"],
        "selected_source_july_identity_proven": False,
    }
    code_sha256 = sha256_bytes(canonical_json({
        item["path"]: item["sha256"] for item in release_info["manifest"]["expected_files"]
        if item["path"].startswith("code/")
    }))
    run_id = sha256_bytes(canonical_json({
        "query_id": QUESTION_ID, "question_sha256": release_info["manifest"]["dataset"]["questions_sha256"],
        "database_sha256": source_identity["snapshot_sha256"],
        "input_package_manifest_sha256": sha256_file(RELEASE_ROOT / "PACKAGE_MANIFEST.json"),
        "model_asset_manifest_sha256": release_info["manifest"]["model_asset_manifest_sha256"],
        "source_code_sha256": code_sha256,
        "image_identity": image_identity,
    }))[:20]
    run_dir = OUTPUT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    records = load_frozen_dataset(RELEASE_ROOT / "dataset")
    query_row = select_frozen_query(records, QUESTION_ID)
    arms = []
    arm_errors = []
    full = _run_arm(arm_name="B6", component_override=None, query_row=query_row,
                    runtime_config=runtime_config, service=service, run_dir=run_dir,
                    progress=progress, completed=0, model_manifest=model_manifest,
                    db_identity=db_identity, source_identity=source_identity,
                    image_identity=image_identity, code_sha256=code_sha256)
    arms.append(full)
    selected_intervention = None
    intervention_selection_rule = None
    intervention_selection_reason = None
    intervention_codepath_verified = False
    if full["status"] != "SUCCESS":
        arm_errors.extend(full.get("result_errors", []))
        status = "BLOCKED"
    else:
        ranking = full["retrieval_candidates_and_ranking_trace"]["ranking"]
        # Predeclared trace-only selection rule: prefer score_blending when it
        # actually blended two or more CE/RRF candidates. If not, fall back to
        # graph_expansion only when B6 inserted a nonempty dynamic graph block.
        score_eligible = (ranking.get("stage_status") == "complete" and
                          ranking.get("score_blending_enabled") is True and
                          ranking.get("score_blending_branch_executed") is True and
                          len(ranking.get("candidate_ids_before_reranker", [])) >= 2 and
                          len(ranking.get("cross_encoder_scores", [])) >= 2 and
                          len(ranking.get("rrf_scores_before_reranker", [])) >= 2)
        assembly = full["retrieval_candidates_and_ranking_trace"].get("context_and_prompt_assembly", {})
        graph_eligible = (
            full["effective_configuration"].get("baseline_components", {}).get("graph_expansion") is True
            and assembly.get("dynamic_graph_expansion_executed") is True
            and assembly.get("dynamic_graph_output_inserted") is True
        )
        if score_eligible:
            selected_intervention = "score_blending"
            intervention_selection_rule = "active Cross-Encoder/RRF blend over at least two candidates"
            intervention_selection_reason = "The B6 trace proves the normalized Cross-Encoder/RRF blend executed on at least two candidates."
        elif graph_eligible:
            selected_intervention = "graph_expansion"
            intervention_selection_rule = "fallback: active dynamic graph expansion inserted into the B6 prompt"
            intervention_selection_reason = "Score blending failed its trace eligibility rule; B6 inserted a nonempty dynamic graph-expansion block, so this downstream component is independently testable."
        else:
            status = "BLOCKED"
            intervention_selection_rule = "score_blending first; graph_expansion fallback only when its output was inserted"
            arm_errors.append("Neither the score_blending stage nor the predeclared graph_expansion fallback met its trace eligibility rule")
        if selected_intervention is not None:
            off = _run_arm(arm_name=f"no_{selected_intervention}",
                           component_override=selected_intervention,
                           query_row=query_row, runtime_config=runtime_config,
                           service=service, run_dir=run_dir, progress=progress,
                           completed=1, model_manifest=model_manifest,
                           db_identity=db_identity, source_identity=source_identity,
                           image_identity=image_identity, code_sha256=code_sha256)
            arms.append(off)
            if off["status"] != "SUCCESS":
                arm_errors.extend(off.get("result_errors", []))
                status = "PARTIAL"
            else:
                isolated = component_isolated(full["effective_configuration"],
                                              off["effective_configuration"],
                                              selected_intervention)
                off_trace = off["retrieval_candidates_and_ranking_trace"]["ranking"]
                raw_rank_equal = all(
                    ranking.get(key) == off_trace.get(key)
                    for key in ("candidate_ids_before_reranker", "rrf_scores_before_reranker",
                                "cross_encoder_scores")
                )
                if selected_intervention == "score_blending":
                    codepath = (off_trace.get("score_blending_enabled") is False and
                                off_trace.get("ranking_formula") == "raw_cross_encoder_score" and
                                off_trace.get("score_blending_branch_executed") is True)
                else:
                    off_assembly = off["retrieval_candidates_and_ranking_trace"].get("context_and_prompt_assembly", {})
                    codepath = (
                        full["effective_configuration"]["baseline_components"].get("graph_expansion") is True
                        and off["effective_configuration"]["baseline_components"].get("graph_expansion") is False
                        and assembly.get("dynamic_graph_expansion_executed") is True
                        and assembly.get("dynamic_graph_output_inserted") is True
                        and off_assembly.get("dynamic_graph_expansion_executed") is False
                        and off_assembly.get("dynamic_graph_output_inserted") is False
                    )
                intervention_codepath_verified = bool(isolated and codepath and raw_rank_equal)
                if not intervention_codepath_verified:
                    status = "BLOCKED"
                    arm_errors.append("Two-arm config, raw-stage trace, or intervention code-path isolation validation failed")
                else:
                    status = "SUCCESS"

    progress.update("finalizing run archive inputs", completed_arms=sum(arm["status"] == "SUCCESS" for arm in arms),
                    tokens=sum(arm.get("total_pipeline_generated_tokens", 0) for arm in arms))
    post_signature = _logical_signature(db_path)
    post_db_sha256 = sha256_file(db_path)
    if post_db_sha256 != db_identity["snapshot_sha256"]:
        status = "BLOCKED"
        arm_errors.append("Read-only retrieval mutated the frozen database snapshot file")
    if post_signature != db_identity["logical_fingerprint"]:
        status = "BLOCKED"
        arm_errors.append("Database logical fingerprint changed during the two-arm smoke")
    prompt_hashes = [arm.get("prompt_sha256") for arm in arms]
    mlx_comparison = (_compare_mlx_reference(RELEASE_ROOT / "diagnostics/mlx_reference", arms[0])
                      if arms else {"available": False, "reason": "No completed CUDA B6 arm is available"})
    run_manifest = {
        "schema_version": "1.0", "status": status,
        "run_id": run_id, "query_id": QUESTION_ID,
        "question": query_row["question"],
        "answerability_label": query_row["answerability_label"],
        "gold_pmids": query_row.get("original_pmids", []),
        "input_package_manifest_sha256": sha256_file(RELEASE_ROOT / "PACKAGE_MANIFEST.json"),
        "dataset": {"row_count": 375, "answerable_count": 300, "designated_negative_count": 75,
                    "sha256": release_info["manifest"]["dataset"]["questions_sha256"]},
        "database": {**source_identity, "after_run_logical_fingerprint": post_signature,
                      "after_run_file_sha256": post_db_sha256,
                      "identity_same_across_arms": True},
        "models": model_manifest,
        "device_preflight": device_preflight,
        "software_versions": version_manifest,
        "image_identity": image_identity,
        "source_code_sha256": code_sha256,
        "arms": [{"arm": arm["arm"], "status": arm["status"],
                  "stable_result_id": arm["stable_result_id"],
                  "answer": arm["parsed_answer"], "parser_status": arm["parser_status"],
                  "prompt_sha256": arm["prompt_sha256"],
                  "pipeline_generation_call_count": len(arm.get("generation_events", [])),
                  "pipeline_generated_token_count": arm.get("total_pipeline_generated_tokens", 0)} for arm in arms],
        "pipeline_generation_call_count": sum(len(arm.get("generation_events", [])) for arm in arms),
        "pipeline_generated_token_count": sum(arm.get("total_pipeline_generated_tokens", 0) for arm in arms),
        "intervention": {"component": selected_intervention,
                         "trace_rule_passed": len(arms) == 2,
                         "selection_rule": intervention_selection_rule,
                         "selection_reason": intervention_selection_reason,
                         "code_path_verified": intervention_codepath_verified,
                         "same_prompt_hash": len(set(prompt_hashes)) == 1 if len(prompt_hashes) == 2 else None,
                         "same_underlying_generator_and_decode": True,
                         "result_effect": "observed; equal answers or contexts are valid null effects" if len(arms) == 2 else None},
        "mlx_reference_comparison": mlx_comparison,
        "elapsed_seconds": round(time.monotonic() - start_all, 3),
        "errors": arm_errors,
        "judge": "not_run",
        "full_matrix_started": False,
    }
    _atomic_json(run_dir / "run_manifest.json", run_manifest)
    _atomic_json(run_dir / "model_index_manifest.json", {
        "models": model_manifest, "database": db_identity,
        "input_release_manifest_sha256": sha256_file(RELEASE_ROOT / "PACKAGE_MANIFEST.json"),
        "model_lock_sha256": sha256_file(RELEASE_ROOT / "models/model_assets.lock.yaml"),
        "runtime_versions": version_manifest,
    })
    with (run_dir / "judge_ready_inputs.jsonl").open("w", encoding="utf-8") as handle:
        for arm in arms:
            judge_row = {"query_id": arm["query_id"], "question": arm["question"],
                         "gold_reference_answer": arm["gold_reference_answer"],
                         "answerability_label": query_row["answerability_label"],
                         "arm": arm["arm"], "retrieved_evidence": arm["selected_evidence"],
                         "raw_generated_response": arm["raw_generated_response"],
                         "parsed_answer": arm["parsed_answer"]}
            handle.write(json.dumps(judge_row, ensure_ascii=False, sort_keys=True) + "\n")
    (run_dir / "stage_status.json").write_text(json.dumps(
        {arm["arm"]: arm["stage_status"] for arm in arms}, ensure_ascii=False,
        sort_keys=True, indent=2) + "\n", encoding="utf-8")
    progress.finish()
    return {"run_directory": str(run_dir), "run_manifest": run_manifest}
