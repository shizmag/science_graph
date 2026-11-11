"""CPU-first command interface for the corrected R1 MLX experiment.

The commands in this module deliberately do not import ``runtime``.  They
freeze and validate the experiment identity and its read-only evidence before
the separate smoke/run commands are allowed to load MLX.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

from ablation_portability.dataset.loader import load_frozen_dataset

from .checkpoint import CheckpointError, DurableResultStore
from .census import build_census_report, validate_census_output_path, write_census_report
from .cohort import CohortRow, prepare_cohort, select_smoke_rows
from .evidence import EvidenceError, ReadOnlyEvidenceStore
from .identity import (
    DATABASE_SHA256,
    DATABASE_SNAPSHOT_ID,
    DATASET_SHA256,
    HISTORICAL_MODEL_REVISION,
    canonical_json,
    file_sha256,
    load_manifest,
    model_identity,
    object_sha256,
    require_manifest,
    runtime_identity,
    write_manifest,
)
from .protocol import (
    ARM_OFF,
    ARM_ON,
    COHORT_SEED,
    EXPERIMENT_ID,
    INPUT_TOKEN_LIMIT,
    MLX_LM_VERSION,
    MLX_VERSION,
    OUTPUT_TOKEN_LIMIT,
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    PYTHON_MAJOR_MINOR,
    SMOKE_GENERATION_BUDGET,
    SMOKE_QUERY_IDS,
    build_prompt,
)
from .telemetry import (
    TELEMETRY_AGGREGATION,
    TELEMETRY_IMPLEMENTATION,
    TELEMETRY_PROTOCOL_VERSION,
    TELEMETRY_SOURCE,
    TELEMETRY_TOP_KS,
    TELEMETRY_UNITS,
    TelemetryCompatibilityError,
    validate_telemetry_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/Volumes/happy-disk/models/llm/OCC-RAG-1.7B")
DEFAULT_DATASET = REPO_ROOT / "ablation_portability/dataset"
# The selected immutable SQLite payload is intentionally outside the clean
# worktree.  The worktree contains its snapshot metadata and validation
# fixtures, but not a second mutable copy of the 185 MB database.
DEFAULT_DATABASE = Path(
    "/Users/vladimirkasterin/python/graph/ablation_portability/"
    "snapshots/candidates/external_db/v1/database.db"
)
DEFAULT_SNAPSHOT_MANIFEST = REPO_ROOT / (
    "ablation_portability/snapshots/candidates/external_db/v1/snapshot_manifest.json"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "/Volumes/happy-disk/codex-worktrees/r1-mlx-structured-context/"
    "revision_round1/experiments/r1_mlx_corrected"
)


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _existing_disk_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _json_dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _args_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    return (
        _path(args.model),
        _path(args.dataset),
        _path(args.database),
        _path(args.artifact_root),
    )


def _runtime_artifact_root(args: argparse.Namespace) -> Path:
    value = getattr(args, "runtime_artifact_root", None)
    if not value:
        raise ValueError(
            "An explicit --runtime-artifact-root is required for runtime execution"
        )
    return _path(value)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("mlx", "mlx-lm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _current_pinned_runtime_identity() -> dict[str, Any]:
    """Return the metadata-only identity allowed to execute the MLX run.

    This helper intentionally lives in the CPU command layer.  It uses package
    metadata but never imports ``mlx`` or ``mlx_lm``; the latter is imported
    only by :func:`_new_runtime_adapter`, after the full frozen-input guard.
    """
    versions = _package_versions()
    if versions != {"mlx": MLX_VERSION, "mlx-lm": MLX_LM_VERSION}:
        raise ValueError(
            "Expected pinned MLX runtime "
            f"mlx=={MLX_VERSION}, mlx-lm=={MLX_LM_VERSION}; got {versions}"
        )
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        raise ValueError(
            "The corrected local experiment requires Apple Silicon Darwin; "
            f"got {platform.system()} {platform.machine()}"
        )
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != PYTHON_MAJOR_MINOR:
        raise ValueError(
            "The corrected local experiment requires CPython "
            f"{PYTHON_MAJOR_MINOR[0]}.{PYTHON_MAJOR_MINOR[1]}; "
            f"got {platform.python_implementation()} {sys.version_info.major}.{sys.version_info.minor}"
        )
    return runtime_identity(
        mlx_version=versions["mlx"],
        mlx_lm_version=versions["mlx-lm"],
    )


def _load_evidence(args: argparse.Namespace) -> tuple[list[dict[str, Any]], ReadOnlyEvidenceStore]:
    _, dataset_path, database_path, _ = _args_paths(args)
    records = load_frozen_dataset(dataset_path)
    evidence = ReadOnlyEvidenceStore(
        database_path,
        expected_sha256=DATABASE_SHA256,
        snapshot_id=DATABASE_SNAPSHOT_ID,
    )
    return records, evidence


def _manifest_path(artifact_root: Path) -> Path:
    return artifact_root / "experiment_manifest.json"


def _cohort_path(artifact_root: Path) -> Path:
    return artifact_root / "cohort_manifest.json"


def _model_file_summary(model_path: Path) -> dict[str, Any]:
    required = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    files: dict[str, Any] = {}
    for name in required:
        path = model_path / name
        files[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return files


def _verified_model_identity(model_path: Path) -> dict[str, Any]:
    identity = model_identity(model_path, model_revision=HISTORICAL_MODEL_REVISION)
    if not identity["weight_matches_historical"]:
        raise ValueError("Local OCC-RAG weights do not match the archived R1 model SHA-256")
    return identity


def _database_metadata(path: Path) -> dict[str, Any]:
    snapshot_path = DEFAULT_SNAPSHOT_MANIFEST.resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    selected_sha = (
        snapshot.get("sha256")
        or snapshot.get("database_sha256")
        or snapshot.get("selected_database_sha256")
        or snapshot.get("baseline_selection", {}).get("selected_database_sha256")
    )
    if selected_sha != DATABASE_SHA256:
        raise ValueError("Selected database snapshot manifest SHA does not match the frozen identity")
    if file_sha256(path) != DATABASE_SHA256:
        raise ValueError("Selected database file SHA-256 does not match the frozen identity")
    return snapshot


def _validate_dataset_identity(dataset_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if file_sha256(dataset_path / "questions.jsonl") != DATASET_SHA256:
        raise ValueError("Frozen questions SHA-256 does not match the R1 cohort identity")
    if len(records) != 375:
        raise ValueError(f"Expected 375 frozen questions; found {len(records)}")
    answerable = sum(record.get("answerability_label") == "answerable" for record in records)
    negatives = sum(record.get("answerability_label") == "designated_negative" for record in records)
    if (answerable, negatives) != (300, 75):
        raise ValueError(f"Unexpected frozen answerability counts: {answerable} answerable, {negatives} negatives")


def _cohort_payload(rows: Sequence[CohortRow], *, manifest_hash: str | None = None) -> dict[str, Any]:
    values = [row.as_dict() for row in rows]
    payload: dict[str, Any] = {
        "schema_version": "r1-cohort-1",
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "seed": COHORT_SEED,
        "rows": values,
        "eligible_count": len(values),
        "activation_positive_count": sum(row.activation_positive for row in rows),
        "activation_negative_count": sum(not row.activation_positive for row in rows),
    }
    if manifest_hash is not None:
        payload["experiment_manifest_hash"] = manifest_hash
    return payload


def _load_cohort(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"Malformed cohort manifest: {path}")
    expected = object_sha256({key: value for key, value in payload.items() if key != "cohort_hash"})
    if payload.get("cohort_hash") != expected:
        raise ValueError(f"Cohort manifest hash mismatch: {path}")
    return payload


def _write_cohort(path: Path, payload: dict[str, Any]) -> str:
    body = dict(payload)
    body["cohort_hash"] = object_sha256({key: value for key, value in body.items() if key != "cohort_hash"})
    # Use the package writer's fsync/replace path by writing a valid cohort
    # shape through the canonical atomic helper exposed by cohort.py.
    from ablation_cuda.runtime.checkpoints import atomic_write_json

    atomic_write_json(path, body)
    _load_cohort(path)
    return str(body["cohort_hash"])


def _build_manifest(args: argparse.Namespace, *, model: dict[str, Any], cohort_hash: str) -> dict[str, Any]:
    model_path, dataset_path, database_path, artifact_root = _args_paths(args)
    return {
        "schema_version": "r1-mlx-corrected-1",
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "arms": [ARM_ON, ARM_OFF],
        "intervention": "static_graph_relations_only",
        "input_token_limit": INPUT_TOKEN_LIMIT,
        "output_token_limit": OUTPUT_TOKEN_LIMIT,
        "smoke_generation_budget": 12,
        "cohort_seed": COHORT_SEED,
        "model_path": str(model_path),
        "model": model,
        "database_path": str(database_path),
        "database_snapshot_id": DATABASE_SNAPSHOT_ID,
        "database_sha256": DATABASE_SHA256,
        "dataset_path": str(dataset_path),
        "dataset_sha256": DATASET_SHA256,
        "cohort_path": str(_cohort_path(artifact_root)),
        "cohort_hash": cohort_hash,
        "runtime": runtime_identity(mlx_version=None, mlx_lm_version=None),
        "network_access": "forbidden",
        "external_judging": "not_run",
        "historical_cuda_archive": "immutable; not merged",
        "model_config_disclosure": (
            "Local config hash is recorded in model.config_sha256 and differs "
            "from the October lock; local weights match the archived weight hash."
        ),
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    model, dataset, database, artifact = _args_paths(args)
    paths = {
        "model": {"path": str(model), "exists": model.is_dir(), "files": _model_file_summary(model)},
        "dataset": {"path": str(dataset), "exists": dataset.is_dir()},
        "database": {"path": str(database), "exists": database.is_file(), "bytes": database.stat().st_size if database.is_file() else None},
        "artifact_root": {"path": str(artifact), "exists": artifact.exists()},
    }
    package_versions: dict[str, str | None] = {}
    for package in ("mlx", "mlx-lm", "pytest"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    payload = {
        "command": "doctor",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "disk_free_bytes": shutil.disk_usage(_existing_disk_parent(artifact)).free,
        "mlx_imported": "mlx" in sys.modules or "mlx_lm" in sys.modules,
        "package_versions": package_versions,
        "paths": paths,
    }
    _json_dump(payload)
    return 0


def _preflight_payload(args: argparse.Namespace, *, require_artifacts: bool) -> dict[str, Any]:
    model_path, dataset_path, database_path, artifact_root = _args_paths(args)
    records = load_frozen_dataset(dataset_path)
    _validate_dataset_identity(dataset_path, records)
    snapshot = _database_metadata(database_path)
    evidence = ReadOnlyEvidenceStore(database_path, expected_sha256=DATABASE_SHA256, snapshot_id=DATABASE_SNAPSHOT_ID)
    census = evidence.population_evidence(records)
    model_files = _model_file_summary(model_path)
    required_present = all(item["exists"] for item in model_files.values())
    verified_model = _verified_model_identity(model_path) if required_present else None
    manifest_path = _manifest_path(artifact_root)
    cohort_path = _cohort_path(artifact_root)
    artifacts = {
        "manifest": {"path": str(manifest_path), "exists": manifest_path.is_file()},
        "cohort": {"path": str(cohort_path), "exists": cohort_path.is_file()},
    }
    if require_artifacts and not all(item["exists"] for item in artifacts.values()):
        raise ValueError("Frozen experiment manifest/cohort is missing; run prepare first")
    positives = [row for row in census if row.get("activation_positive")]
    negatives = [row for row in census if not row.get("activation_positive")]
    return {
        "command": "preflight",
        "model": {"path": str(model_path), "all_required_files_present": required_present, "identity": verified_model, "files": model_files},
        "dataset": {"path": str(dataset_path), "sha256": DATASET_SHA256, "rows": len(records), "answerable": sum(r.get("answerability_label") == "answerable" for r in records), "designated_negative": sum(r.get("answerability_label") == "designated_negative" for r in records)},
        "database": {"path": str(database_path), "snapshot_id": DATABASE_SNAPSHOT_ID, "sha256": DATABASE_SHA256, "schema_counts": snapshot.get("logical_corpus", {}), "activation_positive": len(positives), "activation_negative": len(negatives)},
        "canary_positive": positives[0] if positives else None,
        "canary_negative": negatives[0] if negatives else None,
        "artifacts": artifacts,
        "mlx_imported": "mlx" in sys.modules or "mlx_lm" in sys.modules,
    }


def cmd_preflight(args: argparse.Namespace) -> int:
    payload = _preflight_payload(args, require_artifacts=args.require_artifacts)
    _json_dump(payload)
    return 0 if payload["model"]["all_required_files_present"] else 2


def cmd_prepare(args: argparse.Namespace) -> int:
    model_path, dataset_path, database_path, artifact_root = _args_paths(args)
    if _manifest_path(artifact_root).exists() or _cohort_path(artifact_root).exists():
        raise ValueError(
            f"Refusing to overwrite existing preparation namespace: {artifact_root}"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    records = load_frozen_dataset(dataset_path)
    _validate_dataset_identity(dataset_path, records)
    _database_metadata(database_path)
    evidence = ReadOnlyEvidenceStore(database_path, expected_sha256=DATABASE_SHA256, snapshot_id=DATABASE_SNAPSHOT_ID)
    rows, cohort_meta = prepare_cohort(records, evidence)
    answerable_only = bool(getattr(args, "answerable_only", False))
    selected_rows = [row for row in rows if row.answerability_label == "answerable"] if answerable_only else rows
    selected = select_smoke_rows(rows) if not answerable_only else []
    cohort_payload = _cohort_payload(selected_rows)
    cohort_payload["population_manifest"] = cohort_meta
    cohort_payload["smoke_query_ids"] = [row.query_id for row in selected]
    cohort_hash = _write_cohort(_cohort_path(artifact_root), cohort_payload)
    # One explicit model identity generation is intentional. Its result is
    # frozen into the manifest so later CPU commands do not rehash weights.
    model = _verified_model_identity(model_path)
    manifest = _build_manifest(args, model=model, cohort_hash=cohort_hash)
    manifest_hash = write_manifest(_manifest_path(artifact_root), manifest)
    _json_dump({"manifest_path": str(_manifest_path(artifact_root)), "manifest_hash": manifest_hash, "cohort_path": str(_cohort_path(artifact_root)), "cohort_hash": cohort_hash, "smoke_query_ids": [row.query_id for row in selected], "population": {"eligible": len(selected_rows), "activation_positive": sum(row.activation_positive for row in selected_rows), "activation_negative": sum(not row.activation_positive for row in selected_rows)}})
    return 0


def _runtime_manifest_path(root: Path) -> Path:
    return root / "experiment_manifest.json"


def _runtime_cohort_path(root: Path) -> Path:
    return root / "cohort_manifest.json"


def _tokenizer_identity(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit tokenizer portion of the frozen model identity."""
    values = {
        key: model.get(key)
        for key in ("tokenizer_sha256", "tokenizer_config_sha256")
    }
    if any(value is None for value in values.values()):
        raise ValueError("Frozen model identity has no complete tokenizer identity")
    return values


def _assert_empty_runtime_root(root: Path) -> None:
    if root == _path(DEFAULT_ARTIFACT_ROOT):
        raise ValueError("Runtime artifact root must differ from the immutable pre-runtime root")
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Refusing to use non-empty runtime artifact root: {root}")


def _load_prepared_namespace(
    source_root: Path,
    *,
    expected_hash: str | None,
    manifest_path: Path | None = None,
    cohort_path: Path | None = None,
    full: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = source_root.expanduser().resolve()
    manifest_path = (manifest_path or _manifest_path(source_root)).expanduser().resolve()
    cohort_path = (cohort_path or _cohort_path(source_root)).expanduser().resolve()
    if manifest_path.parent != source_root or cohort_path.parent != source_root:
        raise ValueError("Prepared manifest and cohort must remain inside the source namespace")
    if manifest_path.name != "experiment_manifest.json" or cohort_path.name != "cohort_manifest.json":
        raise ValueError("Prepared namespace uses unexpected manifest/cohort paths")
    if not manifest_path.is_file() or not cohort_path.is_file():
        raise ValueError("Prepared manifest/cohort files are missing")
    manifest = load_manifest(manifest_path)
    if expected_hash and manifest["manifest_hash"] != expected_hash:
        raise ValueError("Pre-runtime manifest does not match --source-manifest-hash")
    if manifest.get("runtime", {}).get("mlx_version") is not None or manifest.get("runtime", {}).get("mlx_lm_version") is not None:
        raise ValueError("Source namespace is not the expected pre-runtime preparation namespace")
    cohort = _load_cohort(cohort_path)
    if cohort.get("cohort_hash") != manifest.get("cohort_hash"):
        raise ValueError("Pre-runtime manifest/cohort identity mismatch")
    if cohort.get("experiment_manifest_hash") not in (None, manifest["manifest_hash"]):
        raise ValueError("Pre-runtime cohort is linked to another experiment manifest")
    if not full and tuple(cohort.get("smoke_query_ids", ())) != SMOKE_QUERY_IDS:
        raise ValueError("Pre-runtime smoke IDs differ from the frozen outcome-blind cohort")
    row_ids = [str(row.get("query_id")) for row in cohort.get("rows", [])]
    if len(row_ids) != len(set(row_ids)) or (not full and not set(SMOKE_QUERY_IDS).issubset(row_ids)):
        raise ValueError("Pre-runtime cohort is missing or duplicates frozen query IDs")
    return manifest, cohort


def _validate_frozen_inputs(
    *,
    model_path: Path,
    dataset_path: Path,
    database_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], ReadOnlyEvidenceStore, dict[str, Any]]:
    records = load_frozen_dataset(dataset_path)
    _validate_dataset_identity(dataset_path, records)
    _database_metadata(database_path)
    local_model = _verified_model_identity(model_path)
    if local_model != manifest.get("model"):
        raise ValueError("Local model identity differs from prepared manifest")
    if str(model_path) != manifest.get("model_path"):
        raise ValueError("Model path differs from prepared manifest")
    if str(dataset_path) != manifest.get("dataset_path"):
        raise ValueError("Dataset path differs from prepared manifest")
    if str(database_path) != manifest.get("database_path"):
        raise ValueError("Database path differs from prepared manifest")
    if manifest.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("Prepared manifest dataset hash differs from frozen dataset")
    if manifest.get("database_sha256") != DATABASE_SHA256 or manifest.get("database_snapshot_id") != DATABASE_SNAPSHOT_ID:
        raise ValueError("Prepared manifest database identity differs from frozen database")
    evidence = ReadOnlyEvidenceStore(
        database_path,
        expected_sha256=DATABASE_SHA256,
        snapshot_id=DATABASE_SNAPSHOT_ID,
    )
    return records, evidence, local_model


def _bundle_rows(
    records: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    evidence: ReadOnlyEvidenceStore,
    *,
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(record["query_id"]): dict(record) for record in records}
    cohort_rows = cohort.get("rows", [])
    if not isinstance(cohort_rows, list):
        raise ValueError("Malformed prepared cohort rows")
    wanted = list(SMOKE_QUERY_IDS) if smoke else [str(row["query_id"]) for row in cohort_rows]
    if smoke and tuple(cohort.get("smoke_query_ids", ())) != SMOKE_QUERY_IDS:
        raise ValueError("Runtime cohort does not contain the frozen smoke IDs")
    selected_rows: list[dict[str, Any]] = []
    bundles: dict[str, Any] = {}
    prepared_by_id = {str(row["query_id"]): row for row in cohort_rows}
    for query_id in wanted:
        if query_id not in by_id or query_id not in prepared_by_id:
            raise ValueError(f"Frozen cohort query is missing from source data: {query_id}")
        record = by_id[query_id]
        bundle = evidence.evidence_for_record(record)
        prepared = prepared_by_id[query_id]
        prompt_on = build_prompt(bundle, ARM_ON)
        prompt_off = build_prompt(bundle, ARM_OFF)
        activation_positive = prompt_on.trace.activation_status == "positive"
        if bool(prepared.get("activation_positive")) != activation_positive:
            raise ValueError(f"Prepared activation flag contradicts evidence for {query_id}")
        if prompt_on.trace.non_target_fingerprint != prompt_off.trace.non_target_fingerprint:
            raise ValueError(f"Non-target evidence differs across arms for {query_id}")
        if activation_positive and prompt_on.prompt == prompt_off.prompt:
            raise ValueError(f"Activation-positive prompt failed to activate for {query_id}")
        if not activation_positive and prompt_on.prompt != prompt_off.prompt:
            raise ValueError(f"Activation-negative prompt unexpectedly differs for {query_id}")
        selected_rows.append(record)
        bundles[query_id] = bundle
    return selected_rows, bundles


def cmd_freeze_runtime(args: argparse.Namespace) -> int:
    model_path, dataset_path, database_path, source_root = _args_paths(args)
    runtime_root = _runtime_artifact_root(args)
    source_root = source_root.resolve()
    if runtime_root == source_root:
        raise ValueError("Runtime and pre-runtime artifact roots must be distinct")
    _assert_empty_runtime_root(runtime_root)
    source_manifest, source_cohort = _load_prepared_namespace(
        source_root, expected_hash=args.source_manifest_hash, full=args.full
    )
    source_cohort_hash = source_cohort["cohort_hash"]
    if args.full:
        source_cohort = dict(source_cohort)
        source_cohort["rows"] = [row for row in source_cohort["rows"] if row.get("answerability_label") == "answerable"]
        source_cohort.pop("smoke_query_ids", None)
        source_cohort["cohort_hash"] = None
        source_cohort["experiment_manifest_hash"] = source_manifest["manifest_hash"]
    records, evidence, local_model = _validate_frozen_inputs(
        model_path=model_path,
        dataset_path=dataset_path,
        database_path=database_path,
        manifest=source_manifest,
    )
    _, bundles = _bundle_rows(records, source_cohort, evidence, smoke=not args.full)
    del bundles  # Evidence activation proof is the precondition; generation is not run here.
    current_runtime = _current_pinned_runtime_identity()
    root_manifest = {
        "schema_version": "r1-mlx-corrected-runtime-1",
        "execution_id": f"{EXPERIMENT_ID}_mlx_{MLX_VERSION}_{MLX_LM_VERSION}_cp{PYTHON_MAJOR_MINOR[0]}{PYTHON_MAJOR_MINOR[1]}",
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "arms": [ARM_ON, ARM_OFF],
        "intervention": "static_graph_relations_only",
        "decoding": {"temperature": 0.0, "max_new_tokens": OUTPUT_TOKEN_LIMIT, "strategy": "greedy"},
        "input_token_limit": INPUT_TOKEN_LIMIT,
        "output_token_limit": OUTPUT_TOKEN_LIMIT,
        "smoke_generation_budget": 0 if args.full else SMOKE_GENERATION_BUDGET,
        "full_primary_cohort": bool(args.full),
        "cohort_seed": COHORT_SEED,
        "model_path": str(model_path),
        "model": local_model,
        "database_path": str(database_path),
        "database_snapshot_id": DATABASE_SNAPSHOT_ID,
        "database_sha256": DATABASE_SHA256,
        "dataset_path": str(dataset_path),
        "dataset_sha256": DATASET_SHA256,
        "cohort_path": str(_runtime_cohort_path(runtime_root)),
        "cohort_hash": None,
        "source_pre_runtime": {
            "manifest_path": str(_manifest_path(source_root)),
            "manifest_hash": source_manifest["manifest_hash"],
            "cohort_path": str(_cohort_path(source_root)),
            "cohort_hash": source_cohort_hash,
        },
        "runtime": current_runtime,
        "network_access": "forbidden",
        "external_judging": "not_run",
        "historical_cuda_archive": "immutable; not merged",
        "model_config_disclosure": source_manifest.get("model_config_disclosure"),
        "entropy_telemetry": _telemetry_manifest_contract(),
    }
    # Complete validation precedes writes. The root must still be empty: a
    # failure during a previous freeze is never silently resumed or overwritten.
    runtime_root.mkdir(parents=True, exist_ok=True)
    cohort_hash = _write_cohort(_runtime_cohort_path(runtime_root), dict(source_cohort))
    root_manifest["cohort_hash"] = cohort_hash
    manifest_hash = write_manifest(_runtime_manifest_path(runtime_root), root_manifest)
    _json_dump({
        "runtime_root": str(runtime_root),
        "manifest_path": str(_runtime_manifest_path(runtime_root)),
        "manifest_hash": manifest_hash,
        "cohort_path": str(_runtime_cohort_path(runtime_root)),
        "cohort_hash": cohort_hash,
        "planned_slots": len(source_cohort.get("rows", [])) * 2,
        "full_primary_cohort": bool(args.full),
        "runtime": current_runtime,
        "generation_started": False,
    })
    return 0


def _validate_runtime_namespace(
    args: argparse.Namespace,
    *,
    smoke: bool,
    allow_legacy_telemetry: bool = False,
    validate_results: bool = True,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    model_path, dataset_path, database_path, _ = _args_paths(args)
    root = _runtime_artifact_root(args)
    if root == _path(DEFAULT_ARTIFACT_ROOT):
        raise ValueError("The pre-runtime namespace cannot be used for inference")
    manifest_path = _runtime_manifest_path(root)
    manifest = load_manifest(manifest_path)
    if args.manifest_hash and manifest["manifest_hash"] != args.manifest_hash:
        raise ValueError("Runtime manifest does not match --manifest-hash")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("mlx_version") is None or runtime.get("mlx_lm_version") is None:
        raise ValueError("Runtime manifest has no frozen MLX package identity")
    current_runtime = _current_pinned_runtime_identity()
    if runtime != current_runtime:
        raise ValueError("Current MLX/Python/hardware runtime identity differs from frozen manifest")
    if manifest.get("schema_version") != "r1-mlx-corrected-runtime-1":
        raise ValueError("Unsupported or non-executable runtime manifest schema")
    if manifest.get("protocol_version") != PROTOCOL_VERSION or manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Runtime manifest protocol/prompt version does not match this code")
    declared_telemetry = manifest.get("entropy_telemetry")
    legacy_telemetry = {
        "status": "unavailable_from_mlx_lm_generate_distribution_api",
        "definitions": ["H20", "H50", "H100"],
    }
    legacy_read_only = allow_legacy_telemetry and declared_telemetry == legacy_telemetry
    if declared_telemetry != _telemetry_manifest_contract() and not legacy_read_only:
        raise ValueError("Runtime manifest predictive-distribution telemetry contract is incompatible")
    cohort_path = _runtime_cohort_path(root)
    declared_cohort_path = manifest.get("cohort_path")
    if not isinstance(declared_cohort_path, str) or _path(declared_cohort_path) != cohort_path.resolve():
        raise ValueError("Runtime manifest cohort_path is not bound to its runtime namespace")
    cohort = _load_cohort(cohort_path)
    if cohort.get("cohort_hash") != manifest.get("cohort_hash"):
        raise ValueError("Runtime manifest/cohort hashes do not match")
    source = manifest.get("source_pre_runtime", {})
    if not isinstance(source, dict):
        raise ValueError("Runtime manifest has no source preparation link")
    source_root = _path(args.artifact_root)
    declared_source_manifest = source.get("manifest_path")
    declared_source_cohort = source.get("cohort_path")
    if not isinstance(declared_source_manifest, str) or not isinstance(declared_source_cohort, str):
        raise ValueError("Runtime manifest source link is incomplete")
    source_manifest_path = _path(declared_source_manifest)
    source_cohort_path = _path(declared_source_cohort)
    if source_manifest_path != _manifest_path(source_root).resolve():
        raise ValueError("Runtime source manifest is not the selected immutable preparation")
    if source_cohort_path != _cohort_path(source_root).resolve():
        raise ValueError("Runtime source cohort is not the selected immutable preparation")
    source_manifest, source_cohort = _load_prepared_namespace(
        source_root,
        expected_hash=source.get("manifest_hash"),
        manifest_path=source_manifest_path,
        cohort_path=source_cohort_path,
    )
    if source.get("manifest_hash") != source_manifest.get("manifest_hash"):
        raise ValueError("Runtime source manifest hash is not self-consistent")
    if source.get("cohort_hash") != source_cohort.get("cohort_hash"):
        raise ValueError("Runtime source cohort hash is not self-consistent")
    for key in (
        "model_path",
        "dataset_path",
        "database_path",
        "dataset_sha256",
        "database_sha256",
        "database_snapshot_id",
        "model",
    ):
        if source_manifest.get(key) != manifest.get(key):
            raise ValueError(f"Runtime manifest differs from linked source field: {key}")
    if manifest.get("full_primary_cohort"):
        source_rows = source_cohort.get("rows", [])
        runtime_rows = cohort.get("rows", [])
        expected_ids = [str(row.get("query_id")) for row in source_rows
                        if row.get("answerability_label") == "answerable"]
        actual_ids = [str(row.get("query_id")) for row in runtime_rows]
        if len(expected_ids) != 300 or len(set(expected_ids)) != 300:
            raise ValueError("Frozen source cohort does not contain exactly 300 unique answerable queries")
        if actual_ids != expected_ids:
            raise ValueError("Runtime cohort is not the deterministic answerable subset of its source cohort")
        if any(row.get("answerability_label") != "answerable" for row in runtime_rows):
            raise ValueError("Runtime cohort contains a non-answerable query")
        if len(actual_ids) != 300 or len(set(actual_ids)) != 300:
            raise ValueError("Runtime cohort must contain exactly 300 unique queries")
    elif source_cohort.get("cohort_hash") != cohort.get("cohort_hash"):
        raise ValueError("Runtime cohort no longer matches its prepared source cohort")
    if not manifest.get("full_primary_cohort") and cohort.get("smoke_query_ids") != list(SMOKE_QUERY_IDS):
        raise ValueError("Runtime cohort smoke IDs do not match frozen protocol IDs")
    records, evidence, local_model = _validate_frozen_inputs(
        model_path=model_path,
        dataset_path=dataset_path,
        database_path=database_path,
        manifest=manifest,
    )
    if local_model != source_manifest.get("model"):
        raise ValueError("Runtime model identity differs from linked pre-runtime manifest")
    rows, bundles = _bundle_rows(records, cohort, evidence, smoke=smoke)
    if not smoke and len(rows) != len(cohort.get("rows", [])):
        raise ValueError("Full cohort has missing or duplicate query identities")
    if validate_results:
        _validate_result_telemetry(
            root,
            manifest,
            allow_legacy=legacy_read_only,
            legacy_manifest=legacy_read_only,
        )
    return root, manifest, cohort, rows, bundles, local_model


def _telemetry_manifest_contract() -> dict[str, Any]:
    return {
        "status": "available_from_generate_step",
        "protocol_version": TELEMETRY_PROTOCOL_VERSION,
        "implementation": TELEMETRY_IMPLEMENTATION,
        "source": TELEMETRY_SOURCE,
        "top_k": list(TELEMETRY_TOP_KS),
        "units": dict(TELEMETRY_UNITS),
        "aggregation": TELEMETRY_AGGREGATION,
        "eos_included_in_aggregation": True,
        "definitions": {
            "h20_bits": "mean local top-20 entropy in bits over all compact generated-token rows, including stop rows",
            "h50_bits": "mean local top-50 entropy in bits over all compact generated-token rows, including stop rows",
            "h100_bits": "mean local top-100 entropy in bits over all compact generated-token rows, including stop rows",
        },
    }


def _validate_result_telemetry(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
    legacy_manifest: bool = False,
) -> dict[str, int]:
    """Validate new payloads; retain explicit legacy unavailable rows unchanged."""

    store = DurableResultStore(
        root,
        _runtime_manifest_path(root),
        str(manifest["manifest_hash"]),
    )
    protocol_counts: dict[str, int] = {}
    for item in store.accepted_rows():
        result = item.get("result")
        if not isinstance(result, Mapping):
            continue
        payload = result.get("telemetry")
        if payload is None:
            # Legacy rows are auditable, but a runtime with the new manifest
            # must not silently continue using a mixture of missing telemetry.
            if result.get("telemetry_source") not in (None, "unavailable"):
                raise ValueError("Accepted result has telemetry source metadata but no payload")
            if not allow_legacy or not legacy_manifest:
                raise ValueError("Legacy unavailable telemetry cannot be continued under the new protocol")
            if result.get("entropy_valid") is not False:
                raise ValueError("Accepted result without predictive telemetry is not explicitly legacy")
            protocol_counts["legacy_unavailable"] = protocol_counts.get("legacy_unavailable", 0) + 1
            continue
        if legacy_manifest:
            raise ValueError("Legacy runtime manifest cannot accept predictive telemetry results")
        if not isinstance(payload, Mapping):
            raise ValueError("Accepted result telemetry must be a mapping")
        try:
            validate_telemetry_payload(payload)
        except TelemetryCompatibilityError as exc:
            raise ValueError(f"Accepted result telemetry is incompatible: {exc}") from exc
        version = str(payload["protocol_version"])
        protocol_counts[version] = protocol_counts.get(version, 0) + 1
        for key, expected in (
            ("telemetry_protocol_version", version),
            ("telemetry_implementation", payload.get("implementation")),
            ("telemetry_source", payload.get("source")),
            ("telemetry_valid", payload.get("valid")),
        ):
            if result.get(key) != expected:
                raise ValueError(f"Accepted result {key} does not match telemetry payload")
    populated_protocols = {key for key in protocol_counts if key != "legacy_unavailable"}
    if len(populated_protocols) > 1:
        raise ValueError("Accepted results mix incompatible predictive-telemetry protocols")
    # The previous smoke's accepted records predate this protocol. They are
    # intentionally preserved but cannot be continued under a new manifest.
    has_legacy_rows = protocol_counts.get("legacy_unavailable", 0) > 0
    if populated_protocols and has_legacy_rows:
        raise ValueError("Runtime accepted rows mix legacy unavailable and predictive telemetry")
    return protocol_counts


def _new_runtime_adapter(model_path: Path):
    # Keep this import below _validate_runtime_namespace: invalid namespaces
    # and over-budget smoke requests must not import MLX or its runtime adapter.
    from .runtime import MLXAdapter

    adapter = MLXAdapter(model_path, output_limit=OUTPUT_TOKEN_LIMIT)
    observed = adapter.require_pinned_environment()
    expected = _current_pinned_runtime_identity()
    if observed != expected:
        raise ValueError("MLX adapter runtime identity changed during execution preflight")
    return adapter


def _execute_runtime(args: argparse.Namespace, *, smoke: bool) -> int:
    requested_budget = args.max_new_generations
    if smoke and requested_budget > SMOKE_GENERATION_BUDGET:
        raise ValueError(f"Smoke generation cap cannot exceed {SMOKE_GENERATION_BUDGET}")
    if requested_budget < 0:
        raise ValueError("Generation budget cannot be negative")
    root, manifest, cohort, rows, bundles, model = _validate_runtime_namespace(args, smoke=smoke)
    manifest_hash = manifest["manifest_hash"]
    manifest_path = _runtime_manifest_path(root)
    store = DurableResultStore(root, manifest_path, manifest_hash)
    states = store.operation_states(
        [str(row["query_id"]) for row in rows],
        [ARM_ON, ARM_OFF],
    )
    pending = [
        key for key, status in states.items()
        if not store.is_complete_operation_status(status)
    ]
    if smoke:
        used = store.generation_calls()
        remaining = max(0, SMOKE_GENERATION_BUDGET - used)
        budget = min(requested_budget, remaining)
    else:
        budget = requested_budget if requested_budget else None
    if not pending:
        _json_dump({
            "runtime_root": str(root),
            "manifest_hash": manifest_hash,
            "smoke": smoke,
            "generation_calls_total": store.generation_calls(),
            "adapter_loaded": False,
            "result": {
                "generated": 0,
                "skipped": sum(status == "SUCCESS" for status in states.values()),
                "terminal": sum(status in {"OVERFLOW", "FAILED_TERMINAL"} for status in states.values()),
                "failed": 0,
                "stopped": 0,
                "budget_exhausted": 0,
            },
        })
        return 0
    if smoke and budget <= 0:
        _json_dump({
            "runtime_root": str(root),
            "manifest_hash": manifest_hash,
            "smoke": True,
            "generation_calls_total": store.generation_calls(),
            "adapter_loaded": False,
            "result": {
                "generated": 0,
                "skipped": sum(status == "SUCCESS" for status in states.values()),
                "terminal": sum(status in {"OVERFLOW", "FAILED_TERMINAL"} for status in states.values()),
                "failed": 0,
                "stopped": 0,
                "budget_exhausted": 1,
            },
        })
        return 0
    if args.stop_after is not None and args.stop_after <= 0:
        _json_dump({
            "runtime_root": str(root),
            "manifest_hash": manifest_hash,
            "smoke": smoke,
            "generation_calls_total": store.generation_calls(),
            "adapter_loaded": False,
            "result": {
                "generated": 0,
                "skipped": 0,
                "terminal": 0,
                "failed": 0,
                "stopped": 1,
                "budget_exhausted": 0,
            },
        })
        return 0
    adapter = _new_runtime_adapter(Path(manifest["model_path"]))
    from .runner import run_pairs

    result = run_pairs(
        rows,
        bundles,
        store=store,
        adapter=adapter,
        model_identity=model,
        tokenizer_identity=_tokenizer_identity(model),
        decoding=manifest["decoding"],
        manifest_hash=manifest_hash,
        max_new_generations=budget,
        stop_after=args.stop_after,
    )
    _json_dump({"runtime_root": str(root), "manifest_hash": manifest_hash, "smoke": smoke, "generation_calls_total": store.generation_calls(), "result": result})
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    return _execute_runtime(args, smoke=True)


def cmd_run(args: argparse.Namespace) -> int:
    return _execute_runtime(args, smoke=False)


def cmd_activation(args: argparse.Namespace) -> int:
    records, evidence = _load_evidence(args)
    census = evidence.population_evidence(records)
    positive = next((row for row in census if row.get("activation_positive")), None)
    negative = next((row for row in census if not row.get("activation_positive")), None)
    result: dict[str, Any] = {
        "population": {
            "eligible": len(census),
            "activation_positive": sum(bool(row.get("activation_positive")) for row in census),
            "activation_negative": sum(not bool(row.get("activation_positive")) for row in census),
            "static_relation_policy": sorted(evidence.STATIC_RELATION_TYPES),
            "excluded_metadata_relation_types": ["HAS_TAG"],
        },
        "positive": positive,
        "negative": negative,
    }
    for label, row in (("positive", positive), ("negative", negative)):
        if row is None:
            continue
        record = next(record for record in records if record["query_id"] == row["query_id"])
        bundle = evidence.evidence_for_record(record)
        on = build_prompt(bundle, ARM_ON)
        off = build_prompt(bundle, ARM_OFF)
        result[label]["on_prompt_sha256"] = on.trace.prompt_sha256
        result[label]["off_prompt_sha256"] = off.trace.prompt_sha256
        result[label]["prompt_differs"] = on.prompt != off.prompt
        result[label]["on_trace"] = asdict(on.trace)
        result[label]["off_trace"] = asdict(off.trace)
    _json_dump(result)
    return 0


def _open_store(args: argparse.Namespace) -> DurableResultStore:
    _, _, _, artifact_root = _args_paths(args)
    if getattr(args, "runtime_artifact_root", None):
        artifact_root = _runtime_artifact_root(args)
    manifest_path = _manifest_path(artifact_root)
    if args.manifest_hash:
        require_manifest(manifest_path, args.manifest_hash)
    manifest = load_manifest(manifest_path)
    return DurableResultStore(artifact_root, manifest_path, manifest["manifest_hash"])


def cmd_status(args: argparse.Namespace) -> int:
    store = _open_store(args)
    progress = store.progress()
    manifest = require_manifest(_runtime_manifest_path(_runtime_artifact_root(args)), args.manifest_hash)
    cohort = _load_cohort(_runtime_cohort_path(_runtime_artifact_root(args)))
    total = len(cohort.get("rows", [])) * 2
    progress.update({"manifest_hash": manifest["manifest_hash"], "preparation_total": total,
                     "generation_total": total, "remaining": total - progress.get("accepted", 0),
                     "completed_pairs": progress.get("accepted", 0) // 2,
                     "state": "complete" if progress.get("accepted", 0) >= total else "in_progress"})
    _json_dump(progress)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    model_path, dataset_path, database_path, artifact_root = _args_paths(args)
    # A runtime root is validated through the guarded execution boundary, not
    # as if it were the nullable-runtime preparation namespace.  This keeps
    # the offline validation command aligned with smoke/run: source links,
    # current runtime identity, immutable evidence and frozen cohort rows are
    # all checked before reporting accumulated results.
    if getattr(args, "runtime_artifact_root", None):
        root, manifest, cohort, rows, bundles, local_model = _validate_runtime_namespace(
            args,
            smoke=False,
            allow_legacy_telemetry=True,
            validate_results=False,
        )
        del cohort, bundles
        manifest_path = _runtime_manifest_path(root)
        store = DurableResultStore(root, manifest_path, manifest["manifest_hash"])
        accepted = store.accepted_rows()
        telemetry_protocol_counts = _validate_result_telemetry(
            root,
            manifest,
            allow_legacy=True,
            legacy_manifest=(manifest.get("entropy_telemetry") != _telemetry_manifest_contract()),
        )
        telemetry_warnings = []
        for item in accepted:
            result = item["result"]
            entropy = result.get("entropy") or {}
            if (
                result.get("entropy_valid") is False
                and entropy.get("source") != "unavailable"
            ):
                telemetry_warnings.append(
                    {
                        "query_id": result.get("query_id"),
                        "arm_id": result.get("arm_id"),
                        "reason": "entropy marked invalid but source label is not unavailable",
                    }
                )
        _json_dump(
            {
                "manifest_hash": manifest["manifest_hash"],
                "cohort_hash": manifest["cohort_hash"],
                "runtime_root": str(root),
                "dataset_rows": len(rows),
                "model_path": str(model_path),
                "model_identity": local_model,
                "progress": store.progress(),
                "accepted": len(accepted),
                "entropy_valid": False,
                "telemetry_warnings": telemetry_warnings,
                "predictive_telemetry_protocol_counts": telemetry_protocol_counts,
                "predictive_telemetry_valid_records": sum(
                    isinstance(item.get("result"), Mapping)
                    and item["result"].get("telemetry_valid") is True
                    for item in accepted
                ),
                "predictive_telemetry_invalid_records": sum(
                    isinstance(item.get("result"), Mapping)
                    and item["result"].get("telemetry_valid") is False
                    for item in accepted
                ),
            }
        )
        return 0

    manifest_path = _manifest_path(artifact_root)
    manifest = load_manifest(manifest_path)
    if args.manifest_hash and manifest["manifest_hash"] != args.manifest_hash:
        raise ValueError("Manifest hash does not match --manifest-hash")
    cohort = _load_cohort(_cohort_path(artifact_root))
    if cohort.get("cohort_hash") != manifest.get("cohort_hash"):
        raise ValueError("Manifest and cohort hashes do not match")
    records = load_frozen_dataset(dataset_path)
    _validate_dataset_identity(dataset_path, records)
    _database_metadata(database_path)
    local_model = _verified_model_identity(model_path)
    if local_model != manifest.get("model"):
        raise ValueError("Local model identity differs from frozen experiment manifest")
    evidence = ReadOnlyEvidenceStore(database_path, expected_sha256=DATABASE_SHA256, snapshot_id=DATABASE_SNAPSHOT_ID)
    census = evidence.population_evidence(records)
    result = {
        "manifest_hash": manifest["manifest_hash"],
        "cohort_hash": cohort["cohort_hash"],
        "dataset_rows": len(records),
        "activation_positive": sum(bool(row.get("activation_positive")) for row in census),
        "activation_negative": sum(not bool(row.get("activation_positive")) for row in census),
        "progress": DurableResultStore(artifact_root, manifest_path, manifest["manifest_hash"]).progress(),
        "model_path": str(model_path),
        "model_identity": local_model,
    }
    _json_dump(result)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    _, _, _, artifact_root = _args_paths(args)
    manifest = load_manifest(_manifest_path(artifact_root))
    store = DurableResultStore(
        artifact_root,
        _manifest_path(artifact_root),
        manifest["manifest_hash"],
    )
    rows = _load_cohort(_cohort_path(artifact_root)).get("rows", [])
    output = _path(args.output) if args.output else artifact_root / "future_judging_export.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    from ablation_cuda.runtime.checkpoints import atomic_write_bytes

    accepted = {
        (item["query_id"], item["arm_id"]): item["result"]
        for item in store.accepted_rows()
    }
    records: list[dict[str, Any]] = []
    for row in rows:
        query_id = row["query_id"]
        for arm_id in (ARM_ON, ARM_OFF):
            record: dict[str, Any] = {"experiment_id": EXPERIMENT_ID, "manifest_hash": manifest["manifest_hash"], "query_id": query_id, "arm_id": arm_id, "status": "MISSING", "response": None}
            payload = accepted.get((query_id, arm_id))
            if payload is not None:
                record.update(
                    {"status": "SUCCESS", "response": payload.get("text"), "result": payload}
                )
            records.append(record)
    data = b"".join((canonical_json(record) + b"\n" for record in records))
    atomic_write_bytes(output, data)
    _json_dump({"path": str(output), "records": len(records), "judging": "not_run"})
    return 0


def cmd_prepare_contexts(args: argparse.Namespace) -> int:
    """Materialize frozen prompts without loading generator weights."""
    from tokenizers import Tokenizer
    from .census import _tokenizer_ids
    from .prepared import make, prepare_all, read

    model_path, dataset_path, database_path, _ = _args_paths(args)
    root = _runtime_artifact_root(args)
    manifest_path = _runtime_manifest_path(root)
    manifest = require_manifest(manifest_path, args.manifest_hash)
    cohort = _load_cohort(_runtime_cohort_path(root))
    if manifest.get("schema_version") != "r1-mlx-corrected-runtime-1" or manifest.get("protocol_version") != PROTOCOL_VERSION or manifest.get("prompt_version") != PROMPT_VERSION or manifest.get("intervention") != "static_graph_relations_only" or manifest.get("arms") != [ARM_ON, ARM_OFF]:
        raise ValueError("Unsupported frozen runtime protocol or intervention")
    if manifest.get("cohort_hash") != cohort.get("cohort_hash"):
        raise ValueError("Frozen manifest/cohort identity mismatch")
    records, evidence, _ = _validate_frozen_inputs(model_path=model_path, dataset_path=dataset_path, database_path=database_path, manifest=manifest)
    by_id = {str(row["query_id"]): row for row in records}
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    bundles: dict[str, Any] = {}

    def build(row: Mapping[str, Any], arm: str):
        query_id = str(row["query_id"])
        if query_id not in bundles:
            if query_id not in by_id:
                raise ValueError(f"Frozen cohort query missing from dataset: {query_id}")
            bundle = evidence.evidence_for_record(by_id[query_id])
            on, off = build_prompt(bundle, ARM_ON), build_prompt(bundle, ARM_OFF)
            if bool(row.get("activation_positive")) != (on.trace.activation_status == "positive") or on.trace.non_target_fingerprint != off.trace.non_target_fingerprint:
                raise ValueError(f"Activation or non-target evidence mismatch for {query_id}")
            bundles[query_id] = (bundle, {ARM_ON: on, ARM_OFF: off})
        bundle, builds = bundles[query_id]
        prompt = builds[arm].prompt
        return make(query_id, arm, prompt, _tokenizer_ids(tokenizer, prompt),
                    manifest_hash=args.manifest_hash, question=bundle.question,
                    evidence_ids=tuple(chunk.chunk_id for chunk in bundle.chunks),
                    activation=bundle.activation_positive, static_relation_count=len(bundle.relations),
                    evidence_identity=object_sha256(asdict(bundle)))

    rows = cohort["rows"]
    items = [(str(row["query_id"]), arm, lambda row=row, arm=arm: build(row, arm))
             for row in rows for arm in (ARM_ON, ARM_OFF)]
    output = root / "prepared_inputs"
    counts = prepare_all(items, output, manifest_path=manifest_path, manifest_hash=args.manifest_hash, cohort=cohort)
    failures = sum(read(output / f"{q}__{arm}.json").status != "prepared" for q, arm, _ in items)
    _json_dump({"prepared": counts["prepared"], "skipped": counts["skipped"], "failed": failures,
                "remaining": failures, "total": len(items), "path": str(output)})
    return 1 if failures else 0

def _load_complete_prepared_inputs(root: Path, manifest: Mapping[str, Any]) -> list[Any]:
    """Validate the immutable generation boundary without constructing context."""
    from .prepared import read
    if manifest.get("arms") != [ARM_ON, ARM_OFF]:
        raise ValueError("Prepared generation requires the frozen ON/OFF arm set")
    cohort = _load_cohort(_runtime_cohort_path(root))
    if cohort.get("cohort_hash") != manifest.get("cohort_hash"):
        raise ValueError("Frozen manifest/cohort identity mismatch")
    rows = cohort.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Frozen cohort has no rows")
    directory = root / "prepared_inputs"
    result: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        query_id = str(row.get("query_id"))
        for arm in (ARM_ON, ARM_OFF):
            path = directory / f"{query_id}__{arm}.json"
            if not path.is_file():
                raise ValueError(f"Missing prepared-input artifact: {path}")
            try:
                item = read(path)
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prepared-input artifact {path}: {exc}") from exc
            if item.manifest_hash != manifest["manifest_hash"] or (item.query_id, item.arm_id) != (query_id, arm):
                raise ValueError(f"Prepared input identity mismatch: {path}")
            if item.status not in ("prepared", "failed"):
                raise ValueError(f"Unsupported prepared input status: {path}")
            key = (item.query_id, item.arm_id)
            if key in seen:
                raise ValueError(f"Duplicate prepared input: {key}")
            seen.add(key)
            result.append(item)
    if len(result) != len(rows) * 2:
        raise ValueError("Prepared-input completeness check failed")
    return result

def cmd_generate_prepared(args: argparse.Namespace) -> int:
    root = _runtime_artifact_root(args)
    manifest_path = _runtime_manifest_path(root)
    manifest = require_manifest(manifest_path, args.manifest_hash)
    if manifest.get("schema_version") != "r1-mlx-corrected-runtime-1":
        raise ValueError("Unsupported runtime manifest schema")
    if manifest.get("protocol_version") != PROTOCOL_VERSION or manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Runtime protocol or prompt version is incompatible")
    if manifest.get("entropy_telemetry", {}).get("protocol_version") != TELEMETRY_PROTOCOL_VERSION:
        raise ValueError("Unsupported telemetry protocol")
    inputs = _load_complete_prepared_inputs(root, manifest)
    store = DurableResultStore(root, manifest_path, args.manifest_hash)
    progress = store.progress()
    total = len(inputs)
    if progress.get("accepted", 0) >= total:
        _json_dump({"stage": "generation", "total": total, "accepted": progress["accepted"], "generation_calls": store.generation_calls(), "model_loaded": False, "result": {"generated": 0, "skipped": total}})
        return 0
    print("R1 MLX | GENERATION | validating prepared inputs: complete", file=sys.stderr, flush=True)
    print("R1 MLX | GENERATION | loading model: started", file=sys.stderr, flush=True)
    started = time.monotonic()
    adapter = _new_runtime_adapter(Path(manifest["model_path"]))
    load_elapsed = time.monotonic() - started
    print(f"R1 MLX | GENERATION | loading model: complete ({load_elapsed:.1f}s)", file=sys.stderr, flush=True)
    from .runner import run_prepared
    completed = progress.get("accepted", 0)
    preexisting = {(row["query_id"], row["arm_id"]) for row in store.accepted_rows()}
    generated_here = 0
    def report(result: dict[str, Any]) -> None:
        nonlocal completed
        nonlocal generated_here
        key = (result.get("query_id"), result.get("arm_id"))
        if key not in preexisting:
            preexisting.add(key)
            completed += 1
            generated_here += 1
        elapsed = time.monotonic() - started
        rate = generated_here / elapsed * 60 if elapsed > 0 else 0.0
        eta = (total - completed) / (generated_here / elapsed) if generated_here and elapsed > 0 else None
        pairs = sum(1 for q in {x.query_id for x in inputs} if all((q, a) in preexisting for a in (ARM_ON, ARM_OFF)))
        eta_text = f"~{int(eta // 60):02d}:{int(eta % 60):02d}" if eta is not None else "calculating..."
        print(f"R1 MLX | GENERATION | {completed}/{total} ({100*completed/total:.1f}%) | pairs={pairs} | accepted={completed} | retryable={store.progress().get('failed_retryable', 0)} | terminal={store.progress().get('failed_terminal', 0)} | remaining={total-completed} | current={result.get('query_id')}/{result.get('arm_id')} | elapsed={int(elapsed//60):02d}:{int(elapsed%60):02d} | throughput={rate:.1f}/min | ETA={eta_text} | checkpoint=persisted", file=sys.stderr, flush=True)
    try:
        result = run_prepared(inputs, store=store, adapter=adapter, model_identity=manifest["model"],
                              tokenizer_identity=_tokenizer_identity(manifest["model"]), decoding=manifest["decoding"],
                              manifest_hash=args.manifest_hash, stop_after=args.stop_after, on_result=report)
    except KeyboardInterrupt:
        current = store.progress()
        print(f"R1 MLX | GENERATION | interrupted | accepted={current.get('accepted', 0)} | remaining={total-current.get('accepted', 0)} | checkpoint=preserved", file=sys.stderr)
        print("Resume command: " + " ".join(sys.argv), file=sys.stderr)
        return 130
    final = store.progress()
    active_elapsed = time.monotonic() - started
    _json_dump({"stage": "generation", "total": total, "progress": final, "result": result, "generation_calls": store.generation_calls(), "model_loaded": True, "active_elapsed_seconds": active_elapsed, "throughput_per_minute": generated_here / active_elapsed * 60 if active_elapsed else 0.0, "checkpointed_before_start": progress.get("accepted", 0)})
    return 0


def cmd_census(args: argparse.Namespace) -> int:
    """Build and atomically write the CPU-only full prompt census."""

    _, dataset_path, database_path, _ = _args_paths(args)
    output = validate_census_output_path(_path(args.output))
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing census report: {output}")
    report = build_census_report(
        dataset_path=dataset_path,
        database_path=database_path,
        tokenizer_path=_path(args.tokenizer),
    )
    report_hash = write_census_report(output, report)
    _json_dump(
        {
            "command": "census",
            "path": str(output),
            "report_sha256": report_hash,
            "counts": report["counts"],
            "distributions": report["distributions"],
            "model_inference": "not_run",
        }
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="r1-mlx")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--database", default=str(DEFAULT_DATABASE))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument(
        "--runtime-artifact-root",
        help="Explicit, new versioned directory for the frozen MLX execution namespace",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--require-artifacts", action="store_true")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--answerable-only", action="store_true")
    prepare_contexts = sub.add_parser("prepare-contexts", help="Persist frozen generator inputs without loading model weights")
    prepare_contexts.add_argument("--manifest-hash", required=True)
    judge_inputs = sub.add_parser("prepare-judge-inputs")
    judge_inputs.add_argument("--manifest-hash", required=True)
    for name in ("run-luna", "run-deepseek"):
        p = sub.add_parser(name); p.add_argument("--manifest-hash", required=True); p.add_argument("--stop-after", type=int); p.add_argument("--max-attempts", type=int, default=2)
    judge_status = sub.add_parser("judge-status"); judge_status.add_argument("--judge", choices=("luna", "deepseek"), required=True)
    judge_export = sub.add_parser("export-judge-scores"); judge_export.add_argument("--judge", choices=("luna", "deepseek"), required=True); judge_export.add_argument("--output", required=True)
    sub.add_parser("activation")
    for name in ("status", "validate", "export"):
        command = sub.add_parser(name)
        command.add_argument("--manifest-hash")
    generation = sub.add_parser("generate-prepared", help="Sequential generation from immutable prepared inputs")
    generation.add_argument("--manifest-hash", required=True)
    generation.add_argument("--stop-after", type=int)
    generation.add_argument("--max-new-slots", type=int, dest="stop_after")
    export = sub.choices["export"]
    export.add_argument("--output")
    census = sub.add_parser("census")
    census.add_argument("--tokenizer", default=str(DEFAULT_MODEL / "tokenizer.json"))
    census.add_argument("--output", required=True)
    freeze_runtime = sub.add_parser("freeze-runtime")
    freeze_runtime.add_argument("--source-manifest-hash", required=True)
    freeze_runtime.add_argument("--full", action="store_true")
    for name in ("smoke", "run"):
        execution = sub.add_parser(name)
        execution.add_argument("--manifest-hash", required=True)
        execution.add_argument(
            "--max-new-generations",
            type=int,
            default=SMOKE_GENERATION_BUDGET if name == "smoke" else 0,
        )
        execution.add_argument("--stop-after", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"prepare-judge-inputs", "run-luna", "run-deepseek", "judge-status", "export-judge-scores"}:
        from . import judging
        root = _runtime_artifact_root(args)
        if args.command == "prepare-judge-inputs": return _json_dump({"prepared": judging.prepare(root, args.manifest_hash)}) or 0
        if args.command.startswith("run-"): return _json_dump({"accepted": judging.run(root, args.command[4:], args.max_attempts, args.stop_after)}) or 0
        if args.command == "judge-status": return _json_dump(judging.status(root, args.judge)) or 0
        return _json_dump({"exported": judging.export(root, args.judge, args.output)}) or 0
    handlers = {
        "doctor": cmd_doctor,
        "preflight": cmd_preflight,
        "prepare": cmd_prepare,
        "prepare-contexts": cmd_prepare_contexts,
        "activation": cmd_activation,
        "status": cmd_status,
        "generate-prepared": cmd_generate_prepared,
        "validate": cmd_validate,
        "export": cmd_export,
        "census": cmd_census,
        "freeze-runtime": cmd_freeze_runtime,
        "smoke": cmd_smoke,
        "run": cmd_run,
    }
    try:
        return handlers[args.command](args)
    except (OSError, ValueError, EvidenceError, CheckpointError) as exc:
        print(f"r1-mlx: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
