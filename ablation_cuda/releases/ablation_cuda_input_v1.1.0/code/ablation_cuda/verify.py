"""Independent, standard-library-only verifier and importer for collaborator results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from ablation_cuda import __version__


PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
EXPECTED_QUERY_ID = "pubmedqa_21084567"
EXPECTED_MODELS = {
    "generator": ("occ-ai/OCC-RAG-1.7B", "e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c"),
    "embedder": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"),
    "cross_encoder": ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825"),
}
CANONICAL_DATASET_SHA256 = "d2820ced7c2994218245887ad753700923262fb91ec983eeacc01b217ec43b6b"
CANONICAL_QUESTIONS_MANIFEST_SHA256 = "e99ab9055dbe883da72f5466afa09629b13f1b9139f516a45f8aa38e21f4cee9"
EXTERNAL_DB_SNAPSHOT_SHA256 = "c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0"
MODEL_ASSET_MANIFEST_SHA256 = "e9912dc029a31ce83aa88010162148ad57fcc6e5082dd6099ba5afab34e7c3bc"
TELEMETRY_REQUIRED = {
    "token_id", "token_text", "logprob", "nll_nats", "nll_bits_per_token",
    "entropy_k20_bits", "entropy_k50_bits", "entropy_k100_bits",
    "retained_mass_k20", "retained_mass_k50", "retained_mass_k100",
    "top1_prob_full", "logit_margin", "top_logit_margins", "telemetry_device", "stop_state",
}

FULL_MATRIX_ARM_IDS = [
    "B6_full", "B6_minus_score_blending", "B6_minus_final_reranking",
    "B6_minus_dynamic_graph_expander", "B6_minus_static_graph_context",
    "B6_minus_intrapaper_graph_candidates", "B6_minus_concept_retrieval",
    "B6_minus_bridge_retrieval", "B6_minus_parent_deduplication",
    "B6_minus_lexical_retrieval", "B6_minus_dense_retrieval",
    "B6_fixed_budget_context", "B6_interaction_graph_candidate_x_dynamic_expander",
    "B6_interaction_reranker_x_blending",
]
FULL_RUN_STATUSES = {"COMPLETE", "PARTIAL", "FAILED", "SCIENTIFIC_VALIDATION_FAILED"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_tar(archive: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile() or member.name in files:
                    raise ValueError(f"Unexpected or duplicate archive entry: {member.name}")
                handle = tar.extractfile(member)
                if handle is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                files[member.name] = handle.read()
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"Cannot read compressed results archive: {exc}") from exc
    raw_manifest = files.get("RESULTS_MANIFEST.json")
    if raw_manifest is None:
        raise ValueError("RESULTS_MANIFEST.json is missing")
    try:
        manifest = json.loads(raw_manifest)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("RESULTS_MANIFEST.json is invalid") from exc
    if manifest.get("package_type") != "ablation_cuda_results" or manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported results archive manifest")
    expected = manifest.get("expected_files")
    if not isinstance(expected, list) or not expected:
        raise ValueError("Results archive file inventory is empty")
    expected_map = {item.get("path"): item for item in expected if isinstance(item, dict)}
    if len(expected_map) != len(expected) or None in expected_map:
        raise ValueError("Results archive file inventory contains duplicate or invalid entries")
    actual_names = set(files) - {"RESULTS_MANIFEST.json"}
    if actual_names != set(expected_map):
        raise ValueError(f"Results archive inventory mismatch; missing={sorted(set(expected_map)-actual_names)}, "
                         f"extra={sorted(actual_names-set(expected_map))}")
    for name, item in expected_map.items():
        data = files[name]
        if len(data) != item.get("size_bytes") or sha256_bytes(data) != item.get("sha256"):
            raise ValueError(f"Results archive internal checksum mismatch: {name}")
    run_manifest = files.get("run_manifest.json")
    if run_manifest is None or sha256_bytes(run_manifest) != manifest.get("run_manifest_sha256"):
        raise ValueError("Run manifest checksum mismatch")
    try:
        run = json.loads(run_manifest)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("run_manifest.json is invalid") from exc
    if manifest.get("run_id") != run.get("run_id") or manifest.get("result_status") != run.get("status"):
        raise ValueError("Results manifest does not match run manifest")
    return files, run


def _full_manifest_probe(archive: Path) -> bool:
    """Identify v2 run archives without reading their large telemetry payloads."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            candidates = [member for member in tar.getmembers()
                          if PurePosixPath(member.name).name == "RESULTS_MANIFEST.json"]
            if len(candidates) != 1:
                return False
            member = candidates[0]
            handle = tar.extractfile(member)
            if handle is None:
                return False
            value = json.loads(handle.read())
            return value.get("package_type") == "ablation_cuda_results" and value.get("schema_version") == "2.0"
    except (tarfile.TarError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _extract_full_archive(archive: Path, destination: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Safely stream a v2 archive to a temporary directory and verify its inventory."""
    seen_members: set[str] = set()
    roots: set[str] = set()
    extracted: dict[str, tuple[Path, int, str]] = {}
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                raw_name = member.name
                path = PurePosixPath(raw_name)
                if ("\\" in raw_name or ":" in raw_name or path.is_absolute()
                        or ".." in path.parts or not path.parts or "\x00" in raw_name):
                    raise ValueError(f"Unsafe archive path: {raw_name}")
                if raw_name in seen_members:
                    raise ValueError(f"Duplicate archive entry: {raw_name}")
                seen_members.add(raw_name)
                if len(path.parts) < 2:
                    raise ValueError(f"Archive member is outside the run directory: {raw_name}")
                roots.add(path.parts[0])
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Unexpected non-file archive member: {raw_name}")
                relative = PurePosixPath(*path.parts[1:])
                relative_text = relative.as_posix()
                target = destination.joinpath(path.parts[0], *relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = tar.extractfile(member)
                if handle is None:
                    raise ValueError(f"Cannot read archive member: {raw_name}")
                digest = hashlib.sha256()
                size = 0
                with target.open("wb") as output:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                extracted[relative_text] = (target, size, digest.hexdigest())
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"Cannot stream compressed results archive: {exc}") from exc
    if len(roots) != 1:
        raise ValueError(f"Full results archive must preserve one run directory; found roots {sorted(roots)}")
    root_name = next(iter(roots))
    run_root = destination / root_name
    manifest_path = run_root / "RESULTS_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("RESULTS_MANIFEST.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("RESULTS_MANIFEST.json is invalid") from exc
    if manifest.get("package_type") != "ablation_cuda_results" or manifest.get("schema_version") != "2.0":
        raise ValueError("Unsupported full results archive manifest")
    expected = manifest.get("expected_files")
    if not isinstance(expected, list) or not expected:
        raise ValueError("Full results archive inventory is empty")
    expected_map: dict[str, dict[str, Any]] = {}
    for item in expected:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Full results archive inventory contains an invalid row")
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or "\\" in item["path"]:
            raise ValueError(f"Unsafe path in results inventory: {item['path']}")
        if item["path"] in expected_map:
            raise ValueError(f"Duplicate file in results inventory: {item['path']}")
        expected_map[item["path"]] = item
    actual_names = set(extracted) - {"RESULTS_MANIFEST.json", "SHA256SUMS"}
    if actual_names != set(expected_map):
        raise ValueError(f"Results archive inventory mismatch; missing={sorted(set(expected_map)-actual_names)}, "
                         f"extra={sorted(actual_names-set(expected_map))}")
    for name, item in expected_map.items():
        _, size, digest = extracted[name]
        if size != item.get("size_bytes") or digest != item.get("sha256"):
            raise ValueError(f"Results archive internal checksum mismatch: {name}")
    checksum_path = run_root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError("Internal SHA256SUMS file is missing")
    listed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("Internal SHA256SUMS contains an invalid row")
        digest, name = match.groups()
        if name in listed:
            raise ValueError(f"Duplicate internal checksum path: {name}")
        listed[name] = digest
    expected_digests = {name: item["sha256"] for name, item in expected_map.items()}
    if listed != expected_digests:
        raise ValueError("Internal SHA256SUMS does not match the verified archive inventory")
    run_manifest_path = run_root / "run_manifest.json"
    if not run_manifest_path.is_file() or sha256_file(run_manifest_path) != manifest.get("run_manifest_sha256"):
        raise ValueError("Run manifest checksum mismatch")
    try:
        run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("run_manifest.json is invalid") from exc
    if manifest.get("run_id") != run.get("run_id") or manifest.get("result_status") != run.get("status"):
        raise ValueError("Results manifest does not match run manifest")
    if run.get("experiment_type") != "full_cuda_ablation" or run.get("status") not in FULL_RUN_STATUSES:
        raise ValueError("Run manifest is not a supported full CUDA ablation result")
    return run_root, manifest, run


def _read_external_digest(archive: Path, expected_sha256: str | None) -> str:
    actual = sha256_file(archive)
    sidecar = Path(str(archive) + ".sha256")
    if expected_sha256:
        expected = expected_sha256.strip().lower()
    elif sidecar.is_file():
        match = re.match(r"^([0-9a-fA-F]{64})\s+", sidecar.read_text(encoding="utf-8"))
        if not match:
            raise ValueError("Archive .sha256 sidecar has an invalid format")
        expected = match.group(1).lower()
    else:
        raise ValueError("Archive SHA-256 evidence is required: provide --expected-sha256 or the .sha256 sidecar")
    if expected != actual:
        raise ValueError(f"Archive SHA-256 mismatch: expected {expected}, computed {actual}")
    return actual


def _read_json_path(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _iter_jsonl_path(path: Path, label: str):
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{label} has invalid JSON on line {line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{label} line {line_number} must be a JSON object")
                yield value
    except OSError as exc:
        raise ValueError(f"{label} is missing or unreadable") from exc


def _expected_arm_hash(entry: dict[str, Any]) -> str:
    config = entry.get("effective_config")
    definition = entry.get("definition")
    if not isinstance(config, dict) or not isinstance(definition, dict):
        raise ValueError(f"Arm {entry.get('arm_id')} lacks an effective config or intervention definition")
    components = config.get("rag_components", {})
    baseline_components = {key: components.get(key, True) for key in components}
    baseline_components.update({
        "hyde": False,
        "reranker": bool(components.get("reranker", True)),
        "graph_neighbors_in_rrf": bool(components.get("graph_neighbors_in_rrf", True)),
        "intent_classifier": bool(components.get("intent_classifier", False)),
        "shannon_estimator_enabled": bool(components.get("shannon_estimator_enabled", True)),
        "deduplicate_parent_chunks": bool(components.get("deduplicate_parent_chunks", True)),
    })
    overrides = {str(row[0]): row[1] for row in definition.get("overrides", [])}
    manifest_effective = {"baseline": "B6", "paper_baseline": "B5", "config": config,
                          "baseline_components": baseline_components, "overrides": overrides}
    return sha256_bytes(canonical_json(manifest_effective))


def _intervention_trace_state(arm_id: str, result: dict[str, Any], *, off_arm: bool) -> bool:
    trace = result.get("retrieval_candidates_and_ranking_trace", {})
    ranking = trace.get("ranking", {})
    graph = trace.get("graph_retrieval", {})
    branches = graph.get("branch_execution", {})
    assembly = trace.get("context_and_prompt_assembly", {})
    if arm_id == "B6_minus_score_blending":
        return (ranking.get("ranking_formula") == "raw_cross_encoder_score"
                and ranking.get("score_blending_branch_executed") is True) if off_arm else (
                    ranking.get("ranking_formula") == "normalized_cross_encoder_rrf_blend"
                    and ranking.get("score_blending_branch_executed") is True)
    if arm_id == "B6_minus_final_reranking":
        return ranking.get("stage_status") == ("reranker_disabled" if off_arm else "complete")
    if arm_id == "B6_minus_dynamic_graph_expander":
        return assembly.get("dynamic_graph_expansion_executed") is (False if off_arm else True)
    if arm_id == "B6_minus_static_graph_context":
        return (assembly.get("static_graph_context_disabled") is True if off_arm
                else assembly.get("static_graph_context_enabled") is True)
    if arm_id == "B6_minus_intrapaper_graph_candidates":
        return ((branches.get("intra_paper_branch_enabled") is False
                 and branches.get("intra_paper_branch_executed") is False) if off_arm else
                (branches.get("intra_paper_branch_enabled") is True
                 and branches.get("intra_paper_branch_executed") is True))
    if arm_id == "B6_minus_concept_retrieval":
        return ((branches.get("concept_branch_enabled") is False
                 and branches.get("concept_lookup_executed") is False) if off_arm else
                (branches.get("concept_branch_enabled") is True
                 and branches.get("concept_lookup_executed") is True))
    if arm_id == "B6_minus_bridge_retrieval":
        return ((branches.get("bridge_branch_enabled") is False
                 and branches.get("bridge_lookup_executed") is False) if off_arm else
                (branches.get("bridge_branch_enabled") is True
                 and branches.get("bridge_lookup_executed") is True))
    if arm_id == "B6_minus_parent_deduplication":
        if off_arm:
            return (ranking.get("parent_deduplication_gate_evaluated") is True
                    and ranking.get("parent_deduplication_branch_executed") is False)
        return ranking.get("parent_deduplication_branch_executed") is True
    if arm_id in {"B6_minus_lexical_retrieval", "B6_minus_dense_retrieval"}:
        key = "lexical" if "lexical" in arm_id else "dense"
        enabled_key = f"{key}_retrieval_enabled"
        current = trace.get("current_trace", {})
        enabled = current.get("retrieval_paths", {}).get(enabled_key)
        seeds = current.get("seed_chunks_from_lexical_dense", {}).get(key, [])
        return (enabled is False and not seeds) if off_arm else (enabled is True and bool(seeds))
    if arm_id == "B6_fixed_budget_context":
        if off_arm:
            return assembly.get("fixed_budget_policy_executed") is True
        return trace.get("context_capture", {}).get("context_trim") is not None
    if arm_id == "B6_interaction_graph_candidate_x_dynamic_expander":
        if off_arm:
            return (branches.get("intra_paper_branch_enabled") is False
                    and branches.get("intra_paper_branch_executed") is False
                    and assembly.get("dynamic_graph_expansion_executed") is False)
        return (branches.get("intra_paper_branch_enabled") is True
                and branches.get("intra_paper_branch_executed") is True
                and assembly.get("dynamic_graph_expansion_executed") is True)
    if arm_id == "B6_interaction_reranker_x_blending":
        if off_arm:
            return (ranking.get("stage_status") == "reranker_disabled"
                    and ranking.get("score_blending_branch_executed") is False)
        return (ranking.get("stage_status") == "complete"
                and ranking.get("score_blending_branch_executed") is True)
    return False


def _validate_token_sidecar(root: Path, result: dict[str, Any]) -> int:
    reference = result.get("telemetry_sidecar")
    if not isinstance(reference, dict):
        raise ValueError(f"Successful operation {result.get('arm')}/{result.get('query_id')} has no token telemetry sidecar")
    relative = PurePosixPath(str(reference.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("Unsafe token telemetry sidecar reference")
    sidecar = root.joinpath(*relative.parts)
    if not sidecar.is_file() or sha256_file(sidecar) != reference.get("sha256"):
        raise ValueError(f"Token telemetry sidecar checksum mismatch: {reference.get('path')}")
    if reference.get("units") != {"entropy": "bits", "log_probability": "nats", "probability": "fraction"}:
        raise ValueError("Token telemetry sidecar units are missing or unsupported")
    total = 0
    seen_events = set()
    try:
        with gzip.open(sidecar, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid telemetry JSONL row {line_number}") from exc
                event_index = row.get("event_index")
                if not isinstance(event_index, int) or event_index in seen_events:
                    raise ValueError("Token telemetry sidecar has a missing or duplicate generation-event index")
                seen_events.add(event_index)
                tokens = row.get("token_telemetry")
                if not isinstance(tokens, list):
                    raise ValueError("Token telemetry sidecar row has no token list")
                for token in tokens:
                    if not isinstance(token, dict) or not TELEMETRY_REQUIRED.issubset(token):
                        raise ValueError("Token telemetry row lacks required lossless metrics")
                    if not str(token.get("telemetry_device", "")).startswith("cuda"):
                        raise ValueError("Token telemetry was not calculated on CUDA")
                    if any(not isinstance(token.get(key), (int, float)) for key in (
                        "logprob", "nll_nats", "nll_bits_per_token", "entropy_k20_bits",
                        "entropy_k50_bits", "entropy_k100_bits", "retained_mass_k20",
                        "retained_mass_k50", "retained_mass_k100", "top1_prob_full", "logit_margin",
                    )):
                        raise ValueError("Token telemetry has a missing numeric metric or unknown unit")
                    total += 1
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError) as exc:
        raise ValueError("Token telemetry sidecar is corrupt or unreadable") from exc
    if total != reference.get("token_count") or total == 0:
        raise ValueError("Token telemetry sidecar token count mismatch or empty telemetry")
    events = result.get("generation_events", [])
    if reference.get("generation_event_count") != len(events) or any(index >= len(events) for index in seen_events):
        raise ValueError("Token telemetry does not map to the stored generation-event list")
    final_indices = [index for index, event in enumerate(events)
                     if event.get("purpose") == "final_answer_generation"]
    if len(final_indices) != 1 or final_indices[0] not in seen_events:
        raise ValueError("Final generation call does not have token-level telemetry")
    return total


def _json_file(files: dict[str, bytes], name: str) -> dict[str, Any]:
    data = files.get(name)
    if data is None:
        raise ValueError(f"Required results file is missing: {name}")
    try:
        value = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Required results file is invalid JSON: {name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Required results file must contain an object: {name}")
    return value


def _remove_component(config: dict[str, Any], component: str) -> dict[str, Any]:
    copy = json.loads(json.dumps(config))
    components = copy.get("rag_components", {})
    components.pop(component, None)
    return copy


def _validate_input_identity(run: dict[str, Any], input_manifest_path: Path | None,
                             questions_path: Path | None) -> dict[str, Any] | None:
    if input_manifest_path is None:
        return None
    manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("package_type") != "ablation_cuda_input":
        raise ValueError("Selected input manifest is not an ablation_cuda_input package")
    if run.get("input_package_manifest_sha256") != sha256_file(input_manifest_path):
        raise ValueError("Result archive was produced from a different frozen input package manifest")
    dataset = run.get("dataset", {})
    if dataset.get("sha256") != manifest.get("dataset", {}).get("questions_sha256"):
        raise ValueError("Result query manifest fingerprint differs from the frozen input package")
    database = run.get("database", {})
    expected_db = manifest.get("database", {})
    if database.get("candidate") != "external_db" or database.get("snapshot_sha256") != expected_db.get("snapshot_sha256"):
        raise ValueError("Result database fingerprint is not the selected external_db input snapshot")
    if questions_path is None:
        sibling = input_manifest_path.parent / "dataset/questions.jsonl"
        questions_path = sibling if sibling.is_file() else None
    if questions_path:
        if sha256_file(questions_path) != dataset.get("sha256"):
            raise ValueError("Frozen questions file SHA-256 differs from the run manifest")
        rows = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = [row for row in rows if row.get("query_id") == EXPECTED_QUERY_ID]
        if len(selected) != 1:
            raise ValueError("Frozen questions manifest does not have exactly one B6 smoke query")
        if selected[0].get("question") != run.get("question"):
            raise ValueError("Result question text differs from the frozen canonical query")
        if selected[0].get("answerability_label") != run.get("answerability_label"):
            raise ValueError("Result answerability label differs from the frozen query")
        if selected[0].get("original_pmids", []) != run.get("gold_pmids", []):
            raise ValueError("Result gold PMID differs from the frozen query record")
    return manifest


def _validate_models(run: dict[str, Any], input_manifest: dict[str, Any] | None) -> None:
    installed = run.get("models", {}).get("models", {})
    assets = None
    if input_manifest:
        assets_path = Path(input_manifest["_path"]).parent / "models/asset_manifest.json"
        if assets_path.is_file():
            assets = json.loads(assets_path.read_text(encoding="utf-8"))
    for role, (model_id, revision) in EXPECTED_MODELS.items():
        item = installed.get(role)
        if not isinstance(item, dict) or item.get("model_id") != model_id or item.get("revision") != revision or not item.get("verified"):
            raise ValueError(f"Pinned model identity or verified asset report missing for {role}")
        if not item.get("files"):
            raise ValueError(f"Pinned model file checksums are missing for {role}")
        if assets is not None:
            source_item = assets.get("model_assets", {}).get(role, {})
            if item.get("files") != source_item.get("files"):
                raise ValueError(f"Downloaded {role} file checksums differ from the frozen model asset manifest")


def validate_results(files: dict[str, bytes], run: dict[str, Any],
                     input_manifest_path: Path | None = None,
                     questions_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if run.get("status") != "SUCCESS":
        raise ValueError(f"Run status is {run.get('status')}; incomplete runs cannot be imported")
    if run.get("query_id") != EXPECTED_QUERY_ID:
        raise ValueError("Run does not contain the frozen B6 query ID")
    if run.get("full_matrix_started") is not False or run.get("judge") != "not_run":
        raise ValueError("Run unexpectedly started the full matrix or a judge")
    if run.get("database", {}).get("identity_same_across_arms") is not True:
        raise ValueError("The same database snapshot identity was not verified across both arms")
    intervention = run.get("intervention", {})
    component = intervention.get("component")
    if component not in {"score_blending", "graph_expansion"}:
        raise ValueError(f"Unsupported or missing trace-selected intervention: {component!r}")
    off_name = f"no_{component}"
    if intervention.get("trace_rule_passed") is not True or intervention.get("code_path_verified") is not True:
        raise ValueError("Run manifest does not prove trace-based intervention eligibility and execution")
    if [item.get("arm") for item in run.get("arms", [])] != ["B6", off_name]:
        raise ValueError("Run manifest does not contain exactly the frozen B6 and selected-off arms")
    if run.get("dataset", {}).get("row_count") != 375 or run.get("dataset", {}).get("answerable_count") != 300 or run.get("dataset", {}).get("designated_negative_count") != 75:
        raise ValueError("Run does not identify the frozen 375-query population")
    input_manifest = _validate_input_identity(run, input_manifest_path, questions_path)
    if input_manifest is not None:
        input_manifest = {**input_manifest, "_path": str(input_manifest_path.resolve())}
    _validate_models(run, input_manifest)

    device = run.get("device_preflight", {}).get("torch_cuda_operation", {})
    if not str(device.get("operation_device", "")).startswith("cuda") or not device.get("result_sum"):
        raise ValueError("Actual CUDA probe operation evidence is missing")
    if "5090" not in str(device.get("device", "")).lower() or device.get("compute_capability") != [12, 0]:
        raise ValueError(f"CUDA device evidence is not an RTX 5090 / Blackwell GPU: {device.get('device')!r}, "
                         f"compute {device.get('compute_capability')}")

    arm_index = {}
    for arm_name in ("B6", off_name):
        arm = _json_file(files, f"arms/{arm_name}.json")
        if arm.get("arm") != arm_name or arm.get("status") != "SUCCESS":
            raise ValueError(f"Arm {arm_name} is missing or incomplete")
        if arm.get("query_id") != EXPECTED_QUERY_ID or arm.get("question") != run.get("question"):
            raise ValueError(f"Arm {arm_name} question identity differs from the frozen run")
        if not arm.get("raw_generated_response") or not arm.get("parsed_answer"):
            raise ValueError(f"Arm {arm_name} raw response or parsed answer is empty")
        prompt = arm.get("prompt")
        if not isinstance(prompt, str) or not prompt or arm.get("prompt_sha256") != sha256_bytes(prompt.encode("utf-8")):
            raise ValueError(f"Arm {arm_name} final prompt or prompt fingerprint is missing")
        event = arm.get("generation_event", {})
        if (event.get("formatted_prompt") != prompt
                or event.get("prompt") != arm.get("input_prompt")):
            raise ValueError(f"Arm {arm_name} exact serialized prompt is missing")
        generation_events = arm.get("generation_events")
        if not isinstance(generation_events, list) or not generation_events:
            raise ValueError(f"Arm {arm_name} has no persisted per-call generation record")
        final_calls = [item for item in generation_events if item.get("purpose") == "final_answer_generation"]
        if len(final_calls) != 1 or final_calls[0].get("event", {}).get("formatted_prompt") != prompt:
            raise ValueError(f"Arm {arm_name} final answer call is not linked to its exact prompt")
        for call_index, call in enumerate(generation_events):
            call_telemetry = call.get("token_telemetry")
            if not isinstance(call_telemetry, list) or not call_telemetry:
                raise ValueError(f"Arm {arm_name} generation call {call_index} has no token telemetry")
            for token_index, token in enumerate(call_telemetry):
                if (not TELEMETRY_REQUIRED.issubset(token)
                        or not str(token.get("telemetry_device", "")).startswith("cuda")):
                    raise ValueError(f"Arm {arm_name} generation call {call_index} token {token_index} is incomplete or not CUDA-computed")
        if arm.get("result_errors"):
            raise ValueError(f"Arm {arm_name} has errors: {arm['result_errors']}")
        stages = arm.get("stage_status", {})
        required_stages = {
            "preprocessing", "lexical_retrieval", "dense_retrieval", "cross_encoder",
            "graph_retrieval", "graph_expansion", "context_construction", "generation",
            "answerability_parser",
        }
        if not required_stages.issubset(stages) or any(stages.get(key) != "complete" for key in required_stages):
            raise ValueError(f"Arm {arm_name} has incomplete required stages: {stages}")
        telemetry = arm.get("token_telemetry")
        if not isinstance(telemetry, list) or not telemetry:
            raise ValueError(f"Arm {arm_name} has no per-token telemetry")
        for index, token in enumerate(telemetry):
            if not isinstance(token, dict) or not TELEMETRY_REQUIRED.issubset(token):
                raise ValueError(f"Arm {arm_name} token {index} has an incomplete telemetry schema")
            if not str(token.get("telemetry_device", "")).startswith("cuda"):
                raise ValueError(f"Arm {arm_name} token {index} telemetry was not computed on CUDA")
        diagnostics = arm.get("device_diagnostics", {})
        actual = diagnostics.get("cuda_inference_results", {})
        if not all(actual.get(key) for key in ("embedding_inference", "cross_encoder", "generation", "token_metrics")):
            raise ValueError(f"Arm {arm_name} is missing actual CUDA neural inference evidence")
        if not diagnostics.get("e5_cuda_calls"):
            raise ValueError(f"Arm {arm_name} has no instrumented E5 inference record")
        if not diagnostics.get("cross_encoder_cuda_calls"):
            raise ValueError(f"Arm {arm_name} has no instrumented Cross-Encoder inference record")
        cross_encoder_roles = {call.get("role") for call in diagnostics["cross_encoder_cuda_calls"]}
        if "final_candidate_reranking" not in cross_encoder_roles:
            raise ValueError(f"Arm {arm_name} has no CUDA Cross-Encoder call at final candidate reranking")
        graph_telemetry = arm.get("retrieval_candidates_and_ranking_trace", {}).get("graph_expander") or {}
        graph_scoring_expected = (
            any(float(item.get("rerank_time", 0.0) or 0.0) > 0
                for item in graph_telemetry.get("hops", []))
            or float(graph_telemetry.get("ingestion", {}).get("rerank_time", 0.0) or 0.0) > 0
        )
        if graph_scoring_expected and "experimental_graph_expander" not in cross_encoder_roles:
            raise ValueError(f"Arm {arm_name} graph-expander trace reports scoring without a CUDA Cross-Encoder call")
        if any(not str(device).startswith("cuda") for device in diagnostics.get("generator_model_device_set", [])):
            raise ValueError(f"Arm {arm_name} generator is not entirely on CUDA")
        if any(not str(device).startswith("cuda") for device in diagnostics.get("cross_encoder_model_device_set", [])):
            raise ValueError(f"Arm {arm_name} Cross-Encoder is not entirely on CUDA")
        if any(not str(device).startswith("cuda") for device in diagnostics.get("e5_query_embedding", {}).get("model_device_set", [])):
            raise ValueError(f"Arm {arm_name} E5 model is not entirely on CUDA")
        if any(not call.get("input_device_set") or any(
            not str(device).startswith("cuda") for device in call["input_device_set"]
        ) for call in diagnostics.get("e5_cuda_calls", [])):
            raise ValueError(f"Arm {arm_name} has E5 inference inputs outside CUDA")
        if any(not call.get("input_device_set") or any(
            not str(device).startswith("cuda") for device in call["input_device_set"]
        ) for call in diagnostics.get("cross_encoder_cuda_calls", [])):
            raise ValueError(f"Arm {arm_name} has Cross-Encoder inference inputs outside CUDA")
        _validate_arm_identity(arm, arm_name)
        arm_index[arm_name] = arm

    full, off = arm_index["B6"], arm_index[off_name]
    if full.get("snapshot_identity", {}).get("snapshot_sha256") != off.get("snapshot_identity", {}).get("snapshot_sha256"):
        raise ValueError("Arms use different input database snapshots")
    if full.get("model_manifest_sha256") != off.get("model_manifest_sha256") or full.get("decoding") != off.get("decoding"):
        raise ValueError("Arms do not share identical model assets and decoding settings")
    full_effective = full.get("effective_configuration", {})
    off_effective = off.get("effective_configuration", {})
    if full_effective.get("baseline") != "B6" or off_effective.get("baseline") != "B6":
        raise ValueError("Both arms must retain the documented B6 baseline")
    if _remove_component(full_effective.get("config", {}), component) != _remove_component(off_effective.get("config", {}), component):
        raise ValueError(f"Effective arm settings differ beyond the selected {component} intervention")
    full_components = dict(full_effective.get("baseline_components", {}))
    off_components = dict(off_effective.get("baseline_components", {}))
    if full_components.pop(component, None) is not True or off_components.pop(component, None) is not False or full_components != off_components:
        raise ValueError(f"B6 active component bundles are not isolated to {component}")
    full_rank = full.get("retrieval_candidates_and_ranking_trace", {}).get("ranking", {})
    off_rank = off.get("retrieval_candidates_and_ranking_trace", {}).get("ranking", {})
    raw_ranking_fields = ("candidate_ids_before_reranker", "rrf_scores_before_reranker", "cross_encoder_scores")
    if any(full_rank.get(key) != off_rank.get(key) for key in raw_ranking_fields):
        raise ValueError("Raw retrieval candidates or ranking inputs changed across the intervention arms")
    if component == "score_blending":
        if full_rank.get("ranking_formula") != "normalized_cross_encoder_rrf_blend" or full_rank.get("score_blending_branch_executed") is not True:
            raise ValueError("B6 ranking trace does not prove the score-blending path executed")
        if off_rank.get("ranking_formula") != "raw_cross_encoder_score" or off_rank.get("score_blending_enabled") is not False or off_rank.get("score_blending_branch_executed") is not True:
            raise ValueError("Intervention trace does not prove the score-blending code path was disabled")
    else:
        if (full_rank.get("ranking_formula") != "normalized_cross_encoder_rrf_blend"
                or off_rank.get("ranking_formula") != full_rank.get("ranking_formula")
                or off_rank.get("score_blending_enabled") is not True):
            raise ValueError("Graph-expansion intervention changed the unrelated ranking stage")
        full_assembly = full.get("retrieval_candidates_and_ranking_trace", {}).get("context_and_prompt_assembly", {})
        off_assembly = off.get("retrieval_candidates_and_ranking_trace", {}).get("context_and_prompt_assembly", {})
        if not (full_assembly.get("dynamic_graph_expansion_executed") is True
                and full_assembly.get("dynamic_graph_output_inserted") is True
                and off_assembly.get("dynamic_graph_expansion_executed") is False
                and off_assembly.get("dynamic_graph_output_inserted") is False):
            raise ValueError("Graph-expansion trace does not prove an inserted B6 block and disabled off-arm path")
    for arm_name in ("B6", off_name):
        effective = arm_index[arm_name]["effective_configuration"]
        if not isinstance(effective.get("config"), dict) or not effective.get("baseline_components"):
            raise ValueError(f"Arm {arm_name} effective configuration is incomplete")
        if not arm_index[arm_name].get("selected_evidence"):
            raise ValueError(f"Arm {arm_name} has no selected evidence context")
        trace = arm_index[arm_name].get("retrieval_candidates_and_ranking_trace", {})
        if not trace.get("trace_counters"):
            raise ValueError(f"Arm {arm_name} is missing retrieval/context/prompt trace counters")
        context_capture = trace.get("context_capture", {})
        if not context_capture.get("context_build") or not context_capture.get("context_trim"):
            raise ValueError(f"Arm {arm_name} is missing static/dynamic context construction evidence")
    return [full, off], input_manifest


def _validate_arm_identity(arm: dict[str, Any], arm_name: str) -> None:
    if arm.get("stable_result_id") != hashlib.sha256(canonical_json({
        "input_identity": arm.get("input_identity"),
        "prompt_sha256": arm.get("prompt_sha256"),
    })).hexdigest():
        raise ValueError(f"Arm {arm_name} stable result identity does not match its configuration and prompt")


def _preview(arms: list[dict[str, Any]], run: dict[str, Any]) -> str:
    rows = ["# CUDA B6 two-arm smoke preview", "",
            f"- Query: `{run['query_id']}` — {run.get('question', '')}",
            f"- Result status: **{run['status']}**", f"- GPU: {run.get('device_preflight', {}).get('torch_cuda_operation', {}).get('device', 'unknown')}", ""]
    for arm in arms:
        rows.extend([f"## {arm['arm']}", "", f"Parser status: `{arm.get('parser_status')}`", "",
                     "Answer:", "", arm.get("parsed_answer", ""), "",
                     f"Retrieved papers: {', '.join(arm.get('retrieved_paper_ids', [])) or '(none recorded)'}. ", ""])
    return "\n".join(rows)


def _apply_overrides(config: dict[str, Any], definition: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    for path, item in definition.get("overrides", []):
        parts = str(path).split(".")
        current = value
        for part in parts[:-1]:
            if not isinstance(current.get(part), dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = item
    return value


def _read_jsonl_rows(path: Path, label: str) -> list[dict[str, Any]]:
    return list(_iter_jsonl_path(path, label))


def _full_operation_identity(run: dict[str, Any], question: dict[str, Any],
                             arm: dict[str, Any], result: dict[str, Any]) -> str:
    effective = result.get("effective_configuration")
    if not isinstance(effective, dict) or effective.get("config") != arm.get("effective_config"):
        raise ValueError(f"Successful result {arm.get('arm_id')}/{question.get('query_id')} has a mismatched effective configuration")
    components = effective.get("baseline_components")
    if not isinstance(components, dict):
        raise ValueError("Effective B6 component bundle is missing")
    config_components = effective["config"].get("rag_components", {})
    expected_components = {key: config_components.get(key, True) for key in config_components}
    expected_components.update({
        "hyde": False,
        "reranker": bool(config_components.get("reranker", True)),
        "graph_neighbors_in_rrf": bool(config_components.get("graph_neighbors_in_rrf", True)),
        "intent_classifier": bool(config_components.get("intent_classifier", False)),
        "shannon_estimator_enabled": bool(config_components.get("shannon_estimator_enabled", True)),
        "deduplicate_parent_chunks": bool(config_components.get("deduplicate_parent_chunks", True)),
    })
    if components != expected_components or effective.get("baseline") != "B6" or effective.get("paper_baseline") != "B5":
        raise ValueError("Effective B6 execution components do not match the frozen arm configuration")
    if effective.get("intervention") is not None:
        raise ValueError("Arm-level intervention metadata differs from the frozen matrix definition")
    decoding = run.get("decoding", {})
    model_manifest_sha256 = result.get("model_manifest_sha256")
    input_id = sha256_bytes(canonical_json({
        "query_id": question["query_id"],
        "question_sha256": sha256_bytes(question["question"].encode("utf-8")),
        "snapshot_sha256": run["database"]["snapshot_sha256"],
        "model_manifest_sha256": model_manifest_sha256,
        "effective_config": effective,
        "decoding": decoding,
    }))
    return sha256_bytes(canonical_json({
        "input_id": input_id,
        "question_row_sha256": sha256_bytes(canonical_json(question)),
        "questions_manifest_sha256": run["dataset"]["questions_manifest_sha256"],
        "dataset_sha256": run["dataset"]["sha256"],
        "code_sha256": run["source_code_sha256"],
        "image": run["image_identity"],
        "telemetry_implementation": run["telemetry_implementation_version"],
    }))


def _validate_full_results(run_root: Path, run: dict[str, Any], archive_sha256: str,
                           *, input_manifest_path: Path | None = None) -> dict[str, Any]:
    if run.get("status") not in FULL_RUN_STATUSES:
        raise ValueError(f"Unsupported full-run status: {run.get('status')}")
    if run.get("internal_baseline") != "B6" or run.get("paper_baseline") != "B5":
        raise ValueError("Internal baseline B6 / paper-facing baseline B5 mapping is missing")
    if run.get("judge", {}).get("remote_requests_made") is not False:
        raise ValueError("Collaborator run manifest does not prove that no remote judge requests were made")
    if run.get("smoke_gate", {}).get("status") != "PASSED":
        raise ValueError("The full run has no passing mandatory CUDA smoke gate")
    device = run.get("device_preflight", {}).get("torch_cuda_operation", {})
    if not str(device.get("operation_device", "")).startswith("cuda") or not device.get("result_sum"):
        raise ValueError("Run manifest is missing an actual CUDA operation probe")
    if "5090" not in str(device.get("device", "")).lower() or device.get("compute_capability") != [12, 0]:
        raise ValueError("Full experiment was not run on an RTX 5090 / Blackwell GPU")

    questions_path = run_root / "questions.jsonl"
    questions = _read_jsonl_rows(questions_path, "questions.jsonl")
    if len(questions) != 375:
        raise ValueError(f"Frozen question count is {len(questions)}, expected 375")
    ids = [row.get("query_id") for row in questions]
    if any(not isinstance(row.get("question"), str) or not row["question"] for row in questions):
        raise ValueError("Frozen question rows contain an empty question")
    if len(set(ids)) != 375 or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Frozen query IDs are missing or duplicated")
    if sha256_file(questions_path) != run.get("dataset", {}).get("sha256"):
        raise ValueError("Frozen questions.jsonl checksum differs from the run dataset identity")
    manifest_path = run_root / "questions_manifest.json"
    question_manifest = _read_json_path(manifest_path, "questions_manifest.json")
    if sha256_file(manifest_path) != run.get("dataset", {}).get("questions_manifest_sha256"):
        raise ValueError("Frozen questions manifest checksum differs from the run dataset identity")
    if (question_manifest.get("row_count") != 375 or question_manifest.get("answerable_count") != 300
            or question_manifest.get("designated_negative_count") != 75
            or question_manifest.get("dataset_sha256") != run["dataset"]["sha256"]):
        raise ValueError("Frozen question manifest does not identify the canonical 375 / 300 / 75 population")
    answerable = sum(row.get("answerability_label") == "answerable" for row in questions)
    negative = sum(row.get("answerability_label") == "designated_negative" for row in questions)
    if answerable != 300 or negative != 75:
        raise ValueError("Question rows do not contain exactly 300 answerable and 75 designated negatives")
    if sha256_bytes(canonical_json(ids)) != run.get("dataset", {}).get("query_ids_sha256"):
        raise ValueError("Ordered query ID fingerprint does not match the run manifest")
    if run["dataset"].get("sha256") != CANONICAL_DATASET_SHA256:
        raise ValueError("Frozen questions do not match the canonical July 375-query dataset identity")
    if sha256_file(manifest_path) != CANONICAL_QUESTIONS_MANIFEST_SHA256:
        raise ValueError("Frozen question manifest does not match the canonical July dataset manifest")

    database = run.get("database", {})
    db_identity = _read_json_path(run_root / "metadata/database_identity.json", "database identity")
    if (database.get("candidate") != "external_db" or database.get("identity_same_across_all_arms") is not True
            or db_identity.get("candidate") != "external_db"
            or database.get("snapshot_sha256") != db_identity.get("snapshot_sha256")
            or database.get("snapshot_id") != db_identity.get("snapshot_id")
            or database.get("snapshot_sha256") != EXTERNAL_DB_SNAPSHOT_SHA256
            or database.get("persisted_vectors") != 22068
            or db_identity.get("logical_fingerprint", {}).get("vector_count") != 22068):
        raise ValueError("Frozen database identity or the 22,068 persisted vectors are not verified")
    if database.get("database_after_run_sha256") != database.get("snapshot_sha256"):
        raise ValueError("The read-only retrieval database changed during the run")
    if database.get("after_run_logical_fingerprint") != database.get("logical_fingerprint"):
        raise ValueError("The frozen database logical fingerprint changed during the run")

    models = run.get("models", {}).get("models", {})
    for role, (model_id, revision) in EXPECTED_MODELS.items():
        item = models.get(role, {})
        if (item.get("model_id") != model_id or item.get("revision") != revision
                or item.get("verified") is not True or not item.get("files")):
            raise ValueError(f"Pinned and verified {role} checkpoint identity is missing")
    asset_manifest_path = PACKAGE_ROOT / "models/asset_manifest.json"
    if (not asset_manifest_path.is_file()
            or sha256_file(asset_manifest_path) != MODEL_ASSET_MANIFEST_SHA256):
        raise ValueError("Author-side pinned model asset manifest is missing or has changed")
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    for role in EXPECTED_MODELS:
        if models[role].get("files") != asset_manifest.get("model_assets", {}).get(role, {}).get("files"):
            raise ValueError(f"Verified {role} checkpoint file checksums differ from the frozen model assets")
    model_hash = sha256_bytes(canonical_json(run["models"]))
    if run.get("run_configuration_hashes", {}).get("model_manifest_sha256") != model_hash:
        raise ValueError("Model manifest checksum differs from the frozen model identity")
    decoding = run.get("decoding", {})
    if (decoding.get("max_new_tokens") != 768 or decoding.get("context_limit") != 12000
            or decoding.get("strategy") != "greedy_argmax" or decoding.get("temperature") != 0.0
            or decoding.get("tokenizer_revision") != EXPECTED_MODELS["generator"][1]
            or run.get("actual_output_token_limit") != 768 or run.get("actual_context_limit_tokens") != 12000):
        raise ValueError("The documented generation decoding, context or output token limits differ")

    arms_manifest = _read_json_path(run_root / "arms_manifest.json", "arms_manifest.json")
    arm_rows = arms_manifest.get("arms")
    if (arms_manifest.get("internal_baseline") != "B6" or arms_manifest.get("paper_baseline") != "B5"
            or arms_manifest.get("query_count") != 375 or arms_manifest.get("arm_count") != 14
            or arms_manifest.get("expected_query_arm_operations") != 5250
            or [row.get("arm_id") for row in arm_rows or []] != FULL_MATRIX_ARM_IDS):
        raise ValueError("Frozen experimental arm matrix differs from the approved 14-arm/5,250-operation matrix")
    if arms_manifest.get("dataset_sha256") != run["dataset"]["sha256"]:
        raise ValueError("Arms manifest has a different frozen question dataset")
    if arms_manifest.get("external_db_snapshot_sha256") != database.get("snapshot_sha256"):
        raise ValueError("Arms manifest points at a different database snapshot")
    arm_by_id = {row["arm_id"]: row for row in arm_rows}
    base_config = arm_by_id["B6_full"].get("effective_config")
    if not isinstance(base_config, dict):
        raise ValueError("Full reference effective configuration is missing")
    definitions = {row["arm_id"]: row.get("definition") for row in arm_rows}
    for arm_id in FULL_MATRIX_ARM_IDS:
        row = arm_by_id[arm_id]
        definition = definitions[arm_id]
        if not isinstance(definition, dict) or definition.get("arm_id") != arm_id:
            raise ValueError(f"Arm definition is missing for {arm_id}")
        expected_config = _apply_overrides(base_config, definition)
        if row.get("effective_config") != expected_config:
            raise ValueError(f"Arm {arm_id} changes settings beyond its declared intervention")
        if row.get("reference_baseline") != "B6" or row.get("paper_baseline_mapping") != "B5":
            raise ValueError(f"Arm {arm_id} does not preserve the B6 -> B5 baseline mapping")
        if row.get("effective_configuration_hash") != _expected_arm_hash(row):
            raise ValueError(f"Arm {arm_id} effective configuration hash is invalid")
    if run.get("matrix", {}).get("planned_query_arm_operations") != 5250:
        raise ValueError("Run operation denominator is not 5,250")

    if input_manifest_path is not None:
        input_manifest = _read_json_path(input_manifest_path, "input package manifest")
        if (sha256_file(input_manifest_path) != run.get("input_package_manifest_sha256")
                or input_manifest.get("dataset", {}).get("questions_sha256") != run["dataset"]["sha256"]
                or input_manifest.get("database", {}).get("snapshot_sha256") != database["snapshot_sha256"]):
            raise ValueError("Input package manifest identity differs from the run")

    generation_rows = _read_jsonl_rows(run_root / "generation_results.jsonl", "generation_results.jsonl")
    expected_pairs = {(arm_id, query_id) for arm_id in FULL_MATRIX_ARM_IDS for query_id in ids}
    observed_pairs = [(row.get("arm_id"), row.get("query_id")) for row in generation_rows]
    if len(observed_pairs) != len(set(observed_pairs)):
        raise ValueError("generation_results.jsonl contains duplicated query-arm records")
    if set(observed_pairs) != expected_pairs or len(generation_rows) != 5250:
        raise ValueError("generation_results.jsonl does not account for every frozen query and approved arm")
    query_by_id = {row["query_id"]: row for row in questions}
    statuses: dict[tuple[str, str], str] = {}
    response_hashes: dict[tuple[str, str], str] = {}
    prompt_hashes: dict[tuple[str, str], str | None] = {}
    telemetry_tokens: dict[tuple[str, str], int] = {}
    intervention_counts = {arm_id: {"reference_active": 0, "off_path_changed": 0} for arm_id in FULL_MATRIX_ARM_IDS}
    results: dict[tuple[str, str], dict[str, Any] | None] = {}
    total_token_rows = 0
    for row in generation_rows:
        arm_id, query_id = row["arm_id"], row["query_id"]
        key = (arm_id, query_id)
        query = query_by_id[query_id]
        arm = arm_by_id[arm_id]
        status = row.get("operation_status")
        if status not in {"SUCCESS", "FAILED", "PARTIAL", "MISSING"}:
            raise ValueError(f"Unknown operation status {status!r} for {arm_id}/{query_id}")
        statuses[key] = status
        relative = PurePosixPath(str(row.get("canonical_result", "")))
        if (relative.is_absolute() or ".." in relative.parts or "\\" in str(relative)
                or relative.parts[:3] != ("baselines", arm_id, "results")
                or relative.name != f"{query_id}.json"):
            raise ValueError(f"Unsafe or mismatched canonical result path for {arm_id}/{query_id}")
        result_path = run_root.joinpath(*relative.parts)
        if status == "MISSING":
            if result_path.exists():
                raise ValueError(f"Missing operation {arm_id}/{query_id} unexpectedly has a result file")
            if row.get("response_sha256") != sha256_bytes(b""):
                raise ValueError("Missing operation response hash is not the empty-response hash")
            results[key] = None
            response_hashes[key] = sha256_bytes(b"")
            prompt_hashes[key] = None
            telemetry_tokens[key] = 0
            continue
        result = _read_json_path(result_path, f"result {arm_id}/{query_id}")
        if result.get("arm") != arm_id or result.get("query_id") != query_id or result.get("question") != query["question"]:
            raise ValueError(f"Canonical result query identity mismatch for {arm_id}/{query_id}")
        if result.get("answerability_label") != query["answerability_label"]:
            raise ValueError(f"Canonical result answerability label differs for {arm_id}/{query_id}")
        if result.get("synthetic_negative") is not (query["answerability_label"] == "designated_negative"):
            raise ValueError(f"Synthetic-negative label differs for {arm_id}/{query_id}")
        if result.get("source_pmids", []) != query.get("original_pmids", []):
            raise ValueError(f"Gold source PMID differs for {arm_id}/{query_id}")
        if result.get("status") != status:
            raise ValueError(f"Operation status index differs from canonical result for {arm_id}/{query_id}")
        raw = result.get("raw_generated_response", "")
        if not isinstance(raw, str) or row.get("response_sha256") != sha256_bytes(raw.encode("utf-8")):
            raise ValueError(f"Response hash mismatch for {arm_id}/{query_id}")
        response_hashes[key] = row["response_sha256"]
        if status == "SUCCESS":
            prompt = result.get("prompt")
            if not isinstance(prompt, str) or not prompt or result.get("prompt_sha256") != sha256_bytes(prompt.encode("utf-8")):
                raise ValueError(f"Exact generation prompt or prompt hash is missing for {arm_id}/{query_id}")
            final_events = [event for event in result.get("generation_events", [])
                            if event.get("purpose") == "final_answer_generation"]
            if (len(final_events) != 1
                    or final_events[0].get("event", {}).get("formatted_prompt") != prompt
                    or final_events[0].get("event", {}).get("prompt") != result.get("input_prompt")):
                raise ValueError(f"Stored final generation call is not linked to the exact serialized prompt for {arm_id}/{query_id}")
            if result.get("stable_result_id") != sha256_bytes(canonical_json({
                    "input_identity": result.get("input_identity"), "prompt_sha256": result["prompt_sha256"]})):
                raise ValueError(f"Stable result hash mismatch for {arm_id}/{query_id}")
            if result.get("effective_configuration_hash") != arm["effective_configuration_hash"]:
                raise ValueError(f"Effective configuration hash mismatch for {arm_id}/{query_id}")
            if result.get("model_manifest_sha256") != model_hash:
                raise ValueError(f"Model identity mismatch for {arm_id}/{query_id}")
            if result.get("snapshot_identity", {}).get("snapshot_sha256") != database["snapshot_sha256"]:
                raise ValueError(f"Database identity mismatch for {arm_id}/{query_id}")
            if result.get("decoding") != decoding:
                raise ValueError(f"Generation decoding differs for {arm_id}/{query_id}")
            expected_identity = _full_operation_identity(run, query, arm, result)
            if result.get("input_identity") != expected_identity:
                raise ValueError(f"Checkpoint input identity mismatch for {arm_id}/{query_id}")
            telemetry_tokens[key] = _validate_token_sidecar(run_root, result)
            total_token_rows += telemetry_tokens[key]
            trace = result.get("retrieval_candidates_and_ranking_trace", {})
            if not trace.get("trace_counters"):
                raise ValueError(f"Retrieval/context trace counters are missing for {arm_id}/{query_id}")
            if arm_id != "B6_full":
                baseline = results.get(("B6_full", query_id))
                if baseline is None and statuses.get(("B6_full", query_id)) == "SUCCESS":
                    baseline = _read_json_path(run_root / "baselines/B6_full/results" / f"{query_id}.json", "B6 result")
                if baseline and baseline.get("status") == "SUCCESS":
                    active = _intervention_trace_state(arm_id, baseline, off_arm=False)
                    disabled = _intervention_trace_state(arm_id, result, off_arm=True)
                    intervention_counts[arm_id]["reference_active"] += int(active)
                    intervention_counts[arm_id]["off_path_changed"] += int(disabled)
            results[key] = result
            prompt_hashes[key] = result["prompt_sha256"]
        else:
            if result.get("telemetry_sidecar"):
                telemetry_tokens[key] = _validate_token_sidecar(run_root, result)
                total_token_rows += telemetry_tokens[key]
            else:
                telemetry_tokens[key] = 0
            results[key] = result
            prompt_hashes[key] = result.get("prompt_sha256")

    counts = {"successful": sum(value == "SUCCESS" for value in statuses.values()),
              "failed": sum(value in {"FAILED", "PARTIAL"} for value in statuses.values()),
              "missing": sum(value == "MISSING" for value in statuses.values())}
    matrix_counts = run.get("matrix", {})
    if (matrix_counts.get("completed_query_arm_operations") != counts["successful"]
            or matrix_counts.get("failed_query_arm_operations") != counts["failed"]
            or matrix_counts.get("missing_query_arm_operations") != counts["missing"]
            or sum(counts.values()) != 5250):
        raise ValueError("Run manifest operation counts disagree with generation_results.jsonl")

    judge_manifest = _read_json_path(run_root / "judge/judge_manifest.json", "judge manifest")
    judge_rows = _read_jsonl_rows(run_root / "judge/judge_inputs.jsonl", "judge inputs")
    mapping = _read_json_path(run_root / "judge/arm_mapping.json", "judge arm mapping")
    if (judge_manifest.get("remote_requests_made") is not False
            or judge_manifest.get("judge_execution") != "not_run"
            or judge_manifest.get("expected_records") != 5250 or len(judge_rows) != 5250
            or judge_manifest.get("exported_records") != 5250
            or len(mapping.get("records", [])) != 5250):
        raise ValueError("Blinded judge export does not account for all 375 × 14 outcomes")
    by_blind_id = {item.get("blinded_response_id"): item for item in mapping.get("records", [])}
    seen_blind = set()
    for item in judge_rows:
        blind_id = item.get("blinded_response_id")
        if not isinstance(blind_id, str) or blind_id in seen_blind or blind_id not in by_blind_id:
            raise ValueError("Blinded judge response IDs are missing, duplicated or unmapped")
        seen_blind.add(blind_id)
        if "arm_id" in item or "arm" in item:
            raise ValueError("Judge input exposes the experimental arm label")
        actual = by_blind_id[blind_id]
        key = (actual.get("arm_id"), actual.get("query_id"))
        if key not in expected_pairs or item.get("query_id") != key[1]:
            raise ValueError("Judge arm mapping does not point to a planned query-arm operation")
        if item.get("response_sha256") != response_hashes[key] or actual.get("response_sha256") != response_hashes[key]:
            raise ValueError(f"Judge response hash differs from the canonical output for {key}")
        if item.get("operation_status") != statuses[key]:
            raise ValueError(f"Judge operation state differs from the canonical output for {key}")
        expected_truth = query_by_id[key[1]]["answerability_label"]
        if item.get("ground_truth_answerability_label") != expected_truth:
            raise ValueError("Judge export changed a ground-truth answerability label")
        if item.get("scoring_eligible") is not (statuses[key] == "SUCCESS" and bool(item.get("raw_output", "").strip())):
            raise ValueError("Judge scoring eligibility differs from the stored generation outcome")
        if statuses[key] == "SUCCESS" and item.get("model_and_prompt_hashes", {}).get("prompt_sha256") != prompt_hashes[key]:
            raise ValueError("Judge prompt hash differs from the canonical stored prompt")
    if len(seen_blind) != 5250 or len(by_blind_id) != 5250:
        raise ValueError("Judge blinded IDs are duplicated or missing")

    telemetry_schema = _read_json_path(run_root / "metadata/token_telemetry_schema.json", "token telemetry schema")
    units = telemetry_schema.get("units", {})
    if (telemetry_schema.get("format") != "Apache Parquet 2.6" or units.get("entropy") != "bits"
            or units.get("log_probability") != "nats" or units.get("probability") != "fraction"):
        raise ValueError("Token telemetry schema or physical units are missing or unsupported")
    required_columns = {
        "run_id", "arm_id", "query_id", "generation_purpose", "token_index", "token_id",
        "token_text", "logprob", "nll_nats", "nll_bits_per_token", "entropy_k20_bits",
        "entropy_k50_bits", "entropy_k100_bits", "retained_mass_k20", "retained_mass_k50",
        "retained_mass_k100", "top1_prob_full", "logit_margin", "top_logit_margins_json",
        "telemetry_device", "stop_state",
    }
    if not required_columns.issubset(set(telemetry_schema.get("columns", []))):
        raise ValueError("Token telemetry schema omits one or more required per-token fields")
    parquet_paths = [run_root / "token_telemetry.parquet"] + [
        run_root / "baselines" / arm_id / "token_telemetry.parquet" for arm_id in FULL_MATRIX_ARM_IDS]
    for path in parquet_paths:
        if not path.is_file() or path.stat().st_size < 8:
            raise ValueError(f"Required Parquet token telemetry is missing or empty: {path.relative_to(run_root)}")
        with path.open("rb") as handle:
            if handle.read(4) != b"PAR1":
                raise ValueError(f"Token telemetry is not Parquet: {path.relative_to(run_root)}")
            handle.seek(-4, os.SEEK_END)
            if handle.read(4) != b"PAR1":
                raise ValueError(f"Token telemetry Parquet footer is corrupt: {path.relative_to(run_root)}")
    report_path = run_root / "completion_report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    expected_report_status = run.get("status")
    if f"Status: **{expected_report_status}**" not in report:
        raise ValueError("Completion report status does not match the run manifest")
    arm_statuses = {row.get("arm_id"): row for row in run.get("arms", [])}
    integrity = {}
    for arm_id in FULL_MATRIX_ARM_IDS:
        manifest_arm = arm_statuses.get(arm_id, {})
        completion = manifest_arm.get("completion") or {}
        expected_count = sum(statuses[(arm_id, query_id)] == "SUCCESS" for query_id in ids)
        expected_failed = sum(statuses[(arm_id, query_id)] in {"FAILED", "PARTIAL"} for query_id in ids)
        expected_missing = sum(statuses[(arm_id, query_id)] == "MISSING" for query_id in ids)
        if (completion.get("planned") != 375 or completion.get("complete") != expected_count
                or completion.get("failed") != expected_failed or completion.get("missing") != expected_missing):
            raise ValueError(f"Per-arm completion accounting is inconsistent for {arm_id}")
        active = intervention_counts[arm_id]["reference_active"]
        changed = intervention_counts[arm_id]["off_path_changed"]
        if arm_id == "B6_full":
            integrity_status = "REFERENCE"
        elif expected_count != 375:
            integrity_status = "INCOMPLETE"
        elif active == 0 or changed == 0:
            integrity_status = "SCIENTIFIC_VALIDATION_FAILED"
        else:
            integrity_status = "PASSED"
        declared = manifest_arm.get("intervention_integrity", {}).get("status")
        if declared != integrity_status:
            raise ValueError(f"Intervention trace integrity status is inconsistent for {arm_id}: {declared!r}/{integrity_status}")
        integrity[arm_id] = {"status": integrity_status,
                             "baseline_active_queries": active,
                             "off_path_changed_queries": changed}
    complete = (run["status"] == "COMPLETE" and counts == {"successful": 5250, "failed": 0, "missing": 0}
                and all(value["status"] in {"REFERENCE", "PASSED"} for value in integrity.values()))
    if run["status"] == "COMPLETE" and not complete:
        raise ValueError("Run claims COMPLETE despite failed operations or intervention validation")
    if run["status"] == "PARTIAL" and complete:
        raise ValueError("Run is marked PARTIAL even though all planned operations validate as complete")
    return {"run_id": run["run_id"], "status": run["status"], "complete": complete,
            "counts": counts, "planned_query_arm_operations": 5250,
            "query_count": 375, "arm_count": 14,
            "total_cuda_token_rows": total_token_rows,
            "intervention_integrity": integrity,
            "archive_sha256": archive_sha256}


def _import_full_archive(archive: Path, archive_sha256: str, *,
                         destination: Path | str | None = None,
                         input_manifest_path: Path | None = None) -> dict[str, Any]:
    import_root = Path(tempfile.mkdtemp(prefix="ablation_full_import_"))
    try:
        run_root, archive_manifest, run = _extract_full_archive(archive, import_root)
        summary = _validate_full_results(run_root, run, archive_sha256,
                                         input_manifest_path=input_manifest_path)
        if archive_manifest.get("run_id") != run.get("run_id"):
            raise ValueError("Archive manifest run identity differs from the run manifest")
        if destination is None:
            destination = PACKAGE_ROOT / "imported" / "full_cuda_ablation" / str(run["run_id"])
        destination = Path(destination).resolve()
        marker_path = destination / "IMPORTED_ARCHIVE.json"
        if destination.exists():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                marker = {}
            if marker.get("archive_sha256") != archive_sha256:
                raise FileExistsError(f"Import destination already exists with different content: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_root, destination)
            atomic = {
                "schema_version": "1.0", "archive_sha256": archive_sha256,
                "run_id": run["run_id"], "imported_status": run["status"],
                "complete": summary["complete"],
            }
            (destination / "IMPORTED_ARCHIVE.json").write_text(
                json.dumps(atomic, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        preview_lines = ["# Imported CUDA ablation results", "",
                         f"- Run: `{summary['run_id']}`",
                         f"- Status: **{summary['status']}**",
                         f"- Complete: **{summary['complete']}**",
                         f"- Query-arm operations: {summary['counts']['successful']} successful, {summary['counts']['failed']} failed, {summary['counts']['missing']} missing of 5,250.",
                         f"- CUDA token telemetry rows: {summary['total_cuda_token_rows']}",
                         "- Independent LLM judge: pending; no remote requests were made.", ""]
        (destination / "import_summary.md").write_text("\n".join(preview_lines), encoding="utf-8")
        summary["verified"] = True
        summary["archive"] = str(archive)
        summary["imported_to"] = str(destination)
        summary["analysis_namespace"] = "full_cuda_ablation"
        return summary
    finally:
        shutil.rmtree(import_root, ignore_errors=True)


def verify_result_archive(archive: Path | str, *, expected_sha256: str | None = None,
                          input_manifest_path: Path | str | None = None,
                          questions_path: Path | str | None = None,
                          destination: Path | str | None = None) -> dict[str, Any]:
    archive = Path(archive).resolve(strict=True)
    archive_sha256 = _read_external_digest(archive, expected_sha256)
    if _full_manifest_probe(archive):
        input_path = Path(input_manifest_path).resolve(strict=True) if input_manifest_path else None
        return _import_full_archive(archive, archive_sha256,
                                    destination=destination,
                                    input_manifest_path=input_path)
    files, run = _read_tar(archive)
    if input_manifest_path is None:
        candidate = PACKAGE_ROOT / "releases" / f"ablation_cuda_input_v{__version__}" / "PACKAGE_MANIFEST.json"
        if (candidate.is_file()
                and run.get("input_package_manifest_sha256") == sha256_file(candidate)):
            input_manifest_path = candidate
    input_manifest_path = Path(input_manifest_path).resolve(strict=True) if input_manifest_path else None
    questions_path = Path(questions_path).resolve(strict=True) if questions_path else None
    arms, _ = validate_results(files, run, input_manifest_path, questions_path)
    preview = _preview(arms, run)
    imported_to = None
    if destination is None:
        default_root = PACKAGE_ROOT / "imported"
        destination = default_root / str(run["run_id"])
    destination = Path(destination).resolve()
    if destination.exists():
        marker = destination / "IMPORTED_ARCHIVE.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("archive_sha256") == archive_sha256:
            imported_to = str(destination)
        else:
            raise FileExistsError(f"Import destination already exists with different content: {destination}")
    else:
        destination.mkdir(parents=True)
        for name, data in files.items():
            output = destination.joinpath(*PurePosixPath(name).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
        (destination / "preview.md").write_text(preview, encoding="utf-8")
        (destination / "IMPORTED_ARCHIVE.json").write_text(json.dumps({
            "schema_version": "1.0", "archive_sha256": archive_sha256,
            "run_id": run["run_id"], "imported_status": "VERIFIED",
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        imported_to = str(destination)
    return {"verified": True, "archive": str(archive), "archive_sha256": archive_sha256,
            "run_id": run["run_id"], "status": run["status"],
            "query_id": run["query_id"], "gpu": run["device_preflight"]["torch_cuda_operation"]["device"],
            "arms": [{"arm": arm["arm"], "parser_status": arm.get("parser_status"),
                      "answer": arm.get("parsed_answer"), "stable_result_id": arm["stable_result_id"]}
                     for arm in arms],
            "imported_to": imported_to, "preview": preview}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and import a Biomedical RAG CUDA result archive")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_result_archive(
        args.archive, expected_sha256=args.expected_sha256,
        input_manifest_path=args.input_manifest, questions_path=args.questions,
        destination=args.destination,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
