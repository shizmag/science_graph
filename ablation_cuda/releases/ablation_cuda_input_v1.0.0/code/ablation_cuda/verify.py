"""Independent, standard-library-only verifier and importer for collaborator results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
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
TELEMETRY_REQUIRED = {
    "token_id", "token_text", "logprob", "nll_nats", "nll_bits_per_token",
    "entropy_k20_bits", "entropy_k50_bits", "entropy_k100_bits",
    "retained_mass_k20", "retained_mass_k50", "retained_mass_k100",
    "logit_margin", "top_logit_margins", "telemetry_device", "stop_state",
}


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


def verify_result_archive(archive: Path | str, *, expected_sha256: str | None = None,
                          input_manifest_path: Path | str | None = None,
                          questions_path: Path | str | None = None,
                          destination: Path | str | None = None) -> dict[str, Any]:
    archive = Path(archive).resolve(strict=True)
    archive_sha256 = _read_external_digest(archive, expected_sha256)
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
