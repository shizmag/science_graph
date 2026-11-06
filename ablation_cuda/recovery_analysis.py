"""Reproducible, read-only analysis of the archived CUDA ablation results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1] / "graphs/run_20261020T065029857103Z_OCC-RAG-1.7B_CUDA_ABLATION"
DEFAULT_ARMS = (
    "B6_full",
    "B6_minus_dynamic_graph_expander",
    "B6_minus_parent_deduplication",
    "B6_minus_static_graph_context",
)
EXPECTED_ANSWERABLE = 300


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def classify_failure(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing_result"
    status = str(row.get("status", "")).upper()
    if status in {"SUCCESS", "SUCCEEDED", "COMPLETE"}:
        return "success"
    errors = row.get("result_errors") or row.get("errors") or []
    if isinstance(errors, dict):
        errors = [errors]
    text = " ".join(str(v) for err in errors for v in (err.values() if isinstance(err, dict) else [err])).lower()
    stages = row.get("stage_status") or {}
    # Specific operational failures take precedence over the generic "generation"
    # stage. In particular, "room" contains the substring "oom"; never classify
    # context overflow as an out-of-memory failure by substring matching.
    if any(term in text for term in ("serialized prompt has", "context overflow", "context limit leaves no room", "prompt_too_long")):
        return "context_overflow"
    failed_stages = [str(k).lower() for k, v in stages.items() if str(v).lower() not in {"success", "succeeded", "complete", "skipped", "not_required", "none"}] if isinstance(stages, dict) else []
    import re
    if any(term in text for term in ("out of memory", "cuda out of memory")) or re.search(r"\boom\b", text):
        return "generation_oom" if any("generation" in str(e).lower() for e in errors) or "generation" in failed_stages else "resource_oom"
    if any("retriev" in s for s in failed_stages) or "retriev" in text:
        return "retrieval_failure"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any("generation" in s for s in failed_stages) or "generation" in text:
        return "generation_failure"
    if any("parse" in s for s in failed_stages) or "parser" in text:
        return "parse_failure"
    return "other_failure"


def mcnemar_exact(reference: Sequence[bool | int], treatment: Sequence[bool | int]) -> dict[str, Any]:
    if len(reference) != len(treatment):
        raise ValueError("paired vectors must have equal length")
    both_success = both_failure = reference_only = treatment_only = 0
    for r, t in zip(reference, treatment):
        r, t = bool(r), bool(t)
        if r and t: both_success += 1
        elif not r and not t: both_failure += 1
        elif r: reference_only += 1
        else: treatment_only += 1
    n = reference_only + treatment_only
    tail = sum(math.comb(n, k) for k in range(min(reference_only, treatment_only) + 1)) / (2 ** n) if n else 1.0
    return {"both_success": both_success, "both_failure": both_failure, "reference_only": reference_only,
            "treatment_only": treatment_only, "discordant": n, "p_value": min(1.0, 2 * tail)}


def paired_bootstrap_ci(reference: Sequence[float], treatment: Sequence[float], *, seed: int = 20261106,
                        n_resamples: int = 10000, confidence: float = 0.95) -> dict[str, Any]:
    if len(reference) != len(treatment):
        raise ValueError("paired vectors must have equal length")
    differences = [float(t) - float(r) for r, t in zip(reference, treatment)]
    n = len(differences)
    if not n:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_pairs": 0, "seed": seed, "n_resamples": n_resamples}
    rng = random.Random(seed)
    estimates = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)]
    estimates.sort()
    alpha = (1 - confidence) / 2
    return {"estimate": sum(differences) / n, "ci_low": estimates[max(0, int(alpha * n_resamples))],
            "ci_high": estimates[min(n_resamples - 1, int((1 - alpha) * n_resamples))],
            "n_pairs": n, "seed": seed, "n_resamples": n_resamples, "confidence": confidence}


def _load_arm(archive: Path, arm: str) -> dict[str, dict[str, Any]]:
    root = archive / "baselines" / arm / "results"
    if not root.is_dir():
        raise ValueError(f"arm result directory missing: {root}")
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        row = _read_json(path)
        query_id = str(row.get("query_id", path.stem))
        if query_id in rows:
            raise ValueError(f"duplicate query {query_id} in arm {arm}")
        rows[query_id] = row
    return rows


def _number(row: dict[str, Any] | None, *keys: str) -> float | None:
    if row is None:
        return None
    for key in keys:
        v = row.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def analyze_archive(archive: Path | str = DEFAULT_ARCHIVE, *, arms: Sequence[str] = DEFAULT_ARMS,
                    expected_answerable: int = EXPECTED_ANSWERABLE, bootstrap_resamples: int = 10000,
                    seed: int = 20261106) -> dict[str, Any]:
    archive = Path(archive).resolve()
    manifest = _read_json(archive / "arms_manifest.json")
    manifest_arms = {entry["arm_id"]: entry for entry in manifest["arms"]}
    if len(arms) != 4:
        raise ValueError("exactly four arms are required")
    absent = [arm for arm in arms if arm not in manifest_arms]
    if absent:
        raise ValueError(f"arms absent from manifest: {absent}")
    query_count = int(manifest.get("query_count", 0))
    expected_ops = int(manifest.get("expected_query_arm_operations", 0))
    if query_count <= 0 or expected_ops != query_count * int(manifest.get("arm_count", 0)):
        raise ValueError("manifest expected query-arm operation count is inconsistent")
    data = {arm: _load_arm(archive, arm) for arm in arms}
    for arm in arms:
        expected_rows = int((manifest_arms[arm].get("completion") or {}).get("planned", query_count))
        if expected_rows != query_count or len(data[arm]) != expected_rows:
            raise ValueError(f"archived result count mismatch for {arm}: expected {expected_rows}, found {len(data[arm])}")
    labels: dict[str, str] = {}
    for arm_rows in data.values():
        for q, row in arm_rows.items():
            label = row.get("answerability_label")
            if label:
                if q in labels and labels[q] != label:
                    raise ValueError(f"answerability label mismatch for {q}")
                labels[q] = label
    answerable = sorted(q for q, label in labels.items() if label == "answerable")
    if len(answerable) != expected_answerable:
        raise ValueError(f"expected {expected_answerable} answerable IDs, found {len(answerable)}")
    ids = sorted(labels)
    if len(ids) != query_count:
        raise ValueError(f"expected {query_count} labeled query IDs, found {len(ids)}")
    arm_report = {}
    for arm in arms:
        rows = data[arm]
        failures = Counter(classify_failure(rows.get(q)) for q in ids)
        completion = manifest_arms[arm].get("completion") or {}
        if failures.get("success", 0) != int(completion.get("complete", failures.get("success", 0))):
            raise ValueError(f"archived success count mismatch for {arm}: manifest says {completion.get('complete')}, results show {failures.get('success', 0)}")
        if sum(v for k, v in failures.items() if k != "success") != int(completion.get("failed", sum(v for k, v in failures.items() if k != "success"))):
            raise ValueError(f"archived failure count mismatch for {arm}")
        success = [classify_failure(rows.get(q)) == "success" for q in answerable]
        available = sum(success)
        times = [_number(rows.get(q, {}).get("timing", {}) if rows.get(q) else None, "elapsed_seconds") for q in answerable]
        tokens = [_number(rows.get(q), "total_pipeline_generated_tokens", "cumulative_new_cuda_tokens") for q in answerable]
        arm_report[arm] = {
            "manifest_completion": manifest_arms[arm].get("completion"),
            "result_rows": len(rows),
            "availability": {"answerable_n": len(answerable), "available": available, "unavailable": len(answerable)-available,
                             "rate": available / len(answerable) if answerable else None},
            "failure_classifications_all_labeled_ids": dict(sorted(failures.items())),
            "operational": {"latency_observed_n": sum(v is not None for v in times), "latency_mean_seconds": (sum(v for v in times if v is not None) / sum(v is not None for v in times)) if any(v is not None for v in times) else None,
                             "generated_tokens_observed_n": sum(v is not None for v in tokens), "generated_tokens_mean": sum(v for v in tokens if v is not None) / max(1, sum(v is not None for v in tokens))},
        }
    comparisons = {}
    reference = arms[0]
    for treatment in arms[1:]:
        r_avail = [classify_failure(data[reference].get(q)) == "success" for q in answerable]
        t_avail = [classify_failure(data[treatment].get(q)) == "success" for q in answerable]
        both = [q for q, r, t in zip(answerable, r_avail, t_avail) if r and t]
        r_times = [_number(data[reference].get(q), "total_pipeline_generated_tokens", "cumulative_new_cuda_tokens") for q in both]
        t_times = [_number(data[treatment].get(q), "total_pipeline_generated_tokens", "cumulative_new_cuda_tokens") for q in both]
        paired_cost = [(r, t) for r, t in zip(r_times, t_times) if r is not None and t is not None]
        comparisons[f"{reference}__vs__{treatment}"] = {
            "availability": mcnemar_exact(r_avail, t_avail),
            "availability_rate_difference_ci": paired_bootstrap_ci(r_avail, t_avail, seed=seed, n_resamples=bootstrap_resamples),
            "quality": {"metric": "Semantic Accuracy", "status": "judge_not_run", "n_pairs": 0, "mean_difference": None, "ci": None,
                        "missing_scores_are_not_zero_imputed": True},
            "paired_token_cost_on_both_available": paired_bootstrap_ci([x[0] for x in paired_cost], [x[1] for x in paired_cost], seed=seed, n_resamples=bootstrap_resamples),
            "jointly_available_ids_n": len(both),
        }
    judge_manifest_path = archive / "judge" / "judge_manifest.json"
    judge_manifest = _read_json(judge_manifest_path) if judge_manifest_path.exists() else None
    judge_rows = _read_jsonl(archive / "judge" / "judge_inputs.jsonl")
    arm_mapping_path = archive / "judge" / "arm_mapping.json"
    arm_mapping = _read_json(arm_mapping_path).get("records", []) if arm_mapping_path.exists() else []
    blinded_to_arm = {item.get("blinded_response_id"): item.get("arm_id") for item in arm_mapping}
    judge_by_arm_id = {(blinded_to_arm.get(r.get("blinded_response_id")), r.get("query_id")): r for r in judge_rows}
    judge_missing = {arm: {"expected": len(answerable), "input_records": sum((arm, q) in judge_by_arm_id for q in answerable),
                            "eligible_inputs": sum((arm, q) in judge_by_arm_id and judge_by_arm_id[(arm, q)].get("scoring_eligible") is True for q in answerable),
                            "scored_records": 0, "missing_scores": len(answerable),
                            "execution_status": (judge_manifest or {}).get("judge_execution", "not_run")} for arm in arms}
    static_activation = {"comparison": "B6_full vs B6_minus_static_graph_context", "paired_records": 0,
                         "prompt_changed": 0, "generator_context_changed": 0, "evidence_identifiers_changed": 0,
                         "activation_status": "unverified"}
    if "B6_full" in arms and "B6_minus_static_graph_context" in arms and arm_mapping:
        by_pair = {}
        for row in judge_rows:
            arm = blinded_to_arm.get(row.get("blinded_response_id"))
            if arm in {"B6_full", "B6_minus_static_graph_context"} and row.get("query_id") in labels:
                by_pair.setdefault(row["query_id"], {})[arm] = row
        for qid in sorted(by_pair):
            pair = by_pair[qid]
            if set(pair) != {"B6_full", "B6_minus_static_graph_context"}:
                continue
            static_activation["paired_records"] += 1
            full, treatment = pair["B6_full"], pair["B6_minus_static_graph_context"]
            if full.get("serialized_generation_prompt") != treatment.get("serialized_generation_prompt"):
                static_activation["prompt_changed"] += 1
            if full.get("actual_generator_visible_context") != treatment.get("actual_generator_visible_context"):
                static_activation["generator_context_changed"] += 1
            if full.get("context_block_identifiers") != treatment.get("context_block_identifiers"):
                static_activation["evidence_identifiers_changed"] += 1
        static_activation["activation_status"] = "inactive_in_saved_prompt_evidence" if static_activation["prompt_changed"] == 0 else "prompt_changed"
    utility = {arm: {"availability": arm_report[arm]["availability"], "conditional_quality_mean": None,
                     "quality_mean": None, "quality_observed_n": 0, "quality_missingness": "not imputed; no judge execution/results in archive",
                     "operational_composite": {"name": "generation_success_times_semantic_accuracy", "value": None,
                                               "formula": "success * Semantic Accuracy; overflow contributes 0 only after successful outputs are judged",
                                               "status": "pending_judge_scores"},
                     "latency_mean_seconds": arm_report[arm]["operational"]["latency_mean_seconds"],
                     "generated_tokens_mean": arm_report[arm]["operational"]["generated_tokens_mean"]} for arm in arms}
    files = []
    for path in sorted(archive.rglob("*")):
        if path.is_file() and path.name != ".DS_Store":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
            files.append({"path": path.relative_to(archive).as_posix(), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size})
    return {"schema_version": "1.0", "archive": str(archive), "population": {"answerable_ids": answerable, "answerable_n": len(answerable), "all_labeled_ids_n": len(ids), "expected_answerable": expected_answerable},
            "arms": arm_report, "paired_comparisons": comparisons,
            "static_context_activation": static_activation,
            "judge_missingness": {"status": (judge_manifest or {}).get("judge_execution", "not_run"), "manifest": judge_manifest, "by_arm": judge_missing},
            "operational_utility": utility,
            "input_registry": {"hash_algorithm": "sha256", "files": files}}


def build_operational_input_registry(archive: Path | str, *, arms: Sequence[str] = DEFAULT_ARMS,
                                     expected_answerable: int = EXPECTED_ANSWERABLE) -> list[dict[str, Any]]:
    """Build one auditable row per answerable query-arm input, without judge imputation."""
    archive = Path(archive).resolve()
    report = analyze_archive(archive, arms=arms, expected_answerable=expected_answerable, bootstrap_resamples=100)
    answerable = report["population"]["answerable_ids"]
    judge_path = archive / "judge" / "judge_manifest.json"
    judge_manifest = _read_json(judge_path) if judge_path.exists() else {"judge_execution": "not_run"}
    rows: list[dict[str, Any]] = []
    for arm in arms:
        by_id = _load_arm(archive, arm)
        for query_id in answerable:
            result = by_id[query_id]
            status = classify_failure(result)
            response = result.get("raw_generated_response") or ""
            prompt = result.get("prompt") or ""
            selected = result.get("selected_evidence") or []
            rows.append({
                "query_id": query_id, "arm_id": arm,
                "answerability_label": result.get("answerability_label"),
                "generation_status": status,
                "result_path": result.get("canonical_result"),
                "prompt_sha256": result.get("prompt_sha256"),
                "prompt_payload_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt else None,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest() if response else None,
                "selected_evidence_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if selected else None,
                "reference_answer_sha256": hashlib.sha256(str(result.get("gold_reference_answer", "")).encode()).hexdigest(),
                "judge_status": "not_run" if judge_manifest.get("judge_execution") == "not_run" else "check_saved_score_records",
                "judge_score_present": False,
                "semantic_accuracy": None,
                "operational_composite": None,
                "operational_composite_definition": "success * Semantic Accuracy; explicit failure-as-zero composite, not Semantic Accuracy",
            })
    return rows

def export_registry(archive: Path | str, output: Path | str, *, arms: Sequence[str] = DEFAULT_ARMS,
                    expected_answerable: int = EXPECTED_ANSWERABLE) -> int:
    rows = build_operational_input_registry(archive, arms=arms, expected_answerable=expected_answerable)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in rows:
            stream.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
    return len(rows)

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--arms", nargs=4, default=list(DEFAULT_ARMS), metavar="ARM")
    parser.add_argument("--expected-answerable", type=int, default=EXPECTED_ANSWERABLE)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--output", type=Path, help="write analysis JSON here; stdout otherwise")
    parser.add_argument("--export-registry", type=Path, help="write deterministic SHA-256 input registry JSONL")
    args = parser.parse_args(argv)
    try:
        report = analyze_archive(args.archive, arms=args.arms, expected_answerable=args.expected_answerable,
                                 bootstrap_resamples=args.bootstrap_resamples, seed=args.seed)
        payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
        if args.export_registry:
            export_registry(args.archive, args.export_registry, arms=args.arms, expected_answerable=args.expected_answerable)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
