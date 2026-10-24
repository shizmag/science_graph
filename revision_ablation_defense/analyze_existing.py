#!/usr/bin/env python3
"""Rebuild and analyze preserved CUDA ablation artifacts; no network/model calls."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20261020T065029857103Z"
RUN = ROOT / "graphs" / f"run_{RUN_ID}_OCC-RAG-1.7B_CUDA_ABLATION"
JUDGE = ROOT / "reports/ablation_luna6_low" / f"run_{RUN_ID}"
OUT = Path(__file__).resolve().parent
TABLES = OUT / "tables"
METRICS = ("semantic_accuracy", "answer_relevance", "faithfulness", "citation_fidelity")
ARM_ORDER = [
    "B6_full", "B6_minus_score_blending", "B6_minus_final_reranking",
    "B6_minus_dynamic_graph_expander", "B6_minus_static_graph_context",
    "B6_minus_intrapaper_graph_candidates", "B6_minus_concept_retrieval",
    "B6_minus_bridge_retrieval", "B6_minus_parent_deduplication",
    "B6_minus_lexical_retrieval", "B6_minus_dense_retrieval",
    "B6_fixed_budget_context", "B6_interaction_graph_candidate_x_dynamic_expander",
    "B6_interaction_reranker_x_blending",
]


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read valid JSON from {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read valid JSONL from {path}: {exc}") from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canon(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to write {path}: {exc}") from exc


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to write table {path}: {exc}") from exc


def score_metric(score_row: dict[str, Any] | None, metric: str) -> float | None:
    if score_row is None or score_row.get("status") != "scored":
        return None
    return float(score_row["metrics"][metric])


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile sample")
    return ordered[min(len(ordered) - 1, int(math.floor(fraction * (len(ordered) - 1))))]


def paired_bootstrap(deltas: list[float], *, seed: int, iterations: int = 10_000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(iterations)]
    return percentile(means, 0.025), percentile(means, 0.975)

def sign_test_p(deltas: list[float]) -> float:
    pos = sum(value > 1e-12 for value in deltas)
    neg = sum(value < -1e-12 for value in deltas)
    n = pos + neg
    if not n:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def rank_biserial(deltas: list[float]) -> float | None:
    nonzero = [(abs(value), 1 if value > 0 else -1) for value in deltas if abs(value) > 1e-12]
    if not nonzero:
        return 0.0
    ordered = sorted(range(len(nonzero)), key=lambda i: nonzero[i][0])
    ranks = [0.0] * len(nonzero)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and abs(nonzero[ordered[end]][0] - nonzero[ordered[index]][0]) <= 1e-12:
            end += 1
        average_rank = (index + 1 + end) / 2
        for position in ordered[index:end]:
            ranks[position] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, (_, sign) in zip(ranks, nonzero, strict=True) if sign > 0)
    negative_rank_sum = sum(rank for rank, (_, sign) in zip(ranks, nonzero, strict=True) if sign < 0)
    total = positive_rank_sum + negative_rank_sum
    return (positive_rank_sum - negative_rank_sum) / total if total else 0.0

def holm_adjust(rows: list[dict[str, Any]], p_key: str, out_key: str, family: str) -> None:
    valid = sorted(((row[p_key], row) for row in rows if row.get(p_key) is not None), key=lambda item: item[0])
    m = len(valid)
    running = 0.0
    for rank, (pvalue, row) in enumerate(valid):
        adjusted = min(1.0, (m - rank) * pvalue)
        running = max(running, adjusted)
        row[out_key] = running
        row["multiplicity_family"] = family


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (sx * sy)


def load_token_means() -> dict[tuple[str, str], dict[str, float]]:
    """Read saved token telemetry only; does not import generation/runtime code."""
    try:
        pyarrow_modules = {name: importlib.import_module(name) for name in ("pyarrow.compute", "pyarrow.parquet")}
    except ImportError as exc:
        raise RuntimeError("Reading saved token_telemetry.parquet requires local pyarrow") from exc
    pc = pyarrow_modules["pyarrow.compute"]
    pq = pyarrow_modules["pyarrow.parquet"]
    columns = ["arm_id", "query_id", "generation_purpose", "token_index",
               "entropy_k20_bits", "entropy_k50_bits", "entropy_k100_bits",
               "retained_mass_k20", "retained_mass_k50", "retained_mass_k100"]
    table = pq.read_table(RUN / "token_telemetry.parquet", columns=columns)
    table = table.filter(pc.equal(table["generation_purpose"], "final_answer_generation"))
    measures = ["entropy_k20_bits", "entropy_k50_bits", "entropy_k100_bits",
                "retained_mass_k20", "retained_mass_k50", "retained_mass_k100"]
    grouped = table.group_by(["arm_id", "query_id"]).aggregate(
        [(measure, "mean") for measure in measures] + [("token_index", "count")]
    )
    output: dict[tuple[str, str], dict[str, float]] = {}
    for row in grouped.to_pylist():
        values = {measure: float(row[f"{measure}_mean"]) for measure in measures}
        values["telemetry_token_count"] = float(row["token_index_count"])
        output[(row["arm_id"], row["query_id"])] = values
    if len(output) != 4265:
        raise AssertionError(f"Expected telemetry for 4,265 successful operations, found {len(output)}")
    return output


def load_sources() -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    questions = {row["query_id"]: row for row in read_jsonl(RUN / "questions.jsonl")}
    metric_rows = {(row["arm_id"], row["query_id"]): row for row in csv.DictReader((RUN / "metrics_details.csv").open(encoding="utf-8"))}
    errors = {(row["arm_id"], row["query_id"]): row for row in read_jsonl(RUN / "logs/errors.jsonl")}
    mapping_doc = read_json(RUN / "judge/arm_mapping.json")
    mapping = {row["blinded_response_id"]: row for row in mapping_doc["records"]}
    inputs = {row["blinded_response_id"]: row for row in read_jsonl(RUN / "judge/judge_inputs.jsonl")}
    score_rows = {row["blinded_response_id"]: row for row in read_jsonl(JUDGE / "judge_scores_by_response.jsonl")}
    generation_rows = {(row["arm_id"], row["query_id"]): row for row in read_jsonl(RUN / "generation_results.jsonl")}
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in ARM_ORDER:
        path = RUN / "baselines" / arm / "retrieved_contexts.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                contexts[(arm, row["query_id"])] = row
        else:
            result_dir = RUN / "baselines" / arm / "results"
            for result_path in result_dir.glob("*.json"):
                result = read_json(result_path)
                trace = result.get("retrieval_candidates_and_ranking_trace", {})
                assembly = trace.get("context_and_prompt_assembly", {})
                contexts[(arm, result["query_id"])] = {
                    "prompt_sha256": assembly.get("serialized_prompt_sha256"),
                    "context_payload_sha256": None,
                    "selected_chunk_ids": assembly.get("context_chunk_ids_after_trimming", []),
                    "context_hash_source": "result trace fallback; no separate retrieved_contexts export",
                }
    mapping_by_operation = {(row["arm_id"], row["query_id"]): row for row in mapping.values()}
    for blinded_id, input_row in inputs.items():
        operation = (mapping[blinded_id]["arm_id"], mapping[blinded_id]["query_id"])
        contexts.setdefault(operation, {})["actual_evaluator_context"] = input_row.get("actual_evaluator_context")
    token_means = load_token_means()
    return questions, metric_rows, errors, {"mapping": mapping, "mapping_by_operation": mapping_by_operation,
                                             "inputs": inputs, "scores": score_rows, "contexts": contexts,
                                             "generation_rows": generation_rows, "token_means": token_means}


def build_canonical(questions: dict[str, Any], details: dict[tuple[str, str], dict[str, Any]], errors: dict[tuple[str, str], dict[str, Any]], judge: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    manifest = read_json(RUN / "arms_manifest.json")
    arm_manifest = {row["arm_id"]: row for row in manifest["arms"]}
    rows: list[dict[str, Any]] = []
    results: dict[tuple[str, str], dict[str, Any]] = {}
    audit = Counter()
    identities: dict[str, set[str]] = defaultdict(set)
    raw_hash_bad: list[str] = []
    missing_results: list[tuple[str, str]] = []
    for arm in ARM_ORDER:
        for query_id, question in questions.items():
            key = (arm, query_id)
            detail = details.get(key)
            if detail is None:
                raise AssertionError(f"Missing metrics row {key}")
            result_path = RUN / "baselines" / arm / "results" / f"{query_id}.json"
            if not result_path.exists():
                missing_results.append(key)
                result: dict[str, Any] = {}
            else:
                result = read_json(result_path)
            results[key] = result
            status = detail["operation_status"]
            available = status == "SUCCESS"
            if result:
                identities["snapshot"].add(result.get("snapshot_identity", {}).get("snapshot_sha256", ""))
                identities["dataset"].add(result.get("snapshot_identity", {}).get("dataset_sha256", ""))
                identities["model_manifest"].add(result.get("model_manifest_sha256", ""))
                identities["effective_config"].add(result.get("effective_configuration_hash", ""))
                if result.get("arm") != arm or result.get("query_id") != query_id:
                    raise AssertionError(f"Result identity mismatch {key}")
            error = errors.get(key, {})
            error_text = error.get("error", "")
            match = re.search(r"Serialized prompt has (\d+) tokens", error_text)
            prompt_tokens = int(match.group(1)) if match else None
            final_event = next((event.get("event", {}) for event in result.get("generation_events", [])
                                if event.get("purpose") == "final_answer_generation"), {})
            if prompt_tokens is None and final_event:
                prompt_tokens = final_event.get("prompt_token_count")
            raw_response = result.get("raw_generated_response")
            raw_hash = sha256_bytes(raw_response.encode("utf-8")) if isinstance(raw_response, str) else None
            if available:
                audit["success_rows"] += 1
                generation_record = judge["generation_rows"].get(key)
                if generation_record is None or generation_record.get("response_sha256") != raw_hash:
                    raw_hash_bad.append(f"{arm}:{query_id}:generation_manifest")
                if not raw_response:
                    audit["success_missing_response"] += 1
                if raw_hash != detail.get("response_sha256") and detail.get("response_sha256"):
                    raw_hash_bad.append(f"{arm}:{query_id}")
            elif status == "PARTIAL":
                audit["failed_rows"] += 1
            maprow = judge["mapping_by_operation"].get(key)
            if maprow is None:
                raise AssertionError(f"Missing judge map {key}")
            inp = judge["inputs"].get(maprow["blinded_response_id"])
            score = judge["scores"].get(maprow["blinded_response_id"])
            context_record = judge["contexts"].get(key)
            if available:
                audit["eligible_rows"] += 1
                if not inp or not inp.get("scoring_eligible"):
                    audit["eligible_missing_input"] += 1
                if not score or score.get("status") != "scored":
                    audit["eligible_missing_score"] += 1
                elif (score.get("response_sha256") != raw_hash or score.get("query_id") != query_id
                      or score.get("blinded_response_id") != maprow["blinded_response_id"]):
                    audit["score_identity_mismatch"] += 1
            else:
                audit["ineligible_rows"] += 1
                if score and score.get("status") == "scored":
                    audit["failed_scored"] += 1
            if maprow["effective_configuration_hash"] != (result.get("effective_configuration_hash") if result else arm_manifest[arm].get("effective_configuration_hash")):
                audit["config_hash_mismatch"] += 1
            token_row = judge["token_means"].get(key, {})
            row = {
                "query_id": query_id,
                "arm_id": arm,
                "answerability_label": question["answerability_label"],
                "source_pmid": question.get("source_pmid"),
                "operation_status": status,
                "failure_reason": error_text or None,
                "prompt_tokens": prompt_tokens,
                "context_tokens": final_event.get("prompt_token_count") if final_event else prompt_tokens,
                "context_chars": (result.get("retrieval_candidates_and_ranking_trace", {}).get("context_capture", {}).get("context_trim", {}).get("input_text_chars") if result else None),
                "generated_tokens": detail.get("final_generation_tokens") or None,
                "generation_available": available,
                "retrieval_recall": float(detail["retrieval_recall"]) if available and detail.get("retrieval_recall") != "" else None,
                "context_precision": float(detail["context_precision"]) if available and detail.get("context_precision") != "" else None,
                "context_hash": sha256_bytes(canon(context_record["actual_evaluator_context"]).encode("utf-8")) if context_record and context_record.get("actual_evaluator_context") is not None else context_record.get("context_payload_sha256") if context_record else None,
                "selected_context_payload_hash": context_record.get("context_payload_sha256") if context_record else None,
                "evaluator_context_hash": sha256_bytes(canon(inp["actual_evaluator_context"]).encode("utf-8")) if inp and inp.get("scoring_eligible") else None,
                "latency": detail.get("elapsed_seconds") or None,
                "h20": token_row.get("entropy_k20_bits"),
                "h50": token_row.get("entropy_k50_bits"),
                "h100": token_row.get("entropy_k100_bits"),
                "retained_mass_k20": token_row.get("retained_mass_k20"),
                "retained_mass_k50": token_row.get("retained_mass_k50"),
                "retained_mass_k100": token_row.get("retained_mass_k100"),
                "entropy_token_count": token_row.get("telemetry_token_count"),
                "parser_status": detail.get("parser_status") or None,
                "raw_response_hash": raw_hash,
                "effective_config_hash": result.get("effective_configuration_hash") if result else arm_manifest[arm].get("effective_configuration_hash"),
                "judge_blinded_response_id": maprow["blinded_response_id"],
                "judge_case_id": score.get("case_id") if score else None,
                "judge_request_id": score.get("request_id") if score else None,
                "judge_scoring_status": score.get("status") if score else None,
                "judge_payload_available": bool(inp and inp.get("scoring_eligible")),
                "semantic_accuracy": score_metric(score, "semantic_accuracy"),
                "answer_relevance": score_metric(score, "answer_relevance"),
                "faithfulness": score_metric(score, "faithfulness"),
                "citation_fidelity": score_metric(score, "citation_fidelity"),
                "retrieval_recall_available": available and detail.get("retrieval_recall") != "",
                "context_precision_available": available and detail.get("context_precision") != "",
                "entropy_available": available and bool(detail.get("h20_bits")) and bool(detail.get("h50_bits")) and bool(detail.get("h100_bits")),
                "parser_available": bool(detail.get("parser_status")),
                "judge_available": bool(score and score.get("status") == "scored"),
            }
            rows.append(row)
    unique_keys = {(r["arm_id"], r["query_id"]) for r in rows}
    if len(rows) != 5250 or len(unique_keys) != 5250:
        raise AssertionError(f"Expected 5250 unique operations; got rows={len(rows)}, keys={len(unique_keys)}")
    if len(questions) != 375 or len(set(questions)) != 375:
        raise AssertionError("Frozen question ID count is not 375 unique")
    if set(questions) != {q for _, q in details}:
        raise AssertionError("Question set differs from metrics detail planned population")
    if len(details) != 5250:
        raise AssertionError(f"Metrics details expected 5250 rows, found {len(details)}")
    if raw_hash_bad:
        raise AssertionError(f"Raw response hash mismatch: {raw_hash_bad[:3]}")
    return rows, {"audit_counts": dict(audit), "missing_result_records": missing_results,
                  "raw_hash_mismatches": raw_hash_bad,
                  "identity_values": {k: sorted(v) for k, v in identities.items()},
                  "arm_manifest": arm_manifest}, results


def judge_system_sha256() -> str:
    import ast
    source = (JUDGE / "run_luna_judge.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SYSTEM" for target in node.targets):
            value = ast.literal_eval(node.value)
            return sha256_bytes(value.encode("utf-8"))
    raise AssertionError("Judge SYSTEM prompt constant not found")


def judge_cache_audit(judge: dict[str, Any], canonical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bad_case_ids = []
    score_conflicts = []
    visible_hashes: dict[str, set[str]] = defaultdict(set)
    for row in judge["inputs"].values():
        if not row.get("scoring_eligible"):
            continue
        payload = {"question": row["question"],
                   "answerability_label": row["ground_truth_answerability_label"],
                   "reference_answer": row["gold_reference_answer"],
                   "context": row["actual_evaluator_context"],
                   "answer": row["cleaned_output"]}
        digest = sha256_bytes(canon(payload).encode("utf-8"))
        score = judge["scores"].get(row["blinded_response_id"])
        grouped[digest].append({"input": row, "score": score, "digest": digest})
        if score and score.get("case_id") != "case_" + digest[:24]:
            bad_case_ids.append(row["blinded_response_id"])
        if score:
            visible_hashes[digest].add(canon(score.get("metrics")))
    for digest in grouped:
        if len(visible_hashes[digest]) > 1:
            score_conflicts.append(digest)
    eligible = sum(len(vals) for vals in grouped.values())
    return {"eligible_slots": eligible, "unique_exact_payloads": len(grouped),
            "duplicate_slots_reused": eligible - len(grouped),
            "max_payload_multiplicity": max(map(len, grouped.values())),
            "payload_identity_fields": ["question", "answerability_label", "reference_answer", "evaluator-visible context", "cleaned answer"],
            "fixed_global_judge_identity": {"model": "gpt-6-luna-low", "omniroute_model_id": "codex/gpt-6-luna-low", "temperature": 0, "reasoning_effort": "low", "system_rubric_sha256": judge_system_sha256()},
            "bad_case_id_count": len(bad_case_ids), "duplicate_payload_score_conflict_count": len(score_conflicts),
            "exact_payload_key_matches_score_case_id": len(bad_case_ids) == 0,
            "note": "Exact per-case reuse is validated for question, designated-answerability label, reference, evaluator-visible context, and cleaned answer. System rubric SHA-256 and model/decode settings (codex/gpt-6-luna-low, temperature=0, reasoning_effort=low) were fixed globally; rubric/script hash and manifest settings are preserved separately. No cross-run cache. No per-batch request/response replay was performed."}


def arm_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {arm: {row["query_id"]: row for row in rows if row["arm_id"] == arm} for arm in ARM_ORDER}


def create_coverage(data: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    base = data["B6_full"]
    output = []
    for arm in ARM_ORDER:
        current = data[arm]
        succ = {q for q, row in current.items() if row["generation_available"]}
        base_succ = {q for q, row in base.items() if row["generation_available"]}
        ans = {q for q, row in current.items() if row["answerability_label"] == "answerable"}
        neg = set(current) - ans
        types = Counter(row["failure_reason"].split(":", 1)[0] if row["failure_reason"] else "none" for row in current.values() if not row["generation_available"])
        output.append({"arm_id": arm, "planned": len(current), "successful": len(succ), "failed": len(current)-len(succ),
                       "coverage_pct": 100 * len(succ) / len(current),
                       "answerable_planned": len(ans), "answerable_success": len(succ & ans),
                       "answerable_coverage_pct": 100 * len(succ & ans) / len(ans),
                       "negative_planned": len(neg), "negative_success": len(succ & neg),
                       "negative_coverage_pct": 100 * len(succ & neg) / len(neg),
                       "shared_success_with_B6": len(succ & base_succ), "success_only_arm": len(succ - base_succ),
                       "success_only_B6": len(base_succ - succ), "failed_both": len(set(current) - (succ | base_succ)),
                       "failure_reason_counts": json.dumps(dict(types), sort_keys=True)})
    return output


def paired_effects(data: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        if arm == "B6_full":
            continue
        common = [q for q in data["B6_full"] if data["B6_full"][q]["generation_available"] and data[arm][q]["generation_available"]]
        for metric in METRICS:
            deltas = [data[arm][q][metric] - data["B6_full"][q][metric] for q in common]
            if any(value is None for value in deltas):
                raise AssertionError(f"missing paired metric for {arm}/{metric}")
            signs = Counter("positive" if x > 1e-12 else "negative" if x < -1e-12 else "tie" for x in deltas)
            seed = 20261020 + sum(map(ord, arm + metric))
            lo, hi = paired_bootstrap(deltas, seed=seed)
            p = sign_test_p(deltas)
            row = {"arm_id": arm, "metric": metric, "n_shared_success": len(common),
                   "mean_delta_vs_B6": statistics.fmean(deltas), "ci95_low": lo, "ci95_high": hi,
                   "positive": signs["positive"], "negative": signs["negative"], "ties": signs["tie"],
                   "paired_rank_biserial": rank_biserial(deltas), "sign_test_p_unadjusted": p,
                   "bootstrap_resamples": 10000, "bootstrap_seed": seed,
                   "ci_method": "query-level paired percentile bootstrap; unadjusted exploratory",
                   "test": "exact two-sided paired sign test; ties omitted"}
            rows.append(row)
    for metric in METRICS:
        fam = [row for row in rows if row["metric"] == metric]
        holm_adjust(fam, "sign_test_p_unadjusted", "holm_p_within_metric", f"13 arm contrasts: {metric}")
    all_rows = [dict(row) for row in rows]
    holm_adjust(all_rows, "sign_test_p_unadjusted", "holm_p_all_52", "52 arm-by-metric contrasts")
    for row, broad in zip(rows, all_rows, strict=True):
        row["holm_p_all_52"] = broad["holm_p_all_52"]
    return rows, all_rows


def create_multiplicity(effects: list[dict[str, Any]], all_effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in effects:
        output.append({"arm_id": row["arm_id"], "metric": row["metric"], "n": row["n_shared_success"],
                       "p_unadjusted_sign_test": row["sign_test_p_unadjusted"],
                       "holm_p_within_metric_family": row["holm_p_within_metric"],
                       "holm_p_all_52_contrasts": row["holm_p_all_52"],
                       "within_metric_family": f"13 component/interaction arms for {row['metric']}",
                       "broad_family": "all 13 non-reference arms x 4 judge outcomes = 52 contrasts",
                       "posthoc_exploratory": True})
    return output


def create_subgroups(data: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for arm in ("B6_minus_parent_deduplication", "B6_minus_dynamic_graph_expander"):
        common = [q for q in data["B6_full"] if data["B6_full"][q]["generation_available"] and data[arm][q]["generation_available"]]
        for subgroup in ("answerable", "designated_negative"):
            qs = sorted(q for q in common if data[arm][q]["answerability_label"] == subgroup)
            for metric in ("semantic_accuracy", "faithfulness", "citation_fidelity", "answer_relevance"):
                deltas = [data[arm][q][metric] - data["B6_full"][q][metric] for q in qs]
                seed = 20261020 + sum(map(ord, arm + subgroup + metric))
                lo, hi = paired_bootstrap(deltas, seed=seed)
                signs = Counter("positive" if x > 1e-12 else "negative" if x < -1e-12 else "tie" for x in deltas)
                output.append({"arm_id": arm, "subgroup": subgroup, "metric": metric, "n_shared_success": len(qs),
                               "mean_delta_vs_B6": statistics.fmean(deltas), "ci95_low": lo, "ci95_high": hi,
                               "positive": signs["positive"], "negative": signs["negative"], "ties": signs["tie"],
                               "bootstrap_resamples": 10000, "bootstrap_seed": seed,
                               "interpretation": "conditional on both arms generating; subgroup analysis exploratory"})
    return output


def operational_utility(data: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for arm in ARM_ORDER:
        deltas = []
        a_values = []
        paired_quality_deltas: list[float] = []
        for q in data["B6_full"]:
            a = data[arm][q]
            b = data["B6_full"][q]
            av = float(a["semantic_accuracy"]) if a["generation_available"] else 0.0
            bv = float(b["semantic_accuracy"]) if b["generation_available"] else 0.0
            a_values.append(av)
            deltas.append(av - bv)
            if a["generation_available"] and b["generation_available"]:
                paired_quality_deltas.append(av - bv)
        seed = 20261020 + sum(map(ord, arm + "operational_semantic_utility"))
        lo, hi = paired_bootstrap(deltas, seed=seed)
        out.append({"arm_id": arm, "n_frozen_queries": len(deltas),
                    "generation_availability": sum(data[arm][q]["generation_available"] for q in data[arm]) / len(deltas),
                    "mean_conditional_semantic_accuracy": mean_or_none([float(data[arm][q]["semantic_accuracy"]) for q in data[arm] if data[arm][q]["generation_available"]]),
                    "mean_policy_operational_utility": statistics.fmean(a_values),
                    "mean_policy_operational_utility_delta_vs_B6": statistics.fmean(deltas),
                    "paired_bootstrap_ci95_low": lo, "paired_bootstrap_ci95_high": hi,
                    "paired_success_quality_delta": statistics.fmean(paired_quality_deltas) if paired_quality_deltas else None,
                    "formula": "U_q = I(generation available) * judge Semantic Accuracy; unavailable=0 by service-utility policy only",
                    "not_a_missing_answer_judge_score": True, "bootstrap_resamples": 10000, "bootstrap_seed": seed})
    return out


def missingness_bounds(data: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for arm in ARM_ORDER:
        if arm == "B6_full":
            continue
        lo_sum = hi_sum = 0.0
        for q in data["B6_full"]:
            a, b = data[arm][q], data["B6_full"][q]
            if a["generation_available"] and b["generation_available"]:
                d = float(a["semantic_accuracy"]) - float(b["semantic_accuracy"])
                lo_sum += d
                hi_sum += d
            elif a["generation_available"]:
                value = float(a["semantic_accuracy"])
                lo_sum += value - 1.0
                hi_sum += value
            elif b["generation_available"]:
                value = float(b["semantic_accuracy"])
                lo_sum += -value
                hi_sum += 1.0 - value
            else:
                lo_sum -= 1.0
                hi_sum += 1.0

        out.append({"arm_id": arm, "n_all_queries": len(data[arm]), "paired_success_n": sum(data[arm][q]["generation_available"] and data['B6_full'][q]["generation_available"] for q in data[arm]),
                    "all_query_semantic_difference_lower_bound": lo_sum / len(data[arm]),
                    "all_query_semantic_difference_upper_bound": hi_sum / len(data[arm]),
                    "assumption": "each unavailable potential judge score lies in [0,1]; no imputation; bounds are sharp score-scale worst/best case"})
    return out


def document_ap(result: dict[str, Any], gold_papers: set[str]) -> float | None:
    evidence = result.get("selected_evidence") or []
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        paper = item.get("paper_id")
        if paper and paper not in seen:
            seen.add(paper)
            ordered_unique.append(paper)
    if not gold_papers or not ordered_unique:
        return None
    hits = 0
    precision_sum = 0.0
    for rank, paper in enumerate(ordered_unique, 1):
        if paper in gold_papers:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(gold_papers)


def create_context_provenance(data: dict[str, dict[str, dict[str, Any]]], results: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        query_ids = [query_id for query_id in data[arm] if data[arm][query_id]["generation_available"]]
        unique_pmids: list[float] = []
        gold_chunk_counts: list[float] = []
        gold_doc_hits: list[float] = []
        text_chars: list[float] = []
        dynamic_chars: list[float] = []
        chunk_precision: list[float] = []
        document_ap_values: list[float] = []
        selected_chunk_counts: list[float] = []
        prompt_tokens: list[float] = []
        prompt_hashes: set[str] = set()
        response_hashes: set[str] = set()
        expansion_executed = 0
        expansion_inserted = 0
        for query_id in query_ids:
            result = results[(arm, query_id)]
            evidence = result.get("selected_evidence") or []
            gold = set(result.get("gold_paper_ids") or [])
            paper_ids = [item.get("paper_id") for item in evidence if item.get("paper_id")]
            unique_pmids.append(float(len(set(paper_ids))))
            gold_chunk_counts.append(float(sum(paper_id in gold for paper_id in paper_ids)))
            gold_doc_hits.append(float(len(set(paper_ids) & gold)))
            selected_chunk_counts.append(float(len(evidence)))
            metric = data[arm][query_id]["context_precision"]
            if metric is not None:
                chunk_precision.append(float(metric))
            ap = document_ap(result, gold)
            if ap is not None:
                document_ap_values.append(ap)
            trace = result.get("retrieval_candidates_and_ranking_trace", {})
            trim = trace.get("context_capture", {}).get("context_trim", {})
            if trim.get("input_text_chars") is not None:
                text_chars.append(float(trim["input_text_chars"]))
            assembly = trace.get("context_and_prompt_assembly", {})
            dynamic = trace.get("dynamic_graph_output") or ""
            dynamic_chars.append(float(len(dynamic)))
            expansion_executed += int(bool(assembly.get("dynamic_graph_expansion_executed")))
            expansion_inserted += int(bool(assembly.get("dynamic_graph_output_inserted")))
            prompt_hash = assembly.get("serialized_prompt_sha256")
            if prompt_hash:
                prompt_hashes.add(prompt_hash)
            raw_response = result.get("raw_generated_response")
            if raw_response:
                response_hashes.add(sha256_bytes(raw_response.encode()))
            event = next((entry.get("event", {}) for entry in result.get("generation_events", [])
                          if entry.get("purpose") == "final_answer_generation"), None)
            if event and event.get("prompt_token_count") is not None:
                prompt_tokens.append(float(event["prompt_token_count"]))
        output.append({
            "arm_id": arm,
            "n_success": len(query_ids),
            "mean_selected_chunks": mean_or_none(selected_chunk_counts),
            "mean_unique_selected_pmids": mean_or_none(unique_pmids),
            "mean_gold_paper_chunks": mean_or_none(gold_chunk_counts),
            "mean_unique_gold_pmids_selected": mean_or_none(gold_doc_hits),
            "mean_chunk_context_precision": mean_or_none(chunk_precision),
            "mean_document_level_AP": mean_or_none(document_ap_values),
            "mean_context_text_chars": mean_or_none(text_chars),
            "mean_serialized_prompt_tokens": mean_or_none(prompt_tokens),
            "mean_dynamic_graph_output_chars": mean_or_none(dynamic_chars),
            "dynamic_expansion_executed_n": expansion_executed,
            "dynamic_output_inserted_n": expansion_inserted,
            "unique_serialized_prompt_hashes": len(prompt_hashes),
            "unique_generation_response_hashes": len(response_hashes),
            "context_precision_note": "Chunk-level MAP may count repeated same-PMID chunks; document-level AP deduplicates selected evidence by paper_id, preserving first rank.",
        })
    base_success = {query_id for query_id, row in data["B6_full"].items() if row["generation_available"]}
    base_prompt = {
        query_id: results[("B6_full", query_id)].get("retrieval_candidates_and_ranking_trace", {}).get("context_and_prompt_assembly", {}).get("serialized_prompt_sha256")
        for query_id in base_success
    }
    base_response = {
        query_id: sha256_bytes(results[("B6_full", query_id)]["raw_generated_response"].encode())
        for query_id in base_success if results[("B6_full", query_id)].get("raw_generated_response")
    }
    for row in output:
        arm = row["arm_id"]
        common = sorted(query_id for query_id in base_success if data[arm][query_id]["generation_available"])
        row["shared_success_n"] = len(common)
        row["paired_prompt_hash_identical_to_B6_n"] = sum(
            results[(arm, query_id)].get("retrieval_candidates_and_ranking_trace", {}).get("context_and_prompt_assembly", {}).get("serialized_prompt_sha256") == base_prompt[query_id]
            for query_id in common
        )
        row["paired_generation_hash_identical_to_B6_n"] = sum(
            sha256_bytes(results[(arm, query_id)]["raw_generated_response"].encode()) == base_response.get(query_id)
            for query_id in common if results[(arm, query_id)].get("raw_generated_response")
        )
        if arm == "B6_full":
            continue
        precision_delta: list[float] = []
        document_ap_delta: list[float] = []
        semantic_delta: list[float] = []
        gold_chunk_delta: list[float] = []
        for query_id in common:
            arm_row = data[arm][query_id]
            base_row = data["B6_full"][query_id]
            if arm_row["context_precision"] is not None and base_row["context_precision"] is not None:
                precision_delta.append(float(arm_row["context_precision"]) - float(base_row["context_precision"]))
            if arm_row["semantic_accuracy"] is not None and base_row["semantic_accuracy"] is not None:
                semantic_delta.append(float(arm_row["semantic_accuracy"]) - float(base_row["semantic_accuracy"]))
            arm_result = results[(arm, query_id)]
            base_result = results[("B6_full", query_id)]
            arm_ap = document_ap(arm_result, set(arm_result.get("gold_paper_ids") or []))
            base_ap = document_ap(base_result, set(base_result.get("gold_paper_ids") or []))
            if arm_ap is not None and base_ap is not None:
                document_ap_delta.append(arm_ap - base_ap)
            arm_gold = set(arm_result.get("gold_paper_ids") or [])
            base_gold = set(base_result.get("gold_paper_ids") or [])
            gold_chunk_delta.append(
                sum(item.get("paper_id") in arm_gold for item in arm_result.get("selected_evidence", []))
                - sum(item.get("paper_id") in base_gold for item in base_result.get("selected_evidence", []))
            )
        row["paired_chunk_precision_delta_vs_B6"] = mean_or_none(precision_delta)
        row["paired_document_AP_delta_vs_B6"] = mean_or_none(document_ap_delta)
        row["paired_document_AP_delta_note"] = "Document AP deduplicates PMID; repeated chunks cannot add unique source coverage. When only one known gold PMID exists and is retrieved, AP may tie at 1.0 despite different repeated-chunk counts."
        row["paired_document_precision_delta_vs_B6"] = mean_or_none(document_ap_delta)
        row["paired_semantic_accuracy_delta_vs_B6"] = mean_or_none(semantic_delta)
        row["corr_semantic_delta_gold_chunk_count_delta"] = correlation(semantic_delta, gold_chunk_delta)
        row["paired_mean_gold_chunk_count_delta"] = mean_or_none([float(value) for value in gold_chunk_delta])
    return output

def create_entropy(data: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        if arm == "B6_full":
            continue
        common = sorted(q for q in data["B6_full"]
                        if data["B6_full"][q]["generation_available"] and data[arm][q]["generation_available"])
        for measure in ("h20", "h50", "h100", "generated_tokens"):
            values = [float(data[arm][q][measure]) - float(data["B6_full"][q][measure])
                      for q in common if data[arm][q][measure] is not None and data["B6_full"][q][measure] is not None]
            seed = 20261020 + sum(map(ord, arm + measure + "entropy"))
            low, high = paired_bootstrap(values, seed=seed) if values else (None, None)
            output.append({"arm_id": arm, "measure": measure, "n_paired": len(values),
                           "mean_delta_vs_B6": mean_or_none(values), "ci95_low": low, "ci95_high": high,
                           "bootstrap_resamples": 10000, "bootstrap_seed": seed,
                           "metric_definition": "mean per-generated-token top-K entropy (bits), locally renormalized over top K logits; not full-vocabulary Shannon entropy" if measure.startswith("h") else "final generation token count",
                           "paired_query_ids_only": True})
        for measure in ("retained_mass_k20", "retained_mass_k50", "retained_mass_k100"):
            values = [float(data[arm][q][measure]) - float(data["B6_full"][q][measure])
                      for q in common if data[arm][q][measure] is not None and data["B6_full"][q][measure] is not None]
            seed = 20261020 + sum(map(ord, arm + measure + "entropy"))
            low, high = paired_bootstrap(values, seed=seed) if values else (None, None)
            output.append({"arm_id": arm, "measure": measure, "n_paired": len(values),
                           "mean_delta_vs_B6": mean_or_none(values), "ci95_low": low, "ci95_high": high,
                           "bootstrap_resamples": 10000, "bootstrap_seed": seed,
                           "metric_definition": "full-softmax probability mass within top K logits; fraction, not entropy",
                           "paired_query_ids_only": True})
        for measure in ("h20", "h50", "h100"):
            pairs = [(float(data[arm][q][measure]) - float(data["B6_full"][q][measure]),
                      float(data[arm][q]["semantic_accuracy"]) - float(data["B6_full"][q]["semantic_accuracy"]))
                     for q in common if data[arm][q][measure] is not None and data["B6_full"][q][measure] is not None]
            output.append({"arm_id": arm, "measure": f"corr_{measure}_delta_semantic_accuracy_delta",
                           "n_paired": len(pairs),
                           "mean_delta_vs_B6": correlation([pair[0] for pair in pairs], [pair[1] for pair in pairs]),
                           "ci95_low": None, "ci95_high": None, "bootstrap_resamples": 0, "bootstrap_seed": None,
                           "metric_definition": "descriptive Pearson correlation of within-query entropy change and Semantic Accuracy change; no causal interpretation",
                           "paired_query_ids_only": True})
    return output

def create_overflow_analysis(data: dict[str, dict[str, dict[str, Any]]], questions: dict[str,Any]) -> list[dict[str,Any]]:
    output=[]
    for arm in ARM_ORDER:
        current=data[arm]
        for group in ("all","answerable","designated_negative"):
            qs=[q for q,row in current.items() if group=="all" or row["answerability_label"]==group]
            failures=[current[q] for q in qs if not current[q]["generation_available"]]
            prompts=[float(current[q]["prompt_tokens"]) for q in qs if current[q]["prompt_tokens"] is not None]
            fail_lens=[len(questions[q]["question"]) for q in qs if not current[q]["generation_available"]]
            success_lens=[len(questions[q]["question"]) for q in qs if current[q]["generation_available"]]
            fail_prompts=[float(current[q]["prompt_tokens"]) for q in qs if not current[q]["generation_available"] and current[q]["prompt_tokens"] is not None]
            success_prompts=[float(current[q]["prompt_tokens"]) for q in qs if current[q]["generation_available"] and current[q]["prompt_tokens"] is not None]
            output.append({"arm_id":arm,"population":group,"planned":len(qs),"success":len(qs)-len(failures),"failures":len(failures),
                           "overflow_failures":sum(bool(current[q]["failure_reason"] and "Serialized prompt has" in current[q]["failure_reason"]) for q in qs),
                           "mean_recorded_serialized_prompt_tokens":mean_or_none(prompts),
                           "mean_failure_prompt_tokens":mean_or_none(fail_prompts),"mean_success_prompt_tokens":mean_or_none(success_prompts),
                           "mean_question_chars_failed":mean_or_none([float(x) for x in fail_lens]),
                           "mean_question_chars_success":mean_or_none([float(x) for x in success_lens]),
                           "prompt_length_note":"token count from tokenizer-recorded generation event for successful outputs and exact overflow exception for failures; question chars descriptive only"})
    return output


def integrity_outputs(rows: list[dict[str,Any]], audit: dict[str,Any], judge: dict[str,Any]) -> dict[str,Any]:
    # Exact planned/query and judge checks independent of summary artifacts.
    counts=Counter(r["operation_status"] for r in rows)
    score_counts=Counter(r["judge_scoring_status"] for r in rows)
    answerability=Counter(r["answerability_label"] for r in rows if r["arm_id"]=="B6_full")
    checks={
        "canonical_rows":len(rows), "unique_arm_query_pairs":len({(r['arm_id'],r['query_id']) for r in rows}),
        "planned_queries_per_arm":{a:len({r['query_id'] for r in rows if r['arm_id']==a}) for a in ARM_ORDER},
        "answerability_population":dict(answerability), "operation_status_counts":dict(counts),
        "judge_status_counts":dict(score_counts), "run_manifest_status":read_json(RUN/'RESULTS_MANIFEST.json')['result_status'],
        "run_package_expected_files":len(read_json(RUN/'RESULTS_MANIFEST.json')['expected_files']),
        "dataset_sha256":read_json(RUN/'questions_manifest.json')['dataset_sha256'],
        "arms_manifest_dataset_sha256":read_json(RUN/'arms_manifest.json')['dataset_sha256'],
        "runtime_dataset_sha256":read_json(RUN/'run_manifest.json')['dataset']['sha256'],
        "external_db_snapshot_sha256":read_json(RUN/'metadata/database_identity.json')['snapshot_sha256'],
        "july_database_identity_proven":read_json(RUN/'metadata/database_identity.json')['july_identity_proven'],
        "model_revisions":read_json(RUN/'run_manifest.json')['model_revisions'],
        "identity_values":audit['identity_values'], "canonical_audit_counts":audit['audit_counts'],
        "missing_result_records":audit['missing_result_records'], "raw_response_hash_mismatches":audit['raw_hash_mismatches'],
        "judge_cache_audit":judge,
    }
    checks["assertions"]={
        "5,250_planned_unique":checks['canonical_rows']==5250 and checks['unique_arm_query_pairs']==5250,
        "375_ids_each_arm":all(v==375 for v in checks['planned_queries_per_arm'].values()),
        "300_answerable_75_negative":answerability==Counter({'answerable':300,'designated_negative':75}),
        "4,265_success_985_failed":counts==Counter({'SUCCESS':4265,'PARTIAL':985}),
        "all_failures_overflow":all('Serialized prompt has' in r['failure_reason'] for r in rows if not r['generation_available']),
        "all_eligible_scored":audit['audit_counts'].get('eligible_missing_score',0)==0,
        "no_failed_scores":audit['audit_counts'].get('failed_scored',0)==0,
        "response_hashes_match":not audit['raw_hash_mismatches'],
        "model_index_dataset_consistent_within_arm_matrix":len(audit['identity_values']['snapshot'])==1 and len(audit['identity_values']['dataset'])==1 and len(audit['identity_values']['model_manifest'])==1,
        "judge_reuse_payload_complete":judge['bad_case_id_count']==0 and judge['duplicate_payload_score_conflict_count']==0,
    }
    return checks


def main() -> None:
    TABLES.mkdir(parents=True,exist_ok=True)
    questions,details,errors,judge=load_sources()
    rows,audit,results=build_canonical(questions,details,errors,judge)
    cache_audit=judge_cache_audit(judge,rows)
    checks=integrity_outputs(rows,audit,cache_audit)
    checks["reported_judge_summary_crosscheck"] = {
        "judge_summary_path": str(JUDGE / "judge_summary.json"),
        "arm_summary_deltas_are_reference_only": True,
        "primary_recomputed_bootstrap": "query-paired, fixed-seed 10,000 resamples; analysis uses regenerated arms/IDs and recomputed source pair table",
        "note": "Saved judge_summary uses its own legacy percentile indexing and arm iterates a Python set; primary table is rederived deterministically and does not adopt order-dependent seed assignment."
    }
    data=arm_rows(rows)
    columns=list(rows[0])
    write_csv(TABLES/"canonical_5250.csv",rows,columns)
    coverage=create_coverage(data)
    write_csv(TABLES/"coverage_by_arm.csv",coverage)
    effects,broad=paired_effects(data)
    write_csv(TABLES/"paired_judge_effects.csv",effects)
    write_csv(TABLES/"multiplicity_adjustments.csv",create_multiplicity(effects,broad))
    write_csv(TABLES/"subgroup_effects.csv",create_subgroups(data))
    write_csv(TABLES/"operational_utility.csv",operational_utility(data))
    write_csv(TABLES/"missingness_bounds.csv",missingness_bounds(data))
    write_csv(TABLES/"overflow_by_arm_subgroup.csv",create_overflow_analysis(data,questions))
    contexts=create_context_provenance(data,results)
    write_csv(TABLES/"context_provenance.csv",contexts)
    write_csv(TABLES/"entropy_paired.csv",create_entropy(data))
    atomic_text_write(OUT/"integrity_audit.json", json.dumps(checks, indent=2, sort_keys=True) + "\n")
    if not all(checks['assertions'].values()):
        raise SystemExit("Integrity assertions failed; inspect integrity_audit.json")
    print(json.dumps({"rows":len(rows),"integrity_assertions":checks['assertions'],"outputs":sorted(p.name for p in TABLES.glob('*.csv'))},indent=2))


if __name__ == "__main__":
    main()
