"""Local, judge-free analysis and blinded judge export generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import yaml

from ablation_cuda.runtime.checkpoints import atomic_write_json, atomic_write_bytes


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mean_available(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows
              if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
    return statistics.fmean(values) if values else None


def _pmid_set(values: list[Any]) -> set[str]:
    result = set()
    for value in values:
        text = str(value).strip()
        if text:
            result.add(text)
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                result.add(digits)
    return result


def _evidence(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result:
        return []
    rows = result.get("selected_evidence", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _final_telemetry(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result:
        return []
    final_events = [item for item in result.get("generation_events", [])
                    if item.get("purpose") == "final_answer_generation"]
    if final_events:
        return final_events[-1].get("token_telemetry", []) or []
    return result.get("token_telemetry", []) or []


def _context_payload(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {"text": "", "graph": "", "dynamic_graph": "", "block_ids": []}
    trace = result.get("retrieval_candidates_and_ranking_trace", {})
    capture = trace.get("context_capture", {})
    trim = capture.get("context_trim", {})
    assembly = trace.get("context_and_prompt_assembly", {})
    dynamic = trace.get("dynamic_graph_output") or ""
    if not assembly.get("dynamic_graph_output_inserted"):
        dynamic = ""
    return {
        "text": trim.get("output_text", ""),
        "graph": trim.get("output_graph", ""),
        "dynamic_graph": dynamic,
        "block_ids": [str(item.get("id")) for item in _evidence(result) if item.get("id") is not None],
    }


def _eligible(result: dict[str, Any] | None) -> tuple[bool, str | None]:
    if not result:
        return False, "missing_operation"
    if result.get("status") != "SUCCESS":
        return False, "operation_not_successful"
    if result.get("result_errors"):
        return False, "operation_has_errors"
    if not isinstance(result.get("raw_generated_response"), str) or not result["raw_generated_response"].strip():
        return False, "empty_raw_response"
    return True, None


def generate_judge_exports(run_dir: Path, questions: list[dict[str, Any]],
                           arms: list[dict[str, Any]], results: dict[tuple[str, str], dict[str, Any] | None],
                           *, dataset_sha256: str, model_manifest_sha256: str) -> None:
    judge_dir = run_dir / "judge"
    mappings = []
    eligible_count = 0
    arm_id_by_name = {arm["arm_id"]: arm for arm in arms}
    inputs_path = judge_dir / "judge_inputs.jsonl"
    judge_dir.mkdir(parents=True, exist_ok=True)
    with inputs_path.open("w", encoding="utf-8") as input_handle:
      for question in questions:
        query_id = question["query_id"]
        for arm in arms:
            arm_id = arm["arm_id"]
            result = results.get((arm_id, query_id))
            raw = result.get("raw_generated_response", "") if result else ""
            cleaned = result.get("parsed_answer", "") if result else ""
            prompt = result.get("prompt", "") if result else ""
            response_id = "blind_" + _sha(f"{dataset_sha256}:{query_id}:{arm_id}")[:20]
            scoring_eligible, ineligible_reason = _eligible(result)
            eligible_count += int(scoring_eligible)
            context = _context_payload(result)
            status = result.get("parser_status") if result else None
            abstention = status == "UNANSWERABLE"
            if not abstention and isinstance(raw, str):
                abstention = any(marker in raw.lower() for marker in (
                    "insufficient information", "cannot be answered", "not enough information",
                    "no relevant evidence",
                ))
            input_record = {
                "blinded_response_id": response_id,
                "query_id": query_id,
                "question": question["question"],
                "gold_reference_answer": question.get("gold_reference_answer", ""),
                "ground_truth_answerability_label": question["answerability_label"],
                "source_pmid": (question.get("original_pmids") or [None])[0],
                "synthetic_negative_status": question["answerability_label"] == "designated_negative",
                "raw_output": raw,
                "cleaned_output": cleaned,
                "parsed_answer": cleaned,
                "parser_status": status,
                "abstention_decision": bool(abstention),
                "operational_answerability_policy": {
                    "prediction": status,
                    "interpretation": "parser status is operational telemetry; UNKNOWN remains explicit",
                },
                "actual_evaluator_context": context,
                "actual_generator_visible_context": context,
                "serialized_generation_prompt": prompt,
                "context_block_identifiers": context["block_ids"],
                "model_and_prompt_hashes": {
                    "model_manifest_sha256": model_manifest_sha256,
                    "effective_configuration_hash": result.get("effective_configuration_hash") if result else None,
                    "prompt_sha256": result.get("prompt_sha256") if result else None,
                    "response_sha256": _sha(raw),
                },
                "response_sha256": _sha(raw),
                "scoring_eligible": scoring_eligible,
                "scoring_ineligibility_reason": ineligible_reason,
                "missing_error_state": (result.get("result_errors", []) if result else ["missing query-arm operation"]),
                "operation_status": result.get("status") if result else "MISSING",
            }
            input_handle.write(json.dumps(input_record, ensure_ascii=False, sort_keys=True,
                                          allow_nan=False, separators=(",", ":")) + "\n")
            mappings.append({"blinded_response_id": response_id,
                             "query_id": query_id, "arm_id": arm_id,
                             "effective_configuration_hash": arm.get("effective_configuration_hash"),
                             "response_sha256": input_record["response_sha256"]})
    atomic_write_json(judge_dir / "arm_mapping.json", {
        "schema_version": "1.0", "blinding": "judge_inputs use deterministic opaque response IDs; mapping is retained separately",
        "records": mappings,
    })
    atomic_write_json(judge_dir / "judge_manifest.json", {
        "schema_version": "1.0", "judge_execution": "not_run",
        "remote_requests_made": False,
        "query_count": len(questions), "arm_count": len(arms),
        "expected_records": len(questions) * len(arms), "exported_records": len(questions) * len(arms),
        "eligible_records": eligible_count,
        "ineligible_records": len(questions) * len(arms) - eligible_count,
        "dataset_sha256": dataset_sha256, "arm_ids_sha256": _sha("\n".join(arm_id_by_name)),
        "response_hash_field": "response_sha256", "mapping_path": "arm_mapping.json",
    })
    atomic_write_json(judge_dir / "judge_schema.json", {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Biomedical RAG blinded judge input",
        "type": "object",
        "required": ["blinded_response_id", "query_id", "question", "gold_reference_answer",
                      "ground_truth_answerability_label", "raw_output", "parser_status",
                      "actual_evaluator_context", "response_sha256", "scoring_eligible", "operation_status"],
        "properties": {
            "blinded_response_id": {"type": "string"}, "query_id": {"type": "string"},
            "question": {"type": "string"}, "gold_reference_answer": {"type": "string"},
            "ground_truth_answerability_label": {"enum": ["answerable", "designated_negative"]},
            "parser_status": {"type": ["string", "null"]}, "scoring_eligible": {"type": "boolean"},
            "response_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    })


def _token_rows(result: dict[str, Any], arm_id: str, run_id: str):
    for event in result.get("generation_events", []):
        event_info = event.get("event", {})
        prompt_hash = _sha(str(event_info.get("prompt", "")))
        for index, token in enumerate(event.get("token_telemetry", [])):
            yield {**token, "run_id": run_id, "arm_id": arm_id,
                   "query_id": result.get("query_id"),
                   "generation_purpose": event.get("purpose"),
                   "generation_cache_hit": bool(event_info.get("cache_hit")),
                   "prompt_sha256": prompt_hash, "token_index": index,
                   "probability_unit": "fraction", "entropy_unit": "bits",
                   "log_probability_unit": "nats"}


def _write_parquet(path: Path, results: dict[tuple[str, str], dict[str, Any] | None],
                   arm_ids: list[str], run_id: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Locked pyarrow dependency is required to write token_telemetry.parquet") from exc
    schema = pa.schema([
        ("run_id", pa.string()), ("arm_id", pa.string()), ("query_id", pa.string()),
        ("generation_purpose", pa.string()), ("generation_cache_hit", pa.bool_()),
        ("prompt_sha256", pa.string()), ("token_index", pa.int32()), ("token_id", pa.int64()),
        ("token_text", pa.string()), ("token", pa.string()),
        ("char_start", pa.int32()), ("char_end", pa.int32()),
        ("logprob", pa.float64()), ("nll_nats", pa.float64()),
        ("nll_bits_per_token", pa.float64()), ("entropy_k20_bits", pa.float64()),
        ("entropy_k50_bits", pa.float64()), ("entropy_k100_bits", pa.float64()),
        ("retained_mass_k20", pa.float64()), ("retained_mass_k50", pa.float64()),
        ("retained_mass_k100", pa.float64()), ("top1_prob_full", pa.float64()),
        ("logit_margin", pa.float64()), ("top_logprobs_json", pa.string()),
        ("top_logit_margins_json", pa.string()), ("telemetry_device", pa.string()),
        ("stop_state", pa.string()), ("is_stop_token", pa.bool_()),
        ("probability_unit", pa.string()), ("entropy_unit", pa.string()),
        ("log_probability_unit", pa.string()),
    ])
    writer = None
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    batch: list[dict[str, Any]] = []
    try:
        writer = pq.ParquetWriter(temporary_path, schema, compression="zstd", version="2.6")
        for arm_id in arm_ids:
            if hasattr(results, "iter_arm"):
                arm_results = results.iter_arm(arm_id)
            else:
                arm_results = (result for (saved_arm_id, _query_id), result in results.items()
                               if saved_arm_id == arm_id)
            for result in arm_results:
                if result is None:
                    continue
                for row in _token_rows(result, arm_id, run_id):
                    normalized = {key: row.get(key) for key in schema.names}
                    normalized["top_logprobs_json"] = json.dumps(row.get("top_logprobs", {}),
                                                                  ensure_ascii=False, sort_keys=True)
                    normalized["top_logit_margins_json"] = json.dumps(row.get("top_logit_margins", {}),
                                                                       ensure_ascii=False, sort_keys=True)
                    batch.append(normalized)
                    if len(batch) >= 50_000:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema), row_group_size=50_000)
                        batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema), row_group_size=50_000)
    finally:
        if writer is not None:
            writer.close()
        if temporary_path.exists():
            temporary_path.replace(path)


def _svg_bar(path: Path, title: str, labels: list[str], values: list[float], *, y_label: str) -> None:
    width, height = 1100, 560
    left, right, top, bottom = 220, 35, 70, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    finite = [value for value in values if math.isfinite(value)]
    maximum = max(finite, default=1.0)
    maximum = maximum if maximum > 0 else 1.0
    gap = 7
    bar_h = max(8, (plot_h - gap * max(0, len(values) - 1)) / max(1, len(values)))
    nodes = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="{left}" y="36" font-family="sans-serif" font-size="22" font-weight="bold">{title}</text>',
             f'<text x="20" y="{top + plot_h/2}" transform="rotate(-90 20 {top + plot_h/2})" font-family="sans-serif" font-size="13">{y_label}</text>']
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * (bar_h + gap)
        amount = max(0.0, value) / maximum * plot_w if math.isfinite(value) else 0.0
        nodes.extend([
            f'<text x="{left-10}" y="{y+bar_h*0.72}" text-anchor="end" font-family="sans-serif" font-size="12">{label}</text>',
            f'<rect x="{left}" y="{y}" width="{amount:.2f}" height="{bar_h:.2f}" fill="#2d74a6"/>',
            f'<text x="{left+amount+5}" y="{y+bar_h*0.72}" font-family="sans-serif" font-size="11">{value:.4g}</text>',
        ])
    nodes.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(nodes), encoding="utf-8")


def finalize_outputs(run_dir: Path, *, run_id: str, status: str,
                     questions: list[dict[str, Any]], arms: list[dict[str, Any]],
                     results: dict[tuple[str, str], dict[str, Any] | None],
                     dataset_sha256: str, model_manifest_sha256: str,
                     planned_operations: int, run_errors: list[str]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    by_arm: dict[str, list[dict[str, Any]]] = {arm["arm_id"]: [] for arm in arms}
    for arm in arms:
        arm_id = arm["arm_id"]
        arm_dir = run_dir / "baselines" / arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        config = arm.get("effective_config", {})
        (arm_dir / "effective_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=True, allow_unicode=True), encoding="utf-8")
        retrieval_rows = []
        generation_rows = []
        parsed_rows = []
        for question in questions:
            query_id = question["query_id"]
            result = results.get((arm_id, query_id))
            relative_result = f"results/{query_id}.json"
            expected = _pmid_set(question.get("original_paper_ids", []) or question.get("original_pmids", []))
            evidence = _evidence(result)
            selected_ids = [str(item.get("id")) for item in evidence if item.get("id") is not None]
            found_values = set(str(item) for item in (result.get("retrieved_paper_ids", []) if result else []))
            found_values.update(str(item.get("paper_id")) for item in evidence if item.get("paper_id"))
            found = _pmid_set(list(found_values))
            relevant = len(found & expected)
            recall = relevant / len(expected) if expected else None
            chunk_papers = [_pmid_set([item.get("paper_id")]) for item in evidence]
            context_relevant = sum(bool(papers & expected) for papers in chunk_papers)
            precision = context_relevant / len(evidence) if evidence else None
            telemetry = _final_telemetry(result)
            parser_status = result.get("parser_status") if result else None
            truth = question.get("answerability_label") == "answerable"
            prediction = True if parser_status == "ANSWERABLE" else False if parser_status == "UNANSWERABLE" else None
            output_row = {
                "arm_id": arm_id, "query_id": query_id,
                "operation_status": result.get("status") if result else "MISSING",
                "retrieval_recall": recall, "context_precision": precision,
                "retrieved_paper_ids": sorted(found_values), "gold_paper_ids": question.get("original_paper_ids", []),
                "selected_chunk_ids": selected_ids,
                "stage_timings": result.get("timing", {}).get("pipeline_component_metrics", {}) if result else {},
                "elapsed_seconds": result.get("timing", {}).get("elapsed_seconds") if result else None,
                "generated_tokens": result.get("total_pipeline_generated_tokens", 0) if result else 0,
                "final_generation_tokens": len(telemetry),
                "parser_status": parser_status, "answerability_truth": truth,
                "answerability_prediction": prediction,
                "answerability_correct": prediction == truth if prediction is not None else None,
                "unknown_prediction": prediction is None,
                "h20_bits": _mean_available(telemetry, "entropy_k20_bits"),
                "h50_bits": _mean_available(telemetry, "entropy_k50_bits"),
                "h100_bits": _mean_available(telemetry, "entropy_k100_bits"),
                "top_k_retained_mass_k20": _mean_available(telemetry, "retained_mass_k20"),
                "top_k_retained_mass_k50": _mean_available(telemetry, "retained_mass_k50"),
                "top_k_retained_mass_k100": _mean_available(telemetry, "retained_mass_k100"),
            }
            details.append(output_row)
            by_arm[arm_id].append(output_row)
            retrieval_rows.append({"query_id": query_id, "canonical_result": relative_result,
                                   "prompt_sha256": result.get("prompt_sha256") if result else None,
                                   "selected_chunk_ids": selected_ids,
                                   "context_payload_sha256": _sha(json.dumps(_context_payload(result), sort_keys=True))})
            generation_rows.append({"query_id": query_id, "canonical_result": relative_result,
                                    "raw_response": result.get("raw_generated_response", "") if result else "",
                                    "response_sha256": _sha(result.get("raw_generated_response", "") if result else ""),
                                    "prompt_sha256": result.get("prompt_sha256") if result else None,
                                    "generation_events": [{"purpose": event.get("purpose"),
                                                           "prompt_sha256": _sha(str(event.get("event", {}).get("formatted_prompt", ""))),
                                                           "cache_hit": bool(event.get("event", {}).get("cache_hit"))}
                                                          for event in (result.get("generation_events", []) if result else [])]})
            parsed_rows.append({"query_id": query_id, "canonical_result": relative_result,
                                "raw_response": result.get("raw_generated_response", "") if result else "",
                                "parsed_answer": result.get("parsed_answer", "") if result else "",
                                "parser_status": parser_status,
                                "answerability_label": question.get("answerability_label"),
                                "answerability_prediction": parser_status,
                                "operation_status": result.get("status") if result else "MISSING",
                                "errors": result.get("result_errors", ["missing query-arm operation"]) if result else ["missing query-arm operation"]})
            if result:
                trace = result.get("retrieval_candidates_and_ranking_trace", {})
                trace_path = arm_dir / "traces" / f"{query_id}.json"
                atomic_write_json(trace_path, trace)
        _write_jsonl(arm_dir / "retrieved_contexts.jsonl", retrieval_rows)
        _write_jsonl(arm_dir / "generations.jsonl", generation_rows)
        _write_jsonl(arm_dir / "parsed.jsonl", parsed_rows)
        _write_parquet(arm_dir / "token_telemetry.parquet", results, [arm_id], run_id)

    _write_jsonl(run_dir / "generation_results.jsonl", (
        {"arm_id": arm["arm_id"], "query_id": question["query_id"],
         "canonical_result": f"baselines/{arm['arm_id']}/results/{question['query_id']}.json",
         "operation_status": (results.get((arm["arm_id"], question["query_id"])) or {}).get("status", "MISSING"),
         "response_sha256": _sha(((results.get((arm["arm_id"], question["query_id"])) or {}).get("raw_generated_response", "")))}
        for question in questions for arm in arms
    ))
    retrieval_compat = {
        "schema_version": "cuda_ablation_retrieved_contexts_v1",
        "compatibility_note": "Canonical July retrieved_contexts.yaml used baseline-keyed nested context blocks. This file is a deterministic index into per-arm canonical result JSON records; exact ordered evidence and generator prompt are in each result record.",
        "query_count": len(questions), "arm_count": len(arms),
        "baselines": {
            arm["arm_id"]: {question["query_id"]: {
                "result_file": f"baselines/{arm['arm_id']}/results/{question['query_id']}.json",
                "prompt_sha256": (results.get((arm["arm_id"], question["query_id"])) or {}).get("prompt_sha256"),
                "selected_chunk_ids": [str(item.get("id")) for item in _evidence(results.get((arm["arm_id"], question["query_id"])))],
            } for question in questions} for arm in arms
        },
    }
    (run_dir / "retrieved_contexts.yaml").write_text(
        yaml.safe_dump(retrieval_compat, sort_keys=False, allow_unicode=True), encoding="utf-8")

    (run_dir / "parsed").mkdir(exist_ok=True)
    for arm in arms:
        _write_jsonl(run_dir / "parsed" / f"{arm['arm_id']}.jsonl",
                     ({"arm_id": arm["arm_id"], "query_id": item["query_id"],
                       "canonical_result": f"baselines/{arm['arm_id']}/results/{item['query_id']}.json",
                       "raw_response": (results.get((arm["arm_id"], item["query_id"])) or {}).get("raw_generated_response", ""),
                       "parsed_answer": (results.get((arm["arm_id"], item["query_id"])) or {}).get("parsed_answer", ""),
                       "parser_status": (results.get((arm["arm_id"], item["query_id"])) or {}).get("parser_status"),
                       "operation_status": (results.get((arm["arm_id"], item["query_id"])) or {}).get("status", "MISSING")}
                      for item in questions))

    _write_parquet(run_dir / "token_telemetry.parquet", results, [arm["arm_id"] for arm in arms], run_id)
    atomic_write_json(run_dir / "metadata/token_telemetry_schema.json", {
        "schema_version": "1.0", "format": "Apache Parquet 2.6", "compression": "zstd",
        "source": "Per-query-arm compressed token sidecars hydrated from atomic checkpoints",
        "units": {"entropy": "bits", "log_probability": "nats", "nll": "nats",
                  "probability": "fraction", "logit_margin": "raw logit units"},
        "columns": ["run_id", "arm_id", "query_id", "generation_purpose",
                    "generation_cache_hit", "prompt_sha256", "token_index", "token_id",
                    "token_text", "logprob", "nll_nats", "nll_bits_per_token",
                    "entropy_k20_bits", "entropy_k50_bits", "entropy_k100_bits",
                    "retained_mass_k20", "retained_mass_k50", "retained_mass_k100",
                    "top1_prob_full", "logit_margin", "top_logprobs_json",
                    "top_logit_margins_json", "telemetry_device", "stop_state"],
    })
    details_path = run_dir / "metrics_details.csv"
    if details:
        with details_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(details[0]))
            writer.writeheader()
            for row in details:
                serialized = {key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                              if isinstance(value, (dict, list)) else value for key, value in row.items()}
                writer.writerow(serialized)
    summaries = []
    cache_by_arm = {}
    cache_rates_by_arm = {}
    for arm in arms:
        rows = by_arm[arm["arm_id"]]
        success_count = sum(row["operation_status"] == "SUCCESS" for row in rows)
        failed_count = sum(row["operation_status"] not in {"SUCCESS", "MISSING"} for row in rows)
        missing_count = sum(row["operation_status"] == "MISSING" for row in rows)
        true_known = [row for row in rows if row["answerability_prediction"] is not None]
        tp = sum(row["answerability_truth"] and row["answerability_prediction"] is True for row in true_known)
        tn = sum(not row["answerability_truth"] and row["answerability_prediction"] is False for row in true_known)
        fp = sum(not row["answerability_truth"] and row["answerability_prediction"] is True for row in true_known)
        fn = sum(row["answerability_truth"] and row["answerability_prediction"] is False for row in true_known)
        values = lambda key: [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
        latencies = values("elapsed_seconds")
        recalls = values("retrieval_recall")
        precisions = values("context_precision")
        entropy50 = values("h50_bits")
        tokens = values("generated_tokens")
        stage_keys = sorted({key for row in rows for key in (row.get("stage_timings") or {})})
        stage_timing_mean = {
            key: statistics.fmean(float((row.get("stage_timings") or {})[key])
                                  for row in rows
                                  if isinstance((row.get("stage_timings") or {}).get(key), (int, float)))
            for key in stage_keys
            if any(isinstance((row.get("stage_timings") or {}).get(key), (int, float)) for row in rows)
        }
        summaries.append({
            "arm_id": arm["arm_id"], "planned_queries": len(questions),
            "completed_queries": success_count, "failed_queries": failed_count,
            "missing_queries": missing_count,
            "coverage": success_count / len(questions) if questions else 0.0,
            "retrieval_recall_mean": statistics.fmean(recalls) if recalls else None,
            "context_precision_mean": statistics.fmean(precisions) if precisions else None,
            "answerability_true_positive": tp, "answerability_true_negative": tn,
            "answerability_false_positive": fp, "answerability_false_negative": fn,
            "answerability_unknown": sum(row["unknown_prediction"] for row in rows),
            "answerability_accuracy_known": (tp + tn) / len(true_known) if true_known else None,
            "latency_mean_seconds": statistics.fmean(latencies) if latencies else None,
            "latency_median_seconds": statistics.median(latencies) if latencies else None,
            "stage_timing_mean_seconds": stage_timing_mean,
            "generated_tokens_mean": statistics.fmean(tokens) if tokens else None,
            "h20_bits_mean": statistics.fmean(values("h20_bits")) if values("h20_bits") else None,
            "h50_bits_mean": statistics.fmean(entropy50) if entropy50 else None,
            "h100_bits_mean": statistics.fmean(values("h100_bits")) if values("h100_bits") else None,
            "retained_mass_k20_mean": statistics.fmean(values("top_k_retained_mass_k20")) if values("top_k_retained_mass_k20") else None,
            "retained_mass_k50_mean": statistics.fmean(values("top_k_retained_mass_k50")) if values("top_k_retained_mass_k50") else None,
            "retained_mass_k100_mean": statistics.fmean(values("top_k_retained_mass_k100")) if values("top_k_retained_mass_k100") else None,
            "judge_status": "PENDING_INDEPENDENT_AUTHOR_EVALUATION",
        })
        event_rows = []
        for question in questions:
            result = results.get((arm["arm_id"], question["query_id"]))
            if result:
                event_rows.extend(result.get("retrieval_candidates_and_ranking_trace", {}).get("stage_cache_events", []))
        hits = {}
        stage_totals: dict[str, dict[str, int]] = {}
        for event in event_rows:
            stage = event.get("stage", "unknown")
            items = max(1, int(event.get("items", 1)))
            stage_totals.setdefault(stage, {"hit_items": 0, "miss_items": 0})
            counter = "hit_items" if event.get("hit") else "miss_items"
            stage_totals[stage][counter] += items
            if event.get("hit"):
                hits[stage] = hits.get(stage, 0) + items
        cache_rates = {
            stage: {**counts,
                    "total_items": counts["hit_items"] + counts["miss_items"],
                    "hit_rate": counts["hit_items"] / (counts["hit_items"] + counts["miss_items"])}
            for stage, counts in sorted(stage_totals.items())
        }
        cache_by_arm[arm["arm_id"]] = hits
        cache_rates_by_arm[arm["arm_id"]] = cache_rates
        summaries[-1]["stage_cache_hit_rates"] = cache_rates
        arm["completion"] = {"planned": len(questions), "complete": success_count,
                              "failed": failed_count, "missing": missing_count,
                              "status": "COMPLETE" if success_count == len(questions) else "PARTIAL"}
        arm["intervention_integrity"] = _arm_integrity(arm, results, questions)

    with (run_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else [])
        if summaries:
            writer.writeheader()
            for summary in summaries:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                                 if isinstance(value, (dict, list)) else value
                                 for key, value in summary.items()})
    lines = ["# Local metrics summary", "", "These measures describe retrieval, context, answerability parsing, latency and generation telemetry. No answer-quality or citation-accuracy score was computed.", ""]
    arms_by_id = {arm["arm_id"]: arm for arm in arms}
    for summary in summaries:
        arm = arms_by_id[summary["arm_id"]]
        lines.extend([f"## {summary['arm_id']}", "",
                      f"- Coverage: {summary['completed_queries']}/{summary['planned_queries']} completed; {summary['failed_queries']} failed; {summary['missing_queries']} missing.",
                      f"- Retrieval recall: {summary['retrieval_recall_mean']}",
                      f"- Context precision: {summary['context_precision_mean']}",
                      f"- Answerability confusion matrix (known parser outcomes): TP {summary['answerability_true_positive']}, TN {summary['answerability_true_negative']}, FP {summary['answerability_false_positive']}, FN {summary['answerability_false_negative']}; UNKNOWN {summary['answerability_unknown']}.",
                      f"- Latency mean/median: {summary['latency_mean_seconds']} / {summary['latency_median_seconds']} seconds.",
                      f"- Mean stage timings (seconds): `{json.dumps(summary['stage_timing_mean_seconds'], sort_keys=True)}`.",
                      f"- Generated token mean: {summary['generated_tokens_mean']}.",
                      f"- Mean H20/H50/H100: {summary['h20_bits_mean']} / {summary['h50_bits_mean']} / {summary['h100_bits_mean']} bits.",
                      f"- Mean retained top-20/50/100 probability mass: {summary['retained_mass_k20_mean']} / {summary['retained_mass_k50_mean']} / {summary['retained_mass_k100_mean']}.",
                      f"- Stage cache hits: `{json.dumps(cache_by_arm[summary['arm_id']], sort_keys=True)}`.",
                      f"- Stage cache hit rates: `{json.dumps(cache_rates_by_arm[summary['arm_id']], sort_keys=True)}`.",
                      f"- Intervention integrity: `{arm['intervention_integrity']['status']}`.", ""])
    (run_dir / "metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")

    _svg_bar(run_dir / "figures/retrieval_recall.svg", "Retrieval recall by arm",
             [item["arm_id"] for item in summaries], [item["retrieval_recall_mean"] or 0.0 for item in summaries], y_label="Experimental arm")
    _svg_bar(run_dir / "figures/latency_mean.svg", "Mean query-arm latency by arm",
             [item["arm_id"] for item in summaries], [item["latency_mean_seconds"] or 0.0 for item in summaries], y_label="Experimental arm")
    _svg_bar(run_dir / "figures_entropy/h50_mean.svg", "Mean per-token H50 by arm",
             [item["arm_id"] for item in summaries], [item["h50_bits_mean"] or 0.0 for item in summaries], y_label="Experimental arm")

    baseline = {row["query_id"]: row["h50_bits"] for row in by_arm.get("B6_full", [])}
    paired = {}
    for arm in arms:
        values = []
        for row in by_arm[arm["arm_id"]]:
            left, right = row.get("h50_bits"), baseline.get(row["query_id"])
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                values.append(float(left) - float(right))
        paired[arm["arm_id"]] = {"paired_query_count": len(values),
                                  "mean_h50_difference_bits_vs_B6": statistics.fmean(values) if values else None}
    atomic_write_json(run_dir / "metadata/local_paired_entropy.json", paired)

    generate_judge_exports(run_dir, questions, arms, results,
                           dataset_sha256=dataset_sha256,
                           model_manifest_sha256=model_manifest_sha256)
    atomic_write_json(run_dir / "metadata/cache_hit_counts.json", cache_by_arm)
    atomic_write_json(run_dir / "metadata/cache_stage_hit_rates.json", cache_rates_by_arm)
    missing_or_failed = sum(summary["failed_queries"] + summary["missing_queries"] for summary in summaries)
    successful_total = sum(item["completed_queries"] for item in summaries)
    failed_total = sum(item["failed_queries"] for item in summaries)
    missing_total = sum(item["missing_queries"] for item in summaries)
    integrity_failed = any(
        arm.get("intervention_integrity", {}).get("status") == "SCIENTIFIC_VALIDATION_FAILED"
        for arm in arms
    )
    if status == "FAILED":
        report_status = "FAILED"
    elif successful_total == planned_operations and failed_total == 0 and missing_total == 0:
        report_status = "SCIENTIFIC_VALIDATION_FAILED" if integrity_failed else "COMPLETE"
    else:
        report_status = "PARTIAL"
    report_lines = ["# Completion report", "", f"- Status: **{report_status}**",
                    f"- Planned query-arm operations: {planned_operations}",
                    f"- Completed successful query-arm operations: {sum(item['completed_queries'] for item in summaries)}",
                    f"- Failed or missing operations: {missing_or_failed}",
                    f"- Query count: {len(questions)}", f"- Arm count: {len(arms)}",
                    f"- Local errors recorded: {len(run_errors)}", "- Independent LLM judge: **PENDING; no remote requests were made**", "",
                    "## Per-arm completion and intervention integrity", ""]
    for arm in arms:
        report_lines.append(f"- `{arm['arm_id']}`: {arm['completion']['status']} ({arm['completion']['complete']}/{arm['completion']['planned']}), intervention integrity `{arm['intervention_integrity']['status']}` — {arm['intervention_integrity'].get('reason', '')}")
    report_lines.extend(["", "The local metrics are descriptive retrieval, context, parser, runtime and token-telemetry statistics. They do not rank unjudged answer quality."])
    (run_dir / "completion_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    atomic_write_json(run_dir / "metadata/local_metrics_summary.json", {
        "schema_version": "1.0", "status": report_status,
        "planned_query_arm_operations": planned_operations,
        "completed": sum(item["completed_queries"] for item in summaries),
        "failed": sum(item["failed_queries"] for item in summaries),
        "missing": sum(item["missing_queries"] for item in summaries),
        "judge": "PENDING", "arms": summaries, "paired_entropy": paired,
        "cache_hit_counts": cache_by_arm, "cache_stage_hit_rates": cache_rates_by_arm,
    })
    return {"arm_summaries": summaries, "cache_hits": cache_by_arm,
            "cache_stage_hit_rates": cache_rates_by_arm,
            "paired_entropy": paired,
            "status": report_status,
            "completed": sum(item["completed_queries"] for item in summaries),
            "failed": sum(item["failed_queries"] for item in summaries),
            "missing": sum(item["missing_queries"] for item in summaries)}


def _arm_integrity(arm: dict[str, Any], results: dict[tuple[str, str], dict[str, Any] | None],
                   questions: list[dict[str, Any]]) -> dict[str, Any]:
    arm_id = arm["arm_id"]
    spec = arm.get("definition", {})
    if arm_id == "B6_full":
        return {"status": "REFERENCE", "reason": "B6 reference path is the comparison anchor."}
    completed = 0
    changed = 0
    active_baseline = 0
    for question in questions:
        baseline = results.get(("B6_full", question["query_id"]))
        candidate = results.get((arm_id, question["query_id"]))
        if not baseline or not candidate or baseline.get("status") != "SUCCESS" or candidate.get("status") != "SUCCESS":
            continue
        completed += 1
        paths = _intervention_path(arm_id, baseline, candidate)
        active_baseline += int(paths[0])
        changed += int(paths[1])
    if completed != len(questions):
        return {"status": "INCOMPLETE", "reason": f"Only {completed}/{len(questions)} paired query traces are complete.",
                "baseline_active_queries": active_baseline, "off_path_changed_queries": changed}
    if arm.get("kind") == "primary" and active_baseline == 0:
        return {"status": "SCIENTIFIC_VALIDATION_FAILED", "reason": "The intervention was inactive in every B6 reference query.",
                "baseline_active_queries": active_baseline, "off_path_changed_queries": changed}
    if changed == 0:
        return {"status": "SCIENTIFIC_VALIDATION_FAILED", "reason": "No off-arm trace proved the intervention changed an execution path.",
                "baseline_active_queries": active_baseline, "off_path_changed_queries": changed}
    return {"status": "PASSED", "reason": "Reference and off-arm traces show the declared path was disabled; answer/context equality is allowed.",
            "baseline_active_queries": active_baseline, "off_path_changed_queries": changed}


def _intervention_path(arm_id: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, bool]:
    left = baseline.get("retrieval_candidates_and_ranking_trace", {})
    right = candidate.get("retrieval_candidates_and_ranking_trace", {})
    if arm_id == "B6_minus_score_blending":
        return (left.get("ranking", {}).get("ranking_formula") == "normalized_cross_encoder_rrf_blend",
                right.get("ranking", {}).get("ranking_formula") == "raw_cross_encoder_score")
    if arm_id == "B6_minus_final_reranking":
        return (left.get("ranking", {}).get("stage_status") == "complete",
                right.get("ranking", {}).get("stage_status") == "reranker_disabled")
    if arm_id == "B6_minus_dynamic_graph_expander":
        return (left.get("context_and_prompt_assembly", {}).get("dynamic_graph_expansion_executed") is True,
                right.get("context_and_prompt_assembly", {}).get("dynamic_graph_expansion_executed") is False)
    if arm_id == "B6_minus_static_graph_context":
        return (left.get("context_and_prompt_assembly", {}).get("static_graph_context_enabled") is True,
                right.get("context_and_prompt_assembly", {}).get("static_graph_context_disabled") is True)
    if arm_id == "B6_minus_intrapaper_graph_candidates":
        left_branch = left.get("graph_retrieval", {}).get("branch_execution", {})
        right_branch = right.get("graph_retrieval", {}).get("branch_execution", {})
        return (left_branch.get("intra_paper_branch_enabled") is True
                and left_branch.get("intra_paper_branch_executed") is True,
                right_branch.get("intra_paper_branch_enabled") is False
                and right_branch.get("intra_paper_branch_executed") is False)
    if arm_id == "B6_interaction_graph_candidate_x_dynamic_expander":
        left_branch = left.get("graph_retrieval", {}).get("branch_execution", {})
        right_branch = right.get("graph_retrieval", {}).get("branch_execution", {})
        baseline_active = (left_branch.get("intra_paper_branch_enabled") is True
                           and left_branch.get("intra_paper_branch_executed") is True
                           and left.get("context_and_prompt_assembly", {}).get("dynamic_graph_expansion_executed") is True)
        candidate_disabled = (right_branch.get("intra_paper_branch_enabled") is False
                              and right_branch.get("intra_paper_branch_executed") is False
                              and right.get("context_and_prompt_assembly", {}).get("dynamic_graph_expansion_executed") is False)
        return baseline_active, candidate_disabled
    if arm_id == "B6_minus_concept_retrieval":
        left_branch = left.get("graph_retrieval", {}).get("branch_execution", {})
        right_branch = right.get("graph_retrieval", {}).get("branch_execution", {})
        return (left_branch.get("concept_branch_enabled") is True
                and left_branch.get("concept_lookup_executed") is True,
                right_branch.get("concept_branch_enabled") is False
                and right_branch.get("concept_lookup_executed") is False)
    if arm_id == "B6_minus_bridge_retrieval":
        left_branch = left.get("graph_retrieval", {}).get("branch_execution", {})
        right_branch = right.get("graph_retrieval", {}).get("branch_execution", {})
        return (left_branch.get("bridge_branch_enabled") is True
                and left_branch.get("bridge_lookup_executed") is True,
                right_branch.get("bridge_branch_enabled") is False
                and right_branch.get("bridge_lookup_executed") is False)
    if arm_id == "B6_minus_ontology_lookup":
        return (left.get("current_trace", {}).get("query_expansion", {}).get("ontology_lookup_enabled") is True,
                right.get("current_trace", {}).get("query_expansion", {}).get("ontology_lookup_enabled") is False)
    if arm_id == "B6_minus_parent_deduplication":
        return (left.get("ranking", {}).get("parent_deduplication_branch_executed") is True,
                right.get("ranking", {}).get("parent_deduplication_gate_evaluated") is True
                and right.get("ranking", {}).get("parent_deduplication_branch_executed") is False)
    if arm_id in {"B6_minus_lexical_retrieval", "B6_minus_dense_retrieval"}:
        key = "lexical" if "lexical" in arm_id else "dense"
        path = "lexical_search" if key == "lexical" else "dense_search"
        trace_key = f"{key}_retrieval_enabled"
        baseline_seeds = left.get("current_trace", {}).get("seed_chunks_from_lexical_dense", {}).get(key, [])
        candidate_seeds = right.get("current_trace", {}).get("seed_chunks_from_lexical_dense", {}).get(key, [])
        return (left.get("current_trace", {}).get("retrieval_paths", {}).get(trace_key) is True
                and bool(baseline_seeds),
                right.get("current_trace", {}).get("retrieval_paths", {}).get(trace_key) is False
                and not candidate_seeds)
    if arm_id == "B6_fixed_budget_context":
        return (left.get("context_capture", {}).get("context_trim") is not None,
                right.get("context_and_prompt_assembly", {}).get("fixed_budget_policy_executed") is True)
    if arm_id == "B6_interaction_reranker_x_blending":
        return (left.get("ranking", {}).get("stage_status") == "complete",
                right.get("ranking", {}).get("stage_status") == "reranker_disabled"
                and right.get("ranking", {}).get("score_blending_branch_executed") is False)
    return False, False
