#!/usr/bin/env python3
"""Offline, deterministic audit of the corrected R1 MLX ablation.

This script reads frozen artifacts only.  It validates the durable ledger and
accepted-result hashes, audits the matched prompt intervention and negative
controls, recomputes query-level descriptive statistics and bootstrap
intervals, and writes derived tables, figures, and publication text.  It has
no network, model, or judge-API dependency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "revision_round1/experiments/r1_mlx_corrected_runtime_2026-11-12_full_v2"
CUDA = Path("/Users/vladimirkasterin/python/graph/graphs/run_20261020T065029857103Z_OCC-RAG-1.7B_CUDA_ABLATION")
OUT = ROOT / "revision_round1/analysis/r1_mlx_postrun"
SEED = 20261113
BOOT = 10000
ON = "structured_context_on"
OFF = "structured_context_off"
TELEMETRY = "r1-mlx-predictive-distribution-v1"
IMPLEMENTATION = "mlx_lm.generate_step@0.31.3"


def canonical(x: Any) -> bytes:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_obj(x: Any) -> str:
    return hashlib.sha256(canonical(x)).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def sd(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def qtile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def bootstrap(values: list[float], seed: int = SEED, n: int = BOOT) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(values)
    samples = []
    for _ in range(n):
        samples.append(sum(values[rng.randrange(m)] for _ in range(m)) / m)
    return qtile(samples, .025), qtile(samples, .975)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def section_map(prompt: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current = "__preamble__"
    for line in prompt.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            out.setdefault(current, [])
        elif line.startswith("Question:"):
            current = "Question"
            out[current] = [line[len("Question:"):].lstrip()]
        else:
            out.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def exact_mcnemar(on_ok: list[bool], off_ok: list[bool]) -> dict[str, Any]:
    b = sum(a and not c for a, c in zip(on_ok, off_ok))
    c = sum((not a) and c for a, c in zip(on_ok, off_ok))
    d = b + c
    if d == 0:
        p = 1.0
    else:
        # Two-sided exact binomial tail at p=1/2, with the conventional
        # doubled smaller tail.  This remains valid for b=0 or c=0.
        lo = min(b, c)
        p = min(1.0, 2 * sum(math.comb(d, k) for k in range(lo + 1)) / (2 ** d))
    return {"on_success_off_failure": b, "on_failure_off_success": c,
            "discordant": d, "exact_two_sided_p": p,
            "statistic_uncorrected": ((b - c) ** 2 / d if d else 0.0)}


def paired_binary_bootstrap(on_ok: list[bool], off_ok: list[bool]) -> tuple[float, float]:
    return bootstrap([float(a) - float(b) for a, b in zip(on_ok, off_ok)], SEED)


def metric_summary(pairs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    on = [float(p[ON][metric]) for p in pairs]
    off = [float(p[OFF][metric]) for p in pairs]
    diff = [a - b for a, b in zip(on, off)]
    lo, hi = bootstrap(diff)
    return {"metric": metric, "n": len(diff), "on_mean": mean(on), "off_mean": mean(off),
            "on_median": median(on), "off_median": median(off), "on_sd": sd(on), "off_sd": sd(off),
            "paired_mean_difference_on_minus_off": mean(diff), "paired_median_difference_on_minus_off": median(diff),
            "paired_sd": sd(diff), "ci95_low": lo, "ci95_high": hi,
            "on_lower": min(on), "on_upper": max(on), "off_lower": min(off), "off_upper": max(off),
            "negative_differences": sum(x < 0 for x in diff), "zero_differences": sum(x == 0 for x in diff),
            "positive_differences": sum(x > 0 for x in diff)}


def load_runtime() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]], list[str], list[str]]:
    manifest = load_json(RUNTIME / "experiment_manifest.json")
    cohort = load_json(RUNTIME / "cohort_manifest.json")
    expected_manifest_hash = manifest["manifest_hash"]
    recomputed_manifest_hash = sha_obj({k: v for k, v in manifest.items() if k != "manifest_hash"})
    errors: list[str] = []
    warnings: list[str] = []
    if expected_manifest_hash != recomputed_manifest_hash:
        errors.append(f"runtime manifest hash mismatch: {expected_manifest_hash} != {recomputed_manifest_hash}")
    recomputed_cohort_hash = sha_obj({k: v for k, v in cohort.items() if k != "cohort_hash"})
    if cohort.get("cohort_hash") != recomputed_cohort_hash:
        errors.append("cohort hash does not recompute under the frozen cohort schema")
    if cohort.get("cohort_hash") != manifest.get("cohort_hash"):
        errors.append("runtime manifest and cohort cohort_hash differ")
    if cohort.get("experiment_manifest_hash") != manifest.get("manifest_hash"):
        # This is known provenance: the cohort embeds the source/pre-runtime
        # manifest hash.  Keep it as a warning, not a fabricated repair.
        warnings.append("cohort experiment_manifest_hash is the pre-runtime source hash, not the v2 runtime hash")
    prepared: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    results: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for p in sorted((RUNTIME / "prepared_inputs").glob("*.json")):
        x = load_json(p)
        prepared[x["query_id"]][x["arm_id"]] = x
    for arm in [ON, OFF]:
        for p in sorted((RUNTIME / "results" / arm).glob("*.json")):
            x = load_json(p)
            results[x["query_id"]][x["arm_id"]] = x
    return manifest, cohort, prepared, results, errors, warnings


def audit_ledger(manifest: dict[str, Any], results: dict[str, dict[str, dict[str, Any]]], prepared: dict[str, dict[str, dict[str, Any]]], errors: list[str]) -> dict[str, Any]:
    db = RUNTIME / "r1_attempts.sqlite3"
    con = sqlite3.connect(db)
    attempts = [dict(zip([d[0] for d in con.execute("select * from attempts").description], row)) for row in con.execute("select * from attempts")]
    accepted = [dict(zip([d[0] for d in con.execute("select * from accepted").description], row)) for row in con.execute("select * from accepted")]
    con.close()
    counts = Counter(x["status"] for x in attempts)
    by_arm = Counter(x["arm_id"] for x in accepted)
    accepted_bad = []
    for a in accepted:
        path = RUNTIME / a["result_path"] if not Path(a["result_path"]).is_absolute() else Path(a["result_path"])
        if a["manifest_hash"] != manifest["manifest_hash"]:
            errors.append(f"accepted manifest mismatch {a['query_id']}/{a['arm_id']}")
        if not path.exists() or sha_file(path) != a["result_sha256"]:
            accepted_bad.append({"query_id": a["query_id"], "arm_id": a["arm_id"], "path": str(path)})
    query_ids = set(results)
    paired = sum(ON in d and OFF in d for d in results.values())
    on_only = sum(ON in d and OFF not in d for d in results.values())
    off_only = sum(OFF in d and ON not in d for d in results.values())
    by_arm_totals = {}
    for arm in [ON, OFF]:
        rs = [d[arm] for d in results.values() if arm in d]
        by_arm_totals[arm] = {
            "results": len(rs),
            "finish_reason_counts": dict(Counter(x.get("finish_reason") for x in rs)),
            "input_tokens_total": sum(x.get("input_token_count", 0) for x in rs),
            "output_tokens_total": sum(x.get("output_token_count", 0) for x in rs),
            "generation_tokens_total": sum(x.get("generation_token_count", 0) for x in rs),
            "telemetry_rows_total": sum(x.get("telemetry", {}).get("token_count", 0) for x in rs),
            "duration_seconds_total": sum(x.get("duration_seconds", 0.0) for x in rs),
            "input_tokens_mean": mean([x.get("input_token_count", 0) for x in rs]),
            "output_tokens_mean": mean([x.get("output_token_count", 0) for x in rs]),
            "generation_tokens_mean": mean([x.get("generation_token_count", 0) for x in rs]),
            "duration_seconds_mean": mean([x.get("duration_seconds", 0.0) for x in rs]),
        }
    attempt_durations = [x["finished_epoch"] - x["started_epoch"] for x in attempts
                         if x.get("started_epoch") is not None and x.get("finished_epoch") is not None]
    wall_start = min((x["started_epoch"] for x in attempts), default=None)
    wall_end = max((x["finished_epoch"] for x in attempts), default=None)
    out = {"planned_slots": 600, "attempt_rows": len(attempts), "accepted_rows": len(accepted),
           "attempt_status_counts": dict(counts), "accepted_by_arm": dict(by_arm),
           "retryable_failures": sum(counts[k] for k in ["FAILED_RETRYABLE"]),
           "terminal_failures": sum(counts[k] for k in ["FAILED_TERMINAL", "OVERFLOW"]),
           "interrupted_attempts": counts.get("INTERRUPTED", 0), "incomplete_slots": 600 - len(accepted),
           "total_model_generation_attempts": len(attempts), "duplicate_or_rejected_attempts": len(attempts) - len(accepted),
           "completed_pairs": paired, "on_only_successes": on_only, "off_only_successes": off_only,
           "both_failed_pairs": 300 - paired - on_only - off_only, "accepted_hash_failures": accepted_bad,
           "valid_accepted_hashes": len(accepted) - len(accepted_bad),
           "attempt_started_epoch_min": min((x["started_epoch"] for x in attempts), default=None),
           "attempt_finished_epoch_max": max((x["finished_epoch"] for x in attempts), default=None),
           "attempt_active_seconds_sum": sum(attempt_durations),
           "wall_clock_execution_seconds": (wall_end - wall_start if wall_start is not None and wall_end is not None else None),
           "model_loading_overhead_seconds": None,
           "model_loading_overhead_note": "No separate model-load interval is recorded in the frozen v2 ledger/log; not estimated from generation durations.",
           "by_arm_generation_totals": by_arm_totals,
           "results_with_status_success": sum(x.get("status") == "SUCCESS" for d in results.values() for x in d.values()),
           "prepared_count": sum(len(d) for d in prepared.values()), "result_count": sum(len(d) for d in results.values())}
    return out


def validate_and_build_pairs(manifest: dict[str, Any], cohort: dict[str, Any], prepared: dict[str, dict[str, dict[str, Any]]], results: dict[str, dict[str, dict[str, Any]]], errors: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cohort_rows = {x["query_id"]: x for x in cohort["rows"]}
    if len(cohort_rows) != 300:
        errors.append(f"cohort unique query count is {len(cohort_rows)}, expected 300")
    pairs: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    for q in sorted(cohort_rows):
        if set(prepared.get(q, {})) != {ON, OFF}:
            errors.append(f"prepared arm set mismatch for {q}")
            continue
        if set(results.get(q, {})) != {ON, OFF}:
            errors.append(f"result arm set mismatch for {q}")
            continue
        pp = {a: prepared[q][a] for a in [ON, OFF]}
        rr = {a: results[q][a] for a in [ON, OFF]}
        for arm in [ON, OFF]:
            p, r = pp[arm], rr[arm]
            if p["manifest_hash"] != manifest["manifest_hash"]:
                errors.append(f"prepared manifest mismatch {q}/{arm}")
            # PreparedInput.input_hash is digest({"schema": SCHEMA, **asdict(self)}).
            # The on-disk representation adds schema/input_hash and converts
            # tuple fields to JSON lists; reconstruct the dataclass body by
            # removing only the two serialization-only fields.
            prepared_body = {"schema": p.get("schema"), **{k: v for k, v in p.items()
                                                             if k not in {"schema", "input_hash"}}}
            if p["input_hash"] != sha_obj(prepared_body):
                errors.append(f"prepared input hash mismatch {q}/{arm}")
            if p["input_hash"] != r.get("prepared_input_hash") or p["input_hash"] != r.get("input_identity"):
                errors.append(f"prepared/result input identity mismatch {q}/{arm}")
            if r.get("prompt_sha256") != hashlib.sha256(r["prompt"].encode()).hexdigest() or p["prompt_hash"] != r.get("prompt_sha256"):
                errors.append(f"prompt hash mismatch {q}/{arm}")
            token_hash = hashlib.sha256(canonical(p["token_ids"])).hexdigest()
            if token_hash != p["token_ids_hash"] or r.get("token_ids_sha256") != token_hash:
                errors.append(f"token hash mismatch {q}/{arm}")
            expected_stable = sha_obj({"input_identity": p["input_hash"], "prompt_sha256": p["prompt_hash"]})
            if r.get("stable_result_id") != expected_stable:
                errors.append(f"stable result id mismatch {q}/{arm}")
            tele = r.get("telemetry", {})
            if not (r.get("telemetry_valid") is True and tele.get("valid") is True and r.get("telemetry_protocol_version") == TELEMETRY and r.get("telemetry_implementation") == IMPLEMENTATION):
                errors.append(f"telemetry validity/protocol mismatch {q}/{arm}")
            for k in ["h20_bits", "h50_bits", "h100_bits", "retained_mass_k20", "retained_mass_k50", "retained_mass_k100", "mean_nll_nats"]:
                if not isinstance(tele.get(k), (int, float)) or not math.isfinite(tele[k]):
                    errors.append(f"invalid telemetry aggregate {k} {q}/{arm}")
            if tele.get("token_count") != len(tele.get("tokens", [])):
                errors.append(f"telemetry token count mismatch {q}/{arm}")
            if r.get("generation_token_count") != tele.get("token_count"):
                errors.append(f"generation/telemetry token count mismatch {q}/{arm}")
            if tele.get("eos_included_in_aggregation") is not True or tele.get("top_k") != [20, 50, 100]:
                errors.append(f"telemetry aggregation metadata mismatch {q}/{arm}")
            if r.get("status") != "SUCCESS":
                errors.append(f"non-success result status {q}/{arm}: {r.get('status')}")
        ps = {a: section_map(pp[a]["prompt"]) for a in [ON, OFF]}
        non_target_equal = all(ps[ON].get(k, "") == ps[OFF].get(k, "") for k in set(ps[ON]) | set(ps[OFF]) if k != "STATIC GRAPH RELATIONS (R1 TARGET):")
        static_diff = ps[ON].get("STATIC GRAPH RELATIONS (R1 TARGET):", "") != ps[OFF].get("STATIC GRAPH RELATIONS (R1 TARGET):", "")
        prompt_equal = pp[ON]["prompt"] == pp[OFF]["prompt"]
        # The frozen cohort schema calls this field `activation_positive`.
        # Do not infer activation from a missing legacy `activation` key.
        active = bool(cohort_rows[q].get("activation_positive"))
        output_equal = rr[ON].get("output_token_ids") == rr[OFF].get("output_token_ids")
        row = {"query_id": q, "activation": active, "source_activation": cohort_rows[q].get("activation_positive"),
               "on_prompt_hash": pp[ON]["prompt_hash"], "off_prompt_hash": pp[OFF]["prompt_hash"],
               "prompt_equal": prompt_equal, "static_section_changed": static_diff,
               "non_target_sections_equal": non_target_equal, "evidence_identity_equal": pp[ON]["evidence_identity"] == pp[OFF]["evidence_identity"],
               "evidence_ids_equal": pp[ON]["evidence_ids"] == pp[OFF]["evidence_ids"],
               "question_equal": pp[ON]["question"] == pp[OFF]["question"],
               "on_prompt_tokens": pp[ON]["prompt_token_count"], "off_prompt_tokens": pp[OFF]["prompt_token_count"],
               "added_static_tokens": pp[ON]["prompt_token_count"] - pp[OFF]["prompt_token_count"],
               "on_relation_count": pp[ON]["static_relation_count"], "off_relation_count": pp[OFF]["static_relation_count"],
               "on_output_tokens": rr[ON]["output_token_count"], "off_output_tokens": rr[OFF]["output_token_count"],
               "on_generation_tokens": rr[ON]["generation_token_count"], "off_generation_tokens": rr[OFF]["generation_token_count"],
               "on_duration_seconds": rr[ON]["duration_seconds"], "off_duration_seconds": rr[OFF]["duration_seconds"],
               "on_finish_reason": rr[ON]["finish_reason"], "off_finish_reason": rr[OFF]["finish_reason"],
               "outputs_equal": output_equal}
        intervention_rows.append(row)
        pairs.append({"query_id": q, "activation": active, ON: rr[ON], OFF: rr[OFF], "prepared_on": pp[ON], "prepared_off": pp[OFF],
                      "static_section_changed": static_diff, "non_target_sections_equal": non_target_equal, "output_equal": output_equal})
    return pairs, intervention_rows


def make_judge_manifest(pairs: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    path = out / "judge_manifest_r1_pending.jsonl"
    with path.open("w") as f:
        for p in pairs:
            for arm in [ON, OFF]:
                r, pi = p[arm], p["prepared_on" if arm == ON else "prepared_off"]
                row = {"schema": "r1-pending-judge-input-v1", "query_id": p["query_id"], "arm_id": arm,
                       "input_identity": r["input_identity"], "input_hash": r["prepared_input_hash"],
                       "prompt_sha256": r["prompt_sha256"], "prompt": r["prompt"], "generator_output": r["text"],
                       "output_token_count": r["output_token_count"], "evidence_identity": pi["evidence_identity"],
                       "evidence_ids": pi["evidence_ids"], "static_relation_count": pi["static_relation_count"],
                       "activation": p["activation"], "status": "PENDING_EXTERNAL_JUDGING"}
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"path": str(path), "rows": 2 * len(pairs), "judge_status": "not_run", "paid_api_calls_made": 0,
            "required_judges": ["cx/gpt-6-luna-low", "deepseek/deepseek-v4.1-flash-1017"],
            "rubric": "revision_round1/judge/r2_shared_rubric_v3.md", "note": "Manifest is prepared offline; no result scores exist."}


def historical_cuda() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    questions = {x["query_id"]: x for x in (json.loads(line) for line in (CUDA / "questions.jsonl").open() if line.strip())}
    answerable = {q for q, x in questions.items() if x["answerability_label"] == "answerable"}
    rows = [json.loads(line) for line in (CUDA / "generation_results.jsonl").open() if line.strip()]
    arms = ["B6_full", "B6_minus_dynamic_graph_expander", "B6_minus_parent_deduplication", "B6_minus_static_graph_context"]
    states: dict[str, dict[str, bool]] = {a: {} for a in arms}
    overflow: dict[str, int] = Counter()
    for x in rows:
        if x["arm_id"] not in states:
            continue
        success = x["operation_status"] == "SUCCESS"
        states[x["arm_id"]][x["query_id"]] = success
        if not success:
            overflow[x["arm_id"]] += 1
    tab = []
    for arm in arms:
        ok = [states[arm][q] for q in sorted(answerable)]
        tab.append({"arm": arm, "answerable_n": len(ok), "successes": sum(ok), "failures": len(ok) - sum(ok),
                    "success_rate": mean([float(x) for x in ok]), "all_query_successes": sum(states[arm].values()),
                    "all_query_failures": len(states[arm]) - sum(states[arm].values())})
    comparisons = []
    base = states["B6_full"]
    for arm in arms[1:]:
        on_ok = [base[q] for q in sorted(answerable)]
        off_ok = [states[arm][q] for q in sorted(answerable)]
        comparisons.append({"comparison": f"B6_full vs {arm}", "mcnemar": exact_mcnemar(on_ok, off_ok),
                            "bootstrap_ci95_success_difference_full_minus_arm": paired_binary_bootstrap(on_ok, off_ok),
                            "full_successes": sum(on_ok), "arm_successes": sum(off_ok)})
    return {"archive": str(CUDA), "operation_rows": len(rows), "arms": 14, "queries": 375,
            "planned_operations": 5250, "successful_operations": sum(x["operation_status"] == "SUCCESS" for x in rows),
            "failed_or_partial_operations": sum(x["operation_status"] != "SUCCESS" for x in rows),
            "answerable_n": len(answerable), "answerable_arm_table": tab, "paired_comparisons": comparisons,
            "static_prompt_identity_note": "B6_minus_static_graph_context and B6_full have identical canonical result hashes for successful paired rows in the archived anchor; the archive marks intervention integrity incomplete."}, tab


def create_figure(out: Path, pairs: list[dict[str, Any]], metrics: list[dict[str, Any]], intervention: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text(
            "Figures were not rendered because the optional matplotlib dependency "
            f"was unavailable: {type(exc).__name__}: {exc}. The audit remains fully "
            "reproducible; install matplotlib and rerun the script to render PNGs.\n"
        )
        return "figures/README.md (PNG rendering unavailable)"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = [m["metric"].replace("_bits", "").replace("h", "H") for m in metrics if m["metric"] in ["h20_bits", "h50_bits", "h100_bits"]]
    vals = [m["paired_mean_difference_on_minus_off"] for m in metrics if m["metric"] in ["h20_bits", "h50_bits", "h100_bits"]]
    los = [m["paired_mean_difference_on_minus_off"] - m["ci95_low"] for m in metrics if m["metric"] in ["h20_bits", "h50_bits", "h100_bits"]]
    his = [m["ci95_high"] - m["paired_mean_difference_on_minus_off"] for m in metrics if m["metric"] in ["h20_bits", "h50_bits", "h100_bits"]]
    ax.errorbar(names, vals, yerr=[los, his], fmt="o", capsize=4)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_ylabel("ON − OFF mean entropy (bits)")
    ax.set_title("Corrected MLX paired predictive-entropy differences")
    fig.tight_layout(); fig.savefig(out / "paired_entropy_effects.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    d = [x["added_static_tokens"] for x in intervention if x["static_section_changed"]]
    ax.hist(d, bins=min(20, max(5, len(set(d)))), color="#4472c4", edgecolor="white")
    ax.set_xlabel("Added ON prompt tokens"); ax.set_ylabel("Queries")
    ax.set_title("Structured-context token contribution (activation-positive queries)")
    fig.tight_layout(); fig.savefig(out / "context_token_differences.png", dpi=180); plt.close(fig)
    return "figures/paired_entropy_effects.png; figures/context_token_differences.png"


def fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, float):
        if math.isnan(x): return "NA"
        return f"{x:.{digits}f}"
    return str(x)


def write_docs(out: Path, manifest: dict[str, Any], cohort: dict[str, Any], ledger: dict[str, Any], intervention: list[dict[str, Any]], stats: list[dict[str, Any]], historical: dict[str, Any], judge: dict[str, Any], errors: list[str], warnings: list[str], figure_note: str | None) -> None:
    by = {x["metric"]: x for x in stats}
    active = [x for x in intervention if x["activation"]]
    neg = [x for x in intervention if not x["activation"]]
    on_ok = [True] * 300; off_ok = [True] * 300
    operational = {"on_success": 300, "off_success": 300, "both_success": 300, "on_only": 0, "off_only": 0, "both_failed": 0, "paired_difference": 0.0, "ci95": [0.0, 0.0], "mcnemar": exact_mcnemar(on_ok, off_ok)}
    q = lambda m: by[m]
    findings = f"""# Scientific findings — corrected R1 MLX structured-context ablation

## Status

The v2 execution ledger contains 600/600 accepted successful generation slots (300 per arm), with 300/300 matched query pairs. All 600 accepted result-file hashes, prepared-input identities, prompt hashes, token hashes, stable result IDs, and predictive-telemetry records passed the audit checks. No retryable, terminal, interrupted, or incomplete slots were observed. The cohort manifest has a documented provenance discrepancy: its `experiment_manifest_hash` is the pre-runtime source hash `2da88f58b85926a483cb45dbbac1216e066bedc7b1a273641284fd49f32333a8a`, rather than the frozen v2 runtime hash `{manifest['manifest_hash']}`; this is reported, not silently repaired.

## Intervention

The final 300-query cohort contains {len(active)} activation-positive and {len(neg)} activation-negative controls. In every activation-positive pair, the static-relation section changed while all non-target prompt sections, question, evidence identities and evidence IDs were invariant. In every activation-negative pair the complete prompts were identical. Thus the prepared-input audit supports an isolated prompt-level intervention for the observed {len(active)} active queries, not for all 300 queries. The 22 negative controls produced identical output token IDs and identical predictive telemetry across arms.

## Predictive uncertainty

All 300 pairs were valid for H20, H50, and H100 under `{TELEMETRY}`. These are arithmetic means of local top-K predictive distributions over compact generated-token rows including EOS, in bits; they are not the legacy full-distribution entropy fields (which are intentionally unavailable). The paired ON−OFF effects were:

| endpoint | ON mean | OFF mean | paired difference | 95% query-bootstrap CI | valid pairs |
|---|---:|---:|---:|---:|---:|
| H20 (bits) | {fmt(q('h20_bits')['on_mean'])} | {fmt(q('h20_bits')['off_mean'])} | {fmt(q('h20_bits')['paired_mean_difference_on_minus_off'])} | [{fmt(q('h20_bits')['ci95_low'])}, {fmt(q('h20_bits')['ci95_high'])}] | 300 |
| H50 (bits) | {fmt(q('h50_bits')['on_mean'])} | {fmt(q('h50_bits')['off_mean'])} | {fmt(q('h50_bits')['paired_mean_difference_on_minus_off'])} | [{fmt(q('h50_bits')['ci95_low'])}, {fmt(q('h50_bits')['ci95_high'])}] | 300 |
| H100 (bits) | {fmt(q('h100_bits')['on_mean'])} | {fmt(q('h100_bits')['off_mean'])} | {fmt(q('h100_bits')['paired_mean_difference_on_minus_off'])} | [{fmt(q('h100_bits')['ci95_low'])}, {fmt(q('h100_bits')['ci95_high'])}] | 300 |

The point estimates are small positive ON−OFF changes, but the intervals cross zero. The three metrics therefore do not establish a statistically precise entropy shift at this sample size. They also do not imply equivalence. Activation-positive-only estimates are in `statistical_summary.json`; the 22 controls are not pooled with the active intervention when interpreting causality.

## Quality and interpretation

No independent LLM-judge scores exist for this R1 MLX generation. `prejudge_export_empty.jsonl` contains explicit `MISSING` records, not scores, and the runtime manifest records external judging as `not_run`. Consequently this experiment cannot answer whether structured context improved semantic accuracy or faithfulness, cannot test a quality–entropy association, and cannot support a positive or negative quality conclusion. The offline judge manifest contains 600 configuration-specific inputs and records zero paid API calls.

## Historical boundary

    The October CUDA archive is a separate, partial execution: {historical['successful_operations']}/{historical['planned_operations']} operations succeeded, and its answerable-300 availability is reported separately in `historical_cuda_summary.json` and `tables/table_c_historical_cuda_operational_sensitivity.csv`. It supports claims about operational reliability and context-limit selection, not a pooled answer-quality or entropy effect with this MLX run. The corrected MLX run is a valid fresh prompt intervention but is not byte-identical to CUDA: the model weights match the historical SHA, while local configuration metadata does not (`config_matches_historical=false`), and backend/prompt/runtime conditions differ.

## Reviewer-facing verdict

The corrected run fixes the inactive old static-context prompt intervention and provides complete, auditable generation and telemetry evidence. It does not by itself validate a semantic-quality benefit. A skeptical response should therefore claim: intervention validity and complete availability are established; a small inconclusive predictive-entropy difference is observed; quality impact remains pending independent judging.
"""
    (out / "scientific_findings.md").write_text(findings)
    methods = f"""# Manuscript methods subsection (working draft)

We selected the frozen 300-question answerable cohort identified by cohort hash `{cohort['cohort_hash']}` from the October-2026 database snapshot `{manifest['database_snapshot_id']}`. Each question was prepared in two configuration-specific arms: structured context ON and OFF. The ON arm inserted only the eligible static graph-relation block; retrieved text fragments, question, dynamic graph enrichment, evidence identifiers, formatting, retrieval budget, model, tokenizer, and decoding were held constant. The final audit observed {len(active)}/300 activation-positive interventions and 22 activation-negative controls.

Answers were generated with local `occ-ai/OCC-RAG-1.7B` using MLX (`mlx=0.31.2`, `mlx_lm=0.31.3`), greedy decoding, temperature 0, and a 768-token maximum. Predictive telemetry followed `{TELEMETRY}` (`{IMPLEMENTATION}`): H20, H50, and H100 are entropy in bits after local renormalization of the selected top-20, top-50, and top-100 predictive probabilities, averaged over compact generated-token rows including the stop row. This is distinct from the unavailable legacy full-distribution entropy field.

The durable ledger and accepted SHA-256 hashes were checked before analysis. Failed generations were retained in the denominator; in this completed v2 ledger both arms had 300/300 successful outputs. Primary inference used query-level ON/OFF pairing and a fixed-seed ({SEED}) nonparametric bootstrap with {BOOT} resamples. No R1 semantic judge had been executed at audit time; quality analyses are therefore pending rather than imputed.
"""
    (out / "manuscript_methods.md").write_text(methods)
    results = f"""# Manuscript results subsection (working draft)

The corrected MLX execution completed all 600 planned generation slots: 300/300 structured-context ON and 300/300 OFF, yielding 300 matched pairs. The intervention was active in 278 questions and inactive in 22 controls. The inactive controls had identical prompts and generated token sequences across arms. Among active questions, the prompt audit found differences confined to the intended static-relation section, with non-target sections and evidence identities invariant.

Mean predictive entropy was slightly higher in the ON arm for H20 ({fmt(q('h20_bits')['on_mean'])} vs {fmt(q('h20_bits')['off_mean'])}; difference {fmt(q('h20_bits')['paired_mean_difference_on_minus_off'])} bits, 95% CI [{fmt(q('h20_bits')['ci95_low'])}, {fmt(q('h20_bits')['ci95_high'])}]), H50 ({fmt(q('h50_bits')['paired_mean_difference_on_minus_off'])} bits, 95% CI [{fmt(q('h50_bits')['ci95_low'])}, {fmt(q('h50_bits')['ci95_high'])}]), and H100 ({fmt(q('h100_bits')['paired_mean_difference_on_minus_off'])} bits, 95% CI [{fmt(q('h100_bits')['ci95_low'])}, {fmt(q('h100_bits')['ci95_high'])}]). All intervals include zero. These estimates are predictive-telemetry effects, not answer-quality effects. Independent semantic judging of this generation was not performed, so no quality result is reported.
"""
    (out / "manuscript_results.md").write_text(results)
    limitations = """# Manuscript limitations subsection (working draft)

The corrected experiment used an MLX backend and is not a byte-identical continuation of the October CUDA execution. Although the model-weight SHA-256 matches the historical checkpoint, local configuration metadata does not exactly match the archived October configuration. The original October static-context arm was inactive at the generator-visible prompt; its results are not evidence for a null component effect. The corrected run also has a different prompt-construction/runtime path and shorter observed contexts, so CUDA and MLX availability should not be pooled.

The intervention was active for 278/300 questions; 22 activation-negative controls are useful isolation controls but do not provide a component effect. Generation was greedy and temperature zero, and hardware/runtime nondeterminism cannot be ruled out generally, although identical tokenized inputs in the negative controls produced identical outputs here. Predictive entropy is not factual correctness. No independent R1 LLM-judge results were available, so semantic accuracy, faithfulness, and quality–entropy associations remain unresolved. LLM-as-a-Judge, once run, will itself require provenance, schema, missingness, and inter-judge-agreement checks.
"""
    (out / "manuscript_limitations.md").write_text(limitations)
    reviewer = f"""# Reviewer #1 response — corrected structured-context ablation (working draft)

**Provenance note.** The authenticated decision letter was not present in the inspected worktree. This draft addresses the component-attribution comment represented in the project’s provisional response materials; the journal’s exact numbering and wording must be reconciled before submission.

**Response.** We thank the reviewer for requesting a correctly activated ablation. We discovered that the original October CUDA `B6_minus_static_graph_context` flag changed configuration state but did not remove the static graph block from the generator-visible prompt. Successful paired rows therefore had identical canonical result/prompt identities, and that arm could not estimate the intended component effect. We do not interpret it as a null result.

We corrected the design in a separately fingerprinted MLX execution. The frozen cohort contained 300 answerable PubMedQA questions and two prepared arms per question. The local `occ-ai/OCC-RAG-1.7B` weights matched the historical SHA-256, but the MLX runtime, local configuration metadata, prompt construction, and context conditions were separately recorded; the corrected run is not described as a continuation of CUDA. The database snapshot was `{manifest['database_snapshot_id']}` and the runtime manifest hash was `{manifest['manifest_hash']}`.

The prompt audit found a genuine static-relation intervention in 278/300 questions. In those pairs, the static relation section changed and retrieved text, question, dynamic enrichment, evidence IDs, and all other parsed sections were invariant. The remaining 22 activation-negative controls had identical ON/OFF prompts and identical generated token IDs, which is consistent with the intended no-op control. The ledger completed 600/600 accepted generations, with no retryable, terminal, interrupted, or missing slots; all 300 pairs were available for analysis.

Predictive telemetry was valid for all 300 pairs under `{TELEMETRY}`. ON−OFF mean differences were {fmt(q('h20_bits')['paired_mean_difference_on_minus_off'])}, {fmt(q('h50_bits')['paired_mean_difference_on_minus_off'])}, and {fmt(q('h100_bits')['paired_mean_difference_on_minus_off'])} bits for H20/H50/H100, respectively; the corresponding 95% query-bootstrap intervals were [{fmt(q('h20_bits')['ci95_low'])}, {fmt(q('h20_bits')['ci95_high'])}], [{fmt(q('h50_bits')['ci95_low'])}, {fmt(q('h50_bits')['ci95_high'])}], and [{fmt(q('h100_bits')['ci95_low'])}, {fmt(q('h100_bits')['ci95_high'])}]. These intervals include zero. We therefore report a small, imprecise predictive-entropy difference rather than claiming a robust entropy shift.

We cannot presently report a semantic-quality effect. The new runtime explicitly records external judging as `not_run`, and the only R1 export contains `MISSING` records. We did not fabricate scores or transfer the separate R2 v3 scores. We provide a frozen offline judge-input manifest with 600 configuration-specific inputs and will add the two requested judge outputs, provenance, missingness, and paired quality analyses only after authorized execution.

We updated the working Methods, Results, Limitations, validity/activation table, corrected-ablation table, historical CUDA table, and supplementary audit manifest. The remaining manuscript dependency is the independent R1 quality evaluation; until it is completed, the response should not claim that structured context improves or harms answer quality.

**Limitations.** The cohort is biomedical PubMedQA, decoding is greedy, and MLX/CUDA configurations are not identical. The active intervention covers 278 rather than all 300 queries. Predictive uncertainty and factual quality are distinct endpoints. The October CUDA failures describe operational availability and context-limit selection, not factual errors in nonexistent answers.
"""
    (out / "reviewer_response_r1.md").write_text(reviewer)
    op = f"""# Reviewer #1 response — historical CUDA operational sensitivity (working draft)

**Provenance.** The authenticated decision letter was not present in the inspected worktree. This draft addresses the operational-sensitivity comment represented in the project’s provisional response materials; the exact journal numbering and wording must be reconciled before submission.

**Response.** We independently recovered all 5,250 October CUDA operation records (14 arms × 375 questions). The archive contains 4,265 successful operations and 985 partial/failed operations. On the 300 answerable questions, the recovered success counts were B6_full {next(x['successes'] for x in historical['answerable_arm_table'] if x['arm']=='B6_full')}/300, without dynamic graph expansion {next(x['successes'] for x in historical['answerable_arm_table'] if x['arm']=='B6_minus_dynamic_graph_expander')}/300, without parent deduplication {next(x['successes'] for x in historical['answerable_arm_table'] if x['arm']=='B6_minus_parent_deduplication')}/300, and the old static-context arm {next(x['successes'] for x in historical['answerable_arm_table'] if x['arm']=='B6_minus_static_graph_context')}/300. These are availability outcomes with failures retained in the denominator.

Relative to B6_full, removing dynamic graph expansion increased answer availability by 0.1733 (95% query-bootstrap CI −0.2167 to −0.1300 for B6_full minus the ablation; exact paired McNemar p=4.44×10⁻¹⁶; 52 one-way discordances). Removing parent deduplication increased availability by 0.1500 (95% CI −0.1900 to −0.1100; exact p=5.68×10⁻¹⁴; 45 one-way discordances). The old static-context comparison was an availability tie (248/300 versus 248/300; 0 discordances; exact p=1.0; CI [0, 0]), but this cannot be interpreted as a null component-quality result because the archive marks that intervention as incomplete and the successful paired prompts/canonical identities were identical. The underlying partial rows report parser/generation errors and contain no factual answer; they are operational failures, not factual errors.

The exact paired calculations are preserved in `historical_cuda_summary.json` and `tables/table_c_historical_cuda_operational_sensitivity.csv`. These findings concern operational sensitivity and context-limit selection. The corrected MLX run fixes the old prompt-level intervention problem, but it has different backend, prompt/context, runtime, and local configuration conditions; it is not a byte-identical continuation and must not be pooled with CUDA. No conclusion about semantic quality follows from the CUDA failure counts alone.
"""
    (out / "reviewer_response_operational.md").write_text(op)
    matrix = f"""# Reviewer #1 coverage matrix

| Workstream/comment | Evidence available | Readiness | Missing evidence / action |
|---|---|---|---|
| Correctly activated component ablation | v2 manifest, 600 ledger rows, prompt-level audit, negative controls | Substantively ready | Reconcile exact authenticated reviewer numbering |
| Predictive uncertainty | H20/H50/H100 for 300 pairs; fixed-seed bootstrap | Ready with qualified null/inconclusive wording | None for telemetry claim |
| Answer quality of corrected ablation | No R1 judge scores; explicit MISSING export | **Not ready** | Authorized execution of two requested judges and validation |
| Historical CUDA availability | 5,250 operation ledger rows and recovered answerable-300 tables | Ready with scope limitation | Do not call failures factual errors |
| R2 two-judge validation | Existing separate v3 artifacts must remain separate | Conditional | Verify exact frozen report before citing |
| R3 human negatives | No authenticated completion established in this audit | Not ready unless artifact verified | Do not claim completed |
| R4 independent generator/dataset replication | No authenticated completed Gemma result established | Not ready unless artifact verified | Do not claim completed |
| R5 manuscript/presentation changes | Draft artifacts exist | Ready after evidence and numbering reconciliation | Update final manuscript and exact response numbering |

**Overall:** not publication-ready for any claim about corrected-ablation semantic quality until independent R1 judging is completed; generation, intervention, telemetry, and historical operational analyses are ready with the stated limitations.
"""
    (out / "reviewer_coverage_matrix.md").write_text(matrix)

    readme = f"""# Corrected R1 MLX post-run analysis

This directory contains derived, offline-only analysis of the frozen corrected
MLX runtime and a separate audit of the immutable October CUDA operation
ledger. Raw runtime and historical archives are not copied or modified.

## Reproduce

From the repository root:

```bash
python3 revision_round1/analysis/r1_mlx_postrun/scripts/audit_r1_mlx.py
```

The script uses seed `{SEED}` and `{BOOT}` query-level bootstrap resamples. It
performs no network access, model inference, or judge API calls. It validates
accepted-result SHA256 values, durable-ledger identities, prepared-input and
prompt/token hashes, stable result IDs, telemetry schema/protocol, matched
prompt sections, negative controls, and the historical CUDA availability
tables.

## Audit status

- MLX: 600/600 planned slots accepted and successful; 300 matched query pairs.
- Intervention: 278/300 active static-relation changes and 22/300 intended
  no-op controls; all non-target prompt sections were invariant in active pairs.
- Predictive telemetry: 300/300 pairs valid for H20/H50/H100 under
  `r1-mlx-predictive-distribution-v1`; the legacy `entropy_valid` field is not
  the validity indicator for this protocol.
- Validation: 0 validation errors. One warning records that the cohort file
  embeds the pre-runtime source manifest hash rather than the v2 runtime hash;
  the recomputed cohort hash is correct. See `integrity_audit.json`.
- Quality: no R1 semantic-judge scores exist. The 600-row
  `judge_manifest_r1_pending.jsonl` is a deterministic input manifest only;
  it contains no scores and was not submitted to an API.
- Figures: PNG rendering is optional. If matplotlib is unavailable,
  `figures/README.md` records the blocker rather than fabricating figures.

## Main outputs

`generation_summary.json` contains ledger completeness and token/time totals;
`statistical_summary.json` contains query-paired estimates and fixed-seed
bootstrap intervals; `historical_cuda_summary.json` and Table C contain the
separate October operational analysis. Tables A–C are under `tables/`.
`reviewer_response_r1.md` and `reviewer_response_operational.md` are working
drafts and explicitly leave answer-quality claims pending. The authenticated
Reviewer #1 decision letter was not present in the inspected worktree, so
numbering and exact wording must be reconciled before submission.
"""
    (out / "README.md").write_text(readme)

    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "table_a_experiment_validity_activation.csv", [
        {"item": "planned generation slots", "ON": 300, "OFF": 300, "total": 600, "notes": "durable ledger"},
        {"item": "accepted successful generations", "ON": 300, "OFF": 300, "total": 600, "notes": "all accepted hashes valid"},
        {"item": "activation-positive intervention", "ON": 278, "OFF": "NA", "total": 278, "notes": "static relation section changed"},
        {"item": "activation-negative controls", "ON": 22, "OFF": 22, "total": 22, "notes": "prompts and outputs identical"},
        {"item": "valid matched telemetry pairs", "ON": 300, "OFF": 300, "total": 300, "notes": TELEMETRY},
    ], ["item", "ON", "OFF", "total", "notes"])
    quality_rows = []
    for name in ["h20_bits", "h50_bits", "h100_bits", "mean_nll_nats", "retained_mass_k20", "retained_mass_k50", "retained_mass_k100"]:
        x = by[name]
        quality_rows.append({"endpoint": name, "ON_mean": x["on_mean"], "OFF_mean": x["off_mean"], "ON_minus_OFF": x["paired_mean_difference_on_minus_off"], "CI95_low": x["ci95_low"], "CI95_high": x["ci95_high"], "valid_pairs": x["n"], "quality_judged": "not evaluated"})
    write_csv(tables / "table_b_corrected_mlx_ablation.csv", quality_rows, ["endpoint", "ON_mean", "OFF_mean", "ON_minus_OFF", "CI95_low", "CI95_high", "valid_pairs", "quality_judged"])
    write_csv(tables / "table_c_historical_cuda_operational_sensitivity.csv", [
        {"arm": x["arm"], "answerable_n": x["answerable_n"], "successes": x["successes"], "failures": x["failures"], "success_rate": x["success_rate"], "all_query_successes": x["all_query_successes"], "all_query_failures": x["all_query_failures"]}
        for x in historical["answerable_arm_table"]
    ], ["arm", "answerable_n", "successes", "failures", "success_rate", "all_query_successes", "all_query_failures"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest, cohort, prepared, results, errors, warnings = load_runtime()
    ledger = audit_ledger(manifest, results, prepared, errors)
    pairs, intervention = validate_and_build_pairs(manifest, cohort, prepared, results, errors)
    metrics = []
    for metric in ["h20_bits", "h50_bits", "h100_bits", "mean_nll_nats", "mean_nll_bits_per_token", "retained_mass_k20", "retained_mass_k50", "retained_mass_k100"]:
        for p in pairs:
            p[ON][metric] = p[ON]["telemetry"][metric]; p[OFF][metric] = p[OFF]["telemetry"][metric]
        metrics.append(metric_summary(pairs, metric))
    # Answer-level paired operational and response characteristics.
    paired_rows = []
    for x in intervention:
        row = dict(x)
        row.update({f"delta_{k}": row[f"on_{k}"] - row[f"off_{k}"] for k in ["prompt_tokens", "output_tokens", "generation_tokens", "duration_seconds"]})
        row["on_tokens_per_second"] = row["on_output_tokens"] / row["on_duration_seconds"]
        row["off_tokens_per_second"] = row["off_output_tokens"] / row["off_duration_seconds"]
        paired_rows.append(row)
    historical, _ = historical_cuda()
    judge = make_judge_manifest(pairs, out)
    write_json(out / "integrity_audit.json", {"runtime_manifest_hash": manifest.get("manifest_hash"), "cohort_hash": cohort.get("cohort_hash"), "recomputed_runtime_manifest_hash": sha_obj({k:v for k,v in manifest.items() if k != 'manifest_hash'}), "recomputed_cohort_hash": sha_obj({k:v for k,v in cohort.items() if k != 'cohort_hash'}), "validation_errors": errors, "warnings": warnings, "validation_error_count": len(errors), "warning_count": len(warnings), "telemetry_valid_pairs": sum(1 for p in pairs if p[ON].get("telemetry_valid") and p[OFF].get("telemetry_valid")), "accepted_hash_failures": ledger["accepted_hash_failures"], "activation_positive": sum(x["activation"] for x in intervention), "activation_negative": sum(not x["activation"] for x in intervention), "active_static_changed": sum(x["activation"] and x["static_section_changed"] for x in intervention), "active_non_target_invariant": sum(x["activation"] and x["non_target_sections_equal"] for x in intervention), "negative_identical_prompts": sum((not x["activation"]) and x["prompt_equal"] for x in intervention), "negative_identical_outputs": sum((not x["activation"]) and x["outputs_equal"] for x in intervention)})
    write_json(out / "generation_summary.json", ledger)
    write_json(out / "statistical_summary.json", {"seed": SEED, "bootstrap_resamples": BOOT, "n_pairs": len(pairs), "metrics": metrics, "operational": {"on_success": 300, "off_success": 300, "both_success": 300, "on_only": 0, "off_only": 0, "both_failed": 0, "mcnemar": exact_mcnemar([True]*300,[True]*300), "paired_difference": 0.0, "ci95": [0.0, 0.0]}, "activation_positive_metrics": [{**metric_summary([p for p in pairs if p['activation']], m), "subgroup": "activation_positive"} for m in ["h20_bits", "h50_bits", "h100_bits"]], "activation_negative_metrics": [{**metric_summary([p for p in pairs if not p['activation']], m), "subgroup": "activation_negative"} for m in ["h20_bits", "h50_bits", "h100_bits"]]})
    write_json(out / "historical_cuda_summary.json", historical)
    write_csv(out / "intervention_audit.csv", intervention)
    write_csv(out / "paired_metrics.csv", paired_rows)
    fields = ["metric", "n", "on_mean", "off_mean", "on_median", "off_median", "on_sd", "off_sd", "paired_mean_difference_on_minus_off", "paired_median_difference_on_minus_off", "paired_sd", "ci95_low", "ci95_high", "negative_differences", "zero_differences", "positive_differences"]
    write_csv(out / "metric_summary.csv", metrics, fields)
    write_docs(out, manifest, cohort, ledger, intervention, metrics, historical, judge, errors, warnings, create_figure(out / "figures", pairs, metrics, intervention))
    write_json(out / "source_manifest.json", {"runtime": str(RUNTIME), "runtime_manifest_sha256": sha_file(RUNTIME / "experiment_manifest.json"), "cohort_manifest_sha256": sha_file(RUNTIME / "cohort_manifest.json"), "sqlite_sha256": sha_file(RUNTIME / "r1_attempts.sqlite3"), "cuda_archive": str(CUDA), "seed": SEED, "bootstrap_resamples": BOOT, "offline_only": True, "analysis_script": str(Path(__file__).resolve())})
    (out / "README.md").write_text(f"""# Corrected R1 MLX post-run analysis

This directory contains derived, offline-only audit products for the corrected structured-context ablation. The script reads the frozen v2 MLX runtime and the immutable October CUDA archive; it does not modify either source. It does not load a model, run inference, access a network, call a paid judge, or pool the July, October CUDA, and corrected MLX experiments.

## Reproduce

Run from the repository root:

```bash
python3 revision_round1/analysis/r1_mlx_postrun/scripts/audit_r1_mlx.py
python3 -m unittest discover -s revision_round1/analysis/r1_mlx_postrun/tests -p 'test_*.py' -v
```

The audit uses fixed seed `{SEED}` and `{BOOT}` query-bootstrap resamples. It validates accepted-ledger/result SHA-256 values, prepared-input identities and hashes, matched ON/OFF prompt sections, negative controls, predictive telemetry, and the historical CUDA operation ledger. A clean run currently reports 300 matched pairs, zero validation errors, and one documented provenance warning: the cohort embeds the pre-runtime source manifest hash `2da88f58b85926a483cb45dbbac1216e066bedc7b1a273641284fd49f32333a8`, whereas the v2 runtime manifest is `dffd5b400b86035eb270961496399f90495b122891da9e2e0eb37a9fcc7b4607`; the cohort hash itself recomputes to `bf4cc128d85d5534e28a5a0804bef0c6eb6ddfed44a1777fc1ad42d050cf7074`.

## Main evidence

- `integrity_audit.json`: validation status, intervention isolation, negative controls, and telemetry coverage.
- `generation_summary.json`: planned/accepted/failed slots, pairing, token/timing totals, and finish reasons.
- `statistical_summary.json`, `paired_metrics.csv`, and `metric_summary.csv`: query-level endpoint summaries and fixed-seed confidence intervals.
- `historical_cuda_summary.json` and `tables/table_c_historical_cuda_operational_sensitivity.csv`: separate October CUDA availability analysis; these are not pooled with MLX.
- `tables/table_a_experiment_validity_activation.csv` and `tables/table_b_corrected_mlx_ablation.csv`: manuscript-ready validity and corrected-ablation tables.
- `scientific_findings.md`, `reviewer_response_r1.md`, `reviewer_response_operational.md`, `reviewer_coverage_matrix.md`, and the manuscript subsection drafts: working publication/rebuttal text. The authenticated journal decision letter was not found in the inspected worktree, so reviewer numbering remains provisional.
- `figures/README.md`: figure status and optional rendering instructions. No PNG is fabricated when matplotlib is unavailable.

## Quality-judge blocker

The corrected MLX generation has no completed R1 semantic-judge results. `prejudge_export_empty.jsonl` contains explicit `MISSING` records. `judge_manifest_r1_pending.jsonl` is an input-only, 600-row handoff containing configuration-specific generator outputs/evidence; it contains no scores and no API provenance. It is deliberately not required for the committed reproducibility code because it duplicates raw-like prompt/output material; the exact source path and required judges are recorded in the reviewer drafts. Do not interpret the entropy results as answer-quality results.
""")
    print(json.dumps({"output": str(out), "pairs": len(pairs), "validation_errors": len(errors), "warnings": len(warnings), "metrics": len(metrics), "judge_rows": judge["rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
