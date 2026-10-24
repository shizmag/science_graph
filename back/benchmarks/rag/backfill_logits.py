#!/usr/bin/env python3
"""Teacher-force logit backfill for an existing RAG benchmark run.

Re-scores existing generated_answer texts under the same prompts used by the
consume-contexts generation path (retrieved_contexts + PromptManager), without
re-running retrieval or LLM-as-judge.

Updates:
  - evaluation_results.yaml  (tokens_info + shannon_diagnostics telemetry)
  - result_metrics.yaml      (shannon_diagnostics / telemetry scalars only)
  - raw_logits.yaml          (compact tokens_info per case/baseline)
  - parse_metrics / stats    (optional, via --parse)

Never touches evaluation_results_judge.yaml or result_metrics_judge.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import yaml

# Project imports: script lives at back/benchmarks/rag/
SCRIPT_DIR = Path(__file__).resolve().parent
BACK_DIR = SCRIPT_DIR.parents[1]  # .../back
if str(BACK_DIR) not in sys.path:
    sys.path.insert(0, str(BACK_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from core.reporting import save_raw_logits_yaml  # noqa: E402
from core.shannon_estimator import (  # noqa: E402
    compute_citation_entropy,
    compute_clr,
    compute_entropy_reduction,
    compute_generation_entropy,
    compute_log_likelihood,
    compute_sequence_telemetry,
)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dump_yaml_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}_{time.time_ns()}")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    tmp.replace(path)


def _index_by_id(results: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in results or []:
        rid = r.get("id")
        if rid is not None:
            out[str(rid)] = r
    return out


def build_generation_prompt(
    prompts: Any,
    query: str,
    baseline: str,
    pre_baseline: dict,
) -> str:
    """Mirror consume_contexts prompt construction from core/generation.py."""
    if baseline == "B0":
        return f"Question: {query}\nAnswer based on your general knowledge."

    enrichment_block = pre_baseline.get("enrichment_block") or ""
    trimmed_text = pre_baseline.get("trimmed_text") or ""
    trimmed_graph = pre_baseline.get("trimmed_graph") or ""

    if enrichment_block and enrichment_block != "No essential knowledge graph enrichment found.":
        return prompts.get_prompt(
            "rag",
            "ask_expander",
            enrichment_block=enrichment_block,
            history_str="",
            query=query,
        )
    return prompts.get_prompt(
        "rag",
        "ask_no_expander",
        context_text=trimmed_text,
        context_graph=trimmed_graph,
        history_str="",
        query=query,
    )


def compact_tokens_for_eval(tokens_info: List[Dict[str, Any]], drop_top: bool = True) -> List[Dict[str, Any]]:
    """Optionally strip top_logprobs to keep evaluation_results.yaml smaller."""
    if not drop_top:
        return tokens_info
    out = []
    for t in tokens_info:
        if not isinstance(t, dict):
            continue
        item = {
            k: t[k]
            for k in (
                "token_id",
                "token_text",
                "token",
                "char_start",
                "char_end",
                "logprob",
                "entropy",
                "msp",
                "logit_margin",
            )
            if k in t
        }
        out.append(item)
    return out


def compute_telemetry_for_answer(
    engine: Any,
    prompt: str,
    answer_text: str,
    query: str,
    baseline: str,
    existing_shannon: Optional[dict] = None,
    h_b0: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Teacher-force score answer under RAG prompt + base prompt; return tokens_info and shannon fields."""
    existing_shannon = dict(existing_shannon or {})

    if not answer_text or not str(answer_text).strip():
        return [], existing_shannon

    score_fn = getattr(engine, "score_text_logprobs", None)
    if not callable(score_fn):
        raise RuntimeError(
            f"LLM engine {type(engine).__name__} has no score_text_logprobs; "
            "teacher-force backfill requires MLX (or compatible) scorer."
        )

    scored_tokens: Any = score_fn(prompt, answer_text)
    if not isinstance(scored_tokens, list):
        scored_tokens = []
    tokens_info: List[Dict[str, Any]] = cast(
        List[Dict[str, Any]], [token for token in scored_tokens if isinstance(token, dict)]
    )
    h_gen = float(compute_generation_entropy(tokens_info))
    h_cit, n_cit = compute_citation_entropy(tokens_info, answer_text)
    ll_rag = float(compute_log_likelihood(tokens_info))

    if baseline == "B0":
        ll_base = ll_rag
        clr = 0.0
    else:
        b0_prompt = f"Question: {query}\nAnswer based on your general knowledge."
        scored_base_tokens: Any = score_fn(b0_prompt, answer_text)
        if not isinstance(scored_base_tokens, list):
            scored_base_tokens = []
        base_tokens: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], [token for token in scored_base_tokens if isinstance(token, dict)]
        )
        ll_base = float(compute_log_likelihood(base_tokens))
        clr = float(compute_clr(ll_rag, ll_base))

    if h_b0 is None and existing_shannon.get("h_gen") is not None and existing_shannon.get("delta_h_gen") is not None:
        # Recover h_b0 ≈ h_gen + delta_h_gen from previously stored values
        try:
            h_b0 = float(existing_shannon["h_gen"]) + float(existing_shannon["delta_h_gen"])
        except (TypeError, ValueError):
            h_b0 = None
    delta_h = float(compute_entropy_reduction(h_b0, h_gen)) if h_b0 is not None else float(
        existing_shannon.get("delta_h_gen") or 0.0
    )

    seq_tel = compute_sequence_telemetry(tokens_info)
    shannon = {
        **existing_shannon,
        "h_gen": round(h_gen, 4),
        "h_citation": round(h_cit, 4),
        "n_citation_tokens": int(n_cit),
        "delta_h_gen": round(delta_h, 4),
        "ll_rag": round(ll_rag, 4),
        "ll_base": round(ll_base, 4),
        "clr": round(clr, 4),
        **seq_tel,
        "msp": seq_tel.get("avg_msp", 0.0),
        "logit_margin": seq_tel.get("avg_logit_margin", 0.0),
    }
    return tokens_info, shannon


def _get_existing_shannon(b_data: dict) -> dict:
    diag = b_data.get("shannon_diagnostics")
    if isinstance(diag, dict) and diag:
        return dict(diag)
    metrics = b_data.get("metrics")
    if isinstance(metrics, dict):
        diag = metrics.get("shannon_diagnostics")
        if isinstance(diag, dict) and diag:
            return dict(diag)
    return {}


def _set_shannon(b_data: dict, shannon: dict) -> None:
    b_data["shannon_diagnostics"] = shannon
    metrics = b_data.get("metrics")
    if isinstance(metrics, dict):
        metrics["shannon_diagnostics"] = shannon
    # Promote scalar telemetry for CSV/stats extractors
    for k in (
        "h_gen",
        "h_citation",
        "n_citation_tokens",
        "delta_h_gen",
        "avg_msp",
        "avg_logit_margin",
        "first_token_msp",
        "first_token_margin",
        "msp",
        "logit_margin",
        "ll_rag",
        "ll_base",
        "clr",
    ):
        if k in shannon and shannon[k] is not None:
            b_data[k] = shannon[k]


def backfill_run(
    run_dir: Path,
    *,
    model_path: Optional[str] = None,
    limit: Optional[int] = None,
    baselines_filter: Optional[List[str]] = None,
    resume: bool = True,
    backup: bool = True,
    compact_eval: bool = True,
    parse_after: bool = True,
    dry_run: bool = False,
) -> dict:
    run_dir = run_dir.resolve()
    eval_path = run_dir / "evaluation_results.yaml"
    metrics_path = run_dir / "result_metrics.yaml"
    contexts_path = run_dir / "retrieved_contexts.yaml"
    config_path = run_dir / "config_snapshot.yaml"
    progress_path = run_dir / ".backfill_logits_progress.json"
    logits_jsonl_path = run_dir / "raw_logits.jsonl"

    for p in (eval_path, contexts_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    print(f"[backfill] run_dir={run_dir}")
    eval_data = _load_yaml(eval_path)
    contexts_raw = _load_yaml(contexts_path)
    metrics_data = _load_yaml(metrics_path) if metrics_path.exists() else None
    cfg_snap = _load_yaml(config_path) if config_path.exists() else {}

    # evaluation_results: {metadata, results:[...]} ; retrieved_contexts may be bare list or {results:[...]}
    if isinstance(eval_data, list):
        eval_data = {"metadata": {}, "results": eval_data}
    results = eval_data.get("results") or []
    if isinstance(contexts_raw, list):
        ctx_results = contexts_raw
    elif isinstance(contexts_raw, dict):
        ctx_results = contexts_raw.get("results") or []
    else:
        ctx_results = []
    ctx_by_id = _index_by_id(ctx_results)
    metrics_by_id = _index_by_id((metrics_data or {}).get("results") or []) if metrics_data else {}

    if baselines_filter:
        baselines = baselines_filter
    else:
        meta = eval_data.get("metadata") or {}
        baselines = meta.get("baselines_evaluated") or list(
            (results[0].get("baselines") or {}).keys()
        ) if results else []

    # Resolve model path like previous run
    if not model_path:
        llm_meta = (eval_data.get("metadata") or {}).get("llm") or {}
        model_path = (
            llm_meta.get("model_name")
            or (cfg_snap.get("llm") or {}).get("local_model_path")
            or (cfg_snap.get("model") or {}).get("local_model_path")
        )
    if not model_path:
        raise RuntimeError("Could not resolve model path; pass --model-path")

    done_keys = set()
    if resume and progress_path.exists():
        try:
            done_keys = set(json.loads(progress_path.read_text(encoding="utf-8")).get("done", []))
            print(f"[backfill] resume: {len(done_keys)} pairs already done")
        except Exception:
            done_keys = set()

    if dry_run:
        n = 0
        for case in results[: (limit or len(results))]:
            for b in baselines:
                if b in (case.get("baselines") or {}):
                    n += 1
        print(f"[backfill] dry-run would score ~{n} (case, baseline) pairs")
        return {"dry_run": True, "pairs": n}

    # Backup once
    if backup:
        for p in (eval_path, metrics_path, run_dir / "raw_logits.yaml"):
            if p.exists():
                bak = p.with_suffix(p.suffix + ".pre_backfill.bak")
                if not bak.exists():
                    shutil.copy2(p, bak)
                    print(f"[backfill] backup -> {bak.name}")

    # Truncate JSONL only on fresh (non-resume) runs so resume can append
    if not resume or not done_keys:
        logits_jsonl_path.write_text("", encoding="utf-8")
    elif not logits_jsonl_path.exists():
        logits_jsonl_path.write_text("", encoding="utf-8")

    # Load prompt manager + MLX engine (same stack as generation pipeline)
    from src.prompts.manager import PromptManager
    from src.llm_engine.mlx_impl import MlxLLMEngine

    prompts = PromptManager()
    print(f"[backfill] loading MLX model: {model_path}")
    engine = MlxLLMEngine(model_path=model_path)
    engine._ensure_model_loaded()

    # Cache B0 H_gen per query for delta_h
    h_b0_cache: Dict[str, float] = {}

    total = 0
    for case in results:
        for b in baselines:
            if b in (case.get("baselines") or {}):
                total += 1
    if limit is not None:
        # limit is max number of cases (queries), not pairs
        cases_to_run = results[:limit]
    else:
        cases_to_run = results

    processed = 0
    skipped = 0
    errors = 0
    t0 = time.perf_counter()

    for case_idx, case in enumerate(cases_to_run):
        case_id = str(case.get("id", f"idx_{case_idx}"))
        query = case.get("query") or ""
        ctx_case = ctx_by_id.get(case_id, {})
        ctx_baselines = (ctx_case.get("baselines") or {}) if ctx_case else {}
        metrics_case = metrics_by_id.get(case_id)

        for baseline in baselines:
            key = f"{case_id}::{baseline}"
            b_data = (case.get("baselines") or {}).get(baseline)
            if not b_data:
                continue
            if key in done_keys:
                skipped += 1
                continue

            answer = b_data.get("generated_answer") or ""
            pre_baseline = ctx_baselines.get(baseline) or {}
            # Prefer pre-retrieved fields; fall back to evaluation payload if present
            for fld in (
                "trimmed_text",
                "trimmed_graph",
                "enrichment_block",
                "context_text",
                "context_graph",
                "pre_rerank_scores",
                "graph_relations",
            ):
                if not pre_baseline.get(fld) and b_data.get(fld) is not None:
                    pre_baseline[fld] = b_data.get(fld)

            try:
                prompt = build_generation_prompt(prompts, query, baseline, pre_baseline)
                existing = _get_existing_shannon(b_data)
                # Seed h_b0 from existing delta if possible once
                h_b0 = h_b0_cache.get(query)
                if h_b0 is None and existing.get("h_gen") is not None and existing.get("delta_h_gen") is not None:
                    try:
                        h_b0 = float(existing["h_gen"]) + float(existing["delta_h_gen"])
                        h_b0_cache[query] = h_b0
                    except (TypeError, ValueError):
                        h_b0 = None

                tokens_info, shannon = compute_telemetry_for_answer(
                    engine,
                    prompt,
                    answer,
                    query,
                    baseline,
                    existing_shannon=existing,
                    h_b0=h_b0,
                )
                if baseline == "B0" and tokens_info:
                    h_b0_cache[query] = float(shannon.get("h_gen") or 0.0)

                eval_tokens = compact_tokens_for_eval(tokens_info, drop_top=compact_eval)
                # Keep full token streams out of the huge evaluation YAML by default;
                # raw_logits.jsonl holds them. Set --full-eval-tokens to embed compact tokens.
                if compact_eval:
                    b_data["tokens_info"] = []
                    metrics = b_data.get("metrics")
                    if isinstance(metrics, dict) and "tokens_info" in metrics:
                        metrics.pop("tokens_info", None)
                else:
                    b_data["tokens_info"] = eval_tokens
                    metrics = b_data.get("metrics")
                    if isinstance(metrics, dict):
                        metrics["tokens_info"] = eval_tokens
                _set_shannon(b_data, shannon)

                # Mirror into result_metrics (judge eval_metrics untouched)
                if metrics_case is not None:
                    mb = (metrics_case.get("baselines") or {}).get(baseline)
                    if isinstance(mb, dict):
                        _set_shannon(mb, shannon)

                # Incremental raw logits (JSONL) — avoids rewriting multi‑MB YAML every N steps
                logits_record = {
                    "id": case_id,
                    "baseline": baseline,
                    "tokens_info": tokens_info,
                    "shannon_diagnostics": {
                        k: shannon.get(k)
                        for k in (
                            "h_gen",
                            "h_citation",
                            "n_citation_tokens",
                            "delta_h_gen",
                            "avg_msp",
                            "avg_logit_margin",
                            "first_token_msp",
                            "first_token_margin",
                            "ll_rag",
                            "ll_base",
                            "clr",
                        )
                    },
                }
                with open(logits_jsonl_path, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(logits_record, ensure_ascii=False) + "\n")

                processed += 1
                done_keys.add(key)
                progress_path.write_text(
                    json.dumps({"done": sorted(done_keys)}, ensure_ascii=False, indent=0),
                    encoding="utf-8",
                )
                if processed % 5 == 0 or processed == 1:
                    elapsed = time.perf_counter() - t0
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(
                        f"[backfill] {processed} scored | last={key} | "
                        f"tokens={len(tokens_info)} avg_msp={shannon.get('avg_msp')} "
                        f"ll_rag={shannon.get('ll_rag')} clr={shannon.get('clr')} "
                        f"({rate:.2f}/s)"
                    )
                # Periodic checkpoint of evaluation/metrics YAMLs (scalars only when compact)
                if processed % 50 == 0:
                    _dump_yaml_atomic(eval_path, eval_data)
                    if metrics_data is not None:
                        _dump_yaml_atomic(metrics_path, metrics_data)
                    print(f"[backfill] checkpoint YAML @ {processed} pairs")

            except Exception as e:
                errors += 1
                print(f"[backfill] ERROR {key}: {e}")
                continue

    # Final save
    _dump_yaml_atomic(eval_path, eval_data)
    if metrics_data is not None:
        _dump_yaml_atomic(metrics_path, metrics_data)
    # Compatibility raw_logits.yaml from in-memory results (may be empty tokens if compact)
    try:
        save_raw_logits_yaml(eval_path, eval_data.get("results") or [], eval_data.get("metadata") or {})
    except Exception as ex:
        print(f"[backfill] raw_logits.yaml final save warning: {ex}")
    # Prefer JSONL as the source of truth for full tokens_info
    print(f"[backfill] full tokens_info stream: {logits_jsonl_path}")
    progress_path.write_text(
        json.dumps({"done": sorted(done_keys), "finished": True}, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )

    summary = {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "run_dir": str(run_dir),
    }
    print(f"[backfill] done: {summary}")

    if parse_after and not dry_run:
        try:
            from parse_metrics import MetricsParser

            parser = MetricsParser(
                run_dir,
                enable_stats=True,
                n_bootstraps=2000,
                random_seed=42,
            )
            parser.run()
            print("[backfill] parse_metrics + stats refreshed")
        except Exception as e:
            print(f"[backfill] parse_metrics failed (telemetry still written): {e}")
            print(f"  You can re-run: python parse_metrics.py {run_dir}")

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Teacher-force logit backfill for a finished RAG run (no judge).")
    ap.add_argument(
        "run_dir",
        type=str,
        nargs="?",
        default=str(Path("/Users/vladimirkasterin/python/graph/graphs/run_20260725_194320_OCC-RAG-1.7B")),
        help="Path to run directory (default: the OCC-RAG-1.7B run with empty logits).",
    )
    ap.add_argument("--model-path", type=str, default=None, help="Override MLX model path.")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N queries (smoke).")
    ap.add_argument("--baselines", type=str, default=None, help="Comma-separated baselines (default: all in run).")
    ap.add_argument("--no-resume", action="store_true", help="Ignore progress file and re-score all.")
    ap.add_argument("--no-backup", action="store_true", help="Skip .pre_backfill.bak copies.")
    ap.add_argument("--full-eval-tokens", action="store_true", help="Keep top_logprobs inside evaluation_results.yaml.")
    ap.add_argument("--no-parse", action="store_true", help="Skip parse_metrics / stats refresh.")
    ap.add_argument("--dry-run", action="store_true", help="Count pairs without loading the model.")
    args = ap.parse_args(argv)

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()] if args.baselines else None
    backfill_run(
        Path(args.run_dir),
        model_path=args.model_path,
        limit=args.limit,
        baselines_filter=baselines,
        resume=not args.no_resume,
        backup=not args.no_backup,
        compact_eval=not args.full_eval_tokens,
        parse_after=not args.no_parse,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
