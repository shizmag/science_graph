"""
Science Graph — RAG Trace Parsing & Loading Utilities.
Provides robust functions to parse, transform, and aggregate graph and evaluation trace logs.
"""

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_graph_retrieval_trace(trace_path: Path) -> Dict[Tuple[str, str], dict]:
    """
    Robustly loads graph_retrieval_trace.jsonl.
    Returns a dictionary mapping (baseline, query_id) -> trace_entry.
    Falls back to (baseline, query) if query_id is missing.
    In case of duplicate keys, keeps the latest entry.
    """
    trace_map = {}
    if not trace_path.exists():
        return trace_map

    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception as e:
                    print(f"Warning: Invalid JSON in {trace_path} at line {line_idx}: {e}")
                    continue

                baseline = entry.get("baseline", "B6")
                query_id = entry.get("query_id")
                query = entry.get("query")

                if query_id:
                    key = (baseline, str(query_id))
                elif query:
                    key = (baseline, str(query))
                else:
                    continue

                trace_map[key] = entry
    except Exception as e:
        print(f"Error loading graph retrieval trace: {e}")

    return trace_map


def parse_graph_retrieval_trace(
    traces_dir: Path, parsed_dir: Path
) -> Tuple[Optional[List[dict]], Optional[dict]]:
    """
    Parses graph_retrieval_trace.jsonl into a structured CSV and summary JSON.

    Args:
        traces_dir: Path to directory containing trace logs.
        parsed_dir: Path to directory where parsed outputs should be written.

    Returns:
        Tuple of (list of parsed CSV row dicts, summary dictionary), or (None, None) if not found.
    """
    trace_file = traces_dir / "graph_retrieval_trace.jsonl"
    if not trace_file.exists():
        trace_file = traces_dir.parent / "graph_retrieval_trace.jsonl"
    if not trace_file.exists():
        print(f"Warning: graph_retrieval_trace.jsonl not found in {traces_dir} or {traces_dir.parent}")
        return None, None

    rows = []
    try:
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as e:
        print(f"Error reading graph_retrieval_trace.jsonl: {e}")
        return None, None

    if not rows:
        return None, None

    parsed_dir.mkdir(parents=True, exist_ok=True)
    csv_file = parsed_dir / "graph_retrieval_trace.parsed.csv"
    headers = [
        "query_id", "baseline", "category", "query",
        "graph_retrieval_enabled", "graph_retrieval_skip_reason",
        "base_candidates_count", "graph_neighbor_paper_ids_count", "graph_chunk_candidates_count",
        "merged_candidates_count_before_reranker", "reranker_input_count_before_limit",
        "reranker_input_count_after_limit", "candidate_count_after_reranker",
        "graph_chunks_survived_final_context_count", "graph_chunks_survived_final_context",
        "graph_survival_rate", "distinct_papers_in_final_context",
        "base_candidate_paper_ids_count", "graph_chunk_candidate_paper_ids_count",
        "final_context_paper_ids_count"
    ]

    csv_rows = []
    for r in rows:
        base_cand_papers = r.get("base_candidate_paper_ids", [])
        graph_chunk_cand_papers = r.get("graph_chunk_candidate_paper_ids", [])
        final_context_papers = r.get("final_context_paper_ids", [])

        base_candidate_paper_ids_count = len(base_cand_papers) if isinstance(base_cand_papers, list) else 0
        graph_chunk_candidate_paper_ids_count = len(graph_chunk_cand_papers) if isinstance(graph_chunk_cand_papers, list) else 0
        final_context_paper_ids_count = len(final_context_papers) if isinstance(final_context_papers, list) else 0

        survived_context = r.get("graph_chunks_survived_final_context", [])
        survived_str = json.dumps(survived_context) if isinstance(survived_context, list) else str(survived_context)

        csv_rows.append({
            "query_id": r.get("query_id") or (r.get("query_concepts", [""])[0] if r.get("query_concepts") else "UNKNOWN"),
            "baseline": r.get("baseline", "B6"),
            "category": r.get("category", "general"),
            "query": r.get("query", ""),
            "graph_retrieval_enabled": r.get("graph_retrieval_enabled", True),
            "graph_retrieval_skip_reason": r.get("graph_retrieval_skip_reason") or "",
            "base_candidates_count": r.get("base_candidates_count", 0),
            "graph_neighbor_paper_ids_count": r.get("graph_neighbor_paper_ids_count", 0),
            "graph_chunk_candidates_count": r.get("graph_chunk_candidates_count", 0),
            "merged_candidates_count_before_reranker": r.get("merged_candidates_count_before_reranker", 0),
            "reranker_input_count_before_limit": r.get("reranker_input_count_before_limit", 0),
            "reranker_input_count_after_limit": r.get("reranker_input_count_after_limit", 0),
            "candidate_count_after_reranker": r.get("candidate_count_after_reranker", 0),
            "graph_chunks_survived_final_context_count": r.get("graph_chunks_survived_final_context_count", 0),
            "graph_chunks_survived_final_context": survived_str,
            "graph_survival_rate": r.get("graph_survival_rate", 0.0),
            "distinct_papers_in_final_context": r.get("distinct_papers_in_final_context", 0),
            "base_candidate_paper_ids_count": base_candidate_paper_ids_count,
            "graph_chunk_candidate_paper_ids_count": graph_chunk_candidate_paper_ids_count,
            "final_context_paper_ids_count": final_context_paper_ids_count
        })

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for crow in csv_rows:
            writer.writerow(crow)

    total_queries = len(csv_rows)
    enabled_count = sum(1 for r in csv_rows if r["graph_retrieval_enabled"])
    queries_with_neighbors = sum(1 for r in csv_rows if r["graph_neighbor_paper_ids_count"] > 0)
    queries_with_chunks = sum(1 for r in csv_rows if r["graph_chunk_candidates_count"] > 0)
    queries_with_survival = sum(1 for r in csv_rows if r["graph_chunks_survived_final_context_count"] > 0)

    avg_base_candidates = statistics.mean([r["base_candidates_count"] for r in csv_rows]) if csv_rows else 0.0
    avg_graph_neighbors = statistics.mean([r["graph_neighbor_paper_ids_count"] for r in csv_rows]) if csv_rows else 0.0
    avg_graph_chunks = statistics.mean([r["graph_chunk_candidates_count"] for r in csv_rows]) if csv_rows else 0.0
    avg_merged_before = statistics.mean([r["merged_candidates_count_before_reranker"] for r in csv_rows]) if csv_rows else 0.0
    avg_survival_rate = statistics.mean([r["graph_survival_rate"] for r in csv_rows]) if csv_rows else 0.0
    avg_distinct_papers = statistics.mean([r["distinct_papers_in_final_context"] for r in csv_rows]) if csv_rows else 0.0

    skip_reasons = {}
    for r in csv_rows:
        reason = r["graph_retrieval_skip_reason"]
        if not r["graph_retrieval_enabled"]:
            reason = "disabled"
        if not reason:
            continue
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    by_category = {}
    categories = set(r["category"] for r in csv_rows)
    for cat in categories:
        cat_rows = [r for r in csv_rows if r["category"] == cat]
        by_category[cat] = {
            "queries": len(cat_rows),
            "queries_with_graph_survival": sum(1 for r in cat_rows if r["graph_chunks_survived_final_context_count"] > 0),
            "avg_graph_chunk_candidates_count": round(statistics.mean([r["graph_chunk_candidates_count"] for r in cat_rows]), 2) if cat_rows else 0.0,
            "avg_graph_survival_rate": round(statistics.mean([r["graph_survival_rate"] for r in cat_rows]), 4) if cat_rows else 0.0
        }

    summary_data = {
        "total_queries": total_queries,
        "queries_with_graph_retrieval_enabled": enabled_count,
        "queries_with_graph_neighbors": queries_with_neighbors,
        "queries_with_graph_chunks": queries_with_chunks,
        "queries_with_graph_survival": queries_with_survival,
        "avg_base_candidates_count": round(avg_base_candidates, 2),
        "avg_graph_neighbor_paper_ids_count": round(avg_graph_neighbors, 2),
        "avg_graph_chunk_candidates_count": round(avg_graph_chunks, 2),
        "avg_merged_candidates_count_before_reranker": round(avg_merged_before, 2),
        "avg_graph_survival_rate": round(avg_survival_rate, 4),
        "avg_distinct_papers_in_final_context": round(avg_distinct_papers, 2),
        "by_category": by_category,
        "skip_reasons": skip_reasons
    }

    with open(parsed_dir / "graph_retrieval_trace.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    return csv_rows, summary_data


def parse_eval_trace(
    traces_dir: Path, parsed_dir: Path
) -> Tuple[Optional[List[dict]], Optional[dict]]:
    """
    Parses eval_trace.jsonl into a structured CSV and summary JSON.

    Args:
        traces_dir: Path to directory containing trace logs.
        parsed_dir: Path to directory where parsed outputs should be written.

    Returns:
        Tuple of (list of parsed CSV row dicts, summary dictionary), or (None, None) if not found.
    """
    trace_file = traces_dir / "eval_trace.jsonl"
    if not trace_file.exists():
        trace_file = traces_dir.parent / "eval_trace.jsonl"
    if not trace_file.exists():
        print(f"Warning: eval_trace.jsonl not found in {traces_dir} or {traces_dir.parent}")
        return None, None

    rows = []
    try:
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as e:
        print(f"Error reading eval_trace.jsonl: {e}")
        return None, None

    if not rows:
        return None, None

    parsed_dir.mkdir(parents=True, exist_ok=True)
    csv_file = parsed_dir / "eval_trace.parsed.csv"
    headers = [
        "query_id", "baseline", "category", "judge_model", "latency_sec",
        "retrieval_recall", "context_precision", "faithfulness",
        "answer_relevance", "citation_fidelity", "semantic_accuracy", "context_fillness",
        "ar_sa_f1", "is_answerable"
    ]

    csv_rows = []
    for r in rows:
        is_ans = r.get("is_answerable")
        if is_ans is None:
            is_ans = True
        else:
            is_ans = str(is_ans).lower() == "true"

        ar_f1 = r.get("ar_sa_f1")
        if ar_f1 is None and is_ans:
            r_relevance = r.get("answer_relevance")
            s_accuracy = r.get("semantic_accuracy")
            if r_relevance is not None and s_accuracy is not None:
                try:
                    r_val = float(r_relevance)
                    s_val = float(s_accuracy)
                    if r_val + s_val > 0:
                        ar_f1 = round(2.0 * (r_val * s_val) / (r_val + s_val), 4)
                    else:
                        ar_f1 = 0.0
                except (ValueError, TypeError):
                    ar_f1 = 0.0

        csv_rows.append({
            "query_id": r.get("query_id") or r.get("id"),
            "baseline": r.get("baseline"),
            "category": r.get("category", "general"),
            "judge_model": r.get("judge_model", ""),
            "latency_sec": r.get("latency_sec"),
            "retrieval_recall": r.get("retrieval_recall"),
            "context_precision": r.get("context_precision"),
            "faithfulness": r.get("faithfulness"),
            "answer_relevance": r.get("answer_relevance"),
            "citation_fidelity": r.get("citation_fidelity"),
            "semantic_accuracy": r.get("semantic_accuracy"),
            "context_fillness": r.get("context_fillness"),
            "ar_sa_f1": ar_f1,
            "is_answerable": is_ans
        })

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for crow in csv_rows:
            writer.writerow(crow)

    total_eval_rows = len(csv_rows)
    queries_evaluated = len(set(r["query_id"] for r in csv_rows))
    baselines = sorted(list(set(r["baseline"] for r in csv_rows if r["baseline"])))

    metrics_summary = {}
    metric_names = [
        "retrieval_recall", "context_precision", "faithfulness",
        "answer_relevance", "citation_fidelity", "semantic_accuracy", "context_fillness",
        "ar_sa_f1"
    ]

    for m in metric_names:
        vals = [r[m] for r in csv_rows if r.get(m) is not None and r[m] != ""]
        if vals:
            metrics_summary[m] = {
                "mean": round(statistics.mean(vals), 4),
                "count": len(vals)
            }

    summary_data = {
        "total_eval_rows": total_eval_rows,
        "queries_evaluated": queries_evaluated,
        "baselines": baselines,
        "metrics": metrics_summary,
        "errors": {
            "count": 0,
            "by_type": {}
        }
    }

    with open(parsed_dir / "eval_trace.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    return csv_rows, summary_data


def parse_all_traces(
    traces_dir: Path, parsed_dir: Path
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Parses all trace files inside traces_dir into parsed CSVs and JSON summaries.

    Returns:
        Tuple of (graph_trace_summary, eval_trace_summary).
    """
    _, g_summary = parse_graph_retrieval_trace(traces_dir, parsed_dir)
    _, e_summary = parse_eval_trace(traces_dir, parsed_dir)
    return g_summary, e_summary
