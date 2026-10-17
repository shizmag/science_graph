import statistics
from typing import Any
from core.metrics import (
    calculate_retrieval_recall,
    calculate_context_precision,
    estimate_prompt_tokens,
    calculate_semantic_accuracy,
    get_is_answerable,
    classify_answerability
)
from core.models import parse_report, ReportOutput

QUALITY_METRICS = [
    "retrieval_recall",
    "context_precision",
    "faithfulness",
    "answer_relevance",
    "citation_fidelity",
    "semantic_accuracy",
    "context_fillness",
    "ar_sa_f1"
]

TELEMETRY_METRICS = [
    "h_gen", "h_citation", "delta_h_gen",
    "avg_msp", "avg_logit_margin",
    "first_token_msp", "first_token_margin",
    "ll_rag", "ll_base", "clr",
    "n_citation_tokens",
]

ALL_METRICS = QUALITY_METRICS + ["latency_sec", "token_output", "token_answer", "token_reasoning"]

METRIC_LABELS = {
    "retrieval_recall": "Retrieval Recall",
    "context_precision": "Context Precision",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "citation_fidelity": "Citation Fidelity",
    "semantic_accuracy": "Semantic Accuracy",
    "context_fillness": "Context Fillness",
    "ar_sa_f1": "AR-SA F1",
    "latency_sec": "Latency (sec)",
    "token_output": "Token Output",
    "token_answer": "Token Answer",
    "token_reasoning": "Token Reasoning",
    "h_gen": "H_gen",
    "h_citation": "H_citation",
    "delta_h_gen": "ΔH_gen",
    "avg_msp": "Avg MSP",
    "avg_logit_margin": "Avg Logit Margin",
    "first_token_msp": "First-Token MSP",
    "first_token_margin": "First-Token Margin",
    "ll_rag": "LL_RAG",
    "ll_base": "LL_Base",
    "clr": "CLR",
    "n_citation_tokens": "Citation Tokens",
}


DETAIL_GRAPH_FIELDS = {
    "graph_retrieval_enabled": (False, bool),
    "graph_retrieval_skip_reason": ("", str),
    
    # Concept diagnostics
    "query_concepts_all_count": (0, int),
    "query_concepts_strong_count": (0, int),
    "query_concepts_dropped_count": (0, int),
    "query_concepts_all": ([], list),
    "query_concepts_strong": ([], list),
    "query_concepts_dropped": ([], list),
    
    # Graph neighbor resolution
    "graph_neighbor_nodes_total": (0, int),
    "graph_neighbor_paper_nodes_count": (0, int),
    "graph_neighbor_local_papers_count": (0, int),
    "graph_neighbor_papers_with_chunks_count": (0, int),
    "graph_neighbor_placeholder_or_external_count": (0, int),
    "graph_neighbor_non_paper_nodes_count": (0, int),
    "graph_neighbor_chunks_retrieved_count": (0, int),
    
    # Graph candidate counts
    "graph_concept_candidate_papers_count": (0, int),
    "graph_bridge_candidate_papers_count": (0, int),
    "graph_chunks_before_rerank_count": (0, int),
    "graph_chunk_candidates_count": (0, int),
    "graph_candidate_source_breakdown": ({}, dict),
    
    # Reranker integration
    "base_candidates_count": (0, int),
    "merged_candidates_count_before_reranker": (0, int),
    "reranker_input_count_before_limit": (0, int),
    "reranker_input_count_after_limit": (0, int),
    "candidate_count_after_reranker": (0, int),
    
    # Graph survival
    "graph_candidate_rerank_positions": ([], list),
    "best_graph_candidate_rank_after_rerank": (None, int),
    "graph_chunks_survived_final_context_count": (0, int),
    "graph_survival_rate": (0.0, float),
    "graph_chunks_survived_final_context": ([], list),
    
    # Final context diversity
    "distinct_papers_in_final_context": (0, int),
    
    # Optional samples/debug lists
    "graph_neighbor_resolution_sample": ([], list),
    "graph_concept_candidate_papers": ([], list),
    "graph_bridge_candidate_papers": ([], list),
    "graph_chunks_before_rerank": ([], list),
}


GRAPH_ENABLED_BASELINES = {"B5", "B6"}


def analyze_metrics(data: Any, trace_map: dict = None) -> dict:
    """Computes all summary statistics, wins, breakdowns, and difficulties from YAML data."""
    # Parsed with Pydantic for strict validation and format unification
    if isinstance(data, ReportOutput):
        report = data
    else:
        report = parse_report(data)
    
    # Mutate a dictionary format of the report to keep compatibility and support in-place mutations
    data_dict = report.model_dump()
    results = data_dict.get("results", [])
    if not results:
        raise ValueError("No results found in the benchmarking data.")
        
    # Find all baselines present in the first result
    first_result = results[0]
    baselines = list(first_result.get("baselines", {}).keys())
    if not baselines:
        for r in results:
            if r.get("baselines"):
                baselines = list(r["baselines"].keys())
                break
    baselines = sorted(baselines)
    baselines = [b for b in baselines if any(r.get("baselines", {}).get(b, {}).get("status") == "success" for r in results)]

    # 0. Merge trace entries if trace_map is provided (or fallback to defaults)
    for r in results:
        q_id = r.get("id")
        query = r.get("query")
        is_ans = get_is_answerable(r)

        for b in baselines:
            b_data = r.setdefault("baselines", {}).setdefault(b, {})
            
            # Pre-calculate deterministic answerability outcome once globally
            gen_ans = b_data.get("generated_answer", "")
            outcome = b_data.get("answerability_outcome")
            if outcome not in ("TP", "FP", "TN", "FN"):
                pred_abst = b_data.get("predicted_abstained")
                if pred_abst is None:
                    try:
                        from core.metrics import detect_abstention
                        pred_abst = detect_abstention(gen_ans)
                    except Exception:
                        gen_lower = gen_ans.lower()
                        pred_abst = any(w in gen_lower for w in ["no information", "information missing", "cannot answer", "insufficient information", "нет информации", "отсутствует"])
                from core.metrics import classify_answerability
                outcome = classify_answerability(is_ans, pred_abst)
                b_data["answerability_outcome"] = outcome

            # Graph diagnostics population with strict baseline isolation
            trace_entry = None
            if b in GRAPH_ENABLED_BASELINES and trace_map:
                if q_id:
                    trace_entry = trace_map.get((b, str(q_id)))
                if not trace_entry and query:
                    trace_entry = trace_map.get((b, str(query)))
            
            for field, (default_val, _) in DETAIL_GRAPH_FIELDS.items():
                if b not in GRAPH_ENABLED_BASELINES:
                    if isinstance(default_val, list):
                        val = []
                    elif isinstance(default_val, dict):
                        val = {}
                    else:
                        val = default_val
                else:
                    val = None
                    if trace_entry:
                        if field == "query_concepts_all_count":
                            val = len(trace_entry.get("query_concepts_all", []))
                        elif field == "query_concepts_strong_count":
                            val = len(trace_entry.get("query_concepts_strong", []))
                        elif field == "query_concepts_dropped_count":
                            val = len(trace_entry.get("query_concepts_dropped", []))
                        else:
                            val = trace_entry.get(field)
                    
                    if val is None:
                        if isinstance(default_val, list):
                            val = []
                        elif isinstance(default_val, dict):
                            val = {}
                        else:
                            val = default_val
                        
                b_data[field] = val
    
    # Pre-calculate missing semantic accuracies for all results/baselines
    missing_semantics = []
    golden_list = []
    generated_list = []
    for r_idx, r in enumerate(results):
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
            eval_metrics = b_data.get("eval_metrics", {})
            sem = None
            if isinstance(eval_metrics, dict):
                sem = eval_metrics.get("semantic_accuracy")
            if sem is None:
                sem = b_data.get("semantic_accuracy")
            if sem is None:
                gold = (r.get("golden_answer") or "").strip()
                gen_raw = (b_data.get("generated_answer") or "").strip()
                if gold and gen_raw:
                    from core.sanitization import extract_clean_answer
                    _, gen = extract_clean_answer(gen_raw)
                    golden_list.append(gold)
                    generated_list.append(gen)
                    missing_semantics.append((r_idx, b))
                
    if missing_semantics:
        computed_sems = calculate_semantic_accuracy(golden_list, generated_list)
        for (r_idx, b), val in zip(missing_semantics, computed_sems):
            b_data = results[r_idx]["baselines"][b]
            if "eval_metrics" not in b_data or not isinstance(b_data["eval_metrics"], dict):
                b_data["eval_metrics"] = {}
            b_data["eval_metrics"]["semantic_accuracy"] = val

    metadata = data_dict.get("metadata") or {}
    original_metadata = metadata.get("original_metadata") or metadata
    max_input_token = original_metadata.get("llm", {}).get("model_max_context")
    if max_input_token is None:
        try:
            from src.config import config
            max_input_token = getattr(config, "llm_model_max_context", 4096)
        except Exception:
            max_input_token = 4096

    # Fill in other deterministic metrics if missing
    for r in results:
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
            if "eval_metrics" not in b_data or not isinstance(b_data["eval_metrics"], dict):
                b_data["eval_metrics"] = {}
            eval_metrics = b_data["eval_metrics"]
            
            # 1. retrieval_recall
            if eval_metrics.get("retrieval_recall") is None:
                rec = b_data.get("retrieval_recall")
                if rec is None:
                    rec = calculate_retrieval_recall(
                        r.get("expected_papers") or [], 
                        b_data.get("retrieved_papers") or []
                    )
                eval_metrics["retrieval_recall"] = rec
                
            # 2. context_precision
            if eval_metrics.get("context_precision") is None:
                prec = b_data.get("context_precision")
                if prec is None:
                    prec = calculate_context_precision(
                        r.get("expected_papers") or [], 
                        b_data.get("retrieved_chunks") or []
                    )
                eval_metrics["context_precision"] = prec
                
            # 3. context_fillness
            if eval_metrics.get("context_fillness") is None:
                fillness = b_data.get("context_fillness")
                if fillness is None:
                    context_token = b_data.get("context_token")
                    max_input_token_val = b_data.get("max_input_token")
                    if context_token is None:
                        context_token = estimate_prompt_tokens(
                            r.get("query") or "", 
                            b_data.get("retrieved_chunks") or [], 
                            b
                        )
                    if max_input_token_val is None:
                        max_input_token_val = max_input_token
                    fillness = round(context_token / max_input_token_val, 4) if max_input_token_val > 0 else 0.0
                    fillness = min(max(fillness, 0.0), 1.0)
                eval_metrics["context_fillness"] = fillness

            # 4. token_output, token_answer, token_reasoning
            if eval_metrics.get("token_output") is None:
                generated_answer = b_data.get("generated_answer", "")
                from core.evaluator import get_clean_judge_answer
                judge_answer = get_clean_judge_answer(generated_answer)
                from core.metrics import count_text_tokens
                token_output = count_text_tokens(generated_answer)
                token_answer = count_text_tokens(judge_answer)
                token_reasoning = max(0, token_output - token_answer)
                eval_metrics["token_output"] = token_output
                eval_metrics["token_answer"] = token_answer
                eval_metrics["token_reasoning"] = token_reasoning

            # 5. ar_sa_f1
            is_ans = get_is_answerable(r)
            if is_ans:
                r_relevance = eval_metrics.get("answer_relevance")
                s_accuracy = eval_metrics.get("semantic_accuracy")
                if r_relevance is not None and s_accuracy is not None:
                    try:
                        r_val = float(r_relevance)
                        s_val = float(s_accuracy)
                        if r_val + s_val > 0:
                            eval_metrics["ar_sa_f1"] = round(2.0 * (r_val * s_val) / (r_val + s_val), 4)
                        else:
                            eval_metrics["ar_sa_f1"] = 0.0
                    except (ValueError, TypeError):
                        eval_metrics["ar_sa_f1"] = 0.0
                else:
                    eval_metrics["ar_sa_f1"] = None
            else:
                eval_metrics["ar_sa_f1"] = None

    # If the input was originally a mutable dict or list, update it in-place to propagate changes
    if isinstance(data, dict):
        if "results" in data:
            data["results"] = results
        else:
            # If it's a dict representing a single case or custom structure
            data.update(data_dict)
    elif isinstance(data, list):
        # If it's a list, update elements in place
        for orig_item, new_item in zip(data, results):
            if isinstance(orig_item, dict) and isinstance(new_item, dict):
                orig_item.update(new_item)

    # Collect raw values per baseline and metric
    raw_values = {b: {m: [] for m in ALL_METRICS} for b in baselines}
    raw_values_ans = {b: {m: [] for m in ALL_METRICS} for b in baselines}
    raw_values_unans = {b: {m: [] for m in ALL_METRICS} for b in baselines}
    for b in baselines:
        raw_values[b]["status"] = []
        raw_values_ans[b]["status"] = []
        raw_values_unans[b]["status"] = []
    
    categories = set()
    category_values = {}
    query_scores = []

    # Dictionary to collect raw values for graph diagnostics per baseline
    graph_raw = {
        b: {
            "enabled": [],
            "skipped": [],
            "concepts_all": [],
            "concepts_strong": [],
            "concepts_dropped": [],
            "neighbor_nodes": [],
            "neighbor_paper_nodes": [],
            "neighbor_local_papers": [],
            "neighbor_papers_with_chunks": [],
            "neighbor_chunks_retrieved": [],
            "base_candidates": [],
            "graph_chunk_candidates": [],
            "merged_before": [],
            "reranker_before": [],
            "reranker_after": [],
            "candidates_after": [],
            "chunks_before_rerank": [],
            "chunks_survived": [],
            "best_rank": [],
            "neighbor_candidates": [],
            "concept_candidates": [],
            "bridge_candidates": [],
            "distinct_papers_final": []
        }
        for b in baselines
    }
    shannon_raw = {
        b: {
            "h_rank_pre_rerank": [],
            "h_rank_post_rerank": [],
            "h_lexical_pre_trim": [],
            "h_lexical_post_trim": [],
            "h_graph_relation_type": [],
            "h_graph_degree": [],
            "h_gen": [],
            "h_citation": [],
            "n_citation_tokens": [],
            "delta_h_gen": [],
            "msp": [],
            "avg_msp": [],
            "logit_margin": [],
            "avg_logit_margin": [],
            "first_token_margin": [],
            "first_token_msp": [],
            "citation_entropy": [],
            "ll_rag": [],
            "ll_base": [],
            "clr": []
        }
        for b in baselines
    }
    category_graph_raw = {}
    
    for r in results:
        q_id = r.get("id", "UNKNOWN")
        q_text = r.get("query", "")
        category = r.get("category", "default")
        categories.add(category)
        
        if category not in category_values:
            category_values[category] = {b: {m: [] for m in ALL_METRICS} for b in baselines}

        if category not in category_graph_raw:
            category_graph_raw[category] = {
                b: {
                    "graph_chunk_candidates": [],
                    "chunks_before_rerank": [],
                    "chunks_survived": [],
                    "concepts_strong": [],
                    "distinct_papers_final": []
                }
                for b in baselines
            }
            
        q_quality_sum = 0.0
        q_quality_count = 0
        
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
                
            is_ans = get_is_answerable(r)
            status = b_data.get("status", "failed")
            raw_values[b]["status"].append(status)
            if is_ans:
                raw_values_ans[b]["status"].append(status)
            else:
                raw_values_unans[b]["status"].append(status)
            
            # Latency
            lat = b_data.get("latency_sec")
            if lat is not None:
                raw_values[b]["latency_sec"].append(lat)
                category_values[category][b]["latency_sec"].append(lat)
                if is_ans:
                    raw_values_ans[b]["latency_sec"].append(lat)
                else:
                    raw_values_unans[b]["latency_sec"].append(lat)
                
            # Quality metrics
            eval_metrics = b_data.get("eval_metrics", {})
            for m in QUALITY_METRICS:
                val = eval_metrics.get(m) if isinstance(eval_metrics, dict) else None
                if val is None:
                    val = b_data.get(m)
                if val is not None:
                    raw_values[b][m].append(val)
                    if is_ans:
                        category_values[category][b][m].append(val)
                        raw_values_ans[b][m].append(val)
                    else:
                        raw_values_unans[b][m].append(val)
                    q_quality_sum += val
                    q_quality_count += 1

            # Token metrics
            for m in ["token_output", "token_answer", "token_reasoning"]:
                val = eval_metrics.get(m) if isinstance(eval_metrics, dict) else None
                if val is None:
                    val = b_data.get(m)
                if val is not None:
                    raw_values[b][m].append(val)
                    category_values[category][b][m].append(val)
                    if is_ans:
                        raw_values_ans[b][m].append(val)
                    else:
                        raw_values_unans[b][m].append(val)

            # Shannon diagnostics collection & offline backfill
            shannon_diag = b_data.get("shannon_diagnostics") or (b_data.get("metrics", {}).get("shannon_diagnostics") if isinstance(b_data.get("metrics"), dict) else {}) or {}
            if not shannon_diag:
                retrieved_chunks = b_data.get("retrieved_chunks", [])
                if retrieved_chunks:
                    from core.shannon_estimator import assemble_retrieval_shannon_fields
                    post_scores = [
                        c.get("score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0)
                        for c in retrieved_chunks
                    ]
                    post_text = b_data.get("trimmed_text") or "\n".join(
                        [
                            c.get("text_content", "") if isinstance(c, dict) else getattr(c, "text_content", "")
                            for c in retrieved_chunks
                        ]
                    )
                    retrieval_fields = assemble_retrieval_shannon_fields(
                        pre_scores=b_data.get("pre_rerank_scores"),
                        post_scores=post_scores,
                        pre_text=b_data.get("context_text"),
                        post_text=post_text,
                        relations=b_data.get("graph_relations"),
                        graph_text=b_data.get("context_graph") or b_data.get("trimmed_graph"),
                    )
                    shannon_diag = {
                        **retrieval_fields,
                        "h_gen": 0.0,
                        "h_citation": 0.0,
                        "n_citation_tokens": 0,
                        "delta_h_gen": 0.0,
                    }
                    b_data["shannon_diagnostics"] = shannon_diag

            tokens_info = b_data.get("tokens_info") or (b_data.get("metrics", {}).get("tokens_info") if isinstance(b_data.get("metrics"), dict) else None)
            if tokens_info and isinstance(tokens_info, list):
                try:
                    from core.shannon_estimator import compute_sequence_telemetry
                    seq_tel = compute_sequence_telemetry(tokens_info)
                    for k, v in seq_tel.items():
                        if k not in shannon_diag or shannon_diag[k] is None:
                            shannon_diag[k] = v
                except Exception:
                    pass

            def _get_val(k_diag: str, k_alt: str | None = None) -> float | None:
                v = shannon_diag.get(k_diag)
                if v is None and k_alt:
                    v = shannon_diag.get(k_alt)
                if v is None:
                    v = b_data.get(k_diag)
                if v is None and k_alt:
                    v = b_data.get(k_alt)
                if v is None and isinstance(eval_metrics, dict):
                    v = eval_metrics.get(k_diag) or (eval_metrics.get(k_alt) if k_alt else None)
                if v is not None:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None
                return None

            s_map = {
                "h_rank_pre_rerank": shannon_diag.get("rank_entropy_pre") if shannon_diag.get("rank_entropy_pre") is not None else shannon_diag.get("h_rank_pre_rerank"),
                "h_rank_post_rerank": shannon_diag.get("rank_entropy_post") if shannon_diag.get("rank_entropy_post") is not None else shannon_diag.get("h_rank_post_rerank"),
                "h_lexical_pre_trim": shannon_diag.get("lexical_entropy_pre") if shannon_diag.get("lexical_entropy_pre") is not None else shannon_diag.get("h_lexical_pre_trim"),
                "h_lexical_post_trim": shannon_diag.get("lexical_entropy_post") if shannon_diag.get("lexical_entropy_post") is not None else shannon_diag.get("h_lexical_post_trim"),
                "h_graph_relation_type": shannon_diag.get("graph_relation_entropy") if shannon_diag.get("graph_relation_entropy") is not None else shannon_diag.get("h_graph_relation_type"),
                "h_graph_degree": shannon_diag.get("graph_degree_entropy") if shannon_diag.get("graph_degree_entropy") is not None else shannon_diag.get("h_graph_degree"),
                "h_gen": shannon_diag.get("generation_entropy") if shannon_diag.get("generation_entropy") is not None else shannon_diag.get("h_gen"),
                "h_citation": shannon_diag.get("citation_entropy") if shannon_diag.get("citation_entropy") is not None else shannon_diag.get("h_citation"),
                "n_citation_tokens": shannon_diag.get("citation_token_count") if shannon_diag.get("citation_token_count") is not None else shannon_diag.get("n_citation_tokens"),
                "delta_h_gen": shannon_diag.get("entropy_reduction") if shannon_diag.get("entropy_reduction") is not None else shannon_diag.get("delta_h_gen"),
                "msp": _get_val("msp", "avg_msp"),
                "avg_msp": _get_val("avg_msp", "msp"),
                "logit_margin": _get_val("logit_margin", "avg_logit_margin"),
                "avg_logit_margin": _get_val("avg_logit_margin", "logit_margin"),
                "first_token_margin": _get_val("first_token_margin"),
                "first_token_msp": _get_val("first_token_msp"),
                "citation_entropy": _get_val("citation_entropy", "h_citation"),
                "ll_rag": _get_val("ll_rag"),
                "ll_base": _get_val("ll_base"),
                "clr": _get_val("clr"),
            }

            # Copy telemetry values into b_data top-level and shannon_diagnostics for full consistency
            for field in [
                "msp", "avg_msp", "logit_margin", "avg_logit_margin",
                "first_token_margin", "first_token_msp", "citation_entropy",
                "ll_rag", "ll_base", "clr"
            ]:
                val = s_map.get(field)
                if val is not None:
                    if b_data.get(field) is None:
                        b_data[field] = val
                    if isinstance(shannon_diag, dict) and shannon_diag.get(field) is None:
                        shannon_diag[field] = val
            for skey, sval in s_map.items():
                if sval is not None:
                    shannon_raw[b][skey].append(float(sval))

            # Collect graph retrieval diagnostics
            enabled = b_data.get("graph_retrieval_enabled", False)
            graph_raw[b]["enabled"].append(enabled)
            
            skip_reason = b_data.get("graph_retrieval_skip_reason", "")
            graph_raw[b]["skipped"].append(bool(skip_reason))
            
            graph_raw[b]["concepts_all"].append(b_data.get("query_concepts_all_count", 0))
            graph_raw[b]["concepts_strong"].append(b_data.get("query_concepts_strong_count", 0))
            graph_raw[b]["concepts_dropped"].append(b_data.get("query_concepts_dropped_count", 0))
            
            graph_raw[b]["neighbor_nodes"].append(b_data.get("graph_neighbor_nodes_total", 0))
            graph_raw[b]["neighbor_paper_nodes"].append(b_data.get("graph_neighbor_paper_nodes_count", 0))
            graph_raw[b]["neighbor_local_papers"].append(b_data.get("graph_neighbor_local_papers_count", 0))
            graph_raw[b]["neighbor_papers_with_chunks"].append(b_data.get("graph_neighbor_papers_with_chunks_count", 0))
            graph_raw[b]["neighbor_chunks_retrieved"].append(b_data.get("graph_neighbor_chunks_retrieved_count", 0))
            
            graph_raw[b]["base_candidates"].append(b_data.get("base_candidates_count", 0))
            graph_raw[b]["graph_chunk_candidates"].append(b_data.get("graph_chunk_candidates_count", 0))
            graph_raw[b]["merged_before"].append(b_data.get("merged_candidates_count_before_reranker", 0))
            graph_raw[b]["reranker_before"].append(b_data.get("reranker_input_count_before_limit", 0))
            graph_raw[b]["reranker_after"].append(b_data.get("reranker_input_count_after_limit", 0))
            graph_raw[b]["candidates_after"].append(b_data.get("candidate_count_after_reranker", 0))
            
            graph_raw[b]["chunks_before_rerank"].append(b_data.get("graph_chunks_before_rerank_count", 0))
            graph_raw[b]["chunks_survived"].append(b_data.get("graph_chunks_survived_final_context_count", 0))
            
            best_rank = b_data.get("best_graph_candidate_rank_after_rerank")
            if best_rank is not None:
                try:
                    graph_raw[b]["best_rank"].append(float(best_rank))
                except (ValueError, TypeError):
                    pass
            
            breakdown = b_data.get("graph_candidate_source_breakdown") or {}
            graph_raw[b]["neighbor_candidates"].append(breakdown.get("graph_neighbor", b_data.get("graph_neighbor_chunks_retrieved_count", 0)))
            graph_raw[b]["concept_candidates"].append(breakdown.get("graph_concept_retrieval", b_data.get("graph_concept_candidate_papers_count", 0)))
            graph_raw[b]["bridge_candidates"].append(breakdown.get("graph_bridge_retrieval", b_data.get("graph_bridge_candidate_papers_count", 0)))
            
            graph_raw[b]["distinct_papers_final"].append(b_data.get("distinct_papers_in_final_context", 0))

            # Category raw graph collection
            category_graph_raw[category][b]["graph_chunk_candidates"].append(b_data.get("graph_chunk_candidates_count", 0))
            category_graph_raw[category][b]["chunks_before_rerank"].append(b_data.get("graph_chunks_before_rerank_count", 0))
            category_graph_raw[category][b]["chunks_survived"].append(b_data.get("graph_chunks_survived_final_context_count", 0))
            category_graph_raw[category][b]["concepts_strong"].append(b_data.get("query_concepts_strong_count", 0))
            category_graph_raw[category][b]["distinct_papers_final"].append(b_data.get("distinct_papers_in_final_context", 0))
                    
        # Compute query average quality score to determine difficulty
        if q_quality_count > 0:
            avg_q_score = q_quality_sum / q_quality_count
            query_scores.append({
                "id": q_id,
                "query": q_text,
                "category": category,
                "avg_score": avg_q_score
            })
            
    # Sort queries by difficulty (hardest first, i.e., lowest score)
    query_scores.sort(key=lambda x: x["avg_score"])
    
    # Compute summary statistics
    summary_stats = {}
    for b in baselines:
        summary_stats[b] = {}
        statuses = raw_values_ans[b]["status"]
        success_rate = (statuses.count("success") / len(statuses)) * 100 if statuses else 0.0
        summary_stats[b]["success_rate"] = success_rate
        
        import math

        def _calc_stats(raw_list):
            clean = [v for v in raw_list if v is not None and isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
            if not clean:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "stdev": 0.0, "count": 0}
            return {
                "mean": statistics.mean(clean),
                "min": min(clean),
                "max": max(clean),
                "median": statistics.median(clean),
                "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
                "count": len(clean)
            }

        for m in ALL_METRICS:
            summary_stats[b][m] = _calc_stats(raw_values_ans[b][m])

        # Shannon diagnostics summary calculation
        s_stats = {}
        for skey, svals in shannon_raw[b].items():
            clean_s = [v for v in svals if v is not None and isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
            s_stats[skey] = round(statistics.mean(clean_s), 4) if clean_s else 0.0
        summary_stats[b]["shannon_summary"] = s_stats

        # Answerable-only metrics
        summary_stats[b]["answerable_only"] = {}
        for m in ALL_METRICS:
            summary_stats[b]["answerable_only"][m] = _calc_stats(raw_values_ans[b][m])

        # Unanswerable-only metrics
        summary_stats[b]["unanswerable_only"] = {}
        for m in ALL_METRICS:
            summary_stats[b]["unanswerable_only"][m] = _calc_stats(raw_values_unans[b][m])

        # Classification metrics
        tp, fn, tn, fp = 0, 0, 0, 0
        for r in results:
            is_ans = get_is_answerable(r)
            b_data = r.get("baselines", {}).get(b, {})
            gen_ans = b_data.get("generated_answer", "")
            outcome = b_data.get("answerability_outcome")
            if outcome not in ("TP", "FP", "TN", "FN"):
                pred_abst = b_data.get("predicted_abstained")
                if pred_abst is None:
                    try:
                        from core.metrics import detect_abstention
                        pred_abst = detect_abstention(gen_ans)
                    except Exception:
                        gen_lower = gen_ans.lower()
                        pred_abst = any(w in gen_lower for w in ["no information", "information missing", "cannot answer", "insufficient information", "нет информации", "отсутствует"])
                from core.metrics import classify_answerability
                outcome = classify_answerability(is_ans, pred_abst)
            
            if outcome == "TP": tp += 1
            elif outcome == "FN": fn += 1
            elif outcome == "TN": tn += 1
            elif outcome == "FP": fp += 1

        total = tp + fn + tn + fp
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall and (precision + recall) > 0) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        fnr = fn / (fn + tp) if (fn + tp) > 0 else None
        
        num_unans = fp + tn
        hallucination_rate = fp / num_unans if num_unans > 0 else None
        answer_rate = (tp + fp) / total if total > 0 else 0.0
        abstention_rate = (tn + fn) / total if total > 0 else 0.0

        summary_stats[b]["classification"] = {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "fpr": fpr,
            "fnr": fnr,
            "hallucination_rate": hallucination_rate,
            "answer_rate": answer_rate,
            "abstention_rate": abstention_rate
        }
        summary_stats[b]["unanswerable_safety"] = {
            "unanswerable_count": num_unans,
            "TN": tn,
            "FP": fp,
            "abstention_accuracy": specificity if specificity is not None else 0.0,
            "hallucination_rate": hallucination_rate if hallucination_rate is not None else 0.0,
            "answer_rate_unans": fp / num_unans if num_unans > 0 else 0.0
        }

        # Calculate graph retrieval aggregates
        total_q = len(results)
        enabled_rate = sum(1 for e in graph_raw[b]["enabled"] if e) / total_q if total_q > 0 else 0.0
        skipped_rate = sum(1 for s in graph_raw[b]["skipped"] if s) / total_q if total_q > 0 else 0.0
        
        avg_concepts = statistics.mean(graph_raw[b]["concepts_all"]) if graph_raw[b]["concepts_all"] else 0.0
        avg_strong = statistics.mean(graph_raw[b]["concepts_strong"]) if graph_raw[b]["concepts_strong"] else 0.0
        avg_dropped = statistics.mean(graph_raw[b]["concepts_dropped"]) if graph_raw[b]["concepts_dropped"] else 0.0
        
        avg_neighbor_nodes = statistics.mean(graph_raw[b]["neighbor_nodes"]) if graph_raw[b]["neighbor_nodes"] else 0.0
        avg_neighbor_paper_nodes = statistics.mean(graph_raw[b]["neighbor_paper_nodes"]) if graph_raw[b]["neighbor_paper_nodes"] else 0.0
        avg_neighbor_local_papers = statistics.mean(graph_raw[b]["neighbor_local_papers"]) if graph_raw[b]["neighbor_local_papers"] else 0.0
        avg_neighbor_papers_with_chunks = statistics.mean(graph_raw[b]["neighbor_papers_with_chunks"]) if graph_raw[b]["neighbor_papers_with_chunks"] else 0.0
        avg_neighbor_chunks_retrieved = statistics.mean(graph_raw[b]["neighbor_chunks_retrieved"]) if graph_raw[b]["neighbor_chunks_retrieved"] else 0.0
        
        avg_base_candidates = statistics.mean(graph_raw[b]["base_candidates"]) if graph_raw[b]["base_candidates"] else 0.0
        avg_graph_chunk_candidates = statistics.mean(graph_raw[b]["graph_chunk_candidates"]) if graph_raw[b]["graph_chunk_candidates"] else 0.0
        avg_merged_before = statistics.mean(graph_raw[b]["merged_before"]) if graph_raw[b]["merged_before"] else 0.0
        avg_reranker_before = statistics.mean(graph_raw[b]["reranker_before"]) if graph_raw[b]["reranker_before"] else 0.0
        avg_reranker_after = statistics.mean(graph_raw[b]["reranker_after"]) if graph_raw[b]["reranker_after"] else 0.0
        avg_candidates_after = statistics.mean(graph_raw[b]["candidates_after"]) if graph_raw[b]["candidates_after"] else 0.0
        
        sum_before = sum(graph_raw[b]["chunks_before_rerank"])
        sum_survived = sum(graph_raw[b]["chunks_survived"])
        survival_rate = sum_survived / sum_before if sum_before > 0 else 0.0
        
        queries_with_chunks = sum(1 for c in graph_raw[b]["chunks_before_rerank"] if c > 0) / total_q if total_q > 0 else 0.0
        queries_with_chunks_survived = sum(1 for c in graph_raw[b]["chunks_survived"] if c > 0) / total_q if total_q > 0 else 0.0
        
        avg_best_rank = statistics.mean(graph_raw[b]["best_rank"]) if graph_raw[b]["best_rank"] else None
        
        avg_neighbor_cand = statistics.mean(graph_raw[b]["neighbor_candidates"]) if graph_raw[b]["neighbor_candidates"] else 0.0
        avg_concept_cand = statistics.mean(graph_raw[b]["concept_candidates"]) if graph_raw[b]["concept_candidates"] else 0.0
        avg_bridge_cand = statistics.mean(graph_raw[b]["bridge_candidates"]) if graph_raw[b]["bridge_candidates"] else 0.0
        
        avg_distinct_papers = statistics.mean(graph_raw[b]["distinct_papers_final"]) if graph_raw[b]["distinct_papers_final"] else 0.0
        
        summary_stats[b]["graph_diagnostics"] = {
            "enabled_rate": enabled_rate,
            "skipped_rate": skipped_rate,
            "avg_concepts": avg_concepts,
            "avg_strong": avg_strong,
            "avg_dropped": avg_dropped,
            "avg_neighbor_nodes": avg_neighbor_nodes,
            "avg_neighbor_paper_nodes": avg_neighbor_paper_nodes,
            "avg_neighbor_local_papers": avg_neighbor_local_papers,
            "avg_neighbor_papers_with_chunks": avg_neighbor_papers_with_chunks,
            "avg_neighbor_chunks_retrieved": avg_neighbor_chunks_retrieved,
            "avg_base_candidates": avg_base_candidates,
            "avg_graph_chunk_candidates": avg_graph_chunk_candidates,
            "avg_merged_before": avg_merged_before,
            "avg_reranker_before": avg_reranker_before,
            "avg_reranker_after": avg_reranker_after,
            "avg_candidates_after": avg_candidates_after,
            "survival_rate": survival_rate,
            "queries_with_chunks": queries_with_chunks,
            "queries_with_chunks_survived": queries_with_chunks_survived,
            "avg_best_rank": avg_best_rank,
            "avg_neighbor_cand": avg_neighbor_cand,
            "avg_concept_cand": avg_concept_cand,
            "avg_bridge_cand": avg_bridge_cand,
            "avg_distinct_papers": avg_distinct_papers
        }
            
    # Compute category statistics
    category_stats = {}
    for cat in sorted(categories):
        category_stats[cat] = {}
        for b in baselines:
            category_stats[cat][b] = {}
            for m in ALL_METRICS:
                vals = category_values[cat][b][m]
                if vals:
                    category_stats[cat][b][m] = statistics.mean(vals)
                else:
                    category_stats[cat][b][m] = 0.0

    category_classification = {}
    for cat in sorted(categories):
        category_classification[cat] = {}
        cat_rows = [r for r in results if r.get("category", "default") == cat]
        for b in baselines:
            tp, fn, tn, fp = 0, 0, 0, 0
            for r in cat_rows:
                b_data = r.get("baselines", {}).get(b, {})
                outcome = b_data.get("answerability_outcome")
                
                if outcome == "TP": tp += 1
                elif outcome == "FN": fn += 1
                elif outcome == "TN": tn += 1
                elif outcome == "FP": fp += 1
            
            total = tp + fn + tn + fp
            category_classification[cat][b] = {
                "TP": tp,
                "FN": fn,
                "TN": tn,
                "FP": fp,
                "total": total,
                "abstention_accuracy": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                "hallucination_rate": fp / (tn + fp) if (tn + fp) > 0 else 0.0
            }

    # Compute category graph statistics
    category_graph_stats = {}
    for cat in sorted(categories):
        category_graph_stats[cat] = {}
        for b in baselines:
            raw_c = category_graph_raw[cat][b]
            avg_chunk_cand = statistics.mean(raw_c["graph_chunk_candidates"]) if raw_c["graph_chunk_candidates"] else 0.0
            
            sum_before_c = sum(raw_c["chunks_before_rerank"])
            sum_survived_c = sum(raw_c["chunks_survived"])
            survival_rate_c = sum_survived_c / sum_before_c if sum_before_c > 0 else 0.0
            
            queries_survived_c = sum(1 for x in raw_c["chunks_survived"] if x > 0) / len(raw_c["chunks_survived"]) if raw_c["chunks_survived"] else 0.0
            
            avg_strong_c = statistics.mean(raw_c["concepts_strong"]) if raw_c["concepts_strong"] else 0.0
            avg_distinct_c = statistics.mean(raw_c["distinct_papers_final"]) if raw_c["distinct_papers_final"] else 0.0
            
            category_graph_stats[cat][b] = {
                "avg_graph_chunk_candidates": avg_chunk_cand,
                "graph_survival_rate": survival_rate_c,
                "queries_with_graph_chunks_survived": queries_survived_c,
                "avg_strong_query_concepts": avg_strong_c,
                "avg_distinct_papers_in_final_context": avg_distinct_c
            }
                    
    # Pairwise Win Rate (how often baseline X > baseline Y for a given metric)
    pairwise_win_rates = {}
    for metric in QUALITY_METRICS + ["latency_sec"]:
        pairwise_win_rates[metric] = {b1: {b2: 0.0 for b2 in baselines} for b1 in baselines}
        
        for b1 in baselines:
            for b2 in baselines:
                if b1 == b2:
                    continue
                wins = 0
                total = 0
                for r in results:
                    val1 = r.get("baselines", {}).get(b1, {}).get("eval_metrics", {}).get(metric) if metric in QUALITY_METRICS else r.get("baselines", {}).get(b1, {}).get(metric)
                    val2 = r.get("baselines", {}).get(b2, {}).get("eval_metrics", {}).get(metric) if metric in QUALITY_METRICS else r.get("baselines", {}).get(b2, {}).get(metric)
                    
                    if val1 is not None and val2 is not None:
                        total += 1
                        if metric == "latency_sec":
                            if val1 < val2:
                                wins += 1
                        else:
                            if val1 > val2:
                                wins += 1
                
                pairwise_win_rates[metric][b1][b2] = (wins / total * 100) if total > 0 else 0.0

    # Collect failure cases
    failure_cases = []
    for r in results:
        q_id = r.get("id", "UNKNOWN")
        q_text = r.get("query", "")
        category = r.get("category", "default")
        
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
            
            skip_reason = b_data.get("graph_retrieval_skip_reason", "")
            neighbor_nodes = b_data.get("graph_neighbor_nodes_total", 0)
            papers_with_chunks = b_data.get("graph_neighbor_papers_with_chunks_count", 0)
            chunks_before = b_data.get("graph_chunks_before_rerank_count", 0)
            chunks_survived = b_data.get("graph_chunks_survived_final_context_count", 0)
            
            cond1 = (neighbor_nodes > 0 and papers_with_chunks == 0)
            cond2 = (chunks_before > 0 and chunks_survived == 0)
            cond3 = bool(skip_reason)
            
            if cond1 or cond2 or cond3:
                failure_cases.append({
                    "query_id": q_id,
                    "query": q_text,
                    "baseline": b,
                    "category": category,
                    "skip_reason": skip_reason,
                    "neighbor_nodes": neighbor_nodes,
                    "papers_with_chunks": papers_with_chunks,
                    "chunks_before": chunks_before,
                    "chunks_survived": chunks_survived
                })
                
    def failure_sort_key(item):
        severity = 0
        if item["chunks_before"] > 0 and item["chunks_survived"] == 0:
            severity = 2
        elif item["neighbor_nodes"] > 0 and item["papers_with_chunks"] == 0:
            severity = 1
        return (severity, item["chunks_before"], item["neighbor_nodes"], item["query_id"])

    failure_cases.sort(key=failure_sort_key, reverse=True)
    top_failures = failure_cases[:5]

    has_graph_trace = bool(trace_map)

    total_ans = sum(1 for r in results if get_is_answerable(r))
    total_unans = len(results) - total_ans
    has_shannon = any(
        any(v > 0 for v in summary_stats[b].get("shannon_summary", {}).values())
        for b in baselines
    )

    return {
        "baselines": baselines,
        "summary": summary_stats,
        "categories": sorted(list(categories)),
        "category_stats": category_stats,
        "category_classification": category_classification,
        "query_difficulty": query_scores,
        "pairwise_win_rates": pairwise_win_rates,
        "total_queries": len(results),
        "total_answerable": total_ans,
        "total_unanswerable": total_unans,
        "has_graph_trace": has_graph_trace,
        "has_shannon": has_shannon,
        "category_graph_stats": category_graph_stats,
        "top_failures": top_failures
    }
