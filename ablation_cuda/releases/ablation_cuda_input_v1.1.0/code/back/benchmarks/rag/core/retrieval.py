import gc
import sys
import time
import yaml
from pathlib import Path
from typing import Any

from core.config import BASELINES_INFO, get_baseline_config, DEFAULT_HYPERPARAMS
from core.stats import BenchmarkStatsCollector
from core.metrics import (
    normalize_optional_text,
    get_is_answerable,
    calculate_retrieval_recall,
    calculate_context_precision,
)
from core.subprocess_runner import format_progress_marker
from src.config import config
from src import console as con


def normalize_component_scores(scores: list[float]) -> list[float]:
    """Scales raw scores from BM25 (unbounded float), Dense Vectors (Cosine/Dot),
    and Graph algorithms into [0, 1] range using Min-Max scaling across the retrieved batch.

    Edge Case Handling: If all scores are identical or if a single document is returned (max == min),
    returns [1.0] for all scores to prevent division by zero / NaN.
    """
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    score_range = max_score - min_score
    return [(s - min_score) / score_range for s in scores]


def _clear_gpu_cache() -> None:
    """Flushes PyTorch MPS/GPU memory cache and executes garbage collection."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_staged_retrieval(args: Any, config: Any, prompts: Any, container: Any, con: Any) -> None:
    """Executes query expansion, dense/lexical retrieval, batch reranking,
    and graph context construction in non-overlapping stages.
    """
    # Determine dataset path
    dataset_path = args.dataset
    if not dataset_path:
        local_dir = Path(__file__).resolve().parents[1]
        dataset_path = local_dir / "golden_dataset.yaml"
        if not dataset_path.exists():
            dataset_path = local_dir / "golden_dataset.example.yaml"
            con.info(f"Using default example dataset: {dataset_path}")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        con.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Load dataset
    from core.config import load_benchmark_dataset
    limit = getattr(args, "limit", None)
    unanswerable_limit = getattr(args, "unanswerable_limit", None)
    try:
        test_cases = load_benchmark_dataset(dataset_path, limit=limit, unanswerable_limit=unanswerable_limit)
    except Exception as e:
        con.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    if not test_cases:
        con.error("Empty or invalid dataset file.")
        sys.exit(1)

    # Resolve unique run directory
    original_output_path = Path(args.output) if getattr(args, "output", None) else Path("reports/retrieved_contexts.yaml")
    
    if args.cloud:
        cloud_val = getattr(config, "llm_cloud_rag_model_name", None)
        llm_model = cloud_val if isinstance(cloud_val, str) else config.data.get("llm", {}).get("cloud", {}).get("model_name", "cloud_model")
    else:
        local_val = getattr(config, "llm_local_rag_model_path", None)
        llm_model = local_val if isinstance(local_val, str) else config.data.get("llm", {}).get("local", {}).get("model_path", "local_model")
        
    if getattr(args, "no_unique_dir", False):
        output_path = original_output_path
        run_dir = output_path.parent
    else:
        from core.config import get_safe_model_name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = get_safe_model_name(llm_model)
        run_dir_name = f"run_retrive_{timestamp}_{safe_model}"
        run_dir = original_output_path.parent / run_dir_name
        output_path = run_dir / original_output_path.name

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed").mkdir(parents=True, exist_ok=True)
    
    args.output = str(output_path)

    # Initialize RAG Service (eager warmup disabled to prevent early model loads)
    con.info("Initializing RAG Service (eager warmup disabled)...")
    try:
        rag_service = container.get_rag_service(use_cloud=args.cloud, warmup=False)
        rag_service.trace_dir = run_dir / "traces"
    except Exception as e:
        con.error(f"Failed to initialize RAG Service: {e}")
        sys.exit(1)

    # Resolve baselines to run
    if args.baselines.lower() == "all":
        baselines_to_run = list(BASELINES_INFO.keys())
    else:
        baselines_to_run = [b.strip().upper() for b in args.baselines.split(",") if b.strip()]

    con.info(f"Staged Retrieval for {len(test_cases)} cases on baselines: {', '.join(baselines_to_run)}")
    con.blank()

    # =========================================================================
    # STAGE 1: LLM Stage (Query Expansion & HyDE)
    # =========================================================================
    con.info("=== STAGE 1: LLM Stage (Query Expansion & HyDE) ===")
    query_expansions_map = {}
    hyde_docs_map = {}

    # Check if any baseline actually requires LLM in Stage 1
    llm_required_in_stage1 = False
    for baseline in baselines_to_run:
        if baseline == "B0":
            continue
        components_settings = get_baseline_config(baseline, config.rag_components)
        if components_settings.get("llm_query_expansion", True) or (config.hyde_enabled and components_settings.get("hyde", False)):
            llm_required_in_stage1 = True
            break

    non_b0_baselines = [b for b in baselines_to_run if b != "B0"]
    retrieval_total = max(len(test_cases) * len(non_b0_baselines), 1)

    if llm_required_in_stage1:
        con.info("Warming up LLM Engine for Stage 1...")
        rag_service.llm_engine._ensure_model_loaded()

    stage1_done = 0
    for case in test_cases:
        query = case.get("query")
        for baseline in baselines_to_run:
            if baseline == "B0":
                continue

            components_settings = get_baseline_config(baseline, config.rag_components)
            orig_components = {name: config.is_component_enabled(name) for name in config.rag_components.keys()}
            orig_hyde = config.data["llm"].get("hyde_enabled", False)

            for k, v in components_settings.items():
                config.data["rag_components"][k] = v
            config.data["llm"]["hyde_enabled"] = components_settings.get("hyde", False)

            try:
                # 1. Query Expansion if enabled for this baseline
                if llm_required_in_stage1 and config.rag_components.get("llm_query_expansion", True):
                    expanded = rag_service._expand_query(query)
                    query_expansions_map[(query, baseline)] = expanded
                else:
                    query_expansions_map[(query, baseline)] = [query]

                # 2. HyDE if enabled for this baseline
                if llm_required_in_stage1 and config.hyde_enabled and config.rag_components.get("hyde", True):
                    hyde_responses = getattr(config, "hyde_count", 1)
                    docs = []
                    for _ in range(hyde_responses):
                        hypothetical = rag_service.llm_engine.generate_response(
                            prompt=prompts.get_prompt("rag", "hyde", query=query),
                            max_tokens=config.hyde_max_tokens
                        )
                        docs.append(hypothetical)
                    hyde_docs_map[(query, baseline)] = docs
                else:
                    hyde_docs_map[(query, baseline)] = []
            except Exception as e:
                con.warning(f"Stage 1 failed for baseline {baseline}, query '{query[:30]}': {e}")
                query_expansions_map[(query, baseline)] = [query]
                hyde_docs_map[(query, baseline)] = []
            finally:
                # Restore original configs
                for k, v in orig_components.items():
                    config.data["rag_components"][k] = v
                config.data["llm"]["hyde_enabled"] = orig_hyde
                stage1_done += 1
                prog_units = int(0.20 * (stage1_done / retrieval_total) * retrieval_total)
                print(format_progress_marker("retrieval", prog_units, retrieval_total), flush=True)

    if llm_required_in_stage1:
        # Explicitly unload LLM model
        rag_service.llm_engine.unload_model()
        _clear_gpu_cache()
    else:
        con.info("No baselines require LLM in Stage 1. Skipping model warmup.")
        print(format_progress_marker("retrieval", int(0.20 * retrieval_total), retrieval_total), flush=True)

    # =========================================================================
    # STAGE 2: Embedder Stage
    # =========================================================================
    con.info("=== STAGE 2: Embedder Stage (Query/HyDE Encoding) ===")
    unique_texts_to_embed = set()

    for case in test_cases:
        query = case.get("query")
        for baseline in baselines_to_run:
            if baseline == "B0":
                continue
            components_settings = get_baseline_config(baseline, config.rag_components)
            if components_settings.get("dense_search", False):
                # Add query variants
                variants = query_expansions_map.get((query, baseline), [query])
                for v in variants:
                    unique_texts_to_embed.add(v)
                # Add HyDE docs
                hyde_docs = hyde_docs_map.get((query, baseline), [])
                for d in hyde_docs:
                    unique_texts_to_embed.add(d)

    texts_list = list(unique_texts_to_embed)
    if texts_list:
        con.info(f"Encoding {len(texts_list)} unique queries/passages in one batch...")
        rag_service.emb_engine._ensure_model_loaded()
        rag_service.emb_engine.get_embeddings(texts_list, is_query=True)
        # Explicitly unload Embedder model
        rag_service.emb_engine.unload_model()
        _clear_gpu_cache()
    else:
        con.info("No queries/passages require embedding in Stage 2. Skipping.")
    print(format_progress_marker("retrieval", int(0.25 * retrieval_total), retrieval_total), flush=True)

    # =========================================================================
    # STAGE 3: DB Retrieval Stage
    # =========================================================================
    con.info("=== STAGE 3: DB Retrieval Stage (SQLite FTS5 & Vector Search) ===")
    stage3_results = {}
    traces_map = {}

    orig_expand_query = rag_service._expand_query
    orig_generate_response = rag_service.llm_engine.generate_response
    orig_classify_intent = rag_service._classify_intent_and_extract_filters

    stage3_done = 0
    for case in test_cases:
        query = case.get("query")
        for baseline in baselines_to_run:
            if baseline == "B0":
                continue

            components_settings = get_baseline_config(baseline, config.rag_components)
            orig_components = {name: config.is_component_enabled(name) for name in config.rag_components.keys()}
            orig_hyde = config.data["llm"].get("hyde_enabled", False)

            for k, v in components_settings.items():
                config.data["rag_components"][k] = v
            config.data["llm"]["hyde_enabled"] = components_settings.get("hyde", False)

            # Disable reranker during DB retrieval stage
            config.data["rag_components"]["reranker"] = False

            # Set up Stage 3 mocks to avoid loading LLM
            current_baseline = baseline
            hyde_iterator = iter(hyde_docs_map.get((query, baseline), []))

            def mocked_expand_query(q):
                return query_expansions_map.get((q, current_baseline), [q])

            def mocked_generate_response(prompt, *args, **kwargs):
                try:
                    return next(hyde_iterator)
                except StopIteration:
                    return ""

            def mocked_classify_intent(q):
                return q, None

            rag_service._expand_query = mocked_expand_query
            rag_service.llm_engine.generate_response = mocked_generate_response
            rag_service._classify_intent_and_extract_filters = mocked_classify_intent

            case_id = case.get("id", "Q")
            con.info(f"[{case_id}] Query: '{query[:60]}...' ({baseline})")

            trace = {
                "query_id": case_id,
                "category": case.get("category", "general"),
                "seed_chunks_from_lexical_dense": {"lexical": [], "dense": []},
                "seed_paper_id_list": [],
                "graph_neighbor_paper_id_list": [],
                "candidate_count_before_reranker": 0,
                "candidate_count_after_reranker": 0,
                "final_context_paper_id_list": [],
                "final_context_token_count": 0,
                "whether_graph_neighbor_chunk_survived_into_final_context": False,
                "answer_token_count": 0
            }
            trace["baseline"] = baseline
            rag_service.current_trace = trace
            traces_map[(query, baseline)] = trace

            collector = BenchmarkStatsCollector(rag_service)
            collector.start()

            try:
                use_reranker = components_settings.get("reranker", False)
                retrieval_limit = 10 if use_reranker else 5

                # Execute retrieve (this will hit the Embedder cache)
                retrieved_with_scores = rag_service.retrieve_relevant_chunks(query, limit=retrieval_limit)
                candidates = [chunk for chunk, _ in retrieved_with_scores]
                rrf_scores = {chunk.id: score for chunk, score in retrieved_with_scores}

                metrics = collector.get_metrics()
                stage3_results[(query, baseline)] = {
                    "candidates": candidates,
                    "rrf_scores": rrf_scores,
                    "metrics": metrics
                }
            except Exception as e:
                con.error(f"Stage 3 failed for baseline {baseline}, query '{query[:30]}': {e}")
            finally:
                collector.stop()
                # Restore configs
                for k, v in orig_components.items():
                    config.data["rag_components"][k] = v
                config.data["llm"]["hyde_enabled"] = orig_hyde
                stage3_done += 1
                prog_units = int((0.25 + 0.35 * (stage3_done / retrieval_total)) * retrieval_total)
                print(format_progress_marker("retrieval", prog_units, retrieval_total), flush=True)

    # Restore original functions
    rag_service._expand_query = orig_expand_query
    rag_service.llm_engine.generate_response = orig_generate_response
    rag_service._classify_intent_and_extract_filters = orig_classify_intent

    # Explicitly unload Embedder model at the end of Stage 3 in case it got loaded
    rag_service.emb_engine.unload_model()
    _clear_gpu_cache()

    # =========================================================================
    # STAGE 4: Reranker Stage
    # =========================================================================
    con.info("=== STAGE 4: Reranker Stage ===")
    pairs_to_score = []
    chunk_mapping = []
    reranker_scores_map = {}
    rerank_latency = 0.0

    has_reranker_baselines = any(get_baseline_config(b, config.rag_components).get("reranker", False) for b in baselines_to_run)

    if has_reranker_baselines:
        con.info("Warming up Reranker model...")
        rag_service._get_reranker()
        reranker = rag_service._reranker

        for case in test_cases:
            query = case.get("query")
            for baseline in baselines_to_run:
                if baseline == "B0":
                    continue
                components_settings = get_baseline_config(baseline, config.rag_components)
                if components_settings.get("reranker", False):
                    res = stage3_results.get((query, baseline))
                    if res:
                        for chunk in res["candidates"]:
                            pairs_to_score.append((query, chunk.text_content))
                            chunk_mapping.append((query, baseline, chunk))

        if pairs_to_score:
            con.info(f"Reranking {len(pairs_to_score)} candidate pairs in chunked batches...")
            t0 = time.perf_counter()
            scores = []
            chunk_size = 256
            for i in range(0, len(pairs_to_score), chunk_size):
                sub_pairs = pairs_to_score[i:i + chunk_size]
                try:
                    import torch
                    with torch.inference_mode():
                        sub_scores = reranker.predict(sub_pairs, batch_size=32, show_progress_bar=False)
                except Exception:
                    sub_scores = reranker.predict(sub_pairs)
                scores.extend(sub_scores)
                _clear_gpu_cache()
            rerank_latency = time.perf_counter() - t0
            con.success(f"Batch reranked {len(pairs_to_score)} pairs in {rerank_latency:.2f} seconds.")

            for idx, (q, b, chunk) in enumerate(chunk_mapping):
                reranker_scores_map[(q, b, chunk.id)] = float(scores[idx])
        else:
            con.info("No candidate pairs to rerank.")
    else:
        con.info("No baselines require Reranker in Stage 4. Skipping.")

    # Apply score blending and select top candidates
    final_chunks_map = {}
    for case in test_cases:
        query = case.get("query")
        for baseline in baselines_to_run:
            if baseline == "B0":
                continue
            res = stage3_results.get((query, baseline))
            if not res:
                continue

            candidates = res["candidates"]
            rrf_scores = res["rrf_scores"]
            components_settings = get_baseline_config(baseline, config.rag_components)

            trace = traces_map.get((query, baseline))
            if trace:
                trace["candidate_count_before_reranker"] = len(candidates)

            if components_settings.get("reranker", False) and candidates:
                c_scores = [reranker_scores_map.get((query, baseline, c.id), 0.0) for c in candidates]
                norm_r = normalize_component_scores(c_scores)

                rrf_vals = [rrf_scores[c.id] for c in candidates]
                norm_rrf = normalize_component_scores(rrf_vals)

                scored_candidates = []
                for idx, c in enumerate(candidates):
                    if components_settings.get("score_blending", True):
                        blended_score = 0.7 * norm_r[idx] + 0.3 * norm_rrf[idx]
                    else:
                        blended_score = float(c_scores[idx])
                    scored_candidates.append((c, blended_score, float(c_scores[idx])))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                candidate_pairs = [(chunk, raw_score) for chunk, _, raw_score in scored_candidates]
                if components_settings.get("deduplicate_parent_chunks", True):
                    from src.services.rag_service import _deduplicate_parent_chunks
                    candidate_pairs = _deduplicate_parent_chunks(candidate_pairs)
                final_chunks_map[(query, baseline)] = candidate_pairs[:5]

                # Add reranker metric proportionally
                res["metrics"]["components"]["reranking"] = {
                    "calls": len(candidates),
                    "time_sec": round(rerank_latency * (len(candidates) / len(pairs_to_score)), 4) if pairs_to_score else 0.0
                }
            else:
                candidate_pairs = [(c, rrf_scores.get(c.id, 1.0)) for c in candidates]
                if components_settings.get("deduplicate_parent_chunks", True):
                    from src.services.rag_service import _deduplicate_parent_chunks
                    candidate_pairs = _deduplicate_parent_chunks(candidate_pairs)
                final_chunks_map[(query, baseline)] = candidate_pairs[:5]

            if trace:
                trace["candidate_count_after_reranker"] = len(final_chunks_map.get((query, baseline), []))

    print(format_progress_marker("retrieval", int(0.70 * retrieval_total), retrieval_total), flush=True)

    # =========================================================================
    # STAGE 5: Graph & Trimming Stage
    # =========================================================================
    con.info("=== STAGE 5: Graph & Trimming Stage ===")
    contexts_to_save = {}
    stage5_done = 0

    for case in test_cases:
        query = case.get("query")
        case_id = case.get("id", "Q")
        contexts_to_save[case_id] = {
            "id": case_id,
            "query": query,
            "category": case.get("category", "general"),
            "golden_answer": normalize_optional_text(case.get("golden_answer")),
            "is_answerable": get_is_answerable(case),
            "expected_papers": case.get("expected_papers", []),
            "baselines": {}
        }

        for baseline in baselines_to_run:
            if baseline == "B0":
                # B0 is Zero-Shot, no retrieval contexts
                contexts_to_save[case_id]["baselines"][baseline] = {
                    "status": "success",
                    "latency_sec": 0.0,
                    "retrieved_papers": [],
                    "retrieved_chunks": [],
                    "trimmed_text": "",
                    "trimmed_graph": "",
                    "enrichment_block": "",
                    "metrics": {
                        "components": {k: {"calls": 0, "time_sec": 0.0} for k in BASELINES_INFO.keys()},
                        "total_io_calls": 0
                    }
                }
                continue

            res = stage3_results.get((query, baseline))
            if not res:
                stage5_done += 1
                prog_units = retrieval_total if stage5_done == retrieval_total else int((0.70 + 0.30 * (stage5_done / retrieval_total)) * retrieval_total)
                print(format_progress_marker("retrieval", prog_units, retrieval_total), flush=True)
                continue

            components_settings = get_baseline_config(baseline, config.rag_components)
            orig_components = {name: config.is_component_enabled(name) for name in config.rag_components.keys()}
            orig_hyde = config.data["llm"].get("hyde_enabled", False)

            for k, v in components_settings.items():
                config.data["rag_components"][k] = v
            config.data["llm"]["hyde_enabled"] = components_settings.get("hyde", False)

            # Advanced Expander for B6 or CUSTOM (if graph expansion enabled)
            if baseline == "B6" or (baseline == "CUSTOM" and components_settings.get("graph_expansion", True)):
                try:
                    from src.services.graph_expander import ExperimentalGraphExpander
                    rag_service.expander = ExperimentalGraphExpander(
                        graph_repo=rag_service.graph_repo,
                        vector_repo=rag_service.vector_repo,
                        llm_engine=rag_service.llm_engine,
                        reranker=rag_service._reranker
                    )
                except Exception as e:
                    con.warning(f"Could not load expander for {baseline}: {e}")
                    rag_service.expander = None
            else:
                rag_service.expander = None

            collector = BenchmarkStatsCollector(rag_service)
            collector.start()

            t_stage5_start = time.perf_counter()
            try:
                final_chunks = final_chunks_map.get((query, baseline), [])
                
                # build context
                context_text, context_graph = rag_service.build_context(final_chunks, limit=5)

                # trim context
                if rag_service.expander and components_settings.get("graph_expansion", True):
                    system_prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block="", history_str="", query=query)
                else:
                    system_prompt = prompts.get_prompt("rag", "ask_no_expander", context_text="", context_graph="", history_str="", query=query)

                model_max_context = getattr(config, "llm_model_max_context", 4096)

                if components_settings.get("context_trimming", True):
                    trimmed_text, trimmed_graph, trimmed_chunks = rag_service.trim_context(
                        context_text=context_text,
                        context_graph=context_graph,
                        final_chunks=final_chunks,
                        query=query,
                        history_str="",
                        system_prompt=system_prompt,
                        model_max_context=model_max_context,
                        reserved_tokens=500
                    )
                else:
                    trimmed_text, trimmed_graph, trimmed_chunks = context_text, context_graph, final_chunks

                # graph expansion (uses Reranker)
                enrichment_block = ""
                if rag_service.expander and components_settings.get("graph_expansion", True):
                    enrichment_block = rag_service.expander.expand(query, trimmed_chunks)

                time.perf_counter() - t_stage5_start
                stage5_metrics = collector.get_metrics()

                # Merge Stage 3 + 5 metrics
                s3_metrics = res["metrics"]
                for component, data in stage5_metrics["components"].items():
                    s3_metrics["components"][component]["calls"] += data["calls"]
                    s3_metrics["components"][component]["time_sec"] = round(s3_metrics["components"][component]["time_sec"] + data["time_sec"], 4)
                s3_metrics["total_io_calls"] += stage5_metrics["total_io_calls"]

                total_latency = sum(comp["time_sec"] for comp in s3_metrics["components"].values())

                # Format retrieved chunks
                chunks_info = []
                for chunk, score in final_chunks:
                    chunks_info.append({
                        "id": chunk.id,
                        "paper_id": chunk.paper_id,
                        "page_number": chunk.page_number,
                        "text_content": chunk.text_content.strip(),
                        "score": round(score, 4)
                    })

                retrieved_papers = list({c.paper_id for c, _ in final_chunks})

                # Shannon stage inputs: pre-rerank scores (Stage 3 RRF) and pre-trim context
                stage3_candidates = res.get("candidates") or []
                stage3_rrf = res.get("rrf_scores") or {}
                pre_rerank_scores = [
                    float(stage3_rrf.get(c.id, 0.0)) for c in stage3_candidates
                ]
                # Prefer structured relations captured during build_context
                graph_relations = list(getattr(rag_service, "_last_graph_relations", None) or [])
                if not graph_relations and context_graph:
                    try:
                        from core.shannon_estimator import parse_graph_relations_from_text
                        graph_relations = parse_graph_relations_from_text(context_graph)
                    except Exception:
                        graph_relations = []

                trace = traces_map.get((query, baseline))
                if trace:
                    final_pids = list(set(c[0].paper_id if isinstance(c, tuple) else c.paper_id for c in trimmed_chunks))
                    trace["final_context_paper_id_list"] = final_pids
                    tokens = rag_service.llm_engine.count_tokens(trimmed_text + trimmed_graph)
                    if not isinstance(tokens, (int, float)):
                        tokens = (len(trimmed_text) + len(trimmed_graph)) // 4
                    trace["final_context_token_count"] = tokens
                    graph_neighbors = trace.get("graph_neighbor_paper_id_list", [])
                    trace["whether_graph_neighbor_chunk_survived_into_final_context"] = any(
                        (c[0].paper_id if isinstance(c, tuple) else c.paper_id) in graph_neighbors for c in trimmed_chunks
                    )

                contexts_to_save[case_id]["baselines"][baseline] = {
                    "status": "success",
                    "latency_sec": round(total_latency, 3),
                    "retrieved_papers": retrieved_papers,
                    "retrieved_chunks": chunks_info,
                    "pre_rerank_scores": pre_rerank_scores,
                    "context_text": context_text,
                    "context_graph": context_graph,
                    "graph_relations": graph_relations,
                    "trimmed_text": trimmed_text,
                    "trimmed_graph": trimmed_graph,
                    "enrichment_block": enrichment_block,
                    "metrics": s3_metrics,
                    "trace": trace
                }
            except Exception as e:
                con.error(f"Stage 5 failed for baseline {baseline}, query '{query[:30]}': {e}")
                trace = traces_map.get((query, baseline))
                contexts_to_save[case_id]["baselines"][baseline] = {
                    "status": "error",
                    "latency_sec": 0.0,
                    "retrieved_papers": [],
                    "retrieved_chunks": [],
                    "trimmed_text": "",
                    "trimmed_graph": "",
                    "enrichment_block": "",
                    "metrics": res["metrics"],
                    "trace": trace
                }
            finally:
                collector.stop()
                for k, v in orig_components.items():
                    config.data["rag_components"][k] = v
                config.data["llm"]["hyde_enabled"] = orig_hyde
                stage5_done += 1
                prog_units = retrieval_total if stage5_done == retrieval_total else int((0.70 + 0.30 * (stage5_done / retrieval_total)) * retrieval_total)
                print(format_progress_marker("retrieval", prog_units, retrieval_total), flush=True)

    # Unload Reranker at the end of Stage 5
    if has_reranker_baselines:
        rag_service._reranker = None
        gc.collect()
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        con.success("Reranker model unloaded and GPU cache cleared")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(list(contexts_to_save.values()), f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Also save to original output path if unique dir was used
    if not getattr(args, "no_unique_dir", False):
        try:
            original_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(original_output_path, "w", encoding="utf-8") as f:
                yaml.dump(list(contexts_to_save.values()), f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as e:
            con.warning(f"Could not save copy to original output path: {e}")

    con.success(f"Stage transition complete. Retrieved contexts saved to: {output_path.resolve()}")

    if not getattr(args, "no_unique_dir", False):
        try:
            import shutil
            original_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, original_output_path)
            con.success(f"Copied output file to: {original_output_path.resolve()}")
        except Exception as e:
            con.warning(f"Could not copy output file to {original_output_path}: {e}")


def evaluate_and_compare(results_file: Path, artifact_dir: Path = None) -> dict:
    """Loads results, computes retrieval metrics (Recall & Precision),
    prints comparison table and saves a Markdown report.
    """
    with open(results_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        con.error("No retrieval data found in results file.")
        return {}

    metrics_summary = {}

    for case in data:
        expected = case.get("expected_papers", [])
        baselines = case.get("baselines", {})

        for baseline_name, b_data in baselines.items():
            if baseline_name not in metrics_summary:
                metrics_summary[baseline_name] = {
                    "recalls": [],
                    "precisions": [],
                    "latencies": [],
                    "success_count": 0,
                    "total_count": 0
                }

            stats = metrics_summary[baseline_name]
            stats["total_count"] += 1

            if b_data.get("status") == "success":
                stats["success_count"] += 1
                retrieved_papers = b_data.get("retrieved_papers", [])
                retrieved_chunks = b_data.get("retrieved_chunks", [])
                latency = b_data.get("latency_sec", 0.0)

                recall = calculate_retrieval_recall(expected, retrieved_papers)
                precision = calculate_context_precision(expected, retrieved_chunks)

                stats["recalls"].append(recall)
                stats["precisions"].append(precision)
                stats["latencies"].append(latency)
            else:
                stats["recalls"].append(0.0)
                stats["precisions"].append(0.0)
                stats["latencies"].append(0.0)

    con.info("\n=== RETRIEVAL STAGE BENCHMARK COMPARISON ===")

    header_fmt = "| {:<15} | {:<12} | {:<12} | {:<17} | {:<13} |"
    row_fmt = "| {:<15} | {:<12} | {:<12.4f} | {:<17.4f} | {:<12.3f}s |"
    sep = "+" + "-"*17 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*19 + "+" + "-"*15 + "+"

    print(sep)
    print(header_fmt.format("Baseline", "Success Rate", "Mean Recall", "Mean Precision", "Mean Latency"))
    print(sep)

    for b_name in sorted(metrics_summary.keys()):
        stats = metrics_summary[b_name]
        total = stats["total_count"]
        success_rate = (stats["success_count"] / total * 100) if total > 0 else 0.0

        mean_recall = sum(stats["recalls"]) / len(stats["recalls"]) if stats["recalls"] else 0.0
        mean_prec = sum(stats["precisions"]) / len(stats["precisions"]) if stats["precisions"] else 0.0
        mean_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0

        b_label = b_name
        if b_name == "CUSTOM":
            b_label = "CUSTOM (Ours)"

        print(row_fmt.format(b_label, f"{success_rate:.1f}%", mean_recall, mean_prec, mean_lat))
    print(sep)
    print()

    return metrics_summary


def save_custom_retrieval_report(metrics_summary: dict, custom_comp: dict, custom_hype: dict, report_path: Path):
    """Saves a rich Markdown report showing configuration diffs and metrics comparison."""
    import core.config
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RAG Custom Retrieval Benchmark Report\n\n")
        f.write("This report displays the retrieval stage performance of your custom configuration compared against active baselines.\n\n")

        f.write("## ⚙️ Custom Run Configuration Overrides\n\n")

        b6_comp = core.config.get_baseline_config("B6", config.rag_components)
        f.write("### Component Settings (vs B6 Full Pipeline)\n\n")
        f.write("| Component | Custom Value | B6 Default | Status |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for k in sorted(custom_comp.keys()):
            custom_val = custom_comp[k]
            b6_val = b6_comp.get(k)
            status = "🟢 **Modified**" if custom_val != b6_val else "Unchanged"
            f.write(f"| `{k}` | `{custom_val}` | `{b6_val}` | {status} |\n")
        f.write("\n")

        f.write("### Hyperparameter Overrides (vs System Defaults)\n\n")
        f.write("| Parameter | Custom Value | Default Value | Status |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")

        has_hype_overrides = False
        for section in sorted(custom_hype.keys()):
            for k in sorted(custom_hype[section].keys()):
                custom_val = custom_hype[section][k]
                def_val = DEFAULT_HYPERPARAMS.get(section, {}).get(k)
                if custom_val != def_val:
                    has_hype_overrides = True
                    f.write(f"| `{section}.{k}` | `{custom_val}` | `{def_val}` | ⚡ **Overridden** |\n")

        if not has_hype_overrides:
            f.write("| *None* | | | | \n")
        f.write("\n\n")

        f.write("## 📊 Retrieval Performance Summary\n\n")
        f.write("| Baseline | Success Rate | Mean Recall | Context Precision | Mean Latency |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")

        for b_name in sorted(metrics_summary.keys()):
            stats = metrics_summary[b_name]
            total = stats["total_count"]
            success_rate = (stats["success_count"] / total * 100) if total > 0 else 0.0
            mean_recall = sum(stats["recalls"]) / len(stats["recalls"]) if stats["recalls"] else 0.0
            mean_prec = sum(stats["precisions"]) / len(stats["precisions"]) if stats["precisions"] else 0.0
            mean_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0

            b_label = b_name
            if b_name == "CUSTOM":
                b_label = "🏆 **CUSTOM (Ours)**"

            f.write(f"| {b_label} | {success_rate:.1f}% | {mean_recall:.4f} | {mean_prec:.4f} | {mean_lat:.3f}s |\n")

        f.write("\n\n")
        f.write("> [!NOTE]\n")
        f.write("> - **Retrieval Recall**: proportion of expected papers retrieved.\n")
        f.write("> - **Context Precision**: Mean Average Precision of the retrieved chunks/papers.\n")


save_markdown_report = save_custom_retrieval_report

