import time
from typing import Dict, List, Any, Tuple

class BenchmarkStatsCollector:
    """Interceptors framework to dynamically wrap RAGService and repository methods.
    
    Measures performance metrics (number of calls and execution latency) of critical components:
    embeddings, vector searches, DB lookups, reranking, graph operations, LLM generations, etc.
    """
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.stats: Dict[str, Dict[str, Any]] = {
            "embedding": {"calls": 0, "time_sec": 0.0},
            "dense_retrieval": {"calls": 0, "time_sec": 0.0},
            "lexical_retrieval": {"calls": 0, "time_sec": 0.0},
            "graph_neighbors": {"calls": 0, "time_sec": 0.0},
            "db_lookups": {"calls": 0, "time_sec": 0.0},
            "reranking": {"calls": 0, "time_sec": 0.0},
            "graph_expansion": {"calls": 0, "time_sec": 0.0},
            "llm_generation": {"calls": 0, "time_sec": 0.0},
            "citation_repair": {"calls": 0, "time_sec": 0.0},
        }
        self.interceptors: List[Tuple[Any, str, Any]] = []
        self.prompt_tokens = 0

    def wrap_method(self, obj: Any, method_name: str, key: str) -> None:
        """Wraps a specific method on an object to increment execution count and sum time."""
        if not obj or not hasattr(obj, method_name):
            return
        orig_method = getattr(obj, method_name)
        stats = self.stats
        
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            stats[key]["calls"] += 1
            if key == "llm_generation":
                prompt = args[0] if args else kwargs.get("prompt", "")
                try:
                    self.prompt_tokens = self.rag_service.llm_engine.count_tokens(prompt)
                except Exception:
                    self.prompt_tokens = len(prompt) // 4
            try:
                return orig_method(*args, **kwargs)
            finally:
                stats[key]["time_sec"] += time.perf_counter() - t0
                
        wrapper.__wrapped__ = orig_method
        setattr(obj, method_name, wrapper)
        self.interceptors.append((obj, method_name, orig_method))

    def start(self) -> None:
        """Starts intercepting and collecting metrics by wrapping target service methods."""
        # 1. Embeddings
        if hasattr(self.rag_service, "emb_engine") and self.rag_service.emb_engine is not None:
            self.wrap_method(self.rag_service.emb_engine, "get_embedding", "embedding")
        
        # 2. Dense retrieval
        if hasattr(self.rag_service, "vector_repo") and self.rag_service.vector_repo is not None:
            self.wrap_method(self.rag_service.vector_repo, "search_similar_chunks", "dense_retrieval")
        
        # 3. Lexical retrieval
        if hasattr(self.rag_service, "vector_repo") and self.rag_service.vector_repo is not None:
            self.wrap_method(self.rag_service.vector_repo, "search_text_fts5", "lexical_retrieval")
        
        # 4. Graph neighbors
        if hasattr(self.rag_service, "graph_repo") and self.rag_service.graph_repo is not None:
            self.wrap_method(self.rag_service.graph_repo, "get_neighbors", "graph_neighbors")
        
        # 5. DB lookups
        if hasattr(self.rag_service, "graph_repo") and self.rag_service.graph_repo is not None:
            for db_method in ["get_paper", "get_author", "get_concept", "get_papers_batch"]:
                self.wrap_method(self.rag_service.graph_repo, db_method, "db_lookups")
            
        # 6. Reranking wrapper
        if hasattr(self.rag_service, "_get_reranker"):
            # If reranker is already loaded, wrap it
            if getattr(self.rag_service, "_reranker", None) is not None:
                reranker = self.rag_service._reranker
                if not hasattr(reranker.predict, "__wrapped_by_bench__"):
                    orig_predict = reranker.predict
                    def wrapped_predict(*p_args, **p_kwargs):
                        t0 = time.perf_counter()
                        self.stats["reranking"]["calls"] += 1
                        try:
                            return orig_predict(*p_args, **p_kwargs)
                        finally:
                            self.stats["reranking"]["time_sec"] += time.perf_counter() - t0
                    wrapped_predict.__wrapped_by_bench__ = True
                    reranker.predict = wrapped_predict
                    self.interceptors.append((reranker, "predict", orig_predict))

            orig_get_reranker = self.rag_service._get_reranker
            stats = self.stats
            def wrapped_get_reranker(*args, **kwargs):
                reranker = orig_get_reranker(*args, **kwargs)
                if reranker and not hasattr(reranker.predict, "__wrapped_by_bench__"):
                    orig_predict = reranker.predict
                    def wrapped_predict(*p_args, **p_kwargs):
                        t0 = time.perf_counter()
                        stats["reranking"]["calls"] += 1
                        try:
                            return orig_predict(*p_args, **p_kwargs)
                        finally:
                            stats["reranking"]["time_sec"] += time.perf_counter() - t0
                    wrapped_predict.__wrapped_by_bench__ = True
                    reranker.predict = wrapped_predict
                return reranker
            self.rag_service._get_reranker = wrapped_get_reranker
            self.interceptors.append((self.rag_service, "_get_reranker", orig_get_reranker))
        
        # 7. LLM generation
        if hasattr(self.rag_service, "llm_engine") and self.rag_service.llm_engine is not None:
            self.wrap_method(self.rag_service.llm_engine, "generate_response", "llm_generation")
            if hasattr(self.rag_service.llm_engine, "generate_response_with_logits"):
                self.wrap_method(self.rag_service.llm_engine, "generate_response_with_logits", "llm_generation")
        
        # 8. Citation repair
        self.wrap_method(self.rag_service, "_validate_and_repair_citations", "citation_repair")
        
        # 9. Expander wrapper
        if getattr(self.rag_service, "expander", None) is not None:
            expander = self.rag_service.expander
            if hasattr(expander, "expand"):
                self.wrap_method(expander, "expand", "graph_expansion")

    def reset(self) -> None:
        """Resets the statistics count without removing the interceptor wrappers."""
        for key in self.stats:
            self.stats[key]["calls"] = 0
            self.stats[key]["time_sec"] = 0.0
        self.prompt_tokens = 0

    def stop(self) -> None:
        """Restores original non-wrapped methods to all intercepted objects."""
        for obj, method_name, orig_method in reversed(self.interceptors):
            setattr(obj, method_name, orig_method)
        self.interceptors.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Formats and returns the collected metrics rounded to 4 decimals."""
        rounded_components = {}
        total_calls = 0
        for k, v in self.stats.items():
            rounded_components[k] = {
                "calls": v["calls"],
                "time_sec": round(v["time_sec"], 4)
            }
            total_calls += v["calls"]
        return {
            "components": rounded_components,
            "total_io_calls": total_calls,
            "prompt_tokens": self.prompt_tokens
        }
