"""Dependency-keyed caches for shared, deterministic stages in B6 arms."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import OrderedDict
from typing import Any

from ablation_cuda.runtime.checkpoints import CheckpointStore, stable_input_hash


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def _role() -> str:
    return ("experimental_graph_expander" if any(
        frame.filename.replace("\\", "/").endswith("/src/services/graph_expander.py")
        for frame in inspect.stack(context=0)
    ) else "final_candidate_reranking")


class StageCache:
    """Stores reusable scores and search candidates under complete dependency keys."""

    def __init__(self, store: CheckpointStore, *, model_manifest: dict[str, Any],
                 snapshot_sha256: str, telemetry_version: str):
        self.store = store
        self.model_manifest = model_manifest
        self.snapshot_sha256 = snapshot_sha256
        self.telemetry_version = telemetry_version
        self.generation_memory: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.generation_memory_limit = 8

    def install(self, service: Any) -> None:
        service._ablation_stage_cache = self
        service._ablation_cache_events = []
        self._wrap_vector_search(service)
        self._wrap_graph_neighbors(service)
        self._wrap_cross_encoder(service)
        self._wrap_generation(service)

    def _event(self, service: Any, stage: str, cache_key: str, *, count: int = 1,
               hit: bool = True) -> None:
        service._ablation_cache_events.append({"stage": stage, "cache_key": cache_key,
                                               "items": count, "hit": hit})

    def get_query_embedding(self, query: str) -> list[float] | None:
        dependency = {
            "query_utf8_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "embedding_model": self.model_manifest["models"]["embedder"]["model_id"],
            "embedding_revision": self.model_manifest["models"]["embedder"]["revision"],
            "preprocessing": "E5 query: prefix, normalized_embeddings=false",
            "source_code_sha256": self.telemetry_version,
        }
        return self.store.cache_get("query_embedding", dependency)

    def put_query_embedding(self, query: str, embedding: list[float]) -> str:
        dependency = {
            "query_utf8_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "embedding_model": self.model_manifest["models"]["embedder"]["model_id"],
            "embedding_revision": self.model_manifest["models"]["embedder"]["revision"],
            "preprocessing": "E5 query: prefix, normalized_embeddings=false",
            "source_code_sha256": self.telemetry_version,
        }
        return self.store.cache_put("query_embedding", dependency, embedding)

    def seed_generation_events(self, events: list[dict[str, Any]]) -> None:
        for item in events:
            event = item.get("event", {})
            if item.get("purpose") and item.get("purpose") != "final_answer_generation":
                # Diagnostic prompts may be reused too, but are less likely to
                # match and can displace final answers from the bounded cache.
                pass
            self._remember_generation(event, item.get("token_telemetry", []))

    def _wrap_vector_search(self, service: Any) -> None:
        repo = service.vector_repo
        if getattr(repo, "_ablation_cache_installed", False):
            return
        dense = repo.search_similar_chunks
        lexical = repo.search_text_fts5

        def materialize(rows: list[dict[str, Any]]):
            by_id = repo.get_chunks_by_ids([item["chunk_id"] for item in rows])
            return [(by_id[item["chunk_id"]], float(item["score"]))
                    for item in rows if item["chunk_id"] in by_id]

        def cached_dense(query_embedding, limit=5, filters=None):
            vector_bytes = bytes(memoryview(__import__("numpy").asarray(query_embedding, dtype="float32")))
            dependencies = {
                "snapshot_sha256": self.snapshot_sha256,
                "query_embedding_sha256": hashlib.sha256(vector_bytes).hexdigest(),
                "limit": int(limit), "filters": filters or {},
                "database_stage": "usearch-cosine-candidate-search-v1",
                "stage_implementation_sha256": self.telemetry_version,
            }
            key = stable_input_hash(dependencies)
            saved = self.store.cache_get("dense_candidates", dependencies)
            if isinstance(saved, list):
                self._event(service, "dense_candidates", key, count=len(saved))
                return materialize(saved)
            result = dense(query_embedding, limit=limit, filters=filters)
            compact = [{"chunk_id": str(chunk.id), "score": float(score)} for chunk, score in result]
            self._event(service, "dense_candidates", key, count=max(1, len(compact)), hit=False)
            self.store.cache_put("dense_candidates", dependencies, compact)
            return result

        def cached_lexical(query, limit=10, filters=None):
            dependencies = {
                "snapshot_sha256": self.snapshot_sha256,
                "query_utf8_sha256": hashlib.sha256(str(query).encode("utf-8")).hexdigest(),
                "limit": int(limit), "filters": filters or {},
                "database_stage": "sqlite-fts5-candidates-v1",
                "stage_implementation_sha256": self.telemetry_version,
            }
            key = stable_input_hash(dependencies)
            saved = self.store.cache_get("lexical_candidates", dependencies)
            if isinstance(saved, list):
                self._event(service, "lexical_candidates", key, count=len(saved))
                return materialize(saved)
            result = lexical(query, limit=limit, filters=filters)
            compact = [{"chunk_id": str(chunk.id), "score": float(score)} for chunk, score in result]
            self._event(service, "lexical_candidates", key, count=max(1, len(compact)), hit=False)
            self.store.cache_put("lexical_candidates", dependencies, compact)
            return result

        repo.search_similar_chunks = cached_dense
        repo.search_text_fts5 = cached_lexical
        repo._ablation_cache_installed = True

    def _wrap_graph_neighbors(self, service: Any) -> None:
        repo = service.graph_repo
        if getattr(repo, "_ablation_cache_installed", False):
            return
        original = repo.get_neighbors

        def cached_neighbors(node_id, max_depth=1, *args, **kwargs):
            dependencies = {
                "snapshot_sha256": self.snapshot_sha256,
                "node_id": str(node_id), "max_depth": int(max_depth),
                "args": list(args), "kwargs": kwargs,
                "graph_traversal": "sqlite-neighbors-v1",
                "stage_implementation_sha256": self.telemetry_version,
            }
            key = stable_input_hash(dependencies)
            saved = self.store.cache_get("graph_candidates", dependencies)
            if isinstance(saved, list):
                self._event(service, "graph_candidates", key, count=len(saved))
                return saved
            result = original(node_id, max_depth, *args, **kwargs)
            # Repository rows contain strings, numeric weights and JSON text.
            compact = [list(row) for row in result]
            self._event(service, "graph_candidates", key, count=max(1, len(compact)), hit=False)
            if _jsonable(compact):
                self.store.cache_put("graph_candidates", dependencies, compact)
            return result

        repo.get_neighbors = cached_neighbors
        repo._ablation_cache_installed = True

    def _wrap_cross_encoder(self, service: Any) -> None:
        reranker = service._get_reranker()
        if getattr(reranker, "_ablation_cache_installed", False):
            return
        original = reranker.predict

        def cached_predict(pairs, *args, **kwargs):
            role = _role()
            simple_args = list(args)
            if not _jsonable(simple_args) or not _jsonable(kwargs):
                return original(pairs, *args, **kwargs)
            keys = []
            values = [None] * len(pairs)
            missing_pairs = []
            missing_indices = []
            for index, pair in enumerate(pairs):
                query, passage = str(pair[0]), str(pair[1])
                dependencies = {
                    "model_id": self.model_manifest["models"]["cross_encoder"]["model_id"],
                    "model_revision": self.model_manifest["models"]["cross_encoder"]["revision"],
                    "query_utf8_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                    "passage_utf8_sha256": hashlib.sha256(passage.encode("utf-8")).hexdigest(),
                    "args": simple_args, "kwargs": kwargs,
                    "score_semantics": "cross-encoder-logit-v1",
                    "stage_implementation_sha256": self.telemetry_version,
                }
                key = stable_input_hash(dependencies)
                keys.append((key, dependencies))
                saved = self.store.cache_get("cross_encoder_pair_score", dependencies)
                if isinstance(saved, dict) and isinstance(saved.get("score"), (int, float)):
                    values[index] = float(saved["score"])
                    self._event(service, "cross_encoder_pair_score", key)
                else:
                    missing_indices.append(index)
                    missing_pairs.append(pair)
            if missing_pairs:
                scores = list(original(missing_pairs, *args, **kwargs))
                if len(scores) != len(missing_pairs):
                    raise RuntimeError("Cross-Encoder cache wrapper received a different score count")
                for index, score in zip(missing_indices, scores):
                    values[index] = float(score)
                    key, dependencies = keys[index]
                    self.store.cache_put("cross_encoder_pair_score", dependencies,
                                         {"score": float(score)})
                self._event(service, "cross_encoder_pair_score",
                            stable_input_hash([key for key, _ in keys]),
                            count=len(missing_pairs), hit=False)
            cached_pair_count = len(pairs) - len(missing_indices)
            if cached_pair_count:
                self._event(service, "cross_encoder_pair_score",
                            stable_input_hash([key for key, _ in keys]),
                            count=cached_pair_count, hit=True)
            if missing_indices and len(missing_indices) != len(pairs):
                service._cuda_cross_encoder_calls.append({
                    "role": role, "pair_count": len(pairs) - len(missing_indices),
                    "cache_hit": True, "cache_hit_count": len(pairs) - len(missing_indices),
                    "inference_executed": False, "input_device_set": [],
                    "score_order_preserved": True,
                })
            elif not missing_indices:
                service._cuda_cross_encoder_calls.append({
                    "role": role, "pair_count": len(pairs), "cache_hit": True,
                    "cache_hit_count": len(pairs), "inference_executed": False,
                    "input_device_set": [], "score_order_preserved": True,
                })
            return values

        reranker.predict = cached_predict
        reranker._ablation_cache_installed = True

    def get_final_ranking(self, *, candidates: list[Any], cross_encoder_scores: list[float],
                          rrf_scores: dict[str, float], score_blending: bool,
                          reranker_weight: float, rrf_weight: float,
                          deduplicate_parent_chunks: bool, limit: int) -> dict[str, Any] | None:
        """Reuse only an identical, fully determined final ranking decision."""
        dependencies = self._ranking_dependencies(
            candidates=candidates, cross_encoder_scores=cross_encoder_scores,
            rrf_scores=rrf_scores, score_blending=score_blending,
            reranker_weight=reranker_weight, rrf_weight=rrf_weight,
            deduplicate_parent_chunks=deduplicate_parent_chunks, limit=limit,
        )
        saved = self.store.cache_get("final_ranking", dependencies)
        if not isinstance(saved, dict):
            return None
        ids = [str(item.id) for item in candidates]
        if saved.get("candidate_ids") != ids or len(saved.get("raw_scores", [])) != len(ids):
            return None
        saved["cache_key"] = stable_input_hash(dependencies)
        return saved

    def put_final_ranking(self, *, candidates: list[Any], cross_encoder_scores: list[float],
                          rrf_scores: dict[str, float], score_blending: bool,
                          reranker_weight: float, rrf_weight: float,
                          deduplicate_parent_chunks: bool, limit: int,
                          value: dict[str, Any]) -> str:
        dependencies = self._ranking_dependencies(
            candidates=candidates, cross_encoder_scores=cross_encoder_scores,
            rrf_scores=rrf_scores, score_blending=score_blending,
            reranker_weight=reranker_weight, rrf_weight=rrf_weight,
            deduplicate_parent_chunks=deduplicate_parent_chunks, limit=limit,
        )
        return self.store.cache_put("final_ranking", dependencies, value)

    def _ranking_dependencies(self, *, candidates: list[Any], cross_encoder_scores: list[float],
                              rrf_scores: dict[str, float], score_blending: bool,
                              reranker_weight: float, rrf_weight: float,
                              deduplicate_parent_chunks: bool, limit: int) -> dict[str, Any]:
        candidate_rows = []
        for candidate, ce_score in zip(candidates, cross_encoder_scores):
            candidate_rows.append({
                "chunk_id": str(candidate.id),
                "text_sha256": hashlib.sha256(
                    str(getattr(candidate, "text_content", "") or "").encode("utf-8")
                ).hexdigest(),
                "cross_encoder_score": float(ce_score),
                "rrf_score": float(rrf_scores.get(candidate.id, 0.0)),
            })
        if len(candidate_rows) != len(candidates):
            raise ValueError("Final ranking cache requires one Cross-Encoder score per candidate")
        return {
            "snapshot_sha256": self.snapshot_sha256,
            "candidate_rows_in_order": candidate_rows,
            "score_blending": bool(score_blending),
            "reranker_weight": float(reranker_weight),
            "rrf_weight": float(rrf_weight),
            "deduplicate_parent_chunks": bool(deduplicate_parent_chunks),
            "limit": int(limit),
            "ranking_implementation": "normalized-cross-encoder-rrf-blend-v1",
            "stage_implementation_sha256": self.telemetry_version,
        }

    def _generation_key(self, event_or_prompt: dict[str, Any] | str,
                        *, max_tokens: Any = None, temp: Any = None) -> str:
        if isinstance(event_or_prompt, dict):
            prompt = str(event_or_prompt.get("prompt", ""))
            max_tokens = event_or_prompt.get("max_new_tokens", max_tokens)
            temp = event_or_prompt.get("temperature", temp)
        else:
            prompt = str(event_or_prompt)
        return stable_input_hash({
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "generator_model": self.model_manifest["models"]["generator"]["model_id"],
            "generator_revision": self.model_manifest["models"]["generator"]["revision"],
            "tokenizer_revision": self.model_manifest["models"]["generator"]["revision"],
            "max_new_tokens": max_tokens, "temperature": temp,
            "stop_token_ids": [151643, 151645, 151683],
            "telemetry_implementation": self.telemetry_version,
        })

    def _remember_generation(self, event: dict[str, Any], telemetry: list[dict[str, Any]]) -> None:
        prompt = event.get("prompt")
        if not isinstance(prompt, str) or not prompt or not telemetry:
            return
        key = self._generation_key(event)
        self.generation_memory[key] = {
            "response": event.get("response", ""),
            "telemetry": telemetry,
            "event": {**event, "token_telemetry": telemetry},
        }
        self.generation_memory.move_to_end(key)
        while len(self.generation_memory) > self.generation_memory_limit:
            self.generation_memory.popitem(last=False)

    def _wrap_generation(self, service: Any) -> None:
        engine = service.llm_engine
        if getattr(engine, "_ablation_generation_cache_installed", False):
            return
        original = engine.generate_response_with_logits

        def cached_generate(prompt, max_tokens=None, temp=None, *args, **kwargs):
            key = self._generation_key(str(prompt), max_tokens=max_tokens, temp=temp)
            cached = self.generation_memory.get(key)
            if cached is not None:
                self.generation_memory.move_to_end(key)
                event = {**cached["event"], "cache_hit": True}
                engine.events.append(event)
                service._ablation_cache_events.append({"stage": "generator_input", "cache_key": key,
                                                       "items": len(cached["telemetry"]), "hit": True})
                return cached["response"], cached["telemetry"]
            service._ablation_cache_events.append({"stage": "generator_input", "cache_key": key,
                                                   "items": 1, "hit": False})
            result = original(prompt, max_tokens=max_tokens, temp=temp, *args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                event = engine.events[-1] if engine.events else {}
                event["cache_hit"] = False
                self._remember_generation(event, result[1])
            return result

        engine.generate_response_with_logits = cached_generate
        engine._ablation_generation_cache_installed = True
