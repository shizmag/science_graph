import time
import logging
import threading
import numpy as np
import difflib
from typing import Optional, Dict, List, Tuple
from src.models import slugify
from src.repository.base import GraphRepository
from src.vector_search import EmbeddingEngine

class EntityResolver:
    """Service to handle caching and resolution of entities."""

    def __init__(
        self, graph_repo: GraphRepository, emb_engine: EmbeddingEngine
    ) -> None:
        """Initialize the EntityResolver with repository and embedding engine."""
        self.graph_repo = graph_repo
        self.emb_engine = emb_engine
        self._aliases_cache: Optional[Dict[str, str]] = None
        self._entity_cache: Dict[
            str, List[Tuple[str, str, str, Optional[np.ndarray], Optional[float]]]
        ] = {}
        self._lock = threading.Lock()

    def _prepare_cache_item(
        self,
        entity_id: str,
        name: str,
        embedding: Optional[List[float]]
    ) -> Tuple[str, str, str, Optional[np.ndarray], Optional[float]]:
        """Precompute and parse entity properties for efficient caching."""
        name_slug = slugify(name) if name else ""
        emb_arr = None
        norm = None
        if embedding is not None and (
            isinstance(embedding, list) or isinstance(embedding, np.ndarray)
        ):
            try:
                emb_arr = np.array(embedding, dtype=np.float32)
                n = float(np.linalg.norm(emb_arr))
                if n > 0:
                    norm = n
                else:
                    emb_arr = None
            except ValueError:
                emb_arr = None
        return (entity_id, name, name_slug, emb_arr, norm)

    def invalidate_cache(self, label: Optional[str] = None) -> None:
        """Invalidate cache for a specific label, or clear all caches."""
        with self._lock:
            if label is None:
                self._entity_cache.clear()
                self._aliases_cache = None
            else:
                if label in self._entity_cache:
                    del self._entity_cache[label]
                if label == "Concept":
                    self._aliases_cache = None

    def invalidate_concept_cache(self) -> None:
        """Invalidate the concept cache for backward compatibility."""
        self.invalidate_cache("Concept")

    def resolve_entity(self, label: str, name: Optional[str]) -> str:
        """Resolve entity name by aliases, slug match, or vector/string similarity."""
        if not name:
            return ""

        name_clean = name.strip()
        slug = slugify(name_clean)
        t0 = time.perf_counter()

        if label == "Concept":
            aliases_map = self._aliases_cache
            if aliases_map is None:
                with self._lock:
                    aliases_map = self._aliases_cache
                    if aliases_map is None:
                        try:
                            self._aliases_cache = (
                                self.graph_repo.get_concept_aliases()
                            )
                        except Exception as e:
                            logging.error(
                                f"Error fetching concept aliases: {e}"
                            )
                            raise
                        aliases_map = self._aliases_cache
            if name_clean.lower() in aliases_map:
                canonical = aliases_map[name_clean.lower()]
                logging.debug(
                    f"resolve_entity '{name}' resolved from aliases_map in "
                    f"{time.perf_counter() - t0:.6f}s"
                )
                return slugify(canonical)

        with self._lock:
            if label not in self._entity_cache:
                try:
                    nodes = self.graph_repo.get_nodes_by_label(label)
                except Exception as e:
                    logging.error(f"Error fetching nodes for label {label}: {e}")
                    raise
                cached_list = []
                for eid, props in nodes:
                    name_val = props.get("name", "")
                    emb_val = props.get("embedding")
                    item = self._prepare_cache_item(eid, name_val, emb_val)
                    cached_list.append(item)
                self._entity_cache[label] = cached_list
            existing_nodes = self._entity_cache[label]

        for eid, _, name_slug, _, _ in existing_nodes:
            if eid == slug or name_slug == slug:
                return eid

        valid_candidates = []
        for eid, _, _, node_emb, node_norm in existing_nodes:
            if node_emb is not None and node_norm is not None:
                valid_candidates.append((eid, node_emb, node_norm))

        if valid_candidates:
            try:
                candidate_emb = self.emb_engine.get_embedding(name_clean)
            except Exception as e:
                logging.debug(f"Failed to get query embedding: {e}")
                candidate_emb = None

            if candidate_emb is not None and (
                isinstance(candidate_emb, list)
                or isinstance(candidate_emb, np.ndarray)
            ):
                try:
                    query_vec = np.array(candidate_emb, dtype=np.float32)
                    query_norm = float(np.linalg.norm(query_vec))
                    if query_norm > 0:
                        node_embs = np.array(
                            [emb for _, emb, _ in valid_candidates],
                            dtype=np.float32
                        )
                        node_norms = np.array(
                            [norm for _, _, norm in valid_candidates],
                            dtype=np.float32
                        )
                        dots = np.dot(node_embs, query_vec)
                        norms_product = query_norm * node_norms

                        sims = np.zeros_like(dots)
                        valid_mask = norms_product > 0
                        sims[valid_mask] = (
                            dots[valid_mask] / norms_product[valid_mask]
                        )

                        if len(sims) > 0:
                            best_idx = int(np.argmax(sims))
                            if sims[best_idx] > 0.95:
                                return valid_candidates[best_idx][0]
                except ValueError as ve:
                    logging.warning(
                        "Embedding dimension mismatch or calculation "
                        f"error in EntityResolver: {ve}"
                    )

        best_ratio = 0.0
        best_eid = None
        for eid, node_name, _, _, _ in existing_nodes:
            if node_name:
                ratio = difflib.SequenceMatcher(
                    None, name_clean.lower(), node_name.lower()
                ).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_eid = eid
        if best_ratio > 0.95 and best_eid is not None:
            return best_eid

        return slug

    def add_resolved_entity_to_cache(
        self,
        label: str,
        entity_id: str,
        name: str,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add a resolved entity to the local cache in a thread-safe manner."""
        with self._lock:
            if label not in self._entity_cache:
                try:
                    nodes = self.graph_repo.get_nodes_by_label(label)
                except Exception as e:
                    logging.error(f"Error fetching nodes for label {label}: {e}")
                    raise
                cached_list = []
                for eid, props in nodes:
                    name_val = props.get("name", "")
                    emb_val = props.get("embedding")
                    item = self._prepare_cache_item(eid, name_val, emb_val)
                    cached_list.append(item)
                self._entity_cache[label] = cached_list

            exists = any(
                eid == entity_id for eid, _, _, _, _ in self._entity_cache[label]
            )
            if not exists:
                item = self._prepare_cache_item(entity_id, name, embedding)
                new_list = list(self._entity_cache[label])
                new_list.append(item)
                self._entity_cache[label] = new_list
                if label == "Concept":
                    self._aliases_cache = None
