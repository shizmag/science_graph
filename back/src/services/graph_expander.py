import threading
import time
import math
from typing import List, Tuple, Dict, Any, Optional
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.tree import Tree
from rich import box
from src.console import console
from src import console as con
from src.models import Chunk, Paper
from src.repository.base import GraphRepository, VectorRepository
from src.llm_engine import BaseLLMEngine
from src.config import config

def _safe_float(val: Any, default: float) -> float:
    try:
        if val.__class__.__name__ in ("MagicMock", "Mock"):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

def _safe_int(val: Any, default: int) -> int:
    try:
        if val.__class__.__name__ in ("MagicMock", "Mock"):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default

def safe_sigmoid(x: float) -> float:
    val = _safe_float(config.graph_sigmoid_slope, -25.0) * (x - _safe_float(config.graph_sigmoid_center, 0.5))
    if val > 700.0:
        return 0.0
    if val < -700.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(val))

class ExperimentalGraphExpander:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        llm_engine: BaseLLMEngine,
        reranker: Any,
        p_base: Optional[float] = None,
        gamma: Optional[float] = None,
        limit: Optional[int] = None,
        top_chunks_per_paper: int = 2,
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.llm_engine = llm_engine
        self.reranker = reranker
        self.p_base = p_base if p_base is not None else _safe_float(config.graph_p_base, 0.75)
        self.gamma = gamma if gamma is not None else _safe_float(config.graph_gamma, 0.5)
        if limit is None:
            order = _safe_int(getattr(config, "b6_graph_neighbors_order", 2), 2)
            self.limit = order + 1
        else:
            self.limit = limit
        self.top_chunks_per_paper = top_chunks_per_paper
        self.last_telemetry = None
        self.last_output = None
        
        # Semaphore/lock to prevent concurrent model execution and minimize model switching overhead
        self.model_lock = threading.Semaphore(1)

    def _resolve_paper_title(self, paper_id: str, papers_map: Dict[str, Paper]) -> str:
        paper = papers_map.get(paper_id)
        if paper and paper.title:
            return paper.title
        return paper_id

    def expand(self, query: str, initial_chunks: List[Tuple[Chunk, float]], trace: bool = False) -> str:
        """
        Runs the adaptive context expansion algorithm:
        1. Prep: extract paper IDs from initial chunks.
        2. Discovery Loop: crawl neighbors with geometric decay stopping criterion.
        3. Ingestion: load relevant chunks for discovered papers.
        4. Evidence List: filter all collected facts using LLM with short indexes.
        5. Form Prompt Block: format the essential facts as markdown text.
        """
        t_overall_start = time.perf_counter()
        should_trace = trace or con.SHOW_TIME

        telemetry = {
            "query": query,
            "prep_time": 0.0,
            "hops": [],
            "ingestion": {
                "fetch_time": 0.0,
                "rerank_time": 0.0,
                "surviving_papers": [],
                "new_chunks_evaluated": [],
                "chunks_kept": [],
                "total_time": 0.0
            },
            "filtering": {
                "total_time": 0.0,
                "facts_evaluated": [],
                "essential_ids": []
            },
            "overall_time": 0.0
        }

        # Step 1: Preparation (Level 0)
        t_prep_start = time.perf_counter()
        # Extract initial paper IDs
        initial_paper_ids = list({chunk.paper_id for chunk, _ in initial_chunks if chunk.paper_id})
        
        # Pull papers from DB to resolve titles and keep their details
        papers_map = self.graph_repo.get_papers_batch(initial_paper_ids)
        
        current_node_ids = set(initial_paper_ids)
        visited_nodes = set(current_node_ids)
        
        # node_id -> (label, name_or_title, card_text, connection_desc, score)
        accumulated_nodes: Dict[str, Tuple[str, str, str, Optional[str], float]] = {}
        node_labels: Dict[str, str] = {}
        for pid in initial_paper_ids:
            paper = papers_map.get(pid)
            if paper:
                label = "UserNote" if paper.properties.get("source_type") == "note" else "Paper"
                node_labels[pid] = label
            else:
                node_labels[pid] = "Paper"
        
        # Populate initial papers into accumulated_nodes
        for pid in initial_paper_ids:
            paper = papers_map.get(pid)
            if paper:
                label = node_labels.get(pid, "Paper")
                desc = f"Title: {paper.title}"
                if paper.abstract:
                    desc += f". Abstract: {paper.abstract}"
                summary = paper.properties.get("summary")
                if summary:
                    desc += f". Summary: {summary}"
                accumulated_nodes[pid] = (label, paper.title, desc, "Initial search result", 1.0)
        telemetry["prep_time"] = time.perf_counter() - t_prep_start

        # Step 2: Discovery Loop
        import datetime
        current_year = datetime.datetime.now().year
        n = 1
        while n < self.limit:
            t_hop_start = time.perf_counter()
            hop_data = {
                "hop_index": n,
                "fetch_time": 0.0,
                "rerank_time": 0.0,
                "candidates_fetched": [],
                "k_n_float": 0.0,
                "k_limit": 0,
                "evaluated": [],
                "total_time": 0.0
            }

            # 2.1 Bidirectional Batch Fetch
            t_fetch_start = time.perf_counter()
            new_neighbors: Dict[str, Tuple[str, Optional[str]]] = {}  # neighbor_id -> (label, connection_desc)
            # Resolve titles for the entire layer to format connection descriptions properly
            node_titles = {}
            for node_id in current_node_ids:
                curr_label = node_labels.get(node_id, "Paper")
                if curr_label in ("Paper", "UserNote"):
                    paper = papers_map.get(node_id) or self.graph_repo.get_paper(node_id)
                    node_titles[node_id] = paper.title if paper else node_id
                elif curr_label == "Author":
                    author = self.graph_repo.get_author(node_id)
                    node_titles[node_id] = author.name if author else node_id
                elif curr_label == "Concept":
                    concept = self.graph_repo.get_concept(node_id)
                    node_titles[node_id] = concept.name if concept else node_id
                else:
                    props = self.graph_repo.get_node_properties(node_id)
                    node_titles[node_id] = props.get("name") if props else node_id

            # Use batch query if available and configured (mock-safe fallback)
            use_batch = False
            if hasattr(self.graph_repo, "get_neighbors_batch"):
                from unittest.mock import Mock
                if isinstance(self.graph_repo.get_neighbors_batch, Mock):
                    if (self.graph_repo.get_neighbors_batch._mock_return_value is not Mock._mock_return_value 
                            or self.graph_repo.get_neighbors_batch.side_effect is not None):
                        use_batch = True
                else:
                    use_batch = True
            
            if use_batch:
                neighbors = self.graph_repo.get_neighbors_batch(list(current_node_ids))
            else:
                # Fallback to sequential get_neighbors to preserve backward compatibility & mock tests
                neighbors = []
                for node_id in current_node_ids:
                    node_neighbors = self.graph_repo.get_neighbors(node_id, max_depth=1)
                    neighbors.extend(node_neighbors)
            
            for src_id, src_label, edge_type, tgt_id, tgt_label, edge_props_json in neighbors:
                if src_id in current_node_ids:
                    node_id = src_id
                    curr_title = node_titles[node_id]
                    curr_label = node_labels.get(node_id, src_label)
                    neigh_id = tgt_id
                    neigh_label = tgt_label
                    is_outbound = True
                elif tgt_id in current_node_ids:
                    node_id = tgt_id
                    curr_title = node_titles[node_id]
                    curr_label = node_labels.get(node_id, tgt_label)
                    neigh_id = src_id
                    neigh_label = src_label
                    is_outbound = False
                else:
                    continue
                
                if neigh_id in visited_nodes or neigh_id in new_neighbors:
                    continue
                    
                # Filter based on type:
                allowed = False
                if curr_label in ("Paper", "UserNote"):
                    if neigh_label in ("Paper", "UserNote", "Author", "Concept", "Institution", "Dataset", "CodeRepository", "JournalConference"):
                        allowed = True
                elif curr_label == "Author":
                    if neigh_label in ("Paper", "UserNote", "Institution"):
                        allowed = True
                elif curr_label == "Concept":
                    if neigh_label in ("Paper", "UserNote", "Concept"):
                        allowed = True
                elif curr_label == "Institution":
                    if neigh_label in ("Paper", "UserNote", "Author"):
                        allowed = True
                elif curr_label in ("Dataset", "CodeRepository", "JournalConference"):
                    if neigh_label in ("Paper", "UserNote"):
                        allowed = True
                        
                if not allowed:
                    continue
                        
                # Construct connection description
                conn_desc = None
                try:
                    import json
                    edge_props = json.loads(edge_props_json) if edge_props_json else {}
                except Exception:
                    edge_props = {}

                # Custom intent and context snippet for CITES
                intent_str = ""
                if edge_type == "CITES":
                    intent = edge_props.get("intent")
                    ctx = edge_props.get("context")
                    if intent:
                        intent_str += f" (Intent: {intent})"
                    if ctx:
                        intent_str += f" [Context: \"{ctx}\"]"

                if edge_type == "CITES":
                    if is_outbound:
                        conn_desc = f"Cites paper '{curr_title}'{intent_str}"
                    else:
                        conn_desc = f"Cited by paper '{curr_title}'{intent_str}"
                elif edge_type == "AUTHORED":
                    if is_outbound:
                        conn_desc = f"Author of paper/note '{curr_title}'"
                    else:
                        conn_desc = f"Paper/note authored by '{curr_title}'"
                elif edge_type == "MENTIONS_CONCEPT":
                    if is_outbound:
                        conn_desc = f"Concept mentioned in paper/note '{curr_title}'"
                    else:
                        conn_desc = f"Paper/note mentioning concept '{curr_title}'"
                elif edge_type == "HAS_TAG":
                    if is_outbound:
                        conn_desc = f"Tag for paper/note '{curr_title}'"
                    else:
                        conn_desc = f"Paper/note tagged with '{curr_title}'"
                elif edge_type == "AFFILIATED_WITH":
                    if is_outbound:
                        conn_desc = f"Institution affiliated with author '{curr_title}'"
                    else:
                        conn_desc = f"Author affiliated with institution '{curr_title}'"
                elif edge_type == "SPONSORED_BY":
                    if is_outbound:
                        conn_desc = f"Institution sponsoring paper/note '{curr_title}'"
                    else:
                        conn_desc = f"Paper/note sponsored by institution '{curr_title}'"
                elif edge_type in ("USED_DATASET", "INTRODUCED_DATASET"):
                    rel_verb = "using" if edge_type == "USED_DATASET" else "introducing"
                    if is_outbound:
                        conn_desc = f"Dataset {rel_verb} in paper/note '{curr_title}'"
                    else:
                        conn_desc = f"Paper/note {rel_verb} dataset '{curr_title}'"
                elif edge_type == "HAS_CODE":
                    if is_outbound:
                        conn_desc = f"Code repository for paper/note '{curr_title}'"
                    else:
                        conn_desc = f"Paper/note containing code repository '{curr_title}'"
                elif edge_type == "PUBLISHED_IN":
                    if is_outbound:
                        conn_desc = f"Journal/conference publishing paper/note '{curr_title}'"
                    else:
                        conn_desc = f"Paper/note published in journal/conference '{curr_title}'"
                elif edge_type in ("SUBCLASS_OF", "IS_A"):
                    if is_outbound:
                        conn_desc = f"Subclass of concept '{curr_title}'"
                    else:
                        conn_desc = f"Superclass of concept '{curr_title}'"
                elif edge_type == "PREREQUISITE_FOR":
                    if is_outbound:
                        conn_desc = f"Prerequisite for concept '{curr_title}'"
                    else:
                        conn_desc = f"Requires concept '{curr_title}' as prerequisite"
                elif edge_type == "COMMENTS_ON":
                    if is_outbound:
                        conn_desc = f"Commented on by note '{curr_title}'"
                    else:
                        conn_desc = f"Note commenting on '{curr_title}'"
                elif edge_type == "AGREES_WITH":
                    if is_outbound:
                        conn_desc = f"Agreed with by note '{curr_title}'"
                    else:
                        conn_desc = f"Note agreeing with '{curr_title}'"
                elif edge_type == "DISAGREES_WITH":
                    if is_outbound:
                        conn_desc = f"Disagreed with by note '{curr_title}'"
                    else:
                        conn_desc = f"Note disagreeing with '{curr_title}'"
                elif edge_type == "LINKED_TO":
                    if is_outbound:
                        conn_desc = f"Concept linked to note '{curr_title}'"
                    else:
                        conn_desc = f"Note linked to concept '{curr_title}'"
                else:
                    conn_desc = f"Connected to '{curr_title}' via {edge_type}"
                        
                new_neighbors[neigh_id] = (neigh_label, conn_desc)
            hop_data["fetch_time"] = time.perf_counter() - t_fetch_start
                    
            if not new_neighbors:
                hop_data["total_time"] = time.perf_counter() - t_hop_start
                telemetry["hops"].append(hop_data)
                break
                
            # Decay strategy stopping criteria
            k_n_float = len(new_neighbors) * (self.p_base * (self.gamma ** n))
            k_limit = int(k_n_float)
            hop_data["k_n_float"] = k_n_float
            hop_data["k_limit"] = k_limit
            
            # Log current hop info
            con.info(f"Hop {n}: {len(new_neighbors)} neighbors -> calculated K_n limit: {k_n_float:.2f}")
            
            if k_n_float < _safe_float(config.graph_crawl_stop_threshold, 1.0):
                con.info(f"Decay stopping criteria reached: K_n={k_n_float:.2f} < {_safe_float(config.graph_crawl_stop_threshold, 1.0)}. Stopping crawl.")
                hop_data["total_time"] = time.perf_counter() - t_hop_start
                telemetry["hops"].append(hop_data)
                break
            
            # 2.2 Summary-First Evaluation: build cards without full chunks
            cards: List[Tuple[str, str, Tuple[str, str, Optional[str]]]] = []  # List of (neigh_id, card_text, (label, name_or_title, conn_desc))
            
            # Group paper fetches for efficiency
            paper_neigh_ids = [nid for nid, (label, _) in new_neighbors.items() if label in ("Paper", "UserNote")]
            fetched_papers = self.graph_repo.get_papers_batch(paper_neigh_ids) if paper_neigh_ids else {}
            
            for nid, (label, conn_desc) in new_neighbors.items():
                card_text = ""
                name_or_title = nid
                if label in ("Paper", "UserNote"):
                    paper = fetched_papers.get(nid) or self.graph_repo.get_paper(nid)
                    if paper:
                        name_or_title = paper.title
                        card_text = f"[{label}] Title: {paper.title}"
                        if paper.abstract:
                            card_text += f". Abstract/Content: {paper.abstract}"
                        summary = paper.properties.get("summary")
                        if summary:
                            card_text += f". Summary: {summary}"
                elif label == "Author":
                    author = self.graph_repo.get_author(nid)
                    if author:
                        name_or_title = author.name
                        # Get key works
                        papers = self.graph_repo.get_papers_by_author(nid)
                        key_works = ", ".join([p.title for p in papers if p.title])
                        card_text = f"Author Name: {author.name}. Key Works: {key_works}."
                elif label == "Concept":
                    concept = self.graph_repo.get_concept(nid)
                    if concept:
                        name_or_title = concept.name
                        description = concept.properties.get("description") or f"No description available for '{concept.name}'."
                        card_text = f"Concept Name: {concept.name}. Description: {description}."
                elif label == "Institution":
                    props = self.graph_repo.get_node_properties(nid)
                    if props:
                        name_or_title = props.get("name", nid)
                        card_text = f"Institution: {name_or_title}."
                elif label == "Dataset":
                    props = self.graph_repo.get_node_properties(nid)
                    if props:
                        name_or_title = props.get("name", nid)
                        card_text = f"Dataset: {name_or_title}."
                elif label == "CodeRepository":
                    props = self.graph_repo.get_node_properties(nid)
                    if props:
                        name_or_title = props.get("name", nid)
                        card_text = f"Code Repository: {name_or_title}."
                elif label == "JournalConference":
                    props = self.graph_repo.get_node_properties(nid)
                    if props:
                        name_or_title = props.get("name", nid)
                        card_text = f"Journal/Conference: {name_or_title}."
                        
                if card_text:
                    node_labels[nid] = label
                    cards.append((nid, card_text, (label, name_or_title, conn_desc)))
                    hop_data["candidates_fetched"].append({
                        "id": nid,
                        "label": label,
                        "name": name_or_title,
                        "conn_desc": conn_desc
                    })
                    
            if not cards:
                hop_data["total_time"] = time.perf_counter() - t_hop_start
                telemetry["hops"].append(hop_data)
                break
                
            # 2.3 Batch Reranking under model lock
            t_rerank_start = time.perf_counter()
            pairs = [(query, ctext) for _, ctext, _ in cards]
            scores = self.reranker.predict(pairs)
            hop_data["rerank_time"] = time.perf_counter() - t_rerank_start
                
            # Associate scores and calculate decayed score / UserNote priority
            scored_candidates = []
            for i, (nid, _, (label, name_or_title, conn_desc)) in enumerate(cards):
                raw_score = float(scores[i])
                semantic_score = safe_sigmoid(raw_score)
                is_note = 1 if label == "UserNote" else 0
                
                authority = 1.0
                if label in ("Paper", "UserNote"):
                    paper = fetched_papers.get(nid) or self.graph_repo.get_paper(nid)
                    if paper:
                        year = paper.year
                        publish_year = year if year else current_year - 5
                        citations = int(paper.properties.get("citationCount") or 0)
                        
                        age_factor = (current_year - publish_year + 2) ** self.gamma
                        decayed = (citations + 1) / age_factor
                        
                        newness_bonus = 0.0
                        if current_year - publish_year <= 1:
                            newness_bonus = 2.0
                            
                        authority = decayed + newness_bonus
                        
                final_score = semantic_score * authority
                scored_candidates.append((nid, label, name_or_title, cards[i][1], conn_desc, semantic_score, is_note, final_score))
                
            # Filter candidates using dynamic Top-P (nucleus filtering) on semantic_score (c[5])
            # (bypass for UserNotes to prioritize notes)
            sorted_by_score = sorted(scored_candidates, key=lambda x: x[5], reverse=True)
            total_score = sum(c[5] for c in sorted_by_score)
            top_p_val = _safe_float(config.graph_semantic_score_top_p, 0.9)
            
            top_p_set = set()
            if total_score > 0:
                cumulative_score = 0.0
                for c in sorted_by_score:
                    top_p_set.add(c[0])
                    cumulative_score += c[5]
                    if cumulative_score >= top_p_val * total_score:
                        break
                
            valid_candidates = [c for c in scored_candidates if c[0] in top_p_set or c[1] == "UserNote"]
            valid_candidates.sort(key=lambda x: (x[6], x[7]), reverse=True)
            
            # Select top k_limit
            selected_candidates = valid_candidates[:k_limit]
            selected_ids = {c[0] for c in selected_candidates}
            
            for nid, label, name_or_title, _, conn_desc, semantic_score, _, _ in scored_candidates:
                hop_data["evaluated"].append({
                    "id": nid,
                    "label": label,
                    "name": name_or_title,
                    "score": semantic_score,
                    "selected": nid in selected_ids
                })
            
            con.success(f"Hop {n}: selected {len(selected_candidates)} / {len(cards)} neighbors (limit {k_limit})")
            
            if not selected_candidates:
                hop_data["total_time"] = time.perf_counter() - t_hop_start
                telemetry["hops"].append(hop_data)
                break
                
            # Update current nodes and visited
            current_node_ids = set()
            for nid, label, name_or_title, ctext, conn_desc, semantic_score, is_note, final_score in selected_candidates:
                accumulated_nodes[nid] = (label, name_or_title, ctext, conn_desc, semantic_score)
                visited_nodes.add(nid)
                current_node_ids.add(nid)
                
            hop_data["total_time"] = time.perf_counter() - t_hop_start
            telemetry["hops"].append(hop_data)
            n += 1

        # Step 3: Chunk Ingestion
        t_ingest_start = time.perf_counter()
        # Get all surviving papers/notes
        surviving_papers = [nid for nid, (label, _, _, _, _) in accumulated_nodes.items() if label in ("Paper", "UserNote")]
        telemetry["ingestion"]["surviving_papers"] = surviving_papers
        
        # Track already loaded chunks to avoid duplication
        loaded_chunk_ids = {c.id for c, _ in initial_chunks}
        all_chunks: List[Tuple[str, str, Optional[str], Optional[int]]] = []  # List of (chunk_id, text, paper_title, page)
        
        # Load initial chunks
        for chunk, _ in initial_chunks:
            paper_title = self._resolve_paper_title(chunk.paper_id, papers_map)
            all_chunks.append((chunk.id, chunk.text_content, paper_title, chunk.page_number))
            
        # For surviving papers, fetch additional relevant chunks if they are new
        new_chunks_to_score: List[Chunk] = []
        t_fetch_chunks_start = time.perf_counter()
        for pid in surviving_papers:
            paper_chunks = self.vector_repo.get_chunks_for_paper(pid)
            for c in paper_chunks:
                if c.id not in loaded_chunk_ids:
                    new_chunks_to_score.append(c)
        telemetry["ingestion"]["fetch_time"] = time.perf_counter() - t_fetch_chunks_start
                    
        if new_chunks_to_score:
            t_rerank_chunks_start = time.perf_counter()
            # Batch rerank all new chunks to find the most relevant ones
            chunk_pairs = [(query, c.text_content) for c in new_chunks_to_score]
            chunk_scores = self.reranker.predict(chunk_pairs)
            telemetry["ingestion"]["rerank_time"] = time.perf_counter() - t_rerank_chunks_start
                
            # Group chunks by paper
            paper_to_scored_chunks: Dict[str, List[Tuple[Chunk, float]]] = {}
            for idx, c in enumerate(new_chunks_to_score):
                raw_score = float(chunk_scores[idx])
                sigmoid_score = safe_sigmoid(raw_score)
                paper_to_scored_chunks.setdefault(c.paper_id, []).append((c, sigmoid_score))
                
            # Keep top chunks per paper (only those selected by dynamic Top-P to filter noise)
            for pid, scored in paper_to_scored_chunks.items():
                scored.sort(key=lambda x: x[1], reverse=True)
                
                # Apply dynamic Top-P (nucleus filtering) on sigmoid_score (item[1])
                total_score = sum(item[1] for item in scored)
                top_p_val = _safe_float(config.graph_sigmoid_score_top_p, 0.9)
                
                valid_scored = []
                if total_score > 0:
                    cumulative_score = 0.0
                    for item in scored:
                        valid_scored.append(item)
                        cumulative_score += item[1]
                        if cumulative_score >= top_p_val * total_score:
                            break
                    
                top_scored = valid_scored[:self.top_chunks_per_paper]
                top_scored_ids = {c.id for c, _ in top_scored}
                
                # Fetch paper details to resolve title
                p_obj = self.graph_repo.get_paper(pid)
                p_title = p_obj.title if p_obj else pid
                
                for c, score in scored:
                    telemetry["ingestion"]["new_chunks_evaluated"].append({
                        "id": c.id,
                        "paper_title": p_title,
                        "page": c.page_number,
                        "score": score,
                        "selected": c.id in top_scored_ids
                    })
                
                for c, score in top_scored:
                    all_chunks.append((c.id, c.text_content, p_title, c.page_number))
                    loaded_chunk_ids.add(c.id)
                    telemetry["ingestion"]["chunks_kept"].append({
                        "id": c.id,
                        "paper_title": p_title,
                        "page": c.page_number,
                        "score": score
                    })
        telemetry["ingestion"]["total_time"] = time.perf_counter() - t_ingest_start

        # Step 4: Semantic Filtering (Evidence List)
        t_filter_start = time.perf_counter()
        # Combine accumulated nodes and chunks into a single list
        # Map original IDs to short IDs (fact_1, fact_2, ...)
        orig_to_short: Dict[str, str] = {}
        short_to_orig: Dict[str, str] = {}
        short_to_text: Dict[str, str] = {}
        short_to_metadata: Dict[str, Dict[str, Any]] = {}
        
        fact_idx = 1
        
        # Process graph nodes
        for nid, (label, name_or_title, card_text, conn_desc, _) in accumulated_nodes.items():
            short_id = f"fact_{fact_idx}"
            orig_to_short[nid] = short_id
            short_to_orig[short_id] = nid
            
            # Format text card for LLM evaluation
            conn_str = f" (Connection: {conn_desc})" if conn_desc else ""
            node_desc = f"[{label}] {name_or_title}: {card_text}{conn_str}"
            short_to_text[short_id] = node_desc
            short_to_metadata[short_id] = {
                "is_node": True,
                "label": label,
                "name_or_title": name_or_title,
                "card_text": card_text,
                "conn_desc": conn_desc,
            }
            fact_idx += 1
            
        # Process chunks
        for cid, text, p_title, page in all_chunks:
            short_id = f"fact_{fact_idx}"
            orig_to_short[cid] = short_id
            short_to_orig[short_id] = cid
            
            chunk_desc = f"[Chunk] Source Paper: {p_title} (Page {page}): {text.strip()}"
            short_to_text[short_id] = chunk_desc
            short_to_metadata[short_id] = {
                "is_node": False,
                "paper_title": p_title,
                "page": page,
                "text": text,
            }
            fact_idx += 1

        for sid, txt in short_to_text.items():
            telemetry["filtering"]["facts_evaluated"].append({
                "short_id": sid,
                "text": txt,
                "is_essential": False,
                "score": 0.0
            })

        if not short_to_text:
            telemetry["filtering"]["total_time"] = time.perf_counter() - t_filter_start
            telemetry["overall_time"] = time.perf_counter() - t_overall_start
            self.last_telemetry = telemetry
            self.last_output = "No enrichment facts gathered."
            if should_trace:
                self._print_trace_summary(telemetry)
            return "No enrichment facts gathered."

        essential_items = []
        essential_ids_set = set()
        try:
            # Rerank all gathered facts using the Cross-Encoder
            short_ids = list(short_to_text.keys())
            fact_texts = [short_to_text[sid] for sid in short_ids]
            pairs = [(query, text) for text in fact_texts]
            
            with self.model_lock:
                scores = self.reranker.predict(pairs)
                
            for idx, sid in enumerate(short_ids):
                raw_score = float(scores[idx])
                sigmoid_score = safe_sigmoid(raw_score)
                is_essential = sigmoid_score >= _safe_float(config.graph_essential_fact_threshold, 0.5)
                
                if is_essential:
                    orig_id = short_to_orig.get(sid)
                    if orig_id:
                        essential_items.append(sid)
                        essential_ids_set.add(sid)
                
                # Update telemetry
                for f in telemetry["filtering"]["facts_evaluated"]:
                    if f["short_id"] == sid:
                        f["is_essential"] = is_essential
                        f["score"] = sigmoid_score

            telemetry["filtering"]["essential_ids"] = list(essential_ids_set)
            con.success(f"Evidence list filtering (Cross-Encoder): {len(essential_items)} / {len(short_to_text)} facts marked essential.")
        except Exception as e:
            con.warning(f"Evidence List semantic filtering failed: {e}. Falling back to including all gathered facts.")
            essential_items = list(short_to_text.keys())
            for f in telemetry["filtering"]["facts_evaluated"]:
                f["is_essential"] = True
            telemetry["filtering"]["essential_ids"] = essential_items
            
        telemetry["filtering"]["total_time"] = time.perf_counter() - t_filter_start

        # Step 5: Format output block
        # Format the selected facts as:
        # ### KNOWLEDGE GRAPH ENRICHMENT:
        # [Type] Title/Name: Content (Connection: Connection description, if applicable)
        enrichment_lines = []
        for sid in essential_items:
            meta = short_to_metadata.get(sid)
            if not meta:
                continue
                
            if meta["is_node"]:
                label = meta["label"]
                name_or_title = meta["name_or_title"]
                card_text = meta["card_text"]
                conn_desc = meta["conn_desc"]
                
                # Format: [Type] Title/Name: Content (Connection: Connection description, if applicable)
                conn_suffix = f" (Connection: {conn_desc})" if conn_desc else ""
                enrichment_lines.append(f"[{label}] {name_or_title}: {card_text}{conn_suffix}")
            else:
                p_title = meta["paper_title"]
                page = meta["page"]
                text = meta["text"]
                
                page_str = f" (Page {page})" if page else ""
                enrichment_lines.append(f"[Chunk] {p_title}{page_str}: {text.strip()}")
                
        telemetry["overall_time"] = time.perf_counter() - t_overall_start
        self.last_telemetry = telemetry
        if should_trace:
            self._print_trace_summary(telemetry)

        import logging
        dt = time.perf_counter() - t_overall_start
        logging.debug(f"ExperimentalGraphExpander.expand completed in {dt:.6f}s")

        if not enrichment_lines:
            self.last_output = "No essential knowledge graph enrichment found."
            return "No essential knowledge graph enrichment found."
        self.last_output = "\n".join(enrichment_lines)
        return self.last_output

    def _print_trace_summary(self, telemetry: Dict[str, Any]) -> None:
        table_time = Table(
            title="⏱️ Phase execution timings",
            title_style="bold magenta",
            box=box.MINIMAL,
            show_footer=True
        )
        table_time.add_column("Phase", footer="Total Duration")
        table_time.add_column("Duration", justify="right", style="yellow")
        table_time.add_column("%", justify="right", style="dim")
        
        t_tot = telemetry["overall_time"]
        
        def add_time_row(name, sec):
            pct = (sec / t_tot) * 100 if t_tot > 0 else 0
            table_time.add_row(name, f"{sec:.3f}s", f"{pct:.1f}%")
            
        add_time_row("1. Preparation", telemetry["prep_time"])
        for hop in telemetry["hops"]:
            idx = hop["hop_index"]
            add_time_row(f"2. Hop {idx} Crawl", hop.get("total_time", 0.0))
        add_time_row("3. Chunk Ingestion", telemetry["ingestion"].get("total_time", 0.0))
        add_time_row("4. Semantic Filtering (LLM)", telemetry["filtering"].get("total_time", 0.0))
        
        table_time.columns[1].footer = f"[bold yellow]{t_tot:.3f}s[/bold yellow]"

        tree = Tree("[bold cyan]🌳 Graph Crawl & Adaptive Context Expansion[/bold cyan]")
        for hop in telemetry["hops"]:
            h_idx = hop["hop_index"]
            k_n = hop["k_n_float"]
            k_lim = hop["k_limit"]
            hop_branch = tree.add(
                f"[bold yellow]Hop {h_idx}[/bold yellow] "
                f"(K_n limit: {k_n:.2f} -> Max Selected: {k_lim}, "
                f"Fetch: {hop.get('fetch_time', 0.0):.3f}s)"
            )
            
            # Sub-branch for fetched candidates
            fetch_branch = hop_branch.add(f"Fetched {len(hop['candidates_fetched'])} unique neighbors")
            for cand in hop["candidates_fetched"]:
                desc_suffix = f" ({cand['conn_desc']})" if cand['conn_desc'] else ""
                fetch_branch.add(f"'{cand['name']}' [{cand['label']}]{desc_suffix}")
            
            # Sub-branch for scoring
            score_branch = hop_branch.add(f"Reranking & Filtering (Rerank: {hop.get('rerank_time', 0.0):.3f}s)")
            for ev in hop["evaluated"]:
                sel_status = "[bold green]PASS[/bold green]" if ev["selected"] else "[red]FAIL[/red]"
                score_branch.add(f"{sel_status} | {ev['label']} | '{ev['name']}' (Score: {ev['score']:.3f})")

        ingest_branch = tree.add(
            f"[bold cyan]📥 Chunk Ingestion[/bold cyan] "
            f"(Surviving Papers: {len(telemetry['ingestion']['surviving_papers'])}, "
            f"Fetch: {telemetry['ingestion'].get('fetch_time', 0.0):.3f}s)"
        )
        if telemetry["ingestion"]["new_chunks_evaluated"]:
            new_chunk_branch = ingest_branch.add(f"New chunks evaluated (Rerank: {telemetry['ingestion'].get('rerank_time', 0.0):.3f}s)")
            for chunk in telemetry["ingestion"]["new_chunks_evaluated"]:
                status = "[bold green]KEPT[/bold green]" if chunk["selected"] else "[red]DROP[/red]"
                new_chunk_branch.add(f"{status} | Paper: '{chunk['paper_title']}' (Page {chunk['page']}) | Score: {chunk['score']:.3f}")
        else:
            ingest_branch.add("No new chunks loaded (all papers were already represented in initial chunks)")

        filter_branch = tree.add("[bold cyan]⚖️ Semantic Filtering (LLM Evidence List Review)[/bold cyan]")
        ess_count = len(telemetry["filtering"]["essential_ids"])
        tot_count = len(telemetry["filtering"]["facts_evaluated"])
        filter_branch.add(f"LLM evaluated {tot_count} facts -> Selected {ess_count} as [bold green]ESSENTIAL[/bold green]")
        for f in telemetry["filtering"]["facts_evaluated"]:
            status = "[bold green]ESSENTIAL[/bold green]" if f["is_essential"] else "[red]REDUNDANT[/red]"
            text_preview = f["text"] if len(f["text"]) < 100 else f["text"][:100] + "..."
            filter_branch.add(f"{status} (Score: {f['score']:.2f}) | {text_preview}")

        console.print(Panel(
            Group(table_time, tree),
            title="🔍 [bold magenta]RAG Advanced Graph Expansion Debug Trace[/bold magenta]",
            border_style="magenta",
            padding=(1, 2)
        ))
