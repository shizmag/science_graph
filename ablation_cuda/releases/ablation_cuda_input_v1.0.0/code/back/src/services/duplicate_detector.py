"""
DuplicateDetector — identifies duplicate documents in the database.
"""

import hashlib
from typing import List, Optional, Tuple

from src.models import Paper, Chunk, slugify
from src.config import config
from src.repository.base import GraphRepository, VectorRepository
from src.vector_search import EmbeddingEngine, split_segment_to_chunks
from src.services.similarity import JaccardSimilarity


def _split_text_to_chunks_raw(
    paper_id: str, text: str, chunk_size: int = None, chunk_overlap: int = None
) -> List[Chunk]:
    """Splits a plain text string (not PDF) into overlapping child Chunk objects with parent context."""
    parent_size = chunk_size or getattr(config, "parent_chunk_size", 2500)
    parent_overlap = chunk_overlap if chunk_overlap is not None else getattr(config, "parent_chunk_overlap", 200)
    
    child_size = getattr(config, "child_chunk_size", 300)
    child_overlap = getattr(config, "child_chunk_overlap", 50)
    
    # 1. Split text into parent chunks
    parent_texts = split_segment_to_chunks(text, parent_size, parent_overlap)
    
    child_chunks = []
    global_child_idx = 0
    parent_idx = 0
    page_num = 1 # plain text usually has page 1
    
    for p_text in parent_texts:
        parent_id = f"{paper_id}#parent_{parent_idx}"
        parent_idx += 1
        
        # 2. Split parent chunk into child chunks
        child_texts = split_segment_to_chunks(p_text, child_size, child_overlap)
        
        for c_text in child_texts:
            child_chunks.append(Chunk(
                id=f"{paper_id}#{global_child_idx}",
                paper_id=paper_id,
                text_content=c_text,
                page_number=page_num,
                parent_id=parent_id,
                parent_text=p_text
            ))
            global_child_idx += 1
            
    return child_chunks


class DuplicateDetector:
    """
    Checks if a document already exists in the graph or vector repository
    using multiple layered heuristics (IDs, DOI, Content Hash, Title-Author similarity,
    Shingle Jaccard, and Vector embedding similarity).
    """

    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine

    def detect_duplicate(self, paper: Paper, full_text: str) -> Optional[Tuple[str, str]]:
        """
        Detects if the given paper/document is already present in the database.
        Returns:
            Optional[Tuple[str, str]]: (duplicate_paper_id, matching_reason) if a duplicate is found,
                                       else None.
        """
        import time
        import logging
        t0 = time.perf_counter()

        # Helper to check if a paper is a placeholder
        def is_placeholder(p: Paper) -> bool:
            return bool(p.properties.get("placeholder") or p.properties.get("is_placeholder"))

        # 1. Exact ID check
        existing = self.graph_repo.get_paper(paper.id)
        if existing and not is_placeholder(existing):
            dt = time.perf_counter() - t0
            logging.debug(f"detect_duplicate found exact ID match for paper {paper.id} in {dt:.6f}s")
            return existing.id, "exact_id"

        # 2. DOI check
        if paper.doi:
            existing_doi = self.graph_repo.find_paper_by_doi(paper.doi)
            if existing_doi and not is_placeholder(existing_doi):
                dt = time.perf_counter() - t0
                logging.debug(f"detect_duplicate found DOI match for paper {paper.id} in {dt:.6f}s")
                return existing_doi.id, "doi"

        # 3. Content Hash check (for incoming text)
        content_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()
        existing_hash = self.graph_repo.find_paper_by_content_hash(content_hash)
        if existing_hash and not is_placeholder(existing_hash):
            dt = time.perf_counter() - t0
            logging.debug(f"detect_duplicate found content hash match for paper {paper.id} in {dt:.6f}s")
            return existing_hash.id, "content_hash"

        # Helper to reconstruct text from chunks
        def reconstruct_text(paper_id: str) -> str:
            chunks = self.vector_repo.get_chunks_for_paper(paper_id)
            def get_idx(chunk):
                try:
                    return int(chunk.id.split('#')[-1])
                except Exception:
                    return chunk.id
            chunks_sorted = sorted(chunks, key=get_idx)
            return " ".join(c.text_content for c in chunks_sorted)

        # Local cache for reconstructed text and shingles mapped by candidate ID
        reconstruct_cache = {}
        shingles_cache = {}

        def get_reconstruct_text_cached(cand_id: str) -> str:
            if cand_id not in reconstruct_cache:
                reconstruct_cache[cand_id] = reconstruct_text(cand_id)
            return reconstruct_cache[cand_id]

        def get_3_shingles_cached(cand_id: str) -> set:
            if cand_id not in shingles_cache:
                txt = get_reconstruct_text_cached(cand_id)
                shingles_cache[cand_id] = JaccardSimilarity.get_3_shingles(txt)
            return shingles_cache[cand_id]

        # Build candidate set of Papers to avoid database-wide scans
        candidates = {}

        def add_candidate(p: Paper):
            if p and p.id != paper.id and not is_placeholder(p):
                candidates[p.id] = p

        # Check DOI candidate
        if paper.doi:
            add_candidate(self.graph_repo.find_paper_by_doi(paper.doi))

        # Check content hash candidate
        add_candidate(self.graph_repo.find_paper_by_content_hash(content_hash))

        # Check case-insensitive title candidate
        if paper.title:
            add_candidate(self.graph_repo.find_paper_by_title(paper.title))

        # Check shared author candidates
        for author_name in paper.authors:
            author_id = slugify(author_name)
            if author_id:
                for p in self.graph_repo.get_papers_by_author(author_id):
                    add_candidate(p)

        # Check vector similarity candidate & embedding similarity check
        chunks = _split_text_to_chunks_raw(paper.id, full_text)
        first_chunk_text = ""
        if chunks:
            first_chunk_text = chunks[0].text_content
        elif full_text.strip():
            first_chunk_text = full_text.strip()
        elif paper.title.strip():
            first_chunk_text = paper.title.strip()

        if first_chunk_text:
            emb = self.emb_engine.get_embeddings([first_chunk_text])[0]
            similar_chunks = self.vector_repo.search_similar_chunks(emb, limit=10)
            for c, sim in similar_chunks:
                if sim >= 0.95:
                    cand_paper = self.graph_repo.get_paper(c.paper_id)
                    if cand_paper and not is_placeholder(cand_paper):
                        add_candidate(cand_paper)
                        if JaccardSimilarity.word_jaccard_similarity(first_chunk_text, c.text_content) >= 0.80:
                            dt = time.perf_counter() - t0
                            logging.debug(f"detect_duplicate found embedding similarity match in {dt:.6f}s")
                            return cand_paper.id, "embedding_similarity"

        # Now detailed checking on all candidates
        shingles_new = set()
        # Data validation: ensure full_text is a valid, non-empty string and not truncated
        if full_text and isinstance(full_text, str) and len(full_text.strip()) > 50:
            shingles_new = JaccardSimilarity.get_3_shingles(full_text)

        for cand_id, cand in candidates.items():
            # A. Exact Title and Author similarity > 0.3 or both empty
            if paper.title and cand.title:
                if paper.title.strip().lower() == cand.title.strip().lower():
                    author_sim = JaccardSimilarity.author_jaccard_similarity(paper.authors, cand.authors)
                    if author_sim > 0.3 or (not paper.authors and not cand.authors):
                        dt = time.perf_counter() - t0
                        logging.debug(f"detect_duplicate found title/author similarity match in {dt:.6f}s")
                        return cand_id, "title_author_similarity"

            # B. Legacy content hash comparison
            cand_hash = cand.properties.get("content_hash")
            cand_text = None
            if not cand_hash:
                cand_text = get_reconstruct_text_cached(cand_id)
                cand_hash = hashlib.sha256(cand_text.encode('utf-8')).hexdigest()
                updated_props = {**cand.properties, "content_hash": cand_hash}
                self.graph_repo.update_node_properties(cand_id, updated_props)

            if cand_hash == content_hash:
                dt = time.perf_counter() - t0
                logging.debug(f"detect_duplicate found content hash match after reconstruction in {dt:.6f}s")
                return cand_id, "content_hash"

            # C. 3-word shingles check (threshold >= 0.70)
            if shingles_new:
                shingles_cand = get_3_shingles_cached(cand_id)
                if shingles_cand:
                    intersection = len(shingles_new.intersection(shingles_cand))
                    union = len(shingles_new.union(shingles_cand))
                    jaccard = intersection / union if union > 0 else 0.0
                    if jaccard >= 0.70:
                        dt = time.perf_counter() - t0
                        logging.debug(f"detect_duplicate found shingle similarity match in {dt:.6f}s")
                        return cand_id, "shingle_similarity"

        dt = time.perf_counter() - t0
        logging.debug(f"detect_duplicate completed in {dt:.6f}s (no duplicates found)")
        return None
