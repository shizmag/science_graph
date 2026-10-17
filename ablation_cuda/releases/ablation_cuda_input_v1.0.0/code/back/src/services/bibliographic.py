import math
import time
import re
import hashlib
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from src.repository.base import GraphRepository

@dataclass
class CanonicalReference:
    work_id: str
    title: str | None
    normalized_title: str | None
    doi: str | None
    arxiv_id: str | None
    url: str | None
    year: str | None
    authors: list[str]
    raw_reference: str
    canonicalization_method: str

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def normalize_doi(doi: str) -> str:
    doi = doi.lower().strip()
    doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', doi)
    doi = re.sub(r'^doi:', '', doi)
    return doi.strip()

def normalize_arxiv(arxiv: str) -> str:
    arxiv = arxiv.lower().strip()
    arxiv = re.sub(r'^arxiv:', '', arxiv)
    arxiv = re.sub(r'^https?://arxiv\.org/(?:abs|pdf)/', '', arxiv)
    m = re.match(r'^(\d{4}\.\d{4,5})', arxiv)
    if m:
        return m.group(1)
    return arxiv

def normalize_url(url: str) -> str:
    from urllib.parse import urlparse
    url = url.strip()
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path
        if path.endswith('/'):
            path = path[:-1]
        normalized = f"{parsed.scheme}://{host}{path}"
        if parsed.params:
            normalized += f";{parsed.params}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized
    except Exception:
        return url.lower()

def normalize_title(title: str) -> str:
    if not title:
        return ""
    title = unicodedata.normalize("NFKC", title)
    title = title.lower()
    title = title.strip(".,;:!?\"'")
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

def canonicalize_reference(raw_reference: str, parsed_metadata: Optional[dict] = None) -> CanonicalReference:
    meta = parsed_metadata or {}
    raw_reference = raw_reference.strip()
    
    # 1. DOI
    doi = meta.get("doi")
    if not doi:
        doi_match = re.search(r'(?:doi:|(?:https?://)?(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)', raw_reference)
        if doi_match:
            doi = doi_match.group(1).rstrip(".,;)")
            
    if doi:
        norm_d = normalize_doi(doi)
        return CanonicalReference(
            work_id=f"work:doi:{norm_d}",
            title=meta.get("title") or doi,
            normalized_title=normalize_title(meta.get("title")),
            doi=norm_d,
            arxiv_id=None,
            url=None,
            year=str(meta.get("year")) if meta.get("year") else None,
            authors=meta.get("authors") or [],
            raw_reference=raw_reference,
            canonicalization_method="doi"
        )
        
    # 2. arXiv ID
    arxiv = meta.get("arxiv_id") or meta.get("arxiv")
    if not arxiv:
        arxiv_match = re.search(r'(?:arxiv:|(?:https?://)?arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)', raw_reference, re.IGNORECASE)
        if arxiv_match:
            arxiv = arxiv_match.group(1)
            
    if arxiv:
        norm_a = normalize_arxiv(arxiv)
        return CanonicalReference(
            work_id=f"work:arxiv:{norm_a}",
            title=meta.get("title") or arxiv,
            normalized_title=normalize_title(meta.get("title")),
            doi=None,
            arxiv_id=norm_a,
            url=None,
            year=str(meta.get("year")) if meta.get("year") else None,
            authors=meta.get("authors") or [],
            raw_reference=raw_reference,
            canonicalization_method="arxiv"
        )
        
    # 3. URL
    url = meta.get("url")
    if not url:
        url_match = re.search(r'(https?://[^\s,;]+)', raw_reference)
        if url_match:
            url = url_match.group(1).rstrip(".,;)")
            
    if url:
        norm_u = normalize_url(url)
        return CanonicalReference(
            work_id=f"work:url:{stable_hash(norm_u)}",
            title=meta.get("title") or url,
            normalized_title=normalize_title(meta.get("title")),
            doi=None,
            arxiv_id=None,
            url=norm_u,
            year=str(meta.get("year")) if meta.get("year") else None,
            authors=meta.get("authors") or [],
            raw_reference=raw_reference,
            canonicalization_method="url"
        )
        
    # 4. Title
    title = meta.get("title")
    if title:
        norm_t = normalize_title(title)
        return CanonicalReference(
            work_id=f"work:title:{stable_hash(norm_t)}",
            title=title,
            normalized_title=norm_t,
            doi=None,
            arxiv_id=None,
            url=None,
            year=str(meta.get("year")) if meta.get("year") else None,
            authors=meta.get("authors") or [],
            raw_reference=raw_reference,
            canonicalization_method="normalized_title_hash"
        )
        
    # 5. Fallback to stable raw hash
    norm_raw = normalize_title(raw_reference)
    return CanonicalReference(
        work_id=f"work:raw:{stable_hash(raw_reference)}",
        title=raw_reference[:100],
        normalized_title=norm_raw,
        doi=None,
        arxiv_id=None,
        url=None,
        year=None,
        authors=[],
        raw_reference=raw_reference,
        canonicalization_method="raw_hash"
    )

def find_citation_context_in_text(text: str, ref: CanonicalReference, marker: Optional[str] = None) -> Optional[str]:
    if not text:
        return None
    sentences = re.split(r'(?<=[.!?])\s+', text)
    patterns = []
    if marker:
        patterns.append(re.compile(re.escape(marker)))
    author_surname = None
    if ref.authors:
        first_author = ref.authors[0]
        parts = [p.strip() for p in first_author.split(",")]
        if len(parts) > 1 and len(parts[0]) > 1:
            author_surname = parts[0]
        else:
            words = first_author.split()
            if words:
                author_surname = words[-1].strip(".,")
    if author_surname and ref.year:
        pat_str = rf"\b{re.escape(author_surname)}\b.*\b{re.escape(str(ref.year))}\b"
        patterns.append(re.compile(pat_str, re.IGNORECASE))
    if ref.title and len(ref.title) > 10:
        words = [re.escape(w) for w in ref.title.split()[:3] if len(w) > 2]
        if words:
            patterns.append(re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE))
    for pat in patterns:
        for idx, sent in enumerate(sentences):
            try:
                if pat.search(sent):
                    start = max(0, idx - 1)
                    end = min(len(sentences), idx + 2)
                    return " ".join(sentences[start:end]).strip()
            except Exception:
                continue
    return None

def resolve_reference_target(repo: GraphRepository, ref: CanonicalReference) -> Tuple[str, bool]:
    if ref.doi:
        local_paper = repo.find_paper_by_doi(ref.doi)
        if local_paper:
            return local_paper.id, True
    if ref.arxiv_id:
        local_paper = repo.find_paper_by_arxiv(ref.arxiv_id)
        if local_paper:
            return local_paper.id, True
    if ref.url:
        local_paper = repo.find_paper_by_url(ref.url)
        if local_paper:
            return local_paper.id, True
    if ref.normalized_title:
        local_paper = repo.find_paper_by_title(ref.title or ref.normalized_title)
        if local_paper:
            return local_paper.id, True
    return ref.work_id, False


class BibliographicProjectionService:
    def __init__(self, graph_repo: GraphRepository):
        self.repo = graph_repo

    def rebuild_projection(self) -> None:
        """
        Runs a complete recomputation of the derived bibliographic layer.
        Recomputes stats, paper vectors, BIBLIOGRAPHIC_COUPLING edges,
        and RELATED_BY_SHARED_REFERENCE edges for chunks.
        """
        # 0. Remap ExternalWork nodes that are now resolved to a newly indexed local Paper
        import json
        all_nodes = self.repo.get_all_nodes()
        external_works = [n for n in all_nodes if n[1] == "ExternalWork"]
        for ext_id, ext_label, ext_props_str in external_works:
            try:
                ext_props = json.loads(ext_props_str)
                ref = CanonicalReference(
                    work_id=ext_id,
                    title=ext_props.get("title"),
                    normalized_title=ext_props.get("normalized_title"),
                    doi=ext_props.get("doi"),
                    arxiv_id=ext_props.get("arxiv_id"),
                    url=ext_props.get("url"),
                    year=ext_props.get("year"),
                    authors=ext_props.get("authors") or [],
                    raw_reference=ext_props.get("raw_reference") or "",
                    canonicalization_method=ext_props.get("canonicalization_method") or "raw_hash"
                )
                target_id, is_local = resolve_reference_target(self.repo, ref)
                if is_local:
                    # Remap all edges and mentions pointing to ext_id to point to target_id instead
                    self.repo.remap_external_work_to_local_paper(ext_id, target_id)
            except Exception:
                pass

        # 1. Load all local indexed papers (N is strictly the count of local non-placeholder papers)
        local_papers = self.repo.get_non_placeholder_papers()
        N = len(local_papers)
        if N == 0:
            # Clean up derived statistics & edges if empty
            self.repo.delete_derived_edges_by_types(["BIBLIOGRAPHIC_COUPLING", "RELATED_BY_SHARED_REFERENCE"])
            return

        # 2. Get CITES edges to build paper -> references mapping
        all_edges = self.repo.get_all_edges()
        paper_citations: Dict[str, Set[str]] = {p.id: set() for p in local_papers}
        
        # Populate citations from CITES edges
        for src_id, tgt_id, edge_type, _ in all_edges:
            if edge_type == "CITES" and src_id in paper_citations:
                paper_citations[src_id].add(tgt_id)

        # 3. Compute DF (document frequency) for each cited work
        df: Dict[str, int] = {}
        for cited_set in paper_citations.values():
            for ref_id in cited_set:
                df[ref_id] = df.get(ref_id, 0) + 1

        # 4. Save stats and calculate IDF
        # idf(r) = log(N / df(r)) using natural logarithm
        stats_to_save = []
        updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        idf: Dict[str, float] = {}
        
        for work_id, df_val in df.items():
            idf_val = math.log(N / df_val)
            idf[work_id] = idf_val
            stats_to_save.append((work_id, df_val, idf_val, N, updated_at))
            
        self.repo.save_reference_corpus_stats(stats_to_save)

        # 5. Compute sparse reference vectors
        # v_A[r] = idf(r) if A cites r, else 0
        vectors_to_save = []
        paper_norms: Dict[str, float] = {}
        
        for paper_id, cited_set in paper_citations.items():
            norm_sq = 0.0
            for ref_id in cited_set:
                weight = idf[ref_id]
                vectors_to_save.append((paper_id, ref_id, weight))
                norm_sq += weight ** 2
            paper_norms[paper_id] = math.sqrt(norm_sq)

        self.repo.save_paper_reference_vectors(vectors_to_save)

        # 6. Delete old derived edges
        self.repo.delete_derived_edges_by_types(["BIBLIOGRAPHIC_COUPLING", "RELATED_BY_SHARED_REFERENCE"])

        # 7. Compute BIBLIOGRAPHIC_COUPLING edges with dual weight
        coupling_edges = []
        paper_ids = list(paper_citations.keys())
        
        for i in range(len(paper_ids)):
            p_A = paper_ids[i]
            c_A = paper_citations[p_A]
            norm_A = paper_norms[p_A]
            len_A = len(c_A)
            if len_A == 0:
                continue
                
            for j in range(i + 1, len(paper_ids)):
                p_B = paper_ids[j]
                c_B = paper_citations[p_B]
                norm_B = paper_norms[p_B]
                len_B = len(c_B)
                if len_B == 0:
                    continue
                    
                shared = c_A.intersection(c_B)
                if shared:
                    # 1. Structural Weight: binary cosine
                    structural_weight = len(shared) / math.sqrt(len_A * len_B)
                    
                    # 2. Specificity Weight: TF-IDF cosine (0.0 if norm is 0)
                    specificity_weight = 0.0
                    if norm_A > 0.0 and norm_B > 0.0:
                        dot = sum(idf[r] ** 2 for r in shared)
                        specificity_weight = dot / (norm_A * norm_B)
                        
                    props = {
                        "derived": True,
                        "method": "bibliographic_coupling_dual_weight",
                        "weight": structural_weight, # traversal-compatible weight
                        "structural_weight": structural_weight,
                        "specificity_weight": specificity_weight,
                        "shared_reference_count": len(shared),
                        "shared_reference_ids": list(shared),
                        "formula_structural": "binary cosine over reference sets: |A∩B| / sqrt(|A|*|B|)",
                        "formula_specificity": "cosine over binary-TF IDF reference vectors; idf(r)=ln(N/df(r))",
                        "explanation": "Structural weight captures observed bibliographic overlap; specificity weight captures how corpus-specific the shared references are."
                    }
                    coupling_edges.append((p_A, p_B, "BIBLIOGRAPHIC_COUPLING", props))
                    coupling_edges.append((p_B, p_A, "BIBLIOGRAPHIC_COUPLING", props))

        if coupling_edges:
            self.repo.save_edges_bulk(coupling_edges)

        # 8. Compute RELATED_BY_SHARED_REFERENCE edges for chunks
        # Uses the same dual-weight model as BIBLIOGRAPHIC_COUPLING:
        #   structural_weight = |Refs(A) ∩ Refs(B)| / sqrt(|Refs(A)| * |Refs(B)|)
        #   specificity_weight = sum(idf(r) for r in shared)
        #   weight = structural_weight  (traversal-compatible)
        # A shared reference always produces an edge, even if idf(r) = 0.
        import logging
        logger = logging.getLogger(__name__)

        mentions = self.repo.get_chunk_reference_mentions()
        if not mentions:
            return

        # Build per-chunk reference sets: chunk_id → {work_id, ...}
        chunk_refs: Dict[str, Set[str]] = {}
        chunk_paper: Dict[str, str] = {}
        for m in mentions:
            cid = m["chunk_id"]
            wid = m["work_id"]
            if cid not in chunk_refs:
                chunk_refs[cid] = set()
                chunk_paper[cid] = m["paper_id"]
            chunk_refs[cid].add(wid)

        # Build inverted index: work_id → [chunk_id, ...]
        work_to_chunks: Dict[str, List[str]] = {}
        for cid, refs in chunk_refs.items():
            for wid in refs:
                if wid not in work_to_chunks:
                    work_to_chunks[wid] = []
                work_to_chunks[wid].append(cid)

        # Accumulate shared reference sets per ordered chunk pair
        pair_shared: Dict[Tuple[str, str], Set[str]] = {}
        for wid, chunk_list in work_to_chunks.items():
            for i in range(len(chunk_list)):
                c_A = chunk_list[i]
                p_A = chunk_paper[c_A]
                for j in range(i + 1, len(chunk_list)):
                    c_B = chunk_list[j]
                    p_B = chunk_paper[c_B]
                    if p_A == p_B:
                        continue  # same paper
                    key = (c_A, c_B) if c_A < c_B else (c_B, c_A)
                    if key not in pair_shared:
                        pair_shared[key] = set()
                    pair_shared[key].add(wid)

        # Cardinality diagnostic (single summary line per rebuild)
        total_directed_edges = len(pair_shared) * 2
        logger.info(
            f"[BibliographicProjection] Shared reference cardinality diagnostic: "
            f"number_of_citing_chunks={len(chunk_refs)}, "
            f"number_of_chunk_pairs={len(pair_shared)}, "
            f"number_of_created_chunk_edges={total_directed_edges}"
        )

        # Build edges with dual weight
        chunk_edges = []
        for (c_A, c_B), shared in pair_shared.items():
            len_A = len(chunk_refs[c_A])
            len_B = len(chunk_refs[c_B])

            structural_weight = len(shared) / math.sqrt(len_A * len_B)
            specificity_weight = sum(idf.get(r, 0.0) for r in shared)

            props = {
                "derived": True,
                "method": "related_by_shared_reference_dual_weight",
                "weight": structural_weight,          # traversal-compatible weight
                "structural_weight": structural_weight,
                "specificity_weight": specificity_weight,
                "shared_reference_count": len(shared),
                "shared_reference_ids": list(shared),
                "formula_structural": "binary cosine over chunk reference sets: |A∩B| / sqrt(|A|*|B|)",
                "formula_specificity": "sum(idf(r)) for r in shared; idf(r)=log(N/df(r))",
                "explanation": (
                    "Structural weight captures observed reference overlap between chunks; "
                    "specificity weight captures how corpus-specific the shared references are."
                )
            }
            chunk_edges.append((c_A, c_B, "RELATED_BY_SHARED_REFERENCE", props))
            chunk_edges.append((c_B, c_A, "RELATED_BY_SHARED_REFERENCE", props))

        if chunk_edges:
            self.repo.save_edges_bulk(chunk_edges)
