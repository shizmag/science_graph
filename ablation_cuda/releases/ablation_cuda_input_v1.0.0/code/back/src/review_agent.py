"""
Auto-Review Agent.

Runs a full agentic cycle:
  1. Hybrid search (Dense + BM25 + RRF + Cross-Encoder rerank)
  2. LLM-based thematic clustering of retrieved chunks
  3. Sequential LLM synthesis of each section
  4. Comparison table generation
  5. Bibliography collection from the graph
  6. Final Markdown report assembly
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models import Chunk
from src.rag import RAGPipeline
from src.repository.base import GraphRepository, VectorRepository
from src.vector_search import EmbeddingEngine
from src.llm_engine import LLMEngine
from src import console as con


class ReviewAgent:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        llm_engine: LLMEngine,
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine
        self._rag = RAGPipeline(graph_repo, vector_repo, embedding_engine, llm_engine)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        topic: str,
        limit: int = 20,
        output_path: Optional[Path] = None,
        fast: bool = False,
    ) -> str:
        """
        Runs the full review pipeline.

        Args:
            topic:       Research topic to review.
            limit:       Number of chunks to retrieve for context.
            output_path: If given, write Markdown report to this file.
            fast:        Skip LLM clustering, use a single default section.

        Returns:
            Markdown report as a string.
        """
        import time
        import logging
        t0 = time.perf_counter()

        con.section(f"Auto-Review: {topic}")
        con.info(f"Retrieving up to [bold]{limit}[/bold] relevant chunks …")

        # Step 1 — Hybrid retrieval
        final_chunks = self._hybrid_retrieve(topic, limit)
        if not final_chunks:
            return f"# Review: {topic}\n\n*No relevant documents found in the index.*\n"

        con.success(f"Retrieved {len(final_chunks)} chunks from {len({c.paper_id for c, _ in final_chunks})} papers")

        # Step 2 — Cluster chunks into thematic sections
        id_to_chunk: Dict[str, Chunk] = {c.id: c for c, _ in final_chunks}
        sections: Dict[str, List[str]]  # section_name → [chunk_ids]

        if fast or not hasattr(self.llm_engine, "cluster_chunks_by_topic"):
            sections = {"Overview": [c.id for c, _ in final_chunks]}
        else:
            sections = self._cluster_chunks(final_chunks, topic)

        # Step 3 — Synthesize sections in parallel
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from src.config import config

        con.info(f"Synthesizing [bold]{len(sections)}[/bold] section(s) in parallel …")
        synthesized: Dict[str, str] = {}
        
        # Semaphore based on chunk_pool_size to avoid Rate Limit Errors
        pool_size = config.data.get("llm", {}).get("chunk_pool_size", 4)
        sem = threading.Semaphore(pool_size)
        write_lock = threading.Lock()

        def synthesize_worker(idx_val: int, sec_name: str, ch_ids: List[str]) -> None:
            section_chunks = [id_to_chunk[cid] for cid in ch_ids if cid in id_to_chunk]
            if not section_chunks:
                return
            chunks_text = self._format_chunks_for_synthesis(section_chunks)
            
            with sem:
                con.dim(f"  [{idx_val}/{len(sections)}] Starting synthesis for {sec_name} ...")
                res = self.llm_engine.synthesize_section(
                    section_name=sec_name,
                    chunks_text=chunks_text,
                    topic=topic,
                )
            
            with write_lock:
                synthesized[sec_name] = res

        with ThreadPoolExecutor(max_workers=len(sections)) as executor:
            for idx, (section_name, chunk_ids) in enumerate(sections.items(), start=1):
                executor.submit(synthesize_worker, idx, section_name, chunk_ids)

        # Step 4 — Comparison table
        con.dim("Building comparison table …")
        paper_ids = list({c.paper_id for c, _ in final_chunks})
        comparison_table = self._build_comparison_table(paper_ids)

        # Step 5 — Bibliography
        bibliography = self._collect_bibliography(paper_ids)

        # Step 6 — Assemble Markdown report
        report = self._render_report(topic, synthesized, comparison_table, bibliography)

        # Write to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            con.success(f"Report saved: [bold]{output_path}[/bold]")

        con.success("Auto-Review complete")
        
        dt = time.perf_counter() - t0
        logging.debug(f"ReviewAgent.run completed in {dt:.6f}s for topic: {topic!r}")
        return report

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _hybrid_retrieve(self, query: str, limit: int) -> List[Tuple[Chunk, float]]:
        """Dense + BM25 + RRF + Cross-Encoder reranking (mirrors RAGPipeline.ask)."""
        query_emb = self.emb_engine.get_embedding(query)
        dense_results = self.vector_repo.search_similar_chunks(query_emb, limit=limit * 2)
        bm25_results = self.vector_repo.search_text_fts5(query, limit=limit * 2)

        if not dense_results and not bm25_results:
            return []

        id_to_chunk: Dict[str, Chunk] = {}
        for chunk, _ in dense_results:
            id_to_chunk[chunk.id] = chunk
        for chunk, _ in bm25_results:
            id_to_chunk[chunk.id] = chunk

        rrf_scores: Dict[str, float] = {}
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (60.0 + rank)
        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (60.0 + rank)

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        candidates = [id_to_chunk[cid] for cid in sorted_ids[: limit * 2] if cid in id_to_chunk]

        if not candidates:
            return []

        try:
            reranker = self._rag._get_reranker()
            pairs = [(query, c.text_content) for c in candidates]
            scores = reranker.predict(pairs)
            scored = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
            return scored[:limit]
        except Exception as e:
            con.warning(f"Reranker unavailable ({e}), using RRF ranking.")
            return [(id_to_chunk[cid], rrf_scores[cid]) for cid in sorted_ids[:limit] if cid in id_to_chunk]

    def _cluster_chunks(
        self, chunks_with_scores: List[Tuple[Chunk, float]], topic: str
    ) -> Dict[str, List[str]]:
        """Ask LLM to group chunks into thematic sections."""
        # Build a compact summary for the LLM prompt
        paper_ids = list({chunk.paper_id for chunk, _ in chunks_with_scores})
        papers = self.graph_repo.get_papers_batch(paper_ids)
        
        summaries = []
        for chunk, score in chunks_with_scores:
            paper = papers.get(chunk.paper_id)
            title = paper.title if paper else chunk.paper_id
            excerpt = chunk.text_content[:200].replace("\n", " ")
            summaries.append({"id": chunk.id, "title": title, "score": round(score, 3), "text": excerpt})

        summaries_json = json.dumps(summaries, ensure_ascii=False, indent=None)
        result = self.llm_engine.cluster_chunks_by_topic(summaries_json, topic)

        if not result or not isinstance(result, dict):
            # Fallback: single section
            return {"Overview": [c.id for c, _ in chunks_with_scores]}

        # Validate all chunk IDs exist; put orphans in "Other"
        known_ids = {c.id for c, _ in chunks_with_scores}
        cleaned: Dict[str, List[str]] = {}
        assigned: set = set()
        for section, ids in result.items():
            valid_ids = [cid for cid in ids if cid in known_ids]
            if valid_ids:
                cleaned[section] = valid_ids
                assigned.update(valid_ids)

        orphans = [c.id for c, _ in chunks_with_scores if c.id not in assigned]
        if orphans:
            cleaned.setdefault("Other", []).extend(orphans)

        return cleaned if cleaned else {"Overview": [c.id for c, _ in chunks_with_scores]}

    def _format_chunks_for_synthesis(self, chunks: List[Chunk]) -> str:
        """Formats chunks with paper metadata for LLM synthesis prompt."""
        paper_ids = list({chunk.paper_id for chunk in chunks})
        papers = self.graph_repo.get_papers_batch(paper_ids)
        
        blocks = []
        for chunk in chunks:
            paper = papers.get(chunk.paper_id)
            title = paper.title if paper else chunk.paper_id
            year = f", {paper.year}" if paper and paper.year else ""
            authors = ""
            if paper and paper.authors:
                first_authors = paper.authors[:2]
                et_al = " et al." if len(paper.authors) > 2 else ""
                authors = f" — {', '.join(first_authors)}{et_al}"
            blocks.append(
                f"[Paper: {title}{authors}{year}, p.{chunk.page_number}]\n"
                f"{chunk.text_content.strip()}"
            )
        return "\n\n---\n\n".join(blocks)

    def _build_comparison_table(self, paper_ids: List[str]) -> str:
        """Builds a Markdown comparison table from paper metadata."""
        rows = []
        papers = self.graph_repo.get_papers_batch(paper_ids)
        for pid in paper_ids:
            paper = papers.get(pid)
            if not paper or not paper.title:
                continue
            # Collect concepts connected to this paper
            neighbors = self.graph_repo.get_neighbors(pid, max_depth=1)
            concepts = []
            for src_id, src_label, edge_type, tgt_id, tgt_label, _ in neighbors:
                if edge_type == "MENTIONS_CONCEPT" and tgt_label == "Concept":
                    concept = self.graph_repo.get_concept(tgt_id)
                    if concept:
                        concepts.append(concept.name)

            authors_str = ""
            if paper.authors:
                a = paper.authors[:2]
                et_al = " et al." if len(paper.authors) > 2 else ""
                authors_str = f"{', '.join(a)}{et_al}"

            rows.append({
                "title": paper.title[:60] + ("…" if len(paper.title) > 60 else ""),
                "authors": authors_str or "—",
                "year": str(paper.year) if paper.year else "—",
                "concepts": ", ".join(concepts[:4]) if concepts else "—",
                "doi": paper.doi or "—",
            })

        if not rows:
            return ""

        lines = [
            "| Title | Authors | Year | Key Concepts | DOI |",
            "|-------|---------|------|-------------|-----|",
        ]
        for r in rows:
            lines.append(
                f"| {r['title']} | {r['authors']} | {r['year']} | {r['concepts']} | {r['doi']} |"
            )
        return "\n".join(lines)

    def _collect_bibliography(self, paper_ids: List[str]) -> List[str]:
        """Assembles APA-style bibliography entries."""
        entries = []
        papers = self.graph_repo.get_papers_batch(paper_ids)
        for pid in paper_ids:
            paper = papers.get(pid)
            if not paper or not paper.title:
                continue
            authors_str = ""
            if paper.authors:
                authors_str = ", ".join(paper.authors) + "."
            year_str = f"({paper.year})." if paper.year else ""
            doi_str = f" https://doi.org/{paper.doi}" if paper.doi else ""
            entry = f"{authors_str} {year_str} *{paper.title}*.{doi_str}"
            entries.append(entry.strip())
        return entries

    def _render_report(
        self,
        topic: str,
        sections: Dict[str, str],
        comparison_table: str,
        bibliography: List[str],
    ) -> str:
        """Assembles the final Markdown document."""
        now = datetime.now().strftime("%Y-%m-%d")
        toc_lines = []
        section_lines = []

        for i, (name, text) in enumerate(sections.items(), start=1):
            anchor = re.sub(r'[^\w\s-]', '', name).strip().lower().replace(' ', '-')
            toc_lines.append(f"  {i}. [{name}](#{anchor})")
            section_lines.append(f"## {name}\n\n{text}\n")

        toc = "\n".join(toc_lines)

        bib_block = ""
        if bibliography:
            bib_items = "\n".join(f"{i}. {e}" for i, e in enumerate(bibliography, start=1))
            bib_block = f"\n## References\n\n{bib_items}\n"

        table_block = ""
        if comparison_table:
            table_block = f"\n## Papers Comparison\n\n{comparison_table}\n"

        sections_text = "\n".join(section_lines)

        report = f"""# Literature Review: {topic}

> **Generated:** {now}  
> **Tool:** Science Graph Auto-Review Agent  

---

## Table of Contents

{toc}
  - [Papers Comparison](#papers-comparison)
  - [References](#references)

---

{sections_text}
{table_block}
{bib_block}
---
*This report was automatically generated by the Science Graph Auto-Review Agent.*
"""
        return report
