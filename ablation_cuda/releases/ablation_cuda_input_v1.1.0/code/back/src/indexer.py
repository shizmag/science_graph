"""
Indexer — Thin Orchestrator for the Ingestion Pipeline.

Workflow for every document type:
    Parse → Enrich Metadata → Extract Concepts → Chunk & Embed → Save Graph → Archive

Parser-specific logic lives in src/parsers/*.
Knowledge extraction lives in ExtractionService.
Metadata enrichment lives in MetadataEnricher.
All graph writes go through GraphRepository / VectorRepository interfaces.
No direct sqlite3 access.
"""

import os
import shutil
import time
import contextlib
import asyncio
import threading
import hashlib
import re
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

from src.models import Paper, Author, Concept, slugify
from src.vector_search import EmbeddingEngine, split_text_to_chunks
from src.repository.base import GraphRepository, VectorRepository
from src.services.extraction_service import ExtractionService, ExtractionResult
from src.services.metadata_enricher import MetadataEnricher
from src.parsers.factory import ParserFactory
from src.parsers.marker_parser import marker_session
from src.config import config
from src import console as con
from src.services.duplicate_detector import DuplicateDetector, _split_text_to_chunks_raw
from src.services.bibliographic import (
    canonicalize_reference,
    resolve_reference_target,
    find_citation_context_in_text,
    BibliographicProjectionService,
)


class DuplicateDocumentError(ValueError):
    """Exception raised when an ingested document is detected as already existing in the database."""
    def __init__(self, message: str, duplicate_paper_id: str):
        super().__init__(message)
        self.duplicate_paper_id = duplicate_paper_id


def create_progress(description: str, total: int):
    import sys
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        ProgressColumn,
    )
    from rich.text import Text

    class IterationSpeedColumn(ProgressColumn):
        def render(self, task):
            speed = task.finished_speed or task.speed
            if speed is None or speed == 0:
                return Text("- sec/it", style="progress.data.speed")
            sec_per_it = 1.0 / speed
            return Text(f"{sec_per_it:.2f} sec/it", style="progress.data.speed")

    is_testing = "pytest" in sys.modules or "unittest" in sys.modules
    disable_progress = is_testing or (not con.console.is_terminal)

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, finished_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        IterationSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=con.console,
        disable=disable_progress,
    )
    task = progress.add_task(description, total=total)
    return progress, task


class Indexer:
    """Orchestrates document parsing, enrichment, extraction, chunking, and graph storage."""

    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        llm_engine: Any = None,
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine

        self._extractor = ExtractionService(llm_engine=llm_engine)
        self._enricher = MetadataEnricher()
        self._duplicate_detector = DuplicateDetector(graph_repo, vector_repo, embedding_engine)

        self._aliases_cache: Optional[Dict[str, str]] = None
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Public Ingestion Interface (Single Document)
    # -------------------------------------------------------------------------

    def index_pdf(
        self,
        file_path: str,
        trace_info: Optional[dict] = None,
        pdf_parser_type: Optional[str] = None,
    ) -> str:
        """Runs the complete ingestion pipeline for a single PDF. Returns the paper ID."""
        kwargs = {}
        if pdf_parser_type is not None:
            kwargs["pdf_parser_type"] = pdf_parser_type
        return asyncio.run(self.index_pdf_async(file_path, trace_info, **kwargs))

    async def index_pdf_async(
        self,
        file_path: str,
        trace_info: Optional[dict] = None,
        pdf_parser_type: Optional[str] = None,
    ) -> str:
        """Runs the complete ingestion pipeline for a single PDF asynchronously."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        con.info(f"Parsing [bold]{os.path.basename(file_path)}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(file_path, pdf_parser_type=pdf_parser_type)
            paper, raw_references, full_text = parser.parse(file_path)

        archive_dir = Path(config.archive_dir)
        archive_path = archive_dir / f"{paper.id}.pdf"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        paper.file_path = str(archive_path)

        if len(paper.authors) < 2:
            with self._trace_stage("NER Author Fallback", trace_info):
                paper.authors = self._ner_fallback_authors(paper.authors, file_path)

        def _archive():
            self._archive_pdf(file_path, archive_path)

        return await self._run_pipeline_async(
            paper=paper,
            full_text=full_text,
            refs_or_links=raw_references,
            is_markdown=False,
            needs_enrichment=True,
            archive_fn=_archive,
            source_path=file_path,
            trace_info=trace_info,
        )

    def index_markdown(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Indexes a Markdown note (.md) into the knowledge graph. Returns the note ID."""
        return asyncio.run(self.index_markdown_async(file_path, trace_info))

    async def index_markdown_async(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Indexes a Markdown note (.md) asynchronously."""
        con.info(f"Parsing note [bold]{os.path.basename(file_path)}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(file_path)
            paper, wiki_links, body = parser.parse(file_path)

        return await self._run_pipeline_async(
            paper=paper,
            full_text=body,
            refs_or_links=wiki_links,
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None,
            source_path=file_path,
            trace_info=trace_info,
        )

    def index_epub(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Indexes an EPUB book into the knowledge graph. Returns the book ID."""
        return asyncio.run(self.index_epub_async(file_path, trace_info))

    async def index_epub_async(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Indexes an EPUB book asynchronously."""
        con.info(f"Parsing EPUB [bold]{os.path.basename(file_path)}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(file_path)
            paper, _, full_text = parser.parse(file_path)

        return await self._run_pipeline_async(
            paper=paper,
            full_text=full_text,
            refs_or_links=[],
            is_markdown=False,
            needs_enrichment=False,
            archive_fn=None,
            source_path=file_path,
            trace_info=trace_info,
        )

    def index_url(self, url: str, trace_info: Optional[dict] = None) -> str:
        """Indexes a webpage URL into the knowledge graph. Returns the page ID."""
        return asyncio.run(self.index_url_async(url, trace_info))

    async def index_url_async(self, url: str, trace_info: Optional[dict] = None) -> str:
        """Indexes a webpage URL asynchronously."""
        con.info(f"Parsing URL [bold]{url}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(url)
            paper, web_links, body = parser.parse(url)

        archive_dir = Path(config.archive_dir)
        archive_path = archive_dir / f"{paper.id}.md"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            archive_path.write_text(body, encoding="utf-8")
            paper.file_path = str(archive_path)
            con.dim(f"Saved local archive of website to {archive_path}")
        except Exception as e:
            con.warning(f"Could not save local archive of website: {e}")

        return await self._run_pipeline_async(
            paper=paper,
            full_text=body,
            refs_or_links=web_links,
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None,
            source_path=None,
            trace_info=trace_info,
        )

    # -------------------------------------------------------------------------
    # Public Ingestion Interface (Batch Ingestion)
    # -------------------------------------------------------------------------

    def index_batch(
        self,
        targets: List[str],
        use_llm: bool = True,
        trace: bool = False,
        chunk_pool_size: Optional[int] = None,
        pdf_parser_type: Optional[str] = None,
    ) -> List[dict]:
        """Synchronous wrapper for batch indexing."""
        return asyncio.run(
            self.index_batch_async(
                targets=targets,
                use_llm=use_llm,
                trace=trace,
                chunk_pool_size=chunk_pool_size,
                pdf_parser_type=pdf_parser_type,
            )
        )

    async def index_batch_async(
        self,
        targets: List[str],
        use_llm: bool = True,
        trace: bool = False,
        chunk_pool_size: Optional[int] = None,
        pdf_parser_type: Optional[str] = None,
    ) -> List[dict]:
        """Unified staged batch ingestion pipeline executing across 4 main stages."""
        self._configure_chunk_pool(chunk_pool_size)
        resolved_targets = self._resolve_batch_targets(targets)
        if not resolved_targets:
            con.warning("No valid targets found to index.")
            return []

        session_traces: List[dict] = []

        # Stage 1: Parsing
        parsed_items, parse_failed_traces = await self._parse_batch_items_async(
            resolved_targets, pdf_parser_type
        )
        session_traces.extend(parse_failed_traces)

        # Stage 1.5: Duplicate detection & filtering
        valid_items, duplicate_traces = await self._filter_batch_duplicates_async(parsed_items)
        session_traces.extend(duplicate_traces)

        # Archiving PDFs
        await self._archive_batch_items_async(valid_items)

        # Parallel Metadata Enrichment
        await self._enrich_batch_items_async(valid_items)

        # Stage 2: Chunking & Embedding
        embeddable_items, chunk_failed_traces = await self._chunk_and_embed_batch_items_async(valid_items)
        session_traces.extend(chunk_failed_traces)

        # Stage 3: LLM Concept Extraction & Summary Generation
        extractable_items, llm_failed_traces = await self._extract_and_summarize_batch_items_async(
            embeddable_items, use_llm
        )
        session_traces.extend(llm_failed_traces)

        # Stage 4: Database Persistence
        persisted_traces = await self._persist_batch_items_async(extractable_items)
        session_traces.extend(persisted_traces)

        return session_traces

    # -------------------------------------------------------------------------
    # Public Re-indexing Interface
    # -------------------------------------------------------------------------

    def reindex_metadata(self, paper_id: str, use_llm: bool = False) -> bool:
        """Re-indexes metadata for a specific paper without regenerating embeddings."""
        paper = self.graph_repo.get_paper(paper_id)
        if not paper:
            con.error(f"Paper not found in database: {paper_id}")
            return False

        con.info(f"Re-indexing metadata for [bold]{paper.title[:60]}[/bold] (ID: {paper_id})")
        self._refresh_paper_metadata_sources(paper)

        api_meta = self._enricher.enrich(paper)
        api_references: List[Dict] = []
        api_citations: List[Dict] = []
        if api_meta:
            paper, api_references, api_citations = self._enricher.apply(paper, api_meta)

        full_text = self._read_local_text(paper)
        text_for_extraction = full_text or f"{paper.title or ''}\n\n{paper.abstract or ''}"

        extraction = self._extractor.extract(
            title=paper.title or "",
            abstract=paper.abstract or "",
            full_text=text_for_extraction,
            use_llm=use_llm,
        )
        self._merge_extracted_authors_and_tags(paper, extraction)

        if len(paper.authors) < 2 and paper.file_path and os.path.exists(paper.file_path) and paper.file_path.lower().endswith(".pdf"):
            paper.authors = self._ner_fallback_authors(paper.authors, paper.file_path)

        with self.graph_repo.transaction():
            self._save_refreshed_metadata_to_graph(paper, extraction, api_references, api_citations)
            if use_llm and self.llm_engine:
                self._extractor.generate_summary(paper, text_for_extraction, graph_repo=self.graph_repo)

        con.success(f"Successfully re-indexed metadata for {(paper.title or paper.id)[:60]}")
        return True

    def reindex_full(self, paper_id: str, pdf_parser_type: Optional[str] = None) -> bool:
        """Performs full reindexing by deleting the paper node and re-indexing the original file/URL."""
        paper = self.graph_repo.get_paper(paper_id)
        if not paper:
            con.error(f"Paper not found in database: {paper_id}")
            return False

        file_path = paper.file_path
        if not file_path:
            con.error(f"No file path or URL stored for paper: {paper_id}")
            return False

        con.info(f"Performing full re-indexing for [bold]{(paper.title or paper_id)[:60]}[/bold] (ID: {paper_id})")

        try:
            self.graph_repo.delete_node(paper_id)
        except Exception as e:
            con.error(f"Failed to delete existing paper records: {e}")
            return False

        try:
            if file_path.startswith(("http://", "https://")):
                self.index_url(file_path)
            elif file_path.lower().endswith(".pdf"):
                kwargs = {}
                if pdf_parser_type is not None:
                    kwargs["pdf_parser_type"] = pdf_parser_type
                self.index_pdf(file_path, **kwargs)
            elif file_path.lower().endswith(".md"):
                self.index_markdown(file_path)
            elif file_path.lower().endswith(".epub"):
                self.index_epub(file_path)
            else:
                con.error(f"Unsupported file type for full re-indexing: {file_path}")
                return False
        except Exception as e:
            con.error(f"Full re-indexing failed: {e}")
            return False

        con.success(f"Successfully performed full re-index for {paper_id}")
        return True

    def reindex_metadata_batch(
        self,
        missing_authors: bool = False,
        missing_tags: bool = False,
        limit: Optional[int] = None,
        use_llm: bool = False,
        chunk_pool_size: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Batch re-indexes paper metadata (authors, tags, etc.) based on filters."""
        self._configure_chunk_pool(chunk_pool_size)
        candidates = self._find_reindex_candidates(missing_authors=missing_authors, missing_tags=missing_tags, limit=limit)

        if not candidates:
            con.success("No papers found matching the re-indexing criteria.")
            return 0, 0

        con.info(f"Starting metadata re-indexing for [bold]{len(candidates)}[/bold] papers …")
        success_count = 0
        progress, task = create_progress("[cyan]Metadata Re-indexing", total=len(candidates))

        with progress:
            with marker_session():
                for paper_id in candidates:
                    try:
                        if self.reindex_metadata(paper_id, use_llm=use_llm):
                            success_count += 1
                    except Exception as e:
                        con.error(f"Failed to re-index {paper_id}: {e}")
                    finally:
                        progress.advance(task, 1)

        con.blank()
        con.success(f"Re-indexed {success_count}/{len(candidates)} papers successfully.")
        return success_count, len(candidates)

    def reindex_full_batch(
        self,
        all_papers: bool = False,
        paper_id: Optional[str] = None,
        limit: Optional[int] = None,
        chunk_pool_size: Optional[int] = None,
        pdf_parser_type: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Batch fully re-indexes papers by re-ingesting original files/URLs."""
        self._configure_chunk_pool(chunk_pool_size)
        if paper_id:
            paper = self.graph_repo.get_paper(paper_id)
            if not paper:
                con.error(f"Paper not found: {paper_id}")
                raise ValueError(f"Paper not found: {paper_id}")
            candidates = [paper_id]
        else:
            candidates = self.graph_repo.get_non_placeholder_paper_ids()

        if limit:
            candidates = candidates[:limit]

        if not candidates:
            con.success("No papers found matching the re-indexing criteria.")
            return 0, 0

        con.info(f"Starting full re-indexing for [bold]{len(candidates)}[/bold] papers …")
        success_count = 0
        progress, task = create_progress("[cyan]Full Re-indexing", total=len(candidates))

        with progress:
            with marker_session():
                for pid in candidates:
                    try:
                        kwargs = {}
                        if pdf_parser_type is not None:
                            kwargs["pdf_parser_type"] = pdf_parser_type
                        if self.reindex_full(pid, **kwargs):
                            success_count += 1
                    except Exception as e:
                        con.error(f"Failed to fully re-index {pid}: {e}")
                    finally:
                        progress.advance(task, 1)

        con.blank()
        con.success(f"Fully re-indexed {success_count}/{len(candidates)} papers successfully.")
        return success_count, len(candidates)

    # -------------------------------------------------------------------------
    # Entity Resolution & Cache Management
    # -------------------------------------------------------------------------

    def invalidate_concept_cache(self) -> None:
        """Flushes the concept aliases and node cache."""
        with self._lock:
            self._aliases_cache = None
            if hasattr(self, "_entity_cache") and "Concept" in self._entity_cache:
                del self._entity_cache["Concept"]

    def resolve_entity(self, label: str, name: str) -> str:
        """Resolves an entity name of a given label to a canonical node ID."""
        if not name:
            return ""

        name_clean = name.strip()
        slug = slugify(name_clean)
        t0 = time.perf_counter()
        import logging

        if label == "Concept":
            alias_match = self._resolve_from_aliases(name_clean)
            if alias_match:
                logging.debug(f"resolve_entity '{name}' resolved from aliases_map in {time.perf_counter() - t0:.6f}s")
                return alias_match

        existing_nodes = self._get_cached_nodes_by_label(label)

        # 1. Exact slug match
        slug_match = self._resolve_by_slug_match(existing_nodes, slug)
        if slug_match:
            logging.debug(f"resolve_entity '{name}' exact slug match in {time.perf_counter() - t0:.6f}s")
            return slug_match

        # 2. Embedding similarity check
        vec_match = self._resolve_by_vector_similarity(existing_nodes, name_clean)
        if vec_match:
            logging.debug(f"resolve_entity '{name}' resolved by vectorized embedding similarity in {time.perf_counter() - t0:.6f}s")
            return vec_match

        # 3. String similarity fallback
        str_match = self._resolve_by_string_similarity(existing_nodes, name_clean)
        if str_match:
            logging.debug(f"resolve_entity '{name}' resolved by string similarity fallback in {time.perf_counter() - t0:.6f}s")
            return str_match

        logging.debug(f"resolve_entity '{name}' fallback to slug in {time.perf_counter() - t0:.6f}s")
        return slug

    def detect_duplicate(self, paper: Paper, full_text: str) -> Optional[Tuple[str, str]]:
        """Detects if the given paper/document is already present in the database."""
        return self._duplicate_detector.detect_duplicate(paper, full_text)

    # -------------------------------------------------------------------------
    # Unified Ingestion Pipeline Orchestration (Single Doc)
    # -------------------------------------------------------------------------

    def _run_pipeline(
        self,
        paper: Paper,
        full_text: str,
        refs_or_links: List[str],
        is_markdown: bool,
        needs_enrichment: bool,
        archive_fn: Optional[Callable[[], None]],
        source_path: Optional[str] = None,
        trace_info: Optional[dict] = None,
    ) -> str:
        return asyncio.run(
            self._run_pipeline_async(
                paper=paper,
                full_text=full_text,
                refs_or_links=refs_or_links,
                is_markdown=is_markdown,
                needs_enrichment=needs_enrichment,
                archive_fn=archive_fn,
                source_path=source_path,
                trace_info=trace_info,
            )
        )

    async def _run_pipeline_async(
        self,
        paper: Paper,
        full_text: str,
        refs_or_links: List[str],
        is_markdown: bool,
        needs_enrichment: bool,
        archive_fn: Optional[Callable[[], None]],
        source_path: Optional[str] = None,
        trace_info: Optional[dict] = None,
    ) -> str:
        """Asynchronous unified ingestion pipeline using a DAG execution flow."""
        dup_info = await asyncio.to_thread(self.detect_duplicate, paper, full_text)
        if dup_info:
            dup_id, reason = dup_info
            raise DuplicateDocumentError(
                f"Document already exists in database (ID: {dup_id}, match: {reason})",
                duplicate_paper_id=dup_id,
            )

        api_references: List[Dict] = []
        api_citations: List[Dict] = []

        async def path_a_enrichment():
            nonlocal paper, api_references, api_citations
            if needs_enrichment:
                t0 = time.perf_counter()
                api_meta = await self._enricher.enrich_async(paper)
                if api_meta:
                    paper, api_references, api_citations = self._enricher.apply(paper, api_meta)
                self._record_stage_time("Metadata Enrichment", time.perf_counter() - t0, trace_info)
            return paper, api_references, api_citations

        async def path_b_extraction():
            t0 = time.perf_counter()
            extraction = await self._extractor.extract_async(
                title=paper.title or "",
                abstract=paper.abstract or "",
                full_text=full_text,
                use_llm=True,
                trace_info=trace_info,
            )
            self._record_stage_time("Concept & Tag Extraction", time.perf_counter() - t0, trace_info)
            return extraction

        orig_size = os.path.getsize(source_path) if source_path and os.path.exists(source_path) else 0

        async def path_c_archive():
            if archive_fn is not None:
                t0 = time.perf_counter()
                await asyncio.to_thread(archive_fn)
                self._record_stage_time("Archiving", time.perf_counter() - t0, trace_info)

        async def path_d_summary():
            t0 = time.perf_counter()
            summary = await self._extractor.generate_summary_async(paper, full_text, graph_repo=None, trace_info=trace_info)
            self._record_stage_time("Summary Generation", time.perf_counter() - t0, trace_info)
            return summary

        enrich_fut = asyncio.create_task(path_a_enrichment())
        extract_fut = asyncio.create_task(path_b_extraction())
        archive_fut = asyncio.create_task(path_c_archive())
        summary_fut = asyncio.create_task(path_d_summary())

        _, extraction, _, summary_text = await asyncio.gather(
            enrich_fut, extract_fut, archive_fut, summary_fut
        )

        if summary_text:
            paper.properties["summary"] = summary_text

        if orig_size > 0 and paper.file_path and os.path.exists(paper.file_path):
            if trace_info is not None:
                trace_info["original_size"] = orig_size
                trace_info["compressed_size"] = os.path.getsize(paper.file_path)

        self._merge_extracted_authors_and_tags(paper, extraction)

        # Chunk & embed
        t0 = time.perf_counter()
        con.dim(f"Chunking and embedding: {(paper.title or paper.id)[:60]}")
        chunks = await self._create_and_embed_chunks_async(paper, full_text, source_path, archive_fn is not None)
        self._record_stage_time("Chunking & Embedding", time.perf_counter() - t0, trace_info)

        # Build Graph Writes
        t0 = time.perf_counter()
        nodes_to_write, edges_to_write = await self._build_graph_writes_async(
            paper=paper,
            extraction=extraction,
            full_text=full_text,
            is_markdown=is_markdown,
            refs_or_links=refs_or_links,
            api_references=api_references,
            api_citations=api_citations,
        )

        chunk_mentions = self._append_chunk_nodes_and_citations(
            paper=paper,
            chunks=chunks,
            is_markdown=is_markdown,
            refs_or_links=refs_or_links,
            api_references=api_references,
            nodes_to_write=nodes_to_write,
            edges_to_write=edges_to_write,
        )

        def _write_bulk_db():
            with self.graph_repo.transaction():
                self.graph_repo.delete_chunk_nodes_for_paper(paper.id)
                self.graph_repo.delete_chunk_reference_mentions_for_paper(paper.id)
                self.graph_repo.delete_edges_by_source(paper.id, ["CITES", "HAS_CHUNK", "RELATED_TO"])
                self.graph_repo.save_nodes_bulk(nodes_to_write)
                self.graph_repo.save_edges_bulk(edges_to_write)
                if chunk_mentions:
                    self.graph_repo.save_chunk_reference_mentions(chunk_mentions)
            self.invalidate_concept_cache()
            if chunks:
                self.vector_repo.save_chunks_bulk(chunks)

        await asyncio.to_thread(_write_bulk_db)

        bib_service = BibliographicProjectionService(self.graph_repo)
        await asyncio.to_thread(bib_service.rebuild_projection)

        self._record_stage_time("Graph & Vector Persistence", time.perf_counter() - t0, trace_info)
        if trace_info is not None:
            trace_info["authors_count"] = len(paper.authors)
            trace_info["concepts_count"] = len(extraction.concepts)
            trace_info["tags_count"] = len(paper.properties.get("tags") or [])
            trace_info["references_count"] = len(edges_to_write)

        con.success(f"Indexed [bold]{(paper.title or paper.id)[:70]}[/bold] (ID: {paper.id[:12]}…)")
        return paper.id

    # -------------------------------------------------------------------------
    # Graph Construction Sub-methods
    # -------------------------------------------------------------------------

    async def _build_graph_writes_async(
        self,
        paper: Paper,
        extraction: ExtractionResult,
        full_text: str,
        is_markdown: bool,
        refs_or_links: List[str],
        api_references: List[Dict],
        api_citations: List[Dict],
    ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Constructs all nodes and edges to write to the graph for an ingested document."""
        nodes_to_write: List[Tuple[str, str, Dict[str, Any]]] = []
        edges_to_write: List[Tuple[str, str, str, Dict[str, Any]]] = []
        added_node_ids: set = set()

        content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        paper.properties["content_hash"] = content_hash

        paper_props = {
            **paper.properties,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "doi": paper.doi,
            "abstract": paper.abstract,
            "file_path": paper.file_path,
            "created_at": paper.created_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        paper_label = "UserNote" if paper.properties.get("source_type") == "note" else "Paper"
        nodes_to_write.append((paper.id, paper_label, paper_props))
        added_node_ids.add(paper.id)

        self._build_authors_writes(paper, nodes_to_write, edges_to_write, added_node_ids)
        await self._build_concepts_writes_async(extraction, paper.id, nodes_to_write, edges_to_write, added_node_ids)
        await self._build_tags_writes_async(paper.properties.get("tags") or [], paper.id, nodes_to_write, edges_to_write, added_node_ids)
        await self._build_institutions_writes_async(extraction, paper.id, nodes_to_write, edges_to_write, added_node_ids)
        await self._build_datasets_writes_async(extraction, paper.id, nodes_to_write, edges_to_write, added_node_ids)
        self._build_code_repos_writes(extraction, paper.id, nodes_to_write, edges_to_write, added_node_ids)
        await self._build_publication_writes_async(extraction, paper.id, nodes_to_write, edges_to_write, added_node_ids)
        await self._build_concept_relations_writes_async(extraction, nodes_to_write, edges_to_write, added_node_ids)

        if paper_label == "UserNote":
            await self._build_user_note_relations_writes_async(paper, nodes_to_write, edges_to_write, added_node_ids)

        await self._build_citations_writes_async(
            paper=paper,
            full_text=full_text,
            is_markdown=is_markdown,
            refs_or_links=refs_or_links,
            api_references=api_references,
            api_citations=api_citations,
            nodes_to_write=nodes_to_write,
            edges_to_write=edges_to_write,
            added_node_ids=added_node_ids,
        )

        return nodes_to_write, edges_to_write

    def _build_authors_writes(
        self, paper: Paper, nodes: list, edges: list, added: set
    ) -> None:
        for author_name in paper.authors:
            author_id = slugify(author_name)
            nodes.append((author_id, "Author", {"name": author_name}))
            edges.append((author_id, paper.id, "AUTHORED", {}))
            added.add(author_id)

    async def _build_concepts_writes_async(
        self, extraction: ExtractionResult, paper_id: str, nodes: list, edges: list, added: set
    ) -> None:
        concepts = getattr(extraction, "concepts", [])
        if not isinstance(concepts, list):
            return
        for item in concepts:
            if not isinstance(item, dict):
                continue
            c_name = item.get("name", "").strip() if isinstance(item.get("name"), str) else ""
            if not c_name:
                continue
            c_desc = item.get("description", "") if isinstance(item.get("description"), str) else ""
            if not c_desc:
                c_desc = await self._extractor.get_concept_description_async(c_name)

            concept_id = self.resolve_entity("Concept", c_name)
            if concept_id not in added:
                emb = await self._safe_get_embedding_async(c_name)
                nodes.append((concept_id, "Concept", {"name": c_name, "description": c_desc, "embedding": emb}))
                self._add_resolved_entity_to_cache("Concept", concept_id, c_name, emb)
                added.add(concept_id)
            edges.append((paper_id, concept_id, "MENTIONS_CONCEPT", {}))

    async def _build_tags_writes_async(
        self, tags: List[str], paper_id: str, nodes: list, edges: list, added: set
    ) -> None:
        for tag in tags:
            tag_id = self.resolve_entity("Concept", tag)
            if tag_id not in added:
                tag_desc = await self._extractor.get_concept_description_async(tag)
                emb = await self._safe_get_embedding_async(tag)
                nodes.append((tag_id, "Concept", {"name": tag, "is_tag": True, "description": tag_desc, "embedding": emb}))
                self._add_resolved_entity_to_cache("Concept", tag_id, tag, emb)
                added.add(tag_id)
            edges.append((paper_id, tag_id, "HAS_TAG", {}))

    async def _build_institutions_writes_async(
        self, extraction: ExtractionResult, paper_id: str, nodes: list, edges: list, added: set
    ) -> None:
        institutions = getattr(extraction, "institutions", [])
        if isinstance(institutions, list):
            for inst_name in institutions:
                if isinstance(inst_name, str):
                    await self._ensure_entity_node_async("Institution", inst_name, nodes, added)

        sponsored_by = getattr(extraction, "sponsored_by", [])
        if isinstance(sponsored_by, list):
            for inst_name in sponsored_by:
                if isinstance(inst_name, str):
                    inst_id = await self._ensure_entity_node_async("Institution", inst_name, nodes, added)
                    edges.append((paper_id, inst_id, "SPONSORED_BY", {}))

        author_institutions = getattr(extraction, "author_institutions", [])
        if isinstance(author_institutions, list):
            for auth_inst in author_institutions:
                if not isinstance(auth_inst, dict):
                    continue
                auth_name = auth_inst.get("author")
                inst_name = auth_inst.get("institution")
                if auth_name and inst_name and isinstance(auth_name, str) and isinstance(inst_name, str):
                    auth_id = slugify(auth_name)
                    inst_id = await self._ensure_entity_node_async("Institution", inst_name, nodes, added)
                    edges.append((auth_id, inst_id, "AFFILIATED_WITH", {}))

    async def _build_datasets_writes_async(
        self, extraction: ExtractionResult, paper_id: str, nodes: list, edges: list, added: set
    ) -> None:
        datasets = getattr(extraction, "datasets", [])
        if isinstance(datasets, list):
            for ds in datasets:
                if not isinstance(ds, dict):
                    continue
                ds_name = ds.get("name")
                relation = ds.get("relation", "USED_DATASET")
                if ds_name and isinstance(ds_name, str):
                    ds_id = await self._ensure_entity_node_async("Dataset", ds_name, nodes, added)
                    edges.append((paper_id, ds_id, relation, {}))

    def _build_code_repos_writes(
        self, extraction: ExtractionResult, paper_id: str, nodes: list, edges: list, added: set
    ) -> None:
        code_repositories = getattr(extraction, "code_repositories", [])
        if isinstance(code_repositories, list):
            for repo_url in code_repositories:
                if isinstance(repo_url, str):
                    repo_id = slugify(repo_url)
                    if repo_id not in added:
                        nodes.append((repo_id, "CodeRepository", {"name": repo_url, "url": repo_url}))
                        added.add(repo_id)
                    edges.append((paper_id, repo_id, "HAS_CODE", {}))

    async def _build_publication_writes_async(
        self, extraction: ExtractionResult, paper_id: str, nodes: list, edges: list, added: set
    ) -> None:
        jc_name = getattr(extraction, "journal_or_conference", None)
        if jc_name and isinstance(jc_name, str):
            jc_id = await self._ensure_entity_node_async("JournalConference", jc_name, nodes, added)
            edges.append((paper_id, jc_id, "PUBLISHED_IN", {}))

    async def _build_concept_relations_writes_async(
        self, extraction: ExtractionResult, nodes: list, edges: list, added: set
    ) -> None:
        concept_relations = getattr(extraction, "concept_relations", [])
        if isinstance(concept_relations, list):
            for rel in concept_relations:
                if not isinstance(rel, dict):
                    continue
                src_c = rel.get("source", "").strip() if isinstance(rel.get("source"), str) else ""
                tgt_c = rel.get("target", "").strip() if isinstance(rel.get("target"), str) else ""
                rel_type = rel.get("relation_type", "SUBCLASS_OF")
                if src_c and tgt_c:
                    src_id = await self._ensure_entity_node_async("Concept", src_c, nodes, added)
                    tgt_id = await self._ensure_entity_node_async("Concept", tgt_c, nodes, added)
                    edges.append((src_id, tgt_id, rel_type, {}))

    async def _build_user_note_relations_writes_async(
        self, paper: Paper, nodes: list, edges: list, added: set
    ) -> None:
        for rel_name, edge_type in [
            ("comments_on", "COMMENTS_ON"),
            ("agrees_with", "AGREES_WITH"),
            ("disagrees_with", "DISAGREES_WITH"),
        ]:
            for target_title in paper.properties.get(rel_name, []) or []:
                edges.append((paper.id, slugify(target_title), edge_type, {}))

        for concept_name in paper.properties.get("linked_to", []) or []:
            concept_id = await self._ensure_entity_node_async("Concept", concept_name, nodes, added)
            edges.append((paper.id, concept_id, "LINKED_TO", {}))

    async def _build_citations_writes_async(
        self,
        paper: Paper,
        full_text: str,
        is_markdown: bool,
        refs_or_links: List[str],
        api_references: List[Dict],
        api_citations: List[Dict],
        nodes_to_write: list,
        edges_to_write: list,
        added_node_ids: set,
    ) -> None:
        if is_markdown:
            for link_target in refs_or_links:
                if link_target.startswith(("http://", "https://")):
                    clean_target = link_target.replace("https://", "").replace("http://", "").strip("/")
                    target_id = slugify(clean_target)
                else:
                    target_id = slugify(link_target)
                if not target_id:
                    continue
                if target_id not in added_node_ids:
                    nodes_to_write.append((target_id, "Paper", {"title": link_target, "is_placeholder": True}))
                    added_node_ids.add(target_id)
                edges_to_write.append((paper.id, target_id, "RELATED_TO", {}))
            return

        cites_list = []
        ref_sources = self._collect_ref_sources(refs_or_links, api_references)

        for raw_ref, meta_dict in ref_sources:
            canonical_ref = canonicalize_reference(raw_ref, meta_dict)
            target_id, is_local = resolve_reference_target(self.graph_repo, canonical_ref)

            if not is_local and target_id not in added_node_ids:
                nodes_to_write.append((
                    target_id,
                    "ExternalWork",
                    {
                        "indexed": False,
                        "source_type": "external_work",
                        "title": canonical_ref.title or "",
                        "normalized_title": canonical_ref.normalized_title or "",
                        "doi": canonical_ref.doi or "",
                        "arxiv_id": canonical_ref.arxiv_id or "",
                        "url": canonical_ref.url or "",
                        "year": canonical_ref.year or "",
                        "authors": canonical_ref.authors or [],
                        "raw_reference": canonical_ref.raw_reference or "",
                        "canonicalization_method": canonical_ref.canonicalization_method,
                        "created_from_paper_id": paper.id,
                        "has_local_fulltext": False,
                    },
                ))
                added_node_ids.add(target_id)

            year_val = self._parse_year_int(canonical_ref.year)
            cites_list.append({
                "source_id": paper.id,
                "target_id": target_id,
                "title": canonical_ref.title or "",
                "author": canonical_ref.authors[0] if canonical_ref.authors else None,
                "year": year_val,
                "properties": {
                    "observed": True,
                    "source": "api_references" if api_references else "reference_list",
                    "reference_id": canonical_ref.work_id,
                    "raw_reference": canonical_ref.raw_reference,
                    "resolver": "local_paper" if is_local else canonical_ref.canonicalization_method,
                    "target_indexed": is_local,
                    "weight": 1.0,
                    "explanation": "Local paper cites this work.",
                },
            })

        for cit in api_citations:
            cit_title = cit.get("title")
            cit_doi = cit.get("doi")
            cit_id = slugify(cit_doi) if cit_doi else (slugify(cit_title[:120]) if cit_title else None)
            if cit_id:
                if cit_id not in added_node_ids:
                    nodes_to_write.append((cit_id, "Paper", {"title": cit_title, "doi": cit_doi, "is_placeholder": True}))
                    added_node_ids.add(cit_id)
                cites_list.append({
                    "source_id": cit_id,
                    "target_id": paper.id,
                    "title": cit_title,
                    "author": cit["authors"][0] if cit.get("authors") else None,
                    "year": cit.get("year"),
                    "properties": {"api_sourced": True},
                })

        classified_cites_edges = await self._classify_cites_edges_async(cites_list, full_text)
        edges_to_write.extend(classified_cites_edges)

    def _append_chunk_nodes_and_citations(
        self,
        paper: Paper,
        chunks: List[Any],
        is_markdown: bool,
        refs_or_links: List[str],
        api_references: List[Dict],
        nodes_to_write: list,
        edges_to_write: list,
    ) -> List[tuple]:
        chunk_mentions = []
        if not chunks:
            return chunk_mentions

        for chunk in chunks:
            nodes_to_write.append((
                chunk.id,
                "Chunk",
                {
                    "paper_id": chunk.paper_id,
                    "parent_id": chunk.parent_id,
                    "page_number": chunk.page_number,
                    "source_type": "chunk",
                    "has_text_in_chunks_table": True,
                },
            ))
            edges_to_write.append((paper.id, chunk.id, "HAS_CHUNK", {}))
            if chunk.parent_id:
                edges_to_write.append((chunk.parent_id, chunk.id, "HAS_CHUNK", {}))

            if not is_markdown:
                ref_sources = self._collect_ref_sources(refs_or_links, api_references)
                for idx, (raw_ref, meta_dict) in enumerate(ref_sources):
                    canonical_ref = canonicalize_reference(raw_ref, meta_dict)
                    target_id, is_local = resolve_reference_target(self.graph_repo, canonical_ref)

                    marker = f"[{idx + 1}]"
                    context = find_citation_context_in_text(chunk.text_content, canonical_ref, marker)
                    if context:
                        edges_to_write.append((
                            chunk.id,
                            target_id,
                            "CITES_IN_CONTEXT",
                            {
                                "observed": True,
                                "paper_id": paper.id,
                                "chunk_id": chunk.id,
                                "reference_id": canonical_ref.work_id,
                                "raw_reference": canonical_ref.raw_reference,
                                "citation_marker": marker,
                                "context": context,
                                "page_number": chunk.page_number,
                                "section": "",
                                "target_indexed": is_local,
                                "weight": 1.0,
                                "explanation": "This local chunk cites or discusses the target work.",
                            },
                        ))
                        chunk_mentions.append((
                            chunk.id,
                            paper.id,
                            target_id,
                            marker,
                            context,
                            chunk.page_number,
                            "",
                            canonical_ref.raw_reference,
                        ))
        return chunk_mentions

    # -------------------------------------------------------------------------
    # Batch Pipeline Helper Stages
    # -------------------------------------------------------------------------

    def _resolve_batch_targets(self, targets: List[str]) -> List[dict]:
        resolved = []
        for tgt in targets:
            tgt_clean = tgt.strip()
            if not tgt_clean:
                continue
            if tgt_clean.startswith(("http://", "https://")):
                resolved.append({"target": tgt_clean, "type": "url"})
            else:
                path = Path(tgt_clean).resolve()
                if not path.exists():
                    con.error(f"Path not found: {path}")
                    raise FileNotFoundError(f"Path not found: {path}")
                if path.is_file():
                    ext = path.suffix.lower().lstrip(".")
                    if ext in ("pdf", "md", "epub"):
                        resolved.append({"target": str(path), "type": ext})
                    else:
                        con.warning(f"Unsupported file type '{ext}' for {path.name}, skipping.")
                elif path.is_dir():
                    allowed = {".pdf", ".md", ".epub"}
                    files = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in allowed]
                    for f in files:
                        resolved.append({"target": str(f), "type": f.suffix.lower().lstrip(".")})
        return resolved

    async def _parse_batch_items_async(
        self, resolved_targets: List[dict], pdf_parser_type: Optional[str]
    ) -> Tuple[List[dict], List[dict]]:
        parsed_items: List[dict] = []
        failed_traces: List[dict] = []

        progress, task = create_progress("[cyan]Parsing Stage", total=len(resolved_targets))
        with progress:
            with marker_session():
                for item in resolved_targets:
                    tgt = item["target"]
                    t = item["type"]
                    trace_info = {
                        "stages": {},
                        "tokens": {},
                        "success": False,
                        "name": os.path.basename(tgt) if t != "url" else tgt,
                    }
                    try:
                        p_item = await self._parse_single_batch_target_async(item, pdf_parser_type, trace_info)
                        parsed_items.append(p_item)
                    except Exception as e:
                        con.error(f"Failed to parse target {tgt}: {e}")
                        trace_info["success"] = False
                        failed_traces.append(trace_info)
                    finally:
                        progress.advance(task, 1)

        return parsed_items, failed_traces

    async def _parse_single_batch_target_async(
        self, item: dict, pdf_parser_type: Optional[str], trace_info: dict
    ) -> dict:
        tgt = item["target"]
        t = item["type"]
        orig_size = os.path.getsize(tgt) if (t != "url" and os.path.exists(tgt)) else 0

        if t == "pdf":
            con.info(f"Parsing [bold]{os.path.basename(tgt)}[/bold]")
            t0 = time.perf_counter()
            parser = ParserFactory.get_parser(tgt, pdf_parser_type=pdf_parser_type)
            paper, raw_references, full_text = await asyncio.to_thread(parser.parse, tgt)
            trace_info["stages"]["Document Parsing"] = time.perf_counter() - t0

            archive_dir = Path(config.archive_dir)
            archive_path = archive_dir / f"{paper.id}.pdf"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            paper.file_path = str(archive_path)

            if len(paper.authors) < 2:
                t0_ner = time.perf_counter()
                paper.authors = await asyncio.to_thread(self._ner_fallback_authors, paper.authors, tgt)
                trace_info["stages"]["NER Author Fallback"] = time.perf_counter() - t0_ner

            def _archive(t_path=tgt, a_path=archive_path):
                self._archive_pdf(t_path, a_path)

            return {
                "item": item,
                "paper": paper,
                "full_text": full_text,
                "refs_or_links": raw_references,
                "is_markdown": False,
                "needs_enrichment": True,
                "archive_fn": _archive,
                "source_path": tgt,
                "orig_size": orig_size,
                "trace_info": trace_info,
            }
        elif t == "md":
            con.info(f"Parsing note [bold]{os.path.basename(tgt)}[/bold]")
            t0 = time.perf_counter()
            parser = ParserFactory.get_parser(tgt)
            paper, wiki_links, body = await asyncio.to_thread(parser.parse, tgt)
            trace_info["stages"]["Document Parsing"] = time.perf_counter() - t0
            return {
                "item": item,
                "paper": paper,
                "full_text": body,
                "refs_or_links": wiki_links,
                "is_markdown": True,
                "needs_enrichment": False,
                "archive_fn": None,
                "source_path": tgt,
                "orig_size": orig_size,
                "trace_info": trace_info,
            }
        elif t == "epub":
            con.info(f"Parsing EPUB [bold]{os.path.basename(tgt)}[/bold]")
            t0 = time.perf_counter()
            parser = ParserFactory.get_parser(tgt)
            paper, _, full_text = await asyncio.to_thread(parser.parse, tgt)
            trace_info["stages"]["Document Parsing"] = time.perf_counter() - t0
            return {
                "item": item,
                "paper": paper,
                "full_text": full_text,
                "refs_or_links": [],
                "is_markdown": False,
                "needs_enrichment": False,
                "archive_fn": None,
                "source_path": tgt,
                "orig_size": orig_size,
                "trace_info": trace_info,
            }
        elif t == "url":
            con.info(f"Parsing URL [bold]{tgt}[/bold]")
            t0 = time.perf_counter()
            parser = ParserFactory.get_parser(tgt)
            paper, web_links, body = await asyncio.to_thread(parser.parse, tgt)
            trace_info["stages"]["Document Parsing"] = time.perf_counter() - t0

            archive_dir = Path(config.archive_dir)
            archive_path = archive_dir / f"{paper.id}.md"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                await asyncio.to_thread(archive_path.write_text, body, encoding="utf-8")
                paper.file_path = str(archive_path)
                con.dim(f"Saved local archive of website to {archive_path}")
            except Exception as e:
                con.warning(f"Could not save local archive of website: {e}")

            return {
                "item": item,
                "paper": paper,
                "full_text": body,
                "refs_or_links": web_links,
                "is_markdown": True,
                "needs_enrichment": False,
                "archive_fn": None,
                "source_path": None,
                "trace_info": trace_info,
            }
        else:
            raise ValueError(f"Unknown target type: {t}")

    async def _filter_batch_duplicates_async(
        self, parsed_items: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        valid_items: List[dict] = []
        duplicate_traces: List[dict] = []

        for p_item in parsed_items:
            paper = p_item["paper"]
            full_text = p_item["full_text"]
            trace_info = p_item["trace_info"]

            dup_info = await asyncio.to_thread(self.detect_duplicate, paper, full_text)
            if dup_info:
                dup_id, reason = dup_info
                con.warning(f"Duplicate detected: Document already exists in database (ID: {dup_id}, match: {reason})")
                trace_info["skipped_duplicate"] = True
                trace_info["success"] = False
                duplicate_traces.append(trace_info)
            else:
                valid_items.append(p_item)

        return valid_items, duplicate_traces

    async def _archive_batch_items_async(self, valid_items: List[dict]) -> None:
        for p_item in valid_items:
            archive_fn = p_item["archive_fn"]
            trace_info = p_item["trace_info"]
            if archive_fn:
                t0_arch = time.perf_counter()
                await asyncio.to_thread(archive_fn)
                trace_info["stages"]["Archiving"] = time.perf_counter() - t0_arch

    async def _enrich_batch_items_async(self, valid_items: List[dict]) -> None:
        items_needing_enrichment = [p for p in valid_items if p["needs_enrichment"]]
        if not items_needing_enrichment:
            return

        progress, task = create_progress("[cyan]Metadata Enrichment Stage", total=len(items_needing_enrichment))
        with progress:
            async def enrich_item(p_item):
                paper = p_item["paper"]
                trace_info = p_item["trace_info"]
                t0_enrich = time.perf_counter()
                api_meta = await self._enricher.enrich_async(paper)
                if api_meta:
                    paper, api_refs, api_cits = self._enricher.apply(paper, api_meta)
                    p_item["paper"] = paper
                    p_item["api_references"] = api_refs
                    p_item["api_citations"] = api_cits
                trace_info["stages"]["Metadata Enrichment"] = time.perf_counter() - t0_enrich
                progress.advance(task, 1)

            tasks = [asyncio.create_task(enrich_item(p)) for p in items_needing_enrichment]
            await asyncio.gather(*tasks)

    async def _chunk_and_embed_batch_items_async(
        self, valid_items: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        failed_traces: List[dict] = []
        progress, chunk_task = create_progress("[cyan]Chunking Stage", total=len(valid_items))

        with progress:
            for p_item in valid_items:
                paper = p_item["paper"]
                full_text = p_item["full_text"]
                source_path = p_item["source_path"]
                trace_info = p_item["trace_info"]
                p_item["failed"] = False

                try:
                    t0_chunk = time.perf_counter()
                    con.dim(f"Chunking: {(paper.title or paper.id)[:60]}")
                    is_pdf_doc = (
                        p_item["archive_fn"] is not None
                        and (source_path or paper.file_path)
                        and (source_path or paper.file_path).endswith(".pdf")
                    )
                    chunks = await self._create_chunks_async(paper, full_text, source_path, is_pdf_doc)
                    p_item["chunks"] = chunks
                    p_item["chunk_time"] = time.perf_counter() - t0_chunk
                except Exception as e:
                    con.error(f"Failed during chunking for {paper.id}: {e}")
                    p_item["failed"] = True
                    trace_info["success"] = False
                    failed_traces.append(trace_info)
                finally:
                    progress.advance(chunk_task, 1)

            items_to_embed = [p for p in valid_items if not p.get("failed", False)]
            all_chunks = []
            for p_item in items_to_embed:
                all_chunks.extend(p_item["chunks"])

            if all_chunks:
                embed_task = progress.add_task("[cyan]Embedding Stage", total=len(all_chunks))
                try:
                    t0_embed = time.perf_counter()
                    embeddings = await asyncio.to_thread(
                        self.emb_engine.get_embeddings, [c.text_content for c in all_chunks]
                    )
                    total_embed_time = time.perf_counter() - t0_embed

                    idx = 0
                    for p_item in items_to_embed:
                        doc_chunks = p_item["chunks"]
                        for chunk in doc_chunks:
                            chunk.embedding = embeddings[idx]
                            idx += 1
                        prop_time = (
                            total_embed_time * (len(doc_chunks) / len(all_chunks))
                            if all_chunks
                            else 0.0
                        )
                        p_item["trace_info"]["stages"]["Chunking & Embedding"] = p_item["chunk_time"] + prop_time
                    progress.advance(embed_task, len(all_chunks))
                except Exception as e:
                    con.error(f"Failed to generate embeddings for batch: {e}")
                    for p_item in items_to_embed:
                        p_item["failed"] = True
                        p_item["trace_info"]["success"] = False
                        failed_traces.append(p_item["trace_info"])

        embeddable_items = [p for p in valid_items if not p.get("failed", False)]
        return embeddable_items, failed_traces

    async def _extract_and_summarize_batch_items_async(
        self, items: List[dict], use_llm: bool
    ) -> Tuple[List[dict], List[dict]]:
        failed_traces: List[dict] = []
        if not items:
            return [], failed_traces

        progress, task = create_progress("[cyan]Concept Extraction Stage", total=len(items))

        with progress:
            async def process_llm_for_item(p_item):
                paper = p_item["paper"]
                full_text = p_item["full_text"]
                trace_info = p_item["trace_info"]
                try:
                    t0_extract = time.perf_counter()
                    extraction = await self._extractor.extract_async(
                        title=paper.title or "",
                        abstract=paper.abstract or "",
                        full_text=full_text,
                        use_llm=use_llm,
                        trace_info=trace_info,
                    )
                    trace_info["stages"]["Concept & Tag Extraction"] = time.perf_counter() - t0_extract
                    p_item["extraction"] = extraction

                    t0_summary = time.perf_counter()
                    summary_text = await self._extractor.generate_summary_async(
                        paper, full_text, graph_repo=None, trace_info=trace_info
                    )
                    trace_info["stages"]["Summary Generation"] = time.perf_counter() - t0_summary
                    p_item["summary_text"] = summary_text
                except Exception as e:
                    con.error(f"Failed during LLM extraction/summary for {paper.id}: {e}")
                    p_item["failed"] = True
                    trace_info["success"] = False
                    failed_traces.append(trace_info)
                finally:
                    progress.advance(task, 1)

            tasks = [asyncio.create_task(process_llm_for_item(p)) for p in items]
            await asyncio.gather(*tasks)

        extractable_items = [p for p in items if not p.get("failed", False)]
        return extractable_items, failed_traces

    async def _persist_batch_items_async(self, items: List[dict]) -> List[dict]:
        session_traces: List[dict] = []
        if not items:
            return session_traces

        progress, task = create_progress("[cyan]Database Persistence Stage", total=len(items))
        with progress:
            for p_item in items:
                paper = p_item["paper"]
                full_text = p_item["full_text"]
                extraction = p_item["extraction"]
                summary_text = p_item["summary_text"]
                chunks = p_item["chunks"]
                is_markdown = p_item["is_markdown"]
                refs_or_links = p_item["refs_or_links"]
                api_references = p_item.get("api_references") or []
                api_citations = p_item.get("api_citations") or []
                trace_info = p_item["trace_info"]

                try:
                    t0_db = time.perf_counter()
                    if summary_text:
                        paper.properties["summary"] = summary_text

                    orig_size = p_item.get("orig_size", 0)
                    if orig_size > 0 and paper.file_path and os.path.exists(paper.file_path):
                        trace_info["original_size"] = orig_size
                        trace_info["compressed_size"] = os.path.getsize(paper.file_path)

                    self._merge_extracted_authors_and_tags(paper, extraction)

                    nodes_to_write, edges_to_write = await self._build_graph_writes_async(
                        paper=paper,
                        extraction=extraction,
                        full_text=full_text,
                        is_markdown=is_markdown,
                        refs_or_links=refs_or_links,
                        api_references=api_references,
                        api_citations=api_citations,
                    )

                    def _write_bulk_db():
                        with self.graph_repo.transaction():
                            self.graph_repo.save_nodes_bulk(nodes_to_write)
                            self.graph_repo.save_edges_bulk(edges_to_write)
                        if chunks:
                            self.vector_repo.save_chunks_bulk(chunks)

                    await asyncio.to_thread(_write_bulk_db)

                    trace_info["stages"]["Graph & Vector Persistence"] = time.perf_counter() - t0_db
                    trace_info["authors_count"] = len(paper.authors)
                    trace_info["concepts_count"] = len(extraction.concepts)
                    trace_info["tags_count"] = len(paper.properties.get("tags") or [])
                    trace_info["references_count"] = len(edges_to_write)
                    trace_info["success"] = True

                    session_traces.append(trace_info)
                    con.success(f"Indexed [bold]{(paper.title or paper.id)[:70]}[/bold] (ID: {paper.id[:12]}…)")
                except Exception as e:
                    con.error(f"Failed to write to database for {paper.id}: {e}")
                    trace_info["success"] = False
                    session_traces.append(trace_info)
                finally:
                    progress.advance(task, 1)

        return session_traces

    # -------------------------------------------------------------------------
    # Helper & Internal Utilities
    # -------------------------------------------------------------------------

    def _resolve_from_aliases(self, name_clean: str) -> Optional[str]:
        if self._aliases_cache is None:
            with self._lock:
                if self._aliases_cache is None:
                    try:
                        self._aliases_cache = self.graph_repo.get_concept_aliases()
                    except Exception:
                        self._aliases_cache = {}
        aliases_map = self._aliases_cache
        if name_clean.lower() in aliases_map:
            return slugify(aliases_map[name_clean.lower()])
        return None

    def _get_cached_nodes_by_label(self, label: str) -> list:
        if not hasattr(self, "_entity_cache"):
            with self._lock:
                if not hasattr(self, "_entity_cache"):
                    self._entity_cache = {}

        if label not in self._entity_cache:
            with self._lock:
                if label not in self._entity_cache:
                    try:
                        self._entity_cache[label] = self.graph_repo.get_nodes_by_label(label)
                    except Exception:
                        self._entity_cache[label] = []
        return self._entity_cache[label]

    def _resolve_by_slug_match(self, existing_nodes: list, target_slug: str) -> Optional[str]:
        for eid, props in existing_nodes:
            if eid == target_slug:
                return eid
            node_name = props.get("name", "")
            if node_name and slugify(node_name) == target_slug:
                return eid
        return None

    def _resolve_by_vector_similarity(self, existing_nodes: list, name_clean: str) -> Optional[str]:
        valid_candidates = [
            (eid, props["embedding"])
            for eid, props in existing_nodes
            if props.get("embedding") and (isinstance(props["embedding"], (list, np.ndarray)))
        ]
        if not valid_candidates:
            return None

        try:
            candidate_emb = self.emb_engine.get_embedding(name_clean)
        except Exception:
            candidate_emb = None

        if candidate_emb is None or not isinstance(candidate_emb, (list, np.ndarray)):
            return None

        query_vec = np.array(candidate_emb, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm <= 0:
            return None

        node_embs = np.array([emb for _, emb in valid_candidates], dtype=np.float32)
        node_norms = np.linalg.norm(node_embs, axis=1)
        dots = np.dot(node_embs, query_vec)
        norms_product = query_norm * node_norms

        sims = np.zeros_like(dots)
        valid_mask = norms_product > 0
        sims[valid_mask] = dots[valid_mask] / norms_product[valid_mask]

        matching_indices = np.where(sims > 0.95)[0]
        if len(matching_indices) > 0:
            return valid_candidates[matching_indices[0]][0]
        return None

    def _resolve_by_string_similarity(self, existing_nodes: list, name_clean: str) -> Optional[str]:
        for eid, props in existing_nodes:
            node_name = props.get("name", "")
            if node_name:
                ratio = difflib.SequenceMatcher(None, name_clean.lower(), node_name.lower()).ratio()
                if ratio > 0.95:
                    return eid
        return None

    def _add_resolved_entity_to_cache(
        self, label: str, entity_id: str, name: str, embedding: Optional[List[float]] = None
    ) -> None:
        existing_nodes = self._get_cached_nodes_by_label(label)
        exists = any(eid == entity_id for eid, _ in existing_nodes)
        if not exists:
            with self._lock:
                self._entity_cache[label].append((entity_id, {"name": name, "embedding": embedding}))
                if label == "Concept":
                    self._aliases_cache = None

    async def _ensure_entity_node_async(
        self, label: str, name: str, nodes: list, added: set
    ) -> str:
        entity_id = self.resolve_entity(label, name)
        if entity_id not in added:
            emb = await self._safe_get_embedding_async(name)
            nodes.append((entity_id, label, {"name": name, "embedding": emb}))
            self._add_resolved_entity_to_cache(label, entity_id, name, emb)
            added.add(entity_id)
        return entity_id

    async def _safe_get_embedding_async(self, text: str) -> Optional[List[float]]:
        try:
            val = await asyncio.to_thread(self.emb_engine.get_embedding, text, False)
            if isinstance(val, list):
                return val
        except Exception:
            pass
        return None

    def _collect_ref_sources(
        self, refs_or_links: List[str], api_references: List[Dict]
    ) -> List[Tuple[str, Optional[Dict]]]:
        ref_sources = []
        if api_references:
            for ref in api_references:
                raw = ref.get("raw_reference") or ref.get("title") or ""
                ref_sources.append((raw, ref))
        else:
            for ref_str in refs_or_links:
                ref_clean = ref_str.strip()
                if len(ref_clean) > 10:
                    ref_sources.append((ref_clean, None))
        return ref_sources

    def _parse_year_int(self, year_str: Optional[str]) -> Optional[int]:
        if not year_str:
            return None
        try:
            match = re.search(r"\d+", year_str)
            return int(match.group(0)) if match else None
        except Exception:
            return None

    def _get_citation_context(
        self, full_text: str, ref_title: str, ref_author: Optional[str] = None, ref_year: Optional[int] = None
    ) -> str:
        if not full_text or not ref_title:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", full_text)
        patterns = []
        if ref_title and len(ref_title) > 8:
            words = [re.escape(w) for w in ref_title.split()[:4] if len(w) > 2]
            if words:
                patterns.append(re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE))
        if ref_author and ref_year:
            patterns.append(re.compile(rf"\b{re.escape(ref_author)}.*\b{ref_year}\b", re.IGNORECASE))
        elif ref_author:
            patterns.append(re.compile(rf"\b{re.escape(ref_author)}\b", re.IGNORECASE))

        for idx, sent in enumerate(sentences):
            for pat in patterns:
                if pat.search(sent):
                    start = max(0, idx - 1)
                    end = min(len(sentences), idx + 2)
                    return " ".join(sentences[start:end]).strip()
        return ""

    async def _classify_cites_edges_async(
        self, cites_list: List[Dict[str, Any]], full_text: str
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        if not cites_list:
            return []

        tasks = []
        metadata = []

        for cit in cites_list:
            ref_title = cit.get("title") or ""
            ref_author = cit.get("author")
            ref_year = cit.get("year")

            context = self._get_citation_context(full_text, ref_title, ref_author, ref_year)
            props = {**cit.get("properties", {})}
            if context:
                props["context"] = context
                tasks.append(self._extractor.classify_citation_intent_async(context, ref_title))
                metadata.append((cit["source_id"], cit["target_id"], props))
            else:
                props["intent"] = "BACKGROUND"
                metadata.append((cit["source_id"], cit["target_id"], props))
                tasks.append(asyncio.sleep(0, result="BACKGROUND"))

        intents = await asyncio.gather(*tasks)

        edges = []
        for (src, tgt, props), intent in zip(metadata, intents):
            props["intent"] = intent or "BACKGROUND"
            edges.append((src, tgt, "CITES", props))
        return edges

    async def _create_chunks_async(
        self, paper: Paper, full_text: str, source_path: Optional[str], is_pdf: bool
    ) -> List[Any]:
        if is_pdf:
            pdf_path = paper.file_path if (paper.file_path and os.path.exists(paper.file_path)) else source_path
            chunks = await asyncio.to_thread(split_text_to_chunks, paper.id, pdf_path)
        else:
            chunks = _split_text_to_chunks_raw(paper.id, full_text)

        if chunks and paper.properties.get("source_type") == "video":
            con.dim("Filtering video transcript chunks for database relevance...")
            chunks = [
                chunk for chunk in chunks
                if self._extractor.is_chunk_relevant(chunk.text_content, paper.title or paper.id)
            ]
        return chunks

    async def _create_and_embed_chunks_async(
        self, paper: Paper, full_text: str, source_path: Optional[str], is_pdf: bool
    ) -> List[Any]:
        chunks = await self._create_chunks_async(paper, full_text, source_path, is_pdf)
        if chunks:
            embeddings = await asyncio.to_thread(
                self.emb_engine.get_embeddings, [c.text_content for c in chunks]
            )
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
        return chunks

    def _merge_extracted_authors_and_tags(
        self, paper: Paper, extraction: ExtractionResult
    ) -> None:
        if extraction.authors:
            existing = {a.lower() for a in paper.authors}
            for a in extraction.authors:
                if a.lower() not in existing:
                    paper.authors.append(a)
                    existing.add(a.lower())

        existing_tags = paper.properties.get("tags") or []
        if extraction.tags:
            seen_tags = {t.lower().strip() for t in existing_tags}
            merged_tags = list(existing_tags)
            for t in extraction.tags:
                t_clean = t.strip()
                if t_clean.lower() not in seen_tags:
                    merged_tags.append(t_clean)
                    seen_tags.add(t_clean.lower())
            paper.properties["tags"] = merged_tags

    def _refresh_paper_metadata_sources(self, paper: Paper) -> None:
        is_webpage = (
            paper.properties.get("source_type") == "webpage"
            or (paper.file_path and paper.file_path.startswith("http"))
        )
        url = paper.properties.get("url") or paper.file_path if is_webpage else None
        if not url:
            return

        try:
            parser = ParserFactory.get_parser(url)
            web_paper, _, _ = parser.parse(url)
            if web_paper.authors:
                paper.authors = list({*paper.authors, *web_paper.authors})
            if web_paper.doi:
                paper.doi = web_paper.doi
            if web_paper.properties.get("arxiv_id"):
                paper.properties["arxiv_id"] = web_paper.properties["arxiv_id"]
            if web_paper.title and len(web_paper.title) > len(paper.title or ""):
                paper.title = web_paper.title
            if web_paper.abstract:
                paper.abstract = web_paper.abstract
            if web_paper.year:
                paper.year = web_paper.year
        except Exception as e:
            con.warning(f"Failed to re-parse URL {url}: {e}")

    def _save_refreshed_metadata_to_graph(
        self, paper: Paper, extraction: ExtractionResult, api_references: List[Dict], api_citations: List[Dict]
    ) -> None:
        self.graph_repo.save_paper(paper)

        self.graph_repo.delete_edges_by_target(paper.id, ["AUTHORED"])
        for author_name in paper.authors:
            author_id = slugify(author_name)
            author = Author(id=author_id, name=author_name)
            self.graph_repo.save_author(author)
            self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")

        self.graph_repo.delete_edges_by_source(paper.id, ["MENTIONS_CONCEPT", "HAS_TAG"])

        for item in extraction.concepts:
            c_name = item.get("name", "").strip()
            if not c_name:
                continue
            c_desc = item.get("description", "") or self._extractor.get_concept_description(c_name)
            concept_id = slugify(c_name)
            concept = Concept(id=concept_id, name=c_name, properties={"description": c_desc})
            self.graph_repo.save_concept(concept)
            self.invalidate_concept_cache()
            self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

        for tag in extraction.tags:
            tag_id = slugify(tag)
            tag_desc = self._extractor.get_concept_description(tag)
            tag_node = Concept(id=tag_id, name=tag, properties={"is_tag": True, "description": tag_desc})
            self.graph_repo.save_concept(tag_node)
            self.invalidate_concept_cache()
            self.graph_repo.add_edge(paper.id, tag_id, "HAS_TAG")

        if api_references or api_citations:
            for ref in api_references:
                ref_title = ref.get("title")
                ref_doi = ref.get("doi")
                ref_id = slugify(ref_doi) if ref_doi else (slugify(ref_title[:120]) if ref_title else None)
                if ref_id:
                    if not self.graph_repo.get_paper(ref_id):
                        placeholder = Paper(id=ref_id, title=ref_title, authors=[], year=None, doi=ref_doi)
                        placeholder.properties["is_placeholder"] = True
                        self.graph_repo.save_paper(placeholder)
                    self.graph_repo.add_edge(paper.id, ref_id, "CITES", {"api_sourced": True})

            for cit in api_citations:
                cit_title = cit.get("title")
                cit_doi = cit.get("doi")
                cit_id = slugify(cit_doi) if cit_doi else (slugify(cit_title[:120]) if cit_title else None)
                if cit_id:
                    if not self.graph_repo.get_paper(cit_id):
                        placeholder = Paper(id=cit_id, title=cit_title, authors=[], year=None, doi=cit_doi)
                        placeholder.properties["is_placeholder"] = True
                        self.graph_repo.save_paper(placeholder)
                    self.graph_repo.add_edge(cit_id, paper.id, "CITES", {"api_sourced": True})

    def _find_reindex_candidates(
        self, missing_authors: bool, missing_tags: bool, limit: Optional[int]
    ) -> List[str]:
        non_placeholders = self.graph_repo.get_non_placeholder_paper_ids()
        candidates = []
        for pid in non_placeholders:
            paper = self.graph_repo.get_paper(pid)
            if not paper:
                continue
            if missing_authors and not paper.authors:
                candidates.append(pid)
            elif missing_tags and not paper.properties.get("tags", []):
                candidates.append(pid)
            elif not missing_authors and not missing_tags:
                candidates.append(pid)

        return candidates[:limit] if limit else candidates

    def _configure_chunk_pool(self, chunk_pool_size: Optional[int]) -> None:
        if chunk_pool_size is not None:
            self._extractor._chunk_pool_size = chunk_pool_size
            self._extractor._sem = None

    def _ner_fallback_authors(self, existing_authors: List[str], file_path: str) -> List[str]:
        """Runs NER on the first PDF page to extract author names when few are known."""
        try:
            from src.ner_engine import extract_persons_from_text
            import fitz

            doc = fitz.open(file_path)
            first_page_text = doc[0].get_text() if len(doc) > 0 else ""
            doc.close()
            ner_names = extract_persons_from_text(first_page_text[:2000])
            candidates = [n for n in ner_names if 1 < len(n.split()) <= 5]
        except Exception:
            return existing_authors

        seen = {a.lower().strip() for a in existing_authors}
        merged = list(existing_authors)
        for name in candidates:
            key = name.lower().strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(name)
        return merged

    def _archive_pdf(self, source_path: str, archive_path: Path) -> None:
        """Compresses (or copies) a PDF to the archive directory."""
        from src.parsers.pdf_parser import PDFParser

        if config.pdf_compression_enabled:
            con.dim(f"Compressing and saving PDF to archive: {archive_path} ...")
            try:
                orig_size = os.path.getsize(source_path)
                PDFParser.compress_and_save_pdf(
                    input_path=source_path,
                    output_path=str(archive_path),
                    dpi_threshold=config.pdf_compression_dpi_threshold,
                    dpi_target=config.pdf_compression_dpi_target,
                    quality=config.pdf_compression_quality,
                )
                if archive_path.exists():
                    new_size = archive_path.stat().st_size
                    ratio = (1 - new_size / orig_size) * 100 if orig_size > 0 else 0
                    con.success(
                        f"PDF compressed: {orig_size / 1024 / 1024:.1f} MB → "
                        f"{new_size / 1024 / 1024:.1f} MB ({ratio:.1f}% saved)"
                    )
                if Path(source_path).resolve() != archive_path.resolve():
                    os.remove(source_path)
            except Exception as e:
                con.warning(f"PDF compression failed: {e}. Falling back to standard copy.")
                if Path(source_path).resolve() != archive_path.resolve():
                    shutil.copy2(source_path, archive_path)
                    os.remove(source_path)
        else:
            if Path(source_path).resolve() != archive_path.resolve():
                shutil.copy2(source_path, archive_path)
                os.remove(source_path)
                con.info(f"PDF moved to archive: {archive_path}")

    def _read_local_text(self, paper: Paper) -> str:
        """Reads the full document text from the local file if available."""
        fp = paper.file_path
        if not fp or not os.path.exists(fp) or fp.startswith("http"):
            return ""
        try:
            if fp.lower().endswith(".pdf") or fp.lower().endswith(".epub"):
                parser = ParserFactory.get_parser(fp)
                _, _, text = parser.parse(fp)
                return text
            else:
                return Path(fp).read_text(encoding="utf-8")
        except Exception as e:
            con.warning(f"Could not read local file {fp} for text extraction: {e}")
            return ""

    @contextlib.contextmanager
    def _trace_stage(self, stage_name: str, trace_info: Optional[dict]):
        """Context manager to measure and accumulate time taken by an ingestion stage."""
        if trace_info is None:
            yield
        else:
            t0 = time.perf_counter()
            yield
            dt = time.perf_counter() - t0
            self._record_stage_time(stage_name, dt, trace_info)

    def _record_stage_time(
        self, stage_name: str, dt: float, trace_info: Optional[dict]
    ) -> None:
        if trace_info is not None:
            stages = trace_info.setdefault("stages", {})
            stages[stage_name] = stages.get(stage_name, 0.0) + dt
