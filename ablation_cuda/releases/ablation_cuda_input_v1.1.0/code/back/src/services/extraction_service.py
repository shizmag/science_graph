"""
ExtractionService — Knowledge Extraction from Document Text.

Encapsulates the "try LLM → fallback to regex taxonomy scan" pattern
that was previously duplicated across index_pdf, index_epub, index_url,
and reindex_metadata in Indexer.

Also provides concept description lookup and LLM summary generation.
"""

from __future__ import annotations

import os
import re
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import config
from src.models import Paper, slugify
from src import console as con
from src.services.normalization_pipeline import NormalizationPipeline
from src.prompts import prompts


@dataclass
class ExtractionResult:
    """Structured output from knowledge extraction on a document."""

    authors: List[str] = field(default_factory=list)
    """Author names discovered by LLM (not from parsers or Semantic Scholar)."""

    concepts: List[Dict[str, Any]] = field(default_factory=list)
    """List of {name, description, aliases} dicts for MENTIONS_CONCEPT edges."""

    tags: List[str] = field(default_factory=list)
    """High-level topic/tag names for HAS_TAG edges."""

    via_llm: bool = False
    """True if concepts/tags were extracted by LLM; False if via regex fallback."""

    institutions: List[str] = field(default_factory=list)
    author_institutions: List[Dict[str, str]] = field(default_factory=list)
    sponsored_by: List[str] = field(default_factory=list)
    datasets: List[Dict[str, str]] = field(default_factory=list)
    code_repositories: List[str] = field(default_factory=list)
    journal_or_conference: Optional[str] = None
    citation_intents: List[Dict[str, str]] = field(default_factory=list)
    concept_relations: List[Dict[str, str]] = field(default_factory=list)


class ExtractionService:
    """
    Extracts structured knowledge (authors, concepts, tags) from document text.

    Priority order:
        1. LLM-based extraction (if llm_engine is provided and use_llm=True)
        2. Regex keyword scan against taxonomy.yaml
        3. Empty result (graceful degradation)
    """

    def __init__(self, llm_engine: Any = None, chunk_pool_size: Optional[int] = None) -> None:
        self.llm_engine = llm_engine
        self.normalization_pipeline = NormalizationPipeline()
        self._chunk_pool_size = chunk_pool_size
        self._sem = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._sem is None or getattr(self, "_sem_loop", None) != current_loop:
            is_cloud = False
            if self.llm_engine:
                is_cloud = (
                    getattr(self.llm_engine, "use_cloud", False)
                    or os.environ.get("SCIENCE_GRAPH_USE_CLOUD") == "1"
                    or config.llm_provider in ("openai", "openai-compatible")
                    or self.llm_engine.__class__.__name__ == "OpenAILLMEngine"
                )

            if config.llm_effective_mtp_mode:
                if self._chunk_pool_size is not None and self._chunk_pool_size > 1:
                    con.warning("Speculative decoding (MTP) is active on local server. Overriding chunk pool concurrency to 1 to run sequentially.")
                limit = 1
            elif self._chunk_pool_size is not None:
                limit = self._chunk_pool_size
            else:

                if is_cloud:
                    cfg_val = getattr(config, "llm_chunk_pool_size", 50)
                    limit = cfg_val if cfg_val > 4 else 50
                else:
                    # For local models, default to 1 because Apple Silicon neural engine/GPU contends on concurrent generation.
                    # If user has explicitly customized llm_chunk_pool_size to something other than 4, we respect it.
                    cfg_val = getattr(config, "llm_chunk_pool_size", 4)
                    limit = cfg_val if cfg_val != 4 else 1
            self._sem = asyncio.Semaphore(limit)
            self._sem_loop = current_loop
        return self._sem

    @property
    def _tax(self) -> Dict[str, Any]:
        return config.taxonomy

    async def _call_llm_extract_async(self, text: str, message: Optional[str] = None) -> Optional[dict]:
        if not self.llm_engine:
            return None
        async with self.semaphore:
            if message:
                con.dim(message)
            from unittest.mock import Mock
            func = getattr(self.llm_engine, "extract_concepts_and_metadata_async", None)
            sync_func = getattr(self.llm_engine, "extract_concepts_and_metadata", None)
            is_sync_mocked = isinstance(sync_func, Mock)
            is_async_mocked = isinstance(func, Mock) and not hasattr(func, "assert_awaited")
            
            if (is_sync_mocked and is_async_mocked) or (sync_func and not func):
                return await asyncio.to_thread(sync_func, text)
                
            if func:
                res = func(text)
                if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                    return await res
                return res
            return None

    async def _call_llm_generate_async(self, prompt: str, task: str = None, message: Optional[str] = None, model: Optional[str] = None) -> str:
        if not self.llm_engine:
            return ""
        async with self.semaphore:
            if message:
                con.dim(message)
            from unittest.mock import Mock
            func = getattr(self.llm_engine, "generate_response_async", None)
            sync_func = getattr(self.llm_engine, "generate_response", None)
            is_sync_mocked = isinstance(sync_func, Mock)
            is_async_mocked = isinstance(func, Mock) and not hasattr(func, "assert_awaited")
            
            kwargs = {}
            if task:
                kwargs["task"] = task
            if model:
                kwargs["model"] = model
                
            if (is_sync_mocked and is_async_mocked) or (sync_func and not func):
                return await asyncio.to_thread(sync_func, prompt, **kwargs)
                
            if func:
                res = func(prompt, **kwargs)
                if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                    return await res
                return res
            return ""

    def extract_from_text_file(
        self,
        content: str,
        filename_stem: str,
        use_llm: bool = True,
    ) -> ExtractionResult:
        """
        Parses raw text content (extracting title, abstract, and full text)
        and extracts authors, concepts, and tags.
        """
        first_line = content.split('\n')[0].strip() if content else ""
        if first_line.startswith("# "):
            title = first_line.lstrip("# ").strip()
            full_text = content[len(first_line):].strip()
        else:
            title = filename_stem
            full_text = content

        paragraphs = [p.strip() for p in re.split(r'\n\n+', full_text) if p.strip()]
        abstract = paragraphs[0][:800] if paragraphs else ""

        return self.extract(title, abstract, full_text, use_llm=use_llm)

    def extract(
        self,
        title: str,
        abstract: str,
        full_text: str,
        use_llm: bool = True,
        trace_info: Optional[dict] = None,
    ) -> ExtractionResult:
        """
        Extracts authors, concepts, and tags from the provided text.

        Args:
            title:     Document title.
            abstract:  Document abstract (may be empty).
            full_text: Full document body text (used for keyword scan).
            use_llm:   Whether to attempt LLM-based extraction first.
            trace_info: Optional dictionary to collect timing and token metrics.

        Returns:
            ExtractionResult with authors, concepts, and tags.
        """
        llm_result = None
        regex_result = self._extract_via_regex(title, abstract, full_text)

        if use_llm and self.llm_engine:
            try:
                llm_result = self._extract_via_llm(title, abstract, full_text, trace_info=trace_info)
                if llm_result is not None:
                    # Merge regex concepts/tags into LLM result
                    # De-duplicate concepts by name (case-insensitive)
                    seen_concepts = {c["name"].lower().strip() for c in llm_result.concepts}
                    for c in regex_result.concepts:
                        name_key = c["name"].lower().strip()
                        if name_key not in seen_concepts:
                            llm_result.concepts.append(c)
                            seen_concepts.add(name_key)

                    # De-duplicate tags (case-insensitive)
                    seen_tags = {t.lower().strip() for t in llm_result.tags}
                    for t in regex_result.tags:
                        tag_key = t.lower().strip()
                        if tag_key not in seen_tags:
                            llm_result.tags.append(t)
                    return self._normalize_extraction_result(llm_result)
            except Exception as e:
                con.warning(f"LLM extraction/normalization failed, falling back to regex: {e}")

        return self._normalize_extraction_result(regex_result)

    async def extract_async(
        self,
        title: str,
        abstract: str,
        full_text: str,
        use_llm: bool = True,
        trace_info: Optional[dict] = None,
    ) -> ExtractionResult:
        """
        Asynchronously extracts authors, concepts, and tags from the provided text.
        """
        # If extract is mocked or overridden, run it inside a thread to respect mocks
        is_mocked = False
        try:
            from unittest.mock import Mock
            if isinstance(self.extract, Mock):
                is_mocked = True
        except ImportError:
            pass

        if is_mocked or type(self).extract != ExtractionService.extract:
            import asyncio
            return await asyncio.to_thread(self.extract, title, abstract, full_text, use_llm, trace_info)

        llm_result = None
        regex_result = self._extract_via_regex(title, abstract, full_text)

        if use_llm and self.llm_engine:
            try:
                llm_result = await self._extract_via_llm_async(title, abstract, full_text, trace_info=trace_info)
                if llm_result is not None:
                    # Merge regex concepts/tags into LLM result
                    # De-duplicate concepts by name (case-insensitive)
                    seen_concepts = {c["name"].lower().strip() for c in llm_result.concepts}
                    for c in regex_result.concepts:
                        name_key = c["name"].lower().strip()
                        if name_key not in seen_concepts:
                            llm_result.concepts.append(c)
                            seen_concepts.add(name_key)

                    # De-duplicate tags (case-insensitive)
                    seen_tags = {t.lower().strip() for t in llm_result.tags}
                    for t in regex_result.tags:
                        tag_key = t.lower().strip()
                        if tag_key not in seen_tags:
                            llm_result.tags.append(t)
                            seen_tags.add(tag_key)

                    return self._normalize_extraction_result(llm_result)
            except Exception as e:
                con.warning(f"Async LLM extraction/normalization failed, falling back to regex: {e}")

        return self._normalize_extraction_result(regex_result)

    def get_concept_description(self, name: str, trace_info: Optional[dict] = None) -> str:
        """
        Returns a one-sentence description for the given concept name.

        Priority:
            1. Taxonomy descriptions dict (case-insensitive match)
            2. LLM-generated description (if llm_engine available)
            3. Generic fallback string
        """
        descriptions: Dict[str, str] = self._tax.get("descriptions", {})
        for k, v in descriptions.items():
            if k.lower() == name.lower():
                return v

        if self.llm_engine:
            try:
                prompt = prompts.get_prompt("extraction", "concept_description", name=name)
                if trace_info is not None:
                    tokens_dict = trace_info.setdefault("tokens", {})
                    tokens_dict["Concept description LLM calls"] = tokens_dict.get("Concept description LLM calls", 0) + self.llm_engine.count_tokens(prompt)
                desc = self.llm_engine.generate_response(prompt, task="extraction").strip()
                desc = re.sub(r'^["\']|["\']$', "", desc).strip()
                if desc:
                    return desc
            except Exception:
                pass

        return f"A key concept representing '{name}' within the AI/ML literature."

    async def get_concept_description_async(self, name: str, trace_info: Optional[dict] = None) -> str:
        """
        Returns a one-sentence description for the given concept name asynchronously.
        """
        descriptions: Dict[str, str] = self._tax.get("descriptions", {})
        for k, v in descriptions.items():
            if k.lower() == name.lower():
                return v

        if self.llm_engine:
            try:
                prompt = prompts.get_prompt("extraction", "concept_description", name=name)
                if trace_info is not None:
                    tokens_dict = trace_info.setdefault("tokens", {})
                    tokens_dict["Concept description LLM calls"] = tokens_dict.get("Concept description LLM calls", 0) + self.llm_engine.count_tokens(prompt)
                desc = await self._call_llm_generate_async(prompt, task="extraction")
                desc = desc.strip()
                desc = re.sub(r'^["\']|["\']$', "", desc).strip()
                if desc:
                    return desc
            except Exception:
                pass

        return f"A key concept representing '{name}' within the AI/ML literature."

    def is_chunk_relevant(self, chunk_text: str, doc_title: str) -> bool:
        """
        Verifies if a specific transcript chunk is relevant/actual to the database.
        Checks for advertising, sponsor integrations, self-promotion, or generic intro/outro chat.
        """
        # 1. Rule-based fast check
        # Common sponsor names and self-promotion phrases (English and Russian)
        patterns = [
            r"\b(sponsor|squarespace|nordvpn|patreon|surfshark|brilliant\.org|expressvpn|audible|skillshare)\b",
            r"\b(raid shadow legends|athletic greens|hellofresh|ridge wallet|wix\.com|grammarly|honey browser)\b",
            r"\b(subscribe|bell icon|my channel|like and share|giveaway|promo code|discount link)\b",
            r"\b(click the link|use my code|curiositystream|nebula)\b",
            r"\b(спонсор|подпишитесь|поставьте лайк|жмите колокольчик|промокод|ссылка в описании|скидка)\b",
            r"\b(реклама|интеграция|наш партнер|подписка)\b"
        ]
        
        chunk_lower = chunk_text.lower()
        matched_rule = False
        for pattern in patterns:
            if re.search(pattern, chunk_lower):
                matched_rule = True
                break
        
        # If no keywords matched, we assume it's relevant (minimizing LLM overhead)
        if not matched_rule:
            return True
            
        # 2. If it matched a rule, let's ask the LLM to verify (if available) to prevent false positives
        if not self.llm_engine:
            # Fallback when LLM is unavailable: filter out chunks matching rule
            con.warning(f"Chunk matches sponsor/promo rule but LLM is not available for double-check. Skipping chunk: '{chunk_text[:60]}...'")
            return False
            
        try:
            from src.llm_schemas import LLMVerificationResponse
            from src.llm_engine import StructuredOutput
            
            prompt = prompts.get_prompt("evaluation", "is_chunk_relevant", doc_title=doc_title, chunk_text=chunk_text)
            structured = StructuredOutput(LLMVerificationResponse)
            validated = structured.generate(self.llm_engine, prompt)
            relevant = validated.relevant
            reason = validated.reason
            if not relevant:
                con.warning(f"Skipping chunk (verified irrelevant by LLM: {reason}): '{chunk_text[:60]}...'")
                return False
        except Exception as e:
            # In case LLM fails, act defensively: accept the chunk to avoid losing data
            con.warning(f"Error during LLM verification of chunk: {e}. Accepting chunk defensively.")
            return True
            
        return True

    def generate_summary(
        self,
        paper: Paper,
        full_text: str,
        graph_repo: Any = None,
        trace_info: Optional[dict] = None,
        temp: Optional[float] = None,
    ) -> Optional[str]:
        """
        Generates an LLM summary for a paper and optionally persists it.

        Args:
            paper:      The Paper object whose summary is to be generated.
            full_text:  Full document text (used as context, first 4000 chars).
            graph_repo: If provided, saves the updated paper after generating summary.
            trace_info: Optional dictionary to collect timing and token metrics.
            temp:       Optional temperature parameter for generation.

        Returns:
            The generated summary string, or None if LLM is unavailable or fails.
        """
        if not self.llm_engine:
            return None

        # Check if this is a video
        source_type = paper.properties.get("source_type", "paper")
        if source_type == "video":
            con.dim(f"Generating structured summary for video [bold]{paper.title[:60]}[/bold] via LLM …")
            try:
                from src.llm_schemas import LLMVideoSummaryResponse
                
                sample_text = full_text[:6000] if full_text else ""
                prompt = prompts.get_prompt("synthesis", "video_summary", title=(paper.title or paper.id), sample_text=sample_text)
                if trace_info is not None:
                    tokens_dict = trace_info.setdefault("tokens", {})
                    tokens_dict["Summary Generation"] = tokens_dict.get("Summary Generation", 0) + self.llm_engine.count_tokens(prompt)
                
                from src.llm_engine import StructuredOutput
                structured = StructuredOutput(LLMVideoSummaryResponse)
                validated = structured.generate(self.llm_engine, prompt, temp=(temp if temp is not None else 0.0))
                
                overview = validated.overview
                themes = validated.themes
                outline = validated.outline
                
                paper.properties["video_overview"] = overview
                paper.properties["video_themes"] = themes
                paper.properties["video_outline"] = outline
                
                # Also compile a fallback markdown summary in case a generic UI requests it
                markdown_summary = f"## 🎥 Video Overview\n\n{overview}\n\n"
                if themes:
                    markdown_summary += "## 🧠 Key Themes\n\n"
                    for theme in themes:
                        markdown_summary += f"- {theme}\n"
                    markdown_summary += "\n"
                if outline:
                    markdown_summary += "## 📝 Detailed Summary\n\n"
                    for outline_item in outline:
                        markdown_summary += f"- {outline_item}\n"
                
                paper.properties["summary"] = markdown_summary
                
                if graph_repo is not None:
                    graph_repo.save_paper(paper)
                con.success(f"Structured video summary generated for {(paper.title or paper.id)[:50]}")
                return markdown_summary
            except Exception as e:
                con.warning(f"Failed to generate structured video summary: {e}. Falling back to standard summary.")

        con.dim(f"Generating summary for [bold]{paper.title[:60]}[/bold] via LLM …")
        try:
            sample_text = full_text[:4000] if full_text else ""
            prompt = prompts.get_prompt("synthesis", "paper_summary", title=(paper.title or paper.id), abstract=(paper.abstract or ""), sample_text=sample_text)
            if trace_info is not None:
                tokens_dict = trace_info.setdefault("tokens", {})
                tokens_dict["Summary Generation"] = tokens_dict.get("Summary Generation", 0) + self.llm_engine.count_tokens(prompt)
            summary = self.llm_engine.generate_response(prompt, task="synthesis", temp=temp)
            if summary:
                paper.properties["summary"] = summary
                if graph_repo is not None:
                    graph_repo.save_paper(paper)
                con.success(f"Summary generated for {(paper.title or paper.id)[:50]}")
                return summary
        except Exception as e:
            con.warning(f"Failed to generate summary: {e}")

        return None

    async def generate_summary_async(
        self,
        paper: Paper,
        full_text: str,
        graph_repo: Any = None,
        trace_info: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Generates an LLM summary for a paper asynchronously.
        """
        if not self.llm_engine:
            return None

        # Check if this is a video
        source_type = paper.properties.get("source_type", "paper")
        if source_type == "video":
            try:
                from src.llm_schemas import LLMVideoSummaryResponse
                import json
                
                sample_text = full_text[:6000] if full_text else ""
                prompt = prompts.get_prompt("synthesis", "video_summary", title=(paper.title or paper.id), sample_text=sample_text)
                if trace_info is not None:
                    tokens_dict = trace_info.setdefault("tokens", {})
                    tokens_dict["Summary Generation"] = tokens_dict.get("Summary Generation", 0) + self.llm_engine.count_tokens(prompt)
                
                async with self.semaphore:
                    con.dim(f"Generating structured summary for video [bold]{paper.title[:60]}[/bold] via LLM …")
                    summary_raw = await self.llm_engine.generate_and_validate_json_async(
                        prompt=prompt,
                        schema_class=LLMVideoSummaryResponse,
                    )
                summary_json = json.loads(summary_raw)
                
                overview = summary_json.get("overview", "")
                themes = summary_json.get("themes", [])
                outline = summary_json.get("outline", [])
                
                paper.properties["video_overview"] = overview
                paper.properties["video_themes"] = themes
                paper.properties["video_outline"] = outline
                
                # Also compile a fallback markdown summary in case a generic UI requests it
                markdown_summary = f"## 🎥 Video Overview\n\n{overview}\n\n"
                if themes:
                    markdown_summary += "## 🧠 Key Themes\n\n"
                    for theme in themes:
                        markdown_summary += f"- {theme}\n"
                    markdown_summary += "\n"
                if outline:
                    markdown_summary += "## 📝 Detailed Summary\n\n"
                    for outline_item in outline:
                        markdown_summary += f"- {outline_item}\n"
                
                paper.properties["summary"] = markdown_summary
                
                if graph_repo is not None:
                    await asyncio.to_thread(graph_repo.save_paper, paper)
                con.success(f"Structured video summary generated for {(paper.title or paper.id)[:50]}")
                return markdown_summary
            except Exception as e:
                con.warning(f"Failed to generate structured video summary: {e}. Falling back to standard summary.")

        try:
            sample_text = full_text[:4000] if full_text else ""
            prompt = prompts.get_prompt("synthesis", "paper_summary", title=(paper.title or paper.id), abstract=(paper.abstract or ""), sample_text=sample_text)
            if trace_info is not None:
                tokens_dict = trace_info.setdefault("tokens", {})
                tokens_dict["Summary Generation"] = tokens_dict.get("Summary Generation", 0) + self.llm_engine.count_tokens(prompt)
            msg = f"Generating summary for [bold]{paper.title[:60]}[/bold] via LLM …"
            summary = await self._call_llm_generate_async(prompt, task="synthesis", message=msg)
            if summary:
                paper.properties["summary"] = summary
                if graph_repo is not None:
                    await asyncio.to_thread(graph_repo.save_paper, paper)
                con.success(f"Summary generated for {(paper.title or paper.id)[:50]}")
                return summary
        except Exception as e:
            con.warning(f"Failed to generate summary: {e}")

        return None

    def split_text_semantically(
        self, text: str, max_chunk_tokens: int, overlap_tokens: int
    ) -> List[str]:
        """Splits text into paragraph-aware chunks with overlap constraint."""
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        def _tokens(t: str) -> int:
            if self.llm_engine:
                return self.llm_engine.count_tokens(t)
            return len(t) // 4

        chunks = []
        current_chunk_paragraphs = []
        current_tokens = 0

        for paragraph in paragraphs:
            p_tokens = _tokens(paragraph)
            
            if p_tokens > max_chunk_tokens:
                if current_chunk_paragraphs:
                    chunks.append("\n\n".join(current_chunk_paragraphs))
                    current_chunk_paragraphs = []
                    current_tokens = 0
                
                sub_paragraphs = [sp.strip() for sp in paragraph.split("\n") if sp.strip()]
                for sp in sub_paragraphs:
                    sp_tokens = _tokens(sp)
                    if current_tokens + sp_tokens > max_chunk_tokens:
                        if current_chunk_paragraphs:
                            chunks.append("\n\n".join(current_chunk_paragraphs))
                            
                            overlap_paragraphs = []
                            overlap_tokens_count = 0
                            for op in reversed(current_chunk_paragraphs):
                                op_tokens = _tokens(op)
                                if overlap_tokens_count + op_tokens <= overlap_tokens:
                                    overlap_paragraphs.insert(0, op)
                                    overlap_tokens_count += op_tokens
                                else:
                                    break
                            current_chunk_paragraphs = overlap_paragraphs
                            current_tokens = overlap_tokens_count
                            
                        if sp_tokens > max_chunk_tokens:
                            chunks.append(sp)
                            current_chunk_paragraphs = []
                            current_tokens = 0
                            continue
                            
                    current_chunk_paragraphs.append(sp)
                    current_tokens += sp_tokens
                continue

            if current_tokens + p_tokens > max_chunk_tokens:
                if current_chunk_paragraphs:
                    chunks.append("\n\n".join(current_chunk_paragraphs))
                    
                    overlap_paragraphs = []
                    overlap_tokens_count = 0
                    for op in reversed(current_chunk_paragraphs):
                        op_tokens = _tokens(op)
                        if overlap_tokens_count + op_tokens <= overlap_tokens:
                            overlap_paragraphs.insert(0, op)
                            overlap_tokens_count += op_tokens
                        else:
                            break
                    current_chunk_paragraphs = overlap_paragraphs
                    current_tokens = overlap_tokens_count

            current_chunk_paragraphs.append(paragraph)
            current_tokens += p_tokens

        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))

        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_via_llm(
        self, title: str, abstract: str, full_text: str, trace_info: Optional[dict] = None
    ) -> Optional[ExtractionResult]:
        """Attempts LLM-based extraction. Uses Map-Reduce if text is too long."""
        if not self.llm_engine:
            return None

        con.debug(f"EXTRACTION_LLM_CALL pid={os.getpid()} service_id={id(self)} llm_engine_id={id(self.llm_engine)} llm_engine_class={self.llm_engine.__class__.__name__} llm_model_path={getattr(self.llm_engine, 'model_path', None)} doc_id={title[:50] if title else 'unknown'} stage=extraction")


        total_text = f"{title}\n\n{abstract}\n\n{full_text}"
        limit = config.llm_extraction_input_limit
        threshold = int(0.85 * limit)
        
        def _get_tokens(text: str) -> int:
            cnt = self.llm_engine.count_tokens(text)
            if not isinstance(cnt, (int, float)):
                return len(text) // 4
            return int(cnt)
        
        total_tokens = 0
        if _get_tokens(total_text) > threshold:
            con.info("Input text exceeds 85% of context window limit. Using Map-Reduce extraction.")
            
            prefix = f"{title}\n\n{abstract}\n\n"
            prefix_tokens = _get_tokens(prefix)
            
            chunk_budget = threshold - prefix_tokens
            max_chunk_tokens = max(chunk_budget, 1000)
            overlap_tokens = int(max_chunk_tokens * 0.15)
            
            chunks = self.split_text_semantically(full_text, max_chunk_tokens, overlap_tokens)
            con.info(f"Split document into {len(chunks)} semantic chunks for map phase.")
            
            all_authors = []
            all_concepts = {}
            all_tags = []
            
            for idx, chunk in enumerate(chunks):
                con.dim(f"Extracting concepts from chunk {idx + 1}/{len(chunks)}...")
                chunk_text = f"{prefix}{chunk}"
                
                # Accumulate tokens for this chunk's LLM call
                total_tokens += _get_tokens(chunk_text)
                
                llm_data = self.llm_engine.extract_concepts_and_metadata(chunk_text)
                if not llm_data:
                    continue

                con.debug(f"LLM_RAW_RESPONSE doc_id={title[:50] if title else 'unknown'} raw_type={type(llm_data).__name__} raw_preview={repr(llm_data)[:1000]}")

                
                all_authors.extend(llm_data.get("authors", []))
                
                for item in llm_data.get("concepts", []):
                    c_name = item.get("name", "").strip()
                    if not c_name:
                        continue
                    c_desc = item.get("description", "").strip()
                    slug = slugify(c_name)
                    if slug:
                        if slug in all_concepts:
                            if len(c_desc) > len(all_concepts[slug]["description"]):
                                all_concepts[slug] = {"name": c_name, "description": c_desc}
                        else:
                            all_concepts[slug] = {"name": c_name, "description": c_desc}
                            
                all_tags.extend(llm_data.get("tags", []))
                
            concepts = list(all_concepts.values())
            
            if trace_info is not None:
                tokens_dict = trace_info.setdefault("tokens", {})
                tokens_dict["Concept & Tag Extraction"] = tokens_dict.get("Concept & Tag Extraction", 0) + total_tokens
            
            return ExtractionResult(
                authors=list(set(all_authors)),
                concepts=concepts,
                tags=list(set(all_tags)),
                via_llm=True
            )
            
        else:
            try:
                # Accumulate tokens for the single call
                total_tokens = _get_tokens(total_text)
                
                llm_data = self.llm_engine.extract_concepts_and_metadata(total_text)
                if not llm_data:
                    return None

                con.debug(f"LLM_RAW_RESPONSE doc_id={title[:50] if title else 'unknown'} raw_type={type(llm_data).__name__} raw_preview={repr(llm_data)[:1000]}")


                raw_concepts = llm_data.get("concepts", [])
                concepts = []
                for item in raw_concepts:
                    c_name = item.get("name", "").strip()
                    if not c_name:
                        continue
                    c_desc = (
                        item.get("description", "").strip()
                        or self.get_concept_description(c_name, trace_info=trace_info)
                    )
                    concepts.append({"name": c_name, "description": c_desc})

                if trace_info is not None:
                    tokens_dict = trace_info.setdefault("tokens", {})
                    tokens_dict["Concept & Tag Extraction"] = tokens_dict.get("Concept & Tag Extraction", 0) + total_tokens

                return ExtractionResult(
                    authors=llm_data.get("authors", []),
                    concepts=concepts,
                    tags=llm_data.get("tags", []),
                    via_llm=True,
                )
            except Exception as e:
                con.warning(f"LLM extraction failed, falling back to regex: {e}")
                return None

    async def _extract_via_llm_async(
        self, title: str, abstract: str, full_text: str, trace_info: Optional[dict] = None
    ) -> Optional[ExtractionResult]:
        """Attempts LLM-based extraction asynchronously. Uses Map-Reduce if text is too long."""
        if not self.llm_engine:
            return None

        total_text = f"{title}\n\n{abstract}\n\n{full_text}"
        limit = config.llm_extraction_input_limit
        threshold = int(0.85 * limit)
        
        def _get_tokens(text: str) -> int:
            cnt = self.llm_engine.count_tokens(text)
            if not isinstance(cnt, (int, float)):
                return len(text) // 4
            return int(cnt)
        
        total_tokens = 0
        if _get_tokens(total_text) > threshold:
            con.info("Input text exceeds 85% of context window limit. Using Map-Reduce extraction.")
            
            prefix = f"{title}\n\n{abstract}\n\n"
            prefix_tokens = _get_tokens(prefix)
            
            chunk_budget = threshold - prefix_tokens
            max_chunk_tokens = max(chunk_budget, 1000)
            overlap_tokens = int(max_chunk_tokens * 0.15)
            
            chunks = self.split_text_semantically(full_text, max_chunk_tokens, overlap_tokens)
            con.info(f"Split document into {len(chunks)} semantic chunks for map phase.")
            
            all_authors = []
            all_concepts = {}
            all_tags = []
            all_institutions = []
            all_author_insts = []
            all_sponsored = []
            all_datasets = []
            all_code_repos = []
            all_journals_confs = []
            all_citations = []
            all_concept_rels = []
            
            async def _extract_chunk(idx, chunk):
                chunk_text = f"{prefix}{chunk}"
                tokens_count = _get_tokens(chunk_text)
                msg = f"Extracting concepts from chunk {idx + 1}/{len(chunks)}..."
                llm_data = await self._call_llm_extract_async(chunk_text, message=msg)
                
                con.debug(f"LLM_RAW_RESPONSE doc_id={title[:50] if title else 'unknown'} raw_type={type(llm_data).__name__} raw_preview={repr(llm_data)[:1000]}")

                
                return llm_data, tokens_count
                
            tasks = [
                asyncio.create_task(_extract_chunk(idx, chunk))
                for idx, chunk in enumerate(chunks)
            ]
            results = await asyncio.gather(*tasks)
            
            for llm_data, tokens_count in results:
                total_tokens += tokens_count
                if not llm_data:
                    continue
                
                all_authors.extend(llm_data.get("authors", []))
                
                for item in llm_data.get("concepts", []):
                    c_name = item.get("name", "").strip()
                    if not c_name:
                        continue
                    c_desc = item.get("description", "").strip()
                    c_aliases = item.get("aliases") or []
                    slug = slugify(c_name)
                    if slug:
                        if slug in all_concepts:
                            if len(c_desc) > len(all_concepts[slug]["description"]):
                                all_concepts[slug] = {"name": c_name, "description": c_desc, "aliases": c_aliases}
                            all_concepts[slug]["aliases"] = list(set(all_concepts[slug]["aliases"] + c_aliases))
                        else:
                            all_concepts[slug] = {"name": c_name, "description": c_desc, "aliases": c_aliases}
                            
                all_tags.extend(llm_data.get("tags", []))
                all_institutions.extend(llm_data.get("institutions", []))
                all_author_insts.extend(llm_data.get("author_institutions", []))
                all_sponsored.extend(llm_data.get("sponsored_by", []))
                all_datasets.extend(llm_data.get("datasets", []))
                all_code_repos.extend(llm_data.get("code_repositories", []))
                jc = llm_data.get("journal_or_conference")
                if jc:
                    all_journals_confs.append(jc)
                all_citations.extend(llm_data.get("citation_intents", []))
                all_concept_rels.extend(llm_data.get("concept_relations", []))
                
            concepts = list(all_concepts.values())
            journal_or_conf = None
            if all_journals_confs:
                journal_or_conf = max(set(all_journals_confs), key=all_journals_confs.count)
            
            if trace_info is not None:
                tokens_dict = trace_info.setdefault("tokens", {})
                tokens_dict["Concept & Tag Extraction"] = tokens_dict.get("Concept & Tag Extraction", 0) + total_tokens
            
            return ExtractionResult(
                authors=list(set(all_authors)),
                concepts=concepts,
                tags=list(set(all_tags)),
                via_llm=True,
                institutions=list(set(all_institutions)),
                author_institutions=all_author_insts,
                sponsored_by=list(set(all_sponsored)),
                datasets=all_datasets,
                code_repositories=list(set(all_code_repos)),
                journal_or_conference=journal_or_conf,
                citation_intents=all_citations,
                concept_relations=all_concept_rels
            )
            
        else:
            try:
                # Accumulate tokens for the single call
                total_tokens = _get_tokens(total_text)
                
                llm_data = await self._call_llm_extract_async(total_text)
                if not llm_data:
                    return None

                con.debug(f"LLM_RAW_RESPONSE doc_id={title[:50] if title else 'unknown'} raw_type={type(llm_data).__name__} raw_preview={repr(llm_data)[:1000]}")


                raw_concepts = llm_data.get("concepts", [])
                concepts = []
                for item in raw_concepts:
                    c_name = item.get("name", "").strip()
                    if not c_name:
                        continue
                    c_desc = (
                        item.get("description", "").strip()
                        or await self.get_concept_description_async(c_name, trace_info=trace_info)
                    )
                    concepts.append({"name": c_name, "description": c_desc, "aliases": item.get("aliases") or []})

                if trace_info is not None:
                    tokens_dict = trace_info.setdefault("tokens", {})
                    tokens_dict["Concept & Tag Extraction"] = tokens_dict.get("Concept & Tag Extraction", 0) + total_tokens

                return ExtractionResult(
                    authors=llm_data.get("authors", []),
                    concepts=concepts,
                    tags=llm_data.get("tags", []),
                    via_llm=True,
                    institutions=llm_data.get("institutions", []),
                    author_institutions=llm_data.get("author_institutions", []),
                    sponsored_by=llm_data.get("sponsored_by", []),
                    datasets=llm_data.get("datasets", []),
                    code_repositories=llm_data.get("code_repositories", []),
                    journal_or_conference=llm_data.get("journal_or_conference"),
                    citation_intents=llm_data.get("citation_intents", []),
                    concept_relations=llm_data.get("concept_relations", [])
                )
            except Exception as e:
                con.warning(f"Async LLM extraction failed, falling back to regex: {e}")
                return None

    def _extract_via_regex(
        self, title: str, abstract: str, full_text: str
    ) -> ExtractionResult:
        """Regex keyword scan against the taxonomy."""
        text_to_scan = f"{title} {abstract} {full_text[:10000]}".lower()

        concepts: List[Dict[str, str]] = []
        seen_concept_names: set = set()
        for keyword, concept_name in self._tax.get("concepts", {}).items():
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_to_scan):
                if concept_name not in seen_concept_names:
                    seen_concept_names.add(concept_name)
                    c_desc = self.get_concept_description(concept_name)
                    concepts.append({"name": concept_name, "description": c_desc})

        tags: List[str] = []
        for keyword, tag_name in self._tax.get("topics", {}).items():
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_to_scan):
                if tag_name not in tags:
                    tags.append(tag_name)

        return ExtractionResult(concepts=concepts, tags=tags, via_llm=False)

    def _normalize_extraction_result(self, result: ExtractionResult) -> ExtractionResult:
        from src.llm_schemas import LLMExtractionResponse, LLMConcept, LLMCitationIntent, LLMConceptRelation, LLMDataset
        
        concepts = [
            LLMConcept(
                name=c["name"],
                description=c.get("description", ""),
                aliases=c.get("aliases") or []
            )
            for c in result.concepts
        ]
        
        datasets = [
            LLMDataset(name=d["name"], relation=d.get("relation", "USED_DATASET"))
            for d in result.datasets
        ]
        
        citations = [
            LLMCitationIntent(target_title=ci["target_title"], intent=ci.get("intent", "BACKGROUND"))
            for ci in result.citation_intents
        ]
        
        concept_rels = [
            LLMConceptRelation(source=cr["source"], target=cr["target"], relation_type=cr["relation_type"])
            for cr in result.concept_relations
        ]
        
        response_model = LLMExtractionResponse(
            authors=result.authors,
            concepts=concepts,
            tags=result.tags,
            institutions=result.institutions,
            author_institutions=result.author_institutions,
            sponsored_by=result.sponsored_by,
            datasets=datasets,
            code_repositories=result.code_repositories,
            journal_or_conference=result.journal_or_conference,
            citation_intents=citations,
            concept_relations=concept_rels
        )
        
        normalized_response = self.normalization_pipeline.normalize_extraction_response(response_model)
        
        normalized_concepts = [
            {"name": c.name, "description": c.description, "aliases": c.aliases}
            for c in normalized_response.concepts
        ]
        
        normalized_datasets = [
            {"name": d.name, "relation": d.relation}
            for d in normalized_response.datasets
        ]
        
        normalized_citations = [
            {"target_title": ci.target_title, "intent": ci.intent}
            for ci in normalized_response.citation_intents
        ]
        
        normalized_concept_rels = [
            {"source": cr.source, "target": cr.target, "relation_type": cr.relation_type}
            for cr in normalized_response.concept_relations
        ]
        
        return ExtractionResult(
            authors=normalized_response.authors,
            concepts=normalized_concepts,
            tags=normalized_response.tags,
            via_llm=result.via_llm,
            institutions=normalized_response.institutions,
            author_institutions=normalized_response.author_institutions,
            sponsored_by=normalized_response.sponsored_by,
            datasets=normalized_datasets,
            code_repositories=normalized_response.code_repositories,
            journal_or_conference=normalized_response.journal_or_conference,
            citation_intents=normalized_citations,
            concept_relations=normalized_concept_rels
        )

    async def classify_citation_intent_async(self, context: str, ref_title: str) -> str:
        if not self.llm_engine:
            return "BACKGROUND"
        try:
            prompt = prompts.get_prompt("extraction", "citation_intent", ref_title=ref_title, context=context)
            cheap_model = getattr(config, "llm_cheap_model_name", "google/gemini-2.5-flash")
            resp = await self._call_llm_generate_async(prompt, task="extraction", message=f"Classifying citation intent for '{ref_title[:40]}'", model=cheap_model)
            resp = resp.strip().upper()
            for code in ["USES_METHOD", "EXTENDS", "COMPARES_WITH", "DISPUTES", "CRITICIZES", "BACKGROUND", "CITES"]:
                if code in resp:
                    if code == "CRITICIZES":
                        return "DISPUTES"
                    if code == "CITES":
                        return "BACKGROUND"
                    return code
        except Exception as e:
            con.warning(f"Failed to classify citation intent: {e}")
        return "BACKGROUND"


