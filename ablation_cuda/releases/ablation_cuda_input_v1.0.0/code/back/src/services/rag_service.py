import json
import asyncio
import hashlib
import os
import re
from typing import List, Tuple, AsyncGenerator, Any, Optional, Dict
from pathlib import Path
try:
    import tiktoken
except ImportError:
    tiktoken = None
from src.models import Chunk
from src.repository.base import GraphRepository, VectorRepository
from src.vector_search import EmbeddingEngine
from src.llm_engine import BaseLLMEngine
from src.config import config
from src import console as con
from src.prompts import prompts

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


def _get_parent_key(chunk: Any) -> Optional[Tuple[str, ...]]:
    """
    Extracts a parent identification key for deduplication.
    Priority:
    1. chunk.paper_id (if present and non-empty string)
    2. chunk.parent_id (if present and non-empty string, normalized to paper base)
    3. chunk.parent_text (if present, non-empty string, and different from chunk.text_content)
    """
    paper_id = getattr(chunk, "paper_id", None)
    if isinstance(paper_id, str) and paper_id.strip():
        return ("paper_id", paper_id.strip())

    parent_id = getattr(chunk, "parent_id", None)
    if isinstance(parent_id, str) and parent_id.strip():
        base_id = parent_id.split("#parent_")[0].strip()
        return ("parent_id", base_id)

    parent_text = getattr(chunk, "parent_text", None)
    text_content = getattr(chunk, "text_content", None)
    if isinstance(parent_text, str) and parent_text.strip():
        if text_content is None or parent_text.strip() != text_content.strip():
            return ("parent_text", parent_text.strip())

    return None


def _deduplicate_parent_chunks(scored_chunks: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    """
    Deduplicates a list of (chunk, score) tuples based on parent chunk keys.
    Preserves the first (highest scoring) occurrence of each parent chunk.
    """
    deduped = []
    seen_parent_keys = set()
    for item in scored_chunks:
        chunk = item[0]
        pkey = _get_parent_key(chunk)
        if pkey is not None:
            if pkey in seen_parent_keys:
                continue
            seen_parent_keys.add(pkey)
        deduped.append(item)
    return deduped


def clean_reasoning_text(text: str) -> str:
    """
    Strips structured thinking/reasoning blocks and extracts the final answer.
    Handles formats like:
      - 1. _analysis..._ or ### 1. _analysis... or 1. _analysis..._analysis:
      - up to the answer section (5. _answer..._ or Final Answer:)
    """
    if not text:
        return text

    # 1. Try to find the last answer marker and take everything after it
    answer_markers = [
        r"(?:###\s*)?Final\s+Answer\s*:?\s*",
        r"(?:###\s*)?5\.\s*_(?:answer|status|reasoning|analysis|source_analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*\s*",
    ]
    combined_pattern = re.compile(
        r"|".join(f"(?:{p})" for p in answer_markers),
        re.IGNORECASE
    )
    
    matches = list(combined_pattern.finditer(text))
    if matches:
        last_match = matches[-1]
        candidate = text[last_match.end():].strip()
        # If there are still answer markers inside candidate, recurse
        if combined_pattern.search(candidate):
            return clean_reasoning_text(candidate)
        text = candidate
    else:
        # 2. Try to strip the entire block from "1. _analysis" to "5. _answer" or "Final Answer"
        text = re.sub(
            r"(?:###\s*)?[1-4]\.\s*_(?:analysis|start|reasoning|status|source_analysis)(?:\.\.\.)?.*?(?=(?:###\s*)?(?:5\.\s*_(?:answer|status|reasoning|analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*|(?:###\s*)?Final\s+Answer\s*:?))",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    # 3. Clean up any remaining section headers/tags
    header_pattern = r"(?:###\s*)?[1-5]\.\s*_(?:analysis|start|reasoning|status|answer|source_analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*"
    text = re.sub(header_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:###\s*)?Final\s+Answer\s*:?", "", text, flags=re.IGNORECASE)
    
    # Mask source ID tags to preserve them during strip_thinking_tokens
    text = re.sub(r"<\|source_id\|>", "__SOURCE_ID_TAG__", text, flags=re.IGNORECASE)
    
    # Clean generic and specific technical tokens
    from src.llm_engine.base import strip_thinking_tokens
    text = strip_thinking_tokens(text)
    
    # Unmask source ID tags
    text = text.replace("__SOURCE_ID_TAG__", "<|source_id|>")
    
    return text.strip()


def parse_reasoning_response(raw_response: str) -> Tuple[str, str]:
    """
    Parses the raw LLM response to extract the status and the final answer.
    
    Returns:
        Tuple[str, str]: (status, answer)
    """
    import logging
    logger = logging.getLogger(__name__)

    if not raw_response or not isinstance(raw_response, str):
        return "UNKNOWN", "Error: Empty or incorrect response from model."

    # Extract status
    status = "UNKNOWN"
    status_match = re.search(r"<\|status_start\|>(.*?)<\|status_end\|>", raw_response, re.DOTALL)
    if status_match:
        status = status_match.group(1).strip()
    else:
        # Fallback if tag is unclosed
        status_unclosed = re.search(r"<\|status_start\|>(.*)", raw_response, re.DOTALL)
        if status_unclosed:
            content = status_unclosed.group(1).split("<|")[0].strip()
            status = content if content else "UNKNOWN"

    # If status is still UNKNOWN, try to extract from 4. _status... section
    if status == "UNKNOWN":
        status_sec_match = re.search(
            r"(?:###\s*)?4\.\s*_(?:status)(?:\.\.\.)?[_a-zA-Z0-9:]*\s*(.*?)(?=(?:###\s*)?(?:5\.\s*_(?:answer)(?:\.\.\.)?[_a-zA-Z0-9:]*|(?:###\s*)?Final\s+Answer\s*:?|$))",
            raw_response,
            re.IGNORECASE | re.DOTALL
        )
        if status_sec_match:
            status_text = status_sec_match.group(1).strip().upper()
            if "UNANSWERABLE" in status_text or "NOT ANSWERABLE" in status_text or "INSUFFICIENT" in status_text or "NOT_ANSWERABLE" in status_text:
                status = "UNANSWERABLE"
            elif "ANSWERABLE" in status_text or "SUFFICIENT" in status_text:
                status = "ANSWERABLE"
            elif "UNKNOWN" in status_text:
                status = "UNKNOWN"

    # Extract answer
    answer_match = re.search(r"<\|answer_start\|>(.*?)<\|answer_end\|>", raw_response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback if tag is unclosed
        answer_unclosed = re.search(r"<\|answer_start\|>(.*)", raw_response, re.DOTALL)
        if answer_unclosed:
            answer = answer_unclosed.group(1).strip()
        else:
            # Fallback to the raw response if no answer tags are present at all.
            # Strip reasoning/status/other tags to isolate text.
            answer = raw_response.strip()
            for tag in ["status", "query_analysis", "source_analysis", "reasoning"]:
                answer = re.sub(rf"<\|{tag}_start\|>.*?<\|{tag}_end\|>", "", answer, flags=re.DOTALL)
                answer = re.sub(rf"<\|{tag}_start\|>.*", "", answer, flags=re.DOTALL)
            
            # Clean reasoning markers and headers from the final answer
            answer = clean_reasoning_text(answer)
            
    logger.info(f"RAG reasoning status: {status}")
    return status, answer


TECHNICAL_TOKEN_RE = re.compile(
    r"(<\|.*?\|>|<<.*?>>|\[/?(?:[A-Z_]{2,}[A-Z0-9_-]*)\]|</?(?:s|pad|unk|turn)>|<\|im_start\|>|<\|im_end\|>|<\|im_sep\|>|<\|start_header_id\|>|<\|end_header_id\|>|<\|eot_id\|>|<\|eom_id\|>|<\|endoftext\|>|<\|assistant\|>|<\|user\|>|<\|system\|>|<\|end\|>|\[INST\]|\[/INST\]|<s>|</s>|<start_of_turn>|<end_of_turn>|<<SYS>>|<</SYS>>|<pad>|<unk>)",
    re.IGNORECASE
)

class StreamTokenCleaner:
    def __init__(self):
        self.in_think = False
        self.buffer = ""

    def process_token(self, token: str) -> str:
        self.buffer += token
        output = ""
        
        while self.buffer:
            if self.in_think:
                # Look for end of think block
                idx = self.buffer.lower().find("</think>")
                if idx != -1:
                    # Found end of think block, discard it and everything before it
                    self.buffer = self.buffer[idx + 8:]
                    self.in_think = False
                else:
                    # End of think block not found yet. Keep only the last 7 characters
                    # to handle cases where "</think>" is split across token boundaries.
                    if len(self.buffer) > 7:
                        self.buffer = self.buffer[-7:]
                    break
            else:
                # Look for start of think block
                idx = self.buffer.lower().find("<think>")
                if idx != -1:
                    # Found start of think block. Emit everything before it.
                    output += self.buffer[:idx]
                    self.buffer = self.buffer[idx + 7:]
                    self.in_think = True
                else:
                    # Start tag not found. Keep at most 6 trailing characters
                    # in case "<think>" is split across token boundaries.
                    check_len = min(len(self.buffer), 6)
                    tail = self.buffer[-check_len:]
                    lt_idx = tail.rfind("<")
                    if lt_idx != -1:
                        split_idx = len(self.buffer) - check_len + lt_idx
                        output += self.buffer[:split_idx]
                        self.buffer = self.buffer[split_idx:]
                        break
                    else:
                        output += self.buffer
                        self.buffer = ""
                        break
        return output


def count_prompt_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


class RAGService:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        llm_engine: BaseLLMEngine,
        expander: Optional[Any] = None,
        warmup: bool = False,
        trace_dir: Optional[Path] = None
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine
        self.expander = expander
        self._reranker = None
        self.trace_dir = trace_dir
        self._last_graph_trace = {}
        if warmup:
            try:
                embedding_engine.get_embedding("warmup query")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Eager embedding engine warmup failed: {e}")
            try:
                self._get_reranker()
                self._reranker.predict([("warmup query", "warmup context")])
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Eager reranker warmup failed: {e}")

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        from sentence_transformers import CrossEncoder
        model_name = config.reranker_model_name
        con.model_msg(f"Loading reranker [bold]{model_name}[/bold] …")
        with con.suppress_stderr(), con.suppress_stdout():
            import torch
            if os.environ.get("ABLA_CUDA_REQUIRE_DEVICE") == "1":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is required for the Cross-Encoder, but is unavailable")
                device = "cuda"
            elif torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            self._reranker = CrossEncoder(model_name, device=device)
            if os.environ.get("ABLA_CUDA_REQUIRE_DEVICE") == "1":
                devices = {str(parameter.device) for parameter in self._reranker.model.parameters()}
                if not devices or not all(item.startswith("cuda") for item in devices):
                    raise RuntimeError(f"Cross-Encoder did not load entirely on CUDA: {sorted(devices)}")
        con.success(f"Reranker ready on {device.upper()}")
        return self._reranker

    def _resolve_node_name(self, node_id: str, label: str) -> str:
        """Resolves node ID to its actual name/title using DB lookup."""
        if label == "Paper":
            paper = self.graph_repo.get_paper(node_id)
            return f"'{paper.title}'" if paper else f"'{node_id}'"
        elif label == "Author":
            author = self.graph_repo.get_author(node_id)
            return author.name if author else node_id
        elif label == "Concept":
            concept = self.graph_repo.get_concept(node_id)
            return concept.name if concept else node_id
        return node_id

    def _get_scored_graph_lines(self, paper_ids: List[str], limit: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Retrieves graph neighbor relations for the given paper IDs, formats them,
        and assigns importance scores.

        Also populates ``self._last_graph_relations`` with structured edge dicts
        for Shannon graph entropy (source/target/type).
        """
        scored_lines: List[Tuple[str, float]] = []
        relations_acc: List[dict] = []
        seen_edges = set()

        for paper_id in paper_ids:
            neighbors = self.graph_repo.get_neighbors(paper_id, max_depth=1)
            for src_id, src_label, edge_type, tgt_id, tgt_label, edge_props in neighbors:
                edge_key = (src_id, tgt_id, edge_type)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)

                    src_name = self._resolve_node_name(src_id, src_label)
                    tgt_name = self._resolve_node_name(tgt_id, tgt_label)

                    try:
                        props = json.loads(edge_props) if edge_props else {}
                    except Exception:
                        props = {}

                    # Try to extract explicit score or weight from properties
                    score_val = props.get("score") or props.get("weight")
                    if score_val is not None:
                        score = float(score_val)
                    else:
                        # Default heuristic scores based on edge type
                        if edge_type == "AUTHORED":
                            score = config.graph_weight_authored
                        elif edge_type == "CITES":
                            score = config.graph_weight_cites
                        elif edge_type == "MENTIONS_CONCEPT":
                            score = config.graph_weight_mentions_concept
                        else:
                            score = config.graph_weight_default

                    # Compact Cypher representation
                    if edge_type == "CITES" and props.get("raw_text"):
                        raw_text = props.get("raw_text")
                        ref_preview = raw_text if len(raw_text) < 100 else raw_text[:100] + "..."
                        ref_preview = ref_preview.replace('\n', ' ').strip()
                        line = f"- ({src_name}:{src_label})-[CITES {{preview: {json.dumps(ref_preview)}}}]->({tgt_name}:{tgt_label})"
                    else:
                        line = f"- ({src_name}:{src_label})-[{edge_type}]->({tgt_name}:{tgt_label})"

                    scored_lines.append((line, score))
                    relations_acc.append({
                        "source": str(src_id),
                        "target": str(tgt_id),
                        "type": str(edge_type),
                        "source_label": str(src_label) if src_label is not None else "",
                        "target_label": str(tgt_label) if tgt_label is not None else "",
                        "source_name": str(src_name) if src_name is not None else "",
                        "target_name": str(tgt_name) if tgt_name is not None else "",
                    })

        if limit is not None:
            # Keep relations aligned with the limited scored lines
            paired = list(zip(scored_lines, relations_acc))
            paired.sort(key=lambda x: x[0][1], reverse=True)
            paired = paired[:limit * 2]
            scored_lines = [p[0] for p in paired]
            relations_acc = [p[1] for p in paired]

        self._last_graph_relations = relations_acc
        return scored_lines

    def build_context(self, similar_chunks: List[tuple[Chunk, float]], limit: Optional[int] = None) -> Tuple[str, str]:
        """
        Builds two context blocks:
        1. Semantic text blocks
        2. Knowledge graph relationships

        Uses get_papers_batch() to avoid N+1 query patterns.
        """
        if config.rag_components.get("deduplicate_parent_chunks", True):
            similar_chunks = _deduplicate_parent_chunks(similar_chunks)

        # ── Batch-fetch all papers in one query ──
        paper_ids = list({chunk.paper_id for chunk, _ in similar_chunks})
        papers_map = self.graph_repo.get_papers_batch(paper_ids)

        # 1. Format text chunks using the pre-fetched map
        text_blocks = []
        for idx, (chunk, score) in enumerate(similar_chunks, start=1):
            paper = papers_map.get(chunk.paper_id)
            title = paper.title if paper else chunk.paper_id
            year_str = f", {paper.year}" if paper and paper.year else ""
            authors_str = f" by {', '.join(paper.authors)}" if paper and paper.authors else ""
            chunk_text = chunk.parent_text if isinstance(chunk.parent_text, str) else chunk.text_content
            doc_text = (
                f"Block {idx} (Score: {score:.3f}) | Paper: {title}{authors_str}{year_str} (Page {chunk.page_number}):\n"
                f"\"\"\"\n{chunk_text.strip()}\n\"\"\""
            )
            text_blocks.append(
                f"<|source_start|><|source_id|>{idx} {doc_text}<|source_end|>"
            )
        context_text = "\n\n".join(text_blocks)

        # 2. Format Graph Subgraph using helper
        if config.rag_components.get("graph_expansion", True):
            scored_lines = self._get_scored_graph_lines(paper_ids, limit=limit)
            graph_lines = [line for line, _ in scored_lines]
            # _last_graph_relations already set by _get_scored_graph_lines
            context_graph = "\n".join(graph_lines) if graph_lines else "No direct graph relations found."
        else:
            self._last_graph_relations = []
            context_graph = "Graph enrichment disabled."
        return context_text, context_graph

    def trim_context(
        self,
        context_text: str,
        context_graph: str,
        final_chunks: list,
        query: str,
        history_str: str,
        system_prompt: str,
        model_max_context: int = 4096,
        reserved_tokens: int = 500
    ) -> tuple[str, str, list]:
        """Возвращает (trimmed_context_text, trimmed_context_graph, trimmed_chunks)."""
        def get_total_tokens(ctx_text: str, ctx_graph: str) -> int:
            combined_text = ctx_text + ctx_graph + system_prompt + query + history_str
            return count_prompt_tokens(combined_text)

        tokens_limit = model_max_context - reserved_tokens
        current_chunks = list(final_chunks)
        current_text = context_text
        current_graph = context_graph

        total_tokens = get_total_tokens(current_text, current_graph)

        if total_tokens <= tokens_limit:
            return current_text, current_graph, current_chunks

        # 1. Prune graph connections first (cut from the tail after sorting descending by score)
        if total_tokens > tokens_limit and config.rag_components.get("graph_expansion", True):
            paper_ids = list({chunk.paper_id for chunk, _ in current_chunks})
            scored_lines = self._get_scored_graph_lines(paper_ids)
            # Sort by score descending (highest score first)
            scored_lines.sort(key=lambda x: x[1], reverse=True)
            
            while len(scored_lines) > 0 and total_tokens > tokens_limit:
                scored_lines.pop()
                current_graph = (
                    "\n".join([line for line, _ in scored_lines])
                    if scored_lines
                    else "No direct graph relations found."
                )
                total_tokens = get_total_tokens(current_text, current_graph)

        if total_tokens <= tokens_limit:
            return current_text, current_graph, current_chunks

        # Group chunks by paper_id to keep document context unified
        paper_chunks = {}
        for chunk, score in final_chunks:
            paper_chunks.setdefault(chunk.paper_id, []).append((chunk, score))

        # Sort chunks within each paper by page_number (restores chronological flow)
        for pid in paper_chunks:
            paper_chunks[pid].sort(key=lambda x: x[0].page_number)

        # Order papers by their highest chunk score
        sorted_papers = sorted(
            paper_chunks.items(),
            key=lambda x: max(score for _, score in x[1]),
            reverse=True
        )

        from copy import copy
        current_papers = []
        for pid, chs in sorted_papers:
            copied_chs = []
            for c, s in chs:
                c_copy = copy(c)
                if isinstance(c_copy.parent_text, str):
                    c_copy.text_content = c_copy.parent_text
                    c_copy.parent_text = None
                copied_chs.append((c_copy, s))
            current_papers.append([pid, copied_chs])

        while total_tokens > tokens_limit:
            # Try to prune from the least relevant paper (last in sorted list)
            pruned = False
            for i in range(len(current_papers) - 1, -1, -1):
                chunks_list = current_papers[i][1]
                if len(chunks_list) > 0:
                    # Prune a sentence from the last chunk.
                    # First and last sentences are preserved; starting with the middle.
                    last_chunk, score = chunks_list[-1]
                    sentences = last_chunk.text_content.strip().split(". ")
                    sentences = [s.strip() for s in sentences if s.strip()]
                    if len(sentences) > 2:
                        # Soft trim: remove the second-to-last sentence (from the middle)
                        sentences.pop(-2)
                        new_text = ". ".join(sentences)
                        if not any(new_text.endswith(p) for p in [".", "?", "!"]):
                            new_text += "."
                        last_chunk.text_content = new_text
                    elif len(sentences) == 2:
                        # Soft trim: remove the last sentence (since only 2 sentences left, no middle exists)
                        sentences.pop()
                        new_text = sentences[0]
                        if not any(new_text.endswith(p) for p in [".", "?", "!"]):
                            new_text += "."
                        last_chunk.text_content = new_text
                    else:
                        # Hard trim: remove the entire chunk if only 1 sentence left
                        chunks_list.pop()
                    pruned = True
                    break

            if not pruned:
                break

            # Rebuild context and recalculate tokens
            flat_active = []
            for _, chs in current_papers:
                flat_active.extend(chs)

            if not flat_active:
                break

            # Rebuild text, and keep graph at its minimum/pruned state
            current_text, _ = self.build_context(flat_active)
            current_graph = "No direct graph relations found." if config.rag_components.get("graph_expansion", True) else "Graph enrichment disabled."
            current_chunks = flat_active
            total_tokens = get_total_tokens(current_text, current_graph)

        return current_text, current_graph, current_chunks

    def _expand_query(self, query: str) -> List[str]:
        if not query or not isinstance(query, str) or not query.strip():
            return [query] if isinstance(query, str) else []

        try:
            max_expanded = _safe_int(config.max_expanded_queries, 3)
        except Exception:
            max_expanded = 3

        if max_expanded <= 1 or len(query) > 200:
            return [query]

        variants = []
        seen = {query.lower()}

        # 1. Short Query Concept/Synonym Lookup from Graph Ontology
        if config.rag_components.get("graph_ontology_lookup", True) and len(query.strip().split()) <= 2:
            try:
                aliases_map = self.graph_repo.get_concept_aliases()
                canonical = aliases_map.get(query.strip().lower())
                if canonical:
                    concept_node = self.graph_repo.get_concept(canonical)
                    if concept_node:
                        props = concept_node.properties if hasattr(concept_node, "properties") else {}
                        aliases = props.get("aliases") or []
                        names = [props.get("name_en"), props.get("name_ru"), props.get("name")]
                        extra_variants = [canonical] + aliases + names
                        for ev in extra_variants:
                            if isinstance(ev, str) and ev.strip():
                                ev_clean = ev.strip()
                                if ev_clean.lower() not in seen:
                                    variants.append(ev_clean)
                                    seen.add(ev_clean.lower())
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Short query concept enrichment failed: {e}")

        # 2. If we still need more variants, call the LLM
        if config.rag_components.get("llm_query_expansion", True) and len(variants) < max_expanded - 1:
            try:
                # Detect language & generate variants on the same language
                is_cyrillic = bool(re.search('[а-яА-ЯёЁ]', query))
                language = "Russian" if is_cyrillic else "English"
                prompt = prompts.get_prompt("rag", "query_expander", query=query, language=language)
                
                from src.llm_schemas import LLMQueryExpansionResponse
                from src.llm_engine import StructuredOutput

                max_tokens = min(_safe_int(config.llm_max_tokens, 1000), 200)
                structured = StructuredOutput(LLMQueryExpansionResponse)
                validated = structured.generate(self.llm_engine, prompt, max_tokens=max_tokens)
                parsed = validated.root

                if isinstance(parsed, list):
                    for v in parsed:
                        if isinstance(v, str):
                            v_clean = v.strip()
                            if v_clean and v_clean.lower() not in seen:
                                variants.append(v_clean)
                                seen.add(v_clean.lower())
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Query expansion failed: {e}")
                
        limit_variants = max_expanded - 1
        return [query] + variants[:limit_variants]

    def _classify_intent_and_extract_filters(self, query: str) -> Tuple[str, Optional[dict]]:
        import re
        import datetime
        
        time_keywords = ["год", "лет", "last", "year", "recent", "новые", "newest", "старые", "oldest"]
        author_keywords = ["автор", "by ", "author", "написал", "wrote"]
        venue_keywords = ["journal", "conference", "журнал", "конференция", "venue"]
        
        query_lower = query.lower()
        has_time = any(w in query_lower for w in time_keywords) or bool(re.search(r'\b\d{4}\b', query))
        has_author = any(w in query_lower for w in author_keywords)
        has_venue = any(w in query_lower for w in venue_keywords)
        
        if not (has_time or has_author or has_venue):
            return query, None
            
        current_year = datetime.datetime.now().year
        
        prompt = (
            f"You are a scientific database query analyzer. The current year is {current_year}.\n"
            "Given a user query, extract metadata filters and return a JSON object with fields:\n"
            "- search_query: clean search query without relative/filter terms\n"
            "- year_start: start year (integer or null)\n"
            "- year_end: end year (integer or null)\n"
            "- author: author name (string or null)\n"
            "- venue: journal/conference name (string or null)\n"
            "Example: 'articles from the last 2 years about convolutional networks'\n"
            f"Output: {{\"search_query\": \"convolutional networks\", \"year_start\": {current_year - 2}, \"year_end\": {current_year}, \"author\": null, \"venue\": null}}\n"
            f"Query: {query}\n"
            "Return ONLY JSON."
        )
        
        try:
            response = self.llm_engine.generate_response(prompt)
            clean_json = self.llm_engine.extract_json(response)
            parsed = json.loads(clean_json)
            
            filters = {}
            for k in ["year_start", "year_end", "author", "venue"]:
                if parsed.get(k) is not None:
                    filters[k] = parsed[k]
                    
            clean_q = parsed.get("search_query")
            if not clean_q or not isinstance(clean_q, str) or not clean_q.strip():
                clean_q = query
            else:
                clean_q = clean_q.strip()
            con.success(f"Extracted filters: {filters} | Clean query: '{clean_q}'")
            return clean_q, filters if filters else None
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Query intent classification failed: {e}")
            return query, None

    def _validate_and_repair_citations(self, response: str, retrieved_chunks: list) -> str:
        """
        Parses LLM response for bracketed numeric citations (e.g., [1], [2])
        and cross-checks them against the available metadata. Removes hallucinated
        citations or maps them to correct document indexes.
        """
        import re
        max_idx = len(retrieved_chunks)
        
        # 1. Parse bracket groups to support multi-citations like [1, 2, 99] or [Block 1; 2]
        bracket_regex = re.compile(r"\[([^\]]+)\]")
        
        def process_bracket(match):
            content = match.group(1)
            # Extract all digits (e.g., "Block 1, 2" -> ["1", "2"])
            nums = re.findall(r"\d+", content)
            valid_nums = []
            for n in nums:
                val = int(n)
                if 1 <= val <= max_idx:
                    valid_nums.append(str(val))
            
            # Reconstruct bracket if there are valid citation numbers
            if valid_nums:
                return f"[{', '.join(valid_nums)}]"
            return ""
            
        repaired = bracket_regex.sub(process_bracket, response)
        
        # 2. Parse standalone "Block X" mentions outside brackets
        block_regex = re.compile(r"\bBlock\s+(\d+)\b", re.IGNORECASE)
        repaired = block_regex.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= max_idx else "", repaired)
        
        # 3. Clean up punctuation spacing and double spaces after deletions
        repaired = re.sub(r"\s+([.,;!?])", r"\1", repaired)  # e.g., "fact ." -> "fact."
        repaired = re.sub(r"\s+", " ", repaired).strip()
        return repaired

    def _deduplicate_candidates(self, chunks: List[Chunk]) -> List[Chunk]:
        import hashlib
        merged_chunks = []
        seen = {} # key -> index in merged_chunks
        
        for chunk in chunks:
            if getattr(chunk, "id", None):
                key = chunk.id
            elif getattr(chunk, "paper_id", None) and hasattr(chunk, "chunk_index"):
                key = (chunk.paper_id, chunk.chunk_index)
            else:
                key = hashlib.md5(chunk.text_content.encode('utf-8')).hexdigest()
                
            if not hasattr(chunk, "retrieval_sources"):
                chunk.retrieval_sources = []
                
            normalized_sources = []
            for s in chunk.retrieval_sources:
                if isinstance(s, dict):
                    normalized_sources.append(s)
                else:
                    normalized_sources.append({"source": str(s)})
                    
            if key in seen:
                existing_chunk = merged_chunks[seen[key]]
                if not hasattr(existing_chunk, "retrieval_sources"):
                    existing_chunk.retrieval_sources = []
                
                # Merge dicts by "source" key
                for s in normalized_sources:
                    if not any(item.get("source") == s.get("source") for item in existing_chunk.retrieval_sources):
                        existing_chunk.retrieval_sources.append(s)
                
                # Merge graph metadata and other properties
                if hasattr(chunk, "graph_metadata"):
                    if not hasattr(existing_chunk, "graph_metadata"):
                        existing_chunk.graph_metadata = chunk.graph_metadata
                for attr in ("candidate_source", "seed_paper_ids", "graph_neighbor_paper_id", "graph_distance", "graph_path_reason", "matched_edge_types"):
                    if hasattr(chunk, attr) and not hasattr(existing_chunk, attr):
                        setattr(existing_chunk, attr, getattr(chunk, attr))
            else:
                seen[key] = len(merged_chunks)
                chunk.retrieval_sources = normalized_sources
                merged_chunks.append(chunk)
                
        # Sort retrieval_sources stably and assign to sources
        source_order = {
            "dense": 0,
            "lexical": 1,
            "graph_neighbor": 2,
            "graph_concept_retrieval": 3,
            "graph_bridge_retrieval": 4,
            "graph_concept": 3,
            "graph_bridge": 4
        }
        for chunk in merged_chunks:
            chunk.retrieval_sources.sort(key=lambda x: source_order.get(x.get("source", ""), 99))
            
            # Map sources to strings for chunk.sources
            sources_set = set()
            for s in chunk.retrieval_sources:
                src = s.get("source")
                if src == "dense":
                    sources_set.add("base_dense")
                elif src == "lexical":
                    sources_set.add("base_lexical")
                elif src in ("graph_neighbor", "graph_neighbors_in_rrf"):
                    sources_set.add("graph_neighbor")
                elif src in ("graph_concept_retrieval", "graph_concept"):
                    sources_set.add("graph_concept")
                elif src in ("graph_bridge_retrieval", "graph_bridge"):
                    sources_set.add("graph_bridge")
                elif src:
                    sources_set.add(src)
            order_list = ["base_dense", "base_lexical", "graph_neighbor", "graph_concept", "graph_bridge"]
            chunk.sources = sorted(list(sources_set), key=lambda x: order_list.index(x) if x in order_list else 99)
            
        return merged_chunks

    def _classify_query_concepts(self, query: str, all_concepts: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Classifies all extracted query concepts into strong and dropped concepts.
        Returns:
            strong_concepts: list of strong concept IDs
            dropped_info: list of dicts with {"concept": concept_id, "reason": reason}
        """
        # Stop/function words
        stop_words = {
            "do", "does", "did", "is", "are", "was", "were", "have", "has", "had", 
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "to", "from", 
            "in", "on", "at", "by", "with", "about", "how", "why", "what", "where", 
            "when", "who", "which", "this", "that", "these", "those", "it", "its", 
            "they", "them", "their", "we", "us", "our", "you", "your", "i", "me", "my", 
            "he", "him", "his", "she", "her",
            "как", "что", "это", "и", "в", "во", "на", "с", "со", "ли", "бы", "же", 
            "то", "этот", "эта", "эти", "для", "о", "об", "обо", "обоих"
        }
        
        # We need to map concept IDs to their canonical name/aliases to check token count
        # and if they are subconcepts of multi-token concepts.
        concepts_data = {}
        for cid in all_concepts:
            concept_node = self.graph_repo.get_concept(cid)
            name = cid
            aliases = []
            if concept_node:
                props = concept_node.properties or {}
                name = props.get("name", cid)
                aliases = props.get("aliases", [])
            concepts_data[cid] = {"name": name, "aliases": aliases}

        # Step 1: Detect multi-token vs single-token
        multi_token_concepts = []
        single_token_concepts = []
        
        dropped_info = []
        
        for cid in all_concepts:
            data = concepts_data[cid]
            name = data["name"]
            aliases = data["aliases"]
            
            # Check if function word
            clean_name = name.lower().strip()
            if clean_name in stop_words or cid.lower() in stop_words:
                dropped_info.append({"concept": cid, "reason": "function_word"})
                continue
                
            # A concept is multi-token if its ID or name or any alias has 2+ words (separated by spaces or underscores/hyphens)
            all_representations = [cid, name] + aliases
            is_multi = False
            for rep in all_representations:
                if not rep:
                    continue
                words = re.split(r'[\s_\-]+', rep.strip())
                words = [w for w in words if w]
                if len(words) >= 2:
                    is_multi = True
                    break
            
            if is_multi:
                multi_token_concepts.append(cid)
            else:
                single_token_concepts.append(cid)
                
        # Step 2: Check subconcept coverage for single-token concepts
        remaining_single_token = []
        for cid in single_token_concepts:
            name = concepts_data[cid]["name"].lower()
            # If this single-token concept's ID or name is a substring of any multi-token concept's ID or name
            is_sub = False
            for mt_cid in multi_token_concepts:
                mt_name = concepts_data[mt_cid]["name"].lower()
                if (cid in mt_cid) or (name in mt_name):
                    is_sub = True
                    break
            if is_sub:
                dropped_info.append({"concept": cid, "reason": "covered_by_more_specific_multitoken_concept"})
            else:
                remaining_single_token.append(cid)
                
        # Step 3: Fetch IDF for single-token concepts if policy is allow_high_idf or idf_if_no_multitoken
        policy = config.graph_retrieval_single_token_concept_policy
        use_idf = config.graph_retrieval_use_concept_idf
        
        strong_single_token = []
        
        def is_technical_concept(cid: str) -> bool:
            name = concepts_data[cid]["name"]
            token_lower = name.lower().strip()
            cid_lower = cid.lower().strip()
            
            whitelist = {
                "decimation", "quantization", "latency", "ablation", "db2", "db3",
                "bandwidth", "throughput", "precision", "recall", "overfitting", "underfitting",
                "embeddings", "gradient", "tensor", "matrices", "matrix", "vector", "vectors",
                "neurons", "neuron", "entropy", "heuristics", "heuristic", "optimization",
                "optima", "minimum", "maximum", "convex", "concave", "stochastic",
                "regression", "classification", "clustering", "dimensionality",
                "variance", "covariance", "eigenvalue", "eigenvector", "f1-score",
                "resnet", "transformer", "bert", "lstm", "rnn", "cnn", "mlp", "gan",
                "hyperparameter", "hyperparameters", "epoch", "epochs", "backpropagation"
            }
            if token_lower in whitelist or cid_lower in whitelist:
                return True
            if any(c.isdigit() for c in token_lower) or any(c.isdigit() for c in cid_lower):
                return True
            technical_suffixes = (
                "ation", "ization", "ency", "ity", "ism", "tropy", "metric", "graph",
                "nomial", "tosis", "lution", "gression", "duction", "morphic", "layer",
                "epoch", "node", "edge", "mesh", "query", "index", "token", "prompt"
            )
            if (len(token_lower) >= 5 and token_lower.endswith(technical_suffixes)) or \
               (len(cid_lower) >= 5 and cid_lower.endswith(technical_suffixes)):
                return True
            return False
            
        if policy == "drop":
            for cid in remaining_single_token:
                if is_technical_concept(cid):
                    strong_single_token.append(cid)
                else:
                    dropped_info.append({"concept": cid, "reason": "generic_single_token"})
        elif policy == "idf_if_no_multitoken":
            if len(multi_token_concepts) > 0:
                for cid in remaining_single_token:
                    if is_technical_concept(cid):
                        strong_single_token.append(cid)
                    else:
                        dropped_info.append({"concept": cid, "reason": "generic_single_token"})
            else:
                if use_idf and remaining_single_token:
                    idfs = self.graph_repo.get_concept_idf(remaining_single_token)
                    for cid in remaining_single_token:
                        if is_technical_concept(cid):
                            strong_single_token.append(cid)
                            continue
                        idf_val = idfs.get(cid, 0.0)
                        if idf_val >= 1.0:
                            strong_single_token.append(cid)
                        else:
                            dropped_info.append({"concept": cid, "reason": "generic_single_token"})
                else:
                    strong_single_token.extend(remaining_single_token)
        elif policy == "allow_high_idf":
            if use_idf and remaining_single_token:
                idfs = self.graph_repo.get_concept_idf(remaining_single_token)
                for cid in remaining_single_token:
                    if is_technical_concept(cid):
                        strong_single_token.append(cid)
                        continue
                    idf_val = idfs.get(cid, 0.0)
                    if idf_val >= 1.0:
                        strong_single_token.append(cid)
                    else:
                        dropped_info.append({"concept": cid, "reason": "generic_single_token"})
            else:
                strong_single_token.extend(remaining_single_token)
        else:
            strong_single_token.extend(remaining_single_token)
            
        strong_concepts = sorted(multi_token_concepts + strong_single_token)
        return strong_concepts, dropped_info

    def search_chunks_within_papers(
        self,
        query: str,
        paper_ids: List[str],
        limit_per_paper: int = 1,
        total_limit: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        if not paper_ids:
            return []
        query_embedding = self.emb_engine.get_embedding(query)
        chunks_with_scores = self.graph_repo.search_chunks_within_papers(
            query_embedding=query_embedding,
            paper_ids=paper_ids,
            limit_per_paper=limit_per_paper
        )
        if total_limit is not None:
            chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
            chunks_with_scores = chunks_with_scores[:total_limit]
        return chunks_with_scores

    def _extract_query_concepts(self, query: str) -> List[str]:
        if not query or not isinstance(query, str) or not query.strip():
            return []
        
        try:
            aliases_map = self.graph_repo.get_concept_aliases()
            concepts = self.graph_repo.get_nodes_by_label("Concept")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to fetch concepts for query concept extraction: {e}")
            return []
            
        name_to_ids = {}
        for node_id, props in concepts:
            cid = node_id
            names = set()
            names.add(cid.lower())
            names.add(cid.replace('_', ' ').replace('-', ' ').lower())
            
            canonical_name = props.get("name")
            if canonical_name:
                names.add(canonical_name.lower().strip())
                
            for alias in props.get("aliases", []):
                if isinstance(alias, str):
                    names.add(alias.lower().strip())
                    
            for name in names:
                if name:
                    if name not in name_to_ids:
                        name_to_ids[name] = set()
                    name_to_ids[name].add(cid)
                    
        for alias, canonical in aliases_map.items():
            from src.models import slugify
            cid = slugify(canonical)
            alias_clean = alias.lower().strip()
            if alias_clean not in name_to_ids:
                name_to_ids[alias_clean] = set()
            name_to_ids[alias_clean].add(cid)
            
        from src.services.normalization_pipeline import get_spacy_nlp
        nlp = get_spacy_nlp()
        
        def lemmatize_phrase(text: str) -> str:
            if not nlp:
                return text.lower()
            doc = nlp(text.lower())
            return " ".join(t.lemma_ for t in doc)
            
        lemma_to_ids = {}
        for phrase, cids in name_to_ids.items():
            phrase_lemma = lemmatize_phrase(phrase)
            if phrase_lemma not in lemma_to_ids:
                lemma_to_ids[phrase_lemma] = set()
            lemma_to_ids[phrase_lemma].update(cids)
            
        query_lower = query.lower()
        query_lemma = lemmatize_phrase(query)
        
        matched_concept_ids = set()
        import re
        
        for phrase, cids in name_to_ids.items():
            pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, query_lower):
                matched_concept_ids.update(cids)
                
        for phrase_lemma, cids in lemma_to_ids.items():
            pattern = r'\b' + re.escape(phrase_lemma) + r'\b'
            if re.search(pattern, query_lemma):
                matched_concept_ids.update(cids)
                
        return sorted(list(matched_concept_ids))

    def _build_selected_sources_card(self, trimmed_chunks: List[Any], query_concepts: List[str]) -> Optional[str]:
        paper_to_indexes = {}
        for idx, item in enumerate(trimmed_chunks, start=1):
            c = item[0] if isinstance(item, tuple) else item
            if getattr(c, "paper_id", None):
                if c.paper_id not in paper_to_indexes:
                    paper_to_indexes[c.paper_id] = []
                paper_to_indexes[c.paper_id].append(idx)
                
        selected_pids = list(paper_to_indexes.keys())
        if len(selected_pids) < 2:
            return None
            
        def ref(pid):
            return f"[{paper_to_indexes[pid][0]}]"

        try:
            concepts_list = self.graph_repo.get_concepts_for_papers(selected_pids)
        except Exception:
            concepts_list = []
            
        paper_concepts = {pid: {} for pid in selected_pids}
        for paper_id, concept_id, concept_name in concepts_list:
            if paper_id in paper_concepts:
                paper_concepts[paper_id][concept_id] = concept_name

        try:
            citations = self.graph_repo.get_citation_neighbors(selected_pids)
        except Exception:
            citations = []
            
        facts = []
        seen_pairs = set()
        
        for q_concept in query_concepts:
            matching_pids = [pid for pid in selected_pids if q_concept in paper_concepts[pid]]
            matching_pids.sort(key=lambda p: paper_to_indexes[p][0])
            for i in range(len(matching_pids)):
                for j in range(i + 1, len(matching_pids)):
                    p1, p2 = matching_pids[i], matching_pids[j]
                    pair_key = (min(p1, p2), max(p1, p2), "query", q_concept)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        c_name = paper_concepts[p1][q_concept]
                        p1_idx = paper_to_indexes[p1][0]
                        p2_idx = paper_to_indexes[p2][0]
                        facts.append((
                            1,
                            0.0,
                            p1_idx,
                            p2_idx,
                            f"- {ref(p1)} and {ref(p2)} both mention concept \"{c_name}\"."
                        ))
                        
        for seed_id, candidate_id, direction, _ in citations:
            if candidate_id in paper_to_indexes:
                pair_key = (min(seed_id, candidate_id), max(seed_id, candidate_id), "citation")
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    if direction == "seed_cites_candidate":
                        fact_text = f"- {ref(seed_id)} cites {ref(candidate_id)}."
                    else:
                        fact_text = f"- {ref(candidate_id)} cites {ref(seed_id)}."
                    p1_idx = paper_to_indexes[seed_id][0]
                    p2_idx = paper_to_indexes[candidate_id][0]
                    facts.append((
                        2,
                        0.0,
                        min(p1_idx, p2_idx),
                        max(p1_idx, p2_idx),
                        fact_text
                    ))
                    
        shared_concepts_map = {}
        for pid in selected_pids:
            for cid, cname in paper_concepts[pid].items():
                if cid not in query_concepts:
                    if cid not in shared_concepts_map:
                        shared_concepts_map[cid] = (cname, set())
                    shared_concepts_map[cid][1].add(pid)
                    
        shared_cids = [cid for cid, (cname, pids) in shared_concepts_map.items() if len(pids) >= 2]
        idfs = {}
        if shared_cids:
            try:
                doc_freqs = self.graph_repo.get_concept_document_frequencies(shared_cids)
                total_papers = self.graph_repo.get_total_paper_count()
                import math
                for cid in shared_cids:
                    df = doc_freqs.get(cid, 0)
                    idfs[cid] = math.log((1 + total_papers) / (1 + df))
            except Exception:
                idfs = {cid: 0.0 for cid in shared_cids}
                
        for cid in shared_cids:
            cname, pids = shared_concepts_map[cid]
            pids_list = list(pids)
            pids_list.sort(key=lambda p: paper_to_indexes[p][0])
            for i in range(len(pids_list)):
                for j in range(i + 1, len(pids_list)):
                    p1, p2 = pids_list[i], pids_list[j]
                    pair_key = (min(p1, p2), max(p1, p2), "shared", cid)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        idf_val = idfs.get(cid, 0.0)
                        p1_idx = paper_to_indexes[p1][0]
                        p2_idx = paper_to_indexes[p2][0]
                        facts.append((
                            3,
                            -idf_val,
                            p1_idx,
                            p2_idx,
                            f"- {ref(p1)} and {ref(p2)} are connected through concept \"{cname}\"."
                        ))
                        
        facts.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        selected_facts = [f[4] for f in facts[:5]]
        if not selected_facts:
            return None
            
        card_content = "Graph links among selected sources:\n" + "\n".join(selected_facts)
        return card_content

    def _parse_trace_dir_from_argv(self) -> Optional[Path]:
        """Helper to extract output directory from command line arguments for fallback."""
        import sys
        for idx, arg in enumerate(sys.argv):
            if arg in ("--output", "-o"):
                if idx + 1 < len(sys.argv):
                    return Path(sys.argv[idx + 1]).parent
            elif arg.startswith("--output="):
                return Path(arg.split("=", 1)[1]).parent
        return None

    def _write_graph_retrieval_trace(self, query: str, final_chunks: List[Any], trimmed_chunks: Optional[List[Any]] = None) -> None:
        import sys
        import json
        from pathlib import Path
        is_benchmark_mode = (self.trace_dir is not None) or any(
            arg in sys.argv for arg in ("--dataset", "--baselines", "run_pipeline.py", "run_benchmarks.py", "run_custom_retrieve.py")
        )
        
        if not (is_benchmark_mode or getattr(config, "graph_retrieval_trace_enabled", False)):
            return
            
        last_trace = getattr(self, "_last_graph_trace", {})
        if not last_trace:
            return
            
        query_id = "Q"
        baseline = "B6"
        category = "general"
        if hasattr(self, "current_trace") and self.current_trace is not None:
            query_id = self.current_trace.get("query_id", "Q")
            baseline = self.current_trace.get("baseline", "B6")
            category = self.current_trace.get("category", "general")
            
        survived_set = set()
        if trimmed_chunks is not None:
            for item in trimmed_chunks:
                c = item[0] if isinstance(item, tuple) else item
                survived_set.add(c.id)
        else:
            for item in final_chunks:
                c = item[0] if isinstance(item, tuple) else item
                survived_set.add(c.id)
                
        graph_chunks_before_rerank = last_trace.get("graph_chunks_before_rerank", [])
        graph_survived_ids = [cid for cid in graph_chunks_before_rerank if cid in survived_set]
        
        before_count = len(graph_chunks_before_rerank)
        survived_count = len(graph_survived_ids)
        survival_rate = survived_count / before_count if before_count > 0 else 0.0
        
        final_context_chunk_ids = list(survived_set)
        final_context_paper_ids = list({
            (item[0].paper_id if isinstance(item, tuple) else item.paper_id)
            for item in (trimmed_chunks if trimmed_chunks is not None else final_chunks)
            if getattr(item[0] if isinstance(item, tuple) else item, "paper_id", None)
        })
        
        rerank_positions = []
        best_rank = None
        for idx, item in enumerate(final_chunks, start=1):
            c = item[0] if isinstance(item, tuple) else item
            sources = getattr(c, "retrieval_sources", [])
            graph_sources = []
            for s in sources:
                src = s.get("source") if isinstance(s, dict) else s
                if src in ("graph_neighbor", "graph_concept", "graph_bridge", "graph_concept_retrieval", "graph_bridge_retrieval"):
                    graph_sources.append(src)
            if graph_sources:
                for gs in graph_sources:
                    rerank_positions.append({
                        "chunk_id": c.id,
                        "paper_id": c.paper_id,
                        "source": gs,
                        "rank_after_rerank": idx,
                        "survived_final_context": c.id in survived_set
                    })
                if best_rank is None or idx < best_rank:
                    best_rank = idx
                    
        if hasattr(self, "current_trace") and self.current_trace is not None:
            self.current_trace["whether_graph_neighbor_chunk_survived_into_final_context"] = survived_count > 0
            
        trace_entry = {
            "query_id": query_id,
            "query": query,
            "baseline": baseline,
            "category": category,
            
            "graph_retrieval_enabled": last_trace.get("graph_retrieval_enabled", getattr(config, "graph_retrieval_enabled", True)),
            "graph_retrieval_skip_reason": last_trace.get("graph_retrieval_skip_reason", None),
            
            "query_concepts_all": last_trace.get("query_concepts_all", []),
            "query_concepts_strong": last_trace.get("query_concepts_strong", []),
            "query_concepts_dropped": last_trace.get("query_concepts_dropped", []),
            
            "graph_neighbor_nodes_total": last_trace.get("graph_neighbor_nodes_total", 0),
            "graph_neighbor_paper_nodes_count": last_trace.get("graph_neighbor_paper_nodes_count", 0),
            "graph_neighbor_local_papers_count": last_trace.get("graph_neighbor_local_papers_count", 0),
            "graph_neighbor_papers_with_chunks_count": last_trace.get("graph_neighbor_papers_with_chunks_count", 0),
            "graph_neighbor_placeholder_or_external_count": last_trace.get("graph_neighbor_placeholder_or_external_count", 0),
            "graph_neighbor_non_paper_nodes_count": last_trace.get("graph_neighbor_non_paper_nodes_count", 0),
            "graph_neighbor_chunks_retrieved_count": last_trace.get("graph_neighbor_chunks_retrieved_count", 0),
            "graph_neighbor_resolution_sample": last_trace.get("graph_neighbor_resolution_sample", []),
            
            "query_concepts": last_trace.get("query_concepts", []),
            "seed_paper_ids": last_trace.get("seed_paper_ids", []),
            "graph_concept_candidate_papers": last_trace.get("graph_concept_candidate_papers", []),
            "graph_bridge_candidate_papers": last_trace.get("graph_bridge_candidate_papers", []),
            "graph_chunks_before_rerank": graph_chunks_before_rerank,
            "graph_chunks_before_rerank_count": before_count,
            "graph_candidate_rerank_positions": rerank_positions,
            "best_graph_candidate_rank_after_rerank": best_rank,
            "graph_chunks_survived_final_context": graph_survived_ids,
            "graph_chunks_survived_final_context_count": survived_count,
            "graph_survival_rate": survival_rate,
            "final_context_paper_ids": sorted(final_context_paper_ids),
            "final_context_chunk_ids": sorted(final_context_chunk_ids),
            "distinct_papers_in_final_context": len(final_context_paper_ids),
            "graph_candidate_source_breakdown": last_trace.get("graph_candidate_source_breakdown", {"graph_concept": 0, "graph_bridge": 0, "graph_neighbor": 0}),
            
            "base_candidates_count": last_trace.get("base_candidates_count", 0),
            "base_candidate_chunk_ids": last_trace.get("base_candidate_chunk_ids", []),
            "base_candidate_paper_ids": last_trace.get("base_candidate_paper_ids", []),
            
            "graph_neighbor_paper_ids_count": last_trace.get("graph_neighbor_paper_ids_count", 0),
            "graph_neighbor_paper_ids_sample": last_trace.get("graph_neighbor_paper_ids_sample", []),
            
            "graph_chunk_candidates_count": last_trace.get("graph_chunk_candidates_count", 0),
            "graph_chunk_candidate_chunk_ids": last_trace.get("graph_chunk_candidate_chunk_ids", []),
            "graph_chunk_candidate_paper_ids": last_trace.get("graph_chunk_candidate_paper_ids", []),
            
            "merged_candidates_count_before_reranker": last_trace.get("merged_candidates_count_before_reranker", 0),
            "merged_candidate_chunk_ids": last_trace.get("merged_candidate_chunk_ids", []),
            
            "reranker_input_count_before_limit": last_trace.get("reranker_input_count_before_limit", len(final_chunks)),
            "reranker_input_count_after_limit": last_trace.get("reranker_input_count_after_limit", len(final_chunks)),
            "candidate_count_after_reranker": len(final_chunks),
        }
        
        # Add new diagnostic fields to trace entry
        trace_entry.update({
            "external_bridge_nodes_seen": last_trace.get("external_bridge_nodes_seen", []),
            "external_bridge_nodes_used": last_trace.get("external_bridge_nodes_used", []),
            "local_candidates_from_intra_paper": last_trace.get("local_candidates_from_intra_paper", 0),
            "local_candidates_from_concepts": last_trace.get("local_candidates_from_concepts", 0),
            "local_candidates_from_direct_edges": last_trace.get("local_candidates_from_direct_edges", 0),
            "local_candidates_from_external_bridges": last_trace.get("local_candidates_from_external_bridges", 0),
        })
        
        if self.trace_dir:
            trace_dir = Path(self.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_file = trace_dir / "graph_retrieval_trace.jsonl"
        else:
            benchmark_dir = self._parse_trace_dir_from_argv()
            if benchmark_dir:
                trace_dir = benchmark_dir
                if trace_dir.name != "traces":
                    trace_dir = trace_dir / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_file = trace_dir / "graph_retrieval_trace.jsonl"
            else:
                trace_file = Path("graph_retrieval_trace.jsonl")
            
        try:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to write graph retrieval trace: {e}")

    def retrieve_relevant_chunks(self, query: str, limit: int = 5, paper_id: Optional[str] = None, filters: Optional[dict] = None, hyde_responses: Optional[int] = None) -> List[tuple[Chunk, float]]:
        # Focused document RAG: cosine similarity search directly over document chunks in Python
        if not query or not isinstance(query, str) or not query.strip():
            return []

        if paper_id:
            import numpy as np
            chunks = self.vector_repo.get_chunks_for_paper(paper_id)
            if not chunks:
                return []
            query_emb = np.array(self.emb_engine.get_embedding(query), dtype=np.float32)
            scored = []
            for c in chunks:
                c_emb = np.array(c.embedding, dtype=np.float32)
                sim = float(np.dot(c_emb, query_emb)) if len(c_emb) == len(query_emb) else 0.0
                scored.append((c, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Apply reranking on candidate chunks if reranker is available
            candidates = [s[0] for s in scored[:limit * 2]]
            self._last_pre_rerank_scores = [float(s[1]) for s in scored[:limit * 2]]
            if config.rag_components.get("reranker", True):
                try:
                    reranker = self._get_reranker()
                    pairs = [(query, c.text_content) for c in candidates]
                    scores = reranker.predict(pairs)
                    scored_candidates = list(zip(candidates, scores))
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    return [(chunk, float(score)) for chunk, score in scored_candidates[:limit]]
                except Exception:
                    return scored[:limit]
            else:
                return scored[:limit]

        # 1. Run Query Intent Classifier if no explicit filters are passed
        if filters is None and config.rag_components.get("intent_classifier", True):
            query, filters = self._classify_intent_and_extract_filters(query)

        # Dense + FTS5 + RRF + Rerank
        query_embedding = None
        expanded_queries = self._expand_query(query)
        
        # Determine dense retrieval limit per query.
        dense_limit = limit * 2 if len(expanded_queries) == 1 else limit
        
        all_dense_results = {}
        if config.rag_components.get("dense_search", True):
            variant_embs = self.emb_engine.get_embeddings(expanded_queries, is_query=True)
            if not isinstance(variant_embs, (list, tuple)):
                variant_embs = [self.emb_engine.get_embedding(q, is_query=True) for q in expanded_queries]
            for variant, variant_emb in zip(expanded_queries, variant_embs):
                if variant == query:
                    query_embedding = variant_emb
                if filters:
                    dense_res = self.vector_repo.search_similar_chunks(variant_emb, limit=dense_limit, filters=filters)
                else:
                    dense_res = self.vector_repo.search_similar_chunks(variant_emb, limit=dense_limit)
                for chunk, score in dense_res:
                    if not hasattr(chunk, "retrieval_sources"):
                        chunk.retrieval_sources = []
                    if not any(s["source"] == "dense" for s in chunk.retrieval_sources):
                        chunk.retrieval_sources.append({"source": "dense"})
                    if chunk.id not in all_dense_results:
                        all_dense_results[chunk.id] = (chunk, score)
                    else:
                        existing_chunk, existing_score = all_dense_results[chunk.id]
                        if score > existing_score:
                            all_dense_results[chunk.id] = (chunk, score)

            # 2. Run HyDE (Hypothetical Document Embeddings) if enabled
            if config.hyde_enabled and config.rag_components.get("hyde", True):
                if hyde_responses is None:
                    import sys
                    argv_hyde = None
                    for idx, arg in enumerate(sys.argv):
                        if arg == "--hyde":
                            if idx + 1 < len(sys.argv):
                                try:
                                    argv_hyde = int(sys.argv[idx + 1])
                                except ValueError:
                                    pass
                        elif arg.startswith("--hyde="):
                            try:
                                argv_hyde = int(arg.split("=", 1)[1])
                            except ValueError:
                                    pass
                    if argv_hyde is not None:
                        hyde_responses = argv_hyde
                    else:
                        hyde_responses = _safe_int(getattr(config, "hyde_count", 1), 1)

                for _ in range(hyde_responses):
                    try:
                        hypothetical = self.llm_engine.generate_response(
                            prompt=prompts.get_prompt("rag", "hyde", query=query),
                            max_tokens=_safe_int(config.hyde_max_tokens, 300)
                        )
                        con.debug(f"Generated hypothetical answer: {hypothetical}")
                        
                        hyp_emb = self.emb_engine.get_embedding(hypothetical)
                        if filters:
                            hyde_res = self.vector_repo.search_similar_chunks(hyp_emb, limit=limit * 2, filters=filters)
                        else:
                            hyde_res = self.vector_repo.search_similar_chunks(hyp_emb, limit=limit * 2)
                            
                        for chunk, score in hyde_res:
                            if not hasattr(chunk, "retrieval_sources"):
                                chunk.retrieval_sources = []
                            if not any(s["source"] == "dense" for s in chunk.retrieval_sources):
                                chunk.retrieval_sources.append({"source": "dense"})
                            if chunk.id not in all_dense_results:
                                all_dense_results[chunk.id] = (chunk, score)
                            else:
                                existing_chunk, existing_score = all_dense_results[chunk.id]
                                if score > existing_score:
                                    all_dense_results[chunk.id] = (chunk, score)
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"HyDE generation failed: {e}")
                            
        if config.rag_components.get("lexical_search", True):
            if filters:
                fts5_results = self.vector_repo.search_text_fts5(query, limit=limit * 2, filters=filters)
            else:
                fts5_results = self.vector_repo.search_text_fts5(query, limit=limit * 2)
            for chunk, _ in fts5_results:
                if not hasattr(chunk, "retrieval_sources"):
                    chunk.retrieval_sources = []
                if not any(s["source"] == "lexical" for s in chunk.retrieval_sources):
                    chunk.retrieval_sources.append({"source": "lexical"})
        else:
            fts5_results = []

        if hasattr(self, "current_trace") and self.current_trace is not None:
            dense_chunk_ids = list(all_dense_results.keys())
            lexical_chunk_ids = [c.id for c, _ in fts5_results]
            self.current_trace["seed_chunks_from_lexical_dense"] = {
                "lexical": lexical_chunk_ids,
                "dense": dense_chunk_ids
            }
            seed_papers = set()
            for chunk, _ in all_dense_results.values():
                if chunk.paper_id:
                    seed_papers.add(chunk.paper_id)
            for chunk, _ in fts5_results:
                if chunk.paper_id:
                    seed_papers.add(chunk.paper_id)
            self.current_trace["seed_paper_id_list"] = list(seed_papers)

        if config.rag_components.get("graph_neighbors_in_rrf", False) and not config.graph_retrieval_enabled:
            paper_ids = set()
            for chunk, _ in all_dense_results.values():
                pid = getattr(chunk, "paper_id", None)
                if pid:
                    paper_ids.add(pid)
            for chunk, _ in fts5_results:
                pid = getattr(chunk, "paper_id", None)
                if pid:
                    paper_ids.add(pid)
            
            neighbor_paper_ids = set()
            order = _safe_int(getattr(config, "b6_graph_neighbors_order", 2), 2)
            for pid in paper_ids:
                neighbors = self.graph_repo.get_neighbors(pid, max_depth=order)
                for src_id, src_label, _, tgt_id, tgt_label, _ in neighbors:
                    if src_label in ("Paper", "UserNote") and src_id not in paper_ids:
                        neighbor_paper_ids.add(src_id)
                    if tgt_label in ("Paper", "UserNote") and tgt_id not in paper_ids:
                        neighbor_paper_ids.add(tgt_id)

            if hasattr(self, "current_trace") and self.current_trace is not None:
                self.current_trace["graph_neighbor_paper_id_list"] = list(neighbor_paper_ids)
            
            if neighbor_paper_ids:
                import numpy as np
                expanded_embs = []
                for variant in expanded_queries:
                    emb = self.emb_engine.get_embedding(variant)
                    expanded_embs.append(np.array(emb, dtype=np.float32))
                
                for neighbor_pid in neighbor_paper_ids:
                    neighbor_chunks = self.vector_repo.get_chunks_for_paper(neighbor_pid)
                    for chunk in neighbor_chunks:
                        cid = getattr(chunk, "id", None)
                        if cid and cid not in all_dense_results:
                            if not hasattr(chunk, "retrieval_sources"):
                                chunk.retrieval_sources = []
                            if not any(s["source"] == "graph_neighbors_in_rrf" for s in chunk.retrieval_sources):
                                chunk.retrieval_sources.append({"source": "graph_neighbors_in_rrf"})
                            max_sim = -1.0
                            c_emb = getattr(chunk, "embedding", None)
                            if c_emb:
                                chunk_emb = np.array(c_emb, dtype=np.float32)
                                for q_emb in expanded_embs:
                                    if len(chunk_emb) == len(q_emb):
                                        sim = float(np.dot(chunk_emb, q_emb))
                                        if sim > max_sim:
                                            max_sim = sim
                            if max_sim == -1.0:
                                max_sim = 0.0
                            all_dense_results[cid] = (chunk, max_sim)

        dense_results = list(all_dense_results.values())
        dense_results.sort(key=lambda x: x[1], reverse=True)
        
        if not dense_results and not fts5_results:
            return []
            
        id_to_chunk = {}
        for chunk, _ in dense_results:
            id_to_chunk[chunk.id] = chunk
        for chunk, _ in fts5_results:
            id_to_chunk[chunk.id] = chunk

        if config.rag_components.get("dynamic_alpha_blending", True):
            fts_weight = _safe_float(config.dynamic_alpha_val_high, 1.0)
            if fts5_results:
                max_bm25 = max(score for _, score in fts5_results)
                if max_bm25 < _safe_float(config.dynamic_alpha_threshold_low, 1.0):
                    fts_weight = _safe_float(config.dynamic_alpha_val_low, 0.2)
                elif max_bm25 < _safe_float(config.dynamic_alpha_threshold_mid, 3.0):
                    fts_weight = _safe_float(config.dynamic_alpha_val_mid, 0.5)
            else:
                fts_weight = 0.0
        else:
            fts_weight = 1.0 if fts5_results else 0.0
            
        dense_weight = 1.0

        if config.rag_components.get("rrf", True):
            rrf_scores = {}
            for rank, (chunk, _) in enumerate(dense_results, start=1):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + dense_weight * (1.0 / (_safe_float(config.rrf_k, 60.0) + rank))
                
            for rank, (chunk, _) in enumerate(fts5_results, start=1):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + fts_weight * (1.0 / (_safe_float(config.rrf_k, 60.0) + rank))
                
            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            candidate_ids = sorted_ids[:limit * 2]
            candidates = [id_to_chunk[cid] for cid in candidate_ids if cid in id_to_chunk]
        else:
            seen_ids = set()
            candidates = []
            for chunk, _ in dense_results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    candidates.append(chunk)
            for chunk, _ in fts5_results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    candidates.append(chunk)
            candidates = candidates[:limit * 2]
            rrf_scores = {c.id: 1.0 for c in candidates}
            sorted_ids = [c.id for c in candidates]
        
        # === DET DETERMINISTIC GRAPH RETRIEVAL ===
        # Gather seed papers
        seed_paper_ids = list({c.paper_id for c in candidates if getattr(c, "paper_id", None)})
        
        # Extract query concepts
        all_query_concepts = self._extract_query_concepts(query)
        strong_query_concepts, dropped_query_concepts = self._classify_query_concepts(query, all_query_concepts)
        
        # Initial configurations
        master_enabled = config.graph_retrieval_enabled
        concept_enabled = config.graph_concept_retrieval_enabled
        bridge_enabled = config.graph_bridge_retrieval_enabled

        graph_retrieval_enabled = master_enabled or concept_enabled or bridge_enabled

        # If master switch is on and no sub-toggles are explicitly enabled, enable both sub-toggles
        if master_enabled and not config.graph_concept_retrieval_enabled and not config.graph_bridge_retrieval_enabled:
            concept_enabled = True
            bridge_enabled = True
        
        skip_reason = None
        if not graph_retrieval_enabled:
            skip_reason = "disabled"

        # Determine budget
        base_candidate_count = len(candidates)
        if config.graph_retrieval_candidate_budget_mode == "fixed":
            derived_graph_candidate_budget = config.graph_retrieval_max_graph_chunk_candidates
        else:  # mirror_base
            derived_graph_candidate_budget = base_candidate_count

        # Telemetry/diagnostics data structure initialization
        telemetry = {
            "query_concepts_all": all_query_concepts,
            "query_concepts_strong": strong_query_concepts,
            "query_concepts_dropped": dropped_query_concepts,
            "seed_paper_ids": seed_paper_ids,
            "graph_neighbor_nodes_total": 0,
            "graph_neighbor_local_papers_count": 0,
            "graph_neighbor_placeholder_or_external_count": 0,
            "graph_neighbor_papers_with_chunks_count": 0,
            "external_bridge_nodes_seen": [],
            "external_bridge_nodes_used": [],
            "local_candidates_from_intra_paper": 0,
            "local_candidates_from_concepts": 0,
            "local_candidates_from_direct_edges": 0,
            "local_candidates_from_external_bridges": 0,
            "graph_chunks_before_rerank_count": 0,
            "graph_chunks_survived_final_context_count": 0,
            "graph_survival_rate": 0.0,
            "graph_retrieval_skip_reason": None,
        }
        telemetry["graph_branch_execution"] = {
            "graph_retrieval_enabled": bool(graph_retrieval_enabled),
            "graph_master_enabled": bool(master_enabled),
            "concept_branch_enabled": bool(concept_enabled),
            "concept_lookup_executed": False,
            "bridge_branch_enabled": bool(bridge_enabled),
            "bridge_lookup_executed": False,
            "intra_paper_branch_executed": False,
        }

        # Sets to hold results and metadata for trace
        unique_chunks = {}
        chunk_to_key = {}
        seen_chunk_ids = set()
        
        layer_0_chunks = []
        layer_1_chunks = []
        layer_2_chunks = []
        layer_3_chunks = []
        layer_4_chunks = []
        graph_candidates = []
        local_seed_papers = []

        # Local variables to track if any candidates were found or filtered
        was_filtered_due_to_no_chunks = False
        has_neighbors_in_graph = False
        has_external_neighbors_but_no_local_neighbors = False
        has_bridges_but_no_local_papers_reachable = False
        has_reachable_local_papers_but_no_chunks = False

        if graph_retrieval_enabled:
            # Query embedding for similarity scoring
            if query_embedding is None:
                query_embedding = self.emb_engine.get_embedding(query)
            chunks_per_paper = config.graph_retrieval_chunks_per_graph_paper

            # --- Layer 0: Existing base retrieval candidates ---
            layer_0_chunks = list(candidates)
            for idx, c in enumerate(layer_0_chunks):
                if not hasattr(c, "retrieval_sources"):
                    c.retrieval_sources = []
                if not any(s.get("source") == "dense" or s.get("source") == "lexical" for s in c.retrieval_sources):
                    c.retrieval_sources.append({"source": "dense"})
                
                # Check compatibility fields
                c.candidate_source = c.retrieval_sources[0].get("source") if c.retrieval_sources else "dense"
                
                c_id = getattr(c, "id", None)
                seen_chunk_ids.add(c_id)
                unique_chunks[c_id] = c
                chunk_to_key[c_id] = (0, idx, getattr(c, "paper_id", None) or "", c_id or "")

            # Identify seed papers
            local_seed_papers = [pid for pid in seed_paper_ids if self.graph_repo.is_local_indexed_paper(pid)]

            # Check seed paper existence for expansion
            if not seed_paper_ids:
                skip_reason = "no_seed_papers"
            elif not local_seed_papers:
                skip_reason = "no_local_seed_papers"
            else:
                # --- Layer 1: Intra-paper local expansion from seed papers ---
                if master_enabled:
                    telemetry["graph_branch_execution"]["intra_paper_branch_executed"] = True
                    for p_id in local_seed_papers:
                        # Get all chunks in the same paper
                        paper_chunks = self.vector_repo.get_chunks_for_paper(p_id)
                        # Filter out already seen in this layer
                        layer_1_seen = set()
                        layer_1_candidates = []
                        for c in paper_chunks:
                            if c.id not in layer_1_seen:
                                layer_1_seen.add(c.id)
                                layer_1_candidates.append(c)
                        if layer_1_candidates:
                            # Score them
                            scored_candidates = []
                            import numpy as np
                            q_vec = np.array(query_embedding, dtype=np.float32)
                            q_norm = np.linalg.norm(q_vec)
                            for c in layer_1_candidates:
                                c_vec = np.array(c.embedding, dtype=np.float32)
                                c_norm = np.linalg.norm(c_vec)
                                sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm)) if q_norm > 0 and c_norm > 0 else 0.0
                                scored_candidates.append((c, sim))
                            
                            # Sort and limit
                            scored_candidates.sort(key=lambda x: x[1], reverse=True)
                            for c, score in scored_candidates[:chunks_per_paper]:
                                # Setup candidate fields
                                src_metadata = {
                                    "source": "graph_neighbor", 
                                    "reason": "intra_paper_expansion",
                                    "paper_id": c.paper_id
                                }
                                if not hasattr(c, "retrieval_sources"):
                                    c.retrieval_sources = []
                                c.retrieval_sources.append(src_metadata)
                                c.graph_metadata = {
                                    "source": "graph_neighbor",
                                    "original_graph_node_id": c.paper_id,
                                    "canonical_paper_id": c.paper_id,
                                    "reason": "intra_paper_expansion"
                                }
                                c.candidate_source = "graph_neighbor"
                                
                                layer_1_chunks.append((c, score))
                                telemetry["local_candidates_from_intra_paper"] += 1

                # --- Layer 2: Local concept-linked expansion ---
                if strong_query_concepts and concept_enabled and all_query_concepts:
                    telemetry["graph_branch_execution"]["concept_lookup_executed"] = True
                    # Fetch concept papers
                    raw_concept_papers = self.graph_repo.get_papers_mentioning_concepts(strong_query_concepts)
                    concept_paper_ids = [p[0] for p in raw_concept_papers]
                    
                    # Filter for locally indexed papers only (chunks_count > 0)
                    local_concept_papers = [pid for pid in concept_paper_ids if self.graph_repo.is_local_indexed_paper(pid)]
                    
                    # Exclude seed papers from concept papers
                    local_concept_papers = [pid for pid in local_concept_papers if pid not in seed_paper_ids]
                    
                    if local_concept_papers:
                        concept_chunks_with_scores = self.search_chunks_within_papers(
                            query=query,
                            paper_ids=local_concept_papers,
                            limit_per_paper=chunks_per_paper
                        )
                        layer_2_seen = set()
                        for c, score in concept_chunks_with_scores:
                            if c.id not in layer_2_seen:
                                layer_2_seen.add(c.id)
                                # Setup concept metadata
                                src_metadata = {
                                    "source": "graph_concept_retrieval", 
                                    "reason": "concept_match",
                                    "paper_id": c.paper_id
                                }
                                if not hasattr(c, "retrieval_sources"):
                                    c.retrieval_sources = []
                                c.retrieval_sources.append(src_metadata)
                                c.graph_metadata = {
                                    "source": "graph_concept_retrieval",
                                    "original_graph_node_id": c.paper_id,
                                    "canonical_paper_id": c.paper_id,
                                    "reason": "concept_match"
                                }
                                c.candidate_source = "graph_concept"
                                
                                layer_2_chunks.append((c, score))
                                telemetry["local_candidates_from_concepts"] += 1

                # --- Layer 3: Direct local graph neighbors & Layer 4: Bridge-to-local expansion ---
                derived_edges = []
                if bridge_enabled and (master_enabled or all_query_concepts):
                    telemetry["graph_branch_execution"]["bridge_lookup_executed"] = True
                    derived_edges = self.graph_repo.get_derived_bridge_edges(local_seed_papers)
                
                # Sets for tracking neighbor resolution statistics
                neighbor_nodes_seen = set()
                local_neighbors = set()
                external_neighbors = set()
                papers_with_chunks = set()
                external_bridges_seen = set()
                external_bridges_used = set()

                layer_3_paper_ids = []
                layer_4_paper_ids = []
                paper_to_bridge = {} # target -> bridge node
                paper_to_seeds = {}  # target -> set of seed papers

                for edge in derived_edges:
                    tgt = edge["target_local_paper_id"]
                    # Skip if it is a seed paper
                    if tgt in seed_paper_ids:
                        continue
                    
                    src = edge["source_local_paper_id"]
                    if tgt not in paper_to_seeds:
                        paper_to_seeds[tgt] = set()
                    paper_to_seeds[tgt].add(src)

                    bridge_type = edge["bridge_relation_type"]
                    bridge_node = edge["bridge_node_id"]
                    
                    if bridge_type == 'DIRECT_LOCAL_CITATION':
                        # Direct local neighbor (Layer 3)
                        neighbor_nodes_seen.add(tgt)
                        local_neighbors.add(tgt)
                        papers_with_chunks.add(tgt)
                        if tgt not in layer_3_paper_ids:
                            layer_3_paper_ids.append(tgt)
                    else:
                        # Bridge connection (Layer 4)
                        if bridge_node:
                            external_bridges_seen.add(bridge_node)
                            neighbor_nodes_seen.add(bridge_node)
                            external_neighbors.add(bridge_node)
                        
                        neighbor_nodes_seen.add(tgt)
                        local_neighbors.add(tgt)
                        papers_with_chunks.add(tgt)
                        
                        if tgt not in layer_4_paper_ids:
                            layer_4_paper_ids.append(tgt)
                            if bridge_node:
                                paper_to_bridge[tgt] = bridge_node

                # Fill neighbor stats for telemetry
                telemetry["graph_neighbor_nodes_total"] = len(neighbor_nodes_seen)
                telemetry["graph_neighbor_local_papers_count"] = len(local_neighbors)
                telemetry["graph_neighbor_placeholder_or_external_count"] = len(external_neighbors)
                telemetry["graph_neighbor_papers_with_chunks_count"] = len(papers_with_chunks)
                telemetry["external_bridge_nodes_seen"] = sorted(list(external_bridges_seen))

                # Handle skip reason statistics tracking variables
                if neighbor_nodes_seen:
                    has_neighbors_in_graph = True
                if external_neighbors and not local_neighbors:
                    has_external_neighbors_but_no_local_neighbors = True
                if external_bridges_seen and not local_neighbors:
                    has_bridges_but_no_local_papers_reachable = True

                # Retrieve chunks for Layer 3
                if layer_3_paper_ids:
                    layer_3_chunks_raw = self.search_chunks_within_papers(
                        query=query,
                        paper_ids=layer_3_paper_ids,
                        limit_per_paper=chunks_per_paper
                    )
                    layer_3_seen = set()
                    for c, score in layer_3_chunks_raw:
                        if c.id not in layer_3_seen:
                            layer_3_seen.add(c.id)
                            connected_seeds = list(paper_to_seeds.get(c.paper_id, set()))
                            src_metadata = {
                                "source": "graph_bridge_retrieval" if bridge_enabled else "graph_neighbor", 
                                "reason": "direct_local_neighbor",
                                "paper_id": c.paper_id,
                                "connected_seed_papers": sorted(connected_seeds)
                            }
                            if not hasattr(c, "retrieval_sources"):
                                c.retrieval_sources = []
                            c.retrieval_sources.append(src_metadata)
                            c.graph_metadata = {
                                "source": "graph_neighbor",
                                "original_graph_node_id": c.paper_id,
                                "canonical_paper_id": c.paper_id,
                                "reason": "direct_local_neighbor"
                            }
                            c.candidate_source = "graph_neighbor"
                            
                            layer_3_chunks.append((c, score))
                            telemetry["local_candidates_from_direct_edges"] += 1

                # Retrieve chunks for Layer 4
                if layer_4_paper_ids:
                    layer_4_chunks_raw = self.search_chunks_within_papers(
                        query=query,
                        paper_ids=layer_4_paper_ids,
                        limit_per_paper=chunks_per_paper
                    )
                    layer_4_seen = set()
                    for c, score in layer_4_chunks_raw:
                        if c.id not in layer_4_seen:
                            layer_4_seen.add(c.id)
                            connected_seeds = list(paper_to_seeds.get(c.paper_id, set()))
                            src_metadata = {
                                "source": "graph_bridge_retrieval", 
                                "reason": "bridge_to_local_expansion",
                                "paper_id": c.paper_id,
                                "connected_seed_papers": sorted(connected_seeds)
                            }
                            if not hasattr(c, "retrieval_sources"):
                                c.retrieval_sources = []
                            c.retrieval_sources.append(src_metadata)
                            c.graph_metadata = {
                                "source": "graph_bridge_retrieval",
                                "original_graph_node_id": c.paper_id,
                                "canonical_paper_id": c.paper_id,
                                "reason": "bridge_to_local_expansion"
                            }
                            c.candidate_source = "graph_bridge"
                            
                            bridge_node = paper_to_bridge.get(c.paper_id)
                            if bridge_node:
                                c.bridge_node_id = bridge_node
                                c.graph_metadata["bridge_node_id"] = bridge_node
                                external_bridges_used.add(bridge_node)
                                
                            layer_4_chunks.append((c, score))
                            telemetry["local_candidates_from_external_bridges"] += 1

                telemetry["external_bridge_nodes_used"] = sorted(list(external_bridges_used))
                
                # Check reachable local papers but no chunks
                if (layer_3_paper_ids or layer_4_paper_ids) and not (layer_3_chunks or layer_4_chunks):
                    has_reachable_local_papers_but_no_chunks = True

            # Merge all active layers sequentially
            layers = [
                (1, layer_1_chunks),
                (2, layer_2_chunks),
                (3, layer_3_chunks),
                (4, layer_4_chunks)
            ]
            
            graph_expansion_chunks = []
            for layer_num, scored_list in layers:
                for idx, (c, score) in enumerate(scored_list):
                    # Check pre-reranker candidate validation: Drop candidates with chunks_count = 0
                    paper_id = getattr(c, "paper_id", None)
                    if paper_id:
                        cnt = self.graph_repo.chunks_count(paper_id)
                        if cnt == 0:
                            was_filtered_due_to_no_chunks = True
                            bridge_id = getattr(c, "bridge_node_id", None)
                            con.warning(
                                f"Dropping invalid graph candidate: {getattr(c, 'id', None)}, "
                                f"reason: paper has 0 chunks, "
                                f"source layer: {layer_num}, "
                                f"bridge node: {bridge_id}"
                            )
                            continue
                        
                    c_id = getattr(c, "id", None)
                    if c_id in unique_chunks:
                        existing_chunk = unique_chunks[c_id]
                        if not hasattr(existing_chunk, "retrieval_sources"):
                            existing_chunk.retrieval_sources = []
                        if not hasattr(c, "retrieval_sources"):
                            c.retrieval_sources = []
                        for s in c.retrieval_sources:
                            if not any(item.get("source") == s.get("source") for item in existing_chunk.retrieval_sources):
                                existing_chunk.retrieval_sources.append(s)
                        
                        # Re-calculate sources attribute
                        sources_set = set()
                        for s in existing_chunk.retrieval_sources:
                            src = s.get("source")
                            if src == "dense":
                                sources_set.add("base_dense")
                            elif src == "lexical":
                                sources_set.add("base_lexical")
                            elif src in ("graph_neighbor", "graph_neighbors_in_rrf"):
                                sources_set.add("graph_neighbor")
                            elif src in ("graph_concept_retrieval", "graph_concept"):
                                sources_set.add("graph_concept")
                            elif src in ("graph_bridge_retrieval", "graph_bridge"):
                                sources_set.add("graph_bridge")
                            elif src:
                                sources_set.add(src)
                        order_list = ["base_dense", "base_lexical", "graph_neighbor", "graph_concept", "graph_bridge"]
                        existing_chunk.sources = sorted(list(sources_set), key=lambda x: order_list.index(x) if x in order_list else 99)
                    else:
                        seen_chunk_ids.add(c_id)
                        unique_chunks[c_id] = c
                        chunk_to_key[c_id] = (layer_num, idx, getattr(c, "paper_id", None) or "", c_id or "")
                        graph_expansion_chunks.append(c)

            # Sort and apply budget limits if fixed budget mode is enabled
            if derived_graph_candidate_budget is not None:
                # Limit the new graph candidates before merging
                graph_expansion_chunks = graph_expansion_chunks[:derived_graph_candidate_budget]

            graph_candidates = graph_expansion_chunks

            # Apply candidate sorting and tie-breaking across all layers
            sorted_chunk_ids = sorted(unique_chunks.keys(), key=lambda cid: chunk_to_key[cid])
            candidates = [unique_chunks[cid] for cid in sorted_chunk_ids]

        # Determine skip_reason if no graph candidates survived
        if graph_retrieval_enabled and not graph_candidates:
            if not seed_paper_ids:
                skip_reason = "no_seed_papers"
            elif not local_seed_papers:
                skip_reason = "no_local_seed_papers"
            elif not all_query_concepts and not has_neighbors_in_graph:
                skip_reason = "no_query_concepts_extracted"
            elif has_external_neighbors_but_no_local_neighbors:
                skip_reason = "only_external_neighbors_found"
            elif has_bridges_but_no_local_papers_reachable:
                skip_reason = "no_local_papers_reachable_through_bridges"
            elif has_reachable_local_papers_but_no_chunks:
                skip_reason = "no_chunks_found_for_reachable_local_papers"
            elif was_filtered_due_to_no_chunks:
                skip_reason = "graph_candidates_filtered_no_chunks"
            else:
                skip_reason = "no_usable_graph_candidates"

        telemetry["graph_retrieval_skip_reason"] = skip_reason

        # Logging / Tracing values
        graph_chunk_ids = [c.id for c in graph_candidates]
        telemetry["graph_chunks_before_rerank"] = graph_chunk_ids
        telemetry["graph_chunks_before_rerank_count"] = len(graph_chunk_ids)

        breakdown = {"graph_neighbor": 0, "graph_concept": 0, "graph_bridge": 0}
        for chunk in graph_candidates:
            src = getattr(chunk, "candidate_source", "graph_neighbor")
            if src in breakdown:
                breakdown[src] += 1
        telemetry["graph_candidate_source_breakdown"] = breakdown

        self._last_graph_trace = {
            "branch_execution": telemetry["graph_branch_execution"],
            "branch_candidate_counts": {
                "intra_paper": telemetry["local_candidates_from_intra_paper"],
                "concept": telemetry["local_candidates_from_concepts"],
                "direct_edges": telemetry["local_candidates_from_direct_edges"],
                "external_bridges": telemetry["local_candidates_from_external_bridges"],
            },
            "graph_neighbors_in_rrf_enabled": bool(config.rag_components.get("graph_neighbors_in_rrf", False)),
            "graph_neighbors_in_rrf_executed": bool(config.rag_components.get("graph_neighbors_in_rrf", False)
                                                      and not config.graph_retrieval_enabled),
            "query_concepts_all": telemetry["query_concepts_all"],
            "query_concepts_strong": telemetry["query_concepts_strong"],
            "query_concepts_dropped": telemetry["query_concepts_dropped"],
            
            "graph_neighbor_nodes_total": telemetry["graph_neighbor_nodes_total"],
            "graph_neighbor_paper_nodes_count": telemetry["graph_neighbor_nodes_total"],  # Compatibility
            "graph_neighbor_local_papers_count": telemetry["graph_neighbor_local_papers_count"],
            "graph_neighbor_papers_with_chunks_count": telemetry["graph_neighbor_papers_with_chunks_count"],
            "graph_neighbor_placeholder_or_external_count": telemetry["graph_neighbor_placeholder_or_external_count"],
            "graph_neighbor_non_paper_nodes_count": 0,
            "graph_neighbor_chunks_retrieved_count": len(graph_chunk_ids),
            "graph_neighbor_resolution_sample": [],
            
            "query_concepts": telemetry["query_concepts_strong"],
            "seed_paper_ids": telemetry["seed_paper_ids"],
            "graph_concept_candidate_papers": [],
            "graph_bridge_candidate_papers": [],
            "graph_chunks_before_rerank": graph_chunk_ids,
            "graph_chunks_before_rerank_count": len(graph_chunk_ids),
            "graph_candidate_source_breakdown": breakdown,
            
            "graph_retrieval_enabled": graph_retrieval_enabled,
            "graph_retrieval_skip_reason": skip_reason,
            "base_candidates_count": len(layer_0_chunks),
            "base_candidate_chunk_ids": [c.id for c in layer_0_chunks],
            "base_candidate_paper_ids": list({c.paper_id for c in layer_0_chunks if getattr(c, "paper_id", None)}),
            "graph_neighbor_paper_ids_count": telemetry["graph_neighbor_papers_with_chunks_count"],
            "graph_neighbor_paper_ids_sample": [],
            "graph_chunk_candidates_count": len(graph_candidates),
            "graph_chunk_candidate_chunk_ids": graph_chunk_ids,
            "graph_chunk_candidate_paper_ids": list({c.paper_id for c in graph_candidates}),
            "merged_candidates_count_before_reranker": len(candidates),
            "merged_candidate_chunk_ids": [c.id for c in candidates],
            "reranker_input_count_before_limit": len(candidates),
            "reranker_input_count_after_limit": len(candidates),

            # New diagnostic fields
            "external_bridge_nodes_seen": telemetry["external_bridge_nodes_seen"],
            "external_bridge_nodes_used": telemetry["external_bridge_nodes_used"],
            "local_candidates_from_intra_paper": telemetry["local_candidates_from_intra_paper"],
            "local_candidates_from_concepts": telemetry["local_candidates_from_concepts"],
            "local_candidates_from_direct_edges": telemetry["local_candidates_from_direct_edges"],
            "local_candidates_from_external_bridges": telemetry["local_candidates_from_external_bridges"],
        }
        # === END DETERMINISTIC GRAPH RETRIEVAL ===

        if not candidates:
            if not getattr(self, "_in_ask", False):
                self._write_graph_retrieval_trace(query, [], None)
            return []
            
        # Capture pre-rerank (RRF) scores for Shannon diagnostics before any CE reordering
        self._last_pre_rerank_scores = [
            float(rrf_scores.get(c.id, 0.0)) for c in candidates
        ]

        returned_chunks = []
        self._last_ranking_trace = {
            "stage_status": "not_started",
            "reranker_enabled": bool(config.rag_components.get("reranker", True)),
            "score_blending_enabled": bool(config.rag_components.get("score_blending", True)),
            "candidate_ids_before_reranker": [str(c.id) for c in candidates],
            "rrf_scores_before_reranker": [float(rrf_scores.get(c.id, 0.0)) for c in candidates],
            "candidates_before_reranker": [
                {
                    "chunk_id": str(c.id), "paper_id": str(c.paper_id),
                    "text_content": c.text_content,
                    "text_sha256": hashlib.sha256((c.text_content or "").encode("utf-8")).hexdigest(),
                    "retrieval_sources": getattr(c, "retrieval_sources", []),
                    "rrf_score": float(rrf_scores.get(c.id, 0.0)),
                }
                for c in candidates
            ],
        }
        if config.rag_components.get("reranker", True):
            try:
                reranker = self._get_reranker()
                pairs = [(query, c.text_content) for c in candidates]
                scores = reranker.predict(pairs)
                
                min_r = min(scores)
                max_r = max(scores)
                range_r = max_r - min_r if max_r > min_r else 1.0
                norm_r = [(s - min_r) / range_r for s in scores]
                
                rrf_vals = [rrf_scores.get(c.id, 0.0) for c in candidates]
                min_rrf = min(rrf_vals)
                max_rrf = max(rrf_vals)
                range_rrf = max_rrf - min_rrf if max_rrf > min_rrf else 1.0
                norm_rrf = [(rrf_scores.get(c.id, 0.0) - min_rrf) / range_rrf for c in candidates]
                
                scored_candidates = []
                for idx, c in enumerate(candidates):
                    if config.rag_components.get("score_blending", True):
                        blended_score = _safe_float(config.score_blend_reranker_weight, 0.7) * norm_r[idx] + _safe_float(config.score_blend_rrf_weight, 0.3) * norm_rrf[idx]
                    else:
                        blended_score = float(scores[idx])
                    scored_candidates.append((c, blended_score, float(scores[idx])))
                    
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                candidate_pairs = [(chunk, raw_score) for chunk, _, raw_score in scored_candidates]
                self._last_ranking_trace.update({
                    "stage_status": "complete",
                    "cross_encoder_scores": [float(value) for value in scores],
                    "normalized_cross_encoder_scores": [float(value) for value in norm_r],
                    "normalized_rrf_scores": [float(value) for value in norm_rrf],
                    "blended_scores_by_candidate": {
                        str(c.id): float(score) for c, score, _ in scored_candidates
                    },
                    "score_blending_branch_executed": True,
                    "ranking_formula": (
                        "normalized_cross_encoder_rrf_blend"
                        if config.rag_components.get("score_blending", True)
                        else "raw_cross_encoder_score"
                    ),
                    "final_candidate_ids": [str(c.id) for c, _, _ in scored_candidates],
                })
                if config.rag_components.get("deduplicate_parent_chunks", True):
                    candidate_pairs = _deduplicate_parent_chunks(candidate_pairs)
                returned_chunks = candidate_pairs[:limit]
            except Exception as e:
                self._last_ranking_trace.update({
                    "stage_status": "error_fallback_to_rrf",
                    "error": str(e),
                    "score_blending_branch_executed": False,
                })
                con.warning(f"Reranking failed ({e}), falling back to RRF ranking.")
                for c in candidates:
                    if c.id not in rrf_scores:
                        rrf_scores[c.id] = 0.0
                sorted_candidates = sorted(candidates, key=lambda x: rrf_scores.get(x.id, 0.0), reverse=True)
                candidate_pairs = [(c, rrf_scores.get(c.id, 0.0)) for c in sorted_candidates]
                if config.rag_components.get("deduplicate_parent_chunks", True):
                    candidate_pairs = _deduplicate_parent_chunks(candidate_pairs)
                returned_chunks = candidate_pairs[:limit]
        else:
            self._last_ranking_trace.update({
                "stage_status": "reranker_disabled",
                "score_blending_branch_executed": False,
            })
            for c in candidates:
                if c.id not in rrf_scores:
                    rrf_scores[c.id] = 0.0
            sorted_candidates = sorted(candidates, key=lambda x: rrf_scores.get(x.id, 0.0), reverse=True)
            candidate_pairs = [(c, rrf_scores.get(c.id, 0.0)) for c in sorted_candidates]
            if config.rag_components.get("deduplicate_parent_chunks", True):
                candidate_pairs = _deduplicate_parent_chunks(candidate_pairs)
            returned_chunks = candidate_pairs[:limit]

        self._last_ranking_trace["returned_chunk_ids"] = [str(c.id) for c, _ in returned_chunks]

        if not getattr(self, "_in_ask", False):
            self._write_graph_retrieval_trace(query, returned_chunks, None)

        if hasattr(self, "current_trace") and self.current_trace is not None:
            self.current_trace["candidate_count_before_reranker"] = len(candidates) if 'candidates' in locals() else 0
            self.current_trace["candidate_count_after_reranker"] = len(returned_chunks)

        return returned_chunks

    def ask(self, query: str, limit: int = 5, history_str: str = "", paper_id: Optional[str] = None, filters: Optional[dict] = None, hyde_responses: Optional[int] = None) -> str:
        self._in_ask = True
        final_chunks = []
        trimmed_chunks = []
        try:
            final_chunks = self.retrieve_relevant_chunks(query, limit, paper_id=paper_id, filters=filters, hyde_responses=hyde_responses)
            if not final_chunks:
                return "No relevant article chunks found in the database. Please index documents first."  # noqa: E501

            # Build initial context
            context_text, context_graph = self.build_context(final_chunks, limit=limit)

            wrapped_query = f"<|query_start|>{query}<|query_end|>"

            # Get system prompt for token counting
            if self.expander and config.rag_components.get("graph_expansion", True):
                system_prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block="", history_str=history_str, query=wrapped_query)
            else:
                system_prompt = prompts.get_prompt("rag", "ask_no_expander", context_text="", context_graph="", history_str=history_str, query=wrapped_query)

            model_max_context = getattr(config, "llm_model_max_context", 4096)

            if config.rag_components.get("context_trimming", True):
                trimmed_text, trimmed_graph, trimmed_chunks = self.trim_context(
                    context_text=context_text,
                    context_graph=context_graph,
                    final_chunks=final_chunks,
                    query=wrapped_query,
                    history_str=history_str,
                    system_prompt=system_prompt,
                    model_max_context=model_max_context,
                    reserved_tokens=500
                )
            else:
                trimmed_text, trimmed_graph, trimmed_chunks = context_text, context_graph, final_chunks

            if hasattr(self, "current_trace") and self.current_trace is not None:
                final_pids = list(set(c[0].paper_id if isinstance(c, tuple) else c.paper_id for c in trimmed_chunks))
                self.current_trace["final_context_paper_id_list"] = final_pids
                tokens = self.llm_engine.count_tokens(trimmed_text + trimmed_graph)
                if not isinstance(tokens, (int, float)):
                    tokens = (len(trimmed_text) + len(trimmed_graph)) // 4
                self.current_trace["final_context_token_count"] = tokens
                graph_neighbors = self.current_trace.get("graph_neighbor_paper_id_list", [])
                self.current_trace["whether_graph_neighbor_chunk_survived_into_final_context"] = any(
                    (c[0].paper_id if isinstance(c, tuple) else c.paper_id) in graph_neighbors for c in trimmed_chunks
                )

            if self.expander and config.rag_components.get("graph_expansion", True):
                if self.expander.reranker is None:
                    self.expander.reranker = self._get_reranker()
                enrichment_block = self.expander.expand(query, trimmed_chunks)
                if not enrichment_block or enrichment_block == "No essential knowledge graph enrichment found.":
                    con.warning("Graph expander found no essential facts. Falling back to raw retrieved context.")
                    prompt = prompts.get_prompt("rag", "ask_no_expander", context_text=trimmed_text, context_graph=trimmed_graph, history_str=history_str, query=wrapped_query)
                else:
                    con.warning("Using graph expansion block.")
                    prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block=enrichment_block, history_str=history_str, query=wrapped_query)
            else:
                prompt = prompts.get_prompt("rag", "ask_no_expander", context_text=trimmed_text, context_graph=trimmed_graph, history_str=history_str, query=wrapped_query)

            con.search_msg("Generating answer …")
            raw_response = self.llm_engine.generate_response(prompt)
            self.last_raw_response = raw_response
            
            status, parsed_answer = parse_reasoning_response(raw_response)
            con.success(f"Reasoning Status: {status}")

            if config.rag_components.get("citation_repair", True):
                try:
                    final_answer = self._validate_and_repair_citations(parsed_answer, trimmed_chunks)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Citation repair failed: {e}")
                    final_answer = parsed_answer
            else:
                final_answer = parsed_answer

            if hasattr(self, "current_trace") and self.current_trace is not None:
                tokens = self.llm_engine.count_tokens(final_answer)
                if not isinstance(tokens, (int, float)):
                    tokens = len(final_answer) // 4
                self.current_trace["answer_token_count"] = tokens

            if config.graph_selected_sources_card_enabled:
                query_concepts = self._extract_query_concepts(query)
                card = self._build_selected_sources_card(trimmed_chunks, query_concepts)
                if card:
                    final_answer = f"{final_answer.strip()}\n\n{card}"

            self._write_graph_retrieval_trace(query, final_chunks, trimmed_chunks)
            return final_answer
        finally:
            self._in_ask = False

    async def generate_stream(self, question: str, limit: int = 5, paper_id: Optional[str] = None, hyde_responses: Optional[int] = None) -> AsyncGenerator[dict, None]:
        import queue
        import threading

        try:
            final_chunks = await asyncio.to_thread(self.retrieve_relevant_chunks, question, limit, paper_id, None, hyde_responses)
        except Exception as e:
            yield {"type": "error", "text": f"Retrieval failed: {e}"}
            return

        if not final_chunks:
            yield {"type": "error", "text": "No documents indexed yet."}
            return

        try:
            wrapped_question = f"<|query_start|>{question}<|query_end|>"
            if self.expander and config.rag_components.get("graph_expansion", True):
                if self.expander.reranker is None:
                    self.expander.reranker = await asyncio.to_thread(self._get_reranker)
                enrichment_block = await asyncio.to_thread(self.expander.expand, question, final_chunks)
                if not enrichment_block or enrichment_block == "No essential knowledge graph enrichment found.":
                    context_text, context_graph = await asyncio.to_thread(self.build_context, final_chunks, limit)
                    prompt = prompts.get_prompt("rag", "stream_no_expander", context_text=context_text, context_graph=context_graph, question=wrapped_question)
                else:
                    prompt = prompts.get_prompt("rag", "stream_expander", enrichment_block=enrichment_block, question=wrapped_question)
            else:
                context_text, context_graph = await asyncio.to_thread(self.build_context, final_chunks, limit)
                prompt = prompts.get_prompt("rag", "stream_no_expander", context_text=context_text, context_graph=context_graph, question=wrapped_question)

        except Exception as e:
            yield {"type": "error", "text": f"Context building failed: {e}"}
            return

        token_queue = queue.Queue()

        def run_mlx_stream():
            try:
                if hasattr(self.llm_engine, "model") and hasattr(self.llm_engine, "tokenizer") and self.llm_engine.model is not None:
                    from mlx_lm import stream_generate
                    gen = stream_generate(
                        model=self.llm_engine.model,
                        tokenizer=self.llm_engine.tokenizer,
                        prompt=prompt,
                        max_tokens=_safe_int(config.llm_max_tokens, 1000),
                    )
                    for response in gen:
                        token_text = response.text if hasattr(response, "text") else str(response)
                        if token_text:
                            token_queue.put(token_text)
                else:
                    full_answer = self.llm_engine.generate_response(prompt)
                    for word in full_answer.split(" "):
                        token_queue.put(word + " ")
            except Exception as e:
                token_queue.put(e)
            finally:
                token_queue.put(None)

        thread = threading.Thread(target=run_mlx_stream)
        thread.start()

        cleaner = StreamTokenCleaner()
        while True:
            item = await asyncio.to_thread(token_queue.get)
            if item is None:
                break
            if isinstance(item, Exception):
                yield {"type": "error", "text": f"Generation failed: {item}"}
                return
            processed_item = cleaner.process_token(item)
            cleaned_item = TECHNICAL_TOKEN_RE.sub("", processed_item)
            if not cleaned_item and item:
                continue
            yield {"type": "token", "text": cleaned_item}

        if config.graph_selected_sources_card_enabled:
            query_concepts = self._extract_query_concepts(question)
            card = self._build_selected_sources_card(final_chunks, query_concepts)
            if card:
                yield {"type": "token", "text": f"\n\n{card}"}

        yield {"type": "done"}


