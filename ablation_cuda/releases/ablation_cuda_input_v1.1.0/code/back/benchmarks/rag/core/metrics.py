import re
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

try:
    import tiktoken
except ImportError:
    tiktoken = None

from core.shannon_estimator import (
    compute_softmax,
    compute_msp,
    compute_logit_margin,
    compute_clr,
    compute_first_token_metrics,
    compute_sequence_telemetry,
    compute_citation_onset_entropy,
)



_embedding_engine = None


def normalize_id(doc_or_chunk_id: object) -> str:
    """Normalizes document or chunk IDs by stripping path prefixes, standardizing separators,
    enforcing lowercase, stripping file extensions, and truncating chunk-level markers to document level.
    """
    if doc_or_chunk_id is None:
        return ""
    s = str(doc_or_chunk_id).strip()
    if not s:
        return ""

    s_lower = s.lower()

    # 1. Strip path prefixes (e.g. "docs/", "data/", "papers/", "corpus/", "dataset/", "files/", "./", "../", absolute paths)
    s_clean = re.sub(r'^(?:.*[/\\])?(?:docs|data|papers|corpus|dataset|files)[/\\]', '', s_lower)
    
    # If still contains path separators and is not a DOI (e.g. starting with '10.')
    if ('/' in s_clean or '\\' in s_clean) and not s_clean.startswith("10."):
        s_clean = s_clean.replace("\\", "/").split("/")[-1]

    # Strip optional paper_ prefix if present to standardize document ID
    s_clean = re.sub(r'^paper_', '', s_clean)

    # Strip standard file extensions (.pdf, .txt, .md, .json, .html)
    s_clean = re.sub(r'\.(pdf|txt|md|json|html)(?=[#_/\\]|$)', '', s_clean)

    # 2. Standardize separators: replace '-', '#', ':', whitespace, '/' with '_'
    s_clean = s_clean.replace("-", "_").replace("#", "_").replace(":", "_").replace("/", "_").replace(" ", "_")

    # 3. Truncation logic: truncate chunk-level markers (_chunk_3, _chunk3, etc.) to document-level
    s_clean = re.sub(r'_chunk.*$', '', s_clean)
    
    # Clean up residual trailing/leading underscores or duplicate underscores
    s_clean = re.sub(r'_+', '_', s_clean).strip("_")

    return s_clean



def normalize_optional_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_is_answerable(case: dict) -> bool:
    is_ans = case.get("is_answerable")
    if is_ans is None:
        return True
    if isinstance(is_ans, str):
        return is_ans.lower() == "true"
    return bool(is_ans)


def detect_abstention(generated_answer: str, final_answer: str | None = None) -> bool:
    ABSTENTION_MARKERS = [
        "unanswerable",
        "cannot answer",
        "can't answer",
        "not enough information",
        "not enough info",
        "insufficient information",
        "not provided in the context",
        "not contained in the context",
        "does not contain",
        "do not contain",
        "no information",
        "missing information",
        "information missing",
        "information not available",
        "no data available",
        "нет информации",
        "недостаточно информации",
        "невозможно ответить",
        "нельзя ответить",
    ]
    gen_lower = generated_answer.lower() if generated_answer else ""
    final_lower = final_answer.lower() if final_answer else ""
    
    try:
        from core.sanitization import extract_clean_answer
        status, extracted_final = extract_clean_answer(generated_answer)
        if status == "UNANSWERABLE":
            return True
        if not final_answer:
            final_lower = extracted_final.lower()
    except Exception:
        pass

    # Clean up non-refusal contexts and quotes to avoid false positives
    for text_to_clean in ["previously unanswerable", "not unanswerable", "'unanswerable'", '"unanswerable"', '“unanswerable”']:
        gen_lower = gen_lower.replace(text_to_clean, "")
        final_lower = final_lower.replace(text_to_clean, "")

    for marker in ABSTENTION_MARKERS:
        if marker in gen_lower:
            return True
        if final_lower and marker in final_lower:
            return True

    if not gen_lower.strip():
        return True
        
    return False


def classify_answerability(is_answerable: bool, predicted_abstained: bool) -> str:
    if is_answerable and not predicted_abstained:
        return "TP"
    if is_answerable and predicted_abstained:
        return "FN"
    if not is_answerable and predicted_abstained:
        return "TN"
    return "FP"



def get_embedding_engine():
    """Lazily loads and returns the EmbeddingEngine from the parent src codebase."""
    global _embedding_engine
    if _embedding_engine is None:
        try:
            # Adjust path to resolve src imports
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from src.vector_search import EmbeddingEngine
            _embedding_engine = EmbeddingEngine()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not load EmbeddingEngine: {e}")
            raise e
    return _embedding_engine


def calculate_retrieval_recall(expected_papers: List[str], retrieved_papers: List[str]) -> float:
    """Computes retrieval recall: proportion of expected papers that were successfully retrieved."""
    if not expected_papers:
        return 1.0
    expected_set = {normalize_id(p) for p in expected_papers if p and normalize_id(p)}
    retrieved_set = {normalize_id(p) for p in retrieved_papers if p and normalize_id(p)}
    if not expected_set:
        return 1.0
    intersection = expected_set.intersection(retrieved_set)
    return round(len(intersection) / len(expected_set), 4)


def calculate_context_precision(expected_papers: List[str], retrieved_chunks: List[Dict[str, Any]]) -> float:
    """Computes context precision (Mean Average Precision on paper retrieval relevance at K)."""
    if not expected_papers:
        return 1.0
    expected_set = {normalize_id(p) for p in expected_papers if p and normalize_id(p)}
    if not expected_set:
        return 1.0
    if not retrieved_chunks:
        return 0.0

    precision_sum = 0.0
    relevant_hits = 0
    for idx, chunk in enumerate(retrieved_chunks):
        if isinstance(chunk, dict):
            raw_id = chunk.get("paper_id") or chunk.get("id") or ""
        else:
            raw_id = getattr(chunk, "paper_id", None) or getattr(chunk, "id", None) or ""
            
        paper_id = normalize_id(raw_id)
        if paper_id and paper_id in expected_set:
            relevant_hits += 1
            precision_sum += relevant_hits / (idx + 1)
            
    if relevant_hits == 0:
        return 0.0
    return round(precision_sum / relevant_hits, 4)


def compute_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes the cosine similarity between two vector embeddings."""
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    norm1 = np.linalg.norm(v1_arr)
    norm2 = np.linalg.norm(v2_arr)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1_arr, v2_arr) / (norm1 * norm2))


def calculate_semantic_accuracy(golden_answers: List[str], generated_answers: List[str]) -> List[float]:
    """Generates embeddings for lists of golden and generated answers and calculates cosine similarity."""
    if not golden_answers or not generated_answers:
        return []
    
    engine = get_embedding_engine()
    gold_embs = engine.get_embeddings(golden_answers, is_query=False)
    gen_embs = engine.get_embeddings(generated_answers, is_query=False)
    
    similarities = []
    for v1, v2 in zip(gold_embs, gen_embs):
        similarities.append(compute_cosine_similarity(v1, v2))
    return similarities


def count_text_tokens(text: str) -> int:
    """Counts token count in text using tiktoken or simple character heuristic."""
    if not text:
        return 0
    if tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    # Simple heuristic fallback (approx. 4 characters per token for English/Russian mixed)
    return max(1, len(text) // 4)


def estimate_prompt_tokens(query: str, retrieved_chunks: List[Dict[str, Any]], baseline: str) -> int:
    """Estimates prompt token count using tiktoken or simple character heuristic."""
    if baseline == "B0":
        prompt = f"Question: {query}\nAnswer based on your general knowledge."
    else:
        system_prompt = (
            "<|im_start|>system\n"
            "You are a research assistant. Synthesize an answer to the user's question using the retrieved text blocks and the knowledge graph connections.\n"
            "Always mention the titles of the papers, years, authors, and page numbers when citation is needed.\n"
            "If the graph contains citing relationships, use them to explain the context (e.g., \"A cited B\").\n\n"
            "Here is the retrieved context:\n\n"
            "### RELEVANT TEXT FRAGMENTS:\n"
        )
        text_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            text_content = chunk.get("text_content", "").strip()
            paper_id = chunk.get("paper_id", "")
            page = chunk.get("page_number", "")
            text_blocks.append(
                f"Block {idx} (Score: 1.000) | Paper: {paper_id} (Page {page}):\n"
                f"\"\"\"\n{text_content}\n\"\"\""
            )
        context_text = "\n\n".join(text_blocks)
        context_graph = "No direct graph relations found."
        prompt = (
            f"{system_prompt}{context_text}\n\n"
            f"### KNOWLEDGE GRAPH CONNECTIONS:\n{context_graph}\n"
            f"<|im_end|>\n<|im_start|>user\nQuestion: {query}\nAnswer:\n<|im_end|>\n<|im_start|>assistant\n"
        )
    
    return count_text_tokens(prompt)
