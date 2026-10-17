"""Shannon Estimator RAG Core Module.

Provides mathematical entropy estimation tools for evaluation of RAG architectures:
- Rank score entropy (softmax, minmax, sum)
- Lexical unigram entropy
- Graph topology entropy (relation type and degree distribution)
- Generation entropy over tokens
- Citation-specific token entropy and span extraction
- Entropy reduction delta measurement
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union


def compute_rank_entropy(
    scores: List[float], method: str = "softmax", tau: float = 1.0
) -> float:
    """Calculates Shannon entropy in bits over normalized candidate rank scores.

    Args:
        scores: List of candidate retrieval or similarity scores.
        method: Normalization method ("softmax", "minmax", "sum"). Default is "softmax".
        tau: Temperature scaling parameter for softmax (tau > 0).

    Returns:
        float: Shannon entropy in bits (log base 2). Returns 0.0 for empty or single item scores.
    """
    if not scores or len(scores) <= 1:
        return 0.0

    if method == "softmax":
        if tau <= 0:
            tau = 1e-6
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / tau) for s in scores]
        sum_exp = sum(exp_scores)
        if sum_exp <= 0:
            return 0.0
        probs = [e / sum_exp for e in exp_scores]
    elif method == "minmax":
        min_s = min(scores)
        max_s = max(scores)
        range_s = max_s - min_s
        if range_s == 0:
            probs = [1.0 / len(scores)] * len(scores)
        else:
            norm = [(s - min_s) / range_s for s in scores]
            sum_norm = sum(norm)
            if sum_norm == 0:
                probs = [1.0 / len(scores)] * len(scores)
            else:
                probs = [n / sum_norm for n in norm]
    elif method in ("sum", "l1", "linear"):
        total = sum(scores)
        if total <= 0:
            return 0.0
        probs = [s / total for s in scores]
    else:
        max_score = max(scores)
        tau_val = tau if tau > 0 else 1.0
        exp_scores = [math.exp((s - max_score) / tau_val) for s in scores]
        sum_exp = sum(exp_scores)
        if sum_exp <= 0:
            return 0.0
        probs = [e / sum_exp for e in exp_scores]

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return max(0.0, float(entropy))


def compute_lexical_entropy(text: str) -> float:
    """Calculates unigram frequency token entropy in bits.

    Args:
        text: Input string.

    Returns:
        float: Shannon entropy of unigram token distribution in bits. Returns 0.0 if empty.
    """
    if not text or not text.strip():
        return 0.0

    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0

    total_tokens = len(tokens)
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1

    entropy = 0.0
    for count in counts.values():
        p = count / total_tokens
        if p > 0:
            entropy -= p * math.log2(p)

    return max(0.0, float(entropy))


def compute_graph_entropy(relations: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates relation_type_entropy and degree_entropy for graph topology in bits.

    Args:
        relations: List of relation dicts representing graph edges.

    Returns:
        Dict with keys "relation_type_entropy" and "degree_entropy".
    """
    if not relations:
        return {"relation_type_entropy": 0.0, "degree_entropy": 0.0}

    type_counts: Dict[str, int] = {}
    node_degrees: Dict[str, int] = {}

    for rel in relations:
        r_type = (
            rel.get("type")
            or rel.get("relation")
            or rel.get("label")
            or rel.get("predicate")
            or "unknown"
        )
        type_counts[str(r_type)] = type_counts.get(str(r_type), 0) + 1

        src = (
            rel.get("source")
            or rel.get("head")
            or rel.get("subject")
            or rel.get("src")
            or rel.get("from")
        )
        dst = (
            rel.get("target")
            or rel.get("tail")
            or rel.get("object")
            or rel.get("dst")
            or rel.get("to")
        )

        if src is not None:
            src_str = str(src)
            node_degrees[src_str] = node_degrees.get(src_str, 0) + 1
        if dst is not None:
            dst_str = str(dst)
            node_degrees[dst_str] = node_degrees.get(dst_str, 0) + 1

    total_edges = len(relations)
    rel_entropy = 0.0
    if total_edges > 0:
        for count in type_counts.values():
            p = count / total_edges
            if p > 0:
                rel_entropy -= p * math.log2(p)

    total_degree = sum(node_degrees.values())
    degree_entropy = 0.0
    if total_degree > 0:
        for d in node_degrees.values():
            p = d / total_degree
            if p > 0:
                degree_entropy -= p * math.log2(p)

    return {
        "relation_type_entropy": max(0.0, float(rel_entropy)),
        "degree_entropy": max(0.0, float(degree_entropy)),
    }


def align_tokens_info(
    full_text: str, clean_text: str, tokens_info: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Aligns token character positions (char_start, char_end) when full_text
    undergoes stripping of thinking tokens or special formatting tags.

    Args:
        full_text: Raw accumulated generated output text.
        clean_text: Final cleaned string after strip_thinking_tokens.
        tokens_info: List of token metric dictionaries.

    Returns:
        List[Dict[str, Any]]: Aligned list of token metric dictionaries matching clean_text.
    """
    if not tokens_info or not full_text:
        return []

    if full_text == clean_text:
        return tokens_info

    # Case 1: clean_text is a contiguous substring of full_text
    offset = full_text.find(clean_text)
    if offset != -1:
        aligned: List[Dict[str, Any]] = []
        clean_len = len(clean_text)
        for t in tokens_info:
            cs = t.get("char_start", t.get("start", 0))
            ce = t.get("char_end", t.get("end", 0))
            if ce > offset and cs < offset + clean_len:
                t_copy = dict(t)
                t_copy["char_start"] = max(0, cs - offset)
                t_copy["char_end"] = min(clean_len, ce - offset)
                aligned.append(t_copy)
        return aligned

    # Case 2: General fallback using sequential matching against clean_text
    aligned = []
    current_pos = 0
    clean_len = len(clean_text)
    for t in tokens_info:
        tok_text = t.get("token_text") or t.get("text") or t.get("token") or ""
        if not tok_text:
            continue
        pos = clean_text.find(tok_text, current_pos)
        if pos != -1:
            t_copy = dict(t)
            t_copy["char_start"] = pos
            t_copy["char_end"] = pos + len(tok_text)
            current_pos = pos + len(tok_text)
            aligned.append(t_copy)
        elif current_pos < clean_len:
            t_copy = dict(t)
            aligned.append(t_copy)

    return aligned if aligned else tokens_info


def find_citation_spans(text: str) -> List[Tuple[int, int]]:
    """Finds character-level [start, end) spans for citation markers.

    Recognized patterns include:
    - [sciq_paper_X]
    - [Block X]
    - [1], [1, 2], [1-3]
    - [paper_1], [doc_2], [ref_1], [id_1], [source_1], [Source_1], [Источник: 1]
    - Paper DOIs: 10.xxxx/... or [10.xxxx/...]
    - Author-year citations: e.g. (Smith et al., 2020) or [Jones, 2019]
    - DOIs and arXiv IDs

    Args:
        text: Input text string.

    Returns:
        List[Tuple[int, int]]: Sorted list of non-overlapping [start, end) character spans.
    """
    if not text:
        return []

    patterns = [
        # [sciq_paper_X]
        r"\[sciq_paper_[^\]]+\]",
        # [Block X] or [Block X, Block Y]
        r"\[Block\s+[^\]]+\]",
        # [1], [1, 2], [1-3]
        r"\[\d+(?:[\s,–-]+\d+)*\]",
        # [paper_1], [doc_2], [ref_1], [id_1], [source_1], [Source_1], [Источник: 1]
        r"\[(?:paper|doc|ref|id|source|Source|Источник)[_\s:]*[^\]]+\]",
        # Paper DOIs in brackets
        r"\[10\.\d{4,9}/[^\]]+\]",
        # Standalone DOIs
        r"\b10\.\d{4,9}/[^\s,;()\]]+",
        # Bracketed author-year: [Jones, 2019] or [Smith et al., 2020]
        r"\[[A-Z][a-zA-Z\s.-]+(?:et\s+al\.)?,\s*\d{4}[a-z]?\]",
        # Parenthetical author-year: (Smith et al., 2020)
        r"\([A-Z][a-zA-Z\s.-]+(?:et\s+al\.)?,\s*\d{4}[a-z]?\)",
        # arXiv IDs
        r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\b",
    ]

    combined_pattern = "|".join(f"(?:{p})" for p in patterns)
    raw_spans = []

    for m in re.finditer(combined_pattern, text):
        raw_spans.append((m.start(), m.end()))

    if not raw_spans:
        return []

    # Merge overlapping or touching citation spans
    raw_spans.sort(key=lambda x: (x[0], x[1]))
    merged_spans: List[Tuple[int, int]] = []
    current_start, current_end = raw_spans[0]

    for next_start, next_end in raw_spans[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged_spans.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_spans.append((current_start, current_end))

    return merged_spans


def _extract_single_token_entropy(t_info: Dict[str, Any]) -> float:
    """Helper function to calculate entropy in bits for a single token dictionary."""
    if "entropy" in t_info and t_info["entropy"] is not None:
        return max(0.0, float(t_info["entropy"]))

    if "probs" in t_info or "top_probs" in t_info:
        probs = t_info.get("probs") or t_info.get("top_probs")
        if isinstance(probs, dict):
            probs_list = list(probs.values())
        elif isinstance(probs, list):
            probs_list = probs
        else:
            probs_list = []
        if probs_list:
            entropy = 0.0
            for p in probs_list:
                if p > 0:
                    entropy -= p * math.log2(p)
            return max(0.0, float(entropy))

    if "top_logprobs" in t_info or "logprobs" in t_info:
        lp_data = t_info.get("top_logprobs") or t_info.get("logprobs")
        if isinstance(lp_data, dict):
            lps = list(lp_data.values())
        elif isinstance(lp_data, list):
            lps = [x.get("logprob", 0.0) if isinstance(x, dict) else x for x in lp_data]
        else:
            lps = []
        if lps:
            probs = [math.exp(lp) for lp in lps]
            sum_p = sum(probs)
            if sum_p > 0:
                probs = [p / sum_p for p in probs]
                entropy = 0.0
                for p in probs:
                    if p > 0:
                        entropy -= p * math.log2(p)
                return max(0.0, float(entropy))

    if "logprob" in t_info and t_info["logprob"] is not None:
        lp = float(t_info["logprob"])
        # Handle both true negative logprob (ln p <= 0) and positive surprisal (-ln p > 0)
        p = math.exp(-lp) if lp > 0 else math.exp(lp)
        if 0 < p <= 1:
            return float(-math.log2(p))

    if "prob" in t_info and t_info["prob"] is not None:
        p = t_info["prob"]
        if 0 < p <= 1:
            return float(-math.log2(p))

    return 0.0


def compute_generation_entropy(tokens_info: List[Dict[str, Any]]) -> float:
    """Calculates average per-token Shannon entropy in bits across generation tokens.

    Args:
        tokens_info: List of token metadata dictionaries.

    Returns:
        float: Average token entropy in bits. Returns 0.0 if list is empty.
    """
    if not tokens_info:
        return 0.0

    total_entropy = sum(_extract_single_token_entropy(t) for t in tokens_info)
    return max(0.0, float(total_entropy / len(tokens_info)))


def compute_citation_entropy(
    tokens_info: List[Dict[str, Any]], generated_text: str
) -> Tuple[float, int]:
    """Maps character citation spans back to token character position ranges,
    and calculates average entropy specifically for citation tokens.

    Args:
        tokens_info: List of token metric dicts.
        generated_text: The complete generated text string.

    Returns:
        Tuple[float, int]: (average citation entropy in bits, number of citation tokens).
    """
    if not tokens_info or not generated_text:
        return (0.0, 0)

    spans = find_citation_spans(generated_text)
    if not spans:
        return (0.0, 0)

    citation_tokens: List[Dict[str, Any]] = []

    has_explicit_range = any(
        ("char_start" in t or "start" in t) and ("char_end" in t or "end" in t)
        for t in tokens_info
    )

    if has_explicit_range:
        for t in tokens_info:
            c_start = t.get("char_start", t.get("start"))
            c_end = t.get("char_end", t.get("end"))
            if c_start is not None and c_end is not None:
                if c_start == c_end:
                    if any(s_start <= c_start < s_end for s_start, s_end in spans):
                        citation_tokens.append(t)
                else:
                    if any(max(c_start, s_start) < min(c_end, s_end) for s_start, s_end in spans):
                        citation_tokens.append(t)

    if not citation_tokens:
        current_idx = 0
        for t in tokens_info:
            tok_text = t.get("token_text") or t.get("text") or t.get("token") or t.get("token_str") or ""
            if not tok_text:
                continue
            pos = generated_text.find(tok_text, current_idx)
            if pos != -1:
                t_start = pos
                t_end = pos + len(tok_text)
                current_idx = t_end
                if any(max(t_start, s_start) < min(t_end, s_end) for s_start, s_end in spans):
                    citation_tokens.append(t)

    if not citation_tokens:
        return (0.0, 0)

    avg_entropy = compute_generation_entropy(citation_tokens)
    return (avg_entropy, len(citation_tokens))


def compute_entropy_reduction(
    h_b0: Optional[float], h_rag: Optional[float]
) -> float:
    """Calculates uncertainty reduction ΔH_gen = h_b0 - h_rag.

    Args:
        h_b0: Baseline model generation entropy.
        h_rag: RAG model generation entropy.

    Returns:
        float: Uncertainty reduction in bits. Returns 0.0 if either value is None.
    """
    if h_b0 is None or h_rag is None:
        return 0.0
    return float(h_b0 - h_rag)


def build_token_char_spans(tokens_info: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Computes exact character start and end offsets (start_char, end_char) for each token index in tokens_info."""
    spans: List[Tuple[int, int]] = []
    curr_offset = 0
    for t in tokens_info:
        c_start = t.get("char_start", t.get("start"))
        c_end = t.get("char_end", t.get("end"))
        if c_start is not None and c_end is not None:
            start_val = int(c_start)
            end_val = int(c_end)
            spans.append((start_val, end_val))
            curr_offset = end_val
        else:
            tok_text = t.get("token") or t.get("token_text") or t.get("text") or t.get("token_str") or ""
            start_val = curr_offset
            end_val = curr_offset + len(str(tok_text))
            spans.append((start_val, end_val))
            curr_offset = end_val
            t["char_start"] = start_val
            t["char_end"] = end_val
    return spans


def map_char_offset_to_token_idx(char_offset: int, token_spans: List[Tuple[int, int]]) -> int:
    """Maps any string character index back to its corresponding token index in token_spans."""
    if not token_spans:
        return 0
    if char_offset <= 0:
        return 0
    last_idx = len(token_spans) - 1
    if char_offset >= token_spans[last_idx][1]:
        return last_idx

    for idx, (s, e) in enumerate(token_spans):
        if s <= char_offset < e:
            return idx
        if char_offset < s:
            return max(0, idx - 1)
    return last_idx


def compute_softmax(
    logits: Union[List[float], Dict[Any, float], Any]
) -> List[float]:
    """Computes numerically stable softmax probabilities from raw logits.

    Formula: p_i = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
    Ensures probabilities satisfy sum(p_i) == 1.0 +- 1e-6.

    Args:
        logits: List, Dict, or array-like of raw logit scores.

    Returns:
        List[float]: Normalized probability distribution.
    """
    if logits is None:
        return []
    if isinstance(logits, dict):
        vals = list(logits.values())
    elif hasattr(logits, "tolist"):
        vals = logits.tolist()
    else:
        try:
            vals = list(logits)
        except TypeError:
            return []

    if not vals:
        return []

    float_vals = [float(x) for x in vals]
    max_z = max(float_vals)
    exp_z = [math.exp(z - max_z) for z in float_vals]
    sum_exp = sum(exp_z)

    if sum_exp <= 0 or math.isnan(sum_exp) or math.isinf(sum_exp):
        n = len(float_vals)
        return [1.0 / n] * n if n > 0 else []

    probs = [ez / sum_exp for ez in exp_z]
    total_p = sum(probs)
    if abs(total_p - 1.0) > 1e-12 and total_p > 0:
        probs = [p / total_p for p in probs]

    return probs


def compute_msp(
    logits_or_probs: Union[List[float], Dict[Any, float], Any]
) -> float:
    """Calculates Maximum Softmax Probability (MSP = max(p)).

    Accepts raw logits or probability distributions as List, Dict, or numpy array.

    Args:
        logits_or_probs: Raw logits or probability distribution.

    Returns:
        float: Maximum softmax probability value in range [0.0, 1.0].
    """
    if logits_or_probs is None:
        return 0.0
    if isinstance(logits_or_probs, dict):
        vals = list(logits_or_probs.values())
    elif hasattr(logits_or_probs, "tolist"):
        vals = logits_or_probs.tolist()
    else:
        try:
            vals = list(logits_or_probs)
        except TypeError:
            return 0.0

    if not vals:
        return 0.0

    float_vals = [float(x) for x in vals]
    total = sum(float_vals)
    if all(x >= 0 for x in float_vals) and math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        probs = float_vals
    else:
        probs = compute_softmax(float_vals)

    return float(max(probs)) if probs else 0.0


def compute_logit_margin(
    logits: Union[List[float], Dict[Any, float], Any]
) -> float:
    """Calculates top-1 vs top-2 logit margin Delta z_{1,2} = z_1 - z_2.

    Where z_1, z_2 are top-1 and top-2 raw logits (or logprobs) sorted in descending order.

    Args:
        logits: Raw logits or logprobs as List, Dict, or numpy array.

    Returns:
        float: Difference z_1 - z_2. Returns 0.0 if fewer than 2 logits are provided.
    """
    if logits is None:
        return 0.0
    if isinstance(logits, dict):
        vals = list(logits.values())
    elif hasattr(logits, "tolist"):
        vals = logits.tolist()
    else:
        try:
            vals = list(logits)
        except TypeError:
            return 0.0

    if len(vals) < 2:
        return 0.0

    float_vals = sorted([float(x) for x in vals], reverse=True)
    return float(float_vals[0] - float_vals[1])


def _extract_token_logits_or_probs(t_info: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """Extracts (logits, probs) data from a single token metadata dictionary."""
    logits = t_info.get("logits") if t_info.get("logits") is not None else t_info.get("top_logits")
    probs = t_info.get("probs") if t_info.get("probs") is not None else t_info.get("top_probs")
    if logits is None and (t_info.get("top_logprobs") is not None or t_info.get("logprobs") is not None):
        lp_data = t_info.get("top_logprobs") if t_info.get("top_logprobs") is not None else t_info.get("logprobs")
        if isinstance(lp_data, dict):
            logits = list(lp_data.values())
        elif isinstance(lp_data, list):
            logits = [x.get("logprob", 0.0) if isinstance(x, dict) else x for x in lp_data]
    return logits, probs


def compute_first_token_metrics(tokens_info: List[Dict[str, Any]]) -> Dict[str, float]:
    """Isolates index 0 of generated response sequence and returns first_token_margin and first_token_msp."""
    if not tokens_info:
        return {"first_token_margin": 0.0, "first_token_msp": 0.0}

    t0 = tokens_info[0]
    if isinstance(t0, dict):
        if "first_token_margin" in t0 and "first_token_msp" in t0:
            return {
                "first_token_margin": float(t0["first_token_margin"]),
                "first_token_msp": float(t0["first_token_msp"]),
            }
        # Prefer compact per-token fields written by MLX/OpenAI engines
        if t0.get("msp") is not None or t0.get("logit_margin") is not None:
            return {
                "first_token_margin": round(float(t0.get("logit_margin", 0.0) or 0.0), 4),
                "first_token_msp": round(float(t0.get("msp", t0.get("prob", 0.0)) or 0.0), 4),
            }
        logits, probs = _extract_token_logits_or_probs(t0)
        data = logits if logits is not None else probs
        margin = compute_logit_margin(data)
        msp = compute_msp(data)
        return {
            "first_token_margin": round(margin, 4),
            "first_token_msp": round(msp, 4),
        }
    return {"first_token_margin": 0.0, "first_token_msp": 0.0}


def compute_sequence_telemetry(tokens_info: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates average sequence MSP and average logit margin across all tokens, plus first-token metrics."""
    first_metrics = compute_first_token_metrics(tokens_info)
    if not tokens_info:
        return {
            "avg_msp": 0.0,
            "avg_logit_margin": 0.0,
            **first_metrics,
        }

    msps: List[float] = []
    margins: List[float] = []
    for t in tokens_info:
        if isinstance(t, dict):
            logits, probs = _extract_token_logits_or_probs(t)
            data = logits if logits is not None else probs
            if data is not None:
                margins.append(compute_logit_margin(data))
                msps.append(compute_msp(data))
            elif "msp" in t or "prob" in t:
                m_val = float(t.get("logit_margin", 0.0))
                p_val = float(t.get("msp", t.get("prob", 0.0)))
                margins.append(m_val)
                msps.append(p_val)

    avg_msp = sum(msps) / len(msps) if msps else 0.0
    avg_margin = sum(margins) / len(margins) if margins else 0.0

    return {
        "avg_msp": round(avg_msp, 4),
        "avg_logit_margin": round(avg_margin, 4),
        **first_metrics,
    }


def compute_citation_onset_entropy(
    tokens_info: List[Dict[str, Any]], generated_text: str
) -> Tuple[float, int]:
    """Uses regex r"\\[|Doc" to find citation start positions in generated response text,
    maps each match's character index to token index t_c via map_char_offset_to_token_idx,
    and computes Shannon Entropy H_{citation} = -sum p_i log_2 p_i at step t_c for the
    token's vocabulary distribution. Averages across multiple citation onsets if present.

    Args:
        tokens_info: List of token metric dicts.
        generated_text: The complete generated text string.

    Returns:
        Tuple[float, int]: (average citation onset entropy in bits, number of citation onset matches).
    """
    if not tokens_info or not generated_text:
        return (0.0, 0)

    matches = list(re.finditer(r"\[|Doc", generated_text))
    if not matches:
        return (0.0, 0)

    spans = build_token_char_spans(tokens_info)
    entropies: List[float] = []

    for match in matches:
        char_idx = match.start()
        t_c = map_char_offset_to_token_idx(char_idx, spans)
        if 0 <= t_c < len(tokens_info):
            ent = _extract_single_token_entropy(tokens_info[t_c])
            entropies.append(ent)

    if not entropies:
        return (0.0, 0)

    avg_entropy = sum(entropies) / len(entropies)
    return (max(0.0, float(avg_entropy)), len(entropies))




def compute_log_likelihood(tokens_info: List[Dict[str, Any]]) -> float:
    """Calculates total sequence log-likelihood sum_i log P(w_i | w_<i, context)."""
    if not tokens_info:
        return 0.0
    total_ll = 0.0
    for t in tokens_info:
        if isinstance(t, dict):
            if "logprob" in t and t["logprob"] is not None:
                total_ll += float(t["logprob"])
            elif "prob" in t and t["prob"] is not None:
                p = float(t["prob"])
                if p > 0:
                    total_ll += math.log(p)
    return float(total_ll)


def compute_clr(ll_rag: float, ll_base: float) -> float:
    """Calculates Contextual Log-Likelihood Ratio CLR = LL_rag - LL_base."""
    return float(ll_rag - ll_base)



_GRAPH_DISABLED_MARKERS = frozenset({
    "",
    "No direct graph relations found.",
    "Graph enrichment disabled.",
})

# Matches real Cypher-style lines produced by rag_service._get_scored_graph_lines, e.g.:
# - ('Long title with spaces':Paper)-[CITES {preview: "..."}]->(work:doi:10.1:ExternalWork)
# - (Author Name:Author)-[AUTHORED]->(Paper Title:Paper)
_GRAPH_LINE_RE = re.compile(
    r"\("
    r"(?:'([^']*)'|\"([^\"]*)\"|([^):\]]+))"  # src name (quoted or bare)
    r":"
    r"([^)]+)"  # src label
    r"\)-"
    r"\["
    r"(\w+)"  # relation type
    r"[^\]]*"
    r"\]->"
    r"\("
    r"(?:'([^']*)'|\"([^\"]*)\"|([^):\]]+))"  # tgt name
    r":"
    r"([^)]+)"  # tgt label
    r"\)"
)


def parse_graph_relations_from_text(graph_text: str) -> List[Dict[str, Any]]:
    """Parse formatted graph context lines into relation dicts for compute_graph_entropy.

    Accepts the Cypher-like lines emitted by RAGService._get_scored_graph_lines.
    Returns an empty list for missing/disabled graph text.
    """
    if not graph_text or graph_text.strip() in _GRAPH_DISABLED_MARKERS:
        return []

    relations: List[Dict[str, Any]] = []
    for match in _GRAPH_LINE_RE.finditer(graph_text):
        src_name = match.group(1) or match.group(2) or (match.group(3) or "").strip()
        src_label = (match.group(4) or "").strip()
        rel_type = match.group(5)
        tgt_name = match.group(6) or match.group(7) or (match.group(8) or "").strip()
        tgt_label = (match.group(9) or "").strip()
        source = f"{src_name}:{src_label}" if src_label else src_name
        target = f"{tgt_name}:{tgt_label}" if tgt_label else tgt_name
        if not source or not target or not rel_type:
            continue
        relations.append({
            "source": source,
            "target": target,
            "type": rel_type,
            "source_label": src_label,
            "target_label": tgt_label,
        })
    return relations


def _coerce_score_list(scores: Optional[List[Any]]) -> List[float]:
    if not scores:
        return []
    out: List[float] = []
    for s in scores:
        try:
            out.append(float(s))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def assemble_retrieval_shannon_fields(
    *,
    pre_scores: Optional[List[Any]] = None,
    post_scores: Optional[List[Any]] = None,
    pre_text: Optional[str] = None,
    post_text: Optional[str] = None,
    relations: Optional[List[Dict[str, Any]]] = None,
    graph_text: Optional[str] = None,
) -> Dict[str, float]:
    """Assemble rank / lexical / graph Shannon fields from staged inputs.

    Semantics:
    - When distinct pre_scores / pre_text are provided, pre metrics use them.
    - When pre is missing, post is used for both pre and post (honest fallback when
      the stage was off or pre boundary was not captured).
    - Graph entropy prefers structured ``relations``; falls back to parsing
      ``graph_text`` with parse_graph_relations_from_text.
    """
    post_score_list = _coerce_score_list(post_scores)
    pre_score_list = _coerce_score_list(pre_scores)
    if not pre_score_list:
        pre_score_list = list(post_score_list)

    post_text_val = post_text if post_text is not None else ""
    pre_text_val = pre_text if pre_text is not None else post_text_val

    rels: List[Dict[str, Any]] = list(relations) if relations else []
    if not rels and graph_text:
        rels = parse_graph_relations_from_text(graph_text)

    graph_ent = compute_graph_entropy(rels)

    return {
        "h_rank_pre_rerank": round(compute_rank_entropy(pre_score_list), 4),
        "h_rank_post_rerank": round(compute_rank_entropy(post_score_list), 4),
        "h_lexical_pre_trim": round(compute_lexical_entropy(pre_text_val), 4),
        "h_lexical_post_trim": round(compute_lexical_entropy(post_text_val), 4),
        "h_graph_relation_type": round(graph_ent["relation_type_entropy"], 4),
        "h_graph_degree": round(graph_ent["degree_entropy"], 4),
    }


def empty_retrieval_shannon_fields() -> Dict[str, float]:
    """Zero retrieval-stage Shannon fields (B0 / empty retrieval)."""
    return {
        "h_rank_pre_rerank": 0.0,
        "h_rank_post_rerank": 0.0,
        "h_lexical_pre_trim": 0.0,
        "h_lexical_post_trim": 0.0,
        "h_graph_relation_type": 0.0,
        "h_graph_degree": 0.0,
    }

