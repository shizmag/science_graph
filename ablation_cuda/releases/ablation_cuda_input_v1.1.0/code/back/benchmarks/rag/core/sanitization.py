import re
from typing import Tuple

def clean_answer_tokens(text: str) -> str:
    if not text:
        return ""
    
    # 1. Strip think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    
    # 2. Mask source_id
    text = re.sub(r"<\|source_id\|>", "__SOURCE_ID_TAG__", text, flags=re.IGNORECASE)
    
    # 3. Model-agnostic generic token stripping
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"<<.*?>>", "", text)
    text = re.sub(r"\[/?(?:[A-Z_]{2,}[A-Z0-9_-]*)\]", "", text)
    text = re.sub(r"</?(?:s|pad|unk|turn)>", "", text, flags=re.IGNORECASE)
    
    # 4. Strip common technical patterns
    technical_patterns = [
        r"<\|im_start\|>", r"<\|im_end\|>", r"<\|im_sep\|>",
        r"<\|start_header_id\|>", r"<\|end_header_id\|>",
        r"<\|eot_id\|>", r"<\|eom_id\|>", r"<\|endoftext\|>",
        r"<\|assistant\|>", r"<\|user\|>", r"<\|system\|>",
        r"<\|end\|>", r"\[INST\]", r"\[/INST\]",
        r"<s>", r"</s>", r"<start_of_turn>", r"<end_of_turn>",
        r"<<SYS>>", r"<</SYS>>", r"<pad>", r"<unk>", r"<turn>"
    ]
    for pattern in technical_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
    # 5. Restore source_id
    text = text.replace("__SOURCE_ID_TAG__", "<|source_id|>")
    return text.strip()

def _fallback_parse_reasoning_response(raw_response: str) -> Tuple[str, str]:
    if not raw_response or not isinstance(raw_response, str):
        return "UNKNOWN", ""
    
    # 1. Extract status
    status = "UNKNOWN"
    status_match = re.search(r"<\|status_start\|>(.*?)<\|status_end\|>", raw_response, re.DOTALL)
    if status_match:
        status = status_match.group(1).strip()
    else:
        status_unclosed = re.search(r"<\|status_start\|>(.*)", raw_response, re.DOTALL)
        if status_unclosed:
            content = status_unclosed.group(1).split("<|")[0].strip()
            status = content if content else "UNKNOWN"
            
    if status == "UNKNOWN":
        status_sec_match = re.search(
            r"(?:###\s*)?4\.\s*_(?:status)(?:\.\.\.)?[_a-zA-Z0-9:]*\s*(.*?)(?=(?:###\s*)?(?:5\.\s*_(?:answer)(?:\.\.\.)?[_a-zA-Z0-9:]*|(?:###\s*)?Final\s+Answer\s*:?|$))",
            raw_response,
            re.IGNORECASE | re.DOTALL
        )
        if status_sec_match:
            status_text = status_sec_match.group(1).strip().upper()
            if any(x in status_text for x in ["UNANSWERABLE", "NOT ANSWERABLE", "INSUFFICIENT", "NOT_ANSWERABLE"]):
                status = "UNANSWERABLE"
            elif any(x in status_text for x in ["ANSWERABLE", "SUFFICIENT"]):
                status = "ANSWERABLE"
                
    # 2. Extract answer
    answer = ""
    # Try XML format first
    answer_match = re.search(r"<\|answer_start\|>(.*?)<\|answer_end\|>", raw_response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer_unclosed = re.search(r"<\|answer_start\|>(.*)", raw_response, re.DOTALL)
        if answer_unclosed:
            answer = answer_unclosed.group(1).strip()
        else:
            # Try Markdown format
            # Find the last answer marker
            answer_markers = [
                r"(?:###\s*)?Final\s+Answer\s*:?\s*",
                r"(?:###\s*)?5\.\s*_(?:answer)(?:\.\.\.)?[_a-zA-Z0-9:]*\s*",
            ]
            combined_pattern = re.compile(
                r"|".join(f"(?:{p})" for p in answer_markers),
                re.IGNORECASE
            )
            
            matches = list(combined_pattern.finditer(raw_response))
            if matches:
                last_match = matches[-1]
                answer = raw_response[last_match.end():].strip()
            else:
                # If no final answer section or 5. _answer section matches, we clean the full text by removing sections 1 to 4
                answer = raw_response.strip()
                answer = re.sub(
                    r"(?:###\s*)?[1-4]\.\s*_(?:analysis|start|reasoning|status|source_analysis)(?:\.\.\.)?.*?(?=(?:###\s*)?(?:5\.\s*_(?:answer|status|reasoning|analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*|(?:###\s*)?Final\s+Answer\s*:?|$))",
                    "",
                    answer,
                    flags=re.IGNORECASE | re.DOTALL
                )
                # Clean headers
                header_pattern = r"(?:###\s*)?[1-5]\.\s*_(?:analysis|start|reasoning|status|answer|source_analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*"
                answer = re.sub(header_pattern, "", answer, flags=re.IGNORECASE)
                answer = re.sub(r"(?:###\s*)?Final\s+Answer\s*:?", "", answer, flags=re.IGNORECASE)
                
            # Strip other technical tags that are outside the answer section but could remain
            for tag in ["status", "query_analysis", "source_analysis", "reasoning"]:
                answer = re.sub(rf"<\|{tag}_start\|>.*?<\|{tag}_end\|>", "", answer, flags=re.DOTALL)
                answer = re.sub(rf"<\|{tag}_start\|>.*", "", answer, flags=re.DOTALL)

    return status, answer

def extract_clean_answer(raw_response: str) -> Tuple[str, str]:
    """
    Extracts status and clean answer from a raw LLM response.
    First tries to use the primary parser from rag_service, falls back on failure.
    """
    if not raw_response or not isinstance(raw_response, str):
        return "UNKNOWN", ""

    normalized_raw = raw_response.lower()
    
    # 1. Check if the response contains markdown reasoning headers but no final answer header
    has_reasoning_headers = any(h in normalized_raw for h in ["1. _analysis", "2. _start", "3. _reasoning", "4. _status"])
    has_final_answer_header = "5. _answer" in normalized_raw or "final answer" in normalized_raw
    
    # 2. Check if the response contains XML/technical reasoning tags but no answer tag
    has_xml_reasoning = any(tag in normalized_raw for tag in ["<|reasoning_start|>", "<|query_analysis_start|>", "<|source_analysis_start|>", "<|status_start|>"])
    has_xml_answer = "<|answer_start|>" in normalized_raw

    if (has_reasoning_headers and not has_final_answer_header) or (has_xml_reasoning and not has_xml_answer):
        status = "UNKNOWN"
        try:
            from src.services.rag_service import parse_reasoning_response
            status, _ = parse_reasoning_response(raw_response)
        except Exception:
            status, _ = _fallback_parse_reasoning_response(raw_response)
        return status, ""

    status = "UNKNOWN"
    answer = ""
    
    try:
        from src.services.rag_service import parse_reasoning_response
        status, answer = parse_reasoning_response(raw_response)
    except Exception:
        status, answer = _fallback_parse_reasoning_response(raw_response)
        
    answer = clean_answer_tokens(answer)
    return status, answer
