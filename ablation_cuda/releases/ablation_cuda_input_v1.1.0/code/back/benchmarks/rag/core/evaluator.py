import os
import sys
import time
import asyncio
import json
import math
import re
import random
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from core.limiter import AsyncRateLimiter
from core.metrics import (
    calculate_retrieval_recall,
    calculate_context_precision,
    estimate_prompt_tokens,
    normalize_optional_text,
    get_is_answerable,
    detect_abstention,
    classify_answerability
)
from core.subprocess_runner import format_progress_marker

class CloudEvaluator:
    """Interacts with the Cloud LLM provider (OpenAI API compatible) to score generated answers."""
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        concurrency: int,
        rpm: int,
        max_retries: int = 5,
        *,
        provider_engine=None,
        temperature: float = 0.0,
        response_format=None,
        strict_output_validation: bool = False,
        request_observer=None,
        max_output_tokens: Optional[int] = None,
        request_guard=None,
    ):
        if provider_engine is None:
            try:
                import openai
            except ImportError:
                raise ImportError("Please install the 'openai' python package to use CloudEvaluator.")
            self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = None
        self.provider_engine = provider_engine
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rate_limiter = AsyncRateLimiter(rpm)
        self.max_retries = max_retries
        self.temperature = temperature
        self.response_format = response_format
        self.strict_output_validation = strict_output_validation
        self.request_observer = request_observer
        self.max_output_tokens = max_output_tokens
        self.request_guard = request_guard

    def can_acquire(self) -> bool:
        """Returns True if the underlying rate limiter has immediate capacity."""
        return self.rate_limiter.can_acquire()

    def has_capacity(self) -> bool:
        """Alias for can_acquire(). Checks if rate limit capacity is available."""
        return self.rate_limiter.has_capacity()

    async def call_llm(self, system_prompt: str, user_prompt: str, max_retries: Optional[int] = None) -> str:
        """Invokes the cloud model using exponential backoff with jitter on error."""
        if max_retries is None:
            max_retries = self.max_retries
        from src import console as con
        async with self.semaphore:
            for attempt in range(max(1, max_retries)):
                await self.rate_limiter.wait()
                if self.request_guard:
                    self.request_guard()
                started_at = time.monotonic()
                try:
                    messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                    ]
                    request_kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                    }
                    if self.max_output_tokens is not None:
                        request_kwargs["max_tokens"] = int(self.max_output_tokens)
                    if self.response_format is not None:
                        request_kwargs["response_format"] = self.response_format
                    if self.provider_engine is None:
                        response = await self.client.chat.completions.create(**request_kwargs)
                    else:
                        response = await asyncio.to_thread(
                            self.provider_engine.chat_completion,
                            messages,
                            model=self.model_name,
                            temperature=self.temperature,
                            response_format=self.response_format,
                            max_tokens=self.max_output_tokens,
                        )
                    
                    if not response or not getattr(response, "choices", None):
                        raise ValueError("Invalid API response: choices list is missing or empty.")
                    
                    choice = response.choices[0]
                    if not choice or not getattr(choice, "message", None) or choice.message.content is None:
                        raise ValueError("Invalid API response: message content is None.")
                    
                    if self.request_observer:
                        usage = getattr(response, "usage", None)
                        self.request_observer({
                            "event": "success",
                            "attempt": attempt + 1,
                            "request_id": getattr(response, "id", None),
                            "model": getattr(response, "model", self.model_name),
                            "elapsed_seconds": round(time.monotonic() - started_at, 6),
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                            "gateway_cost": getattr(usage, "cost", getattr(response, "cost", None)),
                            "gateway_cost_details": getattr(usage, "cost_details", None),
                            "finish_reason": getattr(choice, "finish_reason", None),
                            "response_format": self.response_format,
                            "max_output_tokens": self.max_output_tokens,
                        })
                    return choice.message.content
                except Exception as e:
                    if self.request_observer:
                        self.request_observer({
                            "event": "error",
                            "attempt": attempt + 1,
                            "model": self.model_name,
                            "elapsed_seconds": round(time.monotonic() - started_at, 6),
                            "error_type": type(e).__name__,
                            "status_code": getattr(e, "status_code", None),
                            "response_format": self.response_format,
                            "max_output_tokens": self.max_output_tokens,
                        })
                    wait_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                    
                    # Special handling for Rate Limit (429) or rate limit messages
                    is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
                    if is_rate_limit:
                        import openai
                        headers = {}
                        if isinstance(e, openai.RateLimitError):
                            headers = getattr(getattr(e, "response", None), "headers", {})
                        
                        retry_after = headers.get("retry-after")
                        reset_ms = headers.get("x-ratelimit-reset")
                        
                        sleep_seconds = None
                        if retry_after:
                            try:
                                sleep_seconds = float(retry_after)
                            except ValueError:
                                pass
                        elif reset_ms:
                            try:
                                reset_val = float(reset_ms)
                                if reset_val > 100000000000:
                                    reset_val /= 1000.0
                                if reset_val > 1000000000:
                                    sleep_seconds = max(reset_val - time.time(), 0.0)
                                else:
                                    sleep_seconds = reset_val
                            except ValueError:
                                pass
                        
                        # Fallback parsing from exception string
                        if sleep_seconds is None:
                            import re
                            err_str = str(e)
                            match = re.search(r"['\"]X-RateLimit-Reset['\"]\s*:\s*['\"](\d+)['\"]", err_str)
                            if not match:
                                match = re.search(r"X-RateLimit-Reset.*?(\d+)", err_str)
                            if match:
                                try:
                                    reset_val = float(match.group(1))
                                    if reset_val > 100000000000:
                                        reset_val /= 1000.0
                                    if reset_val > 1000000000:
                                        sleep_seconds = max(reset_val - time.time(), 0.0)
                                    else:
                                        sleep_seconds = reset_val
                                except ValueError:
                                    pass
                        
                        # Apply fallback delay if parsing failed or was invalid
                        if sleep_seconds is None or sleep_seconds <= 0 or sleep_seconds > 120:
                            sleep_seconds = (10 * (attempt + 1)) + random.uniform(1.0, 5.0)
                        
                        # Add a small buffer of 1.0 second to ensure we clear the limit
                        wait_time = sleep_seconds + 1.0
                        con.warning(
                            f"Cloud API Rate Limit Exceeded (429) (Attempt {attempt+1}/{max_retries}). "
                            f"Sleeping for {wait_time:.2f}s before retry..."
                        )
                    else:
                        con.warning(
                            f"Cloud API Call Error (Attempt {attempt+1}/{max(1, max_retries)}): {type(e).__name__}. "
                            f"Retrying in {wait_time:.2f}s..."
                        )
                    
                    status_code = getattr(e, "status_code", None)
                    is_nonretryable_client_error = status_code is not None and 400 <= status_code < 500 and status_code != 429
                    if is_nonretryable_client_error or attempt == max(1, max_retries) - 1:
                        raise e
                    await asyncio.sleep(wait_time)
            raise RuntimeError("Cloud LLM request failed after maximum retries.")

    async def evaluate_metric(self, evaluator_config: Dict[str, Any], metric_name: str, **kwargs) -> Dict[str, Any]:
        """Runs the LLM judge call and cleans/parses the score from output JSON."""
        from src import console as con
        system_prompt = evaluator_config["system_prompt"]
        user_prompt = evaluator_config["user_prompt_template"].format(**kwargs)

        max_parse_retries = 2 if self.strict_output_validation else 3
        for parse_attempt in range(max_parse_retries):
            raw_response = await self.call_llm(system_prompt, user_prompt)
            try:
                parsed = self.clean_and_parse_json(raw_response)
                if "score" not in parsed:
                    raise ValueError("JSON response is missing the 'score' key.")
                return parsed
            except Exception as e:
                con.warning(
                    f"JSON Parse Error for '{metric_name}' (Attempt {parse_attempt+1}/{max_parse_retries}): {e}. "
                    f"Raw response: {raw_response[:200] if raw_response else 'None'}..."
                )
                if parse_attempt == max_parse_retries - 1:
                    if self.strict_output_validation:
                        raise ValueError(f"Independent judge returned invalid structured output: {e}") from e
                    return {"score": 0.0, "error": str(e), "raw_response": raw_response}
                await asyncio.sleep(1.0)
        return {"score": 0.0, "error": "Evaluation parsing failed."}

    async def evaluate_all_metrics(self, evaluator_config: Dict[str, Any], has_context: bool, **kwargs) -> Dict[str, Any]:
        """Runs the unified LLM judge call and cleans/parses all scores from output JSON."""
        from src import console as con
        system_prompt = evaluator_config["system_prompt"]
        user_prompt = evaluator_config["user_prompt_template"].format(**kwargs)

        max_parse_retries = 2 if self.strict_output_validation else 3
        for parse_attempt in range(max_parse_retries):
            raw_response = await self.call_llm(system_prompt, user_prompt)
            try:
                parsed = self.clean_and_parse_json(raw_response)
                
                # Check for standard metrics
                required_keys = ["answer_relevance", "semantic_accuracy"]
                if has_context:
                    required_keys.extend(["faithfulness", "citation_fidelity"])
                
                missing_keys = [k for k in required_keys if k not in parsed]
                if missing_keys:
                    raise ValueError(f"JSON response is missing the following metric keys: {missing_keys}")
                
                for k in required_keys:
                    metric_data = parsed[k]
                    # if the response format returned a plain number instead of dict:
                    if isinstance(metric_data, (int, float)):
                        parsed[k] = {"score": float(metric_data)}
                    elif isinstance(metric_data, dict):
                        if "score" not in metric_data:
                            raise ValueError(f"Metric '{k}' is missing the 'score' key.")
                    else:
                        raise ValueError(f"Metric '{k}' has invalid data type: {type(metric_data)}")
                    if self.strict_output_validation:
                        score = parsed[k]["score"]
                        if isinstance(score, bool) or not isinstance(score, (int, float)):
                            raise ValueError(f"Metric '{k}' score must be numeric.")
                        if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
                            raise ValueError(f"Metric '{k}' score must be finite and within [0, 1].")
                
                return parsed
            except Exception as e:
                con.warning(
                    f"JSON Parse/Validation Error (Attempt {parse_attempt+1}/{max_parse_retries}): {e}. "
                    f"Raw response: {raw_response[:200] if raw_response else 'None'}..."
                )
                if parse_attempt == max_parse_retries - 1:
                    if self.strict_output_validation:
                        raise ValueError(f"Independent judge returned invalid structured output: {e}") from e
                    fallback = {
                        "answer_relevance": {"score": 0.0, "error": str(e)},
                        "semantic_accuracy": {"score": 0.0, "error": str(e)}
                    }
                    if has_context:
                        fallback["faithfulness"] = {"score": 0.0, "error": str(e)}
                        fallback["citation_fidelity"] = {"score": 0.0, "error": str(e)}
                    return fallback
                await asyncio.sleep(1.0)
        
        fallback = {
            "answer_relevance": {"score": 0.0, "error": "Evaluation parsing failed."},
            "semantic_accuracy": {"score": 0.0, "error": "Evaluation parsing failed."}
        }
        if has_context:
            fallback["faithfulness"] = {"score": 0.0, "error": "Evaluation parsing failed."}
            fallback["citation_fidelity"] = {"score": 0.0, "error": "Evaluation parsing failed."}
        return fallback

    def clean_and_parse_json(self, text: str) -> Dict[str, Any]:
        """Cleans Markdown tags and parses JSON dictionary from LLM response text."""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
        else:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                text = text[start:end+1]
        
        return json.loads(text)


def build_context_string(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Builds a formatted multi-block string from retrieved chunks."""
    if not retrieved_chunks:
        return "Context is empty."
    blocks = []
    for idx, chunk in enumerate(retrieved_chunks):
        paper_id = chunk.get("paper_id", "Unknown")
        page = chunk.get("page_number", "Unknown")
        text = chunk.get("text_content", "").strip()
        blocks.append(f"Block {idx+1} (Paper: {paper_id}, Page: {page}):\n{text}")
    return "\n\n".join(blocks)


def get_cloud_credentials(config: Any) -> Tuple[str, str, str]:
    """Retrieves API Key, Base URL, and Model Name for Cloud Evaluator from config or env."""
    from src import console as con
    api_key = config.llm_cloud_api_key
    base_url = config.llm_cloud_base_url
    model_name = config.llm_cloud_model_name

    # Fallbacks to environment variables
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    
    if not base_url:
        if api_key and "openrouter" in api_key.lower():
            base_url = "https://openrouter.ai/api/v1"
        else:
            base_url = "https://api.openai.com/v1"

    if not model_name:
        model_name = "google/gemini-2.5-flash"

    if not api_key:
        con.error("Error: Cloud LLM API Key is not configured.")
        con.info("Please set it in ~/.config/pdf-graph-analyzer/config.yaml under llm.cloud.api_key")
        con.info("Or set the OPENAI_API_KEY / OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    return api_key, base_url, model_name


async def evaluate_baseline_case(
    evaluator: CloudEvaluator,
    prompts: Dict[str, Any],
    case_id: str,
    query: str,
    golden_answer: str,
    expected_papers: List[str],
    baseline_name: str,
    baseline_data: Dict[str, Any],
    checkpoint_data: Dict[str, Any],
    checkpoint_path: Path,
    max_input_token: int = 10000,
    is_answerable: bool = True
) -> Dict[str, Any]:
    """Evaluates a single test case for a baseline, checking checkpoint first."""
    from src import console as con
    retrieved_chunks = baseline_data.get("retrieved_chunks", [])
    
    import hashlib
    # Compute payload hash to prevent reuse of cached metrics if answers/inputs change
    hash_payload = {
        "generated_answer": baseline_data.get("generated_answer", ""),
        "retrieved_chunks": retrieved_chunks,
        "golden_answer": golden_answer,
        "query": query
    }
    payload_str = json.dumps(hash_payload, sort_keys=True, ensure_ascii=False)
    payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:12]
    checkpoint_key = f"{case_id}_{baseline_name}_{payload_hash}"
    
    # Return from checkpoint if already evaluated with all required metrics
    cached = checkpoint_data.get(checkpoint_key, {})
    required = ["retrieval_recall", "answer_relevance", "semantic_accuracy", "context_fillness"]
    if baseline_name != "B0" and retrieved_chunks:
        required.extend(["context_precision", "faithfulness", "citation_fidelity"])
    
    cached_metrics = {}
    cached_details = {}
    has_all_required = False
    if isinstance(cached, dict):
        if "metrics" in cached:
            cached_metrics = cached["metrics"]
            cached_details = cached.get("details", {})
        else:
            cached_metrics = cached
            cached_details = cached.get("eval_details", {})
        
        if "answerability_outcome" in cached_metrics:
            has_all_required = True
        else:
            has_all_required = all(cached_metrics.get(r) is not None for r in required)
    
    if checkpoint_key in checkpoint_data and has_all_required:
        if "answerability_outcome" not in cached_metrics:
            generated_answer = baseline_data.get("generated_answer", "")
            judge_answer = get_clean_judge_answer(generated_answer)
            predicted_abstained = detect_abstention(generated_answer, judge_answer)
            outcome = classify_answerability(is_answerable, predicted_abstained)
            cached_metrics["is_answerable"] = is_answerable
            cached_metrics["predicted_abstained"] = predicted_abstained
            cached_metrics["answerability_outcome"] = outcome
            checkpoint_data[checkpoint_key] = {
                "metrics": cached_metrics,
                "details": cached_details
            }
            save_checkpoint(checkpoint_path, checkpoint_data)

        if "token_output" not in cached_metrics:
            from core.metrics import count_text_tokens
            generated_answer = baseline_data.get("generated_answer", "")
            judge_answer = get_clean_judge_answer(generated_answer)
            token_output = count_text_tokens(generated_answer)
            token_answer = count_text_tokens(judge_answer)
            token_reasoning = max(0, token_output - token_answer)
            cached_metrics["token_output"] = token_output
            cached_metrics["token_answer"] = token_answer
            cached_metrics["token_reasoning"] = token_reasoning
            checkpoint_data[checkpoint_key] = {
                "metrics": cached_metrics,
                "details": cached_details
            }
            save_checkpoint(checkpoint_path, checkpoint_data)
        
        res = dict(cached_metrics)
        if is_answerable:
            r_relevance = res.get("answer_relevance", 0.0)
            s_accuracy = res.get("semantic_accuracy", 0.0)
            if r_relevance is not None and s_accuracy is not None:
                if r_relevance + s_accuracy > 0:
                    res["ar_sa_f1"] = round(2.0 * (r_relevance * s_accuracy) / (r_relevance + s_accuracy), 4)
                else:
                    res["ar_sa_f1"] = 0.0
            else:
                res["ar_sa_f1"] = 0.0
        else:
            res["ar_sa_f1"] = None
        res["eval_details"] = cached_details
        return res

    generated_answer = baseline_data.get("generated_answer", "")
    judge_answer = get_clean_judge_answer(generated_answer)
    predicted_abstained = detect_abstention(generated_answer, judge_answer)
    outcome = classify_answerability(is_answerable, predicted_abstained)

    eval_metrics = {
        "is_answerable": is_answerable,
        "predicted_abstained": predicted_abstained,
        "answerability_outcome": outcome,
        "retrieval_recall": cached_metrics.get("retrieval_recall", 0.0),
        "context_precision": cached_metrics.get("context_precision", 0.0),
        "faithfulness": cached_metrics.get("faithfulness", 0.0),
        "answer_relevance": cached_metrics.get("answer_relevance", 0.0),
        "citation_fidelity": cached_metrics.get("citation_fidelity", 0.0),
        "semantic_accuracy": cached_metrics.get("semantic_accuracy", 0.0),
        "context_fillness": cached_metrics.get("context_fillness", 0.0),
        "token_output": cached_metrics.get("token_output", 0),
        "token_answer": cached_metrics.get("token_answer", 0),
        "token_reasoning": cached_metrics.get("token_reasoning", 0),
    }
    import copy
    eval_details = copy.deepcopy(cached_details)

    if baseline_data.get("status") in ("error", "failed", "timeout"):
        res = dict(eval_metrics)
        res["status"] = baseline_data.get("status")
        res["eval_details"] = eval_details
        checkpoint_data[checkpoint_key] = {
            "metrics": res,
            "details": eval_details
        }
        save_checkpoint(checkpoint_path, checkpoint_data, force=True)

        return res

    retrieved_papers = baseline_data.get("retrieved_papers", [])

    # Calculate traditional retrieval metrics
    if "retrieval_recall" not in cached_metrics or cached_metrics["retrieval_recall"] is None:
        eval_metrics["retrieval_recall"] = baseline_data.get("retrieval_recall")
        if eval_metrics["retrieval_recall"] is None:
            eval_metrics["retrieval_recall"] = calculate_retrieval_recall(expected_papers, retrieved_papers)
    else:
        eval_metrics["retrieval_recall"] = cached_metrics["retrieval_recall"]

    if "context_precision" not in cached_metrics or cached_metrics["context_precision"] is None:
        eval_metrics["context_precision"] = baseline_data.get("context_precision")
        if eval_metrics["context_precision"] is None:
            eval_metrics["context_precision"] = calculate_context_precision(expected_papers, retrieved_chunks)
    else:
        eval_metrics["context_precision"] = cached_metrics["context_precision"]

    # Calculate context fillness
    if "context_fillness" not in cached_metrics or cached_metrics["context_fillness"] is None:
        context_fillness = baseline_data.get("context_fillness")
        if context_fillness is None:
            context_token = baseline_data.get("context_token")
            max_input_token_val = baseline_data.get("max_input_token")
            if context_token is None:
                context_token = estimate_prompt_tokens(query, retrieved_chunks, baseline_name)
            if max_input_token_val is None:
                max_input_token_val = max_input_token
            context_fillness = round(context_token / max_input_token_val, 4) if max_input_token_val > 0 else 0.0
            context_fillness = min(max(context_fillness, 0.0), 1.0)
        eval_metrics["context_fillness"] = context_fillness
    else:
        eval_metrics["context_fillness"] = cached_metrics["context_fillness"]

    # Calculate output token metrics if they were not populated from cached/checkpoint
    if "token_output" not in cached_metrics or cached_metrics["token_output"] is None:
        from core.metrics import count_text_tokens
        token_output = count_text_tokens(generated_answer)
        token_answer = count_text_tokens(judge_answer)
        token_reasoning = max(0, token_output - token_answer)
        eval_metrics["token_output"] = token_output
        eval_metrics["token_answer"] = token_answer
        eval_metrics["token_reasoning"] = token_reasoning

    if outcome in ("TN", "FN", "FP"):
        if outcome == "TN":
            eval_metrics["faithfulness"] = None
            eval_metrics["answer_relevance"] = None
            eval_metrics["citation_fidelity"] = None
            eval_metrics["semantic_accuracy"] = None
            eval_metrics["ar_sa_f1"] = None
        elif outcome == "FP":
            eval_metrics["faithfulness"] = None
            eval_metrics["answer_relevance"] = None
            eval_metrics["citation_fidelity"] = None
            eval_metrics["semantic_accuracy"] = None
            eval_metrics["ar_sa_f1"] = None
            eval_metrics["hallucination"] = True
            eval_metrics["unsupported_answer"] = True
        else: # FN
            eval_metrics["faithfulness"] = 0.0
            eval_metrics["answer_relevance"] = 0.0
            eval_metrics["citation_fidelity"] = 0.0
            eval_metrics["semantic_accuracy"] = 0.0
            eval_metrics["ar_sa_f1"] = 0.0

        checkpoint_data[checkpoint_key] = {
            "metrics": eval_metrics,
            "details": eval_details
        }
        save_checkpoint(checkpoint_path, checkpoint_data, force=True)

        
        rec = eval_metrics.get('retrieval_recall')
        rec_str = f"{rec:.2f}" if rec is not None else "N/A"
        con.info(
            f"  Evaluated {case_id} [{baseline_name}]: (Abstention/Answerability Outcome: {outcome}) "
            f"Recall={rec_str}, "
            f"Faithfulness={eval_metrics['faithfulness']}, "
            f"Relevance={eval_metrics['answer_relevance']}, "
            f"Semantic={eval_metrics['semantic_accuracy']}"
        )
        
        res = dict(eval_metrics)
        res["eval_details"] = eval_details
        return res

    # If clean answer is missing, skip LLM calls
    if not judge_answer.strip():
        checkpoint_data[checkpoint_key] = {
            "metrics": eval_metrics,
            "details": eval_details
        }
        save_checkpoint(checkpoint_path, checkpoint_data, force=True)

        res = dict(eval_metrics)
        res["eval_details"] = eval_details
        return res

    # Prepare LLM evaluator inputs
    context_str = build_context_string(retrieved_chunks)
    has_context = (baseline_name != "B0" and bool(retrieved_chunks))

    llm_metrics_needed = ["answer_relevance", "semantic_accuracy"]
    if has_context:
        llm_metrics_needed.extend(["faithfulness", "citation_fidelity"])
    elif baseline_name != "B0":
        con.warning(
            f"No retrieved chunks found for baseline {baseline_name} in case {case_id}. "
            "Faithfulness, citation fidelity, and context precision will default to 0.0. "
            "Ensure you are passing the full evaluation_results.yaml file as input, "
            "not the simplified evaluation_results_judge.yaml file."
        )

    # Check if we need to call LLM:
    need_llm_call = any(cached_metrics.get(m) is None for m in llm_metrics_needed)

    if need_llm_call:
        prompt_key = "unified_with_context_evaluator" if has_context else "unified_without_context_evaluator"
        if prompt_key in prompts:
            evaluator_config = prompts[prompt_key]
        else:
            # Fallback for backward compatibility/minimal test prompts
            evaluator_config = prompts.get("answer_relevance_evaluator", {"system_prompt": "", "user_prompt_template": ""})

        kwargs = {
            "query": query,
            "golden_answer": golden_answer,
            "answer": judge_answer
        }
        if has_context:
            kwargs["context"] = context_str

        try:
            eval_results = await evaluator.evaluate_all_metrics(
                evaluator_config=evaluator_config,
                has_context=has_context,
                **kwargs
            )
            for m in llm_metrics_needed:
                metric_data = eval_results.get(m, {})
                eval_details[m] = metric_data
                if isinstance(metric_data, dict):
                    score_val = metric_data.get("score")
                elif isinstance(metric_data, (int, float)):
                    score_val = metric_data
                else:
                    score_val = None

                if score_val is None:
                    eval_metrics[m] = 0.0
                else:
                    try:
                        eval_metrics[m] = float(score_val)
                    except (ValueError, TypeError):
                        eval_metrics[m] = 0.0
        except Exception as e:
            con.error(f"Failed executing unified LLM evaluator for {checkpoint_key}: {e}")
            for m in llm_metrics_needed:
                eval_metrics[m] = 0.0
                eval_details[m] = {"score": 0.0, "error": str(e)}
    else:
        for m in llm_metrics_needed:
            eval_metrics[m] = cached_metrics.get(m, 0.0)
            eval_details[m] = cached_details.get(m, {})

    if is_answerable:
        r_relevance = eval_metrics.get("answer_relevance", 0.0)
        s_accuracy = eval_metrics.get("semantic_accuracy", 0.0)
        if r_relevance + s_accuracy > 0:
            eval_metrics["ar_sa_f1"] = round(2.0 * (r_relevance * s_accuracy) / (r_relevance + s_accuracy), 4)
        else:
            eval_metrics["ar_sa_f1"] = 0.0
    else:
        eval_metrics["ar_sa_f1"] = None

    # Save to checkpoint
    checkpoint_data[checkpoint_key] = {
        "metrics": eval_metrics,
        "details": eval_details
    }
    save_checkpoint(checkpoint_path, checkpoint_data, force=True)

    
    rec = eval_metrics.get('retrieval_recall')
    faith = eval_metrics.get('faithfulness')
    rel = eval_metrics.get('answer_relevance')
    sem = eval_metrics.get('semantic_accuracy')
    rec_str = f"{rec:.2f}" if rec is not None else "N/A"
    faith_str = f"{faith:.2f}" if faith is not None else "N/A"
    rel_str = f"{rel:.2f}" if rel is not None else "N/A"
    sem_str = f"{sem:.2f}" if sem is not None else "N/A"
    con.info(
        f"  Evaluated {case_id} [{baseline_name}]: "
        f"Recall={rec_str}, "
        f"Faithfulness={faith_str}, "
        f"Relevance={rel_str}, "
        f"Semantic={sem_str}"
    )

    res = dict(eval_metrics)
    res["eval_details"] = eval_details
    return res


_checkpoint_last_save = 0.0
_checkpoint_counter = 0


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Loads checkpoint JSON dict from path."""
    from src import console as con
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            con.warning(f"Could not load checkpoint: {e}. Starting fresh.")
    return {}


def save_checkpoint(path: Path, data: Dict[str, Any], force: bool = False) -> None:
    """Safely writes checkpoint JSON dict to path using a unique temp file."""
    global _checkpoint_last_save, _checkpoint_counter
    from src import console as con

    now = time.time()
    _checkpoint_counter += 1

    if not force and (now - _checkpoint_last_save < 3.0) and (_checkpoint_counter % 10 != 0):
        return

    _checkpoint_last_save = now
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".tmp.{os.getpid()}_{time.time_ns()}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.replace(path)
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        con.warning(f"Failed to save checkpoint: {e}")



async def run_evaluation(args: Any, config: Any, con: Any) -> None:
    """Core evaluation orchestrator. Runs LLM-as-a-Judge and metrics scoring."""
    script_dir = Path(__file__).resolve().parents[1]
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
        
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    # Bind output path to input path directory if output has default parent but input is in a subdirectory
    if input_path.parent != script_dir and input_path.parent.name not in ("graphs", "reports"):
        if output_path.parent in (script_dir, script_dir / "graphs", script_dir / "reports"):
            output_path = input_path.parent / output_path.name
            args.output = str(output_path)

    if not input_path.exists():
        con.error(f"Input file not found: {input_path}")
        sys.exit(1)

    con.info(f"Loading benchmark results from: {input_path}")
    YamlLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = yaml.load(f, Loader=YamlLoader)

    if not input_data or "results" not in input_data:
        con.error("Invalid benchmark results file structure. Must contain a 'results' key.")
        sys.exit(1)

    # Load judge prompts
    prompts_path = script_dir / "prompts" / "judge_prompts.yaml"
    if not prompts_path.exists():
        con.error(f"Judge prompts file not found: {prompts_path}")
        sys.exit(1)
    
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_dict = yaml.safe_load(f)
    con.success("Loaded judge prompts.")

    # Initialize Cloud Evaluator
    api_key, base_url, model_name = get_cloud_credentials(config)
    con.info(f"Initializing Cloud LLM Evaluator ({model_name}) ...")
    retries = getattr(args, "retries", None) or getattr(config, "llm_evaluation_retries", 5)
    evaluator = CloudEvaluator(api_key, base_url, model_name, args.concurrency, args.rpm, retries)

    # Handle checkpoint
    checkpoint_path = output_path.parent / ".eval_checkpoint.json"
    if args.clear_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        con.info("Cleared existing checkpoint.")
    
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data:
        con.info(f"Resuming evaluation using checkpoint. ({len(checkpoint_data)} items already evaluated)")

    # Resolve baselines to evaluate
    if args.baselines.lower() == "all":
        baselines_to_run = "all"
    else:
        baselines_to_run = [b.strip().upper() for b in args.baselines.split(",") if b.strip()]

    # Limit cases
    cases = input_data["results"]
    if args.limit:
        cases = cases[:args.limit]
        con.info(f"Limiting evaluation to first {args.limit} questions.")

    con.info("Starting evaluation task generation...")

    evaluation_futures = []
    
    for case in cases:
        case_id = case.get("id")
        query = case.get("query")
        golden_answer = normalize_optional_text(case.get("golden_answer"))
        expected_papers = case.get("expected_papers", [])

        for baseline_name, baseline_data in case.get("baselines", {}).items():
            if baselines_to_run != "all" and baseline_name not in baselines_to_run:
                continue

            original_metadata = input_data.get("metadata", {})
            original_metadata = original_metadata.get("original_metadata", original_metadata)
            max_tokens_val = original_metadata.get("llm", {}).get("model_max_context")
            if max_tokens_val is None:
                max_tokens_val = getattr(config, "llm_model_max_context", 4096)

            evaluation_futures.append((
                case_id,
                baseline_name,
                baseline_data,
                evaluate_baseline_case(
                    evaluator,
                    prompts_dict,
                    case_id,
                    query,
                    golden_answer,
                    expected_papers,
                    baseline_name,
                    baseline_data,
                    checkpoint_data,
                    checkpoint_path,
                    max_input_token=max_tokens_val,
                    is_answerable=get_is_answerable(case)
                )
            ))

    con.info(f"Total baseline evaluations to verify: {len(evaluation_futures)}")
    
    results_map = {}
    for case_id, baseline_name, _, coro in evaluation_futures:
        results_map[f"{case_id}_{baseline_name}"] = asyncio.create_task(coro)

    # Execute with live progress logging for the pipeline dashboard
    completed_count = 0
    total_count = len(results_map)
    for fut in asyncio.as_completed(results_map.values()):
        await fut
        completed_count += 1
        con.info(f"Evaluated case {completed_count}/{total_count}")
        print(format_progress_marker("evaluation", completed_count, total_count), flush=True)

    con.info("All evaluations complete. Aggregating results...")
    save_checkpoint(checkpoint_path, checkpoint_data, force=True)

    
    final_results = []
    summary_stats = {}

    for case in cases:
        case_id = case.get("id")
        query = case.get("query")
        golden_answer = normalize_optional_text(case.get("golden_answer"))
        expected_papers = case.get("expected_papers", [])

        case_output = {
            "id": case_id,
            "category": case.get("category", "general"),
            "query": query,
            "golden_answer": golden_answer,
            "expected_papers": expected_papers,
            "is_answerable": get_is_answerable(case),
            "baselines": {}
        }

        for baseline_name, baseline_data in case.get("baselines", {}).items():
            if baselines_to_run != "all" and baseline_name not in baselines_to_run:
                continue

            key = f"{case_id}_{baseline_name}"
            eval_res = results_map[key].result()
            eval_metrics = {k: v for k, v in eval_res.items() if k != "eval_details"}
            eval_details = eval_res.get("eval_details", {})

            if baseline_name not in summary_stats:
                summary_stats[baseline_name] = {
                    "latency_sec": [],
                    "retrieval_recall": [],
                    "context_precision": [],
                    "faithfulness": [],
                    "answer_relevance": [],
                    "citation_fidelity": [],
                    "semantic_accuracy": [],
                    "context_fillness": [],
                    "ar_sa_f1": [],
                    "token_output": [],
                    "token_answer": [],
                    "token_reasoning": [],
                }

            is_ans = case.get("is_answerable")
            if is_ans is None:
                is_ans = True
            else:
                is_ans = str(is_ans).lower() == "true"

            latency = baseline_data.get("latency_sec")
            if latency is not None and is_ans:
                summary_stats[baseline_name]["latency_sec"].append(latency)

            for k, val in eval_metrics.items():
                if k not in summary_stats[baseline_name]:
                    continue
                if baseline_name == "B0" and k in ("faithfulness", "citation_fidelity", "context_precision"):
                    continue
                if baseline_name != "B0" and not baseline_data.get("retrieved_chunks") and k in ("faithfulness", "citation_fidelity", "context_precision"):
                    continue
                if not is_ans:
                    continue
                if val is None:
                    continue
                summary_stats[baseline_name][k].append(val)

            shannon_diag = baseline_data.get("shannon_diagnostics") or (baseline_data.get("metrics", {}).get("shannon_diagnostics") if isinstance(baseline_data.get("metrics"), dict) else None)
            case_output["baselines"][baseline_name] = {
                "status": baseline_data.get("status", "success"),
                "latency_sec": latency,
                "is_answerable": eval_metrics.get("is_answerable", get_is_answerable(case)),
                "predicted_abstained": eval_metrics.get("predicted_abstained", False),
                "answerability_outcome": eval_metrics.get("answerability_outcome", "TP"),
                "retrieved_papers": baseline_data.get("retrieved_papers", []),
                "eval_metrics": eval_metrics,
                "eval_details": eval_details,
                "generated_answer": baseline_data.get("generated_answer", ""),
                "retrieved_chunks": baseline_data.get("retrieved_chunks", []),
                "context_token": baseline_data.get("context_token"),
                "max_input_token": baseline_data.get("max_input_token"),
                "context_fillness": baseline_data.get("context_fillness"),
                "trace": baseline_data.get("trace"),
                "metrics": baseline_data.get("metrics"),
            }
            if shannon_diag:
                case_output["baselines"][baseline_name]["shannon_diagnostics"] = shannon_diag

        final_results.append(case_output)

    # Compute averages for summary
    final_summary = {}
    for baseline, metrics in summary_stats.items():
        final_summary[baseline] = {}
        for m_name, values in metrics.items():
            numeric = [v for v in values if v is not None]
            if numeric:
                final_summary[baseline][f"avg_{m_name}"] = round(sum(numeric) / len(numeric), 4)
            else:
                final_summary[baseline][f"avg_{m_name}"] = 0.0

    # Write report
    output_report = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original_metadata": input_data.get("metadata", {}),
            "evaluation_llm": {
                "model_name": model_name,
                "provider": base_url
            }
        },
        "summary": final_summary,
        "results": final_results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output_report, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Write eval trace
    trace_dir = output_path.parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    eval_trace_path = trace_dir / "eval_trace.jsonl"
    try:
        with open(eval_trace_path, "w", encoding="utf-8") as f:
            for case_out in final_results:
                case_id = case_out.get("id")
                category = case_out.get("category", "general")
                is_ans = get_is_answerable(case_out)
                for baseline_name, b_data in case_out.get("baselines", {}).items():
                    metrics = b_data.get("eval_metrics", {})
                    entry = {
                        "query_id": case_id,
                        "baseline": baseline_name,
                        "category": category,
                        "is_answerable": is_ans,
                        "retrieval_recall": metrics.get("retrieval_recall"),
                        "context_precision": metrics.get("context_precision"),
                        "faithfulness": metrics.get("faithfulness"),
                        "answer_relevance": metrics.get("answer_relevance"),
                        "citation_fidelity": metrics.get("citation_fidelity"),
                        "semantic_accuracy": metrics.get("semantic_accuracy"),
                        "context_fillness": metrics.get("context_fillness"),
                        "ar_sa_f1": metrics.get("ar_sa_f1"),
                        "latency_sec": b_data.get("latency_sec"),
                        "judge_model": model_name,
                        "token_output": metrics.get("token_output"),
                        "token_answer": metrics.get("token_answer"),
                        "token_reasoning": metrics.get("token_reasoning"),
                        "eval_details": b_data.get("eval_details", {})
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        con.info(f"Saved evaluation trace to {eval_trace_path}")
    except Exception as e:
        con.warning(f"Could not save evaluation trace: {e}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    con.success(f"Evaluation finished successfully! Report saved to: {output_path}")

    # Save simplified LLM-judge reports with evaluated metrics
    try:
        from core.reporting import save_judge_report, save_individual_judge_reports
        judge_output_path = output_path.with_name(output_path.stem + "_judge" + output_path.suffix)
        save_judge_report(output_report, judge_output_path)
        save_individual_judge_reports(output_report, output_path.parent, output_path.stem, output_path.suffix)
        con.success(f"Judge reports updated and saved to: {judge_output_path} and {output_path.parent / 'baselines'}/")
    except Exception as e:
        con.warning(f"Could not save judge reports: {e}")

    # Rich summary table
    if HAS_RICH:
        table = Table(title="RAG baselines - Aggregated Evaluation Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Baseline", style="cyan")
        table.add_column("Recall", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Faithfulness", justify="right")
        table.add_column("Relevance", justify="right")
        table.add_column("Citations", justify="right")
        table.add_column("Semantic", justify="right")
        table.add_column("Fillness", justify="right")
        table.add_column("Latency (s)", justify="right")

        for baseline in sorted(final_summary.keys()):
            stats = final_summary[baseline]
            recall = f"{stats.get('avg_retrieval_recall', 0.0):.2%}" if baseline != "B0" else "N/A"
            precision = f"{stats.get('avg_context_precision', 0.0):.2%}" if baseline != "B0" else "N/A"
            faithfulness = f"{stats.get('avg_faithfulness', 0.0):.2%}" if baseline != "B0" else "N/A"
            relevance = f"{stats.get('avg_answer_relevance', 0.0):.2%}"
            citations = f"{stats.get('avg_citation_fidelity', 0.0):.2%}" if baseline != "B0" else "N/A"
            semantic = f"{stats.get('avg_semantic_accuracy', 0.0):.2%}"
            fillness = f"{stats.get('avg_context_fillness', 0.0):.2%}"
            latency = f"{stats.get('avg_latency_sec', 0.0):.2f}s"

            table.add_row(
                baseline,
                recall,
                precision,
                faithfulness,
                relevance,
                citations,
                semantic,
                fillness,
                latency
            )

        from rich.console import Console
        Console().print(table)


def get_clean_judge_answer(generated_answer: str) -> str:
    """Helper to extract clean final answer for judge metrics / reports."""
    if not generated_answer:
        return ""
    from core.sanitization import extract_clean_answer
    _, judge_answer = extract_clean_answer(generated_answer)
    return judge_answer


def _fallback_parse_reasoning_response(raw_response: str) -> Tuple[str, str]:
    """
    Fallback implementation of parse_reasoning_response that does not depend on
    heavy imports (like tiktoken, etc.) and is self-contained.
    """
    if not raw_response or not isinstance(raw_response, str):
        return "UNKNOWN", "Error: Empty or incorrect response from model."

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
    answer_match = re.search(r"<\|answer_start\|>(.*?)<\|answer_end\|>", raw_response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer_unclosed = re.search(r"<\|answer_start\|>(.*)", raw_response, re.DOTALL)
        if answer_unclosed:
            answer = answer_unclosed.group(1).strip()
        else:
            answer = raw_response.strip()
            for tag in ["status", "query_analysis", "source_analysis", "reasoning"]:
                answer = re.sub(rf"<\|{tag}_start\|>.*?<\|{tag}_end\|>", "", answer, flags=re.DOTALL)
                answer = re.sub(rf"<\|{tag}_start\|>.*", "", answer, flags=re.DOTALL)
            
            # Clean reasoning markers and headers
            answer = _fallback_clean_reasoning_text(answer)
            
    return status, answer


def _fallback_clean_reasoning_text(text: str) -> str:
    if not text:
        return text

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
        if combined_pattern.search(candidate):
            return _fallback_clean_reasoning_text(candidate)
        text = candidate
    else:
        text = re.sub(
            r"(?:###\s*)?[1-4]\.\s*_(?:analysis|start|reasoning|status|source_analysis)(?:\.\.\.)?.*?(?=(?:###\s*)?(?:5\.\s*_(?:answer|status|reasoning|analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*|(?:###\s*)?Final\s+Answer\s*:?))",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    header_pattern = r"(?:###\s*)?[1-5]\.\s*_(?:analysis|start|reasoning|status|answer|source_analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*"
    text = re.sub(header_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:###\s*)?Final\s+Answer\s*:?", "", text, flags=re.IGNORECASE)
    
    text = re.sub(r"<\|source_id\|>", "__SOURCE_ID_TAG__", text, flags=re.IGNORECASE)
    
    # Strip thinking and technical tokens
    text = _fallback_strip_thinking_tokens(text)
    
    text = text.replace("__SOURCE_ID_TAG__", "<|source_id|>")
    return text.strip()


def _fallback_strip_thinking_tokens(text: str) -> str:
    if not text:
        return text
    # Remove closed think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove unclosed think blocks at the end
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)

    # Model-agnostic generic token stripping
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"<<.*?>>", "", text)
    text = re.sub(r"\[/?(?:[A-Z_]{2,}[A-Z0-9_-]*)\]", "", text)
    text = re.sub(r"</?(?:s|pad|unk|turn)>", "", text, flags=re.IGNORECASE)

    # Patterns for technical formatting tokens
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
        
    return text
