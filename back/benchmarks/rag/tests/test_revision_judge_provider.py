import asyncio
import json
from types import SimpleNamespace

import pytest

from core.evaluator import CloudEvaluator
from revision_round1.judge.run_revision_pilot import build_judge_context


class MockProviderEngine:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        content = self.content.pop(0) if isinstance(self.content, list) else self.content
        return SimpleNamespace(
            id="mock-request",
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=None),
        )


def valid_payload():
    return json.dumps({
        "answer_relevance": {"score": 0.7, "explanation": "relevant"},
        "semantic_accuracy": {"score": 0.8, "explanation": "mostly correct"},
        "faithfulness": {"score": 0.9, "statements": []},
        "citation_fidelity": {"score": 0.5, "citations": []},
    })


def test_provider_engine_uses_json_mode_and_validates_all_scores():
    provider = MockProviderEngine(valid_payload())
    events = []
    evaluator = CloudEvaluator(
        api_key="unused",
        base_url="http://127.0.0.1:20128/v1",
        model_name="agy/gemini-3.8-flash-low",
        concurrency=3,
        rpm=0,
        max_retries=2,
        provider_engine=provider,
        temperature=0.0,
        response_format={"type": "json_object"},
        strict_output_validation=True,
        request_observer=events.append,
    )
    prompt = {"system_prompt": "system", "user_prompt_template": "{query} {context} {golden_answer} {answer}"}
    result = asyncio.run(evaluator.evaluate_all_metrics(prompt, has_context=True, query="q", context="c", golden_answer="g", answer="a"))
    assert result["semantic_accuracy"]["score"] == 0.8
    assert len(provider.calls) == 1
    assert provider.calls[0]["response_format"] == {"type": "json_object"}
    assert provider.calls[0]["temperature"] == 0.0
    assert events[0]["event"] == "success"
    assert events[0]["prompt_tokens"] == 10


def test_provider_engine_receives_explicit_output_token_ceiling():
    provider = MockProviderEngine(valid_payload())
    evaluator = CloudEvaluator(
        api_key="unused",
        base_url="http://127.0.0.1:20128/v1",
        model_name="agy/gemini-3.8-flash-low",
        concurrency=1,
        rpm=0,
        max_retries=1,
        provider_engine=provider,
        temperature=0.0,
        response_format={"type": "json_object"},
        strict_output_validation=True,
        max_output_tokens=512,
    )
    prompt = {"system_prompt": "system", "user_prompt_template": "{query} {context} {golden_answer} {answer}"}
    asyncio.run(evaluator.evaluate_all_metrics(prompt, has_context=True, query="q", context="c", golden_answer="g", answer="a"))
    assert provider.calls[0]["max_tokens"] == 512


def test_request_guard_blocks_provider_call_before_submission():
    provider = MockProviderEngine(valid_payload())

    def reject():
        raise RuntimeError("bounded request budget exhausted")

    evaluator = CloudEvaluator(
        api_key="unused",
        base_url="http://127.0.0.1:20128/v1",
        model_name="agy/gemini-3.8-flash-low",
        concurrency=1,
        rpm=0,
        max_retries=1,
        provider_engine=provider,
        request_guard=reject,
    )
    with pytest.raises(RuntimeError, match="bounded request budget exhausted"):
        asyncio.run(evaluator.call_llm("system", "user"))
    assert provider.calls == []


def test_strict_provider_does_not_replace_invalid_score_with_zero():
    payload = json.loads(valid_payload())
    payload["semantic_accuracy"]["score"] = 1.01
    provider = MockProviderEngine(json.dumps(payload))
    evaluator = CloudEvaluator(
        api_key="unused",
        base_url="http://127.0.0.1:20128/v1",
        model_name="agy/gemini-3.8-flash-low",
        concurrency=2,
        rpm=0,
        max_retries=1,
        provider_engine=provider,
        response_format={"type": "json_object"},
        strict_output_validation=True,
    )
    prompt = {"system_prompt": "system", "user_prompt_template": "{query} {context} {golden_answer} {answer}"}
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        asyncio.run(evaluator.evaluate_all_metrics(prompt, has_context=True, query="q", context="c", golden_answer="g", answer="a"))
    assert len(provider.calls) == 2


def test_strict_provider_retries_malformed_json_once():
    provider = MockProviderEngine(["{malformed", valid_payload()])
    evaluator = CloudEvaluator(
        api_key="unused",
        base_url="http://127.0.0.1:20128/v1",
        model_name="agy/gemini-3.8-flash-low",
        concurrency=3,
        rpm=0,
        max_retries=2,
        provider_engine=provider,
        response_format={"type": "json_object"},
        strict_output_validation=True,
    )
    prompt = {"system_prompt": "system", "user_prompt_template": "{query} {context} {golden_answer} {answer}"}
    result = asyncio.run(evaluator.evaluate_all_metrics(prompt, has_context=True, query="q", context="c", golden_answer="g", answer="a"))
    assert result["semantic_accuracy"]["score"] == 0.8
    assert len(provider.calls) == 2


def test_revision_judge_uses_historical_retrieved_chunk_context():
    baseline = {
        "context_text": "This broader aggregate context must not be used.",
        "retrieved_chunks": [
            {"paper_id": "pmid:1", "page_number": 2, "text_content": "Only the stored chunk text."},
        ],
    }
    rendered = build_judge_context(baseline)
    assert "Only the stored chunk text." in rendered
    assert "This broader aggregate context must not be used." not in rendered
    assert "pmid:1" in rendered
