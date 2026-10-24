"""Unit tests for teacher-force logit backfill (no real model)."""

from unittest.mock import MagicMock

from backfill_logits import (
    build_generation_prompt,
    compact_tokens_for_eval,
    compute_telemetry_for_answer,
)


def test_build_generation_prompt_b0():
    prompts = MagicMock()
    p = build_generation_prompt(prompts, "What is X?", "B0", {})
    assert "What is X?" in p
    assert "general knowledge" in p
    prompts.get_prompt.assert_not_called()


def test_build_generation_prompt_expander():
    prompts = MagicMock()
    prompts.get_prompt.return_value = "EXPANDED"
    pre = {"enrichment_block": "graph facts here", "trimmed_text": "t", "trimmed_graph": "g"}
    p = build_generation_prompt(prompts, "Q?", "B6", pre)
    assert p == "EXPANDED"
    prompts.get_prompt.assert_called_once()
    assert prompts.get_prompt.call_args[0][:2] == ("rag", "ask_expander")


def test_build_generation_prompt_no_expander():
    prompts = MagicMock()
    prompts.get_prompt.return_value = "PLAIN"
    pre = {"enrichment_block": "", "trimmed_text": "ctx", "trimmed_graph": ""}
    p = build_generation_prompt(prompts, "Q?", "B1", pre)
    assert p == "PLAIN"
    assert prompts.get_prompt.call_args[0][:2] == ("rag", "ask_no_expander")


def test_compact_tokens_for_eval_drops_top_logprobs():
    tokens = [
        {
            "token_id": 1,
            "token_text": "Hi",
            "logprob": -0.1,
            "entropy": 0.5,
            "msp": 0.9,
            "logit_margin": 1.2,
            "top_logprobs": {"Hi": -0.1, "Hello": -2.0},
            "char_start": 0,
            "char_end": 2,
        }
    ]
    out = compact_tokens_for_eval(tokens, drop_top=True)
    assert "top_logprobs" not in out[0]
    assert out[0]["msp"] == 0.9
    assert out[0]["logprob"] == -0.1


def test_compute_telemetry_for_answer_mocked():
    engine = MagicMock()
    rag_tokens = [
        {
            "token_id": 10,
            "token_text": "Yes",
            "logprob": -0.2,
            "entropy": 0.4,
            "msp": 0.85,
            "logit_margin": 1.5,
            "top_logprobs": {"Yes": -0.2, "No": -1.7},
            "char_start": 0,
            "char_end": 3,
        },
        {
            "token_id": 11,
            "token_text": ".",
            "logprob": -0.05,
            "entropy": 0.1,
            "msp": 0.95,
            "logit_margin": 3.0,
            "top_logprobs": {".": -0.05},
            "char_start": 3,
            "char_end": 4,
        },
    ]
    base_tokens = [
        {**rag_tokens[0], "logprob": -0.5, "msp": 0.7},
        {**rag_tokens[1], "logprob": -0.1, "msp": 0.9},
    ]

    def score(prompt, answer):
        if "general knowledge" in prompt:
            return base_tokens
        return rag_tokens

    engine.score_text_logprobs.side_effect = score

    tokens_info, shannon = compute_telemetry_for_answer(
        engine,
        prompt="RAG PROMPT",
        answer_text="Yes.",
        query="Is it true?",
        baseline="B1",
        existing_shannon={
            "h_rank_pre_rerank": 2.0,
            "h_gen": 0.5,
            "delta_h_gen": 0.3,
        },
        h_b0=0.8,
    )

    assert len(tokens_info) == 2
    assert shannon["avg_msp"] > 0
    assert shannon["ll_rag"] != 0.0
    assert shannon["ll_base"] != 0.0
    assert shannon["clr"] == round(shannon["ll_rag"] - shannon["ll_base"], 4) or abs(
        shannon["clr"] - (shannon["ll_rag"] - shannon["ll_base"])
    ) < 1e-6
    assert shannon["h_rank_pre_rerank"] == 2.0  # preserved retrieval fields
    assert "first_token_msp" in shannon
    assert engine.score_text_logprobs.call_count == 2


def test_compute_telemetry_for_answer_rejects_malformed_token_container() -> None:
    engine = MagicMock()
    engine.score_text_logprobs.side_effect = ["not-a-token-list", None]

    tokens_info, shannon = compute_telemetry_for_answer(
        engine,
        prompt="RAG PROMPT",
        answer_text="Answer.",
        query="Question?",
        baseline="B1",
        h_b0=0.0,
    )

    assert tokens_info == []
    assert shannon["ll_rag"] == 0.0
    assert shannon["ll_base"] == 0.0
    assert shannon["clr"] == 0.0
