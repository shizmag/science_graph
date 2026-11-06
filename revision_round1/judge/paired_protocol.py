"""Offline paired-identical-ID judge protocol and hard-disabled call boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

BASELINES = ("B3", "B5")
ACCEPTED_EVIDENCE_STATUS = {"exact_archived", "verified_reconstructed"}

@dataclass(frozen=True)
class JudgeInput:
    query_id: str
    paper_baseline: str
    query: str
    golden_answer: str
    candidate_answer: str
    generator_visible_evidence: str
    rubric: str
    token_budget: int
    evidence_status: str


def make_pair(*, query_id: str, query: str, golden_answer: str,
              generator_visible_evidence: str, rubric: str, b3_answer: str, b5_answer: str,
              token_budget: int, evidence_status: str) -> tuple[JudgeInput, JudgeInput]:
    """Create same-ID paired inputs; only candidate answer and baseline label vary.

    Evidence provenance is mandatory: top retrieved chunks cannot be passed as a
    substitute unless the caller has verified that they are the full generator input.
    """
    if not query_id.strip() or token_budget <= 0:
        raise ValueError("query_id must be nonempty and token_budget must be positive")
    if evidence_status not in ACCEPTED_EVIDENCE_STATUS:
        raise ValueError("judging requires exact archived or verified-reconstructed generator-visible evidence")
    if not all(isinstance(x, str) and x.strip() for x in
               (query, golden_answer, generator_visible_evidence, rubric, b3_answer, b5_answer)):
        raise ValueError("both answers and all shared prompt fields must be nonempty strings")
    return (
        JudgeInput(query_id, "B3", query, golden_answer, b3_answer, generator_visible_evidence, rubric, token_budget, evidence_status),
        JudgeInput(query_id, "B5", query, golden_answer, b5_answer, generator_visible_evidence, rubric, token_budget, evidence_status),
    )


def validate_pair(pair: tuple[JudgeInput, JudgeInput]) -> None:
    left, right = pair
    if (left.query_id != right.query_id or left.paper_baseline != "B3" or right.paper_baseline != "B5"
        or left.query != right.query or left.golden_answer != right.golden_answer
        or left.generator_visible_evidence != right.generator_visible_evidence
        or left.rubric != right.rubric or left.token_budget != right.token_budget
        or left.evidence_status != right.evidence_status
        or left.evidence_status not in ACCEPTED_EVIDENCE_STATUS):
        raise ValueError("paired inputs must share ID, query, reference, complete evidence, rubric, and budget")


def paired_call_budget(pairs: Mapping[str, tuple[JudgeInput, JudgeInput]], *, judges: tuple[str, ...],
                       output_token_budget_per_call: int) -> dict[str, int]:
    if not judges or len(set(judges)) != len(judges):
        raise ValueError("judge IDs must be nonempty and unique")
    if output_token_budget_per_call <= 0:
        raise ValueError("output token budget must be positive")
    for query_id, pair in pairs.items():
        validate_pair(pair)
        if pair[0].query_id != query_id:
            raise ValueError("pair mapping key must equal query ID")
    input_tokens = sum(item.token_budget for pair in pairs.values() for item in pair) * len(judges)
    calls = len(pairs) * 2 * len(judges)
    return {"paired_query_ids": len(pairs), "judge_count": len(judges), "evaluations": calls,
            "input_token_budget": input_tokens, "output_token_budget": calls * output_token_budget_per_call}


def score_with_fake_judge(pair: tuple[JudgeInput, JudgeInput], judge: Callable[[JudgeInput], object]) -> dict[str, object]:
    """Testable local seam: injected fake only; production network boundary stays disabled."""
    validate_pair(pair)
    return {item.paper_baseline: judge(item) for item in pair}


def real_judge_call(*_args: object, **_kwargs: object) -> None:
    """Hard stop. No model/provider/network invocation is implemented in this workstream."""
    raise RuntimeError("Real judge calls are disabled by design; use an injected fake judge in tests.")
