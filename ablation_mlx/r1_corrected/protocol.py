"""Pure, dependency-free protocol objects for the corrected R1 intervention.

The ON/OFF contrast in this module changes only static graph relations.  Text
chunks and the dynamic-enrichment field are deliberately held constant.  This
keeps the intervention auditable without importing the application service or
requiring a network embedding service.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Iterable, Sequence

PROTOCOL_VERSION = "r1-mlx-corrected-2026-11-09"
PROMPT_VERSION = "r1-mlx-structured-context-additive-v2"
EXPERIMENT_ID = "r1_mlx_corrected_structured_context_v1"
ARM_ON = "structured_context_on"
ARM_OFF = "structured_context_off"
INPUT_TOKEN_LIMIT = 12_000
OUTPUT_TOKEN_LIMIT = 768
SMOKE_GENERATION_BUDGET = 12
COHORT_SEED = "r1-mlx-corrected-cohort-v1"
MLX_VERSION = "0.31.2"
MLX_LM_VERSION = "0.31.3"
PYTHON_MAJOR_MINOR = (3, 12)
# This is frozen from the outcome-blind CPU preparation.  Execution must use
# these IDs rather than reselection after a model result is available.
SMOKE_QUERY_IDS = (
    "pubmedqa_15703931",
    "pubmedqa_18041059",
    "pubmedqa_18926458",
    "pubmedqa_27040842",
    "pubmedqa_9199905",
    "pubmedqa_unans_16962519",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class EvidenceChunk:
    """One deterministic retrieved chunk from the frozen local snapshot."""

    chunk_id: str
    paper_id: str
    text: str
    page_number: int = 0


@dataclass(frozen=True)
class GraphRelation:
    """One graph edge eligible for the targeted static-context section."""

    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    source_label: str = ""
    target_label: str = ""
    properties: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "source_label": self.source_label,
            "target_label": self.target_label,
            "properties": dict(self.properties),
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """All generator inputs other than the arm-specific static graph section."""

    query_id: str
    question: str
    source_ids: tuple[str, ...] = ()
    chunks: tuple[EvidenceChunk, ...] = ()
    relations: tuple[GraphRelation, ...] = ()
    dynamic_enrichment: str = ""
    retrieval_identity: str = ""
    evidence_note: str = ""

    @property
    def activation_positive(self) -> bool:
        return bool(self.relations)

    @property
    def non_target_fingerprint(self) -> str:
        """Hash every input held invariant by the ON/OFF comparison."""

        payload = {
            "query_id": self.query_id,
            "question": self.question,
            "source_ids": list(self.source_ids),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                }
                for chunk in self.chunks
            ],
            "dynamic_enrichment": self.dynamic_enrichment,
            "retrieval_identity": self.retrieval_identity,
        }
        return _sha256(_canonical(payload))

    @property
    def evidence_hash(self) -> str:
        payload = {
            "non_target_fingerprint": self.non_target_fingerprint,
            "relations": [relation.as_dict() for relation in self.relations],
            "evidence_note": self.evidence_note,
        }
        return _sha256(_canonical(payload))


@dataclass(frozen=True)
class PromptTrace:
    arm_id: str
    static_gate_enabled: bool
    available_relation_count: int
    selected_relation_ids: tuple[str, ...]
    activation_status: str
    activation_explanation: str
    non_target_fingerprint: str
    evidence_hash: str
    prompt_sha256: str
    token_ids_sha256: str | None = None
    token_count: int | None = None
    input_overflow: bool | None = None


@dataclass(frozen=True)
class PromptBuild:
    prompt: str
    trace: PromptTrace
    token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class EntropyTelemetry:
    """Validated telemetry summary; unavailable fields remain ``None``.

    H20/H50/H100 are means of the full predictive-distribution Shannon
    entropy over the first 20/50/100 generated tokens.  A selected-token
    log-probability is not a substitute for this quantity.
    """

    h20: float | None = None
    h50: float | None = None
    h100: float | None = None
    token_count: int = 0
    source: str = "unavailable"
    distribution_observations: int = 0


def entropy_windows(entropies: Sequence[float]) -> EntropyTelemetry:
    """Compute H20/H50/H100 from per-token full-distribution entropies."""

    values = tuple(float(value) for value in entropies)
    if any(value < 0 for value in values):
        raise ValueError("Predictive-distribution entropy cannot be negative")

    def mean_prefix(size: int) -> float | None:
        prefix = values[:size]
        return sum(prefix) / len(prefix) if prefix else None

    return EntropyTelemetry(
        h20=mean_prefix(20),
        h50=mean_prefix(50),
        h100=mean_prefix(100),
        token_count=len(values),
        source="full_predictive_distribution",
        distribution_observations=len(values),
    )


def _format_chunks(chunks: Iterable[EvidenceChunk]) -> str:
    rows = []
    for index, chunk in enumerate(chunks, start=1):
        rows.append(
            f"[{index}] chunk_id={chunk.chunk_id} page={chunk.page_number}\n{chunk.text}"
        )
    return "\n\n".join(rows)


def _format_relations(relations: Iterable[GraphRelation]) -> str:
    rows = []
    for index, relation in enumerate(relations, start=1):
        source = relation.source_id
        target = relation.target_id
        labels = ""
        if relation.source_label or relation.target_label:
            labels = f" ({relation.source_label or '?'} -> {relation.target_label or '?'})"
        rows.append(
            f"[{index}] {source} -[{relation.relation_type}]-> {target}{labels}"
        )
    return "\n".join(rows)


def _build_prompt_text(bundle: EvidenceBundle, static_relations: Sequence[GraphRelation]) -> str:
    text = _format_chunks(bundle.chunks)
    graph = _format_relations(static_relations)
    enrichment = bundle.dynamic_enrichment.strip()
    return (
        "<|im_start|>system\n"
        "You are a biomedical research assistant. Synthesize an answer using "
        "the retrieved text and the separately labeled knowledge-graph sections.\n"
        "Cite the supplied evidence when relevant and do not invent citations.\n\n"
        "### RELEVANT TEXT FRAGMENTS:\n"
        f"{text}\n\n"
        "### STATIC GRAPH RELATIONS (R1 TARGET):\n"
        f"{graph}\n\n"
        "### DYNAMIC KNOWLEDGE GRAPH ENRICHMENT (HELD CONSTANT):\n"
        f"{enrichment}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: {bundle.question}\n"
        "Answer in English:\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_prompt(
    bundle: EvidenceBundle,
    arm_id: str,
    *,
    token_ids: Sequence[int] | None = None,
    input_token_limit: int = INPUT_TOKEN_LIMIT,
) -> PromptBuild:
    """Build one generator-visible prompt and an auditable intervention trace."""

    if arm_id not in {ARM_ON, ARM_OFF}:
        raise ValueError(f"Unknown R1 arm: {arm_id}")
    enabled = arm_id == ARM_ON
    selected = bundle.relations if enabled else ()
    prompt = _build_prompt_text(bundle, selected)
    if bundle.activation_positive:
        explanation = (
            "Static graph relations are available; ON inserts the selected relation "
            "section and OFF removes only that section."
            if enabled
            else "Static graph relations are available; OFF removes the targeted section."
        )
        activation_status = "positive"
    else:
        activation_status = "negative"
        explanation = (
            "No relevant static graph relations were available for this query; "
            "identical ON/OFF prompts are expected."
        )

    token_hash: str | None = None
    token_count: int | None = None
    overflow: bool | None = None
    normalized_tokens: tuple[int, ...] | None = None
    if token_ids is not None:
        normalized_tokens = tuple(int(item) for item in token_ids)
        token_hash = _sha256(_canonical(list(normalized_tokens)))
        token_count = len(normalized_tokens)
        overflow = token_count > input_token_limit

    trace = PromptTrace(
        arm_id=arm_id,
        static_gate_enabled=enabled,
        available_relation_count=len(bundle.relations),
        selected_relation_ids=tuple(relation.relation_id for relation in selected),
        activation_status=activation_status,
        activation_explanation=explanation,
        non_target_fingerprint=bundle.non_target_fingerprint,
        evidence_hash=bundle.evidence_hash,
        prompt_sha256=_sha256(prompt.encode("utf-8")),
        token_ids_sha256=token_hash,
        token_count=token_count,
        input_overflow=overflow,
    )
    return PromptBuild(prompt=prompt, trace=trace, token_ids=normalized_tokens)


def attach_token_ids(build: PromptBuild, token_ids: Sequence[int], *, input_token_limit: int = INPUT_TOKEN_LIMIT) -> PromptBuild:
    """Attach tokenizer output without changing the already-frozen prompt text."""

    normalized = tuple(int(item) for item in token_ids)
    trace = replace(
        build.trace,
        token_ids_sha256=_sha256(_canonical(list(normalized))),
        token_count=len(normalized),
        input_overflow=len(normalized) > input_token_limit,
    )
    return PromptBuild(prompt=build.prompt, trace=trace, token_ids=normalized)
