"""Deterministic, outcome-blind cohort preparation for the corrected R1 run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ablation_cuda.runtime.checkpoints import atomic_write_json

from .evidence import ReadOnlyEvidenceStore
from .identity import object_sha256
from .protocol import (
    ARM_ON,
    COHORT_SEED,
    INPUT_TOKEN_LIMIT,
    OUTPUT_TOKEN_LIMIT,
    PROTOCOL_VERSION,
    PROMPT_VERSION,
    EvidenceBundle,
    build_prompt,
)


@dataclass(frozen=True)
class CohortRow:
    query_id: str
    question: str
    answerability_label: str
    historical_category: str
    source_pmid: str
    source_ids: tuple[str, ...]
    chunk_count: int
    context_chars: int
    relation_count: int
    activation_positive: bool
    activation_explanation: str
    historical_overflow: bool | None
    length_stratum: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_ids"] = list(self.source_ids)
        return value


def _stable_rank(seed: str, query_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).hexdigest()


def _length_strata(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted(rows, key=lambda row: (int(row["context_chars"]), row["query_id"]))
    if not ordered:
        return {}
    n = len(ordered)
    result: dict[str, str] = {}
    for index, row in enumerate(ordered):
        fraction = index / max(n - 1, 1)
        result[row["query_id"]] = "short" if fraction < 1 / 3 else "long" if fraction >= 2 / 3 else "median"
    return result


def _historical_overflow(record: Mapping[str, Any]) -> bool | None:
    for key in ("historical_overflow", "context_overflow", "overflow"):
        value = record.get(key)
        if isinstance(value, bool):
            return value
    metadata = record.get("source_metadata")
    if isinstance(metadata, Mapping):
        for key in ("historical_overflow", "context_overflow", "overflow"):
            value = metadata.get(key)
            if isinstance(value, bool):
                return value
    return None


def prepare_cohort(
    records: Iterable[Mapping[str, Any]],
    evidence_store: ReadOnlyEvidenceStore,
    *,
    seed: str = COHORT_SEED,
    input_token_limit: int = INPUT_TOKEN_LIMIT,
) -> tuple[list[CohortRow], dict[str, Any]]:
    """Build CPU-only diagnostics without inspecting generated quality.

    Token counts are deliberately character-based here because cohort
    preparation must not require MLX.  The runner replaces them with exact
    tokenizer counts before each generation and persists overflow explicitly.
    """
    records = list(records)
    diagnostics: list[dict[str, Any]] = []
    bundles: dict[str, EvidenceBundle] = {}
    for record in records:
        bundle = evidence_store.evidence_for_record(dict(record))
        bundles[bundle.query_id] = bundle
        prompt_on = build_prompt(bundle, ARM_ON, input_token_limit=input_token_limit)
        row = {
            "query_id": bundle.query_id,
            "question": bundle.question,
            "answerability_label": str(record.get("answerability_label", "")),
            "historical_category": str(record.get("historical_category", "")),
            "source_pmid": str(record.get("source_pmid", "")),
            "source_ids": bundle.source_ids,
            "chunk_count": len(bundle.chunks),
            "context_chars": len(prompt_on.prompt),
            "relation_count": len(bundle.relations),
            "activation_positive": bool(bundle.relations),
            "activation_explanation": prompt_on.trace.activation_explanation,
            "historical_overflow": _historical_overflow(record),
        }
        diagnostics.append(row)
    strata = _length_strata(diagnostics)
    rows = [
        CohortRow(length_stratum=strata[row["query_id"]], **row)
        for row in sorted(diagnostics, key=lambda row: row["query_id"])
    ]
    positive = [row for row in rows if row.activation_positive]
    negative = [row for row in rows if not row.activation_positive]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "cohort_seed": seed,
        "input_token_limit": input_token_limit,
        "output_token_limit": OUTPUT_TOKEN_LIMIT,
        "selection_is_outcome_blind": True,
        "eligible_count": len(rows),
        "activation_positive_count": len(positive),
        "activation_negative_count": len(negative),
        "activation_positive_query_ids": [row.query_id for row in positive],
        "activation_negative_query_ids": [row.query_id for row in negative],
        "rows": [row.as_dict() for row in rows],
    }
    manifest["manifest_hash"] = object_sha256(manifest)
    return rows, manifest


def select_smoke_rows(rows: Sequence[CohortRow], *, count: int = 6, seed: str = COHORT_SEED) -> list[CohortRow]:
    """Select six IDs deterministically before any new output is observed."""
    if count != 6:
        raise ValueError("The authorized smoke cohort is exactly six query IDs")
    positive = [row for row in rows if row.activation_positive]
    negative = [row for row in rows if not row.activation_positive]
    if len(positive) < 2 or not negative:
        raise ValueError("Frozen cohort cannot satisfy the positive/negative smoke requirements")
    by_rank = lambda row: _stable_rank(seed, row.query_id)
    selected: list[CohortRow] = []

    def take(pool: list[CohortRow], predicate) -> None:
        for row in sorted(pool, key=by_rank):
            if predicate(row) and row not in selected:
                selected.append(row)
                return

    take(positive, lambda row: row.length_stratum == "short")
    take(positive, lambda row: row.length_stratum == "long")
    take(negative, lambda row: True)
    for row in sorted(rows, key=by_rank):
        if row not in selected:
            selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Frozen cohort contains fewer than six eligible query IDs")
    return sorted(selected, key=lambda row: row.query_id)


def write_cohort_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    expected = payload.pop("manifest_hash", None)
    actual = object_sha256(payload)
    if expected is not None and expected != actual:
        raise ValueError("Cohort manifest hash is inconsistent")
    payload["manifest_hash"] = actual
    atomic_write_json(path, payload)
    written = json.loads(path.read_text(encoding="utf-8"))
    if written.get("manifest_hash") != actual or object_sha256(
        {key: value for key, value in written.items() if key != "manifest_hash"}
    ) != actual:
        raise ValueError("Cohort manifest changed during atomic write")
    return actual
