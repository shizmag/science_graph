"""CPU-only prompt census for the corrected R1 MLX experiment.

This module intentionally has no import path through ``runtime`` and no MLX
dependency.  It freezes the exact generator-visible ON/OFF prompts using the
same read-only evidence store and prompt builder used by the experiment, then
records enough compact identity to audit the full 375-query cohort before any
model execution is authorized.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path
from typing import Any

from ablation_portability.dataset.loader import load_frozen_dataset

from .evidence import ReadOnlyEvidenceStore
from .identity import (
    DATABASE_SHA256,
    DATABASE_SNAPSHOT_ID,
    DATASET_SHA256,
    canonical_json,
    file_sha256,
    object_sha256,
)
from .protocol import ARM_OFF, ARM_ON, INPUT_TOKEN_LIMIT, build_prompt, attach_token_ids


CENSUS_SCHEMA_VERSION = "r1-mlx-prompt-census-v1"
EXPECTED_ROWS = 375
EXPECTED_ANSWERABLE = 300
EXPECTED_DESIGNATED_NEGATIVE = 75


def validate_census_output_path(path: Path) -> Path:
    """Resolve a report path and refuse the two protected runtime namespaces."""

    resolved = Path(path).expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[2]
    protected_roots = (
        repository_root / "revision_round1/experiments/r1_mlx_corrected",
        repository_root / "revision_round1/experiments/r1_mlx_corrected_runtime_2026-11-10",
    )
    for protected in protected_roots:
        protected = protected.resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise ValueError(f"Census output path is inside a protected experiment root: {resolved}")
    return resolved


def _historical_overflow(record: Mapping[str, Any]) -> bool | None:
    """Use the same historical metadata convention as cohort preparation."""

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


def _tokenizer_ids(tokenizer: Any, prompt: str) -> tuple[int, ...]:
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    return tuple(int(value) for value in ids)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _distribution(values: Iterable[int | float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": math.fsum(numbers) / len(numbers),
        "p50": _percentile(numbers, 0.50),
        "p90": _percentile(numbers, 0.90),
        "p95": _percentile(numbers, 0.95),
        "p99": _percentile(numbers, 0.99),
    }


def _arm_payload(build: Any, token_ids: Sequence[int]) -> dict[str, Any]:
    attached = attach_token_ids(build, token_ids, input_token_limit=INPUT_TOKEN_LIMIT)
    trace = attached.trace
    return {
        "prompt_sha256": trace.prompt_sha256,
        "token_ids_sha256": trace.token_ids_sha256,
        "token_count": trace.token_count,
        "token_ids": list(attached.token_ids or ()),
        "input_overflow": trace.input_overflow,
        "prompt_char_count": len(attached.prompt),
        "selected_relation_count": len(trace.selected_relation_ids),
        "selected_relation_ids": list(trace.selected_relation_ids),
    }


def _subset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    on = [row["arms"]["on"] for row in rows]
    off = [row["arms"]["off"] for row in rows]
    deltas = [int(row["static_token_contribution"]) for row in rows]
    return {
        "count": len(rows),
        "on_token_count": _distribution(item["token_count"] for item in on),
        "off_token_count": _distribution(item["token_count"] for item in off),
        "static_token_contribution": _distribution(deltas),
        "prompt_char_count": {
            "on": _distribution(item["prompt_char_count"] for item in on),
            "off": _distribution(item["prompt_char_count"] for item in off),
        },
        "input_overflow": {
            "on": sum(bool(item["input_overflow"]) for item in on),
            "off": sum(bool(item["input_overflow"]) for item in off),
            "either": sum(bool(a["input_overflow"] or b["input_overflow"]) for a, b in zip(on, off)),
            "both": sum(bool(a["input_overflow"] and b["input_overflow"]) for a, b in zip(on, off)),
        },
        "historical_overflow": {
            "true": sum(row["historical_overflow"] is True for row in rows),
            "false": sum(row["historical_overflow"] is False for row in rows),
            "unknown": sum(row["historical_overflow"] is None for row in rows),
        },
    }


def _assemble_report(
    records: Sequence[Mapping[str, Any]],
    evidence: ReadOnlyEvidenceStore,
    tokenizer: Any,
    *,
    tokenizer_path: Path,
    dataset_path: Path,
    database_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        bundle = evidence.evidence_for_record(record)
        on_base = build_prompt(bundle, ARM_ON)
        off_base = build_prompt(bundle, ARM_OFF)
        on_ids = _tokenizer_ids(tokenizer, on_base.prompt)
        off_ids = _tokenizer_ids(tokenizer, off_base.prompt)
        on = _arm_payload(on_base, on_ids)
        off = _arm_payload(off_base, off_ids)
        activation_positive = bool(bundle.relations)
        non_target_invariant = (
            on_base.trace.non_target_fingerprint == off_base.trace.non_target_fingerprint
        )
        prompt_equal = on_base.prompt == off_base.prompt
        if not non_target_invariant:
            raise ValueError(f"Non-target evidence differs across arms for {bundle.query_id}")
        if activation_positive == prompt_equal:
            raise ValueError(
                f"ON/OFF intervention isolation failed for {bundle.query_id}: "
                f"activation_positive={activation_positive}, prompt_equal={prompt_equal}"
            )
        rows.append(
            {
                "query_id": bundle.query_id,
                "row_index": record.get("row_index"),
                "question": bundle.question,
                "answerability_label": record.get("answerability_label"),
                "historical_category": record.get("historical_category"),
                "synthetic_negative_category": record.get("synthetic_negative_category"),
                "source_pmid": record.get("source_pmid"),
                "source_ids": list(bundle.source_ids),
                "relation_count": len(bundle.relations),
                "relation_types": sorted({relation.relation_type for relation in bundle.relations}),
                "relation_ids": [relation.relation_id for relation in bundle.relations],
                "activation_positive": activation_positive,
                "historical_overflow": _historical_overflow(record),
                "non_target_invariant": non_target_invariant,
                "prompt_equal": prompt_equal,
                "static_token_contribution": int(on["token_count"]) - int(off["token_count"]),
                "static_prompt_char_contribution": int(on["prompt_char_count"]) - int(off["prompt_char_count"]),
                "arms": {"on": on, "off": off},
            }
        )

    rows.sort(key=lambda row: str(row["query_id"]))
    if len(rows) != EXPECTED_ROWS or len({row["query_id"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} unique frozen census rows; found {len(rows)}")
    answerable = [row for row in rows if row["answerability_label"] == "answerable"]
    designated_negative = [
        row for row in rows if row["answerability_label"] == "designated_negative"
    ]
    activation_positive = [row for row in rows if row["activation_positive"]]
    activation_negative = [row for row in rows if not row["activation_positive"]]
    if (len(answerable), len(designated_negative)) != (
        EXPECTED_ANSWERABLE,
        EXPECTED_DESIGNATED_NEGATIVE,
    ):
        raise ValueError(
            "Unexpected frozen answerability counts: "
            f"{len(answerable)} answerable, {len(designated_negative)} designated negative"
        )
    if (len(activation_positive), len(activation_negative)) != (347, 28):
        raise ValueError(
            "Unexpected static activation counts: "
            f"{len(activation_positive)} positive, {len(activation_negative)} negative"
        )

    subsets = {
        "all": rows,
        "answerable": answerable,
        "designated_negative": designated_negative,
        "activation_positive": activation_positive,
        "activation_negative": activation_negative,
        "historical_overflow": [row for row in rows if row["historical_overflow"] is True],
    }
    report: dict[str, Any] = {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "purpose": "offline_generator_visible_prompt_census; no_model_inference",
        "dataset": {
            "path": str(dataset_path),
            "sha256": file_sha256(dataset_path / "questions.jsonl"),
            "rows": len(rows),
        },
        "database": {
            "path": str(database_path),
            "snapshot_id": evidence.snapshot_id,
            "sha256": evidence.expected_sha256,
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": file_sha256(tokenizer_path),
            "special_tokens": "add_special_tokens=False",
        },
        "input_token_limit": INPUT_TOKEN_LIMIT,
        "static_relation_policy": sorted(evidence.STATIC_RELATION_TYPES),
        "excluded_metadata_relation_types": ["HAS_TAG"],
        "counts": {
            "rows": len(rows),
            "answerable": len(answerable),
            "designated_negative": len(designated_negative),
            "activation_positive": len(activation_positive),
            "activation_negative": len(activation_negative),
            "historical_overflow_true": sum(row["historical_overflow"] is True for row in rows),
            "historical_overflow_false": sum(row["historical_overflow"] is False for row in rows),
            "historical_overflow_unknown": sum(row["historical_overflow"] is None for row in rows),
        },
        "invariants": {
            "non_target_invariant_rows": sum(row["non_target_invariant"] for row in rows),
            "positive_rows_with_different_prompts": sum(
                row["activation_positive"] and not row["prompt_equal"] for row in rows
            ),
            "negative_rows_with_equal_prompts": sum(
                not row["activation_positive"] and row["prompt_equal"] for row in rows
            ),
            "max_input_token_limit": INPUT_TOKEN_LIMIT,
        },
        "distributions": {name: _subset_summary(values) for name, values in subsets.items()},
        "rows": rows,
    }
    report["report_sha256"] = object_sha256(report)
    return report


def build_census_report(
    *,
    dataset_path: Path,
    database_path: Path,
    tokenizer_path: Path,
    expected_dataset_sha256: str = DATASET_SHA256,
    expected_database_sha256: str = DATABASE_SHA256,
    snapshot_id: str = DATABASE_SNAPSHOT_ID,
) -> dict[str, Any]:
    """Build the complete frozen census using CPU tokenization only."""

    dataset_path = Path(dataset_path).expanduser().resolve()
    database_path = Path(database_path).expanduser().resolve()
    tokenizer_path = Path(tokenizer_path).expanduser().resolve()
    records = load_frozen_dataset(dataset_path)
    actual_dataset_sha256 = file_sha256(dataset_path / "questions.jsonl")
    if actual_dataset_sha256 != expected_dataset_sha256:
        raise ValueError("Frozen dataset SHA-256 does not match the expected identity")
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} frozen records; found {len(records)}")
    evidence = ReadOnlyEvidenceStore(
        database_path,
        expected_sha256=expected_database_sha256,
        snapshot_id=snapshot_id,
    )
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The pinned tokenizers package is required for the exact CPU census"
        ) from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return _assemble_report(
        records,
        evidence,
        tokenizer,
        tokenizer_path=tokenizer_path,
        dataset_path=dataset_path,
        database_path=database_path,
    )


def write_census_report(path: Path, report: Mapping[str, Any]) -> str:
    """Atomically write a caller-selected derived report path."""

    path = validate_census_output_path(path)
    payload = dict(report)
    expected = payload.get("report_sha256")
    body = dict(payload)
    body.pop("report_sha256", None)
    actual = object_sha256(body)
    if expected not in (None, actual):
        raise ValueError("Census report hash is inconsistent")
    payload["report_sha256"] = actual
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing census report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical_json(payload) + b"\n")
        # A hard-link publication is atomic and, unlike replace(), cannot
        # overwrite a report another process created after the existence check.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return actual
