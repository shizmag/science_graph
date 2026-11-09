"""Corrected R1 structured-context ablation for the local MLX runtime."""

from .protocol import (
    ARM_OFF,
    ARM_ON,
    INPUT_TOKEN_LIMIT,
    OUTPUT_TOKEN_LIMIT,
    EvidenceBundle,
    EvidenceChunk,
    GraphRelation,
    PromptBuild,
    PromptTrace,
    attach_token_ids,
    build_prompt,
    EntropyTelemetry,
    entropy_windows,
)

__all__ = [
    "ARM_OFF",
    "ARM_ON",
    "INPUT_TOKEN_LIMIT",
    "OUTPUT_TOKEN_LIMIT",
    "EvidenceBundle",
    "EvidenceChunk",
    "GraphRelation",
    "PromptBuild",
    "PromptTrace",
    "attach_token_ids",
    "build_prompt",
    "EntropyTelemetry",
    "entropy_windows",
]
