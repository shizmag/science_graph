"""Sequential, idempotent execution for the corrected R1 paired protocol."""

from __future__ import annotations

from dataclasses import asdict
from functools import wraps
import time
from typing import Any, Callable, Mapping, Sequence

from .checkpoint import CheckpointError, DurableResultStore
from .identity import query_input_identity
from .protocol import (
    ARM_OFF,
    ARM_ON,
    INPUT_TOKEN_LIMIT,
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    EvidenceBundle,
    attach_token_ids,
    build_prompt,
)


class RunError(RuntimeError):
    """Raised when the frozen run cannot safely continue."""


def _serialized_run(function):
    """Prevent concurrent processes from reconciling each other's attempts."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        store = kwargs.get("store")
        if store is None:
            raise RunError("A checkpoint store is required")
        with store.execution_session():
            return function(*args, **kwargs)

    return wrapped


def _token_ids(tokenizer: Any, prompt: str) -> list[int]:
    # HuggingFace tokenizers are callable, while mlx-lm's TokenizerWrapper
    # exposes the underlying tokenizer's ``encode`` method but is not itself
    # callable.  Keep the fake/HF path used by the CPU tests and support the
    # actual MLX adapter without changing the token identity protocol.
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
    elif callable(tokenizer):
        encoded = tokenizer(prompt, add_special_tokens=False)
    else:
        raise RunError("Tokenizer is neither callable nor exposes encode()")
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise RunError("Tokenizer output has no input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    try:
        return [int(value) for value in encoded]
    except (TypeError, ValueError) as exc:
        raise RunError("Tokenizer returned an invalid input_ids sequence") from exc


def _adapter_ready(adapter: Any) -> None:
    if getattr(adapter, "model", None) is None or getattr(adapter, "tokenizer", None) is None:
        load = getattr(adapter, "load", None)
        if not callable(load):
            raise RunError("The MLX adapter has no loaded tokenizer/model")
        load()
    if getattr(adapter, "tokenizer", None) is None:
        raise RunError("The MLX adapter did not expose a tokenizer after load")
    if not callable(getattr(adapter, "generate", None)):
        raise RunError("The MLX adapter has no generate method")


def _counts(
    *,
    generated: int,
    skipped: int,
    terminal: int,
    failed: int,
    stopped: int = 0,
    budget_exhausted: int = 0,
) -> dict[str, int]:
    return {
        "generated": generated,
        "skipped": skipped,
        "terminal": terminal,
        "failed": failed,
        "stopped": stopped,
        "budget_exhausted": budget_exhausted,
    }


def _mark_raced_attempt(
    store: DurableResultStore,
    attempt_id: str,
    *,
    error_message: str,
) -> None:
    """Close a losing attempt without writing a failure for the operation.

    A concurrent process may have accepted the same query/arm after this
    attempt started.  The accepted result is authoritative; this attempt is
    merely recorded as interrupted provenance.
    """
    try:
        store.finish_attempt(
            attempt_id,
            status="INTERRUPTED",
            error_class="AcceptedByConcurrentAttempt",
            error_message=error_message,
        )
    except CheckpointError:
        # The same attempt may already be SUCCESS or terminal if the exception
        # happened after its durable commit.  Never overwrite that state.
        pass


@_serialized_run
def run_pairs(
    rows: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, EvidenceBundle],
    *,
    store: DurableResultStore,
    adapter: Any,
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    decoding: Mapping[str, Any],
    manifest_hash: str,
    max_new_generations: int | None = None,
    stop_after: int | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    """Run missing ON/OFF operations and return durable progress counts.

    ``max_new_generations`` counts only calls to ``adapter.generate``.  An
    input overflow is a durable terminal outcome, not an inference.  A
    process restart first reconciles stale RUNNING attempts as INTERRUPTED;
    previously accepted SUCCESS results are then skipped and never regenerated.
    """
    if manifest_hash != store.manifest_hash:
        raise RunError("Runner manifest hash does not match the checkpoint store")
    store.recover_inflight()
    budget = max_new_generations if max_new_generations is not None else len(rows) * 2
    generated = skipped = terminal = failed = processed = 0

    for row in rows:
        query_id = str(row["query_id"])
        bundle = bundles.get(query_id)
        if bundle is None:
            raise RunError(f"Missing evidence bundle for {query_id}")
        for arm_id in (ARM_ON, ARM_OFF):
            if stop_after is not None and processed >= stop_after:
                return _counts(
                    generated=generated,
                    skipped=skipped,
                    terminal=terminal,
                    failed=failed,
                    stopped=1,
                )

            build = build_prompt(bundle, arm_id)
            # Keep model/tokenizer construction out of empty invocations.  The
            # CLI performs a stronger durable operation-state preflight for
            # all-accepted and over-budget resumes; this local lazy boundary
            # also avoids loading an adapter when ``rows`` is empty.
            _adapter_ready(adapter)
            token_ids = _token_ids(adapter.tokenizer, build.prompt)
            build = attach_token_ids(build, token_ids, input_token_limit=INPUT_TOKEN_LIMIT)
            identity = query_input_identity(
                manifest_hash=manifest_hash,
                query_id=query_id,
                question=bundle.question,
                evidence_hash=bundle.evidence_hash,
                arm_id=arm_id,
                prompt_sha256=build.trace.prompt_sha256,
                token_ids=token_ids,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                decoding=dict(decoding),
            )

            existing = store.accepted(query_id, arm_id, identity)
            if existing is not None:
                skipped += 1
                processed += 1
                if on_result is not None:
                    on_result(existing["result"])
                continue

            # Check the inference cap before opening a RUNNING attempt.  This
            # leaves a budget-deferred operation cleanly eligible for resume.
            if not build.trace.input_overflow and generated >= budget:
                return _counts(
                    generated=generated,
                    skipped=skipped,
                    terminal=terminal,
                    failed=failed,
                    budget_exhausted=1,
                )

            base = {
                "protocol_version": PROTOCOL_VERSION,
                "prompt_version": PROMPT_VERSION,
                "query_id": query_id,
                "arm_id": arm_id,
                "question": bundle.question,
                "prompt": build.prompt,
                "prompt_sha256": build.trace.prompt_sha256,
                "token_ids": token_ids,
                "token_ids_sha256": build.trace.token_ids_sha256,
                "input_token_count": build.trace.token_count,
                "input_overflow": build.trace.input_overflow,
                "evidence_hash": bundle.evidence_hash,
                "non_target_fingerprint": bundle.non_target_fingerprint,
                "input_identity": identity,
                "model_identity": dict(model_identity),
                "tokenizer_identity": dict(tokenizer_identity),
                "decoding": dict(decoding),
            }

            try:
                attempt_id = store.begin_attempt(query_id, arm_id, identity)
            except CheckpointError:
                # A different process may have accepted between the first
                # read and begin.  A verified accepted result wins; any other
                # checkpoint error is a real identity/lifecycle failure.
                raced = store.accepted(query_id, arm_id, identity)
                if raced is None:
                    raise
                skipped += 1
                processed += 1
                if on_result is not None:
                    on_result(raced["result"])
                continue

            base["attempt_id"] = attempt_id

            if build.trace.input_overflow:
                result = dict(base)
                result.update({"status": "OVERFLOW", "finish_reason": "input_context_limit"})
                store.record_terminal(
                    attempt_id=attempt_id,
                    query_id=query_id,
                    arm_id=arm_id,
                    input_identity=identity,
                    status="OVERFLOW",
                    result=result,
                )
                terminal += 1
                processed += 1
                if on_result is not None:
                    on_result(result)
                continue

            started = time.perf_counter()
            # Persist the budget marker before entering inference. A crash in
            # this narrow interval conservatively consumes one call.
            store.mark_generation_started(attempt_id)
            generated += 1  # the call itself is part of provenance even if it raises
            try:
                generated_result = adapter.generate(build.prompt)
                if isinstance(generated_result, Mapping):
                    result = dict(generated_result)
                else:
                    result = {"text": str(generated_result)}
                result = {**base, **result}
                result.setdefault("status", "SUCCESS")
                result.setdefault(
                    "duration_seconds", float(time.perf_counter() - started)
                )
                if result["status"] == "SUCCESS":
                    try:
                        store.commit_success(
                            attempt_id=attempt_id,
                            query_id=query_id,
                            arm_id=arm_id,
                            input_identity=identity,
                            result=result,
                        )
                    except CheckpointError:
                        raced = store.accepted(query_id, arm_id, identity)
                        if raced is None:
                            raise
                        _mark_raced_attempt(
                            store,
                            attempt_id,
                            error_message="A concurrent attempt committed the accepted result",
                        )
                        skipped += 1
                        processed += 1
                        if on_result is not None:
                            on_result(raced["result"])
                        continue
                else:
                    status = str(result["status"])
                    if status not in {"OVERFLOW", "FAILED_RETRYABLE", "FAILED_TERMINAL", "INTERRUPTED"}:
                        status = "FAILED_RETRYABLE"
                    store.record_terminal(
                        attempt_id=attempt_id,
                        query_id=query_id,
                        arm_id=arm_id,
                        input_identity=identity,
                        status=status,
                        result=result,
                        error_class=result.get("error_class"),
                        error_message=result.get("error_message"),
                    )
                    failed += 1
                processed += 1
                if on_result is not None:
                    on_result(result)
            except Exception as exc:
                # A commit may have completed before an exception was raised
                # by the adapter/caller boundary.  Never append a failure to an
                # already accepted operation.
                raced = store.accepted(query_id, arm_id, identity)
                if raced is not None:
                    _mark_raced_attempt(
                        store,
                        attempt_id,
                        error_message="Accepted result found while reconciling an exception",
                    )
                    skipped += 1
                    processed += 1
                    if on_result is not None:
                        on_result(raced["result"])
                    continue
                failure = {
                    **base,
                    "status": "FAILED_RETRYABLE",
                    "error_class": type(exc).__name__,
                    "error_message": str(exc),
                    "duration_seconds": float(time.perf_counter() - started),
                }
                try:
                    store.record_terminal(
                        attempt_id=attempt_id,
                        query_id=query_id,
                        arm_id=arm_id,
                        input_identity=identity,
                        status="FAILED_RETRYABLE",
                        result=failure,
                        error_class=type(exc).__name__,
                        error_message=str(exc),
                    )
                except CheckpointError as finalize_error:
                    raise RunError(
                        f"Could not durably finalize {query_id}/{arm_id}"
                    ) from finalize_error
                failed += 1
                processed += 1
                if on_result is not None:
                    on_result(failure)

    return _counts(
        generated=generated,
        skipped=skipped,
        terminal=terminal,
        failed=failed,
    )
