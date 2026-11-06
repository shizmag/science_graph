"""Safety and numerical tests for local paired-replication infrastructure.

All generation in this module is provided by a deterministic fake; no MLX
weights, real model adapter, or external judge is loaded or called.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from revision_replication.entropy import summarize_token_telemetry, topk_telemetry
from revision_replication.gemma_mlx import AUTHORIZATION, GemmaMLXAdapter
from revision_replication.provenance import ReplicationProvenance
from revision_replication.runner import PairedRunner


def valid_provenance() -> ReplicationProvenance:
    return ReplicationProvenance(
        query_registry_sha256="a" * 64,
        dataset_id="synthetic-test",
        dataset_sha256="b" * 64,
        corpus_id="synthetic-corpus",
        corpus_sha256="c" * 64,
        retrieval_config_sha256="d" * 64,
        context_origin="new_frozen_mlX_contexts",
        generator_id="gemma-3-text-4b-it",
        generator_revision="local-sha256:weight-file",
        tokenizer_revision="local-sha256:tokenizer-config",
        quantization="4bit_group64",
        prompt_template_sha256="e" * 64,
        generation_settings={"temperature": 0, "max_tokens": 16},
        entropy_definition="local_renormalized_topk_raw_pre_sampling_logprobs",
        evaluation_protocol_id="test-v1",
        paper_to_code={"B3": "B4", "B5": "B6"},
    )


def test_topk_entropy_and_retained_mass_use_correct_denominators():
    # Probabilities are exactly [0.5, 0.3, 0.2]; K=2 is locally renormalized.
    logits = [math.log(0.5), math.log(0.3), math.log(0.2)]
    got = topk_telemetry(logits, emitted_token_id=1, ks=(2, 3))
    p = [0.5 / 0.8, 0.3 / 0.8]
    expected_h2 = -sum(x * math.log2(x) for x in p)
    expected_h3 = -sum(x * math.log2(x) for x in (0.5, 0.3, 0.2))
    assert got["entropy_k2_bits"] == pytest.approx(expected_h2)
    assert got["entropy_k3_bits"] == pytest.approx(expected_h3)
    assert got["retained_mass_k2"] == pytest.approx(0.8)
    assert got["retained_mass_k3"] == pytest.approx(1.0)
    assert got["nll_nats"] == pytest.approx(-math.log(0.3))
    assert summarize_token_telemetry([got])["token_count"] == 1


def test_entropy_rejects_bad_logits():
    with pytest.raises(ValueError):
        topk_telemetry([], 0)
    with pytest.raises(ValueError):
        topk_telemetry([0.0, float("nan")], 0)
    with pytest.raises(ValueError):
        topk_telemetry([0.0], 1)


def test_gemma_adapter_authorization_precedes_any_model_load():
    loads = []
    adapter = GemmaMLXAdapter(loader=lambda path: loads.append(path))
    with pytest.raises(PermissionError):
        adapter.generate("test", authorization=None)
    assert loads == []


def test_fake_gemma_adapter_uses_template_and_collects_full_distribution():
    class Tokenizer:
        chat_template = "fake-template"
        eos_token_id = 99
        eos_token_ids = [99]

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is False and add_generation_prompt is True
            return "<user>hello<assistant>"

        def encode(self, text, *, add_special_tokens):
            assert text == "<user>hello<assistant>"
            assert add_special_tokens is False
            return [1, 2, 3]

    token = SimpleNamespace(token=1, text="a", logprobs=[math.log(0.7), math.log(0.2), math.log(0.1)])
    seen = {}

    def stream(**kwargs):
        seen.update(kwargs)
        yield token

    adapter = GemmaMLXAdapter(
        loader=lambda _path: (object(), Tokenizer()),
        streamer=stream,
        sampler_factory=lambda: "greedy",
        context_limit=20,
    )
    result = adapter.generate("hello", max_tokens=5, authorization=AUTHORIZATION)
    assert seen["sampler"] == "greedy"
    assert result["status"] == "success"
    assert result["token_ids"] == [1]
    assert result["token_count"] == 1
    assert result["token_telemetry"][0]["retained_mass_k100"] == pytest.approx(1)


def test_gemma_context_overflow_does_not_truncate_or_stream():
    class Tokenizer:
        chat_template = "fake"
        eos_token_id = 99
        def apply_chat_template(self, *_args, **_kwargs): return "long"
        def encode(self, *_args, **_kwargs): return [1, 2, 3, 4]

    streamed = []
    adapter = GemmaMLXAdapter(
        loader=lambda _path: (object(), Tokenizer()),
        streamer=lambda **_kwargs: streamed.append(True),
        context_limit=5,
    )
    result = adapter.generate("x", max_tokens=2, authorization=AUTHORIZATION)
    assert result["status"] == "context_overflow"
    assert streamed == []


class FakeGenerator:
    requires_real_execution_authorization = False

    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = outcomes or {}

    def generate(self, prompt, *, max_tokens, authorization=None):
        self.calls.append(prompt)
        if prompt in self.outcomes:
            raise self.outcomes[prompt]
        return {"status": "success", "text": f"answer:{prompt}", "token_ids": [1], "token_count": 1}


def rows_for(*query_ids: str) -> list[dict]:
    return [
        {"query_id": qid, "code_baseline": code, "prompt": f"{qid}:{code}", "evidence_ids": [f"ev-{qid}"], "evidence": [f"text-{qid}"]}
        for qid in query_ids
        for code in ("B4", "B6")
    ]


def test_plan_only_records_frozen_pair_without_calling_generator(tmp_path):
    generator = FakeGenerator()
    runner = PairedRunner(generator=generator, provenance=valid_provenance(), output_dir=tmp_path)
    result = runner.run(rows_for("q1", "q2"), cohort=["q1", "q2"], run_id="test", dry_run=True)
    assert result["planned_operations"] == 4
    assert generator.calls == []
    assert json.loads((tmp_path / "run_manifest.json").read_text())["execution_authorized"] is False


def test_fake_paired_run_resumes_idempotently_and_writes_final_manifest(tmp_path):
    rows = rows_for("q1")
    generator = FakeGenerator()
    runner = PairedRunner(generator=generator, provenance=valid_provenance(), output_dir=tmp_path)
    result = runner.run(rows, cohort=["q1"], run_id="test", dry_run=False)
    assert result["completed_operations"] == 2
    assert len(generator.calls) == 2
    first_manifest = (tmp_path / "results_manifest.json").read_bytes()
    result2 = runner.run(rows, cohort=["q1"], run_id="test", dry_run=False)
    assert result2["completed_operations"] == 2
    assert len(generator.calls) == 2
    assert (tmp_path / "results_manifest.json").read_bytes() == first_manifest
    records = [json.loads(line) for line in (tmp_path / "operations.jsonl").read_text().splitlines()]
    assert [r["status"] for r in records] == ["success", "success"]


def test_failure_is_checkpointed_and_counted_without_retry(tmp_path):
    rows = rows_for("q1")
    failed_prompt = "q1:B4"
    generator = FakeGenerator({failed_prompt: RuntimeError("synthetic error")})
    runner = PairedRunner(generator=generator, provenance=valid_provenance(), output_dir=tmp_path)
    result = runner.run(rows, cohort=["q1"], run_id="test", dry_run=False)
    assert result["completed_operations"] == 2
    records = [json.loads(line) for line in (tmp_path / "operations.jsonl").read_text().splitlines()]
    failed = next(r for r in records if r["query_id"] == "q1" and r["code_baseline"] == "B4")
    assert failed["status"] == "model_error"
    assert failed["error_type"] == "RuntimeError"
    assert (tmp_path / "results_manifest.json").exists()


def test_export_cuda_recovery_tables_preserves_missing_quality(tmp_path):
    from revision_replication.exports import export_cuda_recovery_tables

    source = tmp_path / "analysis.json"
    source.write_text(json.dumps({
        "arms": {"B6_full": {"availability": {"answerable_n": 2, "available": 1, "unavailable": 1, "rate": 0.5}, "failure_classifications_all_labeled_ids": {"success": 1, "context_overflow": 1}}},
        "judge_missingness": {"by_arm": {"B6_full": {"input_records": 2, "eligible_inputs": 1, "scored_records": 0, "execution_status": "not_run"}}},
        "paired_comparisons": {},
    }))
    assert export_cuda_recovery_tables(source, tmp_path / "csv") == {"arms": 1, "comparisons": 0}
    text = (tmp_path / "csv" / "cuda_arm_availability.csv").read_text()
    assert '""' in text or ",," in text
    assert "not_run" in text


def test_resume_skips_checkpointed_pair_and_runs_only_missing_operation(tmp_path):
    rows = rows_for("q1")
    runner = PairedRunner(generator=FakeGenerator(), provenance=valid_provenance(), output_dir=tmp_path)
    plan = runner.plan(rows, cohort=["q1"], run_id="interrupted")
    prompt = rows[0]["prompt"]
    record = {
        "run_id": "interrupted", "query_id": "q1", "code_baseline": "B4",
        "paper_baseline": "B3", "input_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "status": "success", "text": "already completed",
    }
    (tmp_path / "operations.jsonl").write_text(json.dumps(record) + "\n")
    result = runner.run(rows, cohort=["q1"], run_id="interrupted", dry_run=False)
    assert result["completed_operations"] == plan["planned_operations"] == 2
    assert runner.generator.calls == ["q1:B6"]
    assert len((tmp_path / "operations.jsonl").read_text().splitlines()) == 2


def test_runner_rejects_duplicate_query_baseline_rows(tmp_path):
    rows = rows_for("q1")
    rows.append(dict(rows[0]))
    runner = PairedRunner(generator=FakeGenerator(), provenance=valid_provenance(), output_dir=tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        runner.plan(rows, cohort=["q1"], run_id="duplicate")
