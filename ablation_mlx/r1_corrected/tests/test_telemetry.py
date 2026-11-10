"""Network-free tests for predictive-distribution telemetry and the adapter seam."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import math

from revision_replication.entropy import topk_telemetry

from ablation_mlx.r1_corrected import runtime, telemetry
from ablation_mlx.r1_corrected.census import validate_census_output_path, write_census_report
from ablation_mlx.r1_corrected.cli import _telemetry_manifest_contract
from ablation_mlx.r1_corrected.identity import object_sha256
from ablation_mlx.r1_corrected.telemetry import (
    TokenTelemetryAccumulator,
    TelemetryCompatibilityError,
    reduce_token_distribution,
    validate_telemetry_payload,
)


class TelemetryMathTests(unittest.TestCase):
    def test_matches_historical_raw_logit_reference(self) -> None:
        values = [-2.0, 0.0, 3.5, -1.0, 1.0]
        actual = reduce_token_distribution(values, 2, requested_ks=(2, 4, 100))
        expected = topk_telemetry(values, 2, ks=(2, 4, 100))
        for key in (
            "token_id",
            "logprob",
            "nll_nats",
            "nll_bits_per_token",
            "entropy_k2_bits",
            "retained_mass_k2",
            "entropy_k4_bits",
            "retained_mass_k4",
            "entropy_k100_bits",
            "retained_mass_k100",
        ):
            self.assertAlmostEqual(actual[key], expected[key], places=12, msg=key)

    def test_uniform_distribution_has_expected_top_k_entropy_and_mass(self) -> None:
        actual = reduce_token_distribution([0.0] * 101, 7)
        self.assertAlmostEqual(actual["entropy_k20_bits"], math.log2(20), places=12)
        self.assertAlmostEqual(actual["entropy_k50_bits"], math.log2(50), places=12)
        self.assertAlmostEqual(actual["entropy_k100_bits"], math.log2(100), places=12)
        self.assertAlmostEqual(actual["retained_mass_k20"], 20 / 101, places=12)
        self.assertAlmostEqual(actual["retained_mass_k50"], 50 / 101, places=12)
        self.assertAlmostEqual(actual["retained_mass_k100"], 100 / 101, places=12)
        self.assertAlmostEqual(actual["top1_prob_full"], 1 / 101, places=12)
        self.assertEqual(actual["logit_margin"], 0.0)

    def test_peaked_and_near_deterministic_distributions_are_finite(self) -> None:
        for values in ([0.0, -20.0, -20.0], [1000.0, -1000.0, -1000.0]):
            actual = reduce_token_distribution(values, 0)
            for key, value in actual.items():
                if key != "token_id" and value is not None:
                    self.assertTrue(float(value) == float(value), key)
            self.assertGreaterEqual(actual["retained_mass_k20"], 0.0)
            self.assertLessEqual(actual["retained_mass_k20"], 1.0)
            self.assertLess(actual["entropy_k20_bits"], 1e-6)

    def test_additive_logprob_shift_and_top_k_truncation(self) -> None:
        first = reduce_token_distribution([0.0, -1.0, -2.0], 1, requested_ks=(2, 5))
        shifted = reduce_token_distribution([100.0, 99.0, 98.0], 1, requested_ks=(2, 5))
        for key in first:
            if key != "token_id":
                self.assertAlmostEqual(first[key], shifted[key], places=12, msg=key)
        self.assertAlmostEqual(first["retained_mass_k5"], 1.0, places=12)
        self.assertGreater(first["entropy_k5_bits"], first["entropy_k2_bits"])

    def test_one_element_vocabulary_has_no_margin_and_validates(self) -> None:
        accumulator = TokenTelemetryAccumulator()
        accumulator.add([0.0], 0, stop_state="eos")
        payload = accumulator.finalize(finish_reason="eos", stop_token_id=151643)
        self.assertIsNone(payload["mean_logit_margin"])
        self.assertIsNone(payload["tokens"][0]["logit_margin"])
        validate_telemetry_payload(payload)

    def test_empty_invalid_and_dense_payload_rejection(self) -> None:
        payload = TokenTelemetryAccumulator().finalize(finish_reason="empty")
        validate_telemetry_payload(payload)
        dense = dict(payload)
        dense["extension"] = {"logprobs": [0.0, -1.0]}
        with self.assertRaises(TelemetryCompatibilityError):
            validate_telemetry_payload(dense)
        with self.assertRaises(ValueError):
            reduce_token_distribution([], 0)
        with self.assertRaises(ValueError):
            reduce_token_distribution([float("nan")], 0)

    def test_accumulator_means_and_json_persistence(self) -> None:
        accumulator = TokenTelemetryAccumulator()
        accumulator.add([0.0, -1.0, -2.0], 0, stop_state="continue")
        accumulator.add([-1.0, 0.0, -2.0], 1, stop_state="eos")
        payload = accumulator.finalize(finish_reason="eos", stop_token_id=151643)
        validate_telemetry_payload(payload)
        self.assertEqual(payload["token_count"], 2)
        self.assertAlmostEqual(
            payload["mean_nll_nats"],
            sum(row["nll_nats"] for row in payload["tokens"]) / 2,
            places=12,
        )
        restored = json.loads(json.dumps(payload))
        validate_telemetry_payload(restored)


@contextmanager
def _fake_mlx_modules(generate_step):
    mlx = types.ModuleType("mlx")
    mlx.__path__ = []
    core = types.ModuleType("mlx.core")
    core.array = lambda values: tuple(values)
    mlx.core = core
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.__path__ = []
    generate = types.ModuleType("mlx_lm.generate")
    generate.generate_step = generate_step
    mlx_lm.generate = generate
    with patch.dict(
        sys.modules,
        {"mlx": mlx, "mlx.core": core, "mlx_lm": mlx_lm, "mlx_lm.generate": generate},
    ):
        yield


class _FakeRuntimeTokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool = False):
        del prompt, add_special_tokens
        return [11, 12]

    def decode(self, token_ids):
        return "".join(f"<{int(token_id)}>" for token_id in token_ids)


class AdapterTelemetryTests(unittest.TestCase):
    def _adapter(self) -> runtime.MLXAdapter:
        adapter = runtime.MLXAdapter(Path("/unused/model"), output_limit=8)
        adapter.load = lambda: None  # type: ignore[method-assign]
        adapter.model = object()
        adapter.tokenizer = _FakeRuntimeTokenizer()
        return adapter

    def test_first_distribution_aligns_with_first_generated_token_and_eos_is_compact_only(self) -> None:
        seen: dict[str, object] = {}

        def fake_generate_step(prompt, model, *, max_tokens):
            seen["prompt"] = prompt
            seen["model"] = model
            seen["max_tokens"] = max_tokens
            yield 1, [0.0, -1.0, -2.0]
            yield 2, [-1.0, 0.0, -2.0]

        adapter = self._adapter()
        with patch.object(runtime, "STOP_TOKEN_IDS", (2,)), patch.object(
            telemetry, "STOP_TOKEN_IDS", (2,)
        ), _fake_mlx_modules(fake_generate_step):
            result = adapter.generate("prompt", max_tokens=8)

        self.assertEqual(seen["prompt"], (11, 12))
        self.assertEqual(seen["max_tokens"], 8)
        self.assertEqual(result["output_token_ids"], [1])
        self.assertEqual(result["output_token_count"], 1)
        self.assertEqual(result["generation_token_count"], 2)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["telemetry"]["token_count"], 2)
        self.assertEqual(result["telemetry"]["tokens"][1]["stop_state"], "eos")
        expected = reduce_token_distribution([0.0, -1.0, -2.0], 1)
        self.assertAlmostEqual(result["telemetry"]["tokens"][0]["logprob"], expected["logprob"])

    def test_length_and_empty_generation_preserve_compact_protocol(self) -> None:
        def fake_length(prompt, model, *, max_tokens):
            yield 0, [0.0, -1.0]
            yield 1, [-1.0, 0.0]

        def fake_empty(prompt, model, *, max_tokens):
            if False:
                yield 0, [0.0]

        adapter = self._adapter()
        with patch.object(runtime, "STOP_TOKEN_IDS", (99,)), patch.object(
            telemetry, "STOP_TOKEN_IDS", (99,)
        ), _fake_mlx_modules(fake_length):
            result = adapter.generate("prompt", max_tokens=2)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["telemetry"]["tokens"][1]["stop_state"], "max_tokens")
        validate_telemetry_payload(result["telemetry"])

        with _fake_mlx_modules(fake_empty):
            empty = adapter.generate("prompt", max_tokens=2)
        self.assertEqual(empty["finish_reason"], "empty")
        self.assertFalse(empty["telemetry"]["valid"])
        validate_telemetry_payload(empty["telemetry"])

    def test_generation_limits_reject_non_integer_without_loading(self) -> None:
        adapter = runtime.MLXAdapter(Path("/unused/model"), output_limit=8)
        adapter.load = lambda: self.fail("invalid generation options must not load")  # type: ignore[method-assign]
        for value in (True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.generate("prompt", max_tokens=value)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            runtime.MLXAdapter(Path("/unused/model"), output_limit=True)  # type: ignore[arg-type]


class CensusTests(unittest.TestCase):
    def test_audited_full_census_artifact_hash_and_counts(self) -> None:
        report_path = (
            Path(__file__).resolve().parents[3]
            / "revision_round1/derived/r1_mlx_prompt_census_2026-11-10.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_hash = report.pop("report_sha256")
        self.assertEqual(object_sha256(report), report_hash)
        self.assertEqual(report["counts"]["rows"], 375)
        self.assertEqual(report["counts"]["answerable"], 300)
        self.assertEqual(report["counts"]["designated_negative"], 75)
        self.assertEqual(report["counts"]["activation_positive"], 347)
        self.assertEqual(report["counts"]["activation_negative"], 28)
        self.assertEqual(report["invariants"]["non_target_invariant_rows"], 375)
        self.assertEqual(report["distributions"]["all"]["input_overflow"]["either"], 0)
        self.assertEqual(report["counts"]["historical_overflow_unknown"], 375)

    def test_census_module_import_does_not_import_runtime(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import ablation_mlx.r1_corrected.census; "
                "print('ablation_mlx.r1_corrected.runtime' in sys.modules)",
            ],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(probe.stdout.strip(), "False")

    def test_census_output_path_guard(self) -> None:
        root = Path(__file__).resolve().parents[3]
        for protected in (
            root / "revision_round1/experiments/r1_mlx_corrected",
            root / "revision_round1/experiments/r1_mlx_corrected_runtime_2026-11-10",
        ):
            with self.subTest(protected=protected), self.assertRaises(ValueError):
                validate_census_output_path(protected / "new-report.json")
        safe = root / "revision_round1/derived/future-report.json"
        self.assertEqual(validate_census_output_path(safe), safe.resolve())

    def test_census_writer_round_trip_and_refuses_bad_hash(self) -> None:
        report = {"schema_version": "test", "rows": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "census.json"
            digest = write_census_report(path, report)
            restored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(restored["report_sha256"], digest)
            with self.assertRaises(ValueError):
                write_census_report(path, {**report, "report_sha256": "wrong"})
            with self.assertRaises(FileExistsError):
                write_census_report(path, report)


class TelemetryManifestTests(unittest.TestCase):
    def test_new_manifest_telemetry_contract_is_explicit(self) -> None:
        contract = _telemetry_manifest_contract()
        self.assertEqual(contract["protocol_version"], telemetry.TELEMETRY_PROTOCOL_VERSION)
        self.assertEqual(contract["implementation"], telemetry.TELEMETRY_IMPLEMENTATION)
        self.assertEqual(contract["source"], telemetry.TELEMETRY_SOURCE)
        self.assertEqual(contract["top_k"], [20, 50, 100])
        self.assertEqual(contract["aggregation"], telemetry.TELEMETRY_AGGREGATION)
        self.assertTrue(contract["eos_included_in_aggregation"])


if __name__ == "__main__":
    unittest.main()
