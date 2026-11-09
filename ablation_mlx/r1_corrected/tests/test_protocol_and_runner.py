"""Network-free tests for the corrected R1 protocol and durable runner."""

from __future__ import annotations

import json
import importlib.util
from contextlib import closing
from pathlib import Path
import tempfile
import unittest

from ablation_mlx.r1_corrected.checkpoint import CheckpointError, DurableResultStore
from ablation_mlx.r1_corrected.identity import write_manifest
from ablation_mlx.r1_corrected.protocol import (
    ARM_OFF,
    ARM_ON,
    EvidenceBundle,
    EvidenceChunk,
    GraphRelation,
    INPUT_TOKEN_LIMIT,
    build_prompt,
)
from ablation_mlx.r1_corrected.runner import run_pairs


class FakeTokenizer:
    def __call__(self, prompt: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": list(range(max(1, len(prompt) // 20)))}


class FakeAdapter:
    def __init__(self, *, failures: int = 0):
        self.model = object()
        self.tokenizer = FakeTokenizer()
        self.failures = failures
        self.calls: list[str] = []

    def load(self) -> None:
        return None

    def generate(self, prompt: str) -> dict[str, object]:
        self.calls.append(prompt)
        if len(self.calls) <= self.failures:
            raise RuntimeError("synthetic generation failure")
        return {
            "status": "SUCCESS",
            "text": "synthetic answer",
            "output_token_count": 2,
            "finish_reason": "synthetic",
        }


def _bundle(query_id: str = "q1") -> EvidenceBundle:
    return EvidenceBundle(
        query_id=query_id,
        question="What is the synthetic answer?",
        source_ids=("source-1",),
        chunks=(EvidenceChunk("c1", "source-1", "A deterministic chunk."),),
        relations=(
            GraphRelation("r1", "source-1", "concept-1", "MENTIONS_CONCEPT"),
        ),
        dynamic_enrichment="constant enrichment",
        retrieval_identity="synthetic-retrieval-v1",
    )


class ProtocolTests(unittest.TestCase):
    def test_positive_on_off_changes_only_static_relation_section(self) -> None:
        bundle = _bundle()
        on = build_prompt(bundle, ARM_ON)
        off = build_prompt(bundle, ARM_OFF)

        self.assertNotEqual(on.prompt, off.prompt)
        self.assertIn("MENTIONS_CONCEPT", on.prompt)
        self.assertNotIn("MENTIONS_CONCEPT", off.prompt)
        self.assertEqual(on.trace.non_target_fingerprint, off.trace.non_target_fingerprint)
        self.assertTrue(on.trace.activation_status == "positive")
        self.assertTrue(off.trace.activation_status == "positive")

    def test_negative_control_explains_identical_prompts(self) -> None:
        bundle = EvidenceBundle(
            query_id="negative",
            question="Can this be answered?",
            chunks=(EvidenceChunk("c1", "source", "text"),),
        )
        on = build_prompt(bundle, ARM_ON)
        off = build_prompt(bundle, ARM_OFF)

        self.assertEqual(on.prompt, off.prompt)
        self.assertEqual(on.trace.activation_status, "negative")
        self.assertIn("identical", on.trace.activation_explanation)


class DurableRunnerTests(unittest.TestCase):
    def _store(self, root: Path) -> DurableResultStore:
        manifest = root / "manifest.json"
        manifest_hash = write_manifest(manifest, {"test": "r1"})
        store = DurableResultStore(root / "state", manifest, manifest_hash)
        store.test_manifest_hash = manifest_hash  # type: ignore[attr-defined]
        return store

    def _kwargs(self, store: DurableResultStore, adapter: FakeAdapter):
        return {
            "rows": [{"query_id": "q1", "question": _bundle().question}],
            "bundles": {"q1": _bundle()},
            "store": store,
            "adapter": adapter,
            "model_identity": {"model": "fake"},
            "tokenizer_identity": {"tokenizer": "fake"},
            "decoding": {"temperature": 0},
            "manifest_hash": store.test_manifest_hash,  # type: ignore[attr-defined]
        }

    def test_resume_and_third_invocation_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-test-") as directory:
            root = Path(directory)
            store = self._store(root)
            adapter = FakeAdapter()
            kwargs = self._kwargs(store, adapter)

            first = run_pairs(**kwargs, max_new_generations=2, stop_after=1)
            second = run_pairs(**kwargs, max_new_generations=2)
            third = run_pairs(**kwargs, max_new_generations=2)

            self.assertEqual(first["generated"], 1)
            self.assertEqual(second["generated"], 1)
            self.assertEqual(second["skipped"], 1)
            self.assertEqual(third["generated"], 0)
            self.assertEqual(third["skipped"], 2)
            self.assertEqual(len(adapter.calls), 2)
            self.assertEqual(store.progress()["accepted"], 2)

    def test_failure_is_durable_and_retryable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-test-") as directory:
            root = Path(directory)
            store = self._store(root)
            adapter = FakeAdapter(failures=1)
            kwargs = self._kwargs(store, adapter)

            first = run_pairs(**kwargs, max_new_generations=1, stop_after=1)
            self.assertEqual(first["generated"], 1)
            self.assertEqual(first["failed"], 1)
            self.assertEqual(store.progress()["failed_retryable"], 1)
            self.assertEqual(len(list((root / "state" / "outcomes").rglob("*.json"))), 1)

            second = run_pairs(**kwargs, max_new_generations=2)
            self.assertEqual(second["generated"], 2)  # ON retry and OFF pair
            self.assertEqual(store.progress()["accepted"], 2)

    def test_manifest_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-test-") as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest_hash = write_manifest(manifest, {"test": "r1"})
            store = DurableResultStore(root / "state", manifest, manifest_hash)
            write_manifest(manifest, {"test": "changed"})

            with self.assertRaises(CheckpointError):
                store.progress()
            with self.assertRaises(CheckpointError):
                store.execution_session().__enter__()

    def test_accepted_payload_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-test-") as directory:
            root = Path(directory)
            store = self._store(root)
            adapter = FakeAdapter()
            kwargs = self._kwargs(store, adapter)
            run_pairs(**kwargs, max_new_generations=2)
            accepted = store.accepted("q1", ARM_ON, next(iter(
                self._accepted_identity(store, "q1", ARM_ON)
            )))
            self.assertIsNotNone(accepted)
            assert accepted is not None
            result_path = accepted["result_path"]
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["text"] = "tampered"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CheckpointError):
                store.accepted("q1", ARM_ON, accepted["result"]["input_identity"])

    def test_overflow_is_durable_but_never_calls_model_or_consumes_budget(self) -> None:
        bundle = EvidenceBundle(
            query_id="large",
            question="large prompt",
            # FakeTokenizer emits roughly one token per 20 characters, so the
            # fixture is intentionally larger than 12,000 * 20 characters.
            chunks=(EvidenceChunk("large-chunk", "paper", "z" * (INPUT_TOKEN_LIMIT * 20 + 1)),),
        )
        with tempfile.TemporaryDirectory(prefix="r1-test-") as directory:
            root = Path(directory)
            store = self._store(root)
            adapter = FakeAdapter()
            kwargs = self._kwargs(store, adapter)
            kwargs["rows"] = [{"query_id": "large", "question": bundle.question}]
            kwargs["bundles"] = {"large": bundle}

            result = run_pairs(**kwargs, max_new_generations=1)

            self.assertEqual(result["generated"], 0)
            self.assertEqual(result["terminal"], 2)
            self.assertEqual(result["budget_exhausted"], 0)
            self.assertEqual(adapter.calls, [])
            self.assertEqual(store.progress()["overflow"], 2)

    def test_zero_generation_budget_leaves_operation_pending(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-test-") as directory:
            root = Path(directory)
            store = self._store(root)
            adapter = FakeAdapter()
            result = run_pairs(**self._kwargs(store, adapter), max_new_generations=0)
            self.assertEqual(result["generated"], 0)
            self.assertEqual(result["budget_exhausted"], 1)
            self.assertEqual(result["stopped"], 0)
            self.assertEqual(
                store.progress(),
                {"accepted": 0, "attempts": 0, "generation_calls": 0},
            )
            self.assertEqual(adapter.calls, [])

    def test_generation_marker_migrates_old_schema_and_survives_reopen(self) -> None:
        """The cumulative smoke counter must survive an old SQLite namespace."""
        import sqlite3

        with tempfile.TemporaryDirectory(prefix="r1-old-schema-") as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest_hash = write_manifest(manifest, {"test": "old-schema"})
            state = root / "state"
            state.mkdir()
            database = state / "r1_attempts.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE attempts (
                        attempt_id TEXT PRIMARY KEY,
                        manifest_hash TEXT NOT NULL,
                        query_id TEXT NOT NULL,
                        arm_id TEXT NOT NULL,
                        input_identity TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result_path TEXT,
                        error_class TEXT,
                        error_message TEXT,
                        started_epoch REAL NOT NULL,
                        finished_epoch REAL
                    );
                    CREATE TABLE accepted (
                        manifest_hash TEXT NOT NULL,
                        query_id TEXT NOT NULL,
                        arm_id TEXT NOT NULL,
                        input_identity TEXT NOT NULL,
                        result_path TEXT NOT NULL,
                        result_sha256 TEXT NOT NULL,
                        attempt_id TEXT NOT NULL,
                        accepted_epoch REAL NOT NULL,
                        PRIMARY KEY(manifest_hash, query_id, arm_id)
                    );
                    """
                )
                connection.commit()

            store = DurableResultStore(state, manifest, manifest_hash)
            with closing(sqlite3.connect(database)) as connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(attempts)")
                }
            self.assertIn("generation_started", columns)

            attempt_id = store.begin_attempt("q-old", ARM_ON, "old-input")
            store.mark_generation_started(attempt_id)
            self.assertEqual(store.generation_calls(), 1)
            reopened = DurableResultStore(state, manifest, manifest_hash)
            self.assertEqual(reopened.generation_calls(), 1)

    def test_operation_states_keep_retryable_work_pending(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-state-") as directory:
            root = Path(directory)
            store = self._store(root)
            attempt_id = store.begin_attempt("q-pending", ARM_ON, "pending-input")
            store.finish_attempt(
                attempt_id,
                status="FAILED_RETRYABLE",
                error_class="Synthetic",
                error_message="retry me",
            )
            states = store.operation_states(["q-pending"], [ARM_ON])
            self.assertEqual(states, {("q-pending", ARM_ON): "FAILED_RETRYABLE"})
            self.assertFalse(store.is_complete_operation_status(states[("q-pending", ARM_ON)]))

    def test_running_attempt_is_reconciled_as_interrupted(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory(prefix="r1-test-") as directory:
            root = Path(directory)
            store = self._store(root)
            store.begin_attempt("q-crash", ARM_ON, "synthetic-input-identity")
            self.assertEqual(store.recover_inflight(), 1)
            self.assertEqual(store.progress()["interrupted"], 1)
            with closing(sqlite3.connect(store.database)) as connection:
                row = connection.execute(
                    "SELECT status,error_class FROM attempts WHERE query_id='q-crash'"
                ).fetchone()
            self.assertEqual(row, ("INTERRUPTED", "ProcessRestart"))

    def test_execution_lock_rejects_second_runner_and_releases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-test-") as directory:
            root = Path(directory)
            store = self._store(root)
            second_store = DurableResultStore(store.root, store.manifest_path, store.manifest_hash)
            with store.execution_session():
                with self.assertRaisesRegex(CheckpointError, "Another process is already running"):
                    with second_store.execution_session():
                        self.fail("concurrent experiment lock unexpectedly succeeded")
            with second_store.execution_session():
                pass

    def test_manifest_and_cohort_writers_persist_validated_hashes(self) -> None:
        from ablation_mlx.r1_corrected.cohort import write_cohort_manifest
        from ablation_mlx.r1_corrected.identity import object_sha256

        with tempfile.TemporaryDirectory(prefix="r1-test-") as directory:
            root = Path(directory)
            path = root / "manifest.json"
            manifest_hash = write_manifest(path, {"fixed": ["a", "b"]})
            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["manifest_hash"], manifest_hash)

            cohort_body = {"selection_seed": "fixed-v1", "query_ids": ["q1", "q2"]}
            cohort = {**cohort_body, "manifest_hash": object_sha256(cohort_body)}
            cohort_path = root / "cohort.json"
            self.assertEqual(write_cohort_manifest(cohort_path, cohort), cohort["manifest_hash"])
            self.assertEqual(json.loads(cohort_path.read_text(encoding="utf-8")), cohort)

    def test_production_expander_templates_preserve_all_three_sections(self) -> None:
        manager_path = Path(__file__).resolve().parents[3] / "back/src/prompts/manager.py"
        spec = importlib.util.spec_from_file_location("r1_prompt_manager", manager_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prompts = module.PromptManager(str(manager_path.parent))
        kwargs = {
            "context_text": "RETRIEVED_TEXT_SENTINEL",
            "context_graph": "STATIC_GRAPH_SENTINEL",
            "enrichment_block": "DYNAMIC_ENRICHMENT_SENTINEL",
            "history_str": "",
            "query": "question sentinel",
            "question": "question sentinel",
        }
        for template_name in ("ask_expander", "stream_expander"):
            rendered = prompts.get_prompt("rag", template_name, **kwargs)
            for sentinel in ("RETRIEVED_TEXT_SENTINEL", "STATIC_GRAPH_SENTINEL", "DYNAMIC_ENRICHMENT_SENTINEL"):
                self.assertIn(sentinel, rendered)

    @staticmethod
    def _accepted_identity(store: DurableResultStore, query_id: str, arm_id: str):
        with store.database.open("rb"):
            import sqlite3
            connection = sqlite3.connect(store.database)
            try:
                row = connection.execute(
                    "SELECT input_identity FROM accepted WHERE query_id=? AND arm_id=?",
                    (query_id, arm_id),
                ).fetchone()
            finally:
                connection.close()
        if row is None:
            return []
        return [row[0]]


if __name__ == "__main__":
    unittest.main()
