"""CPU-only command-layer tests; these must not import or load MLX."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

from ablation_mlx.r1_corrected import cli
from ablation_mlx.r1_corrected.checkpoint import DurableResultStore
from ablation_mlx.r1_corrected.identity import object_sha256, write_manifest
from ablation_mlx.r1_corrected.protocol import ARM_ON


class CliTests(unittest.TestCase):
    def _run(self, *argv: str) -> tuple[int, dict[str, object]]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(list(argv))
        return code, json.loads(output.getvalue())

    def test_doctor_is_cpu_only_and_uses_repo_defaults(self) -> None:
        sys.modules.pop("mlx", None)
        sys.modules.pop("mlx_lm", None)
        code, payload = self._run("doctor")
        self.assertEqual(code, 0)
        self.assertFalse(payload["mlx_imported"])
        self.assertNotIn("mlx", sys.modules)
        self.assertNotIn("mlx_lm", sys.modules)
        self.assertEqual(payload["paths"]["database"]["path"], str(cli.DEFAULT_DATABASE))
        self.assertEqual(payload["paths"]["dataset"]["path"], str(cli.DEFAULT_DATASET))

    def test_preflight_verifies_real_frozen_inputs_without_runtime_import(self) -> None:
        sys.modules.pop("mlx", None)
        sys.modules.pop("mlx_lm", None)
        code, payload = self._run("preflight")
        self.assertEqual(code, 0)
        self.assertEqual(payload["dataset"]["rows"], 375)
        self.assertEqual(payload["database"]["activation_positive"], 347)
        self.assertEqual(payload["database"]["activation_negative"], 28)
        self.assertFalse(payload["model"]["identity"]["config_matches_historical"])
        self.assertNotIn("mlx", sys.modules)
        self.assertNotIn("mlx_lm", sys.modules)

    def test_preflight_rejects_a_database_with_the_wrong_fingerprint(self) -> None:
        code = cli.main(
            [
                "--database",
                "/Users/vladimirkasterin/python/graph/ablation_portability/snapshots/candidates/repo_db/v1/database.db",
                "preflight",
            ]
        )
        self.assertEqual(code, 2)

    def test_export_includes_only_verified_accepted_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-export-test-") as directory:
            root = Path(directory)
            manifest_path = root / "experiment_manifest.json"
            cohort_path = root / "cohort_manifest.json"
            output_path = root / "export.jsonl"
            manifest_hash = write_manifest(
                manifest_path,
                {
                    "experiment_id": "r1_mlx_corrected_structured_context_v1",
                    "cohort_hash": "cohort-hash",
                },
            )
            cohort_payload = {
                "rows": [{"query_id": "q1"}, {"query_id": "q2"}],
            }
            cohort_payload["cohort_hash"] = object_sha256(cohort_payload)
            cohort_path.write_text(json.dumps(cohort_payload), encoding="utf-8")
            store = DurableResultStore(root, manifest_path, manifest_hash)
            identity = "input-q1-on"
            attempt_id = store.begin_attempt("q1", ARM_ON, identity)
            result = {
                "status": "SUCCESS",
                "query_id": "q1",
                "arm_id": ARM_ON,
                "input_identity": identity,
                "prompt": "<prompt>evidence</prompt>",
                "prompt_sha256": hashlib.sha256(
                    b"<prompt>evidence</prompt>"
                ).hexdigest(),
                "text": "raw answer",
            }
            store.commit_success(
                attempt_id=attempt_id,
                query_id="q1",
                arm_id=ARM_ON,
                input_identity=identity,
                result=result,
            )

            # A valid-looking file which is not indexed as accepted must not
            # enter export. It represents an orphan/interrupted write.
            orphan_dir = root / "results" / ARM_ON
            orphan_dir.mkdir(parents=True, exist_ok=True)
            (orphan_dir / "q2.orphan.json").write_text(
                json.dumps({"status": "SUCCESS", "text": "must not export"}),
                encoding="utf-8",
            )

            code, payload = self._run(
                "--artifact-root",
                str(root),
                "export",
                "--output",
                str(output_path),
            )
            self.assertEqual(code, 0)
            self.assertEqual(payload["records"], 4)
            records = [json.loads(line) for line in output_path.read_text().splitlines()]
            by_key = {(item["query_id"], item["arm_id"]): item for item in records}
            self.assertEqual(by_key[("q1", ARM_ON)]["status"], "SUCCESS")
            self.assertEqual(by_key[("q1", ARM_ON)]["response"], "raw answer")
            self.assertEqual(by_key[("q2", ARM_ON)]["status"], "MISSING")
            self.assertEqual(by_key[("q2", ARM_ON)]["response"], None)


if __name__ == "__main__":
    unittest.main()
