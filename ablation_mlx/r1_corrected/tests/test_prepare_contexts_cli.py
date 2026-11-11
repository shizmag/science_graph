import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ablation_mlx.r1_corrected.identity import object_sha256, write_manifest
from ablation_mlx.r1_corrected.protocol import ARM_ON, ARM_OFF, PROTOCOL_VERSION, PROMPT_VERSION


class PrepareContextsCLITest(unittest.TestCase):
    def test_subprocess_prepare_resume_and_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cohort = {"rows": [{"query_id": "q", "activation_positive": False}]}
            cohort["cohort_hash"] = object_sha256(cohort)
            (root / "cohort_manifest.json").write_text(json.dumps(cohort))
            manifest = {"schema_version": "r1-mlx-corrected-runtime-1", "protocol_version": PROTOCOL_VERSION,
                        "prompt_version": PROMPT_VERSION, "intervention": "static_graph_relations_only",
                        "arms": [ARM_ON, ARM_OFF], "cohort_hash": cohort["cohort_hash"]}
            mh = write_manifest(root / "experiment_manifest.json", manifest)
            # Only expensive external data/tokenizer dependencies are replaced; the real
            # parser, command, protocol prompt builder and artifact writer run in-process.
            code = '''
import sys, types
from ablation_mlx.r1_corrected import cli
from ablation_mlx.r1_corrected.protocol import EvidenceBundle
class Store:
 def evidence_for_record(self, record): return EvidenceBundle(query_id=record["query_id"], question="Question?")
class Tokenizer:
 @staticmethod
 def from_file(path): return Tokenizer()
 def encode(self, prompt, add_special_tokens=False): return [1,2,3]
sys.modules["tokenizers"] = types.SimpleNamespace(Tokenizer=Tokenizer)
cli._validate_frozen_inputs = lambda **kwargs: ([{"query_id":"q"}], Store(), {})
sys.exit(cli.main())
'''
            def run(hash_value=mh):
                return subprocess.run([sys.executable, "-c", code, "--runtime-artifact-root", str(root),
                                       "prepare-contexts", "--manifest-hash", hash_value],
                                      capture_output=True, text=True)
            first = run()
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads(first.stdout)["prepared"], 2)
            paths = sorted((root / "prepared_inputs").glob("*.json"))
            self.assertEqual(len(paths), 2)
            paths[1].unlink()  # Simulate interruption after one atomic publication.
            resumed = run()
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual((json.loads(resumed.stdout)["prepared"], json.loads(resumed.stdout)["skipped"]), (1, 1))
            self.assertEqual(json.loads(run().stdout)["skipped"], 2)
            self.assertNotEqual(run("wrong").returncode, 0)
            paths[0].write_text("corrupt")
            self.assertNotEqual(run().returncode, 0)
            help_result = subprocess.run([sys.executable, "-m", "ablation_mlx.r1_corrected.cli", "prepare-contexts", "--help"], capture_output=True, text=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--manifest-hash", help_result.stdout)


if __name__ == "__main__":
    unittest.main()
