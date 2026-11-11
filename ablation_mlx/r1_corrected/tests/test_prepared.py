import tempfile
import unittest
from pathlib import Path

from ablation_mlx.r1_corrected.prepared import make, read, prepare_all, write, PreparedManifestMismatch
from ablation_mlx.r1_corrected.runner import run_prepared
from ablation_mlx.r1_corrected.checkpoint import DurableResultStore
from ablation_mlx.r1_corrected.identity import write_manifest, object_sha256
from ablation_mlx.r1_corrected.protocol import ARM_OFF, ARM_ON


class PreparedInputTests(unittest.TestCase):
    def frozen(self, root, **changes):
        cohort = {"rows": [{"query_id": "q"}]}
        cohort["cohort_hash"] = object_sha256(cohort)
        manifest = {"cohort_hash": cohort["cohort_hash"], "arms": [ARM_ON, ARM_OFF],
                    "database": "db-a", "intervention": "on-off", "prompt_version": "v1", **changes}
        path = root / (object_sha256(manifest) + ".json")
        return path, write_manifest(path, manifest), cohort

    def test_manifest_binding_rejects_other_experiment_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, mh, cohort = self.frozen(root)
            items = [("q", ARM_ON, lambda: make("q", ARM_ON, "p", [1], manifest_hash=mh))]
            self.assertEqual(prepare_all(items, root, manifest_path=path, manifest_hash=mh, cohort=cohort)["prepared"], 1)
            self.assertEqual(prepare_all(items, root, manifest_path=path, manifest_hash=mh, cohort=cohort)["skipped"], 1)
            for change in ({"database": "db-b"}, {"intervention": "other"}, {"prompt_version": "v2"}, {"experiment_id": "b"}):
                other, other_hash, _ = self.frozen(root, **change)
                with self.assertRaises(PreparedManifestMismatch):
                    prepare_all(items, root, manifest_path=other, manifest_hash=other_hash, cohort=cohort)
            (root / "q__structured_context_on.json").write_text("not json")
            with self.assertRaises(ValueError):
                prepare_all(items, root, manifest_path=path, manifest_hash=mh, cohort=cohort)

    def test_round_trip_and_hash_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q__on.json"
            item = make("q", ARM_ON, "prompt", [1, 2], question="Q")
            write(path, item)
            self.assertEqual(read(path), item)
            text = path.read_text().replace("prompt", "changed")
            path.write_text(text)
            with self.assertRaises(ValueError):
                read(path)

    def test_wrong_slot_and_nonmember_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, mh, cohort = self.frozen(root)
            artifact = root / "q__structured_context_on.json"
            write(artifact, make("q", ARM_OFF, "p", [1], manifest_hash=mh))
            items = [("q", ARM_ON, lambda: self.fail("must not rebuild"))]
            with self.assertRaisesRegex(ValueError, "different slot"):
                prepare_all(items, root, manifest_path=path, manifest_hash=mh, cohort=cohort)
            with self.assertRaisesRegex(ValueError, "not in frozen cohort"):
                prepare_all([("other", ARM_ON, lambda: None)], root, manifest_path=path, manifest_hash=mh, cohort=cohort)

    def test_rehashed_prompt_or_tokens_are_rejected(self):
        import json
        from ablation_mlx.r1_corrected.prepared import digest
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "item.json"
            item = make("q", ARM_ON, "p", [1], manifest_hash="a")
            for field, value in (("prompt", "changed"), ("token_ids", [2])):
                data = item.to_dict()
                data[field] = value
                data["input_hash"] = digest({key: val for key, val in data.items() if key != "input_hash"})
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    read(path)

    def test_prepare_all_is_resumable_and_records_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, mh, cohort = self.frozen(root)
            items = [("q", ARM_ON, lambda: make("q", ARM_ON, "p", [1], manifest_hash=mh)),
                     ("q", ARM_OFF, lambda: (_ for _ in ()).throw(RuntimeError("bad")))]
            self.assertEqual(prepare_all(items[:1], root, manifest_path=path, manifest_hash=mh, cohort=cohort)["prepared"], 1)
            self.assertEqual(prepare_all(items, root, manifest_path=path, manifest_hash=mh, cohort=cohort), {"prepared": 0, "failed": 1, "skipped": 1})
            self.assertEqual(prepare_all(items, root, manifest_path=path, manifest_hash=mh, cohort=cohort), {"prepared": 0, "failed": 0, "skipped": 2})
            self.assertEqual(read(root / "q__structured_context_off.json").status, "failed")

    def test_prepared_runner_loads_once_and_resumes_without_rebuilding(self):
        class Adapter:
            def __init__(self): self.loads = 0; self.calls = []; self.model = None; self.tokenizer = None
            def load(self): self.loads += 1; self.model = object(); self.tokenizer = object()
            def generate(self, prompt): self.calls.append(prompt); return {"text": prompt}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "manifest.json"; manifest_hash = write_manifest(manifest, {})
            store = DurableResultStore(root, manifest, manifest_hash)
            inputs = [make("q", arm, arm, [1, 2]) for arm in (ARM_ON, ARM_OFF)]
            adapter = Adapter()
            first = run_prepared(inputs, store=store, adapter=adapter, model_identity={"id": "fake"},
                                 tokenizer_identity={"id": "fake"}, decoding={"temperature": 0},
                                 manifest_hash=manifest_hash, stop_after=1)
            second = run_prepared(inputs, store=store, adapter=adapter, model_identity={"id": "fake"},
                                  tokenizer_identity={"id": "fake"}, decoding={"temperature": 0},
                                  manifest_hash=manifest_hash)
            third = run_prepared(inputs, store=store, adapter=adapter, model_identity={"id": "fake"},
                                 tokenizer_identity={"id": "fake"}, decoding={"temperature": 0},
                                 manifest_hash=manifest_hash)
            self.assertEqual(first["stopped"], 1); self.assertEqual(second["generated"], 1)
            self.assertEqual(third["skipped"], 2); self.assertEqual(adapter.loads, 1)
            self.assertEqual(adapter.calls, [ARM_ON, ARM_OFF])

    def test_completed_run_does_not_load_adapter_again(self):
        class Adapter:
            def __init__(self): self.loads = 0; self.model = None; self.tokenizer = None
            def load(self): self.loads += 1; self.model = object(); self.tokenizer = object()
            def generate(self, prompt): return {"text": prompt}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "manifest.json"; mh = write_manifest(manifest, {})
            item = make("q", ARM_ON, "p", [1]); store = DurableResultStore(root, manifest, mh)
            run_prepared([item], store=store, adapter=Adapter(), model_identity={}, tokenizer_identity={}, decoding={}, manifest_hash=mh)
            adapter = Adapter()
            result = run_prepared([item], store=store, adapter=adapter, model_identity={}, tokenizer_identity={}, decoding={}, manifest_hash=mh)
            self.assertEqual(result["skipped"], 1); self.assertEqual(adapter.loads, 0)


if __name__ == "__main__":
    unittest.main()
