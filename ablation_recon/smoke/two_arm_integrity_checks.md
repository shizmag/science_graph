# Two-arm micro-ablation checks

After E2E smoke, run reference internal B6 (paper B5) and one B6 intervention that is exercised and has observable intermediate effect. Selected historical query has candidates before rerank but none in final context; graph-off may prove candidate suppression, not altered generated evidence. If no intervention is observable, gate fails.

Checks:
- Serialize full effective config and assert only intended intervention/derived flags differ.
- Match query bytes, manifest, DB/index/model/image fingerprints and decoding protocol.
- Rerun upstream retrieval when changed; no cache may bypass intervention.
- Preserve candidate source IDs/scores, graph records, selected contexts, prompt hash and stage counts.
- Prove target branch changed and unrelated paths remain enabled/observable.
- For downstream cache, prove cached inputs identical and justify reuse.
- Hold tokenizer, prompt rules, generator, stop and entropy metric fixed.
- Fail on DB split, CPU fallback, hidden config drift, cache collision or missing trace.
- No judge call.

YAML difference is not proof. Runtime trace and invariants are required.
