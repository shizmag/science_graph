# Author guide: verify and import collaborator results

The collaborator sends one full-run `.tar.gz` archive and its adjacent `.sha256` file. Verify and import on macOS or another Python 3 host with the standard library:

```sh
./ablation verify-results /path/to/run_<id>_OCC-RAG-1.7B_CUDA_ABLATION.tar.gz
```

The author verifier does not need CUDA, Docker, uv, PyYAML, model downloads, or access to the collaborator's database. It checks the archive checksum and safe member paths, internal inventory and checksums, frozen 375-query identity, selected database/model identities, the 14-arm matrix, query-arm counts, exact prompts and response hashes, telemetry schemas/units, intervention traces, and judge-export completeness.

Valid partial archives can be imported; their status remains `PARTIAL`, `FAILED`, or `SCIENTIFIC_VALIDATION_FAILED`, and `complete` remains false. Imports are stored in a separate `ablation_cuda/imported/full_cuda_ablation/<run_id>/` namespace. Existing historical July results are not modified.

The import contains `judge/judge_inputs.jsonl`, the separate `judge/arm_mapping.json`, and `judge/judge_schema.json`. Use the author's independent evaluation workflow with the mapping retained outside judge prompts, then join scores by `blinded_response_id`. The archive itself contains no judge scores and the verifier makes no remote requests.
