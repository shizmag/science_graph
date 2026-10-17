# Output schema

The full runner creates one `graphs/run_<id>_OCC-RAG-1.7B_CUDA_ABLATION/` directory and a same-tree `.tar.gz` archive.

## Canonical records

- `baselines/<arm_id>/results/<query_id>.json`: one query-arm record with exact question, labels, selected ordered evidence, final prompt, raw and parsed answer, parser state, retrieval/graph/context traces, effective configuration hash, model/database identity, stage times, and errors.
- `baselines/<arm_id>/results/<query_id>.telemetry.jsonl.gz`: lossless compact per-generation-event token telemetry, with CUDA device and explicit entropy/log-probability/probability units.
- `baselines/<arm_id>/effective_config.yaml`: frozen effective arm configuration.
- `baselines/<arm_id>/{retrieved_contexts,generations,parsed}.jsonl`: inspectable arm-level indexes and raw/parsed output records.
- `baselines/<arm_id>/traces/<query_id>.json`: retrieval, ranking, graph, prompt, and trimming trace.
- `generation_results.jsonl`: exactly one row for each planned query-arm pair, including missing and failed operations.

`retrieved_contexts.yaml` is a compatibility index (`cuda_ablation_retrieved_contexts_v1`) into canonical per-arm records. Exact ordered evidence text and generator-visible prompt are in each canonical JSON result; the compatibility file does not duplicate them.

## Analysis and judge export

- `token_telemetry.parquet` and per-arm Parquet files: token-level log probability, NLL, H20/H50/H100, retained top-K mass, top-1 probability, logit gap, stop state, and units. Dense vocabulary logits are not stored.
- `metrics_details.csv`, `metrics_summary.csv`, `metrics_summary.md`: judge-free retrieval recall, context precision, parser answerability outcomes, timing and token statistics. No answer-quality, faithfulness, or citation score is present.
- `metadata/cache_hit_counts.json`, `metadata/cache_stage_hit_rates.json`: arm-level reusable-stage cache counts and hit-rate denominators, calculated from trace events.
- `figures/` and `figures_entropy/`: local retrieval, latency and token-entropy plots only.
- `judge/judge_inputs.jsonl`: all 375 × arm outcomes, including UNKNOWN, abstentions, failed/missing states, and scoring eligibility.
- `judge/arm_mapping.json`: blinded response ID to query/arm mapping. Keep this outside any blinded judge prompt.
- `judge/judge_manifest.json`, `judge/judge_schema.json`: export accounting and data contract; judge status is `not_run` and remote requests are false.

## Integrity and completion

`run_manifest.json`, `arms_manifest.json`, `questions_manifest.json`, `SHA256SUMS`, and `RESULTS_MANIFEST.json` bind run identity, the frozen dataset, the selected external database, pinned models, matrix and file inventory. The completion report distinguishes `COMPLETE`, `PARTIAL`, `FAILED`, and `SCIENTIFIC_VALIDATION_FAILED`. Per-arm completion and intervention-integrity states are recorded separately.

No judge-derived `evaluation_results.yaml`, `result_metrics_judge.yaml`, or incompatible `raw_logits.yaml` is generated.
