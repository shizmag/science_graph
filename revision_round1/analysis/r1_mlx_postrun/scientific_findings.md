# Scientific findings — corrected R1 MLX structured-context ablation

## Status

The v2 execution ledger contains 600/600 accepted successful generation slots (300 per arm), with 300/300 matched query pairs. All 600 accepted result-file hashes, prepared-input identities, prompt hashes, token hashes, stable result IDs, and predictive-telemetry records passed the audit checks. No retryable, terminal, interrupted, or incomplete slots were observed. The cohort manifest has a documented provenance discrepancy: its `experiment_manifest_hash` is the pre-runtime source hash `2da88f58b85926a483cb45dbbac1216e066bedc7b1a273641284fd49f32333a8a`, rather than the frozen v2 runtime hash `dffd5b400b86035eb270961496399f90495b122891da9e2e0eb37a9fcc7b4607`; this is reported, not silently repaired.

## Intervention

The final 300-query cohort contains 278 activation-positive and 22 activation-negative controls. In every activation-positive pair, the static-relation section changed while all non-target prompt sections, question, evidence identities and evidence IDs were invariant. In every activation-negative pair the complete prompts were identical. Thus the prepared-input audit supports an isolated prompt-level intervention for the observed 278 active queries, not for all 300 queries. The 22 negative controls produced identical output token IDs and identical predictive telemetry across arms.

## Predictive uncertainty

All 300 pairs were valid for H20, H50, and H100 under `r1-mlx-predictive-distribution-v1`. These are arithmetic means of local top-K predictive distributions over compact generated-token rows including EOS, in bits; they are not the legacy full-distribution entropy fields (which are intentionally unavailable). The paired ON−OFF effects were:

| endpoint | ON mean | OFF mean | paired difference | 95% query-bootstrap CI | valid pairs |
|---|---:|---:|---:|---:|---:|
| H20 (bits) | 0.9620 | 0.9584 | 0.0037 | [-0.0178, 0.0241] | 300 |
| H50 (bits) | 1.0234 | 1.0183 | 0.0051 | [-0.0181, 0.0273] | 300 |
| H100 (bits) | 1.0500 | 1.0442 | 0.0058 | [-0.0182, 0.0288] | 300 |

The point estimates are small positive ON−OFF changes, but the intervals cross zero. The three metrics therefore do not establish a statistically precise entropy shift at this sample size. They also do not imply equivalence. Activation-positive-only estimates are in `statistical_summary.json`; the 22 controls are not pooled with the active intervention when interpreting causality.

## Quality and interpretation

No independent LLM-judge scores exist for this R1 MLX generation. `prejudge_export_empty.jsonl` contains explicit `MISSING` records, not scores, and the runtime manifest records external judging as `not_run`. Consequently this experiment cannot answer whether structured context improved semantic accuracy or faithfulness, cannot test a quality–entropy association, and cannot support a positive or negative quality conclusion. The offline judge manifest contains 600 configuration-specific inputs and records zero paid API calls.

## Historical boundary

    The October CUDA archive is a separate, partial execution: 4265/5250 operations succeeded, and its answerable-300 availability is reported separately in `historical_cuda_summary.json` and `tables/table_c_historical_cuda_operational_sensitivity.csv`. It supports claims about operational reliability and context-limit selection, not a pooled answer-quality or entropy effect with this MLX run. The corrected MLX run is a valid fresh prompt intervention but is not byte-identical to CUDA: the model weights match the historical SHA, while local configuration metadata does not (`config_matches_historical=false`), and backend/prompt/runtime conditions differ.

## Reviewer-facing verdict

The corrected run fixes the inactive old static-context prompt intervention and provides complete, auditable generation and telemetry evidence. It does not by itself validate a semantic-quality benefit. A skeptical response should therefore claim: intervention validity and complete availability are established; a small inconclusive predictive-entropy difference is observed; quality impact remains pending independent judging.
