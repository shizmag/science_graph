# Corrected R1 MLX post-run analysis

This directory contains derived, offline-only audit products for the corrected structured-context ablation. The script reads the frozen v2 MLX runtime and the immutable October CUDA archive; it does not modify either source. It does not load a model, run inference, access a network, call a paid judge, or pool the July, October CUDA, and corrected MLX experiments.

## Reproduce

Run from the repository root:

```bash
python3 revision_round1/analysis/r1_mlx_postrun/scripts/audit_r1_mlx.py
python3 -m unittest discover -s revision_round1/analysis/r1_mlx_postrun/tests -p 'test_*.py' -v
```

The audit uses fixed seed `20261113` and `10000` query-bootstrap resamples. It validates accepted-ledger/result SHA-256 values, prepared-input identities and hashes, matched ON/OFF prompt sections, negative controls, predictive telemetry, and the historical CUDA operation ledger. A clean run currently reports 300 matched pairs, zero validation errors, and one documented provenance warning: the cohort embeds the pre-runtime source manifest hash `2da88f58b85926a483cb45dbbac1216e066bedc7b1a273641284fd49f32333a8`, whereas the v2 runtime manifest is `dffd5b400b86035eb270961496399f90495b122891da9e2e0eb37a9fcc7b4607`; the cohort hash itself recomputes to `bf4cc128d85d5534e28a5a0804bef0c6eb6ddfed44a1777fc1ad42d050cf7074`.

## Main evidence

- `integrity_audit.json`: validation status, intervention isolation, negative controls, and telemetry coverage.
- `generation_summary.json`: planned/accepted/failed slots, pairing, token/timing totals, and finish reasons.
- `statistical_summary.json`, `paired_metrics.csv`, and `metric_summary.csv`: query-level endpoint summaries and fixed-seed confidence intervals.
- `historical_cuda_summary.json` and `tables/table_c_historical_cuda_operational_sensitivity.csv`: separate October CUDA availability analysis; these are not pooled with MLX.
- `tables/table_a_experiment_validity_activation.csv` and `tables/table_b_corrected_mlx_ablation.csv`: manuscript-ready validity and corrected-ablation tables.
- `scientific_findings.md`, `reviewer_response_r1.md`, `reviewer_response_operational.md`, `reviewer_coverage_matrix.md`, and the manuscript subsection drafts: working publication/rebuttal text. The authenticated journal decision letter was not found in the inspected worktree, so reviewer numbering remains provisional.
- `figures/README.md`: figure status and optional rendering instructions. No PNG is fabricated when matplotlib is unavailable.

## Quality-judge blocker

The corrected MLX generation has no completed R1 semantic-judge results. `prejudge_export_empty.jsonl` contains explicit `MISSING` records. `judge_manifest_r1_pending.jsonl` is an input-only, 600-row handoff containing configuration-specific generator outputs/evidence; it contains no scores and no API provenance. It is deliberately not required for the committed reproducibility code because it duplicates raw-like prompt/output material; the exact source path and required judges are recorded in the reviewer drafts. Do not interpret the entropy results as answer-quality results.
