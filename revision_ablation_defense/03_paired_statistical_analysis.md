# Paired judge effects and multiplicity

## Analysis rule

Primary contrasts are each non-reference arm minus `B6_full`, on the intersection of successful query IDs. Difference is reported on the original 0–1 scale (multiply by 100 for percentage points). Confidence intervals are two-sided query-level paired percentile bootstrap intervals (10,000 resamples, reproducible per-contrast seed in CSV). A two-sided exact paired sign test is the specified nonparametric test; ties are omitted from its denominator. Rank-biserial effect is computed from signed ranks with midranks for ties. The 13 arm contrasts are post-hoc/exploratory; choices are not described as preregistered.

Holm family 1: 13 intervention/interaction contrasts for each judge metric separately. Additional conservative family: all 13 × 4 = 52 arm-by-metric contrasts. Report raw effects and intervals regardless of adjusted p-values. Bootstrap intervals are unadjusted; do not infer multiplicity-adjusted significance from a nominal 95% CI.

## Principal contrasts (ablation − B6 full)

| Intervention | Metric | Paired N | Mean difference (pp) | Query-bootstrap 95% CI (pp) | Sign counts + / − / tie | Sign-test p | Holm within metric | Holm all 52 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Parent deduplication off | Semantic Accuracy | 296 | +12.47 | [+7.88, +16.97] | 139 / 70 / 87 | 2.09e-6 | 2.71e-5 | 1.08e-4 |
| Parent deduplication off | Answer Relevance | 296 | +9.26 | [+4.98, +13.55] | 130 / 77 / 89 | 2.81e-4 | .00365 | .0132 |
| Parent deduplication off | Faithfulness | 296 | +10.20 | [+6.05, +14.33] | 139 / 81 / 76 | 1.12e-4 | .00145 | .00548 |
| Parent deduplication off | Citation Fidelity | 296 | +9.36 | [+2.65, +16.03] | 80 / 50 / 166 | .0107 | .118 | .481 |
| Dynamic GraphExpander off | Semantic Accuracy | 296 | +7.44 | [+2.22, +12.77] | 133 / 99 / 64 | .0300 | .361 | 1.000 |
| Dynamic GraphExpander off | Answer Relevance | 296 | +2.82 | [−2.04, +7.69] | 126 / 99 / 71 | .0828 | .828 | 1.000 |
| Dynamic GraphExpander off | Faithfulness | 296 | +7.33 | [+2.29, +12.11] | 148 / 93 / 55 | .000479 | .00527 | .0220 |
| Dynamic GraphExpander off | Citation Fidelity | 296 | −12.80 | [−18.84, −6.73] | 30 / 72 / 194 | 3.89e-5 | .000466 | .00194 |

Effects and intervals recompute from response-level score records; raw values are proportion differences (pp shown here). The answerable/negative, entropy, all-arm, multiplicity, and full canonical outputs are machine-readable CSVs in `tables/`.

## Interpretation

- **Parent deduplication:** strongest evidence in this run; Semantic Accuracy and the two non-citation quality dimensions remain significant after both Holm families. Citation Fidelity’s positive nominal difference is not significant after correction.
- **GraphExpander:** Semantic Accuracy increase is not supported after either correction. Faithfulness improvement and Citation Fidelity loss remain significant in both correction families. The reversal of negative-query point estimate and the operational coverage change should be reported with it, not omitted.
- **Other arms:** report full table; no cherry-picking. Several scores tie exactly because contexts or model inputs were identical, or nearly so. Table ties are valid observations but cannot establish clinical equivalence.
- **Historical claim boundary:** this new same-runtime CUDA arm analysis does not isolate the cause of historical July B3–B5 differences. Do not state that component ablation “explains” or “decomposes” that historical contrast.

## Test and uncertainty limits

The judge is one LLM and judge scores are operational rubric scores, not validated biomedical truth. Exact bootstrap intervals quantify query resampling uncertainty conditional on the observed paired cases; they do not incorporate judge calibration error, corpus sampling, or failure selection. The paired sign test was chosen for a distribution-robust directional check and does not model score magnitude. Multiple-testing adjustment applies to the reported test family, not the bootstrap intervals.
