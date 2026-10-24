# Answerable vs designated-negative subgroup results

Negative labels are **synthetic designated-negative** labels in the frozen set. Corpus-wide unanswerability has not been independently human-validated. Interpret judge Semantic Accuracy under the saved rubric and these labels, not as a clinical-safety endpoint.

All contrasts below are ablation minus B6 on jointly successful cases, with 10,000 query-level paired bootstrap replicates and unadjusted exploratory intervals. Small subgroup Ns do not support a definitive subgroup interaction claim.

| Intervention | Group | N | Semantic Accuracy Δ (pp) | Paired 95% CI (pp) | + / − / tie |
|---|---|---:|---:|---:|---:|
| Parent deduplication off | Answerable | 248 | +14.96 | [+9.89, +20.02] | 136 / 66 / 46 |
| Parent deduplication off | Designated-negative | 48 | −0.42 | [−9.06, +8.23] | 3 / 4 / 41 |
| GraphExpander off | Answerable | 248 | +10.79 | [+5.24, +16.27] | 127 / 89 / 32 |
| GraphExpander off | Designated-negative | 48 | −9.90 | [−25.00, +4.58] | 6 / 10 / 32 |

For GraphExpander off, answerable Faithfulness changes +10.04 pp (N=248; 95% CI +4.90 to +15.21); answerable Citation Fidelity changes −15.27 pp (95% CI −22.08 to −8.46). On designated negatives Faithfulness is −6.67 pp (N=48; CI −19.58 to +6.25), and Citation Fidelity is 0.00 pp (CI −10.42 to +10.42). Parent-dedup-off designated-negative quality changes are near zero with broad intervals. CSV contains all four metrics per subgroup.

## Availability and parser status are separate

- B6 produces 248/300 answerable and 48/75 designated-negative outputs. The no-expander arm produces all 300 and all 75.
- B6 parser outcomes among generated negatives: 7 `UNANSWERABLE`, 2 `ANSWERABLE`, 39 `UNKNOWN`; 27 negatives have no generated output. No-expander: 11 `UNANSWERABLE`, 2 `ANSWERABLE`, 62 `UNKNOWN` among all 75.
- `UNKNOWN` is an operational parser label, not a semantic abstention. A missing output is not an abstention.
- If reporting parser-derived UQ-FPR, provide its denominator and label definition; it is not clinical safety evidence. Example: parser `ANSWERABLE` on designated negatives is 2/75 for B6 and 2/75 for no-expander; B6 has 48 generated negatives and 27 unavailable outputs, while no-expander has 75. The parser is often `UNKNOWN`, so these counts are not a calibrated safety rate.

## Interpretation

GraphExpander removal improves the conditional answerable-query point estimate while worsening designated-negative Semantic Accuracy by about ten points; the latter interval crosses zero and N=48. It also lowers Citation Fidelity in answerable queries. The component therefore shows a meaningful possible safety/grounded-citation trade-off, not a universal benefit of disabling expansion. The designated-negative sample is synthetic and small, and should not support clinical-safety claims. No new unanswerability labels or human validation were added.
