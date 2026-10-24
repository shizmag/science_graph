# Entropy and answer-quality analysis

## Definition

CUDA telemetry computes softmax from raw, unprocessed full-vocabulary logits, records full-softmax log probability and retained mass, and computes `entropy_k20_bits`, `entropy_k50_bits`, `entropy_k100_bits` after locally renormalizing probability among the top K logits. These are **truncated, locally renormalized top-K entropies**, in bits, not full-vocabulary Shannon entropy. The unit in the table is first the mean per generated answer token for each query, then query-paired difference; the 375-operation planned cohort is not the denominator when generation failed.

All 4,265 successful operations have final-answer token telemetry in the saved Parquet; failed operations have no generated-token entropy. The three large B6 pairings each have N=296 with telemetry for every paired successful answer. Retained mass is full-softmax probability mass within each top K, not locally renormalized entropy.

| Ablation vs B6 full | N | ΔH20 bits [95% CI] | ΔH50 bits [95% CI] | ΔH100 bits [95% CI] | Δ retained mass K20 | Δ mass K50 | Δ mass K100 | Δ Semantic Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent deduplication off | 296 | +0.0430 [+0.0091,+0.0768] | +0.0453 [+0.0087,+0.0817] | +0.0461 [+0.0084,+0.0830] | −0.000233 [−0.000862,+0.000393] | −0.000020 [−0.000326,+0.000296] | +0.000045 [−0.000141,+0.000238] | +12.47 pp |
| Dynamic GraphExpander off | 296 | −0.0548 [−0.0920,−0.0170] | −0.0573 [−0.0967,−0.0169] | −0.0579 [−0.0992,−0.0163] | +0.000204 [−0.000454,+0.000884] | −0.000069 [−0.000405,+0.000262] | −0.000114 [−0.000324,+0.000091] | +7.44 pp (unadjusted; not multiplicity-significant) |

Entropy confidence intervals above are 10,000 query-level paired bootstrap intervals, unadjusted; retained-mass interval units are fractions. Corresponding Semantic Accuracy uncertainty/multiplicity is in `paired_judge_effects.csv`.

## Interpretation

These two interventions are direct evidence against a simple monotonic rule that lower top-K generation entropy necessarily means higher answer quality: both interventions increase conditional Semantic Accuracy on their paired observed support, but one raises entropy and one lowers it. The no-parent-dedup Semantic Accuracy finding remains Holm-significant; no-expander Semantic Accuracy does not. Across queries, H50-change and Semantic-Accuracy-change are weak positive descriptive correlations (r=.166 parent-off; r=.123 no-expander); no causal interpretation follows. Retained top-K mass is almost unchanged, so entropy movement is not explained by a broad loss/gain in top-K probability mass. This is within one model, frozen cohort, and one judge; it does not establish an entropy-quality law or cause.
