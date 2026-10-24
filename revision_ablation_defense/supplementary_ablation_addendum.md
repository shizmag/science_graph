# Supplementary technical addendum: post-review CUDA component sensitivity

## Purpose and scope

This addendum reports a post-review controlled sensitivity analysis addressing whether specific evidence/context components show distinguishable behavior. The paper-facing B5 configuration is internal `B6_full`. The new experiment ran internal B6 plus 13 ablation/interaction arms under one common frozen set of 375 queries (300 answerable; 75 synthetic designated-negative), the new frozen `external_db` snapshot, fixed model revisions, tokenizer, decoding policy and CUDA/Torch runtime. The result applies to this run and observed query population. Historical July results used an older MLX environment and are not re-executed here; the July B3–B5 difference is not decomposed causally by this experiment.

## Operation and judge accounting

There were 5,250 planned query-arm operations. 4,265 generated answers succeeded (81.24%); 985 failed before generation due to serialized prompts exceeding the configured 12,000-token context cap. Failed answers were retained as missing and not assigned judge scores. The independent LLM judge was OmniRoute `codex/gpt-6-luna-low`, with four [0,1] rubric dimensions: Answer Relevance, Semantic Accuracy vs the supplied reference, Faithfulness to evaluator-visible context, and Citation Fidelity. 4,265/4,265 eligible outputs were scored. Exact duplicate case inputs (question, answerability label, reference, evaluator-visible context, cleaned answer) were judged once and reused: 2,297 unique inputs and 1,968 reused slots. One judge is not human ground truth; cache integrity is not judge-validity evidence.

## Estimands

- **Conditional quality:** each ablation compared with B6 only on common successful query IDs. N=296 for the two leading interventions. This is not a 375-query quality estimate.
- **Generation availability:** successful valid generation / all 375 planned queries. B6=296/375; no GraphExpander=375/375; parent-deduplication off=347/375.
- **Service operational utility:** `U_q = I(generation available) × Semantic Accuracy of delivered answer`; a missing response receives zero only under this policy-defined service composite. It is not a missing-answer judge score, nor a clinical utility measure.

## Main paired judge contrasts

Differences are ablation minus B6, in percentage points. Intervals are unadjusted query-level paired bootstrap 95% percentile intervals, 10,000 resamples. Paired exact sign tests omit ties; Holm adjustment is reported for 13 contrasts within each metric and, separately, a conservative 52-arm×metric family. These contrasts are post-hoc/exploratory.

| Intervention | N | Semantic Accuracy Δ [95% CI] | Answer Relevance Δ [95% CI] | Faithfulness Δ [95% CI] | Citation Fidelity Δ [95% CI] | All-52 Holm p (SA / Faith / Cite) |
|---|---:|---:|---:|---:|---:|---:|
| Parent deduplication off | 296 | +12.47 [+7.88,+16.97] | +9.26 [+4.98,+13.55] | +10.20 [+6.05,+14.33] | +9.36 [+2.65,+16.03] | .000108 / .00548 / .481 |
| Dynamic GraphExpander off | 296 | +7.44 [+2.22,+12.77] | +2.82 [−2.04,+7.69] | +7.33 [+2.29,+12.11] | −12.80 [−18.84,−6.73] | 1.000 / .0220 / .00194 |

Parent-deduplication Semantic Accuracy remained significant after both Holm families; its Citation Fidelity did not. No-expander Semantic Accuracy did not remain significant after either family; its Faithfulness increase and Citation Fidelity decrease did. Full 13-arm × 4-metric values appear in `tables/paired_judge_effects.csv` and `tables/multiplicity_adjustments.csv`.

## Answerability subgroup sensitivity

Among the 296 jointly successful no-expander/B6 query IDs: answerable cases (N=248) have Semantic Accuracy +10.79 pp (95% CI +5.24,+16.27); designated-negative cases (N=48) have −9.90 pp (CI −25.00,+4.58). For parent-deduplication off, answerable cases (N=248) show +14.96 pp (CI +9.89,+20.02); designated-negative (N=48) show −0.42 pp (CI −9.06,+8.23). These subgroup analyses are exploratory. The negative labels are synthetic and not human-verified corpus-wide; they do not establish clinical safety.

## Context and prompt evidence

Parent deduplication off changes context composition. Both arms select ~5 chunks, but unique selected PMIDs average 4.75 in B6 and 1.89 in the no-dedup arm. Gold-PMID chunks average .94 and 3.94. The paired chunk-level Context Precision difference is +55.96 pp; this measure treats each relevant chunk as a separate relevant rank and can count multiple chunks from one target PMID. A document-deduplicated AP removes that repeated-chunk contribution and has 0.0 pp paired difference here; these single-document questions often have only one relevant PMID. This supports a context-concentration hypothesis but does not show that repeated chunks caused answer-quality improvement.

The GraphExpander-off trace disables expansion and inserts no dynamic graph output for all 375 queries; paired prompts/responses differ from B6 on all 296 common successful queries. B6’s dynamic enrichment output averages ~22.9k characters among successful outputs. The no-expander arm has full generation coverage and markedly shorter final prompts. At least one overflow trace contains 210,710 characters of dynamic enrichment output, but the failed prompt assembly trace records that this dynamic block was not inserted; overflow error reports 51,004 serialized tokens. The precise component of final serialization responsible for each overflow is therefore not asserted as proven. The fixed-budget branch executed but yielded prompts and response bytes identical to B6 on the 296 common successes and had the same 79 failures; it did not demonstrate an operational prompt-limit remedy.

## Entropy and quality

CUDA telemetry stores per-token top-K entropy computed after local renormalization over logits ranks 1…K, in bits, plus full-softmax retained top-K mass. It is not full-vocabulary Shannon entropy. On the same 296 pairs: parent-deduplication off changes H20/H50/H100 by +.0430/+ .0453/+ .0461 bits while Semantic Accuracy rises +12.47 pp; no-expander changes H20/H50/H100 by −.0548/−.0573/−.0579 bits while Semantic Accuracy has an unadjusted +7.44 pp estimate. Retained-mass shifts are small. These observations show the paired interventions do not follow one monotonic entropy-quality direction; they do not show entropy causes quality.

## Limitations and conclusion

The main conditional quality finding is strongest for parent-chunk deduplication off. Dynamic graph expansion displays an explicit trade-off between judged Faithfulness/Citation Fidelity, availability, and designated-negative point estimates. Several nominal component-off arms are downstream no-ops or low-activation for this cohort. All component estimates are from one LLM judge, one new frozen snapshot and one run, with arm-dependent prompt-overflow missingness. The analysis supports component sensitivity of evidence/context construction; it does not identify a unique cause of the historical July B3–B5 contrast, establish human validation, or prove clinical safety.

**Machine-readable companion tables:** `tables/coverage_by_arm.csv`, `paired_judge_effects.csv`, `multiplicity_adjustments.csv`, `operational_utility.csv`, `subgroup_effects.csv`, `entropy_paired.csv`, `context_provenance.csv`, `overflow_by_arm_subgroup.csv`, `missingness_bounds.csv`. **Reconstruction:** `analyze_existing.py`, `integrity_audit.json`, `canonical_5250.csv`.
