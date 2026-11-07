# R2 API v3 scientific audit and provisional freeze

## Scope and disposition

**400 of 400 v3 result records are technically accepted; none is silently replaced.** Each was checked against the frozen 100-query cohort, configuration-specific B3/B5 evidence and input hashes, v3 rubric hash, requested/returned model, accepted request-body hashes, scoring schema, and a separate completed raw provider attempt containing the identical parsed JSON. Prior v2 and heuristic outputs were not included. The 415 attempts comprise 400 successful and 15 invalid/unsuccessful attempts, preserved in place. No judge or generation was invoked during this audit. `frozen_sha256.txt` creates a post-hoc integrity snapshot, not independent proof of what was sent at inference time.

**Publication-ready status is conditional on human adjudication of the exceptions below**, particularly how evidence-content assertions are treated under Faithfulness and the flagged negative label. The frozen numerical outputs are not changed by audit annotations.

The raw captured records include request hashes rather than a separately captured wire-level request body. Reconstructing the request using the frozen code, exact task, rubric and the three documented output limits matches every accepted hash. This validates consistency with recorded fingerprints, not a packet-level network capture. Both models saw scientifically identical reconstructed messages for an item; transport/model fields differ. The initial manifest records `f503db9`, whereas actual execution also used uncommitted changes and subsequent commits; see `execution_code_provenance.json`. Models returned `gpt-6-luna` (accepted normalized OmniRoute alias) and `deepseek/deepseek-v4.1-flash-1017`. Provider response IDs and raw text are distinct across every paired provider record; no parser reuse, duplicate provider content, or deterministic score injection was found in the recorded artifacts. This is not independent attestation of provider-internal execution or Luna reasoning effort.

Output limits changed from 1,400 to 3,000 and then 6,000 after responses exhausted the cap. Accepted Luna records: 2, 130, 68 respectively; DeepSeek: 1, 130, 69. The scientific messages and rubric remain the same under the reconstructed request hashes; transport generation settings differed, which can affect score distributions. Answer Relevance is an exploratory v3 addition; Negative Safety is revision-added supplementary, not a prospective primary endpoint. No v2 scores were pooled. Citation Fidelity is not a primary endpoint.

## Independent paired recomputation

The independent implementation reproduced all 12 reported v3 cohort/metric/judge contrasts, 5,000-resample query-paired percentile intervals and pooled agreement point estimates from raw result files. Historical bootstrap replay required the original filesystem result-file iteration order; a different order with the same seed yields Monte Carlo variation in interval endpoints. The results below are complete-pair means, not marginal all-response means.

| Cohort | Judge | Endpoint | N pairs | B3 | B5 | B3−B5 (95% CI) |
|---|---|---|---:|---:|---:|---:|
| Answerable | Luna | Semantic Accuracy | 80 | 0.788 | 0.281 | 0.506 (0.419, 0.594) |
| Answerable | DeepSeek | Semantic Accuracy | 80 | 0.812 | 0.331 | 0.481 (0.381, 0.588) |
| Answerable | Luna | Faithfulness | 78 | 0.923 | 0.577 | 0.346 (0.250, 0.442) |
| Answerable | DeepSeek | Faithfulness | 79 | 0.905 | 0.595 | 0.310 (0.203, 0.418) |
| Negative | Luna | Negative Safety | 19 | 0.895 | 0.921 | −0.026 (−0.158, 0.079) |
| Negative | DeepSeek | Negative Safety | 20 | 0.850 | 0.925 | −0.075 (−0.225, 0.050) |
| Negative | Luna | Faithfulness | 5 | 0.700 | 0.800 | −0.100 (−0.400, 0.200) |
| Negative | DeepSeek | Faithfulness | 10 | 0.900 | 0.950 | −0.050 (−0.150, 0.000) |

All other effective Ns and exploratory relevance estimates are in `audit_results.json`. No missing or failed generation was assigned a quality score; all 200 archived answers in this cohort were nonempty and scored. This does not characterize operational availability outside the selected cohort.

## Faithfulness applicability

| Judge | Cohort | B3 scored / not applicable | B5 scored / not applicable | Complete pairs |
|---|---|---:|---:|---:|
| Luna | Answerable (80) | 79 / 1 | 79 / 1 | 78 |
| DeepSeek | Answerable (80) | 80 / 0 | 79 / 1 | 79 |
| Luna | Negative (20) | 11 / 9 | 11 / 9 | 5 |
| DeepSeek | Negative (20) | 15 / 5 | 12 / 8 | 10 |

The subset is selected by response behavior and each judge's applicability decisions. Different Ns and sets can induce selection bias and preclude direct comparisons of complete-pair means across judges or cohorts. Negative Faithfulness is especially fragile.

## Perfect Negative Safety agreement investigation

All 40 negative observations and each judge's rationale and raw-attempt filename are in `audit_results.json`. Of 39 jointly scored observations, 35 were (1,1), one (0.5,0.5), and three (0,0). One further observation was Luna null versus DeepSeek 0, because Luna flagged a potential label conflict (`pubmedqa_unans_18235194`, B3). Thus the perfect numerical agreement excludes an applicability disagreement. Overall quadratic-weighted kappa and ICC are 1.0 on the 39 observations; these estimates cannot establish correctness. Only four jointly scored cases are below 1; most are straightforward abstentions. In some cluster-bootstrap replicates all scores were identical and correlation/ICC/kappa were undefined. Conditional degenerate [1,1] intervals are not evidence of perfect underlying reliability.

Representative check: for `pubmedqa_unans_10430303` B5, DeepSeek identifies substitution of unrelated laser-ablation and clavicle studies while Luna also flags unrelated evidence; both score 0. For `pubmedqa_unans_18670651` B5, both score 0.5 but Luna objects to a claim about absent information while DeepSeek identifies an additional misattributed odds-ratio claim. Matching numbers thus do not imply matching reasoning; the former interpretation should be reviewed against the supplementary anchors. The 39 score-pair distributions and separate response IDs/content are consistent with independent outputs but cannot rule out hidden provider dependencies or shared priors.

## Agreement by configuration and paired contrast

ICC is absolute-agreement single-measure ICC(A,1); the audit separately computed Spearman and mean absolute difference (MAD). Parenthesized ICC intervals use 2,000 query-cluster bootstrap replicates, retaining both configuration observations for a sampled query. Undefined replicates are reported in the JSON. Comparisons of judge-specific B3−B5 differences use only queries with all four scores.

| Metric | Scope | N | MAD | Spearman | ICC (cluster 95% CI) |
|---|---|---:|---:|---:|---:|
| Answerable Semantic Accuracy | B3 | 80 | 0.088 | 0.706 | 0.725 (0.557, 0.853) |
| Answerable Semantic Accuracy | B5 | 80 | 0.100 | 0.809 | 0.793 (0.687, 0.877) |
| Answerable Semantic Accuracy | paired difference | 80 | 0.150 | 0.805 | 0.802 (0.708, 0.870) |
| Answerable Faithfulness | B3 | 79 | 0.051 | 0.692 | 0.684 (0.461, 0.863) |
| Answerable Faithfulness | B5 | 79 | 0.089 | 0.860 | 0.862 (0.756, 0.937) |
| Answerable Faithfulness | paired difference | 78 | 0.122 | 0.836 | 0.836 (0.731, 0.910) |
| Negative Safety | B3 / B5 | 19 / 20 | 0 / 0 | 1 / 1 | 1 / 1, unstable |
| Negative Safety | paired difference | 19 | 0 | 1 | 1, unstable |
| Negative Faithfulness | B3 / B5 | 9 / 10 | 0 / 0.05 | 1 / 0.994 | 1 / 0.842, unstable |
| Negative Faithfulness | paired difference | 4 | 0.125 | 0.816 | 0.727, unstable |

## Missing rationales and human adjudication

`missing_faithfulness_rationale_annotations.csv` separately identifies all 21 records, their original generated answers, and audit assessments; no source output is edited. All 21 responses are abstentions and lack an explicit biomedical result; 20 null scores are reasonable under a *narrow biomedical-factual-claim* reading, although several contain claims about what the evidence does or does not contain. That scope interpretation and the absent rationales remain limitations, not proof those meta-claims are correct.

**Material exception:** Luna `e87b421dc64b3e7b43ae2af8`, answerable query `pubmedqa_16097998`, B3, says the supplied sources contain no relevant information. Block 1 explicitly addresses the question and concludes screening seems reasonable; Luna itself noted that contradiction for Semantic Accuracy but marked Faithfulness not applicable. DeepSeek scored Faithfulness 0 for the same evidence-content assertion. Human adjudication must decide whether such evidence-content claims are in scope, without retroactively changing the frozen rubric. This is an applicability disagreement, not proof of an incorrect numerical score. For orientation only, excluding this entire query from DeepSeek's paired answerable Faithfulness analysis changes N 79 to 78 and the contrast from 0.310 to 0.327; this is an exploratory sensitivity calculation, not a replacement estimate.

A separate human review is required for Luna's `pubmedqa_unans_18235194` B3 label conflict and for the 0.5 negative-safety rationale described above. The audit does not adjudicate clinical truth or establish independent human ground truth.

## Historical reconstruction and citation limitations

The July recovery report classifies the final post-trim evidence payloads as exact archived for the selected inputs, with B3/code B4 using archived trimmed text and graph, and B5/code B6 using the archived enrichment block. The original fully serialized prompts/tokenizer rendering were not archived; **byte-identical full historical prompts are not claimed**. B3 has numbered block markers; B5 enrichment uses source labels without a verified numbered-block mapping. Thus generated B5 `[Block N]` targets may be unresolved and are not automatically unsupported claims. The recovery report's 37 unresolvable records and 375 complete-evidence pairs refer to its wider historical registry, not necessarily this selected cohort. Do not infer zero unresolved citations for the selected sample from that report.

## Manuscript methods and limitations

Two independent API routes from different model families scored the same 100 frozen historical query pairs using the unified v3 rubric, validated references for answerable Semantic Accuracy, and each generated answer's own recovered generator-visible evidence for Faithfulness. We used query-paired bootstrap intervals and report revision-added Negative Safety separately. Matching scores do not demonstrate factual correctness; provider family independence is based on exposed routing/model metadata, not independent provider attestation. Reconstructed evidence, missing rationale documentation, nonrandom applicability, result-token-cap changes, unresolved citations and label validity limit interpretation. The evidence-content Faithfulness scope requires human adjudication.

## Reviewer #1, Comment #2 — draft

We repeated the evaluation on a frozen 100-query paired historical cohort with two different model families and shared scoring instructions. Each judge received the same historical answer and its configuration-specific recovered evidence, rather than substituting retrieved chunks or requiring B3/B5 evidence to match. We report query-paired answerable Semantic Accuracy and Faithfulness, separate revision-added negative-query abstention results, agreement, effective sample sizes and evidence-recovery limitations. Citation Fidelity is not reinstated as a primary metric; unresolved historical citation targets are not automatically counted as unsupported. The report explicitly flags label and Faithfulness-applicability cases for human adjudication.
