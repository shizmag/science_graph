# Integrated reviewer-response draft — provisional wording

> **Not submission-ready until author checks official decision letter, reviewer numbering, and claims against the exact submission.** Exact journal decision letter was not located; comments below are locally preserved excerpts only. No manuscript source/PDF/page references are claimed. The author has decided against a major manuscript rewrite; response describes supplemental standalone evidence and does not say that sections/figures were revised.

## Reviewer 1

### R1.1 — Component attribution *(local wording provisional)*

**Comment:** “The main finding is driven primarily by B5, which simultaneously changes score blending, graph candidate expansion, ontology lookup, structured context, and other components. It is therefore difficult to determine which design choice causes the substantial quality degradation. A small controlled ablation of the main B5 components, particularly graph expansion, ontology enrichment, and structured context construction, would considerably strengthen the conclusions.”

**Response:** We agree that the original B1–B5 comparison changes multiple components together and cannot identify which component caused the historical B3–B5 quality difference. We conducted an additional post-review sensitivity analysis comparing a freshly executed full B6 reference (paper-facing B5) against 13 component-off or interaction arms under one frozen 375-query population (300 answerable and 75 synthetic designated-negative), one newly frozen `external_db` snapshot, common model/tokenizer revisions, and CUDA/Torch execution. This is a within-run component-sensitivity analysis, not a new B3 result and not a causal decomposition of the historical July MLX B3–B5 contrast.

The matrix planned 5,250 operations; 4,265 generations succeeded and were scored on four judge dimensions, and 985 operations failed before generation because serialized prompts exceeded the 12,000-token cap. Failed operations remain unscored; they were not assigned zero answer quality. We report availability on all 375 planned queries separately from matched-success quality on each arm’s common successful IDs.

The clearest quality signal is parent-chunk deduplication off: Semantic Accuracy increased by 12.47 percentage points on the 296 query IDs where B6 and this arm both generated (query-paired bootstrap 95% CI +7.88 to +16.97 pp). A paired sign test yielded p=2.09×10⁻⁶; Holm-adjusted p was 2.71×10⁻⁵ across the 13 post-hoc Semantic Accuracy contrasts and 1.08×10⁻⁴ across all 52 arm-by-metric contrasts. Faithfulness increased 10.20 pp (CI +6.05 to +14.33; all-52 adjusted p=.00548), Answer Relevance 9.26 pp (CI +4.98 to +13.55; adjusted p=.0132); Citation Fidelity increased 9.36 pp but did not survive correction (adjusted p=.481). These are judge-assigned outcomes from one LLM on one frozen cohort, conditional on joint generation.

Disabling dynamic graph expansion showed a different pattern. On the same 296 paired successful queries, Semantic Accuracy was +7.44 pp (95% CI +2.22 to +12.77), but this did not survive Holm correction (p=.361 in the Semantic Accuracy family; p=1.0 across all 52 contrasts). Faithfulness increased 7.33 pp (CI +2.29 to +12.11; all-52 adjusted p=.0220), while Citation Fidelity decreased 12.80 pp (CI −18.84 to −6.73; adjusted p=.00194). Answer Relevance was +2.82 pp (CI −2.04 to +7.69). Removing the expander also changed generation availability from B6’s 296/375 to 375/375.

Subgroup estimates point to heterogeneity: no-expander Semantic Accuracy was +10.79 pp among answerable pairs (N=248, CI +5.24 to +16.27) but −9.90 pp among designated-negative pairs (N=48, CI −25.00 to +4.58). These 75 negatives were synthetically designated and not independently human-validated as corpus-wide unanswerable. Parser `UNKNOWN` is not abstention, and a missing generation is neither an abstention nor a judged answer.

Saved evidence traces make context construction a plausible contributor, not a proven mediator. Parent-dedup-off still selects about five chunks but averages 1.97 unique PMIDs vs 4.75 for B6; target-paper chunks average 3.86 vs .94. Chunk-level Context Precision increases by about 56 pp, but counts repeated chunks from a single relevant PMID; a document-deduplicated source metric shows no improvement in unique gold-source coverage for these single-PMID questions. The data support an evidence-construction hypothesis, not the claim that repetition caused improved answer quality.

Several named arms had little or no downstream input change: static graph context off and fixed-budget context produced exactly the same serialized prompts and response bytes as B6 for all 296 common successful cases. Concept retrieval off changed only 1/296 prompts/responses; intra-paper candidates off changed 24/294. These should be read as execution-path no-ops or low-activation interventions in this workload, not proof that the components are generally ineffective.

The accompanying analysis therefore gives component-level sensitivity evidence while leaving the historical causal question open. We do not claim the new run explains or decomposes the July MLX B3–B5 quality difference.

**Supplemental evidence:** `revision_ablation_defense/supplementary_ablation_addendum.md` and machine-readable files in `revision_ablation_defense/tables/`. **Manuscript changes:** none claimed; the author chose not to undertake a major rewrite. Editorial risk: if the journal requires the requested results to be incorporated directly into the manuscript rather than supplied as an addendum, this response alone may not satisfy that requirement.

### R1.2 — Entropy estimator sensitivity *(local wording provisional)*

**Comment:** The reviewer asks for top-K entropy sensitivity at K=20/50/100 and retained top-K probability mass.

**Response:** The saved CUDA token telemetry permits a within-run paired sensitivity analysis. Entropy is the per-generated-token top-K distribution renormalized over the top K, in bits; it is not full-vocabulary Shannon entropy. On paired successful cases, parent-deduplication off changes H20/H50/H100 by +.0430/+ .0453/+ .0461 bits while Semantic Accuracy changes +12.47 pp. Expander off changes H20/H50/H100 by −.0548/−.0573/−.0579 bits while its conditional Semantic Accuracy estimate is +7.44 pp (not multiplicity-significant). Retained top-K probability mass changes are near zero for both interventions. Thus the top-K direction is stable across the reported K values for these paired contrasts, but this is not full-vocabulary entropy and not a claim about historical baseline ordering.

**Evidence:** `tables/entropy_paired.csv`, definition in `06_entropy_quality_analysis.md`. **Manuscript changes:** none claimed. The addendum can be supplied as supporting information; direct manuscript integration may still be editorially requested.

### R1.3 — LLM judge reliability *(local wording provisional)*

**Comment:** The reviewer requests human validation or a second independent judge and agreement results.

**Response:** This CUDA ablation was scored by one independent LLM judge, OmniRoute `codex/gpt-6-luna-low`, with a fixed rubric. We verified exact duplicate-input score reuse: 4,265 eligible slots correspond to 2,297 exact payloads; 1,968 duplicate slots reused the score for an identical question, answerability label, reference, evaluator-visible context, and cleaned answer. Duplicate score tuples agree, hashes/IDs reconcile, and no eligible score is missing. This is a score-reuse/integrity audit, not human or second-judge validation, and does not establish judge validity. We do not claim independent human verification.

**Evidence:** `01_integrity_and_accounting.md`, `integrity_audit.json`, archived judge manifest and rubric. **Remaining limitation:** the request for independent human or second-model validation remains unresolved; additional evaluation is outside the current run and was not fabricated.

### R1.4 — Synthetic designated-negative questions *(local wording provisional)*

**Comment:** The reviewer requests confirmation that synthetic unanswerables are truly unanswerable and manual validation.

**Response:** We preserve the dataset designation rather than upgrading it to verified unanswerability. The 75 negative examples were not independently human-reviewed against the full retrieval corpus. We report answerable/negative subgroup results separately, identify parser status as a different construct, and do not interpret these findings as clinical safety evidence.

**Remaining limitation:** corpus-level human unanswerability validation remains unperformed.

### R1.5 — Selection bias / missing outcomes *(local wording provisional)*

**Comment:** The reviewer requests effective sample sizes and a failure penalty or combined utility.

**Response:** The canonical operation table includes all 5,250 planned query-arm slots. Conditional paired answer-quality comparisons use common successfully generated IDs; for the two largest interventions N=296. Generation availability is reported over all 375 planned IDs (B6 296/375; no-expander 375/375; parent-dedup-off 347/375). We also provide a clearly labelled operational composite `U_q = I(generation available) × judge Semantic Accuracy delivered`; unavailable means receive zero only under this service-utility policy, not as a judge quality score. It rises by .2299 for no-expander and .2133 for parent-dedup-off versus B6 on all 375 queries. All 985 failures are serialized prompt-cap errors. We do not assume MCAR; sharp [0,1] score-scale bounds are broad and retained as sensitivity bounds, not imputed outcomes.

**Evidence:** `02_missingness_and_coverage.md`, `tables/coverage_by_arm.csv`, `operational_utility.csv`, `missingness_bounds.csv`.

### R1.6 — Paired statistics / effect sizes / uncertainty *(local wording provisional)*

**Comment:** The reviewer requests sample sizes, effect sizes, confidence intervals, adjusted p-values, and cautious interpretation of small configuration-level correlations.

**Response:** The accompanying paired table reports N, mean query-level difference, bootstrap CI, sign counts, rank-biserial effect, exact paired sign-test p-value, and Holm correction. Within-metric and all-52 contrast families are separately identified and post-hoc. For Semantic Accuracy, parent-dedup-off remains significant under both corrections; the no-expander Semantic Accuracy estimate does not. All unadjusted effects remain reported. No five-configuration correlation is used to claim causal or stable association.

**Evidence:** `03_paired_statistical_analysis.md`, `tables/paired_judge_effects.csv` and `multiplicity_adjustments.csv`.

### Generalizability / retrieval and context attribution

The experiment uses one biomedical dataset cohort, one generator and one new frozen database snapshot. The saved run manifest explicitly does not establish that the July database identity matches this new snapshot. We have not tested a second generator/dataset, and do not claim generalization. Retrieval/context metrics and source-deduplicated context summaries accompany the judge results, with the caveat that chunk-level precision can reward repeated same-PMID chunks. The run provides evidence about sensitivity within this new CUDA arm matrix only.

## Reviewer 2 and unresolved comments

The locally preserved response draft contains separate Reviewer 2 examples and other comments, but this task’s verified new evidence is focused on the post-review CUDA ablation. Do not assert that prior manuscript changes described in that intermediate response are present in the actual submission. Do not fabricate line/page citations. Reconcile every existing response claim against the final submitted manuscript/PDF and official letter before sending.
