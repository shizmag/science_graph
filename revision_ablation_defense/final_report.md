# Final report — decision-ready

## Verdict: `READY_WITH_EXPLICIT_LIMITATIONS`

### 1. Is ablation accounting internally consistent?

**Yes.** Independently rebuilt ledger has 5,250 unique `(query_id, arm_id)` operations (375 queries × 14 arms), 4,265 successful generated outputs, 985 failures, and 4,265 scored eligible outputs. Every arm contains the same 375 IDs. All 985 failures are logged serialized-prompt context-cap errors. No failure has a judge score. Run checksum file lists 14,891 package files; all listed files verified.

### 2. Is judge score reuse valid?

**Valid for duplicate exact input payloads.** 2,297 unique inputs scored once and 1,968 exact duplicate slots reused those scores. The digest/case ID is consistent with question, answerability label, reference, evaluator-visible context and cleaned output; duplicate score tuples agree; mapping/raw hashes reconcile. Global judge rubric and model/decode fields were fixed in the single run and are separately preserved, but not inside per-case key. No cross-run cache. Not a validation of judge accuracy; not an independent per-batch request replay.

### 3. Parent-deduplication result after adjustment?

**Strongest result, within matched-success cohort.** No-parent-dedup Semantic Accuracy +12.47 pp (N=296, query-paired 95% CI +7.88 to +16.97), all-52 Holm p=.000108; within-SA family Holm p=.000027. Faithfulness and Answer Relevance likewise survive broad Holm; Citation Fidelity does not (p=.481). Label as exploratory post-hoc and conditional, not definitive biomedical truth. Evidence concentration and context chunk metric inflation temper mechanism claims.

### 4. Dynamic GraphExpander result after adjustment?

**Mixed and correctly not “a component win.”** No-expander Semantic Accuracy +7.44 pp (N=296, CI +2.22 to +12.77), but SA adjusted p=.361 within metric and 1.0 across 52. Faithfulness +7.33 pp survives broad correction (p=.022); Citation Fidelity −12.80 pp survives (p=.00194). Availability improves from 296/375 to 375/375. Report trade-off.

### 5. Which effects differ by answerability?

No-expander SA +10.79 pp on answerable (N=248, CI +5.24,+16.27), −9.90 pp on designated-negative (N=48, CI −25.00,+4.58). Parent-off SA +14.96 pp answerable (N=248, CI +9.89,+20.02), −0.42 pp designated-negative (N=48, CI −9.06,+8.23). Negative labels are synthetic, small, not human-validated. Do not claim safety.

### 6. How serious are prompt overflows?

Substantial for full-cohort service availability (985/5,250 operations; B6 specifically 79/375). Not an absent outcome in the paired common-success analysis (all N=296 include B6 success) but selection restricts target to the jointly served subset. No-MCAR claim. Use availability and optional policy composite alongside conditional quality; never score missing answers as observed 0.

### 7. Can missingness be handled without experiment repeat?

**Transparently, yes; not eliminated.** Report N and coverage per arm, full-cohort generation availability, paired conditional quality, service-utility policy composite, and wide `[0,1]` potential-score bounds. State the score-policy formula and that it is not a judge score. These guard against misleading completeness claims; they do not recover missing answer quality.

### 8. Which are genuine execution-path interventions?

- Parent deduplication off: effective configuration changes and context/prompt outputs change in most paired cases.
- Dynamic GraphExpander off: actual expansion path disabled for all; dynamic output not inserted; all paired prompts differ.
- Concept retrieval off: branch disabled, but only one common prompt/output differs; low realized activation.
- Intrapaper candidates off: branch disabled, 24/294 common outputs differ.
- Static graph context off: flag changes but selected prompt branch never inserts static context; prompt/output exact ties.
- Fixed-budget context: branch flag executes but prompts and outputs exact ties and same overflow count; not evidence that final expanded serialization is bounded.
- Reranker×blending is a combined interaction and must not be attributed to either main effect.

### 9. Most defensible interpretations

1. Parent chunk deduplication was associated with higher conditional judge quality on matched successful answers in this cohort.
2. Dynamic graph expansion altered prompts, quality rubric scores, citations and coverage, with competing outcomes.
3. Evidence/context construction plausibly contributes; repeated target chunks increase chunk relevance and concentration, but do not establish mediation.
4. Entropy can move either direction alongside higher Semantic Accuracy; no universal monotonic relation or causal effect inferred.
5. Exact prompt/response no-op arms estimate no independent downstream quality difference in the observed support.

### 10. Claims not to put in reviewer response

- “The ablation causally explains the original historical B3–B5 quality gap.”
- “Graph expansion is harmful” or “graph expansion is beneficial” without citing Faithfulness/Citation/negative-query counterevidence.
- “All 375 queries have paired quality scores” or “missing answers scored zero.”
- “The negative cohort is clinically validated unanswerable” / “the system is clinically safe.”
- “The answer-quality judge was validated by humans,” “second judge confirms” or “judge reliability established.”
- “Fixed-budget intervention solved prompt overflow.”
- “Repeated target chunks caused higher answer quality.”
- “No effect means a component is unnecessary.”
- Large arm-level latency/speedup claims without comparable stage timing/cold-start treatment.

### 11. Evidence that can be submitted immediately

Standalone addendum, reviewer drafts, integrity report, all required analysis notes, full 5,250-row canonical ledger, 14-arm coverage table, paired metrics, both Holm tables, subgroup effects, service utility, overflow and missingness bounds, context provenance, entropy sensitivity, checksum/source manifest, and short rerun script. The pipeline and document source were regenerated from preserved artifacts. The manuscript and pre-existing supplement files were not edited.

### 12. What requires author sign-off

- Verify the official decision letter and exact quotation/numbering. Locally preserved response drafts are provisional; no official letter located.
- Confirm permission/format to attach this technical addendum without major manuscript rewrite. If journal requires these changes in manuscript body, that remains editorial risk.
- Confirm text/content from the final submitted manuscript and published supplemental records; no page/line number or manuscript-change statement is supplied.
- Explicitly acknowledge the new database snapshot differs in provenance from July identity, CUDA vs MLX, no new B3, single judge, synthetic negatives, and missingness.
- Decide whether to submit isolated saved examples; they are selected illustrations and not representative.

## Verification record

- `.venv/bin/python revision_ablation_defense/analyze_existing.py`: PASS; all integrity assertions true; regenerated 5,250-row ledger and tables.
- Focused tests (`test_backfill_logits.py`, `test_parse_metrics_pipeline.py`, `test_connector.py`, `test_statistics.py`): 43 passed; 2 environment/config warnings (soupsieve absent, pytest did not recognize `asyncio_mode` plugin option).
- `pyright --project pyrightconfig.json` on touched Python files: 0 errors, 0 warnings, 0 informations.
- `cd back && pyright --pythonversion 3.12 ...` on relevant RAG modules: 0 errors, 0 warnings, 0 informations.
- `python -m py_compile` on changed Python files: PASS.
- Scoped `git diff --check` on substantive changed code: PASS. Repository-wide `git diff --check` exits 2 due existing all-staged content under gitignore; scoped check passes.
- Full `make ci` was not run; user requested focused audit and explicitly discouraged unrelated CI. No model generation, judge API call, database rebuild or manuscript edit occurred.

## Reviewer-facing evidence summary (≤2 pages)

We performed an additional post-review within-run CUDA component-sensitivity analysis, using internal B6 as the reference corresponding to paper B5, and 13 component-off/interaction arms. The experiment used 375 frozen PubMedQA queries, a newly frozen `external_db` snapshot, the same generator/model revisions and decoder across arms. It planned 5,250 operations. 4,265 outputs were generated and scored; 985 operations failed before generation because serialized prompts exceeded the 12,000-token context cap. Failures were not scored or assigned zero answer quality. Every arm uses the same query IDs. The June/July historical MLX B3–B5 difference is not decomposed here: the runtime and database provenance differ, B6 maps to paper B5, and no new B3 arm was run.

The strongest conditional quality evidence concerns parent-chunk deduplication. Among the 296 query IDs where both B6 and the no-parent-dedup arm generated, Semantic Accuracy increased +12.47 percentage points (query-paired 95% CI +7.88 to +16.97). Holm-adjusted p=.000027 within the 13 Semantic Accuracy contrasts and p=.000108 across all 52 arm-by-metric contrasts. Answer Relevance (+9.26 pp) and Faithfulness (+10.20 pp) also survive the broad correction; Citation Fidelity (+9.36 pp) does not. This is a post-hoc single-judge sensitivity result conditional on observed common successes, not clinical ground truth or a proof that repeated chunks caused the score improvement.

The GraphExpander-off contrast is heterogeneous. On 296 paired successful IDs, Semantic Accuracy is +7.44 pp (CI +2.22 to +12.77), but not significant after Holm correction (SA-family p=.361; all-52 p=1). Faithfulness increases +7.33 pp (all-52 adjusted p=.022), while Citation Fidelity decreases −12.80 pp (all-52 adjusted p=.00194). The no-expander arm succeeds on all 375 planned queries vs B6’s 296/375, so output availability must be shown separately. Subgroup Semantic Accuracy changes are +10.79 pp among answerable pairs (N=248, CI +5.24,+16.27), and −9.90 pp among 48 synthetic designated-negative pairs (CI −25.00,+4.58). These negative labels have not been human-validated as corpus-wide unanswerables.

Evidence traces show no-parent-deduplication selects similar chunk counts but fewer unique PMIDs (mean 1.89 vs B6 4.75) and more chunks from the gold PMID (3.94 vs .94). The +55.96 pp chunk-level Context Precision increase can count repeated chunks; a source-deduplicated measure does not show more unique gold-source coverage. This makes evidence concentration/context construction a plausible hypothesis, not a demonstrated mediator. Some nominal interventions—static graph context off and fixed-budget context—produced identical serialized prompts/responses to B6 for all 296 common successful cases; they are downstream no-ops for this cohort, not evidence of zero general component effect. The fixed-budget arm did not reduce its 79 B6-matching overflow failures.

Saved CUDA telemetry uses entropy locally renormalized over top K logits (H20/H50/H100), not full-vocabulary Shannon entropy. Parent-dedup-off increases H20/H50/H100 while increasing conditional Semantic Accuracy; no-expander decreases H20/H50/H100 while its conditional Semantic Accuracy estimate also rises. This shows no universal monotonic entropy-quality direction in these observed contrasts; no causal entropy claim is made.

All outputs are standalone supporting materials; no manuscript files were modified. The exact official decision letter and final source/PDF were not available for page/line verification. Before submission, authors should check the official wording and confirm whether the journal accepts a technical addendum without a major manuscript rewrite. Evidence substantiates a nuanced component-sensitivity response, while historical attribution, human/second-judge validity, human validation of negatives, and generalization remain unresolved.
