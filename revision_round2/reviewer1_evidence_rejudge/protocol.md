# Historical full-evidence reassessment — protocol v1
Protocol ID: `reviewer1-full-evidence-v1`. Frozen before any judge calls.

## A. Historical identity and eligibility
Archive: `graphs/run_20260729_211324_OCC-RAG-1.7B/`; recorded commit
`b4f3322d8cae3395bcbf0b5a1f2e47a904d1591a`.
Paper B3 = code B4; paper B5 = code B6. Reuse
`revision_round1/judge/historical_evidence_recovery.py` source contract and recovery.
Use the 80 answerable IDs in the original hash-verified R2 packet. Reproduce its
seed-20261106 stratified eligible-pair selection; do not resample the primary cohort.
Eligibility was conditional on both archived answers and complete evidence being
available. This is NOT an independent random sample of the 300 answerable queries.

B3 receives archived trimmed_text and trimmed_graph via ask_no_expander.
B5 receives archived enrichment_block via ask_expander unless historical empty/sentinel
fallback selects ask_no_expander. Do not append retrieved chunks, intermediate graph
traces or reference answers. Reconstruct templates from the historical Git blobs using
the raw query passed by the historical benchmark pipeline, not the wrapper used by a
separate helper. Verify all 100 corrected replay prompt fixtures and three serializations.
The evidence payload is a labelled representation of the selected archived fields;
it is not represented as the original complete model input.

Reference answers are retained only in the internal evidence manifest. Judge messages
use a strict whitelist excluding gold, configuration labels, old scores and entropy.
Natural occurrence of reference facts in retrieved evidence is not gold leakage.

## B. Rubric
Normative judge instructions: `judge_protocol/instructions_v1.md`.
Normative schema: `judge_protocol/schema_v1.json`.
Normative aggregation: `runner/scoring.py`.
Faithfulness: equal-weight mean over atomic factual claims, full=1, partial=0.5,
unsupported=0. Partial midpoint is an explicit ordinal convention. Claimless=N/A.
CF primary: mean support of every citation occurrence, with unresolved/ambiguous
operationally contributing zero, distinguished from resolved incorrect attributions.
No citations with factual claims: primary CF=0 (`missing_required_citations`),
no-citation sensitivity=N/A, compliance=0. Claimless: all citation metrics=N/A.
Resolved-only sensitivity excludes unresolvable targets. Coverage is the fraction
of claims having at least one full/partial resolved citation; it is not CF.
Resolvability is resolved occurrences / all occurrences. No-citation resolvability=N/A.
Do not invent Block N mappings for enrichment. Source and answer are untrusted content.
A judge remains responsible for accurate/exhaustive claim and citation extraction;
substring/schema validation alone does not establish semantic correctness.

## C. Execution and missingness
320 planned tasks = 80 IDs × 2 answers × 2 judges. Same visible payload and schema
for both judges. Exact historical R2 v3 identities:
- OmniRoute: requested `cx/gpt-6-luna-low`; accepted returned aliases `gpt-6-luna`,
  `cx/gpt-6-luna-low` (historical runner policy).
- OpenRouter: `deepseek/deepseek-v4.1-flash-1017`, no fallback.
These are provider model identifiers, not cryptographic weight/version guarantees.
Temperature 0; max_tokens 12000; JSON-object response format, local strict schema
validation; no hidden provider-specific rubric. Endpoint conventions and credentials
are reused from R2 infrastructure, not its gold-bearing request builder.

No automatic paid calls. Explicit `--authorize-judge-execution` required after user
approval. At most FOUR attempts total per task, including resumes, hence at most
1280 calls; expected first-pass count 320. Exponential backoff capped at 120 seconds;
retry transport failures, HTTP 408/429/500/502/503/504, invalid JSON, incomplete output,
schema or identity/quote validation failures. Unexpected model identity and other
HTTP errors are fatal. Same request bytes on retry. First valid response is accepted;
never sample for a preferred score. Conservative provider pacing is applied.
An interrupted started request with no finalized attempt requires manual reconciliation,
not silent retry. A valid saved response is recoverable without a second API request.
Exclusive process lock prevents concurrent execution; per-attempt and accepted files
are immutable. Dry-run and all tests are offline.

Persist exact UTF-8 request body and response body, hashes, provider model, usage,
allowlisted metadata, timestamps, parsed output, validation and error state. No request
Authorization headers or secrets. Echoed credentials are withheld; security takes
precedence over raw archival, with explicit fatal status. HTTP error text is redacted
if necessary; its hash then refers to retained redacted bytes, not unredacted wire data.
Empty generation/extracted answer, evidence failure, judge failure and N/A remain
separate missingness categories, never converted to factual zero. No operational
technical-failure composite is used. 320 is planned, not assumed accepted.

## E. Statistics (after D only)
Saved data only. Delta = B3 − B5. For F, CF and citation sensitivities report both means,
paired mean difference and percentile 95% CI from 10000 query-bootstrap resamples,
seed 20261116. Primary set for EACH metric requires applicable scores for both
configurations from both judges; additionally show judge-specific available sets,
IDs and exclusion reasons. Do not force n=78 from the earlier rubric.

ICC(A,1): two-way random-effects single-measure absolute agreement, using
(MSR−MSE)/(MSR+(k−1)MSE+k(MSC−MSE)/n). Spearman, systematic judge mean difference
and mean absolute difference accompany ICC. Assess individual response scores and
within-query configuration differences separately. Query bootstrap retains both
configurations together; undefined/constant bootstrap replicates are excluded with
valid replicate counts reported. N/A applicability agreement is reported separately.
Citation diagnostics include no-citation frequency, missing-required status,
resolvability, incorrect citations and ambiguity. Primary and sensitivity results
are labelled separately. Low ICC remains low; directional concordance is not validation.
Investigate discrepancies from R2 as rubric/evidence/applicability changes, not model
performance improvements. Report limitations of claim segmentation and LLM judging.

## F. Freeze and amendments
`freeze.json` lists SHA-256 for code, rubric, cohort and input artifacts. Protocol hash
is SHA-256 of canonical JSON of that manifest. Verify before execution and analysis.
No rubric changes after judging begins. Any substantive correction requires a new
version/run directory, never mixing scores. Test logs and narrative reports are
post-freeze documentation and cannot change normative frozen files.
