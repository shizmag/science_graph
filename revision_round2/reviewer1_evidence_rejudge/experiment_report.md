# Historical full-evidence F/CF reassessment — implementation report

## Status and objective
**IMPLEMENTED AND VERIFIED / READY FOR API EXECUTION / PENDING EXPERIMENTAL RESULTS.**
Address Reviewer #1 by assessing identical archived B3/B5 answers with two independent
judges using the full recoverable archived generator-visible evidence, rather than
retrieved-chunks-only evaluator contexts. No paid judging or new generation occurred.

## Historical provenance, cohort and evidence
Archive `run_20260729_211324_OCC-RAG-1.7B`, recorded source commit
`b4f3322d8cae3395bcbf0b5a1f2e47a904d1591a`. Paper B3 maps to code B4, B5 to code B6.
The original R2 packet is checksum-verified. Its fixed80 answerable cohort is retained;
seeded eligible-pair selection was reproduced. Availability-conditioned selection is
an explicit limitation, not independent random sampling of all 300 answerable queries.

160/160 answer–evidence records verified; zero evidence exclusions, zero empty cleaned
answers, all archived generations successful. Raw and cleaned answers remain separate.
80 ask_no_expander and 80 ask_expander paths are verified against archived fields and
historical templates. 100/100 corrected prompt fixtures match. Three serializations
match. See `provenance_checks.json` and `provenance_validation.md` for evidence boundaries.
Archived evidence is not synonymous with byte-exact original tokenizer input.

## Independent judges and frozen rubric
OmniRoute `cx/gpt-6-luna-low` and OpenRouter
`deepseek/deepseek-v4.1-flash-1017`, matching R2 v3 configuration. Provider identifiers
are not independently verified weight versions. Temperature 0, max_tokens 12000,
no OpenRouter fallback. Same instructions/schema and same per-answer visible inputs.
Gold, configuration labels, previous scores and entropy are excluded.

F uses atomic claim support weights 1/0.5/0. Citation Fidelity uses per-citation support,
resolvability and explicit source spans, never guessed Block numbering. Primary CF
penalizes missing required citations and unresolvability operationally, while recording
separate statuses. No-citation-N/A and resolved-only sensitivities are prespecified.
Coverage and compliance remain distinct. Claimless and empty answers are not numeric
factual zeros. Frozen details: `protocol.md`, `judge_protocol/`, `runner/scoring.py`.
Protocol hash: see `local_dry_run.json`; all normative files in `freeze.json`.

## Request retention and execution
320 planned tasks, no calls made, zero accepted evaluations. Expected 320 first-pass
requests, bounded at four attempts per task (1280 maximum). Exact requests, raw responses,
hashes, usage, metadata, validation failures and timestamps are retained. The first
valid response wins; checkpoints and attempts are immutable, with crash reconciliation
and exclusive execution locking. Credential echoes trigger a security stop.

## Local validation
28 offline tests pass (see `local_test_results.txt`). They cover 160-input identity,
hashes, blinding, same inputs across judges, partial support, absent/incorrect/mixed/
unresolvable/ambiguous citations, claimless and empty outputs, invalid schema/JSON,
HTTP 429 and network retries, bounded attempts, restart, crash recovery, uncertain
requests, model mismatch, checkpoint tampering, fallback branches, and synthetic
same-ID/cluster-bootstrap statistics. An unrelated repository pytest configuration
warning (`asyncio_mode`) is present; it does not fail these tests.

## Paired results, reliability and missingness
**PENDING RESULTS**. No F/CF means, B3−B5 differences, ICC or Spearman estimates exist
under this protocol yet. Current analysis output explicitly records planned=320,
accepted=0 and missing judge results; missing results are not zeros. Historical R2
80-SA/78-F applicability counts are not imposed on the new protocol.

The saved-data-only analysis reports each metric's four-score common query set,
judge-specific available sets and exclusions, means and paired differences, 10000
query-bootstrap percentile CIs (seed 20261116), ICC(A,1), Spearman, systematic judge
bias and paired-difference agreement. Individual response bootstrap clusters both
configurations by query. Applicability agreement is separate with cluster-bootstrap CI.
Constant/undefined reliability estimates remain undefined with valid-resample counts.

## Citation diagnostics and sensitivities
**PENDING RESULTS**: no-citation rate, missing-required frequency, resolvability,
resolved incorrect attributions, ambiguity, coverage, compliance, primary CF,
resolved-only CF and no-citation-N/A CF. Unresolvable is not relabelled factually wrong.

## Limitations and next steps
- Complete archived evidence recovery does not prove original model-input identity.
- No exact historical corpus/index fingerprint is established.
- Fixed80 is availability-conditioned; generalization beyond it is limited.
- LLM claim segmentation and citation extraction can be imperfect. Schema, substring
  checks and mean-contrast agreement do not establish semantic validity.
- Provider versions may drift behind named model IDs; returned identities are retained.
- Citation ambiguity may be structurally more common in enrichment evidence; report
  it rather than creating an invented target mapping.
- Low ICC, conflicting directions, N/A changes or disagreement with R2 must be reported
  and diagnosed, not discarded. A human adjudication study remains a possible next
  validation step, not claimed completed here.

After authorization execute the README command, inspect coverage and failures, run
the deterministic analysis, then replace PENDING RESULTS in manuscript/reviewer drafts
only with saved-data estimates. No current cost tariff was verified; obtain provider
pricing and a spending decision before execution. Public release needs manual security
and licensing review in addition to the supplied automated screening.
