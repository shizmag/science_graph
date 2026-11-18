# Response to Reviewer #1 — DRAFT, PENDING RESULTS

**Editorial instruction: this is not a completed-experiment claim. Do not submit until
judging, coverage review and analysis are completed and placeholders replaced.**

Thank you for identifying two distinct weaknesses: incomplete generator-evidence
presentation to the judges, and insufficient validation of Faithfulness and Citation
Fidelity, including the very low previously reported ICC values. We agree that neither
agreement on mean contrast direction nor the Semantic Accuracy pilot establishes
validity or absolute agreement for these other dimensions.

We have implemented and locally verified a separate historical reassessment protocol.
The original evaluator formatted retrieved chunks as enumerated blocks; this context
could omit post-trim graph material and the final enrichment block used by generation.
The archive retains `trimmed_text`, `trimmed_graph` and `enrichment_block`, together
with raw generations, persisted cleaned answers, configuration snapshots and run
metadata. Therefore new B3/B5 generations are unnecessary for evaluating the archived
answers against the archived evidence.

We recovered 160 answer–evidence records on the same fixed 80 answerable query IDs
used in the previous R2 experiment. Paper B3 corresponds to code B4 and paper B5 to
code B6. The archive records source commit
`b4f3322d8cae3395bcbf0b5a1f2e47a904d1591a`. Using the existing historical recovery
implementation, we verified source/run checksums and mapped the saved fields to the
historical prompt branches: 80 B3 records use `ask_no_expander` with trimmed text and
graph; 80 B5 records use `ask_expander` with the final enrichment block. The historical
empty-enrichment fallback is implemented and tested, although not needed in these
80 B5 records. We did not concatenate intermediate retrieval fields into evidence.
Reconstruction matched 100/100 existing corrected prompt fixtures, and deterministic
serialization and UTF-8 hashes were checked. Reference answers are retained separately
for provenance, not supplied to the judges.

These checks establish reproducible recovery of archived generator-visible evidence,
not byte-exact recovery of the original complete tokenizer input. The original input
serialization and exact historical corpus/index fingerprint remain unestablished.
Matching replay outputs cannot resolve that uncertainty. The cohort is conditional
on availability of paired answers and evidence, not an independent random sample of
all 300 answerable queries; conclusions will be limited accordingly.

The frozen protocol uses identical blind inputs, instructions and schema for the two
R2-configured judges, OmniRoute Luna (`cx/gpt-6-luna-low`) and OpenRouter DeepSeek
(`deepseek/deepseek-v4.1-flash-1017`). No configuration labels, gold answers, entropy,
old scores or other-judge outputs enter their messages. Faithfulness is aggregated
from atomic claim support (full=1, partial=0.5, unsupported=0), with explicit N/A for
claimless answers. Citation Fidelity separately records citation presence, source
resolvability, claim attribution and source support. Unresolvable or ambiguous targets
are never assigned an invented source. In particular, B5 Block-N references are not
mapped to the Nth retrieved chunk unless the archived visible evidence actually
establishes that mapping. The primary required-citation policy and the no-citation-N/A
and resolved-only sensitivity analyses are fixed before judging.

Every request, response and failure will be retained with hashes and provider metadata;
bounded retries accept only the first valid response and do not select favorable scores.
There are 320 planned evaluations. **PENDING RESULTS: currently zero paid requests and
zero accepted judgments under this protocol.**

For each metric the primary comparison will use exactly those query IDs with all four
applicable B3/B5 × judge scores. Judge-specific available sets and reasons for exclusion
will also be reported. The deterministic saved-data analysis computes means, paired
B3−B5 contrasts and 95% query-bootstrap intervals using 10,000 resamples. Agreement
is assessed separately for individual response scores, paired contrasts, and
applicability status, including ICC(A,1), Spearman and systematic judge differences.
The bootstrap preserves the two configuration responses within each query.

**PENDING RESULTS:** insert common-set sizes, each judge's F/CF means and paired
contrasts with intervals, ICC(A,1) and Spearman with intervals, applicability agreement,
citation diagnostics, missingness, and sensitivity results here. State explicitly
whether ICC remains low. If new F results differ from R2, diagnose rubric, evidence
and applicability differences; do not suppress contradictory findings. No directional
agreement will be presented as sufficient validation of individual scores.

The implemented supplement documents provenance and remaining limitations. The
present package closes the technical preparation and evidence-identity checks, but
the reviewer's request for empirical F/CF reassessment remains open until the independent
judging and analysis are performed. Additional human validation may still be warranted,
particularly if absolute agreement remains poor.
