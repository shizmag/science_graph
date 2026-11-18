# Proposed Supplementary Material — DRAFT, PENDING RESULTS

## Historical Generator-Visible Evidence Recovery and Independent Faithfulness/Citation Reassessment

We constructed a historical reassessment without regenerating candidate answers or
performing new retrieval. The archived run `run_20260729_211324_OCC-RAG-1.7B` records
source commit `b4f3322d8cae3395bcbf0b5a1f2e47a904d1591a`. Paper configurations B3 and
B5 map to implementation configurations B4 and B6, respectively. The cohort comprises
the fixed 80 answerable IDs from the prior R2 packet, conditional on availability of
paired answers and recoverable evidence. It is not a fresh random sample from all
answerable queries.

For B3, the historical no-expander branch receives archived `trimmed_text` and
`trimmed_graph`. For B5, the expander branch receives `enrichment_block`; the historical
fallback uses the no-expander branch when enrichment is empty or the sentinel
“No essential knowledge graph enrichment found.” All 80 selected B5 responses use
the expander branch. Intermediate retrieved chunks were not appended to either branch.
Raw generation and persisted cleaned answer identities were preserved separately.
The original retrieved-chunks-only evaluation context could omit this final graph or
enrichment evidence, motivating reassessment.

All 160 records passed source-to-prompt mapping and hash checks, with no exclusions
or empty cleaned answers. Historical-template reconstruction matched 100 existing
corrected replay prompt fixtures. The prompt manifest records deterministic UTF-8
hashes and three identical serializations. This verifies archived evidence recovery
and reproducible prompt reconstruction, not the original complete tokenizer input.
The original serialized model input and exact historical index fingerprint remain
unestablished; output matches do not demonstrate input-byte identity.

The prespecified protocol uses the two prior R2 judge configurations, OmniRoute Luna
and OpenRouter DeepSeek, with identical per-answer blinded messages and schemas.
Reference answers, configuration names, previous evaluations and entropy are excluded.
Faithfulness is the mean atomic factual-claim support with weights 1, 0.5 and 0 for
full, partial and unsupported claims. Claimless outputs are N/A. Citation Fidelity
scores explicit citation attributions against uniquely identifiable archived sources;
missing citations and unresolved targets are separately flagged. Primary CF assigns
operational zero to required-but-absent citations and unresolvable targets, without
calling ambiguity a demonstrated factual error. Resolved-only and no-citation-N/A
sensitivities, Citation Coverage, Citation Resolvability and Citation Compliance are
reported separately. No B5 citation mapping is inferred from retrieved-chunk order.

Each accepted evaluation and all retry attempts are retained. Primary comparisons
use metric-specific common IDs with all four applicable scores. Confidence intervals
use 10,000 query-level bootstrap resamples, seed 20261116. ICC(A,1), Spearman,
systematic interjudge differences and applicability agreement distinguish individual
score reliability from agreement on paired B3−B5 differences. Both configuration
responses remain together in bootstrap clusters.

### Tables to populate only from saved-data analysis

| Metric | Common queries | Luna B3 / B5 | Luna paired Δ [95% CI] | DeepSeek B3 / B5 | DeepSeek paired Δ [95% CI] |
|---|---|---|---|---|---|
| Faithfulness | PENDING RESULTS | PENDING | PENDING | PENDING | PENDING |
| Citation Fidelity | PENDING RESULTS | PENDING | PENDING | PENDING | PENDING |

| Metric / unit | ICC(A,1) [95% CI] | Spearman [95% CI] | Systematic judge difference | Applicability agreement |
|---|---|---|---|---|
| F / individual responses | PENDING | PENDING | PENDING | PENDING |
| F / paired differences | PENDING | PENDING | PENDING | — |
| CF / individual responses | PENDING | PENDING | PENDING | PENDING |
| CF / paired differences | PENDING | PENDING | PENDING | — |

Citation diagnostic and sensitivity tables must include absent citations, unresolved
and ambiguous targets, resolved incorrect citations, coverage, compliance, resolved-only
CF and no-citation-N/A CF, together with their denominators. Missingness tables must
separate generation failure, empty answer, evidence failure, judge failure and metric N/A.
Currently planned=320 and accepted=0; no new empirical reliability conclusion is available.

### Interpretation constraints
Directional agreement on the cohort does not validate individual scores. Low ICC must
be reported without qualification by desired contrast direction. Cohort selection,
LLM claim/citation extraction, provider version uncertainty, historical input identity
and citation ambiguity limit inference. Any disagreement with previous R2 F results
requires explicit rubric/evidence/applicability diagnostics. Additional human validation
is not supplied by this experiment and may remain necessary.

## Main-text changes after results
Replace any claim that the original pilot validated F/CF with the actual new estimates
and limitations. Cite this supplement for archived evidence recovery and common-ID
comparisons. Do not replace the earlier low ICC values with unperformed results.
