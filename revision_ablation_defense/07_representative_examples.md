# Saved qualitative examples

## Selection protocol

Examples are selected by deterministic scan of frozen `query_id` order and explicit, predeclared case properties: first high-contrast answerable parent-dedup case where no-dedup adds selected chunks from the gold paper; first answerable no-expander pair with higher judge Semantic Accuracy; first designated-negative no-expander pair with lower judge Semantic Accuracy; first exact generation no-op from concept-off; first answerable B6 prompt overflow where no-expander succeeds. These are illustrations only, not population samples. Excerpts are literal saved text snippets; ellipses mark truncation by this report.

## 1. Parent deduplication: target paper appears repeatedly

**Query:** `pubmedqa_10173769` (source PMID 10173769, answerable).
**Question:** “Longer term quality of life and outcome in stroke patients: is the Barthel index alone an adequate measure of outcome?”
**Gold:** “Yes. The place of the Barthel index as the standard outcome measure for populations of stroke patients is still justified for long term follow up, and may be a proxy for different outcome measures intended for the assessment of other domains.”

**B6 evidence:** top excerpt from gold PMID `...#0`: “To consider whether the Barthel Index alone provides sufficient information about the long term outcome of stroke.” Other selected PMIDs include lung cancer quality-of-life demographics and colorectal process indicators; four unique selected PMIDs across five chunks.
**No-dedup evidence:** gold PMID chunks `#0`, `#6`, `#3` (three of five chunks; only three unique PMIDs overall). Gold `#0` is the context/aim excerpt; other excerpts provide additional portions from same source paper.
**B6 answer:** “The text does not provide a definitive answer … It only mentions the index was used in the study.”
**No-dedup answer:** “The Barthel index is justified as a standard outcome measure for long-term follow-up … though it may be a proxy for other measures.” It includes `[Block 1]` citation.

| Arm | Semantic Accuracy | Answer Relevance | Faithfulness | Citation Fidelity | H50 bits | Prompt tokens | Parser |
|---|---:|---:|---:|---:|---:|---:|---|
| B6 full | .25 | .50 | .70 | 0 | 1.176 | 6,944 | UNKNOWN |
| no parent dedup | 1.00 | .95 | 1.00 | 1.00 | .924 | 3,679 | ANSWERABLE |

This example shows added target-PMID chunk coverage and lower entropy with improved judged quality, opposite entropy direction from the cohort-level parent-off paired mean (H50 +.045 bits). It is selected by outcome contrast and not representative. More chunks from one gold paper may inflate chunk-level Context Precision and reduce source diversity; it does not establish that repetition causes quality gains.

## 2. GraphExpander: answerable pair with higher Semantic Accuracy after disabling

Same frozen query `pubmedqa_10173769`. No-expander answer: “The Barthel index is considered an adequate measure for long-term follow-up, though it may be a proxy for other domains.” Judge scores: B6 `.25/.70/0/.50`; no-expander `.90/.90/0/.90` (SA/Faithfulness/Citation/ Relevance). B6 dynamic block begins with unrelated Kosmos graph-enrichment material; no-expander serialized prompt is 3,065 tokens vs B6 6,944. Selected evidence chunks remain the same. This illustrates a possible noise/context mechanism, but the judge’s per-case rationale and source context remain model-assessed; prompt content is not a randomized mediator.

## 3. GraphExpander: designated-negative counterexample

**Query:** `pubmedqa_unans_10430303`; synthetic category `unanswerable-missing-alternative-intervention`, source PMID 10430303. Question asks how a cholecystectomy study compares with alternative therapies or unexamined control cohorts. Gold reference is the preserved empty string for this negative. B6 receives SA 1.0, Faithfulness 1.0, Citation 0; it marks the requested external comparison UNANSWERABLE. No-expander receives SA 0, Faithfulness .2, Citation 0; it provides an unsupported comparison-style outcome. This is one saved example of a possible cost of removing expansion on a designated negative, not evidence of true corpus-level unanswerability or safety. Do not call `UNANSWERABLE` parser status human-validated abstention.

## 4. Near-no-op intervention counterexample

`B6_minus_concept_retrieval` disables the configured concept lookup; trace confirms lookup branch did not execute. Across 296 common successful responses, 295 prompt hashes and response hashes equal B6. Query `pubmedqa_15112004` is the first frozen-ID case where the no-concept prompt differs; both arms retain the same relevant chunks and answer hash remains identical, with only a tiny judge-score difference. Population-level no-effect claim is not justified; say intervention had little downstream activation in this cohort and many exact-input ties.

## 5. Prompt-overflow example

**Query:** `pubmedqa_10375486` (answerable): “Are variations in the use of carotid endarterectomy explained by population Need?”
B6 retrieves five chunks and assembles 10,576 context-text characters plus 1,371 graph-context characters. Its saved dynamic expansion output is 210,710 characters. Trace says expansion executed but was not inserted in the final prompt; generation fails before an output: “Serialized prompt has 51004 tokens … (12000).” No-exansion succeeds with 3,183 prompt tokens and a judge score, but that output is not paired B6 quality evidence because B6 has no answer. This identifies a prompt-path/size association in a saved trace, not definitive proof that the 210,710-character expansion was serialized as that failed prompt. Exact token accounting comes from the runtime error; do not claim a single prompt component caused the overflow.
