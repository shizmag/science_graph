# Component mechanisms, prompts, context provenance and latency

## Parent-chunk deduplication off

The intended branch change is present in the effective configuration (`rag_components.deduplicate_parent_chunks=False`), and selected evidence/prompt hashes differ from B6 for 274/296 paired successes. Results show a consistent **evidence concentration** pattern on the paired set:

- Both arms select about five chunks/query.
- B6 selects mean 4.75 unique PMIDs; no-dedup selects 1.97.
- Mean gold-PMID chunks rise from 0.94 to 3.86. Paired gold-PMID presence is unchanged: 279/296 cases retrieve the gold paper in each arm, and all 296 pairs tie on this binary hit indicator. The change is repeated chunk multiplicity, not additional unique gold-PMID coverage.
- Paired gold-chunk-count change is +2.92 chunks/query; its correlation with per-query Semantic Accuracy difference is r=.104 (descriptive, weak). The source-PMID hit indicator does not change in the paired cases.
- Chunk-level Context Precision changes +55.96 pp. It is mean average precision over ranked chunks with the gold paper treated as relevant, so multiple chunks from one target PMID count repeatedly. After deduplicating ranked evidence by `paper_id` and preserving first occurrence, the document-level AP difference is 0.00 pp for this data’s one-gold-PMID queries; document-level AP does not support the large chunk-precision increase as more unique-source coverage.
- Overall arm means: unique selected PMIDs 4.75→1.89; gold-paper selected chunks .94→3.94; mean input context text characters 9,984→9,936; mean serialized prompt tokens 5,833→2,802 (success-only, not paired same query composition); B6 dynamic output averages 22,874 characters, no-dedup 9,990.

Paired mean Semantic Accuracy change and gold-chunk-count change are associated only weakly (r=.104). Removing dedup may alter which chunks survive context construction and allows repeated target-paper excerpts to displace unrelated low-ranked items; that is plausible but does not establish repetition as the causal answer-quality mechanism. Strongest population claim is the matched-success judge-score contrast; source-level evidence diversity actually falls.

## Dynamic GraphExpander off

The component-off path is real: effective config disables `rag_components.graph_expansion`; per-query trace records expansion not executed and dynamic enrichment not inserted for all 375 queries. Compared with B6, mean prompt length falls (success-only 5,833→3,188 tokens) while selected chunk counts and source metrics are broadly similar. Paired on 296 B6 successes, exact serialized prompt hash and response hash match B6 in 0/296 cases, so this is not an identical-input no-op. The interaction arm also disables graph candidate expansion, so it is not a clean independent “expander alone” contrast.

The effect is heterogeneous:

- B6 full: dynamic branch expands candidates and inserts enrichment in all 296 successful examples; its mean saved dynamic-output field is ~22,874 chars.
- Disabling expansion removes that insertion, shortens prompt, and eliminates all 79 B6 overflows; it improves matched conditional Semantic Accuracy +7.44 pp (not Holm-significant) and Faithfulness +7.33 pp (Holm-significant), but reduces Citation Fidelity −12.80 pp (Holm-significant).
- Answerable group Semantic Accuracy rises +10.79 pp (N=248); designated-negative point estimate falls −9.90 pp (N=48, CI crosses zero).

In PubMedQA 10173769, B6’s top selected chunk from the gold PMID contains only study context/aim (“To consider whether the Barthel Index alone provides sufficient information…”), while B6 also inserts a long unrelated Kosmos graph-enrichment block; the gold excerpt supplied to the judge does not show the reference’s result sentence. Removing expansion leaves the same five selected chunks but omits that large enrichment block; answer turns correct and citation fidelity declines because generated answer has no matching citation. This is an illustrative, judge-scored case, not a population explanation.

### Overflow forensic review

For B6, overflow failures average 33,132 serialized-prompt tokens, range 12,278–74,605; successful prompts average 5,833. All B6 overflow records execute the GraphExpander but trace `dynamic_graph_output_inserted=false` on the failed-generation path. In the earliest frozen answerable overflow example (`pubmedqa_10375486`), the question is “Are variations in the use of carotid endarterectomy explained by population Need?”, retrieved evidence includes five chunks (one target PMID), context text is ~10,576 characters, context graph ~1,371 characters, but saved dynamic expansion output is ~210,710 characters and the attempted serialized prompt is 51,004 tokens. The failed prompt was rejected before any generation token, parser output, or judge score. Dynamic payload length is not itself proof it was inserted into the final prompt on failure; traces explicitly say it was not inserted, while prompt-overflow exception records the oversized serialized prompt. The exact failure-stage serialization logic should not be inferred beyond those records.

The trace shows ordinary semantic context trimming (`context_trimming=true`) executes before dynamic expansion, but its trim metadata bounds assembled text/graph context, not the later appended dynamic enrichment block. The fixed-budget arm changes `context_trimming=false` and reports its deterministic tokenizer-prefix guard executed on all 375 rows, but its successful prompts and responses are exactly equal to B6 for all 296 common successes, and its failure count remains 79. Therefore the saved records do **not** show that its guard bounded the final serialized expanded prompt. Do not claim a verified prompt-packing fix. No code changes were made to packing and no generation was rerun.

## Near-zero / no-op interventions

| Arm | Trace evidence | Same B6 prompt / response among B6-common successes | Classification |
|---|---|---:|---|
| `B6_minus_static_graph_context` | Effective config false, but static context not inserted in the selected prompt branch; dynamic expander branch used. | 296/296 and 296/296 | Execution-configuration intervention, but no downstream input change for this workload; observed no-op. |
| `B6_fixed_budget_context` | Policy flag executed and config disables ordinary trimming, but final prompts are unchanged. | 296/296 and 296/296 | Non-informative for answer-quality effect here; not evidence fixed-budget safety works. |
| `B6_minus_concept_retrieval` | Concept lookup disabled on trace; only 1 of 296 shared prompts/responses changes. | 295/296 and 295/296 | Genuine upstream branch-off but low activation/near-no-op in this cohort. |
| `B6_minus_intrapaper_graph_candidates` | Intra-paper candidate branch disabled; 24 of 294 shared prompts/responses change. | 270/294 and 270/294 | Genuine but low-yield intervention; quality differences small. |
| `B6_interaction_reranker_x_blending` | Combined path differs, but this is a joint interaction, not a component-only effect. | 30/247 and 30/247 | Genuine combined intervention; do not attribute to either component alone. |

Static context and fixed-budget rows are exact downstream-input no-ops on the B6-success support. Their observed zero score deltas are not independently estimated nonzero effects. No-effect arms show low cache hits on actual generation inputs for these unchanged examples; cached cross-encoder/graph computations primarily affect upstream runtime. Do not claim large latency speedups from arm-level end-to-end means without comparable per-stage timing and initialization adjustment.

## Latency provenance

The top-level elapsed time is a pipeline wall-clock measure and mixes retrieval, graph expansion, reranking, generation, prompt-overflow rejection, and cache effects. Several arm-level stage timing summaries are empty, and initialization/cold-start comparability is not complete. Graph/cross-encoder cache counters differ across arms. Treat runtime as descriptive only; omit speedup claims from the reviewer response. No inference was rerun to obtain timing.
