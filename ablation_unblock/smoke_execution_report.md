# One-query B6 smoke execution

## Result

`SUCCESS`: the local B6 pipeline completed a genuine retrieval and generation for
the frozen fixture. Process exit code was 0; all smoke validation gates passed;
the external/cloud evaluator was skipped. The machine-readable run manifest is
[`smoke_run/run_manifest.json`](smoke_run/run_manifest.json), and all 19 smoke
files are preserved in [`smoke_results.tar.zst`](smoke_results.tar.zst).

## Query and answer

- ID: `pubmedqa_21084567` (frozen canonical row 138), answerable.
- Question: “Are stroke patients' reports of home blood pressure readings
  reliable?”
- Gold reference: “Yes. This group of elderly stroke patients were able to
  record their BPs reliably at home. Any bias was small and would be unlikely to
  affect management. Since BP readings in a GP surgery are often a poor
  indication of true BP, GPs might consider using hypertensive patients' records
  of home BP monitoring to help guide treatment decisions.”
- Parsed result: “The study provides evidence suggesting that stroke patients
  can reliably record their home BP readings.” Parser status was `ANSWERABLE`;
  B6 generation status was `success`. The raw model response is preserved in
  `smoke_run/generation.json` and the actual final prompt in `smoke_run/prompt.txt`.

## Database and model identity

- Baseline: author-selected `external_db`, snapshot ID
  `external_db-20261016T193200Z`.
- Selected snapshot SHA-256:
  `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`.
- Restored, pre-run SQLite SHA-256:
  `bc54d962762d0ec9c150853f10a4fed7b4cec1fa42cdef064ec2d87a9fb7b79d`.
- The worker startup log and pipeline config snapshot record the same resolved
  restored-copy path. SQLite changed the clone's physical hash while it was
  opened (`c4884514b938ff5739bf61f72f01eb628349840c67d8ac4a47d4cc30383632ce`),
  but post-run integrity, FTS5, vectors, and the frozen logical corpus summary
  all matched. The selected snapshot file itself remained unchanged.
- The selected corpus has 346 documents, 22,068 chunks/vectors (768 dimensions),
  9,680 graph nodes, and 11,712 graph edges. July database identity remains
  unproven.
- The locked local models were `occ-ai/OCC-RAG-1.7B`,
  `intfloat/multilingual-e5-base` revision
  `d128750597153bb5987e10b1c3493a34e5a4502a`, and
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` revision
  `1427fd652930e4ba29e8149678df786c240d8825`.

## Retrieval, ranking, and graph evidence

The E5 embedder ran on MPS; FTS5 lexical and dense retrieval each produced 20
seed chunks. Ten candidates reached reranking; the Cross-Encoder ran on MPS and
scored them. Five chunks remained in final retrieval and all five were visible
to the generator. The expected PubMed paper is the first final chunk. Using the
single distinct expected paper ID and persisted output gives Recall 1.0 and
Context Precision 1.0. This is one positive fixture, not a 375-query benchmark.

Graph evidence is reported separately:

- The trace lists 322 graph candidate papers. The stage log shows hop 1 selected
  2 of 29 neighbors and hop 2 selected 1 of 7.
- Zero graph-neighbor chunks survived into final context; the final-context
  trace contains five seed/retrieval papers.
- The retrieval artifact has 10 graph relationships and 1,844 characters of
  trimmed graph text, but the captured generator prompt does not contain that
  graph block. The persisted trace says no graph-neighbor chunk survived.
- `graph_neighbors_in_rrf` was enabled in the B6 component configuration, but
  this run has no counterfactual ranking trace establishing a causal ranking
  change. Graph expansion execution/candidate production is not treated as
  effective final-context graph evidence.

## Stage results and timing

| Stage | Evidence and timing |
|---|---|
| Query expansion | Local MLX model loaded; four structured-output attempts failed with the same unterminated JSON string at character 954. Pipeline fell back to the original question and continued. |
| Dense retrieval | E5 initialized on MPS (32.538 s); dense stage 0.115 s; 20 seeds. |
| Lexical retrieval | FTS5 stage 0.150 s; 20 seeds. |
| Cross-Encoder | Initialized on `mps:0` (1.428 s); 10 candidates reranked; 0.757 s across recorded calls (the worker log reports a 10-pair batch at 0.25 s). |
| Graph expansion | Two hops; 0.543 s component time; no graph-neighbor chunk survived final context. |
| Generation | Local MLX generator; 12.118 s component time and 11.343 s captured generation call; one response persisted. |
| End-to-end timing | B6-reported latency 13.891 s; retrieval-reported latency 1.771 s; full smoke process wall time 298.079 s including restore, setup, cold model initialization, and stage work. |

Available compact telemetry is preserved in `smoke_run/telemetry.json`: 870
reported prompt tokens and 223 token records. No independent full-vocabulary
entropy or logits were synthesized. The parser result is reported separately
from generation status.

## Archive verification

`smoke_results.tar.zst` SHA-256 is
`c0b091bcf05db718849efd3322b8a5f48c302d07f8fb364bb21d575c87806e61`. The package
verifier reported `valid: true`, `smoke_completion_status: SUCCESS`, and 19
files. A clean extraction roundtrip reproduced all 19 payload files byte for
byte. The extracted copy is retained in `ablation_unblock/archive_roundtrip/`.
