| Gate | Status | Evidence |
|---|---|---|
| Canonical 375 recovered | PASS — `EXACT_SOURCE_RECOVERED` | July `retrieved_contexts.yaml`; 375 unique IDs, 300 answerable, 75 designated negatives. |
| Manifest frozen | PASS | `questions.jsonl` SHA-256 `d2820ced7c2994218245887ad753700923262fb91ec983eeacc01b217ec43b6b`; deterministic extraction and dataset `SHA256SUMS`. |
| Database baseline identified | PASS, as new baseline only | Author-selected `external_db`, snapshot SHA-256 `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`; July database identity remains unproven. |
| Source article coverage | PASS | All 375 frozen query sources resolve to exact paper IDs and chunked records in indexed `external_db` metadata. |
| Zero Recall explained | UNRESOLVED for the supplied report | Exact all-zero/37.566 report and its query-level trace are absent from scoped artifacts; see `zero_recall_diagnosis.md`. |
| Real retrieval verified | PASS for the frozen fixture | 20 lexical and 20 dense seeds; five final chunks; expected source at rank 1; recomputed Recall and Context Precision are 1.0. |
| B6 generation verified | PASS | Local MLX answer generated and persisted; parser status `ANSWERABLE`; B6 status `success`. |
| Smoke archive verified | PASS | `smoke_results.tar.zst`, SHA-256 `c0b091bcf05db718849efd3322b8a5f48c302d07f8fb364bb21d575c87806e61`; verifier passed 19-file roundtrip. |

# Biomedical RAG unblock report

The canonical query population is now frozen, every source document in that
population exists in the selected `external_db` snapshot, and one real B6 query
completed through local retrieval and generation. This clears the one-query
local acceptance gate for moving to the remote CUDA portability phase. It does
not explain the separate all-zero table whose report and query trace are not
present, and it does not establish July database identity.

Frozen inputs: [`questions.jsonl`](../ablation_portability/dataset/questions.jsonl),
[`questions_manifest.json`](../ablation_portability/dataset/questions_manifest.json),
and [`SHA256SUMS`](../ablation_portability/dataset/SHA256SUMS).

## Answers to the requested questions

1. **Why did reconnaissance find the source while portability did not?**
   Reconnaissance used the project-root `graphs/` directory. The portability
   workflow checked the absent nested path
   `back/benchmarks/rag/graphs/...`. The generated directories are git-ignored,
   but the actual canonical July file was present at the project root; see
   `canonical_dataset_recovery.md`.

2. **Were the exact 375 questions recovered?** Yes. The original July
   `retrieved_contexts.yaml` was recovered and validated as
   `EXACT_SOURCE_RECOVERED`, not reconstructed or resampled.

3. **If reconstructed, which fields came from which artifacts?** The dataset
   was not reconstructed. Per-field provenance records the original July source
   field and cross-checks: IDs, questions, row order, answerability, source
   paper/PMID, and archived reference answer come from the July YAML; positive
   references are crosschecked against the PubMedQA golden and sampled records;
   negative category and provenance are crosschecked against the synthetic
   negative trace. The 1,875-row historical metrics file independently
   crosschecks the 375-query run membership. Details and hashes are in
   `canonical_dataset_recovery.md` and `questions_provenance.csv`.

4. **Was the zero-Recall report one query or many?** Unknown. The exact report
   file, run ID, query IDs, and denominator were not recovered. The displayed
   aggregate table does not state its population. A separate saved retrieval
   artifact has three rows, but there is no evidence that it generated the
   all-zero table.

5. **Is the expected source document in the selected database?** Yes. Indexed
   metadata reports exact source paper-ID and PMID coverage for all 375 frozen
   records (300 answerable, 75 designated negatives). The smoke also retrieved
   its expected PMID 21084567 paper as the first final chunk.

6. **Did retrieval return candidates?** Yes for the verified B6 fixture: 20
   lexical seeds, 20 dense seeds, 10 reranker inputs, and five final evidence
   chunks. Recomputed single-document Recall and Context Precision are both
   1.0. This does not substitute for the missing all-zero report's own trace.

7. **Is there a routing or identifier bug?** None was observed in the verified
   fixture. The worker startup log, expected pre-run database hash, and saved
   configuration identify the same restored clone for the pipeline; the source
   snapshot stayed unchanged. The expected and retrieved paper IDs normalize to
   the same identifier. The normalizer does not equate a bare numeric PMID with
   a paper slug, so a different report could still have an ID-shape mismatch; its
   query-level IDs are unavailable.

8. **Why was B6 much slower?** The supplied 37.566-second report cannot be
   attributed to a stage without its trace. In the local smoke, B6-reported
   latency was 13.891 seconds (12.118 seconds in generation and 1.771 seconds
   for retrieval); Cross-Encoder scoring was 0.757 seconds and ran on MPS.
   Cold initialization took about 89 seconds for each of two MLX generator
   initializations and 32.5 seconds for the E5 embedder; full process wall time
   was 298 seconds. This separate run's MPS evidence does not explain the
   absent report or support attributing its latency to CPU Cross-Encoder work.

9. **Did a real B6 query complete?** Yes. `pubmedqa_21084567` completed with
   B6 `success`, process exit code 0, persisted retrieval, a captured final
   prompt, and a persisted local generated answer. All smoke gates passed.

10. **Which stages ran and which were skipped?** FTS5 lexical retrieval, E5
    dense retrieval, Cross-Encoder scoring, two-hop graph expansion, prompt
    assembly, MLX generation, and answer parsing ran. Query expansion attempted
    four structured outputs, failed JSON parsing, then fell back to the original
    question. Cloud evaluation was skipped. Graph expansion enumerated 322
    candidate papers and selected three intermediate neighbors, but no graph
    neighbor survived final context and the graph text block was absent from
    the captured generator prompt. Graph execution is therefore not reported as
    final graph evidence. The parser returned `ANSWERABLE`; this is reported
    separately from successful generation. CUDA was not exercised.

11. **What remains before the CUDA portability phase?** Nothing blocks starting
    that phase's remote one-query portability work. Transfer and checksum the
    frozen dataset and `external_db` snapshot, provision and record a pinned
    CUDA/PyTorch runtime, add and verify CUDA device selection (the current
    Cross-Encoder chooses MPS or CPU), run the same frozen query on the remote
    snapshot, and package/verify its traces. Do not start the full ablation
    matrix until that remote smoke passes. The all-zero report remains a
    separate unresolved diagnosis; obtain its report and per-query trace to
    settle its exact cause.

## Implementation and verification

The freeze workflow now points at the actual root `graphs/` source, validates
all 375 historical IDs and required fields, writes deterministic
`questions.jsonl`, manifest, provenance, and checksums, and rejects missing rows
or implicit resampling. The genuine B6 smoke exposed an existing control-flow
error in `back/benchmarks/rag/core/generation.py`: the generation loop was
unreachable inside the checkpoint loader. The existing loop was moved back into
`run_benchmarking`; retrieval and ranking algorithms were not changed. The
smoke identity check was also made robust to Rich wrapping a long database path,
with a regression test.

Focused tests passed: nine dataset, metric, routing, archive, and real-smoke
checks in the final group; five focused generation, wrapped-path identity,
archive, and model-lock checks in the preceding group. The smoke archive verifier
validated all 19 payload files and a byte-identical extraction roundtrip. No full
benchmark suite, mass generation, embedding regeneration, or full ablation
runner was run.

## Evidence files

- `canonical_dataset_recovery.md`, `questions_provenance.csv`
- `corpus_coverage_report.md`, `smoke_fixture.md`
- `zero_recall_diagnosis.md`, `smoke_execution_report.md`
- `smoke_run/` complete local execution traces
- `smoke_results.tar.zst` and `archive_roundtrip/`
- `remaining_blockers.md`

The candidate snapshot archive remains the previously verified
`ablation_portability/package/candidate_snapshots.tar.zst` with SHA-256
`76659734a0f23772f37bbca607f69b0a4b38ea801ee6d210ae81e8bbf95bb097`. Its
restoration and USEARCH rebuild checks were not repeated.
