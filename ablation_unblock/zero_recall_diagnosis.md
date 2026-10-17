# Zero-recall report diagnosis

## Verdict: `UNRESOLVED` for the supplied all-zero table

The exact report containing the stated `0.008 / 0.007 / 0.141 / 0.464 / 37.566`
latencies and all-zero Recall/Context Precision was not found in the scoped
research and benchmark artifacts. A scoped search including git-ignored output
under `graphs/`, benchmark reports, `ablation_recon/`, `ablation_portability/`,
`ablation_unblock/`, and the revision directories found no `37.566` occurrence
in the pre-existing artifacts (this diagnosis records the user-supplied value).
The report therefore has no recoverable file, run ID, timestamp, dataset record
count, query IDs, effective database path, or metric denominators. The table alone
cannot establish whether it summarizes one query or many.

The closest saved report is
`back/benchmarks/rag/reports/retrieval_custom_report.md` (file timestamp
2026-07-29 21:19:34; SHA-256
`82eb70cf0833321494e83af8549b7808e34c1727609d07af4b6ce722489a406c`). It is a
different report: it shows nonzero Recall/Context Precision (B1 1.0000/1.0000,
B2 0.9973/0.9956, B4 0.8587/0.8475, B5 0.8587/0.8475, B6 0.9067/0.8650) and
latencies 0.002/0.003/0.048/0.055/0.701 seconds. It labels the custom component
settings unchanged from B6 defaults, so those settings do not represent a
component-off intervention.

The neighboring `custom_retrieved_contexts.yaml` contains three IDs—
`pubmedqa_21084567`, `pubmedqa_25432938`, and `pubmedqa_unans_17244100`—and each
has a `success` status plus nonempty outputs for B1, B2, B4, B5, and B6. Its
SHA-256 is `ad08733dabb65e462049ceeb323191eb6b6b896a053d8d133da41486c3d60d63`.
There is no saved run manifest proving that this three-row artifact supplied the
different all-zero table, so it is not used as that table's population.

## What the code and surviving run evidence establish

- The retrieval runner records `status: success` when its stage completes and
  persists `retrieved_chunks`, even if that list is empty. The report's
  execution-success rate is status-based. Thus 100% execution success alone does
  not mean candidates or relevant evidence were returned.
- `calculate_retrieval_recall()` uses the number of **distinct normalized
  expected paper IDs** as its denominator. Empty expected IDs (including IDs that
  normalize to empty) produce 1.0. A zero requires a nonempty normalized expected
  set with no normalized match, or an upstream/cached zero.
- `calculate_context_precision()` also returns 1.0 when no expected IDs exist;
  with nonempty expectations, empty retrieved chunks or no relevant chunk gives
  0.0. Its nonzero score is mean average precision over relevant chunks.
- The normalizer handles common path, separator, and chunk forms, but it does not
  translate a numeric PMID such as `21084567` into a slug such as
  `pubmed_21084567_...`. No evidence shows that this mismatch caused the missing
  table. In the completed smoke, the expected and retrieved `paper_id` values
  normalize to the same ID.
- The canonical source population is fully represented in the author-selected
  `external_db` snapshot: all 375 query source PMIDs and exact paper IDs map to
  chunked indexed metadata. That rules out corpus absence for this frozen
  population/snapshot, not for an unknown report database or query set.
- The genuine B6 smoke returned five chunks, with the expected source article at
  rank 1. Recomputing metrics from its persisted expected ID, paper IDs, and
  chunks yields Recall 1.0 and Context Precision 1.0. This proves working
  retrieval for the fixture; it does not establish why another run's table was
  zero.

No evidence identifies a general retrieval implementation or database-routing
failure. The missing report leaves a dataset/corpus mismatch, empty retrieval,
identifier mismatch, a legitimate retrieval miss, or upstream metric values
possible for that run. No one cause is selected without its query-level trace.

## B6 latency evidence

The reported 37.566-second value cannot be profiled without that report's run
trace. In the separate local smoke, B6 latency was 13.891 seconds, retrieval
latency 1.771 seconds, and local generator calls accounted for 12.118 seconds.
Cross-Encoder scoring took 0.757 seconds across 14 recorded scoring calls, with
1.428 seconds initialization on `mps:0`. The embedder initialized on MPS in
32.538 seconds, and MLX generator initialization events took about 88.96 and
89.17 seconds. The full smoke process wall time was 298.079 seconds, which
includes cold initialization and setup outside the B6-reported latency.

This run used MPS for the Cross-Encoder; it provides no evidence that CPU scoring
caused the separate 37.566-second report. The current Cross-Encoder device choice
is MPS or CPU, not CUDA. The supplied latency remains unassigned to a stage.
