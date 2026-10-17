# SQLite candidate comparison

> **Follow-up update — 2026-10-17:** The canonical July source was found at
> project-root `graphs/run_20260729_211324_OCC-RAG-1.7B/retrieved_contexts.yaml`.
> The earlier portability search expected it under the nested
> `back/benchmarks/rag/graphs/` path. See
> [`ablation_unblock/canonical_dataset_recovery.md`](../../ablation_unblock/canonical_dataset_recovery.md).
> This path correction freezes the query population; it does not prove which
> database July used. The candidate comparison below remains the earlier direct
> comparison of saved October retrieval evidence.

## Decision state

`external_db` is the author-selected new frozen baseline. This is a corpus choice for future within-run B6 component ablations; it does not establish historical July identity. The repository candidate remains preserved as a separate, nonselected snapshot.

The earlier portability report searched under
`back/benchmarks/rag/graphs/run_20260729_211324_OCC-RAG-1.7B/` and did not find the
canonical artifact there. Follow-up located it at project-root
`graphs/run_20260729_211324_OCC-RAG-1.7B/retrieved_contexts.yaml`; the exact
population is now frozen. The three-row October B2 artifact described below
remains a separate corpus comparison and does not establish July database
identity.

## Direct comparison available here

The available October artifact was sampled deterministically at its first, middle,
and last rows: `pubmedqa_21084567`, `pubmedqa_25432938`, and
`pubmedqa_unans_17244100`. Its B2 retrieval output contains 11 saved chunks.

| Evidence | Repo DB | External DB |
|---|---:|---:|
| Exact chunk IDs found | 11/11 | 11/11 |
| Archived chunk text SHA-256 matches | 11/11 | 11/11 |
| Saved paper IDs match | 11/11 | 11/11 |
| Corresponding paper nodes exist | 11/11 | 11/11 |
| Archived parent IDs available | 0/11 | 0/11 |
| Archived source titles available | 0/11 | 0/11 |

These are exact identity and text matches for the October B2 artifact. They do not
distinguish the databases and do not establish July provenance. The full row-level
comparison is in `available_smoke_candidate_comparison.json`.

## Candidate contents

| Candidate | Documents/chunks | Vectors | Graph | July identity | Classification |
|---|---:|---:|---|---|---|
| `repo_db` | 300 document IDs / 2,331 chunks / 312 parents | 2,331 × 768D float32 | 307 nodes, 615 `HAS_TAG` edges, 7 concepts | Unproven | `NEW_BASELINE_CANDIDATE` |
| `external_db` | 346 document IDs / 22,068 chunks / 2,845 parents; 300 `UserNote` nodes | 22,068 × 768D float32 | 9,680 nodes, 11,712 edges, including `CITES` and `MENTIONS_CONCEPT` | Unproven | `NEW_BASELINE_CANDIDATE` |

Both contain the benchmark's 300 PubMedQA notes. The external database has a much
larger corpus and graph. Its configuration path appears in the October snapshot,
but this is not proof that July retrieval and generation used the same file.

## Author decision record

The author selected `external_db` as the new baseline because it includes citation
and concept graph relations and a broader corpus. It contains 19,737 more chunks
than `repo_db`. This decision is recorded in `baseline_decision.json` and in the
selected snapshot manifest. It is not a historical equivalence claim. The
repository snapshot is retained for inspection but is not selected. All future
arms must be regenerated on the selected snapshot; July quality scores must not be
paired with outputs from this new baseline.

## Stored-vector compatibility

The local pinned `intfloat/multilingual-e5-base` checkpoint was run on a fixed
sample of 16 chunk IDs per candidate. Before inference, the comparison plan set
these criteria: every vector cosine agreement at least 0.999, relative L2 norm
difference at most 0.001, matching 768 dimensions, and mean top-10 dense ranking
overlap at least 0.90. Both candidates passed every criterion. The minimum cosine
was `0.9999999999987099` for `repo_db` and `0.9999999999980604` for `external_db`;
maximum relative norm differences were `1.22e-7` and `1.26e-7`, respectively.
All 16 sampled top-10 rankings matched exactly. This supports confident reuse of
the stored vectors with this pinned model and the project's `passage: ` document
prefix, mean-token pooling, 512-token truncation, and unnormalized encode output.
Per-vector evidence and the prewritten plan are stored in each candidate's
`e5_compatibility/` directory.

## Database and index roundtrips

Each candidate snapshot was restored twice in separate temporary directories.
Both restores passed SQLite integrity, foreign-key checks, vector dimension and
finite-value checks, FTS5 search, and graph count checks. USEARCH indexes were
rebuilt from the persisted 768D float32 blobs. Exact fixture retrieval results
were identical across the independent restores. Approximate top-10 overlap was
98% over the five fixed chunk-vector fixtures; USEARCH index files had different
SHA-256 values after independent builds, and their order was not identical for
all fixtures. The SQLite vectors and logical identifiers are stable, while the
binary HNSW sidecar is rebuilt per target machine.

| Candidate | SQLite SHA-256 | Independent restore | E5 vector status |
|---|---|---|---|
| `repo_db` | `23adc428e8f21743144ea5f0a316931bb0158ef02a8d8fc5e1d1ad408f3ac005` | Passed twice; exact fixture rankings identical | `CONFIDENT` |
| `external_db` | `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0` | Passed twice; exact fixture rankings identical | `CONFIDENT` |
