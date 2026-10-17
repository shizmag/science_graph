> **Current status update — 2026-10-17:** This report records the 2026-10-16
> portability checkpoint and its blocked smoke. Follow-up work found the July
> source at the project-root `graphs/run_20260729_211324_OCC-RAG-1.7B/` path,
> froze the exact 375-query dataset, verified all 375 source articles in the
> author-selected `external_db` corpus, and completed a genuine local B6
> retrieval-plus-generation run. The full current gate table and evidence are in
> [`ablation_unblock/final_report.md`](../ablation_unblock/final_report.md).
> `external_db` remains a new baseline; July database identity remains unproven.

The status table below is retained as the historical 2026-10-16 checkpoint.

| Gate | Status | Evidence | Blocker |
|---|---|---|---|
| Frozen 375 questions | BLOCKED | Extractor, canonical JSONL policy, manifest validator, and frozen loader implemented | Canonical July `retrieved_contexts.yaml` has not been restored; no questions were synthesized |
| Database comparison | PARTIAL | Both candidates compared to the only available October B2 artifact: 11/11 saved chunks match each database by ID, text hash, and paper ID | Canonical July B6 retrieval artifact is missing, so its top-five evidence cannot be compared |
| Snapshot generated | PASS | SQLite backup API snapshots for both candidates, with source main/WAL/SHM identities recorded | None |
| Snapshot restore verified | PASS | Two independent restores per candidate plus third restore after archive extraction; integrity, FTS, vectors, graph, and identifiers validated | USEARCH binary index is rebuilt per target machine |
| E5 vector compatibility | PASS, sampled | 16 deterministic chunks per candidate; vector cosine, relative norm, dimension, and top-10 ranking criteria passed | This is a sample-level comparison, not a full-corpus re-embedding audit |
| DB path isolation | PASS | Explicit path and expected SHA-256 propagate through pipeline subprocesses; conflicting HOME default and hash mismatch tests pass | The actual B6 pipeline has not run yet |
| Local B6 E2E | BLOCKED | `pubmedqa_19302863` preflight attempted against author-selected snapshot | Missing `ablation_portability/dataset/questions_manifest.json`; pipeline and model inference were not invoked |
| Portable archive | PARTIAL | `candidate_snapshots.tar.zst` verifies; 42 files, two validated snapshots, selected-baseline record, code, and model lock | Input archive needs the frozen dataset; smoke archive needs a real query execution |

# Phase 1 report

## Decision and scope

The author-selected new baseline is `external_db`, recorded in
`snapshots/baseline_decision.json`. Its SQLite snapshot SHA-256 is
`c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`. This selection
is for future within-run component ablations. It does not establish that this was
the database used in July. The repository database snapshot is preserved as a
separate candidate.

The phase is incomplete because the canonical July population artifact is absent.
The extractor deliberately fails closed rather than reconstructing the population
from a sample or a seed. The requested one-query run therefore did not reach the
pipeline, retrieval, model loading, or generation stages.

## 1. Database selected and why

`external_db` was selected because it contains the 300 PubMedQA note documents and
a broader 22,068-chunk corpus with citation (`CITES`) and concept
(`MENTIONS_CONCEPT`) graph relations. `repo_db` contains 2,331 chunks and a smaller
tag-only graph. The external database offers graph paths relevant to a future
graph-enabled B6 ablation. This changes the corpus relative to `repo_db`, so all
future arms must use the exact selected snapshot.

The author rationale is in `snapshots/database_author_decision.md`; the machine
record and selected database identity are in `snapshots/baseline_decision.json`.

## 2. Match to historical retrieval evidence

The documented canonical July artifact,
`back/benchmarks/rag/graphs/run_20260729_211324_OCC-RAG-1.7B/retrieved_contexts.yaml`,
is absent. The only saved retrieval artifact available is the three-row October
B2 `smoke_test_output/retrieved_contexts.yaml`. A deterministic first/middle/last
sample from that file contains 11 chunks. Both candidate databases match all 11 by
exact chunk ID, chunk text SHA-256, saved paper ID, PMID, and paper-node presence.
The artifact does not include parent IDs or source titles. It cannot distinguish
the candidates. See `snapshots/available_smoke_candidate_comparison.json` and
`snapshots/candidate_comparison.md`.

## 3. July database identity

July database identity is **not proven**. The available October evidence is not the
July B6 run and matches both databases equally. Neither candidate is labeled as
the historical July database.

## 4. Stored embedding reuse

Stored embeddings are reusable with **sample-level confidence** for the pinned
`intfloat/multilingual-e5-base` model, revision
`d128750597153bb5987e10b1c3493a34e5a4502a` (the requested prefix resolves to this
commit). Before inference, the comparison criteria were set to per-vector cosine
agreement ≥ 0.999, relative L2 norm difference ≤ 0.001, 768 matching dimensions,
and mean top-10 ranking overlap ≥ 0.90.

Sixteen deterministic chunk IDs per database were embedded using the local pinned
checkpoint on Apple Silicon. For `repo_db`, minimum cosine was
`0.9999999999987099`, maximum relative norm delta was `1.22e-7`, and all sampled
top-10 rankings matched. For `external_db`, minimum cosine was
`0.9999999999980604`, maximum relative norm delta was `1.26e-7`, and all sampled
top-10 rankings matched. The preprocessing inspection found chunk text input,
`passage: ` document and `query: ` query prefixes, mean-token pooling, 512-token
truncation, no explicit output normalization, and cosine-distance retrieval.
Detailed sample results and the predeclared thresholds are stored with the
candidate snapshots under `e5_compatibility/`.

This supports reuse of the saved vectors without an immediate full-corpus
re-embedding. It does not prove every stored vector's lineage.

## 5. SQLite and USEARCH transfer

Both source databases were read using SQLite read-only connections and captured
with `sqlite3.Connection.backup`, which includes a consistent WAL-aware database
state. Each snapshot manifest records source main, WAL, and SHM identities before
and after backup. The original source identities were stable across capture.
Snapshot databases use DELETE journal mode and have no required SQLite sidecar.

The stored float32 vectors remain in SQLite. The USEARCH binary sidecar is not
treated as portable and is not copied as authoritative state. The target-side
rebuild command reads saved vector blobs and stable chunk IDs:

```bash
python -m ablation_portability.scripts.rebuild_vector_index <restored-database.db>
```

It does not recompute document embeddings. Independent approximate index builds
can produce different binary hashes and minor ordering differences; validation
compares persisted vector identities and retrieval fixtures.

## 6. Authoritative database path

The new execution path takes `--db-path` and `--db-sha256`, resolves the explicit
path once, verifies its digest, records it in the run manifest, and passes the same
identity to retrieval and generation subprocesses. It does not need the external
user YAML or a working-directory fallback. Conflicting machine defaults and a
deliberately wrong digest are covered by tests. Production changes are limited to
`back/benchmarks/rag/core/config.py`, `run_pipeline.py`, `run_custom_retrieve.py`,
and `run_benchmarks.py`.

Because the one-query pipeline did not start, same-path delivery is verified by
the implementation and focused tests, not by a completed run trace.

## 7. Real B6 query execution

No real B6 query executed. The smoke preflight used the selected external snapshot
and target ID `pubmedqa_19302863`, then stopped while loading the frozen dataset:

```text
FileNotFoundError: .../ablation_portability/dataset/questions_manifest.json
```

The exact preflight record and log are under
`smoke/preflight_after_author_choice_20261016/`. Pipeline exit code is null because
the pipeline was never invoked. No retrieval candidates, Cross-Encoder scores,
prompt, generated answer, parser result, or generation telemetry were produced.
This is a blocker, not a successful or partial inference smoke.

## 8. Graph stages

No graph retrieval stage ran in the new B6 smoke. The available October B2 archive
was only used for identity comparison. It is not evidence that the chosen query
would exercise any particular graph path in the selected baseline.

## 9. Generated files

Implementation and evidence are under `ablation_portability/`:

- `scripts/`: deterministic extractor, snapshot, restore, validation, roundtrip,
  candidate comparison, E5 comparison, USEARCH rebuild, smoke, packaging, and
  package verification commands.
- `dataset/loader.py`: strict frozen-manifest loader. The canonical
  `questions.jsonl` and `questions_manifest.json` were not generated because the
  source is absent.
- `snapshots/candidates/{repo_db,external_db}/v1/`: candidate SQLite snapshots,
  manifests, logical inspection, validation fixtures, and roundtrip evidence.
- `snapshots/candidate_comparison.md`,
  `snapshots/available_smoke_candidate_comparison.json`,
  `snapshots/database_author_decision.md`, and `snapshots/baseline_decision.json`.
- `models/model_assets.lock.yaml`: exact public model IDs and revisions, expected
  local file hashes, tokenizer identities, runtimes, and download policy. Model
  checkpoints are not duplicated in the archive.
- `smoke/preflight_after_author_choice_20261016/`: honest blocked preflight log and
  manifest.
- `package/candidate_snapshots.tar.zst` and `package/SHA256SUMS`.
- `tests/test_portability.py` and the isolated `pyproject.toml`/`uv.lock`.

The isolated project environment is `ablation_portability/.venv`; no global
packages were installed.

## 10. Independent package restore

`package/candidate_snapshots.tar.zst` SHA-256 is
`76659734a0f23772f37bbca607f69b0a4b38ea801ee6d210ae81e8bbf95bb097`. The verifier
found 42 payload files, checked their checksums and schemas, extracted both SQLite
snapshots into a clean temporary location, validated their logical contents, then
performed a third restore and rebuilt each USEARCH index from stored vectors.
Both candidate snapshots passed.

This is a verified **candidate snapshot set**, not the requested frozen input
archive. The input archive cannot be built until the exact question dataset exists;
the smoke-results archive is also not built because a real smoke did not execute.

## 11. CUDA runtime remaining

CUDA has not been tested or implemented in this phase. The next infrastructure
stage must prepare a pinned CUDA/PyTorch runtime, fetch and hash-check the locked
public model assets on the collaborator host, extract and validate this selected
snapshot, rebuild USEARCH from the stored vectors, then run the same one-query B6
smoke with explicit database identity. It must capture actual device/runtime,
retrieval, generation, and available token telemetry before starting an ablation
matrix.

## 12. Windows and WSL2 remaining

Neither Windows nor WSL2 was tested. Validate archive extraction, Python/runtime
installation, SQLite restore and FTS behavior, USEARCH build/load behavior, model
path resolution, subprocess path propagation, CUDA visibility, and a real one-query
smoke under WSL2. The macOS arm64 results do not establish Windows/WSL2 portability.

## 13. Work required before a collaborator runs the full ablation

First restore the exact canonical July `retrieved_contexts.yaml` and rerun the
deterministic freeze and manifest verification. Confirm the 375 IDs, 300 positive
labels, 75 designated-negative labels and categories, then build and verify the
selected external input archive. Run the real local B6 case from an isolated
restore and package its complete diagnostics. Next validate the selected archive
and one-query run on the collaborator's CUDA/WSL2 runtime. Only after those gates
pass should the collaborator regenerate the complete B6 and all component-off
arms from the same selected snapshot. Do not pair old July scores with outputs
from this new baseline.

## Final verdict

`PORTABLE_SNAPSHOT_READY_E2E_BLOCKED`

The SQLite candidate snapshots are portable and were restored and validated from
the packaged archive. The external candidate is author-selected, stored-vector
compatibility passed the declared sample criteria, and the explicit path isolation
code passed focused tests. The frozen 375-query population is still unavailable,
so the input dataset archive and genuine one-query B6 execution remain blocked.
