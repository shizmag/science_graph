# Full CUDA run design

## Frozen experiment identity

The run resolves all arms before generation and writes `arms_manifest.json` and `.yaml`. Internal implementation B6 maps to paper-facing baseline B5. Every arm uses the same ordered 375-query dataset, source labels and gold answers, restored `external_db` snapshot, E5 and Cross-Encoder checkpoints, OCC-RAG-1.7B BF16 weights, tokenizer, decoding settings, prompt serialization policy, 12,000-token context budget, and 768-token output cap. The run records image/source identity, model file checksums, GPU/driver/CUDA/PyTorch versions, seed and deterministic settings.

The approved matrix has 14 arms and 5,250 query-arm operations:

1. `B6_full`
2. `B6_minus_score_blending`
3. `B6_minus_final_reranking`
4. `B6_minus_dynamic_graph_expander`
5. `B6_minus_static_graph_context`
6. `B6_minus_intrapaper_graph_candidates`
7. `B6_minus_concept_retrieval`
8. `B6_minus_bridge_retrieval`
9. `B6_minus_parent_deduplication`
10. `B6_minus_lexical_retrieval`
11. `B6_minus_dense_retrieval`
12. `B6_fixed_budget_context`
13. `B6_interaction_graph_candidate_x_dynamic_expander`
14. `B6_interaction_reranker_x_blending`

The manifest documents inactive or coupled branches. Ontology query enrichment is not a primary arm because every canonical query is longer than the branch's two-word activation limit. The matrix is not changed in response to generated answer quality.

## Hardware acceptance gate

The `run` and `resume` commands execute diagnostics and installation, then run the Phase 2 two-arm smoke on the actual container GPU. The smoke must record CUDA operations for query embeddings, Cross-Encoder scoring, generation and token metrics; exercise the restored SQLite/USEARCH retrieval; serialize results; and prove a meaningful disabled execution path in traces. The full-run gate also checks that the current device is an RTX 5090 with compute capability 12.0 and matches the device that passed smoke. `torch.cuda.is_available()` alone cannot pass the gate. Smoke failures are archived and stop before the 375-query workload.

## Query-arm checkpointing and resume

`checkpoints/index.sqlite3` is a transactional index to atomically replaced result JSON records. Token-level records are kept in a compressed JSONL sidecar; each result stores its path and SHA-256. Input identities include exact query bytes, question row, dataset and manifest fingerprints, corpus snapshot, complete effective configuration, model manifest, decoding policy, runtime image/source identity, and telemetry implementation. A stable result ID also binds the serialized prompt hash. Corrupt, stale, or incompatible results are recomputed; only integrity-checked successful records are reused.

Transient failures receive at most three attempts with bounded backoff. OOM retries clear CUDA allocator state without changing model weights, precision, prompt, limits, or decoding. Failure records and retry history remain available. Ctrl+C and ordinary exceptions finalize a partial report and archive. `run` reuses a compatible incomplete run; `--new-run` creates another identity, and completed output directories are never overwritten.

## Caching and execution

The SQLite stage cache is dependency-keyed for query embeddings, lexical/dense candidates, graph neighbors, query/chunk Cross-Encoder scores, and exact generation inputs. Cache dependencies include source/config/model/corpus identity. Generation reuse requires the exact prompt, decoding, tokenizer/model identity, and telemetry implementation. Cache hits remain explicitly traced; a cache hit does not count as proof that an intervention branch ran.

Retrieval and graph orchestration use CPU/SQLite/USEARCH as appropriate. Neural query embeddings, Cross-Encoder calls, graph-expander scoring, generation, and token probability calculations use CUDA. Checkpoint/result iteration is bounded; local analysis streams token rows into Parquet.

## Progress and artifacts

`logs/progress.json`, `logs/events.jsonl`, and `logs/errors.jsonl` are updated during execution. ETA appears only after three measured query-arm timings. Model startup is recorded separately from per-operation throughput. Progress includes planned/completed/failed work, retries, stage-cache and generation-cache use, generated tokens, rate, GPU memory when available, and output location.

The runner creates the `graphs/run_<UTC>_OCC-RAG-1.7B_CUDA_ABLATION/` tree, computes judge-free retrieval/context/parser/runtime/token statistics, exports every query-arm outcome for later blinded judging, writes a completion state, and packages the experiment plus internal checksums. The full result archive excludes downloaded model caches and the database copy.
