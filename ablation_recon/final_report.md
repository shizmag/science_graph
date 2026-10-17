| Area | Verified state | Portability status | Blocking issue | Next required action |
|---|---|---|---|---|
| Generator | OCC-RAG-1.7B BF16 Qwen3 checkpoint/tokenizer identified | Public pinned revision | CUDA and telemetry untested | Implement same-checkpoint Torch CUDA |
| Embedding model | multilingual-e5-base, 768D prefixes | Public/local weights | Persisted vector lineage missing | Compare CUDA vectors and retrieval |
| Cross-Encoder | mMARCO MiniLM revision identified | Public/pinnable | Source selects MPS/CPU | Add CUDA and separate uses |
| Questions | 375 rows verified, 300/75 | Manifest can be extracted | No standalone frozen file | Extract/hash exact population |
| Corpus | 300 notes and local DBs exist | Staging feasible | July identity unresolved | Recover or approve new baseline |
| Vector index | Float32 768D SQLite blobs and USEARCH | Snapshot/rebuild plausible | model lineage/sidecar portability unknown | Restore/rebuild/validate |
| Graph | SQLite nodes and edges | Single DB | DB path split | One explicit path |
| Dependencies | Python >=3.12, broad app closure | Minimal OCI feasible | No CUDA benchmark lock | Build pinned image |
| CUDA | Torch 2.12/CUDA13 metadata for Blackwell | metadata only | no RTX run | test on RTX 5090 |
| Windows | WSL2 Docker GPU documented | conditional | no project test | test Win11 WSL2 |
| One-query smoke | archived pubmedqa_19302863 candidate | not run locally | incomplete runtime/mutation risk | isolated restored copy |

# Verdict: READY_WITH_EXPLICIT_BLOCKERS

The evidence is sufficient to plan implementation but not to ask a collaborator to run the experiment. July DB identity is unproven; current defaults can split retrieval and generation databases; vector provenance is incomplete; CE CUDA selection is absent; settings leak/couple; no end-to-end CUDA run occurred. Strict July identity requires recovering that snapshot. A new frozen DB must be labeled a new baseline.

# Q1. Which exact models are required?

Three learned assets are used by the observed full B6 path.

1. Generator: occ-ai/OCC-RAG-1.7B; local /Volumes/happy-disk/models/llm/OCC-RAG-1.7B; revision e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c. Qwen3ForCausalLM, BF16 safetensors 3,441,185,608 bytes; cache SHA256 metadata 913448d005d2af695a738da177120c9e741dc8a5b2328f717d7ed108c1fd2e4f. Same snapshot tokenizer/template. MIT/public: [OCC-RAG-1.7B model card](https://huggingface.co/occ-ai/OCC-RAG-1.7B).
2. Embedder: intfloat/multilingual-e5-base; local /Volumes/happy-disk/models/embeddings/intfloat/multilingual-e5-base; revision d128750597153bb5987e10b1c3493a34e5. SentenceTransformer mean pooling, 768D, query:/passage: prefixes, max512. Main file 1,112,201,288 bytes; metadata hash a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7. MIT/public: [multilingual-e5-base model card](https://huggingface.co/intfloat/multilingual-e5-base).
3. Cross-Encoder: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 revision 1427fd652930e4ba29e8149678df786c240d8825, XLM-R classifier, max512, weights 470,592,698 bytes; metadata hash 5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4. Apache/public: [mMARCO Cross-Encoder model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).

Conditional en_core_web_sm is unpinned; all-MiniLM-L6-v2 is inactive 384D default; dslim/bert-base-NER and Marker/PDF models are ingestion paths, not proven query-time requirements.

# Q2. Can every model be downloaded automatically?

The three core checkpoints are public and pinned, so automatic download is feasible in principle; no large downloads were performed. spaCy model is unpinned and may auto-download; pin/stage or disable. No private/custom generator weights were found. The manifest and DB/index lack verified remote source and need author staging.

# Q3. Can the historical OCC-RAG generator run on CUDA without changing model identity?

Strongest path is direct PyTorch + Transformers loading of exact BF16 HF weights, tokenizer and template; no conversion is needed. Keep greedy decode, context 12000, output 768, EOS/stop behavior. Transformers exposes output_logits: [Transformers text generation docs](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation). This is not execution proof. vLLM full logits telemetry is unverified; no verified GGUF for llama.cpp. CUDA parity is unresolved.

# Q4. Which model produced stored embeddings? Can index be reused?

Historical configuration identifies multilingual-e5-base; query/document prefix behavior is implemented in back/src/vector_search.py:14-123. SQLite vector blobs and cosine USEARCH are implemented in back/src/repository/sqlite_impl.py:1671-1697,1871-1900. Vectors have no revision/prefix/pooling/normalization fingerprint. Reuse only after accepted provenance and representative CUDA vector/ranking checks; dimension alone is insufficient.

# Q5. Where is actual corpus?

300 Markdown notes at back/benchmarks/rag/dataset/pubmedqa/notes/, about 3.1MB, derived from 1,075,513-byte local parquet. Repo DB back/science_graph.db is 14,065,664 bytes; external ~/.local/share/pdf-graph-analyzer/graph.db is 185,470,976 bytes. External PDF archive ~86,736 KiB is not mapped to benchmark chunks. July path unknown.

# Q6. Where are embeddings, lexical, graph and ontology assets?

SQLite schema/FTS5/graph storage and USEARCH sidecar are implemented in back/src/repository/sqlite_impl.py:49-80,1648-1701,1729 onward; vector blobs are written at :1871-1900. No Postgres/Qdrant/Neo4j. External DB counts: 22,068 chunks, 2,845 parents, 9,680 nodes, 11,712 edges. Repo DB: 2,331 chunks, 312 parents, 307 nodes, 615 tag edges. SQLite 3.50.4, user_version0, quick_check ok. Ontology data are SQLite graph records; spaCy is conditional query preprocessing.

# Q7. Can current DB transfer?

Yes, with SQLite consistent online backup, restore, FTS/schema/count/vector/graph validation and target-side USEARCH rebuild from vector blobs. Do not copy live DB alone during WAL. Compare retrieval fixture rankings with tolerance. Restore is not historical identity/equivalence.

# Q8. Is rebuild necessary?

Only if accepted snapshot cannot be used or vector provenance fails. Notes and E5 exist but source revision, chunking, graph extraction and July DB identity are not frozen. Rebuild creates new baseline unless validated equivalent. No rebuild occurred.

# Q9. Are 375 questions standalone?

Exact set is in archived retrieved_contexts.yaml: 300 answerable and 75 negative from 400 candidates. No thin manifest exists. load_benchmark_dataset in back/benchmarks/rag/core/config.py:418-499 can resample seed42; extract/hash exact rows and disable runtime sampling. Negative construction and source revision remain unknown.

# Q10. Which dependencies fail on Linux CUDA?

back/pyproject.toml:1-60 declares mlx-lm and other application packages; MLX is an Apple-only runtime, so provider MLX will not work as CUDA. The inspected platform markers mean Linux dependency resolution is not proven to fail solely because mlx-lm is declared. Native risks: USEARCH, Torch/Torchvision, llama-cpp-python, tokenizers, PDF/OCR. Existing image is CPU-only/unlocked. Use dedicated locked image and exclude unused ingestion packages.

# Q11. Can Windows and Linux share one environment?

Conditionally, the same Linux x86_64 CUDA OCI image can run on Linux and Docker Desktop WSL2 Linux containers. Vendor metadata documents Windows GPU route, not project test. Linux needs driver/Docker/NVIDIA toolkit. Windows needs current WSL2/kernel, NVIDIA WSL driver and Docker Desktop. Do not auto-install host drivers. Start with Windows 11 WSL2.

# Q12. Can full B6 run on one real local question?

No safe query executed. Host is macOS ARM64/no NVIDIA, runtime packages missing, and SQLite repository/USEARCH constructors may initialize schema, WAL or sidecar (back/src/repository/sqlite_impl.py:49-80,1648-1697). Archived trace is not an E2E probe. Candidate: pubmedqa_19302863 on isolated restored snapshot.

# Q13. What remains untested on RTX?

Generator CUDA/logits, E5 CUDA, CE CUDA, full pipeline, entropy parity, memory and driver/runtime, USEARCH restore, WSL GPU, and archive roundtrip. Metadata only.

# Q14. Which components are independently ablatable?

Lexical, dense, RRF, score blending with fixed CE inputs, and trimming when active are plausible. Source card is post-generation formatting and cannot affect generation entropy. Final reranker shares CE with expander (back/benchmarks/rag/core/generation.py:192-226; back/src/services/graph_expander.py:435,543,667); graph_expansion bundles static/dynamic context (back/src/services/rag_service.py:400-464,2184-2222); ontology flag is partial (:576-595,983-1056); nested config leaks (back/src/config.py:981-994); graph-neighbor-in-RRF is inactive under master graph (:1502-1554). Historical concept/bridge outputs are zero and no graph chunk reaches context. See dependency matrix.

# Q15. What must happen before collaborator run?

1. Recover July DB or approve a new baseline.
2. Freeze/hash exact 375 manifest.
3. Pass one DB path to all subprocesses.
4. Validate or rebuild vectors/index.
5. Normalize config and split coupled controls.
6. Implement CUDA generator and CE.
7. Define entropy and preserve scores.
8. Build pinned Linux image.
9. Restore/rebuild USEARCH and validate.
10. Pass one-query GPU smoke and two-arm integrity check.
11. Validate result archive roundtrip without Judge.
12. Resolve license/artifact-store/disk budget.

See blockers.md and implementation_roadmap.md.
