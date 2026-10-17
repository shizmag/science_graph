# Executive summary

**Verdict: READY_WITH_EXPLICIT_BLOCKERS.** There is enough evidence to implement a bounded snapshot-and-run workflow, but not to ask the collaborator to run a scientific ablation yet.

The exact historical 375-query population is recoverable from the archived retrieved_contexts.yaml: 300 answerable and 75 negative rows from a 400-row pool. A thin standalone manifest does not exist. Generator, E5 embedder and mMARCO Cross-Encoder are public identifiable Hugging Face checkpoints with local cache evidence. The stored 768D vectors do not encode their own E5 revision and preprocessing fingerprint.

OCC-RAG-1.7B is an unquantized BF16 Qwen3-family safetensors checkpoint. Loading this exact checkpoint/tokenizer with PyTorch and Transformers is the strongest identity-preserving CUDA approach and gives the best path to logits. Upstream Blackwell metadata supports feasibility; no RTX execution occurred. Current Cross-Encoder selection is MPS or CPU only.

The data store is SQLite with chunk text, float32 vector blobs, FTS5, graph nodes/edges and a USEARCH cosine sidecar. A SQLite online backup plus target-side sidecar rebuild is a practical transfer design, but July index identity is missing and cross-platform sidecar compatibility is unverified. Current process routing can send retrieval to repository-local DB and generation to an external configured DB.

Historical B6 traces show 1,053 graph-neighbor candidate records overall, zero concept/bridge candidates and no graph chunks surviving into final contexts. The selected query has seven pre-rerank graph candidates and zero selected. Candidate presence is not proof graph evidence affected the generated answer. Configuration OR leakage, shared reranker use and a bundled static/dynamic graph flag prevent several naive ablations.

| Area | Verified state | Portability | Blocker | Next action |
|---|---|---|---|---|
| Generator | OCC-RAG-1.7B BF16 checkpoint/tokenizer identified | Public pinned revision | CUDA path and telemetry untested | Add exact-checkpoint Torch CUDA and validate stops/logits |
| Embedder | multilingual-e5-base, 768D, prefixes known | Public | DB vector provenance absent | Compare CUDA vectors/rankings |
| Cross-Encoder | mMARCO MiniLM revision known | Public | Current device selector MPS/CPU | Add CUDA and prove both uses |
| Questions | 375 rows verified, 300/75 | Can extract thin manifest | No standalone hash/source revision | Freeze and hash exact population |
| Corpus | 300 notes and SQLite candidates exist | Staging feasible | July database/index unresolved | Recover or approve a new baseline |
| Vector index | 768D float32 in SQLite plus USEARCH | Snapshot feasible | E5 lineage and sidecar portability | Restore, rebuild, validate fixture rankings |
| Graph | Nodes/edges in SQLite | Single DB | DB path split | Pass same explicit DB to all stages |
| Python | >=3.12; broad lock includes MLX/native deps | Minimal OCI feasible | No CUDA benchmark closure | Pin dedicated image/runtime |
| CUDA | PyTorch 2.12/CUDA13 metadata for Blackwell | Metadata only | No target GPU test | Validate RTX 5090 |
| Windows | WSL2/Docker GPU route vendor documented | Conditional | No host test/prereqs | Target Win11 WSL2 and test |
| One-query smoke | Archived candidate pubmedqa_19302863 | Not locally executed | Missing packages and mutation risk | Run on isolated restored DB |

Strict July index reproduction requires recovery of the exact DB snapshot. If the scientific team accepts a current frozen snapshot, it must be labeled as a new baseline.
