# Biomedical RAG CUDA phase 2 — final report

**Report date:** 2026-10-17  
**Phase status:** portable implementation and CPU-side gates are complete; the RTX 5090 two-arm smoke and returned-archive gate are pending. The 375-query generation matrix was not started.

The author-selected `external_db` snapshot is frozen as a new experimental baseline. It does not establish historical July database identity. The original Phase 1 and local MLX smoke artifacts remain in place.

## Acceptance questions

### 1. Can the Linux CUDA image build reproducibly?

**Status: `IMPLEMENTED_AND_TESTED`.** The `linux/amd64` image completed under Docker Desktop's amd64 emulation on this Apple Silicon Mac. The pinned Python base and locked runtime installed 90 packages with `uv sync --locked`; the image compiled 12,907 Python files. Its local image ID was `sha256:f30cc99e4cc9f8b06932fcb65badb4a9ba38b1b592ac778c5efddb46b872d0e5` (5,779,239,453 bytes).

The build candidate used the same Python base digest and dependency lock as the final v1.0.0 release. I then mounted the final packaged source into that image and imported the query runtime, validated the frozen input package, and confirmed SQLite FTS5. The final `Dockerfile` also performs these import and FTS5 checks. A second byte-for-byte image rebuild was not run, so reproducibility evidence is the pinned base, package hashes, and successful locked build rather than a repeated-build comparison. The launchers save Buildx's OCI digest for each collaborator build and each run records it.

Pinned runtime: Python 3.12.13; PyTorch 2.12.0+cu130 / CUDA runtime 13.0; Transformers 5.20.1; tokenizers 0.23.3; SentenceTransformers 5.7.0; USEARCH 2.26.4. The runtime lock excludes MLX, PDF/OCR ingestion, and plotting dependencies. PyTorch's official release material describes CUDA 13 and Blackwell support: [PyTorch 2.12 release](https://pytorch.org/blog/pytorch-2-12-release-blog/) and [previous-version wheel commands](https://pytorch.org/get-started/previous-versions/).

### 2. Can all models download automatically?

**Status: `IMPLEMENTED_NOT_HARDWARE_TESTED`.** `smoke` runs installation and asset acquisition first. The downloader uses immutable Hugging Face revisions, persistent cache directories, resumable `snapshot_download`, revision metadata checks, and file SHA-256 checks from the frozen model asset manifest. Cache hits are validated; damaged files trigger a refresh of that revision. A mock-cache/revision/hash test passed. No full checkpoint download was run locally.

Pinned assets are OCC-RAG `e7d9f8b912dc1ac9a66ab42f0d07a7663d6ed58c`, E5 `d128750597153bb5987e10b1c3493a34e5a4502a`, and mMARCO Cross-Encoder `1427fd652930e4ba29e8149678df786c240d8825`. The E5 revision is the full commit resolved from the prefix recorded in the recovered model evidence. The query-time spaCy model is also acquired automatically at `en_core_web_md` 3.8.0.

### 3. Can the selected database restore without re-embedding?

**Status: `IMPLEMENTED_AND_TESTED`.** I restored the selected SQLite snapshot into an isolated temporary location, validated the file SHA and logical fingerprint, checked all 22,068 stored 768-dimensional vectors, and rebuilt the USEARCH sidecar from those persisted vectors. The report recorded `recomputed_embeddings: false`. The five chunk fixtures, FTS fixture, and three graph-node fixtures replayed against the selected snapshot. Invalid vectors fail closed; the installer has no re-embedding path.

### 4. Is the 375-question manifest correctly loaded?

**Status: `IMPLEMENTED_AND_TESTED`.** The frozen loader and package validator read 375 records: 300 answerable and 75 designated negatives. The exact `questions.jsonl` SHA-256 is `d2820ced7c2994218245887ad753700923262fb91ec983eeacc01b217ec43b6b`; the summary manifest SHA-256 is `e99ab9055dbe883da72f5466afa09629b13f1b9139f516a45f8aa38e21f4cee9`. The smoke query is loaded by ID `pubmedqa_21084567`, canonical row 138, with gold PMID 21084567.

### 5. Does the generator use the original weights?

**Status: `IMPLEMENTED_NOT_HARDWARE_TESTED`.** The CUDA backend loads the pinned OCC-RAG `model.safetensors` and tokenizer locally, requires BF16 parameters, rejects quantized/MTP directories, and does not convert or substitute weights. It uses greedy argmax, the 768-token budget, the 12,000-token context bound, and checkpoint EOS IDs `151643`, `151645`, and `151683`. The model files are checksum-frozen. Actual download and CUDA loading of those weights remain untested.

The runtime records the exact formatted prompt and tokenizer revision. The backend changes from historical MLX to Transformers/PyTorch SDPA; output-sequence identity is measured against the packaged Phase 1 reference when a GPU run occurs, never assumed.

### 6. Is CUDA logit telemetry implemented and numerically tested?

**Status: `IMPLEMENTED_NOT_HARDWARE_TESTED`.** The generator computes raw-logit reductions per step on the logits' current CUDA device and retains compact values only. Each emitted token records locally renormalized H20/H50/H100, top-K full-vocabulary retained mass, selected-token log probability, NLL, top-logit margins, token ID, token text, and stop state. Full-vocabulary logits are not accumulated across steps. A CPU reference-distribution test checks entropy, retained mass, selected-token probability, NLL, and margins numerically. The CUDA implementation itself has not run on NVIDIA hardware.

### 7. Does the Cross-Encoder use CUDA for all required roles?

**Status: `IMPLEMENTED_NOT_HARDWARE_TESTED`.** CUDA is selected explicitly and CPU fallback is fatal. The final ranking and `ExperimentalGraphExpander` share the same instrumented Cross-Encoder. Per-call traces label final candidate reranking versus graph-expander scoring, capture model and input devices, and fail a smoke if a scoring role reported by the graph trace has no corresponding CUDA inference. These checks have CPU unit coverage but no actual CUDA inference evidence yet.

### 8. What exact host prerequisites remain?

**Status: `IMPLEMENTED_NOT_HARDWARE_TESTED`.** The host launchers check prerequisites and provide remediation without changing drivers, administrative settings, or restarting the host.

- **Linux:** Docker Engine with Buildx in Linux mode, working NVIDIA Linux driver at least 580.65.06 for the pinned CUDA 13 / Blackwell target, `nvidia-smi`, NVIDIA Container Toolkit (`nvidia-ctk`), and an RTX 5090 visible inside a `--gpus all` container.
- **Windows:** WSL2, Docker Desktop in Linux-container mode with WSL2 integration, Windows NVIDIA WSL CUDA driver at least 580.88, `nvidia-smi.exe`, and successful `wsl.exe -e nvidia-smi` GPU passthrough.

This development host is macOS and has no NVIDIA device. PowerShell/Windows execution was unavailable; those script paths were checked statically only.

### 9. What was actually executed locally?

**Status: `IMPLEMENTED_AND_TESTED`.** Local execution included:

- Python compile checks across `ablation_cuda`, portability code, and the touched query runtime.
- **18 focused tests passed**, covering locked dependency and asset parsing, mock cache/revision validation, SQLite restore and USEARCH reconstruction, real fixture replay, bootstrap checks where statically testable, identity/isolation, telemetry math, result import, and corrupt/incomplete archive rejection. Pytest emitted one unrelated warning because the existing environment has an unknown `asyncio_mode` option.
- `bash -n` for both executable shell entrypoints and `git diff --check`.
- A complete `linux/amd64` dependency image build under emulation, followed by imports of `EmbeddingEngine`, `RAGService`, and `run_query_on_baseline` plus final-package validation inside that image. It reported the expected versions above and `torch.cuda.is_available() == false`.
- Final input release construction and manifest/checksum audit; it contains one database (`snapshot/database.db`) and no `repo_db` snapshot. The returned shell launcher retains executable mode in the tar archive. A scan found no `sk-or-v1-` credential prefix.
- Extracted the exact final archive in a clean temporary directory, validated its full inventory, restored the SQLite snapshot byte-exactly, and rebuilt the USEARCH index from 22,068 persisted vectors. The run reported `recomputed_embeddings: false`.

No full model download, CUDA inference, CUDA B6 query, component-off query, or cloud judge ran locally.

### 10. What remains untested until an RTX 5090 is available?

**Status: `BLOCKED`.** A supported RTX 5090 host must pass host and in-container driver/runtime preflight, then actually execute CUDA E5 query embedding, final and graph-expander Cross-Encoder scoring, OCC-RAG BF16 generation, and per-token CUDA probability reductions. It must complete the full B6 arm and a trace-eligible component-off arm, verify frozen database identity throughout, and export a result archive that passes the author verifier. No result may be described as a CUDA smoke until that happens.

### 11. What command should the collaborator execute?

**Status: `IMPLEMENTED_NOT_HARDWARE_TESTED`.** After receiving and extracting the single input archive, run one command from its root:

```bash
./ablation.sh smoke
```

Windows PowerShell:

```powershell
.\ablation.ps1 smoke
```

`smoke` performs preflight, builds the image if needed, installs and validates assets, restores the snapshot, then executes only the frozen query under B6 and one trace-selected off arm. It first selects `no_score_blending` only if the B6 trace shows the blend ran on at least two candidates. Its predeclared fallback is `no_graph_expansion` only if B6 inserted dynamic graph-expansion output. If neither trace rule passes, it records a blocked run instead of searching for a favorable intervention. No full matrix or judge is launched.

### 12. What files should the collaborator return?

**Status: `IMPLEMENTED_NOT_HARDWARE_TESTED`.** Return both generated files:

```text
return/ablation_cuda_results_v1_<runid>.tar.gz
return/ablation_cuda_results_v1_<runid>.tar.gz.sha256
```

The archive includes the run manifest, both arm records, raw responses, parsed answers, exact prompts, retrieved candidates/evidence, trace counters, both effective configurations, model/index manifests, per-call token telemetry, timing/device evidence, logs, and judge-ready inputs. It contains no judge output.

### 13. Can the author verify and import those files?

**Status: `IMPLEMENTED_AND_TESTED`.** On macOS, without NVIDIA hardware, run:

```bash
./ablation verify-results /path/to/ablation_cuda_results_v1_<runid>.tar.gz
```

The standard-library-only verifier checks the external archive SHA, internal inventory and checksums, frozen question and database identity, pinned model revisions/file hashes, both effective configurations, raw output and prompt fields, per-token telemetry, stage completion, intervention traces, and actual RTX 5090/CUDA evidence. It rejects incomplete/error runs, path-traversal entries, corrupt files, mismatched identities, missing arms, and absent device evidence. It imports to a new run-specific directory and writes a preview with both answers. Unit tests exercised valid import roundtrip, both intervention forms, corrupted archive rejection, and incomplete-arm rejection; no collaborator archive exists yet.

### 14. Is the full ablation matrix safe to implement next?

**Status: `BLOCKED`.** The full 375-query generation matrix is **`NOT_IMPLEMENTED` by design** in this phase and was not started. First complete the real RTX 5090 two-arm smoke, return its archive, and pass author-side verification. The previous local MLX smoke is diagnostic only: it showed ten reranked candidates and an active score-blending path, while graph-expansion candidates did not survive into final context. The CUDA run must make its own trace-based intervention selection.

## Frozen release and transfer action

The input package is `releases/ablation_cuda_input_v1.0.0.tar.gz`, SHA-256:

```text
9c552757487cfa37920f761384000c152679dd6a34cb6aa4726d616de86c9f37
```

It contains the exact 375-question manifest and only the selected `external_db` snapshot. Its snapshot SHA-256 is `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0`; fingerprint: 346 papers, 22,068 chunks/vectors at 768 dimensions, 9,680 graph nodes, and 11,712 edges. The source package manifest SHA-256 is `785303a356f1ce03c591bce8aa44da59ef60967bd6aeb977d11444242aeb3bb1`.

No artifact store is configured. The remaining staging action is to copy this exact archive and its matching `.sha256` file into the author's private collaborator transfer location. No public upload has occurred. The archived source records base commit `f53b1999bf006d826e6284b006bb7081b4581f36`, a dirty working tree, and independent checksums for packaged source files.

## Historical configuration and database notes

`external_db` is the author-selected new baseline. **Historical July database identity is not proven.** The package freezes the effective B6 configuration, disables cloud credentials, isolates execution from machine-specific YAML, and records documented differences from the historical MLX configuration. The runtime writes per-arm traces for graph branch execution, candidate generation/selection, context insertion, and prompt assembly. The configuration audit found `graph_neighbors_in_rrf` enabled by the B6 component builder but inactive when the graph-retrieval master branch is active; flags alone are not treated as evidence of effect.

The frozen B6 bundle enables lexical and dense search, RRF, final Cross-Encoder reranking, score blending, context trimming, parent deduplication, graph retrieval and dynamic graph expansion, graph tracing, the selected-source card, and the Shannon estimator. Nested graph retrieval enables concept and bridge branches even though their top-level component switches are false under the documented OR semantics. HyDE, LLM query expansion, dynamic-alpha blending, citation repair, and intent classification are disabled. The selected-source card is post-generation output; its enabled flag is not treated as evidence that it affected answer context. The provider changes to `cuda_torch`; model path, database path, cloud credentials, and spaCy package resolution are made portable. The machine-readable configuration and historical differences are in the input package.

The prior local MLX run for `pubmedqa_21084567` passed retrieval and generation on Apple Silicon. Its prompt, answer, and available telemetry are included as a diagnostic reference. The CUDA run will compare prompt bytes, token sequences, and aligned telemetry where possible; different output sequences will not be reported as byte-identical historical reproduction.

## Files

- Author release and result verifier: [`release.py`](release.py), [`author_cli.py`](author_cli.py), [`verify.py`](verify.py)
- Runtime and CUDA smoke: [`runtime/`](runtime/), [`back/src/llm_engine/cuda_torch_impl.py`](../back/src/llm_engine/cuda_torch_impl.py)
- Host launchers and image: [`ablation.sh`](ablation.sh), [`ablation.ps1`](ablation.ps1), [`Dockerfile`](Dockerfile)
- Locks and checkpoint manifests: [`runtime/uv.lock`](runtime/uv.lock), [`requirements.lock`](requirements.lock), [`models/model_assets.lock.yaml`](models/model_assets.lock.yaml), and [`models/asset_manifest.json`](models/asset_manifest.json)
- Schemas and tests: [`schemas/`](schemas/), [`tests/test_cuda_portability.py`](tests/test_cuda_portability.py)
- Collaborator input bundle: [`releases/ablation_cuda_input_v1.0.0.tar.gz`](releases/ablation_cuda_input_v1.0.0.tar.gz)
