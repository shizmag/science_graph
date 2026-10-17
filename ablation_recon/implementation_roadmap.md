# Implementation roadmap

| Stage | Tasks | Likely files | Acceptance | Dependencies and complexity |
|---|---|---|---|---|
| A Source/artifact freeze | Decide historical DB vs new baseline; extract/hash 375 manifest; pin corpus/index/models/config; redact secrets | new staging tools, run_pipeline.py, core/config.py | exact 300/75 IDs/text/order; accepted DB fingerprint; hashes or explicit unknowns | author backup/license decisions; high scientific risk, medium-high effort |
| B Portable snapshot | SQLite online backup; restore; validate FTS/vector/graph; rebuild USEARCH; one path across subprocesses | sqlite_impl.py, pipeline and CLI | clean Linux restore, same DB fingerprint all stages, retrieval fixture parity or new-baseline label | A; medium effort, high impact |
| C CUDA environment | Minimal pinned Linux image; exclude MLX/ingestion; exact HF Torch generator; CE CUDA | benchmark lock, Dockerfile, engine factory, rag_service.py | image digest/lock/SBOM; generator/E5/CE execute on RTX; devices proven, memory measured | GPU required; high effort/kernel risk |
| D One-query E2E | Real frozen query, full traces, metric definition, archive | pipeline/tracing/persistence | exit 0; lexical+dense+graph, CE/generator CUDA, answer/parser/telemetry, archive verifies | A-C; medium-high |
| E Two-arm micro-ablation | Full B6 and one actually exercised component-off; invariants/cache identity | config resolver, flags, checkpoint/cache | one intended intervention; trace effect; all other fingerprints fixed | D; no-op/config risk |
| F Full harness | Approved matrix, progress, resume, atomic checkpoints, caching and packaging | new harness/core generation | interrupted resume, no contamination, all arms and complete archive | E; high |
| G Return/import | Validate archive and prepare later independent Judge input, no request | new importer/schema | valid one-to-one data imports; tampering/mismatch rejected | F; medium |

Primary target Linux x86_64. Windows means Windows 11 WSL2/Docker Desktop after host prerequisites. Do not begin full matrix until A-C pass and relevant graph/intervention path is observable.
