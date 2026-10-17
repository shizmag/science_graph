# Container strategy

| Option | Strength | Risk | Recommendation |
|---|---|---|---|
| Prebuilt pinned OCI | immutable digest, lock and SBOM; shared Linux image | registry/transfer size; host GPU prerequisites | primary |
| Runtime build | reproducible if base/package locks pinned | native builds, toolchains, registry variability, disk | development fallback |
| Native uv | package pinning | host CUDA/native differences and OS matrix | not first choice |

Create minimal linux/amd64 CUDA benchmark image, exclude MLX and ingestion dependencies, pass host GPU, mount DB/output and persistent model cache. One process is simplest; if subprocesses remain, mount exact same manifest/DB and pass explicit paths. SQLite needs no service or exposed port. Network only outbound for asset fetch/store. Add preflight for GPU, FTS5, writable disk and hashes.

Existing docker-compose.yml is web stack; back/Dockerfile.backend is generic Python slim/no CUDA and does not prove this runtime. Keep models outside image. If registry unavailable, provide checksummed OCI archive after measuring size.
