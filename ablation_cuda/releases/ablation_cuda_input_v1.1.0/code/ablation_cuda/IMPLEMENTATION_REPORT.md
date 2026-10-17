# Phase 3 implementation report

## Collaborator workflow

- Supported hosts: x86_64 Linux with Docker Engine/Buildx and NVIDIA Container Toolkit, or x64 Windows 11 with Docker Desktop Linux containers, WSL2 GPU integration, and an RTX 5090.
- One-command starts: `./ablation.sh run` on Linux; `.\ablation.ps1 run` in PowerShell.
- Host prerequisites: working NVIDIA driver and GPU passthrough, Docker, network access for first image/model acquisition, and sufficient disk space. The launchers do not install drivers or make privileged host changes.
- The input distribution contains the launchers, pinned runtime source/dependencies, Dockerfile, asset manifests, frozen 375-query dataset, and portable `external_db` snapshot. Models are acquired and checksummed automatically on first install.
- Frozen matrix: 14 arms, 375 questions each, 5,250 planned query-arm operations. Internal B6 maps to paper-facing B5.
- Full output example: `graphs/run_20261017T120000000000Z_OCC-RAG-1.7B_CUDA_ABLATION/`.
- Complete archive: `run_<id>_OCC-RAG-1.7B_CUDA_ABLATION.tar.gz`; incomplete archives include an explicit status suffix and retain the run directory.
- `run` resumes compatible incomplete work; a distinct run after completion requires `--new-run`. Completed `resume` only re-verifies identity and repackages existing results. Each new/resumed incomplete workload must pass the real CUDA smoke gate.

## Local verification

| Check | Status | Evidence |
|---|---|---|
| Focused Phase 3 infrastructure suite | `PASSED` | 12 tests, including matrix identity, checkpoint resume, interruption persistence, judge export, archive verification, and completed-run idempotence. |
| CUDA and portability suites | `PASSED` | 49 tests, including the completed-run idempotence case. |
| CPU/infrastructure interruption roundtrip | `PASSED` | Synthetic completed result restored from a fresh checkpoint index, validated for resume, packaged as partial, then verified and imported without claiming completion. |
| Input release validation/build | `PASSED` | Versioned v1.1.0 bundle built and validated; 375/300/75 population, frozen snapshot, persisted vectors, launchers, runtime source and locked dependencies included. |
| Launcher syntax | `PASSED` | `/bin/bash -n` on Linux launchers. PowerShell parser unavailable on this development host. |
| RTX 5090 CUDA smoke | `BLOCKED` | Development host is macOS arm64 and has no NVIDIA CUDA device or `nvidia-smi`. No claim of CUDA execution is made. |
| Full 5,250-operation scientific matrix | `NOT_RUN` | Correctly withheld because the mandatory real RTX 5090 smoke has not passed. |

The author imports collaborator output with `./ablation verify-results /path/to/run_<id>_OCC-RAG-1.7B_CUDA_ABLATION.tar.gz`. No remote LLM judge is called by the collaborator workflow.
