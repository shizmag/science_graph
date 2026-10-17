# Biomedical RAG CUDA phase 2

This directory contains the author-side release builder and verifier, the
benchmark-only Linux CUDA runtime, Linux and Windows host launchers, pinned
runtime dependencies, frozen checkpoint lock and asset manifests in `models/`,
schemas, and CPU tests. The released execution scope is
one frozen `pubmedqa_21084567` query under B6 and one trace-selected component
off arm. It does not start the 375-query matrix or call a cloud judge.

## Author workflow

Build and validate the versioned input package:

```bash
./ablation prepare-release
```

The package contains the canonical 375-query manifest, the author-selected
`external_db` SQLite snapshot, stored vectors and graph data, retrieval
fixtures, effective configuration, model locks, code, and checksums. It excludes
the other candidate database. The selected snapshot is a new experimental
baseline; this package does not claim July database identity.

By default, the command leaves the archive in `ablation_cuda/releases/` and
prints its exact path and SHA-256. If `ABLA_ARTIFACT_STORE` is unset, the author
must stage that archive and its `.sha256` file in the private collaborator
transfer location. A local artifact directory can be configured with
`ABLA_ARTIFACT_STORE=/path/to/private-transfer`; private S3 requires a configured
`s3://` destination and the explicit `--authorize-remote-upload` option. No
public upload is performed.

Verify and import the returned result archive on the author machine:

```bash
./ablation verify-results /path/to/ablation_cuda_results_v1_<runid>.tar.gz
```

The verifier checks the external archive SHA-256 and all internal checksums,
compares the result to the frozen question, database, and model manifests,
validates both arm configurations and CUDA evidence, rejects missing/error
states, writes a two-answer `preview.md`, and imports into a new directory
without replacing previous research artifacts.

## Collaborator workflow

The collaborator receives and extracts the single versioned input archive, then
runs one command from its root.

Linux:

```bash
./ablation.sh smoke
```

Windows PowerShell with Docker Desktop and WSL2:

```powershell
.\ablation.ps1 smoke
```

`smoke` performs host and in-container CUDA preflight, builds the pinned Linux
image if needed, downloads and validates the pinned public checkpoints, restores
the database, rebuilds USEARCH from stored vectors, then runs only the two
frozen arms. Model cache, database/index, logs, outputs, and returned archives
use persistent host volumes. The generated
`return/ablation_cuda_results_v1_<runid>.tar.gz` and matching `.sha256` file are
the files to return.

The launchers also provide `doctor`, `install`, `status`, `pack`, and `verify`.
`verify` runs the same standard-library-only verifier inside the pinned image
and imports to `return/imported/`. No command silently changes a host driver,
performs administrative setup, or restarts a host.

## Host prerequisites

Linux requires Docker Engine with Buildx, Linux containers, an NVIDIA Linux
driver supporting CUDA 13 / Blackwell (the preflight minimum is 580.65.06), the
NVIDIA Container Toolkit, and a visible RTX 5090. Windows requires WSL2, Docker
Desktop in Linux-container mode with WSL2 integration, NVIDIA's Windows WSL CUDA
driver (preflight minimum 580.88), and working GPU passthrough into both WSL2 and
Docker. The launchers print remediation guidance and do not install or replace
drivers or change administrative host settings.

The image is pinned to the official Python 3.12 slim Bookworm manifest digest;
the locked runtime uses PyTorch 2.12.0 with CUDA 13.0 wheels and exact versions
and hashes in `runtime/uv.lock`. The image records the built OCI digest in the
run manifest. Runtime operation checks execute real CUDA operations and reject
CPU fallback; a macOS build or CPU test is not evidence that CUDA passed.

## Local CPU validation

From the repository root, using its Python 3.12 portability environment:

```bash
PYTHONPATH="$PWD/back:$PWD/back/benchmarks/rag:$PWD" \
  ./ablation_portability/.venv/bin/python -m pytest -q ablation_cuda/tests
```

These tests cover locked asset metadata and cache handling, SQLite restore and
vector-index reconstruction, fixture replay, bootstrap behavior where it can be
checked statically, configuration/result identity, intervention isolation,
telemetry math, archive roundtrips, and corrupt/incomplete archive rejection.
They do not claim Windows or NVIDIA hardware execution.
