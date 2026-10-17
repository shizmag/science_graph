# Collaborator guide: full CUDA ablation

## Supported host

- 64-bit x86_64 Linux with Docker Engine, Docker Buildx, NVIDIA Container Toolkit, and an NVIDIA RTX 5090.
- x64 Windows 11 with Docker Desktop using Linux containers, WSL2 GPU integration, and an NVIDIA RTX 5090.
- Host NVIDIA drivers must already work. The launcher checks Linux driver 580.65.06+ or Windows/WSL driver 580.88+ for the pinned CUDA 13 runtime. It does not install drivers, change system configuration, or make privileged host changes.

Network access is needed for the first image build and pinned public model downloads. Allow adequate local disk space for image layers, model caches, logs, and experiment outputs. The collaborator package includes project code and frozen inputs; no repository clone or host Python/pip/uv/Hugging Face/database commands are required.

## Start

Linux:

```sh
./ablation.sh run
```

Windows PowerShell:

```powershell
.\ablation.ps1 run
```

`run` checks the host, builds the included Linux/amd64 image if needed, downloads and verifies pinned checkpoints, restores the frozen `external_db` snapshot, rebuilds USEARCH from the 22,068 persisted vectors, and verifies the frozen 375-query dataset. It then runs the mandatory real CUDA smoke: E5, Cross-Encoder, OCC-RAG generation, token logits/entropy, retrieval/context, and a trace-proven component-off path. Only a passing RTX 5090 smoke allows the 5,250-operation full matrix to start.

The full matrix has 14 arms × 375 questions. Progress is printed in a Rich panel on an interactive terminal and as periodic structured lines otherwise. `status` reads the live progress snapshot. Checkpoints and logs are written continuously.

## Resume and outputs

`run` resumes a compatible incomplete run automatically. Add `--new-run` to create a distinct run after an earlier run has completed. To resume a named run explicitly:

```sh
./ablation.sh resume run_<id>_OCC-RAG-1.7B_CUDA_ABLATION
```

The PowerShell command is `.\ablation.ps1 resume <run-directory-name>`. A failed smoke leaves diagnostic artifacts and stops before full generation. After correcting the reported issue, rerun `run` or `resume`.

The experiment directory is under the state volume's `outputs/graphs/`. The portable results archive and `.sha256` sidecar are under `return/`. The launcher prints their exact paths, checksum, status, and transfer instructions. Copy the full-run archive and sidecar to the author. Do not send model caches or database files.

The collaborator runs generation and local telemetry only. No remote LLM judge is called. The author runs independent judging after importing the archive.
