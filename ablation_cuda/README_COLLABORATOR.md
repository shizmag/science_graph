# Collaborator guide: full CUDA ablation

## Supported host

- 64-bit x86_64 Linux with Docker Engine, Docker Buildx, NVIDIA Container Toolkit, and an NVIDIA RTX 5090.
- x64 Windows 11 with Docker Desktop using Linux containers, WSL2 GPU integration, and an NVIDIA RTX 5090.
- Host NVIDIA drivers must already work. The launcher checks Linux driver 580.65.06+ or Windows/WSL driver 580.88+ for the pinned CUDA 13 runtime. It does not install drivers, change system configuration, or make privileged host changes.

Network access is needed for the first image build and pinned public model downloads. The v1.2.3 bundled release includes the selected SQLite snapshot and all 375 questions; no repository clone, Git LFS, original database, historical benchmark directory, or author filesystem path is required.

Before downloading or extracting the archive, reserve at least 54 GiB across the release, persistent data, return, and Docker storage locations. This includes 1 GiB of archive/extraction space, 8 GiB for model cache, 1 GiB for SQLite and USEARCH restoration, 16 GiB for experiment outputs, 16 GiB for the returned archive, and 12 GiB for Docker image/build storage. The measured public model-weight floor is 5,023,979,594 bytes. The launchers check host volumes, and the container checks its writable Docker storage before model downloads. Keep at least 12 GiB available inside Docker Desktop's Linux disk image as well.

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

Database, model, home/config, log, output, and return directories are persistent bind mounts outside the immutable input release. By default they live under `~/.local/state/ablation_cuda` on Linux or `%LOCALAPPDATA%\ablation_cuda` on Windows; `ABLA_STATE_DIR` and `ABLA_RETURN_DIR` can move them to other writable persistent storage. Docker builds use only the release `code/` directory; the snapshot, dataset, model files, and results are outside the build context and image layers. SQLite restore and USEARCH rebuild use the same `/db/database.db` identity for graph retrieval and vector retrieval. The restored database and index are verified and reused on repeated launches.

The full matrix has 14 arms × 375 questions. Progress is printed in a Rich panel on an interactive terminal and as periodic structured lines otherwise. `status` reads the live progress snapshot. Checkpoints and logs are written continuously.

## Resume and outputs

`run` resumes a compatible incomplete run automatically. Add `--new-run` to create a distinct run after an earlier run has completed. To resume a named run explicitly:

```sh
./ablation.sh resume run_<id>_OCC-RAG-1.7B_CUDA_ABLATION
```

The PowerShell command is `.\ablation.ps1 resume <run-directory-name>`. A failed smoke leaves diagnostic artifacts and stops before full generation. After correcting the reported issue, rerun `run` or `resume`.

The experiment directory is under the state volume's `outputs/graphs/`. The portable results archive and `.sha256` sidecar are under `return/`. The launcher prints their exact paths, checksum, status, and transfer instructions. Copy the full-run archive and sidecar to the author. Do not send model caches or database files.

The collaborator runs generation and local telemetry only. No remote LLM judge is called. The author runs independent judging after importing the archive.

To unpack the transferred archive, verify the sidecar before extraction:

```sh
sha256sum -c ablation_cuda_input_v1.2.3.tar.gz.sha256
tar -xzf ablation_cuda_input_v1.2.3.tar.gz
```

In PowerShell, compare `(Get-FileHash -Algorithm SHA256 .\ablation_cuda_input_v1.2.3.tar.gz).Hash` with the first value in `ablation_cuda_input_v1.2.3.tar.gz.sha256`, then run `tar -xzf .\ablation_cuda_input_v1.2.3.tar.gz`. A missing file, partial extraction, or changed payload is rejected before model downloads. Run the launcher from the extracted release directory.
