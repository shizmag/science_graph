# Phase 3.1 — Database Snapshot Distribution Verification

**Date:** 2026-10-19  
**Transfer release:** `ablation_cuda_input_v1.2.3.tar.gz` (bundled mode)

## Release identity

The supplied v1.1.0 archive was checked without modification. Its SHA-256 is the recorded `812c879c90b5b057ed977a58202dbd65253ebc33fa0b65c7dceb3172eaa13fb1`, and it contains the selected SQLite snapshot and all 375 dataset questions. The release builder does not include a USEARCH index; the index is intentionally rebuilt on the destination platform.

The final v1.2.3 archive includes:

| Asset | Identity |
| --- | --- |
| Archive | 104,414,463 bytes; SHA-256 `29288cfed280d6a4bc2ee803a6f536a6aa4750ec875217934b079360522070ec` |
| Snapshot | `external_db-20261016T193200Z`; 185,470,976 bytes (176.85 MiB); SHA-256 `c7fdd7c6fed69b788563896efaa8cbfe204216d6ff82043ff3a7414267aba8f0` |
| Dataset | 375 questions: 300 answerable and 75 designated negative; questions SHA-256 `d2820ced7c2994218245887ad753700923262fb91ec983eeacc01b217ec43b6b` |
| Retrieval fixtures | SHA-256 `4a575b00df48be6741187a49bd3bc926bffb641947ae7adcf666c0b9c19e9d07` |

The archive SHA-256 is also in `ablation_cuda_input_v1.2.3.tar.gz.sha256`. v1.1.0 remains byte-for-byte unchanged. v1.2.3 is the transfer target; it includes the clean-launch fixes and broader data/cache exclusions.

## Clean-environment acceptance

The v1.2.3 archive was copied to a fresh temporary directory, checksum-verified, and extracted there. The acceptance process ran packaged code from that extraction with a separate Python 3.12 runtime and persistent test directories. A macOS sandbox denied reads and writes to the original repository, the author-local `graph.db` directory, and author model-cache paths. The test did not import code from the repository or read a database outside the extracted release.

`python -m ablation_cuda.runtime.cli initialize` completed twice:

- First launch restored `/db/database.db` byte-for-byte from the bundled snapshot and rebuilt USEARCH from stored vectors. SQLite `quick_check` returned `ok`; the database had 346 documents, 22,068 chunks, 9,680 graph nodes, 11,712 graph edges, and 22,068 valid 768-dimensional vectors.
- The five chunk retrieval fixtures, FTS5 fixture, and three graph-node fixtures passed.
- The rebuilt test-host USEARCH index contained 22,068 rows at 768 dimensions, SHA-256 `512a8b2bdafdc3f07a8719872ae15afa1399ecfcecbaf276f73a51324ac84035`. The second launch reused that verified index (`usearch_rebuilt_from_persisted_vectors: false`) and the same database SHA.
- A recovery check replaced the managed database with a partial file and left a partial staging file. The next initialization quarantined invalid state, removed the staging file, restored the canonical database, rebuilt USEARCH, and reported `recomputed_embeddings: false`.
- Appending a byte to a copied archive caused its SHA sidecar check to fail. Removing `dataset/questions.jsonl` from an extracted release caused initialization to reject the inventory with a missing/extra-file error.
- The optional remote builder produced a temporary, untransferred variant with `snapshot/database.db` omitted and the exact same snapshot SHA/size and 375-question manifest. Its resolver was exercised against a local `file:` test source copied from the extracted snapshot and enforced the same size and canonical SHA. No private artifact URL or remote storage service is configured or required.

## Portability, identity, and storage

- Git no longer tracks database snapshots, SQLite backups, USEARCH indexes, model weights, or large result archives. Four previously tracked generated payloads are staged for index removal; their local working copies are preserved and ignored. `git ls-files` found no tracked database/index/archive/model-weight payloads. The repository has no `.gitattributes` LFS rules; the Git LFS command is unavailable here. Small manifests, schemas, code, and checksum sidecars remain eligible for Git.
- The release has no author home-directory, external-SSD, historical benchmark-directory, `science_graph.db`, or `~/.local/share/pdf-graph-analyzer/graph.db` dependency. The original repository and local DB paths were inaccessible during clean initialization.
- The Linux launcher builds from the release `code/` directory only. The SQLite snapshot, dataset, model cache, restored DB, USEARCH index, outputs, and return archive stay outside that context and the image layers. The generated `.dockerignore` and `back/.dockerignore` also exclude databases, indexes, model weights/caches, data directories, Parquet results, and archives.
- Retrieval and query generation both receive the managed `/db/database.db` path. Runtime and app-config checks require the restored path and expected database SHA; the benchmark resolver rejects implicit project/home DB discovery in strict mode. Missing or mismatched DB state is fatal.
- Linux persistent state defaults to `~/.local/state/ablation_cuda`; Windows defaults to `%LOCALAPPDATA%\ablation_cuda`. Database, model cache, logs, isolated home/config, outputs, and return files are separate persistent mounts. The return directory defaults under the state directory, outside the immutable release; launchers reject managed directories inside the release. The Linux launcher exercised this separation before stopping on the host architecture check.
- PowerShell passes Docker `--mount` specifications as argument-array entries so Windows paths containing spaces remain single arguments. It checks bind-directory write access, host free space, Linux-container mode, WSL2, and GPU prerequisites. `pwsh` was unavailable here, so Windows syntax and Docker Desktop/WSL2 behavior received static review only. The generic `docker-compose.yml` has relative mounts and no author-local DB dependency; its backend context has a `.dockerignore` for local research data.
- The conservative reservation is 54 GiB total: 1 GiB for archive/extraction work, 8 GiB for model cache, 1 GiB for SQLite/index, 16 GiB for outputs, 16 GiB for returned evidence, and 12 GiB for Docker image/build storage. The measured public model-weight floor is 5,023,979,594 bytes. Launchers report actionable host-capacity failures; runtime preflight checks mounted filesystems and Docker storage before model acquisition or restoration. Because transfer is a manual single-archive handoff, the recipient must reserve the documented space before downloading/extracting; the launcher repeats the capacity check after extraction.
- Run outputs live under `outputs/graphs/run_...`; the portable results archive is written under the separate return mount. The packer archives the run directory and its evidence only, not the input SQLite file, USEARCH index, or model cache. The full CUDA result archive was not produced on this machine.

## Checks and limitations

Passed: v1.1.0 checksum/content audit; v1.2.3 checksum and clean extraction; complete dataset and package-manifest validation; SQLite restore and integrity checks; vector, graph, FTS5, and retrieval-fixture validation; USEARCH rebuild and repeat-launch reuse; partial-restore recovery; corrupted-archive and partial-extraction rejection; remote builder identity and resolver checksum path; simulated low-space rejection with an actionable 42 GiB diagnostic; `uv lock --check`; `bash -n`; Python syntax parsing; Git and Docker-context audits.

CUDA and Windows acceptance remain outstanding. This machine is macOS `arm64`; Docker reports Linux `aarch64`, while the locked image targets `linux/amd64`. `nvidia-smi` and PowerShell are unavailable. The launcher correctly stopped before installation on the architecture check. The Linux/amd64 image build, RTX 5090 CUDA smoke, pinned model downloads, 5,250-operation run, and Windows 11/Docker Desktop/WSL2 launch were not executed.

## Transfer instructions

Transfer the single payload `ablation_cuda_input_v1.2.3.tar.gz` and provide its SHA-256 above (the small `.sha256` sidecar may be sent with it). No Git checkout, Git LFS fetch, original database, or model cache is needed. Ask the recipient to reserve 54 GiB before transfer/extraction.

Linux:

```sh
shasum -a 256 ablation_cuda_input_v1.2.3.tar.gz
# Compare with 29288cfed280d6a4bc2ee803a6f536a6aa4750ec875217934b079360522070ec
mkdir -p ablation_cuda_input_v1.2.3
tar -xzf ablation_cuda_input_v1.2.3.tar.gz -C ablation_cuda_input_v1.2.3
cd ablation_cuda_input_v1.2.3
./ablation.sh run
```

Windows PowerShell:

```powershell
(Get-FileHash -Algorithm SHA256 .\ablation_cuda_input_v1.2.3.tar.gz).Hash
# Compare with 29288cfed280d6a4bc2ee803a6f536a6aa4750ec875217934b079360522070ec
New-Item -ItemType Directory -Force .\ablation_cuda_input_v1.2.3 | Out-Null
tar -xzf .\ablation_cuda_input_v1.2.3.tar.gz -C .\ablation_cuda_input_v1.2.3
Set-Location .\ablation_cuda_input_v1.2.3
.\ablation.ps1 run
```
