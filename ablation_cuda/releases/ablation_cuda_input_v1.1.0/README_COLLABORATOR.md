# Biomedical RAG full CUDA ablation

This self-contained distribution contains the exact frozen 375-query dataset, the selected `external_db` snapshot, runtime source, Linux and PowerShell launchers, and locked dependencies. The original July database identity remains unproven.

## Run the experiment

Linux: `./ablation.sh run`

Windows PowerShell: `./ablation.ps1 run`

The launcher performs host and Docker checks, builds the pinned Linux image from the included source, acquires pinned public model revisions, restores SQLite and rebuilds USEARCH from its 22,068 stored vectors, then runs a real CUDA B6 plus component-off smoke. The full 5,250 query-arm workload starts only if that smoke passes on an RTX 5090. It writes live progress and resumable checkpoints. A compatible interrupted run is resumed by `run`; use `--new-run` only to start a distinct experiment.

## Host requirements

Use x86_64 Linux with Docker Engine, Buildx, NVIDIA Container Toolkit and an NVIDIA RTX 5090, or x64 Windows 11 with Docker Desktop Linux containers, WSL2 GPU integration and an NVIDIA RTX 5090. Host NVIDIA drivers must already be installed and working. Linux requires driver 580.65.06 or newer; Windows/WSL requires 580.88 or newer for this CUDA 13 image. The launcher reports missing prerequisites and remediation; it does not install drivers or change privileged host settings. Ensure adequate free disk space for model caches, build layers, logs and results.

If the smoke fails, the full workload does not start. Its diagnostic directory and archive remain available. Fix the reported host or runtime problem and run `./ablation.sh resume` (or the PowerShell equivalent).

On completion, send the printed full-run `.tar.gz` archive and matching `.sha256` file to the author. The archive contains prompts, retrieved evidence, generations, traces, telemetry, local metrics and blinded judge inputs; it excludes model caches and the database copy. No cloud judge is called on the collaborator machine.
