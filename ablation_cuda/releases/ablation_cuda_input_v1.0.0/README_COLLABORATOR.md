# Biomedical RAG CUDA release

This frozen package contains the exact 375-question manifest and the author-selected `external_db` snapshot only. The original July database identity remains unproven.

Run `./ablation.sh doctor`, then `./ablation.sh smoke`. `smoke` builds the pinned Linux image, checks NVIDIA access, restores SQLite, rebuilds USEARCH from stored vectors, downloads pinned public model revisions, and runs only `pubmedqa_21084567` under B6 and one trace-selected component-off arm. It prefers `score_blending=false` when the ranking trace proves an active blend over at least two candidates; otherwise it uses `graph_expansion=false` only if B6 inserted a dynamic graph block. It exports the result archive into `return/`.

Windows users run `./ablation.ps1 doctor` and `./ablation.ps1 smoke` in PowerShell from this directory. Docker Desktop must use Linux containers and WSL2 GPU passthrough must already work. The scripts do not install or change host drivers.

Return the generated `return/ablation_cuda_results_v1_<runid>.tar.gz` archive and its matching `.sha256` file. The author can verify/import it with `./ablation verify-results <archive>`. No cloud judge is called.
