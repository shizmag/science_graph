# CUDA feasibility

NVIDIA lists RTX 5090 at 32GB and CC12.0; PyTorch 2.12 release metadata discusses CUDA13 Blackwell support. Sources: [RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/), [CUDA GPU table](https://developer.nvidia.com/cuda/gpus), [PyTorch 2.12 release notes](https://pytorch.org/blog/pytorch-2-12-release-blog/). This is metadata, not project execution.

Generator BF16 weights are ~3.44GB disk; KV cache, activations and full logits add VRAM. 12K context may matter. 32GB is plausible but peak unknown. E5 CUDA inference is plausible, unrun. Cross-Encoder is small but current source chooses CPU/MPS. USEARCH is CPU-side. BF16 and attention kernels must be validated in chosen wheel; avoid mandatory custom kernels before proving compatibility. Host RAM, throughput, batches, peak VRAM and full-logit disk are unknown.

Implemented: embedder device selection can use CUDA. Metadata-supported: recent Torch/Blackwell and container path. Untested: generator, E5, CE on GPU, full pipeline, sidecar restore, Windows WSL, memory/performance. No local GPU or nvidia-smi. Use CUDA_VERIFIED only after target run records GPU, driver, framework, parameter placement and real inference activity.
