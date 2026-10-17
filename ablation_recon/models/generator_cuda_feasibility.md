# Generator CUDA feasibility

Primary: load exact OCC BF16 Hugging Face safetensors and tokenizer directly with PyTorch + Transformers. There is no MLX-to-Torch conversion needed. Keep template, prompt, tokenizer, context cap 12000, output cap 768, temperature 0/do_sample false, EOS IDs and stop behavior fixed. Current MLX wrapper uses the tokenizer chat template when no format marker is present and strips thinking tokens from returned text (back/src/llm_engine/mlx_impl.py:557-650); preserve raw generated sequence separately from parsed/stripped answer.

Comparison:
- PyTorch + Transformers: same weights/tokenizer; API exposes unprocessed logits with output_logits and return_dict_in_generate; most direct full-vocabulary H20/H50/H100 path. Primary recommendation.
- vLLM: model card lists architecture support, but full-distribution logit telemetry path is not verified in this project. Candidate only after output/stop parity.
- llama.cpp CUDA: requires an unverified GGUF; quantization can alter scores and full-vocab entropy path is unknown.
- MLX: historical Apple backend, not CUDA.

Project mlx_impl.py:557-650 uses MLX; project has no CUDA Torch generation engine. Transformers docs: [Transformers text generation docs](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation).

Current MLX telemetry in mlx_impl.py:124-193,198-305 selects top-k entries and renormalizes selected probabilities before entropy, storing H20/H50/H100, retained mass and top-one probability. It does not store full-vocabulary logits. Historical raw_logits.yaml is compact tokens, not raw distribution. Current telemetry differs from July-era single top-k metric. Define metric and preserve actual scores; do not claim full entropy.

On target GPU record model/tokenizer revisions, Torch/CUDA/driver, GPU, dtype, attention, prompt hash, tokens, stop, per-token score and entropy. Compare fixed prompt against reference with tolerances. Test EOS, stop sequence, 768 cap and near-12000 input. NVIDIA RTX 5090 specs are 32GB/CC12.0 and PyTorch 2.12 release describes CUDA13 Blackwell support: [RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/), [CUDA GPU table](https://developer.nvidia.com/cuda/gpus), [PyTorch 2.12 release notes](https://pytorch.org/blog/pytorch-2-12-release-blog/). Metadata only.
