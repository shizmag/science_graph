# Dependency inventory

Root pyproject.toml Python >=3.12 and workspace back (lines 1-18). Root/back uv locks differ: inspected Transformers 5.12.0 vs 5.9.0; both Torch 2.12.0. back/pyproject.toml:1-60 broadly declares mlx-lm, llama-cpp-python, SentenceTransformers, Torchvision, Transformers, USEARCH, spaCy, rank-bm25, datasets, Marker/PDF dependencies. back/Dockerfile.backend:1-31 installs manifest, not pinned lock.

Local selected env macOS ARM64/Python 3.13.12, Torch2.12, Transformers5.9, SentenceTransformers5.5.1; lacks mlx_lm, usearch, rank_bm25 and spaCy. Root .venv Python3.14.5 lacks PyYAML. No install occurred.

Fixed-snapshot runtime needs Python, YAML/config, SQLite FTS5, NumPy, Torch, Transformers, SentenceTransformers, USEARCH, benchmark/parser and three model assets. spaCy is conditional. Parquet/pandas, Marker/PDF/OCR, datasets and NER are not needed with fixed DB. MLX is Apple-only runtime; its dependency being platform-marked in lock means resolver failure is not proven, but provider MLX will not run as CUDA. Lexical path is SQLite FTS5, not rank-bm25. Native wheel/build risks include USEARCH, Torch/Torchvision, llama-cpp-python, tokenizers and PDF/OCR. Existing Compose and Python slim backend are not a CUDA harness.
