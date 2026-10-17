# The manifest-list digest pins the official Python base for linux/amd64 and
# linux/arm64. Collaborator launchers select linux/amd64 for one common CUDA
# runtime on native Linux and Docker Desktop/WSL2.
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ARG UV_VERSION=0.10.9
ARG RELEASE_VERSION=1.0.0
LABEL org.opencontainers.image.title="Biomedical RAG benchmark CUDA runtime" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.base.name="python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    PYTHONPATH=/app/code:/app/code/back:/app/code/back/benchmarks/rag

WORKDIR /app
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"
COPY code/ /app/code/
RUN uv sync --project /app/code/ablation_cuda/runtime --locked --no-dev --no-install-project \
    && /app/code/ablation_cuda/runtime/.venv/bin/python -c "import sqlite3, torch, transformers, tokenizers, sentence_transformers, usearch; from src.vector_search import EmbeddingEngine; from src.services.rag_service import RAGService; from core.generation import run_query_on_baseline; from ablation_cuda.runtime.smoke import validate_release; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE fts_probe USING fts5(text)'); c.close(); assert torch.__version__.startswith('2.12.0+cu130'), torch.__version__; assert torch.version.cuda == '13.0', torch.version.cuda" \
    && rm -rf /tmp/uv-cache

ENV ABLA_RELEASE_ROOT=/app/release \
    ABLA_DB_ROOT=/db \
    ABLA_MODEL_ROOT=/models \
    ABLA_OUTPUT_ROOT=/outputs \
    ABLA_LOG_ROOT=/logs \
    HOME=/state/home

ENTRYPOINT ["/app/code/ablation_cuda/runtime/.venv/bin/python", "-m", "ablation_cuda.runtime.cli"]
