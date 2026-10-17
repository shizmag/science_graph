"""
LLM Engine Package.
"""

from src.llm_engine.base import (
    BaseLLMEngine,
    strip_thinking_tokens,
    ResilientParser,
    retry_with_temp_decay,
    retry_with_temp_decay_async,
    StructuredOutput,
)
from src.llm_engine.mlx_impl import MlxLLMEngine, ConstrainedLogitsProcessor, build_mlx_tokenizer_data
from src.llm_engine.gguf_impl import GgufLLMEngine
from src.llm_engine.openai_impl import OpenAILLMEngine, AsyncRateLimiter
from src.llm_engine.cuda_torch_impl import CudaTorchLLMEngine
from src.llm_engine.factory import LLMEngine

_local_engine_singleton = None
_cloud_engine_singleton = None
_local_rag_engine_singleton = None
_cloud_rag_engine_singleton = None

__all__ = [
    "LLMEngine",
    "BaseLLMEngine",
    "MlxLLMEngine",
    "GgufLLMEngine",
    "OpenAILLMEngine",
    "CudaTorchLLMEngine",
    "strip_thinking_tokens",
    "ResilientParser",
    "retry_with_temp_decay",
    "retry_with_temp_decay_async",
    "ConstrainedLogitsProcessor",
    "build_mlx_tokenizer_data",
    "AsyncRateLimiter",
    "StructuredOutput",
]

