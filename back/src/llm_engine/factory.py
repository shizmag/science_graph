"""
LLM Engine Factory.
"""

import os
from src.config import config
from src.llm_engine.base import BaseLLMEngine

def LLMEngine(use_cloud: bool = False, purpose: str = "index", *args, **kwargs) -> BaseLLMEngine:
    """Factory for returning the correct LLM Engine based on config/parameters."""
    import src.llm_engine
    is_cloud = use_cloud or os.environ.get("SCIENCE_GRAPH_USE_CLOUD") == "1" or config.llm_provider in ("openai", "openai-compatible")
    
    if purpose == "rag":
        if is_cloud:
            if src.llm_engine._cloud_rag_engine_singleton is None:
                from src.llm_engine import OpenAILLMEngine
                model_name = config.llm_cloud_rag_model_name
                src.llm_engine._cloud_rag_engine_singleton = OpenAILLMEngine(model_name=model_name, *args, **kwargs)
            return src.llm_engine._cloud_rag_engine_singleton
        else:
            if src.llm_engine._local_rag_engine_singleton is None:
                provider = config.llm_provider
                model_path = config.llm_local_rag_model_path
                if provider == "gguf":
                    from src.llm_engine import GgufLLMEngine
                    src.llm_engine._local_rag_engine_singleton = GgufLLMEngine(model_path=model_path, *args, **kwargs)
                elif provider == "cuda_torch":
                    from src.llm_engine import CudaTorchLLMEngine
                    src.llm_engine._local_rag_engine_singleton = CudaTorchLLMEngine(model_path=model_path, *args, **kwargs)
                elif provider in ("openai", "openai-compatible"):
                    from src.llm_engine import OpenAILLMEngine
                    from pathlib import Path
                    model_name = Path(model_path).name if model_path else "local-model"
                    base_url = config.llm_local_base_url
                    src.llm_engine._local_rag_engine_singleton = OpenAILLMEngine(
                        model_name=model_name,
                        base_url=base_url,
                        api_key="sk-optiq-local",
                        *args, **kwargs
                    )
                else:
                    from src.llm_engine import MlxLLMEngine
                    src.llm_engine._local_rag_engine_singleton = MlxLLMEngine(model_path=model_path, *args, **kwargs)
            return src.llm_engine._local_rag_engine_singleton
    else:
        if is_cloud:
            if src.llm_engine._cloud_engine_singleton is None:
                from src.llm_engine import OpenAILLMEngine
                src.llm_engine._cloud_engine_singleton = OpenAILLMEngine(model_name=config.llm_cloud_model_name, *args, **kwargs)
            return src.llm_engine._cloud_engine_singleton
        else:
            if src.llm_engine._local_engine_singleton is None:
                provider = config.llm_provider
                model_path = config.llm_local_model_path
                if provider == "gguf":
                    from src.llm_engine import GgufLLMEngine
                    src.llm_engine._local_engine_singleton = GgufLLMEngine(model_path=model_path, *args, **kwargs)
                elif provider == "cuda_torch":
                    from src.llm_engine import CudaTorchLLMEngine
                    src.llm_engine._local_engine_singleton = CudaTorchLLMEngine(model_path=model_path, *args, **kwargs)
                elif provider in ("openai", "openai-compatible"):
                    from src.llm_engine import OpenAILLMEngine
                    from pathlib import Path
                    model_name = Path(model_path).name if model_path else "local-model"
                    base_url = config.llm_local_base_url
                    src.llm_engine._local_engine_singleton = OpenAILLMEngine(
                        model_name=model_name,
                        base_url=base_url,
                        api_key="sk-optiq-local",
                        *args, **kwargs
                    )
                else:
                    from src.llm_engine import MlxLLMEngine
                    src.llm_engine._local_engine_singleton = MlxLLMEngine(model_path=model_path, *args, **kwargs)
            return src.llm_engine._local_engine_singleton


def create_provider_engine(
    provider_name: str,
    *,
    model_name: str,
    api_key: str,
    base_url: str,
) -> BaseLLMEngine:
    """Create a non-singleton engine for an explicitly selected API provider.

    Revision judges use this path to avoid changing the application's default
    generator or reusing the production judge singleton. Localhost endpoints are
    treated as gateways; this factory never starts or discovers a local model.
    """
    if provider_name.lower() not in {"omniroute", "openai-compatible"}:
        raise ValueError(f"Unsupported provider engine: {provider_name}")
    from src.llm_engine import OpenAILLMEngine

    return OpenAILLMEngine(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        manage_local_server=False,
        discover_local_model=False,
        serialize_local_requests=False,
    )
