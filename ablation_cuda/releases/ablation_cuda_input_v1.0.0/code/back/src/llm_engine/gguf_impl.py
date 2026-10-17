"""
GGUF LLM Engine implementation using llama-cpp-python for cross-platform local inference.
"""

import os
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel

from src.config import config
from src import console as con
from src.llm_engine.base import BaseLLMEngine, strip_thinking_tokens


class GgufLLMEngine(BaseLLMEngine):
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.llm_local_model_path
        self.model = None

        # GGUF models are single files. However, we check exists to be flexible
        # in case the user configures a directory or file path.
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Local GGUF model path not found: {self.model_path}\n"
                f"  Please verify the model path in config.yaml."
            )

    def unload_model(self):
        if self.model is not None:
            import gc
            try:
                # llama_cpp.Llama instances have a close method to free resources
                if hasattr(self.model, "close"):
                    self.model.close()
            except Exception as e:
                con.warning(f"Error closing GGUF model: {e}")
            self.model = None
            gc.collect()
            con.success("GGUF LLM model unloaded and memory cleared")

    def _ensure_model_loaded(self):
        if self.model is None:
            model_name = Path(self.model_path).name
            con.model_msg(f"Loading GGUF LLM [bold]{model_name}[/bold] …")

            try:
                from llama_cpp import Llama
            except ImportError:
                con.error("Failed to import llama-cpp-python.")
                con.info("Please install it in your environment: uv add llama-cpp-python")
                raise ImportError(
                    "llama-cpp-python is not installed. "
                    "Run `uv add llama-cpp-python` to install it for GGUF model support."
                )

            # Get GGUF-specific settings from config or use defaults
            gguf_config = config.data["llm"].get("gguf", {})
            n_gpu_layers = gguf_config.get("n_gpu_layers", -1)
            n_ctx = gguf_config.get("n_ctx", config.llm_model_max_context or 4096)

            with con.suppress_stderr(), con.suppress_stdout():
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )

            con.success(f"GGUF LLM ready: [bold]{model_name}[/bold] (GPU layers: {n_gpu_layers}, Context: {n_ctx})")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        self._ensure_model_loaded()
        try:
            return len(self.model.tokenize(text.encode("utf-8"), special=True))
        except Exception:
            return len(text) // 4

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: Optional[str] = None) -> str:
        self._ensure_model_loaded()
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            if task == "extraction":
                resolved_max_tokens = config.llm_extraction_output_limit
            elif task == "clustering":
                resolved_max_tokens = config.llm_clustering_output_limit
            elif task == "synthesis":
                resolved_max_tokens = config.llm_synthesis_output_limit
            
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp

        # Check if the prompt is already formatted with chat template structure
        is_formatted = any(
            tag in prompt
            for tag in [
                "<|im_start|>",
                "<|start_header_id|>",
                "[INST]",
                "<start_of_turn>",
                "<|im_end|>"
            ]
        )

        if not is_formatted:
            # Use create_chat_completion to automatically apply the model's chat template
            try:
                messages = [{"role": "user", "content": prompt}]
                response_dict = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                )
                response = response_dict["choices"][0]["message"]["content"]
            except Exception as e:
                con.warning(f"GGUF chat completion failed: {e}. Falling back to raw completion.")
                response_dict = self.model(
                    prompt,
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                )
                response = response_dict["choices"][0]["text"]
        else:
            response_dict = self.model(
                prompt,
                max_tokens=resolved_max_tokens,
                temperature=temp,
            )
            response = response_dict["choices"][0]["text"]

        return strip_thinking_tokens(response)

    def generate_json(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        self._ensure_model_loaded()
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp
        schema = schema_class.model_json_schema()

        # Check if the prompt is already formatted
        is_formatted = any(
            tag in prompt
            for tag in [
                "<|im_start|>",
                "<|start_header_id|>",
                "[INST]",
                "<start_of_turn>",
                "<|im_end|>"
            ]
        )

        # Try using llama-cpp-python's built-in JSON schema response format
        try:
            if not is_formatted:
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a strict JSON extractor. Output ONLY valid JSON matching this schema: {schema}"
                    },
                    {"role": "user", "content": prompt}
                ]
                response_dict = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                    response_format={
                        "type": "json_object",
                        "schema": schema,
                    }
                )
                response = response_dict["choices"][0]["message"]["content"]
            else:
                response_dict = self.model(
                    prompt,
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                    response_format={
                        "type": "json_object",
                        "schema": schema,
                    }
                )
                response = response_dict["choices"][0]["text"]
        except Exception as e:
            con.warning(f"GGUF response_format JSON generation failed: {e}. Falling back to unconstrained generation.")
            # Fallback to unconstrained generation, parsing will be done by ResilientParser
            if not is_formatted:
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a strict JSON extractor. Output ONLY valid JSON matching this schema: {schema}"
                    },
                    {"role": "user", "content": prompt}
                ]
                response_dict = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                )
                response = response_dict["choices"][0]["message"]["content"]
            else:
                response_dict = self.model(
                    prompt,
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                )
                response = response_dict["choices"][0]["text"]

        return strip_thinking_tokens(response)

    async def generate_response_async(
        self,
        prompt: str,
        max_tokens: int = None,
        temp: float = None,
        task: str = None,
        model: Optional[str] = None,
    ) -> str:
        import asyncio
        await asyncio.sleep(0)
        return self.generate_response(prompt, max_tokens, temp, task, model=model)

    async def generate_json_async(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        import asyncio
        await asyncio.sleep(0)
        return self.generate_json(prompt, schema_class, temp, max_tokens)
