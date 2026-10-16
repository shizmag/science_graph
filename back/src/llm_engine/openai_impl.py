"""
OpenAI LLM Engine implementation for cloud model provider.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel
from src.config import config
from src import console as con
from src.llm_engine.base import BaseLLMEngine, strip_thinking_tokens, _local_request_lock


class AsyncRateLimiter:
    """Enforces a minimum interval between requests to avoid overloading providers."""
    def __init__(self, delay: float):
        self.delay = delay
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        if self.delay <= 0:
            return
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.delay:
                sleep_time = self.delay - elapsed
                await asyncio.sleep(sleep_time)
            self.last_request_time = time.monotonic()


class OpenAILLMEngine(BaseLLMEngine):
    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None,
        *,
        manage_local_server: bool = True,
        discover_local_model: bool = True,
        serialize_local_requests: bool = True,
    ):
        import openai
        api_key = api_key or config.llm_cloud_api_key
        if not api_key and config.llm_provider == "openai-compatible":
            api_key = "sk-optiq-local"

        base_url = base_url or config.llm_cloud_base_url
        self.model_name = model_name or config.llm_cloud_model_name

        if not api_key:
            con.error("API key is not configured for OpenAI/OpenRouter.")
            raise ValueError("Missing API key for OpenAI provider")

        # Determine MTP details
        mtp_requested = config.llm_enable_mtp
        mtp_file_found = config.llm_mtp_file_found
        mtp_effective = config.llm_effective_mtp_mode
        model_path = config.llm_model_path
        actual_base_url = base_url or "https://api.openai.com/v1"

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"MTP requested: {str(mtp_requested).lower()}")
        logger.info(f"MTP file found: {str(mtp_file_found).lower()}")
        logger.info(f"MTP effective mode: {'enabled' if mtp_effective else 'disabled'}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Backend base URL: {actual_base_url}")
        logger.info("Request mode: one-shot")
        logger.info("Token limits source: existing project config")
        logger.info("Token estimation source: existing tiktoken mechanism")

        if mtp_requested and not mtp_file_found:
            logger.warning("MTP file (mtp.safetensors) is missing in model path. MTP will be disabled.")
            con.warning("MTP file (mtp.safetensors) is missing in model path. MTP will be disabled.")

        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url

        # Automatically manage background server lifecycle for local endpoints
        self.server_process = None
        is_local = base_url and ("localhost" in base_url or "127.0.0.1" in base_url)
        
        # Only start server if we are not running a pytest test run
        if is_local and manage_local_server and not os.environ.get("PYTEST_CURRENT_TEST"):
            server_running = False
            try:
                import openai
                temp_client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=1.0)
                temp_client.models.list()
                server_running = True
            except Exception:
                pass

            if not server_running:
                cmd = config.llm_expected_launch_command
                if cmd:
                    con.info(f"Local server not detected on {base_url}. Starting background server: [bold]{cmd}[/bold]")
                    import subprocess
                    env = os.environ.copy()
                    local_bin = os.path.expanduser("~/.local/bin")
                    if local_bin not in env.get("PATH", ""):
                        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"
                    
                    self.server_process = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env
                    )
                    
                    import atexit
                    def cleanup(proc):
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                    atexit.register(cleanup, self.server_process)
                    
                    # Wait for server to become ready
                    import time
                    ready = False
                    temp_client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=1.0)
                    for _ in range(60): # 30 seconds max
                        try:
                            temp_client.models.list()
                            ready = True
                            break
                        except Exception:
                            if self.server_process.poll() is not None:
                                break
                            time.sleep(0.5)
                    
                    if not ready:
                        try:
                            self.server_process.terminate()
                            self.server_process.wait(timeout=3)
                        except Exception:
                            pass
                        raise ConnectionError(
                            f"Failed to start local server using command: {cmd}\n"
                            "Please ensure optiq is installed and the model path is correct."
                        )
                    con.success("Local LLM server is up and ready!")

        self.client = openai.OpenAI(**client_args)
        self._is_local = is_local
        self._serialize_local_requests = serialize_local_requests
        self.rate_limiter = AsyncRateLimiter(config.llm_request_delay)

        # Auto-discover the actual model name from the local server.
        # Local servers (optiq, ollama, vLLM) expose loaded models via /v1/models.
        # Using the wrong name causes the server to try downloading from HuggingFace.
        if is_local and discover_local_model and not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                models_response = self.client.models.list()
                available = [m.id for m in models_response.data]
                if config.llm_model_path in available and is_local and available:
                    discovered = config.llm_model_path
                else:
                    discovered = available[-1]
                if discovered != self.model_name:
                    con.info(f"Auto-discovered local model name: [bold]{discovered}[/bold] (was: {self.model_name})")
                    self.model_name = discovered
            except Exception:
                pass  # Fall back to configured model_name

        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            self.tokenizer = None

        con.success(f"OpenAI API LLM ready: [bold]{self.model_name}[/bold]")

    def _call_completions_with_lock(self, *args, **kwargs):
        if self._is_local and self._serialize_local_requests:
            with _local_request_lock:
                return self.client.chat.completions.create(*args, **kwargs)
        else:
            return self.client.chat.completions.create(*args, **kwargs)

    def chat_completion(self, messages, *, model=None, temperature=0.0, response_format=None, max_tokens=None):
        """Call the configured OpenAI-compatible provider with explicit chat messages.

        This method lets evaluators reuse the application's provider engine while
        keeping system/user prompts and structured-output settings explicit.
        """
        kwargs = {
            "model": model or self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return self._call_completions_with_lock(**kwargs)

    def _truncate_to_context(self, text: str, max_input_tokens: int) -> str:
        if self.tokenizer is None:
            return text[:max_input_tokens * 4]
        try:
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) <= max_input_tokens:
                return text
            return self.tokenizer.decode(token_ids[:max_input_tokens])
        except Exception:
            return text[:max_input_tokens * 4]

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: Optional[str] = None) -> str:
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
        model_to_use = model if model else self.model_name

        try:
            response = self._call_completions_with_lock(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=resolved_max_tokens,
                temperature=temp,
            )
        except Exception as e:
            import openai
            is_connection_error = isinstance(e, (openai.APIConnectionError, ConnectionError, OSError))
            is_local = "localhost" in str(self.client.base_url) or "127.0.0.1" in str(self.client.base_url)
            if is_local and is_connection_error:
                raise ConnectionError(
                    f"Failed to connect to local OpenAI-compatible server at {self.client.base_url}.\n"
                    "Please ensure that your local server (e.g., 'optiq serve' or 'ollama') is running and healthy."
                ) from e
            raise e
        return strip_thinking_tokens(response.choices[0].message.content)

    async def generate_response_async(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: Optional[str] = None) -> str:
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
        model_to_use = model if model else self.model_name

        max_retries = 3
        backoff = config.llm_retry_backoff
        last_err = None

        for attempt in range(max_retries):
            await self.rate_limiter.wait()
            try:
                response = await asyncio.to_thread(
                    self._call_completions_with_lock,
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                )
                return strip_thinking_tokens(response.choices[0].message.content)
            except Exception as e:
                last_err = e
                import openai
                is_connection_error = isinstance(e, (openai.APIConnectionError, ConnectionError, OSError))
                is_local = "localhost" in str(self.client.base_url) or "127.0.0.1" in str(self.client.base_url)
                if is_local and is_connection_error:
                    raise ConnectionError(
                        f"Failed to connect to local OpenAI-compatible server at {self.client.base_url}.\n"
                        "Please ensure that your local server (e.g., 'optiq serve' or 'ollama') is running and healthy."
                    ) from e
                if attempt == max_retries - 1:
                    raise e
                sleep_time = backoff * (2 ** attempt)
                con.warning(f"LLM API request failed: {e}. Retrying in {sleep_time}s...")
                await asyncio.sleep(sleep_time)
        raise last_err

    def generate_json(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp

        messages = [
            {
                "role": "system",
                "content": f"You are a strict JSON extractor. Output ONLY valid JSON matching this schema: {schema_class.model_json_schema()}"
            },
            {"role": "user", "content": prompt}
        ]

        # Local servers (optiq/ollama) don't support structured outputs (BaseModel as response_format)
        # so skip that tier entirely and go straight to JSON mode / basic fallback.
        if not self._is_local:
            # Try structured outputs (cloud providers only)
            try:
                response = self._call_completions_with_lock(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                    response_format=schema_class,
                )
                return strip_thinking_tokens(response.choices[0].message.content)
            except Exception:
                pass

        # Try JSON mode
        try:
            response = self._call_completions_with_lock(
                model=self.model_name,
                messages=messages,
                max_tokens=resolved_max_tokens,
                temperature=temp,
                response_format={"type": "json_object"},
            )
            return strip_thinking_tokens(response.choices[0].message.content)
        except Exception:
            # Basic fallback (no response_format)
            response = self._call_completions_with_lock(
                model=self.model_name,
                messages=messages,
                max_tokens=resolved_max_tokens,
                temperature=temp,
            )
            return strip_thinking_tokens(response.choices[0].message.content)

    async def generate_json_async(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp

        messages = [
            {
                "role": "system",
                "content": f"You are a strict JSON extractor. Output ONLY valid JSON matching this schema: {schema_class.model_json_schema()}"
            },
            {"role": "user", "content": prompt}
        ]

        async def _call_with_format(response_format):
            max_retries = 3
            backoff = config.llm_retry_backoff
            last_err = None
            for attempt in range(max_retries):
                await self.rate_limiter.wait()
                try:
                    kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": resolved_max_tokens,
                        "temperature": temp,
                    }
                    if response_format is not None:
                        kwargs["response_format"] = response_format
                    response = await asyncio.to_thread(self._call_completions_with_lock, **kwargs)
                    return strip_thinking_tokens(response.choices[0].message.content)
                except Exception as e:
                    last_err = e
                    if attempt == max_retries - 1:
                        raise e
                    sleep_time = backoff * (2 ** attempt)
                    con.warning(f"LLM API JSON request failed: {e}. Retrying in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
            raise last_err

        # Local servers (optiq/ollama) don't support structured outputs (BaseModel as response_format)
        # so skip that tier entirely and go straight to JSON mode / basic fallback.
        if not self._is_local:
            # Try structured outputs (cloud providers only)
            try:
                return await _call_with_format(schema_class)
            except Exception:
                pass

        # Try JSON mode
        try:
            return await _call_with_format({"type": "json_object"})
        except Exception:
            # Basic fallback (no response_format)
            return await _call_with_format(None)

    def generate_response_with_logits(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temp: Optional[float] = None,
        task: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
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

        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": resolved_max_tokens,
            "temperature": temp,
            "logprobs": True,
            "top_logprobs": 5,
        }

        try:
            response = self._call_completions_with_lock(**kwargs)
        except Exception:
            try:
                kwargs.pop("top_logprobs", None)
                response = self._call_completions_with_lock(**kwargs)
            except Exception:
                kwargs.pop("logprobs", None)
                response = self._call_completions_with_lock(**kwargs)

        choice = response.choices[0]
        tokens_info = []
        full_text = ""
        if hasattr(choice, "logprobs") and choice.logprobs and getattr(choice.logprobs, "content", None):
            for item in choice.logprobs.content:
                t_text = item.token
                c_start = len(full_text)
                full_text += t_text
                c_end = len(full_text)
                lp = item.logprob
                top_lps = {}
                if getattr(item, "top_logprobs", None):
                    for top in item.top_logprobs:
                        top_lps[top.token] = top.logprob
                tokens_info.append({
                    "token_text": t_text,
                    "logprob": lp,
                    "top_logprobs": top_lps,
                    "char_start": c_start,
                    "char_end": c_end
                })
        else:
            full_text = choice.message.content or ""

        clean_text = strip_thinking_tokens(full_text)
        try:
            from core.shannon_estimator import align_tokens_info
            aligned_tokens = align_tokens_info(full_text, clean_text, tokens_info)
        except Exception:
            aligned_tokens = tokens_info

        return clean_text, aligned_tokens

