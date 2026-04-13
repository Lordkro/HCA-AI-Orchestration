"""Ollama API client for LLM inference.

Provides a robust async client with:
- Retry logic with exponential backoff
- Token/context window estimation and management
- Streaming and non-streaming chat
- Model preloading and validation
- Performance metrics tracking
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
import structlog

logger = structlog.get_logger()


# ============================================================
# Token Estimation
# ============================================================

# Rough heuristic: ~4 characters per token for English text.
# This is intentionally conservative to avoid overflowing context.
CHARS_PER_TOKEN_ESTIMATE = 3.5


def estimate_tokens(text: str) -> int:
    """Estimate the token count for a string (conservative)."""
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate total tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        # Each message has overhead (~4 tokens for role + formatting)
        total += 4
        total += estimate_tokens(msg.get("content", ""))
    total += 2  # Start/end tokens
    return total


# ============================================================
# Data Classes
# ============================================================


@dataclass
class GenerationStats:
    """Statistics from a single LLM generation call."""
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class ClientStats:
    """Cumulative statistics for the Ollama client."""
    total_requests: int = 0
    total_failures: int = 0
    total_retries: int = 0
    total_tokens_used: int = 0
    total_duration_seconds: float = 0.0
    requests_by_model: dict[str, int] = field(default_factory=dict)

    def record(self, stats: GenerationStats) -> None:
        """Record stats from a generation call."""
        self.total_requests += 1
        self.total_tokens_used += stats.total_tokens
        self.total_duration_seconds += stats.duration_seconds
        model_count = self.requests_by_model.get(stats.model, 0)
        self.requests_by_model[stats.model] = model_count + 1


# ============================================================
# Ollama Client
# ============================================================


class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when Ollama server is unreachable."""


class OllamaModelError(OllamaError):
    """Raised when a requested model is not available."""


class OllamaTimeoutError(OllamaError):
    """Raised when a request times out after all retries."""


class OllamaContextOverflowError(OllamaError):
    """Raised when the input exceeds the context window."""


class OllamaClient:
    """Async client for the Ollama REST API.

    Features:
    - Automatic retries with exponential backoff
    - Token estimation and context window management
    - Streaming and non-streaming chat completions
    - Generation statistics tracking
    - Model availability validation
    """

    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        default_model: str = "qwen2.5:32b",
        timeout: int = 120,
        num_ctx: int = 8192,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.stats = ClientStats()
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=15.0),
        )
        self._available_models: list[str] = []

    # --------------------------------------------------------
    # Retry Logic
    # --------------------------------------------------------

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_data: dict | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        """Make an HTTP request with exponential backoff retry."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    # For streaming, we return the response directly
                    # Caller is responsible for the context manager
                    response = await self._client.request(
                        method, path, json=json_data
                    )
                else:
                    response = await self._client.request(
                        method, path, json=json_data
                    )

                # Check for model not found errors
                if response.status_code == 404:
                    model = (json_data or {}).get("model", "unknown")
                    raise OllamaModelError(
                        f"Model '{model}' not found. Pull it first with: "
                        f"ollama pull {model}"
                    )

                response.raise_for_status()
                return response

            except httpx.ConnectError as e:
                last_error = OllamaConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Is the Ollama server running? Error: {e}"
                )
            except httpx.TimeoutException as e:
                last_error = OllamaTimeoutError(
                    f"Request timed out after {self.timeout}s (attempt {attempt + 1}). "
                    f"The model may be loading or the generation is too slow. Error: {e}"
                )
            except OllamaModelError:
                raise  # Don't retry model-not-found errors
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_error = OllamaError(f"Ollama server error: {e}")
                else:
                    raise OllamaError(f"Ollama request failed: {e}") from e
            except httpx.HTTPError as e:
                last_error = OllamaError(f"HTTP error: {e}")

            # Exponential backoff
            if attempt < self.max_retries:
                delay = self.retry_base_delay * (2 ** attempt)
                self.stats.total_retries += 1
                logger.warning(
                    "ollama_retry",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    delay_seconds=delay,
                    error=str(last_error),
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        self.stats.total_failures += 1
        raise last_error or OllamaError("Request failed after all retries")

    # --------------------------------------------------------
    # Context Window Management
    # --------------------------------------------------------

    def check_context_fit(
        self, messages: list[dict[str, str]], max_completion: int = 4096
    ) -> tuple[bool, int, int]:
        """Check if messages fit within the context window.

        Returns:
            (fits, estimated_input_tokens, available_for_completion)
        """
        input_tokens = estimate_messages_tokens(messages)
        available = self.num_ctx - input_tokens
        fits = available >= max_completion
        return fits, input_tokens, max(0, available)

    def trim_messages_to_fit(
        self,
        messages: list[dict[str, str]],
        max_completion: int = 4096,
        keep_system: bool = True,
    ) -> list[dict[str, str]]:
        """Trim conversation history to fit within the context window.

        Strategy: Keep the system message and the most recent messages,
        dropping older messages from the middle.
        """
        fits, input_tokens, available = self.check_context_fit(messages, max_completion)
        if fits:
            return messages

        logger.warning(
            "trimming_context",
            input_tokens=input_tokens,
            num_ctx=self.num_ctx,
            max_completion=max_completion,
            message_count=len(messages),
        )

        # Separate system messages from conversation
        system_msgs = []
        conversation = []
        for msg in messages:
            if msg.get("role") == "system" and keep_system:
                system_msgs.append(msg)
            else:
                conversation.append(msg)

        # Always keep the last message (current prompt) and system
        if not conversation:
            return messages  # Nothing to trim

        # Calculate how much space system + last message take
        reserved_tokens = estimate_messages_tokens(system_msgs + [conversation[-1]])
        budget = self.num_ctx - max_completion - reserved_tokens

        # Add messages from most recent backwards until budget exhausted
        trimmed_conversation = []
        running_tokens = 0
        for msg in reversed(conversation[:-1]):
            msg_tokens = estimate_tokens(msg.get("content", "")) + 4
            if running_tokens + msg_tokens > budget:
                break
            trimmed_conversation.insert(0, msg)
            running_tokens += msg_tokens

        result = system_msgs + trimmed_conversation + [conversation[-1]]
        logger.info(
            "context_trimmed",
            original_messages=len(messages),
            trimmed_messages=len(result),
            estimated_tokens=estimate_messages_tokens(result),
        )
        return result

    # --------------------------------------------------------
    # Core API Methods
    # --------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a completion from Ollama (non-streaming)."""
        model = model or self.default_model
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if system:
            payload["system"] = system

        prompt_tokens = estimate_tokens(prompt)
        if system:
            prompt_tokens += estimate_tokens(system)

        logger.debug(
            "ollama_generate",
            model=model,
            prompt_len=len(prompt),
            estimated_tokens=prompt_tokens,
        )

        start = time.monotonic()
        response = await self._request_with_retry("POST", "/api/generate", json_data=payload)
        elapsed = time.monotonic() - start
        data = response.json()

        text = data.get("response", "")

        # Extract real token counts if available, else estimate
        eval_count = data.get("eval_count", estimate_tokens(text))
        prompt_eval_count = data.get("prompt_eval_count", prompt_tokens)

        stats = GenerationStats(
            model=model,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
            duration_seconds=elapsed,
            tokens_per_second=eval_count / elapsed if elapsed > 0 else 0,
        )
        self.stats.record(stats)

        logger.info(
            "ollama_generate_complete",
            model=model,
            response_len=len(text),
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            duration_s=round(elapsed, 2),
            tok_per_s=round(stats.tokens_per_second, 1),
        )
        return text

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        auto_trim: bool = True,
    ) -> str:
        """Chat completion using Ollama's /api/chat endpoint.

        Args:
            messages: List of chat messages (role + content).
            model: Model to use (defaults to self.default_model).
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            auto_trim: If True, automatically trim messages to fit context window.
        """
        model = model or self.default_model

        # Context window management
        if auto_trim:
            messages = self.trim_messages_to_fit(messages, max_completion=max_tokens)
        else:
            fits, input_tokens, available = self.check_context_fit(messages, max_tokens)
            if not fits:
                raise OllamaContextOverflowError(
                    f"Input ({input_tokens} tokens) + completion ({max_tokens} tokens) "
                    f"exceeds context window ({self.num_ctx} tokens). "
                    f"Only {available} tokens available for completion."
                )

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }

        input_tokens_est = estimate_messages_tokens(messages)
        logger.debug(
            "ollama_chat",
            model=model,
            message_count=len(messages),
            estimated_input_tokens=input_tokens_est,
        )

        start = time.monotonic()
        response = await self._request_with_retry("POST", "/api/chat", json_data=payload)
        elapsed = time.monotonic() - start
        data = response.json()

        text = data.get("message", {}).get("content", "")

        eval_count = data.get("eval_count", estimate_tokens(text))
        prompt_eval_count = data.get("prompt_eval_count", input_tokens_est)

        stats = GenerationStats(
            model=model,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
            duration_seconds=elapsed,
            tokens_per_second=eval_count / elapsed if elapsed > 0 else 0,
        )
        self.stats.record(stats)

        logger.info(
            "ollama_chat_complete",
            model=model,
            response_len=len(text),
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            duration_s=round(elapsed, 2),
            tok_per_s=round(stats.tokens_per_second, 1),
        )
        return text

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        auto_trim: bool = True,
    ) -> AsyncIterator[str]:
        """Streaming chat completion.

        Yields tokens as they are generated. Also tracks stats.
        """
        model = model or self.default_model

        if auto_trim:
            messages = self.trim_messages_to_fit(messages, max_completion=max_tokens)

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }

        logger.debug("ollama_chat_stream_start", model=model, message_count=len(messages))

        start = time.monotonic()
        token_count = 0

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("message", {}).get("content", "")
                    if token:
                        token_count += 1
                        yield token
                    # Check if this is the final message with stats
                    if data.get("done", False):
                        elapsed = time.monotonic() - start
                        eval_count = data.get("eval_count", token_count)
                        stats = GenerationStats(
                            model=model,
                            prompt_tokens=data.get("prompt_eval_count", 0),
                            completion_tokens=eval_count,
                            total_tokens=data.get("prompt_eval_count", 0) + eval_count,
                            duration_seconds=elapsed,
                            tokens_per_second=eval_count / elapsed if elapsed > 0 else 0,
                        )
                        self.stats.record(stats)
                        logger.info(
                            "ollama_chat_stream_complete",
                            model=model,
                            tokens=eval_count,
                            duration_s=round(elapsed, 2),
                            tok_per_s=round(stats.tokens_per_second, 1),
                        )

    # --------------------------------------------------------
    # Model Management
    # --------------------------------------------------------

    async def list_models(self) -> list[dict]:
        """List all available models on the Ollama server."""
        response = await self._request_with_retry("GET", "/api/tags")
        models = response.json().get("models", [])
        self._available_models = [m.get("name", "") for m in models]
        return models

    async def ensure_model_available(self, model: str | None = None) -> bool:
        """Check if a model is available, refresh cache if needed."""
        model = model or self.default_model
        if not self._available_models:
            await self.list_models()

        if model not in self._available_models:
            # Refresh in case model was just pulled
            await self.list_models()

        available = model in self._available_models
        if not available:
            logger.error(
                "model_not_available",
                model=model,
                available=self._available_models,
            )
        return available

    async def get_model_info(self, model: str | None = None) -> dict:
        """Get detailed information about a model."""
        model = model or self.default_model
        try:
            response = await self._request_with_retry(
                "POST", "/api/show", json_data={"name": model}
            )
            return response.json()
        except OllamaError:
            return {}

    async def preload_model(self, model: str | None = None) -> bool:
        """Preload a model into memory for faster first inference.

        Sends a minimal request to force Ollama to load the model weights.
        """
        model = model or self.default_model
        logger.info("preloading_model", model=model)
        try:
            await self._request_with_retry(
                "POST",
                "/api/generate",
                json_data={
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
            logger.info("model_preloaded", model=model)
            return True
        except OllamaError as e:
            logger.error("model_preload_failed", model=model, error=str(e))
            return False

    # --------------------------------------------------------
    # Health & Diagnostics
    # --------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def get_stats(self) -> dict:
        """Get cumulative client statistics."""
        return {
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "total_retries": self.stats.total_retries,
            "total_tokens_used": self.stats.total_tokens_used,
            "total_duration_seconds": round(self.stats.total_duration_seconds, 2),
            "avg_tokens_per_second": round(
                self.stats.total_tokens_used / self.stats.total_duration_seconds, 1
            ) if self.stats.total_duration_seconds > 0 else 0,
            "requests_by_model": self.stats.requests_by_model,
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
        logger.info("ollama_client_closed", stats=self.get_stats())
