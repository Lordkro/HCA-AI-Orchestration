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
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
import structlog

from hca.core.metrics import (
    ollama_circuit_breaker_state,
    ollama_circuit_breaker_tripped_total,
    ollama_concurrent_requests,
    ollama_up,
    record_ollama_request,
)

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


class OllamaCircuitBreakerOpenError(OllamaError):
    """Raised when the circuit breaker is open (Ollama deemed unavailable)."""


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
        default_model: str = "qwen3:14b",
        timeout: int = 120,
        num_ctx: int = 8192,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        max_concurrent: int = 1,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.max_concurrent = max_concurrent
        self.circuit_breaker_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_breaker_recovery_timeout = circuit_breaker_recovery_timeout
        self.stats = ClientStats()
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=15.0),
        )
        self._available_models: list[str] = []
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_requests = 0
        self._cb_state = 0  # 0=closed, 1=half-open, 2=open
        self._cb_failure_count = 0
        self._cb_last_failure_time = 0.0
        self._cb_tripped_count = 0

    async def _acquire_concurrency_slot(self) -> None:
        """Acquire a concurrency slot, blocking if at capacity."""
        await self._semaphore.acquire()
        self._active_requests += 1
        ollama_concurrent_requests.set(self._active_requests)

    def _release_concurrency_slot(self) -> None:
        """Release a concurrency slot."""
        self._active_requests -= 1
        ollama_concurrent_requests.set(self._active_requests)
        self._semaphore.release()

    # --------------------------------------------------------
    # Circuit Breaker
    # --------------------------------------------------------

    def _cb_check(self) -> None:
        """Check circuit breaker state; raises if open and not yet recoverable."""
        now = time.monotonic()
        if self._cb_state == 2:  # Open
            if now - self._cb_last_failure_time >= self.circuit_breaker_recovery_timeout:
                self._cb_state = 1  # Half-open — allow probe
                ollama_circuit_breaker_state.set(1)
                logger.info("circuit_breaker_half_open")
            else:
                raise OllamaCircuitBreakerOpenError(
                    f"Circuit breaker open for "
                    f"{now - self._cb_last_failure_time:.0f}s "
                    f"(recovery in {self.circuit_breaker_recovery_timeout}s)"
                )

    def _cb_record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._cb_failure_count += 1
        self._cb_last_failure_time = time.monotonic()
        if self._cb_failure_count >= self.circuit_breaker_failure_threshold:
            self._cb_state = 2  # Open
            self._cb_tripped_count += 1
            ollama_circuit_breaker_state.set(2)
            ollama_circuit_breaker_tripped_total.inc()
            ollama_up.set(0)
            logger.error(
                "circuit_breaker_opened",
                failure_count=self._cb_failure_count,
                recovery_timeout_s=self.circuit_breaker_recovery_timeout,
            )

    def _cb_record_success(self) -> None:
        """Record a success; closes the circuit if it was open."""
        if self._cb_state != 0:
            self._cb_state = 0
            self._cb_failure_count = 0
            ollama_circuit_breaker_state.set(0)
            ollama_up.set(1)
            logger.info("circuit_breaker_closed")
        self._cb_failure_count = 0

    # --------------------------------------------------------
    # Retry Logic
    # --------------------------------------------------------

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_data: dict | None = None,
    ) -> httpx.Response:
        """Make an HTTP request with exponential backoff retry."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(method, path, json=json_data)

                # Check for model not found errors
                if response.status_code == 404:
                    model = (json_data or {}).get("model", "unknown")
                    raise OllamaModelError(
                        f"Model '{model}' not found. Pull it first with: ollama pull {model}"
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
                delay = self.retry_base_delay * (2**attempt)
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
        self._cb_record_failure()
        model = (json_data or {}).get("model", "unknown")
        record_ollama_request(model, "error", 0, 0, 0)
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
    # Streaming Collector (used by chat & generate internally)
    # --------------------------------------------------------

    async def _stream_collect(
        self,
        path: str,
        payload: dict,
        *,
        content_key: str = "response",
    ) -> tuple[str, dict]:
        """Stream a request and collect all tokens into a single string.

        Uses streaming so the httpx timeout applies per-chunk rather than
        to the total generation time.  This is critical for thinking models
        like qwen3 that produce long internal chains before any output.

        Args:
            path: API path (e.g. "/api/chat" or "/api/generate").
            payload: JSON payload (must have "stream": True).
            content_key: Dot-separated key path to extract text from each chunk.
                         "response" for /api/generate, "message.content" for /api/chat.

        Returns:
            (collected_text, final_chunk_data) — the final chunk usually
            carries eval_count / prompt_eval_count stats.
        """
        last_error: Exception | None = None
        keys = content_key.split(".")
        model = payload.get("model", "unknown")

        for attempt in range(self.max_retries + 1):
            try:
                chunks: list[str] = []
                final_data: dict = {}
                logger.info(
                    "stream_collect_request",
                    path=path,
                    model=payload.get("model"),
                    attempt=attempt + 1,
                )
                async with self._client.stream("POST", path, json=payload) as response:
                    if response.status_code == 404:
                        model = payload.get("model", "unknown")
                        raise OllamaModelError(
                            f"Model '{model}' not found. Pull it first with: ollama pull {model}"
                        )
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Navigate the key path to extract text
                        value = data
                        for k in keys:
                            if isinstance(value, dict):
                                value = value.get(k, "")
                            else:
                                value = ""
                                break
                        if value:
                            chunks.append(str(value))
                        # The last chunk (done=true) carries stats
                        if data.get("done"):
                            final_data = data
                return "".join(chunks), final_data

            except httpx.ConnectError as e:
                last_error = OllamaConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Is the Ollama server running? Error: {e}"
                )
            except httpx.TimeoutException as e:
                last_error = OllamaTimeoutError(
                    f"Stream timed out (attempt {attempt + 1}). Error: {e}"
                )
            except OllamaModelError:
                raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_error = OllamaError(f"Ollama server error: {e}")
                else:
                    raise OllamaError(f"Ollama request failed: {e}") from e
            except httpx.HTTPError as e:
                last_error = OllamaError(f"HTTP error: {e}")

            if attempt < self.max_retries:
                delay = self.retry_base_delay * (2**attempt)
                self.stats.total_retries += 1
                logger.warning(
                    "ollama_stream_retry",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    delay_seconds=delay,
                    error=str(last_error),
                )
                await asyncio.sleep(delay)

        self.stats.total_failures += 1
        self._cb_record_failure()
        record_ollama_request(model, "error", 0, 0, 0)
        raise last_error or OllamaError("Stream request failed after all retries")

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
        """Generate a completion from Ollama using streaming internally.

        Uses streaming to avoid total-time timeouts with thinking models
        like qwen3, while still returning the complete text.
        """
        model = model or self.default_model
        self._cb_check()
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "num_gpu": 20,
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

        await self._acquire_concurrency_slot()
        try:
            start = time.monotonic()
            text, final_data = await self._stream_collect(
                "/api/generate", payload, content_key="response"
            )
        finally:
            self._release_concurrency_slot()
        elapsed = time.monotonic() - start

        # Strip qwen3 <think>...</think> blocks from output
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()

        # Extract real token counts if available, else estimate
        eval_count = final_data.get("eval_count", estimate_tokens(text))
        prompt_eval_count = final_data.get("prompt_eval_count", prompt_tokens)

        stats = GenerationStats(
            model=model,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
            duration_seconds=elapsed,
            tokens_per_second=eval_count / elapsed if elapsed > 0 else 0,
        )
        self.stats.record(stats)
        record_ollama_request(model, "ok", elapsed, prompt_eval_count, eval_count)

        logger.info(
            "ollama_generate_complete",
            model=model,
            response_len=len(text),
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            duration_s=round(elapsed, 2),
            tok_per_s=round(stats.tokens_per_second, 1),
        )
        self._cb_record_success()
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
        self._cb_check()

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
            "stream": True,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "num_gpu": 20,
            },
        }

        input_tokens_est = estimate_messages_tokens(messages)
        logger.info(
            "ollama_chat_start",
            model=model,
            message_count=len(messages),
            estimated_input_tokens=input_tokens_est,
        )

        await self._acquire_concurrency_slot()
        try:
            start = time.monotonic()
            text, final_data = await self._stream_collect(
                "/api/chat", payload, content_key="message.content"
            )
        finally:
            self._release_concurrency_slot()
        elapsed = time.monotonic() - start

        # Strip qwen3 <think>...</think> blocks from output
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()

        eval_count = final_data.get("eval_count", estimate_tokens(text))
        prompt_eval_count = final_data.get("prompt_eval_count", input_tokens_est)

        stats = GenerationStats(
            model=model,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
            duration_seconds=elapsed,
            tokens_per_second=eval_count / elapsed if elapsed > 0 else 0,
        )
        self.stats.record(stats)
        record_ollama_request(model, "ok", elapsed, prompt_eval_count, eval_count)

        logger.info(
            "ollama_chat_complete",
            model=model,
            response_len=len(text),
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            duration_s=round(elapsed, 2),
            tok_per_s=round(stats.tokens_per_second, 1),
        )
        self._cb_record_success()
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
        self._cb_check()

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

        await self._acquire_concurrency_slot()
        try:
            start = time.monotonic()
            token_count = 0

            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    async with self._client.stream("POST", "/api/chat", json=payload) as response:
                        if response.status_code == 404:
                            raise OllamaModelError(
                                f"Model '{model}' not found. Pull it first with: ollama pull {model}"
                            )
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
                                    record_ollama_request(
                                        model, "ok", elapsed,
                                        stats.prompt_tokens, stats.completion_tokens,
                                    )
                                    logger.info(
                                        "ollama_chat_stream_complete",
                                        model=model,
                                        tokens=eval_count,
                                        duration_s=round(elapsed, 2),
                                        tok_per_s=round(stats.tokens_per_second, 1),
                                    )
                    self._cb_record_success()
                    return  # Success — exit retry loop

                except OllamaModelError:
                    raise
                except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                    last_error = e
                    if attempt < self.max_retries:
                        delay = self.retry_base_delay * (2**attempt)
                        self.stats.total_retries += 1
                        logger.warning(
                            "ollama_stream_retry",
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            delay_seconds=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)

            self.stats.total_failures += 1
            record_ollama_request(model, "error", 0, 0, 0)
            self._cb_record_failure()
            raise OllamaError(f"Streaming request failed after all retries: {last_error}")
        finally:
            self._release_concurrency_slot()
    # --------------------------------------------------------

    async def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        auto_trim: bool = True,
    ) -> tuple[str, list[dict]]:
        """Chat completion with tool/function calling support.

        Uses non-streaming mode (tools require ``stream: false``).
        Returns (text_response, tool_calls) where tool_calls is a list
        of ``{"name": str, "arguments": dict}`` dicts.

        Args:
            messages: Chat messages (role + content).
            tools: Tool definitions in Ollama/OpenAI function-calling format.
            model: Model override.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            auto_trim: If True, trim messages to fit context window.
        """
        model = model or self.default_model
        self._cb_check()

        if auto_trim:
            messages = self.trim_messages_to_fit(messages, max_completion=max_tokens)

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": tools,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }

        input_tokens_est = estimate_messages_tokens(messages)
        logger.info(
            "ollama_chat_with_tools_start",
            model=model,
            message_count=len(messages),
            tool_count=len(tools),
            estimated_input_tokens=input_tokens_est,
        )

        await self._acquire_concurrency_slot()
        try:
            start = time.monotonic()
            response = await self._request_with_retry("POST", "/api/chat", json_data=payload)
        finally:
            self._release_concurrency_slot()
        elapsed = time.monotonic() - start
        data = response.json()

        message = data.get("message", {})
        text = message.get("content", "").strip()

        # Strip think blocks from text portion
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()

        # Extract tool calls
        raw_calls = message.get("tool_calls", [])
        tool_calls: list[dict] = []
        for call in raw_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            raw_args = func.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    raw_args = {}
            tool_calls.append({"name": name, "arguments": raw_args})

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
        record_ollama_request(model, "ok", elapsed, prompt_eval_count, eval_count)
        self._cb_record_success()

        logger.info(
            "ollama_chat_with_tools_complete",
            model=model,
            response_len=len(text),
            tool_call_count=len(tool_calls),
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            duration_s=round(elapsed, 2),
        )
        return text, tool_calls

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

        Sends a minimal keep_alive request to force Ollama to load the model
        weights without generating any tokens.  Uses a short timeout and no
        retries so it doesn't block actual agent requests.
        """
        model = model or self.default_model
        logger.info("preloading_model", model=model)
        try:
            await self._client.post(
                "/api/generate",
                json={
                    "model": model,
                    "keep_alive": "5m",
                },
                timeout=30.0,
            )
            logger.info("model_preloaded", model=model)
            return True
        except Exception as e:
            logger.warning("model_preload_skipped", model=model, error=str(e))
            return False

    # --------------------------------------------------------
    # Health & Diagnostics
    # --------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = await self._client.get("/api/tags")
            ok = response.status_code == 200
            ollama_up.set(1 if ok else 0)
            return ok
        except httpx.HTTPError:
            ollama_up.set(0)
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
            )
            if self.stats.total_duration_seconds > 0
            else 0,
            "requests_by_model": self.stats.requests_by_model,
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        ollama_up.set(0)
        await self._client.aclose()
        logger.info("ollama_client_closed", stats=self.get_stats())
