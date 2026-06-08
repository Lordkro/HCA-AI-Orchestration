"""Ollama API client for LLM inference.

Provides a robust async client with:
- Retry logic with exponential backoff
- Token/context window estimation and management
- LLM response caching (in-memory LRU)
- Streaming and non-streaming chat
- Model preloading and validation
- Performance metrics tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

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

# Rough heuristic: ~4 characters per token for English text,
# ~3 for code/symbol-dense text.  Conservative to avoid overflow.
CHARS_PER_TOKEN_ESTIMATE = 3.5

# Characters that signal code-heavy content (denser tokenisation)
_CODE_HEAVY_CHARS = set("{}[]();:<>!=+-*/&|^~#@\"'\\`")


def estimate_tokens(text: str) -> int:
    """Estimate the token count for a string.

    Uses a slightly better heuristic that accounts for code density.
    Conservative by design to avoid overflowing the context window.
    """
    if not text:
        return 0
    ratio = sum(1 for ch in text if ch in _CODE_HEAVY_CHARS) / max(len(text), 1)
    # Code-heavy content: ~3 chars/token; prose: ~4 chars/token
    effective_cpt = CHARS_PER_TOKEN_ESTIMATE - ratio
    return max(1, int(len(text) / max(effective_cpt, 2.0)))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate total tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4
        total += estimate_tokens(msg.get("content", ""))
    total += 2
    return total


# ============================================================
# LRU Cache for LLM Responses
# ============================================================


class _LLMResponseCache:
    """Size-bounded in-memory cache for LLM responses.

    Keys are SHA-256 hashes of (model, messages_json, temperature, top_p).  This
    catches repeated identical prompts within a short window — common during
    tool-call retry loops and parallel agent runs.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._cache: OrderedDict[str, tuple[str | None, list[dict] | None]] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        raw = json.dumps(
            [model, messages, temperature, top_p, max_tokens],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> tuple[str | None, list[dict] | None] | None:
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            self._cache.move_to_end(key)
            return result
        self._misses += 1
        return None

    def put(
        self,
        key: str,
        text: str | None,
        tool_calls: list[dict] | None,
    ) -> None:
        self._cache[key] = (text, tool_calls)
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0,
        }


# ============================================================
# Cost Estimation
# ============================================================

# Approximate per-1K-token costs (USD) for popular local models.
# These are rough guidelines — actual costs depend on hardware and energy.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "qwen3:14b": {"input_per_1k": 0.002, "output_per_1k": 0.002},
    "qwen2.5-coder:14b": {"input_per_1k": 0.002, "output_per_1k": 0.002},
    "qwen3:8b": {"input_per_1k": 0.001, "output_per_1k": 0.001},
    "qwen2.5-coder:7b": {"input_per_1k": 0.001, "output_per_1k": 0.001},
    "llama3.2:3b": {"input_per_1k": 0.0005, "output_per_1k": 0.0005},
    "qwen2.5-coder:3b": {"input_per_1k": 0.0005, "output_per_1k": 0.0005},
    "phi-4:latest": {"input_per_1k": 0.001, "output_per_1k": 0.001},
    "llama3.2:1b": {"input_per_1k": 0.0003, "output_per_1k": 0.0003},
}
_DEFAULT_PRICING = {"input_per_1k": 0.001, "output_per_1k": 0.001}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of an LLM call."""
    pricing = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
    return round(input_cost + output_cost, 6)


# ============================================================
# Ollama Client
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
    cost_estimate_usd: float = 0.0

    def __post_init__(self) -> None:
        if not self.cost_estimate_usd and self.total_tokens > 0:
            self.cost_estimate_usd = estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)


@dataclass
class ClientStats:
    """Cumulative statistics for the Ollama client."""

    total_requests: int = 0
    total_failures: int = 0
    total_retries: int = 0
    total_tokens_used: int = 0
    total_duration_seconds: float = 0.0
    total_cost_estimate_usd: float = 0.0
    requests_by_model: dict[str, int] = field(default_factory=dict)

    def record(self, stats: GenerationStats) -> None:
        """Record stats from a generation call."""
        self.total_requests += 1
        self.total_tokens_used += stats.total_tokens
        self.total_duration_seconds += stats.duration_seconds
        self.total_cost_estimate_usd += stats.cost_estimate_usd
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
    - In-memory LLM response cache (LRU, configurable size)
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
        cache_maxsize: int = 256,
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
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_requests = 0
        self._cb_state = 0  # 0=closed, 1=half-open, 2=open
        self._cb_failure_count = 0
        self._cb_last_failure_time = 0.0
        self._cb_tripped_count = 0
        self._cache = _LLMResponseCache(maxsize=cache_maxsize)
        self._last_stats: GenerationStats | None = None

    @property
    def last_stats(self) -> GenerationStats | None:
        """GenerationStats from the most recent chat call."""
        return self._last_stats

    @property
    def cache_stats(self) -> dict[str, Any]:
        """LLM response cache statistics."""
        return self._cache.stats

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

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        auto_trim: bool = True,
        use_cache: bool = True,
    ) -> str:
        """Chat completion using Ollama's /api/chat endpoint.

        Args:
            messages: List of chat messages (role + content).
            model: Model to use (defaults to self.default_model).
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Max tokens to generate.
            auto_trim: If True, automatically trim messages to fit context window.
            use_cache: If True, check the in-memory response cache before calling the LLM.
        """
        model = model or self.default_model
        self._cb_check()

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

        # LLM response cache: skip the HTTP call for identical prompts
        if use_cache:
            cache_key = self._cache._make_key(model, messages, temperature, top_p, max_tokens)
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached_text, _ = cached
                logger.info(
                    "ollama_chat_cache_hit",
                    model=model,
                    message_count=len(messages),
                )
                self._last_stats = GenerationStats(
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    duration_seconds=0,
                    tokens_per_second=0,
                    cost_estimate_usd=0,
                )
                return cached_text or ""

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
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
        self._last_stats = stats
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

        if use_cache:
            self._cache.put(cache_key, text, None)

        return text

    async def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        auto_trim: bool = True,
        use_cache: bool = True,
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
            top_p: Nucleus sampling parameter.
            max_tokens: Max tokens to generate.
            auto_trim: If True, trim messages to fit context window.
            use_cache: If True, check the in-memory response cache before calling the LLM.
        """
        model = model or self.default_model
        self._cb_check()

        if auto_trim:
            messages = self.trim_messages_to_fit(messages, max_completion=max_tokens)

        # LLM response cache: skip the HTTP call for identical prompts
        if use_cache:
            cache_key = self._cache._make_key(model, messages, temperature, top_p, max_tokens)
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached_text, cached_tool_calls = cached
                logger.info(
                    "ollama_chat_with_tools_cache_hit",
                    model=model,
                    message_count=len(messages),
                )
                self._last_stats = GenerationStats(
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    duration_seconds=0,
                    tokens_per_second=0,
                    cost_estimate_usd=0,
                )
                return (cached_text or "", cached_tool_calls or [])

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": tools,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
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
        self._last_stats = stats
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

        if use_cache:
            self._cache.put(cache_key, text, tool_calls)

        return text, tool_calls

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
            "total_cost_estimate_usd": round(self.stats.total_cost_estimate_usd, 4),
            "avg_tokens_per_second": round(
                self.stats.total_tokens_used / self.stats.total_duration_seconds, 1
            )
            if self.stats.total_duration_seconds > 0
            else 0,
            "requests_by_model": self.stats.requests_by_model,
            "cache": self._cache.stats,
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        ollama_up.set(0)
        await self._client.aclose()
        logger.info("ollama_client_closed", stats=self.get_stats())
