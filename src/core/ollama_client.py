"""Ollama API client for LLM inference."""

from __future__ import annotations

from typing import AsyncIterator

import httpx
import structlog

logger = structlog.get_logger()


class OllamaClient:
    """Async client for the Ollama REST API."""

    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        default_model: str = "qwen2.5:32b",
        timeout: int = 120,
        num_ctx: int = 8192,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

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

        logger.debug("ollama_generate", model=model, prompt_len=len(prompt))

        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        text = data.get("response", "")
        logger.debug(
            "ollama_response",
            model=model,
            response_len=len(text),
            eval_count=data.get("eval_count"),
            eval_duration_ms=data.get("eval_duration", 0) / 1_000_000,
        )
        return text

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Chat completion using Ollama's /api/chat endpoint."""
        model = model or self.default_model
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

        logger.debug("ollama_chat", model=model, message_count=len(messages))

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        text = data.get("message", {}).get("content", "")
        logger.debug(
            "ollama_chat_response",
            model=model,
            response_len=len(text),
            eval_count=data.get("eval_count"),
        )
        return text

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Streaming chat completion."""
        model = model or self.default_model
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

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token

    async def list_models(self) -> list[dict]:
        """List all available models on the Ollama server."""
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
