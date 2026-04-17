"""Tests for the Ollama client.

Tests token estimation, context window trimming, stats tracking,
and client initialization — all without a real Ollama server.
"""

from __future__ import annotations

import pytest

from src.core.ollama_client import (
    ClientStats,
    GenerationStats,
    OllamaClient,
    estimate_messages_tokens,
    estimate_tokens,
)


# ============================================================
# Token Estimation
# ============================================================


class TestTokenEstimation:
    """Tests for token estimation functions."""

    def test_estimate_tokens_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short_string(self) -> None:
        # "Hello" = 5 chars → 5 / 3.5 ≈ 1
        result = estimate_tokens("Hello")
        assert result == 1

    def test_estimate_tokens_longer_string(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        # 43 chars → 43 / 3.5 ≈ 12
        result = estimate_tokens(text)
        assert result == 12

    def test_estimate_tokens_scales_linearly(self) -> None:
        short = estimate_tokens("abc")
        long = estimate_tokens("abc" * 100)
        # Should be roughly 100x
        assert long >= short * 90  # Allow some rounding variance

    def test_estimate_messages_tokens_empty_list(self) -> None:
        result = estimate_messages_tokens([])
        # Just the start/end overhead (2)
        assert result == 2

    def test_estimate_messages_tokens_single(self) -> None:
        messages = [{"role": "user", "content": "Hello world"}]
        result = estimate_messages_tokens(messages)
        # 2 (base) + 4 (per-msg overhead) + estimate_tokens("Hello world")
        content_tokens = estimate_tokens("Hello world")
        assert result == 2 + 4 + content_tokens

    def test_estimate_messages_tokens_multiple(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = estimate_messages_tokens(messages)
        expected = 2  # base
        for msg in messages:
            expected += 4 + estimate_tokens(msg["content"])
        assert result == expected

    def test_estimate_messages_tokens_missing_content(self) -> None:
        messages = [{"role": "user"}]
        # Should not crash, treats missing content as ""
        result = estimate_messages_tokens(messages)
        assert result == 2 + 4 + 0  # base + overhead + 0 tokens


# ============================================================
# Context Window Trimming
# ============================================================


class TestContextTrimming:
    """Tests for context window management."""

    def setup_method(self) -> None:
        self.client = OllamaClient(
            base_url="http://localhost:11434",
            num_ctx=200,  # Small window for easy testing
            max_retries=0,
        )

    def test_check_context_fit_fits(self) -> None:
        messages = [{"role": "user", "content": "short"}]
        fits, input_tokens, available = self.client.check_context_fit(messages, max_completion=50)
        assert fits is True
        assert input_tokens > 0
        assert available > 0

    def test_check_context_fit_too_large(self) -> None:
        messages = [{"role": "user", "content": "x" * 2000}]
        fits, input_tokens, available = self.client.check_context_fit(messages, max_completion=50)
        assert fits is False

    def test_trim_messages_no_trim_needed(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = self.client.trim_messages_to_fit(messages, max_completion=50)
        assert result == messages

    def test_trim_messages_keeps_system_and_last(self) -> None:
        # Fill context beyond capacity
        messages = [
            {"role": "system", "content": "System prompt"},
        ]
        # Add many messages to exceed context
        for i in range(20):
            messages.append({"role": "user", "content": f"Message {i} " + "x" * 50})
            messages.append({"role": "assistant", "content": f"Response {i} " + "y" * 50})
        messages.append({"role": "user", "content": "Final question"})

        result = self.client.trim_messages_to_fit(messages, max_completion=50)

        # System prompt should be first
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System prompt"
        # Last message should be preserved
        assert result[-1]["content"] == "Final question"
        # Should be shorter than original
        assert len(result) < len(messages)

    def test_trim_messages_empty_conversation(self) -> None:
        messages = [{"role": "system", "content": "System prompt"}]
        result = self.client.trim_messages_to_fit(messages, max_completion=50)
        assert result == messages

    def test_trim_preserves_most_recent(self) -> None:
        """Trimming should keep recent messages, not old ones."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old message " + "x" * 100},
            {"role": "assistant", "content": "old response " + "y" * 100},
            {"role": "user", "content": "recent message"},
            {"role": "assistant", "content": "recent response"},
            {"role": "user", "content": "current question"},
        ]
        result = self.client.trim_messages_to_fit(messages, max_completion=50)
        # Current question must always be there
        assert result[-1]["content"] == "current question"
        # System must be there
        assert result[0]["content"] == "sys"


# ============================================================
# Client Stats
# ============================================================


class TestClientStats:
    """Tests for statistics tracking."""

    def test_record_increments_counters(self) -> None:
        stats = ClientStats()
        gen = GenerationStats(
            model="test-model",
            total_tokens=100,
            duration_seconds=2.5,
        )
        stats.record(gen)
        assert stats.total_requests == 1
        assert stats.total_tokens_used == 100
        assert stats.total_duration_seconds == 2.5
        assert stats.requests_by_model["test-model"] == 1

    def test_record_multiple_models(self) -> None:
        stats = ClientStats()
        stats.record(GenerationStats(model="model-a", total_tokens=50, duration_seconds=1.0))
        stats.record(GenerationStats(model="model-b", total_tokens=80, duration_seconds=2.0))
        stats.record(GenerationStats(model="model-a", total_tokens=60, duration_seconds=1.5))

        assert stats.total_requests == 3
        assert stats.total_tokens_used == 190
        assert stats.requests_by_model["model-a"] == 2
        assert stats.requests_by_model["model-b"] == 1

    def test_initial_stats_are_zero(self) -> None:
        stats = ClientStats()
        assert stats.total_requests == 0
        assert stats.total_failures == 0
        assert stats.total_retries == 0


# ============================================================
# Client Initialization
# ============================================================


class TestOllamaClientInit:
    """Tests for OllamaClient construction."""

    def test_defaults(self) -> None:
        client = OllamaClient()
        assert client.base_url == "http://ollama:11434"
        assert client.default_model == "qwen3.5:27b"
        assert client.num_ctx == 8192
        assert client.max_retries == 3

    def test_custom_params(self) -> None:
        client = OllamaClient(
            base_url="http://localhost:11434/",
            default_model="llama3:8b",
            timeout=60,
            num_ctx=4096,
            max_retries=5,
            retry_base_delay=1.0,
        )
        assert client.base_url == "http://localhost:11434"  # trailing slash stripped
        assert client.default_model == "llama3:8b"
        assert client.num_ctx == 4096
        assert client.max_retries == 5

    def test_get_stats_empty(self) -> None:
        client = OllamaClient()
        stats = client.get_stats()
        assert stats["total_requests"] == 0
        assert stats["total_failures"] == 0
        assert isinstance(stats["requests_by_model"], dict)
