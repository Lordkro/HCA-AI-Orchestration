"""Tests for the Ollama client.

Tests token estimation, context window trimming, stats tracking,
and client initialization — all without a real Ollama server.
"""

from __future__ import annotations

from hca.core.ollama_client import (
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
        short = estimate_tokens("abcde")  # 5 chars → 5 / 3.5 ≈ 1
        long = estimate_tokens("abcde" * 100)  # 500 chars → 500 / 3.5 ≈ 142
        assert long >= short * 90  # 142 >= 1 * 90 → meaningful check

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
        assert client.default_model == "qwen3:14b"
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
        assert stats["total_cost_estimate_usd"] == 0.0
        assert "cache" in stats
        assert isinstance(stats["requests_by_model"], dict)

    def test_get_stats_includes_cache(self) -> None:
        client = OllamaClient()
        cache = client.cache_stats
        assert cache["size"] == 0
        assert cache["hits"] == 0
        assert cache["misses"] == 0


# ============================================================
# LRU Cache Tests
# ============================================================


class TestLLMResponseCache:
    """Tests for the in-memory LLM response cache."""

    def test_cache_miss_returns_none(self) -> None:
        from hca.core.ollama_client import _LLMResponseCache

        cache = _LLMResponseCache(maxsize=4)
        key = cache._make_key("model", [{"role": "user", "content": "hi"}], 0.7, 0.9, 100)
        assert cache.get(key) is None

    def test_cache_hit_after_put(self) -> None:
        from hca.core.ollama_client import _LLMResponseCache

        cache = _LLMResponseCache(maxsize=4)
        key = cache._make_key("model", [{"role": "user", "content": "hi"}], 0.7, 0.9, 100)
        cache.put(key, "hello", [])
        result = cache.get(key)
        assert result is not None
        assert result[0] == "hello"
        assert result[1] == []

    def test_cache_evicts_lru(self) -> None:
        from hca.core.ollama_client import _LLMResponseCache

        cache = _LLMResponseCache(maxsize=2)
        keys = []
        for i in range(3):
            k = cache._make_key("m", [{"role": "user", "content": str(i)}], 0.7, 0.9, 100)
            cache.put(k, f"resp-{i}", [])
            keys.append(k)
        # First key should be evicted
        assert cache.get(keys[0]) is None
        # Second and third should still be there
        assert cache.get(keys[1]) is not None
        assert cache.get(keys[2]) is not None

    def test_cache_tracks_hit_rate(self) -> None:
        from hca.core.ollama_client import _LLMResponseCache

        cache = _LLMResponseCache(maxsize=4)
        key = cache._make_key("m", [{"role": "user", "content": "hi"}], 0.7, 0.9, 100)
        # Miss
        cache.get(key)
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 1
        # Hit
        cache.put(key, "hello", [])
        cache.get(key)
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1
        assert cache.stats["hit_rate"] == 0.5

    def test_cache_key_different_model(self) -> None:
        from hca.core.ollama_client import _LLMResponseCache

        cache = _LLMResponseCache(maxsize=4)
        k1 = cache._make_key("model-a", [{"role": "user", "content": "hi"}], 0.7, 0.9, 100)
        k2 = cache._make_key("model-b", [{"role": "user", "content": "hi"}], 0.7, 0.9, 100)
        cache.put(k1, "resp-a", [])
        # Different model → different key → miss
        assert cache.get(k2) is None
        assert cache.get(k1) is not None


# ============================================================
# Cost Estimation Tests
# ============================================================


class TestCostEstimation:
    """Tests for the cost estimation helper."""

    def test_estimate_cost_zero_tokens(self) -> None:
        from hca.core.ollama_client import estimate_cost

        assert estimate_cost("qwen3:14b", 0, 0) == 0.0

    def test_estimate_cost_known_model(self) -> None:
        from hca.core.ollama_client import estimate_cost

        # 1000 input + 500 output tokens on qwen3:14b
        # (1000/1000)*0.002 + (500/1000)*0.002 = 0.002 + 0.001 = 0.003
        cost = estimate_cost("qwen3:14b", 1000, 500)
        assert cost == 0.003

    def test_estimate_cost_unknown_model_default(self) -> None:
        from hca.core.ollama_client import estimate_cost

        cost = estimate_cost("unknown:latest", 1000, 500)
        # Uses default pricing: 0.001 per 1K for both
        assert cost == 0.0015

    def test_generation_stats_auto_cost(self) -> None:
        stats = GenerationStats(
            model="qwen3:14b",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )
        assert stats.cost_estimate_usd == 0.003

    def test_generation_stats_zero_cost(self) -> None:
        stats = GenerationStats(
            model="qwen3:14b",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_estimate_usd=0.0,
        )
        assert stats.cost_estimate_usd == 0.0

    def test_client_stats_tracks_cost(self) -> None:
        stats = ClientStats()
        stats.record(GenerationStats(
            model="qwen3:14b",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        ))
        stats.record(GenerationStats(
            model="qwen3:8b",
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
        ))
        assert stats.total_cost_estimate_usd > 0
        assert stats.total_requests == 2


# ============================================================
# Token Estimation (improved heuristic) Tests
# ============================================================


class TestImprovedTokenEstimation:
    """Tests for the code-density-aware token estimator."""

    def test_code_denser_than_prose(self) -> None:
        # Code has more symbols → denser → more tokens per char
        assert estimate_tokens("x" * 100) <= estimate_tokens(
            "{" * 50 + "}" * 50
        )
