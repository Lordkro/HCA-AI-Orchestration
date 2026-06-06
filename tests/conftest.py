"""Shared test fixtures for HCA Orchestration test suite.

All tests run fully offline — no Ollama server, no Redis required.
- Database: real SQLite via temp files
- Ollama: mock httpx transport
- MessageBus: mock (AsyncMock)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from hca.core.database import Database
from hca.core.models import (
    AgentMessage,
    AgentRole,
    MessagePayload,
    MessageType,
    Priority,
)

# ============================================================
# Database Fixture (real SQLite in temp dir)
# ============================================================


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Provide an initialized Database backed by a temp file."""
    db_path = str(tmp_path / "test.db")
    database = Database(f"sqlite:///{db_path}")
    await database.initialize()
    yield database
    await database.close()


# ============================================================
# Mock Ollama Client
# ============================================================


class MockOllamaClient:
    """A mock OllamaClient that returns canned responses without HTTP."""

    def __init__(self, default_response: str = "Mock LLM response.") -> None:
        self.default_response = default_response
        self.chat_calls: list[dict[str, Any]] = []
        self._health = True

        # Expose same attributes as real client
        self.base_url = "http://mock:11434"
        self.default_model = "mock-model"
        self.timeout = 10
        self.num_ctx = 8192
        self.max_retries = 1
        self.retry_base_delay = 0.01
        self.max_concurrent = 1

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        auto_trim: bool = False,
    ) -> str:
        self.chat_calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        )
        return self.default_response

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
    ) -> tuple[str, list[dict]]:
        self.chat_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        )
        return self.default_response, []

    async def health_check(self) -> bool:
        return self._health

    async def preload_model(self, model: str | None = None) -> bool:
        return True

    async def close(self) -> None:
        pass

    def get_stats(self) -> dict[str, Any]:
        return {"total_requests": len(self.chat_calls)}


@pytest.fixture
def mock_ollama() -> MockOllamaClient:
    """Provide a mock OllamaClient."""
    return MockOllamaClient()


# ============================================================
# Mock Message Bus
# ============================================================


class MockPubSub:
    """Mock Redis pubsub object that never yields messages."""

    def __init__(self) -> None:
        self._closed = False

    async def subscribe(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def unsubscribe(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def aclose(self) -> None:
        self._closed = True

    async def listen(self) -> AsyncIterator[dict]:  # type: ignore[return]
        """Block forever (or until cancelled)."""
        while True:
            await asyncio.sleep(3600)


class MockRedis:
    """Minimal mock redis client with pubsub support."""

    def pubsub(self, *args: Any, **kwargs: Any) -> MockPubSub:
        return MockPubSub()


class MockMessageBus:
    """A mock MessageBus that records publishes and returns nothing on consume."""

    def __init__(self) -> None:
        self.published: list[AgentMessage] = []
        self.acknowledged: list[tuple[str, str]] = []
        self.dead_lettered: list[tuple[str, str, AgentMessage, str]] = []
        self.ui_events: list[tuple[str, dict]] = []
        self._connected = True
        self.redis = MockRedis()

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def setup_agent_streams(self, agents: list[AgentRole]) -> None:
        pass

    async def publish(self, message: AgentMessage) -> str:
        self.published.append(message)
        return f"mock-{len(self.published)}"

    async def consume(
        self,
        agent: AgentRole,
        *,
        last_id: str = ">",
        block_ms: int = 2000,
        count: int = 1,
    ) -> list[tuple[str, str, AgentMessage]]:
        # Return nothing by default; tests can override
        return []

    async def claim_stale_messages(
        self, agent: AgentRole, *, min_idle_ms: int = 120_000
    ) -> list[tuple[str, str, AgentMessage]]:
        return []

    async def acknowledge(self, stream: str, entry_id: str) -> None:
        self.acknowledged.append((stream, entry_id))

    async def move_to_dead_letter(
        self,
        stream: str,
        entry_id: str,
        message: AgentMessage,
        *,
        reason: str = "",
    ) -> None:
        self.dead_lettered.append((stream, entry_id, message, reason))

    async def publish_ui_event(self, event_type: str, data: dict[str, Any]) -> None:
        self.ui_events.append((event_type, data))

    async def trim_streams(self) -> None:
        pass

    async def get_pending_count(self, agent: AgentRole) -> int:
        return 0

    async def get_stream_length(self, stream: str) -> int:
        return 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "messages_published": len(self.published),
            "messages_consumed": 0,
            "connected": self._connected,
        }


@pytest.fixture
def mock_bus() -> MockMessageBus:
    """Provide a mock MessageBus."""
    return MockMessageBus()


# ============================================================
# Message Factory Helper
# ============================================================


def make_message(
    *,
    sender: AgentRole = AgentRole.USER,
    recipient: AgentRole = AgentRole.PM,
    msg_type: MessageType = MessageType.SYSTEM,
    project_id: str = "test-project-1",
    task_id: str = "",
    content: str = "Test message content",
    priority: Priority = Priority.NORMAL,
    metadata: dict[str, str] | None = None,
) -> AgentMessage:
    """Create an AgentMessage for testing."""
    return AgentMessage(
        sender=sender,
        recipient=recipient,
        type=msg_type,
        project_id=project_id,
        task_id=task_id,
        payload=MessagePayload(
            content=content,
            metadata=metadata or {},
        ),
        priority=priority,
    )
