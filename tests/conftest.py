"""Shared test fixtures for HCA Orchestration test suite.

All tests run fully offline — no Ollama server, no Redis required.
- Database: real SQLite via temp files
- Ollama: mock httpx transport
- MessageBus: mock (AsyncMock)
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.database import Database
from src.core.models import (
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
        self.generate_calls: list[dict[str, Any]] = []
        self._health = True

        # Expose same attributes as real client
        self.base_url = "http://mock:11434"
        self.default_model = "mock-model"
        self.timeout = 10
        self.num_ctx = 8192
        self.max_retries = 1
        self.retry_base_delay = 0.01

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        auto_trim: bool = False,
    ) -> str:
        self.chat_calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return self.default_response

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.generate_calls.append({"prompt": prompt, **kwargs})
        return self.default_response

    async def health_check(self) -> bool:
        return self._health

    async def preload_model(self, model: str | None = None) -> bool:
        return True

    async def close(self) -> None:
        pass

    def get_stats(self) -> dict[str, Any]:
        return {"total_requests": len(self.chat_calls) + len(self.generate_calls)}


@pytest.fixture
def mock_ollama() -> MockOllamaClient:
    """Provide a mock OllamaClient."""
    return MockOllamaClient()


# ============================================================
# Mock Message Bus
# ============================================================


class MockMessageBus:
    """A mock MessageBus that records publishes and returns nothing on consume."""

    def __init__(self) -> None:
        self.published: list[AgentMessage] = []
        self.acknowledged: list[tuple[str, str]] = []
        self.dead_lettered: list[tuple[str, str, AgentMessage, str]] = []
        self.ui_events: list[tuple[str, dict]] = []
        self._connected = True

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
