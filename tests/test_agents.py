"""Tests for the BaseAgent framework and concrete agent routing.

Uses MockOllamaClient and MockMessageBus from conftest —
no real Ollama or Redis required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base_agent import AgentStats, BaseAgent
from src.core.database import Database
from src.core.models import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    MessagePayload,
    MessageType,
    Priority,
)
from tests.conftest import MockMessageBus, MockOllamaClient, make_message


# ============================================================
# Concrete test agent
# ============================================================


class StubAgent(BaseAgent):
    """Minimal concrete agent for testing BaseAgent behavior."""

    def __init__(self, **kwargs) -> None:
        super().__init__(role=AgentRole.PM, **kwargs)
        self.processed: list[AgentMessage] = []
        self.raise_on_process: Exception | None = None

    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        if self.raise_on_process:
            raise self.raise_on_process
        self.processed.append(message)
        # Return a response to test the send path
        return self.create_message(
            recipient=message.sender,
            msg_type=MessageType.ANSWER,
            project_id=message.project_id,
            content=f"Processed: {message.payload.content}",
        )


# ============================================================
# Agent Stats
# ============================================================


class TestAgentStats:
    """Tests for the AgentStats dataclass."""

    def test_initial_values(self) -> None:
        stats = AgentStats()
        assert stats.messages_received == 0
        assert stats.messages_sent == 0
        assert stats.llm_calls == 0

    def test_snapshot(self) -> None:
        stats = AgentStats()
        stats.messages_received = 5
        stats.llm_calls = 3
        stats.total_think_seconds = 9.0
        stats.started_at = 1000.0

        snap = stats.snapshot()
        assert snap["messages_received"] == 5
        assert snap["llm_calls"] == 3
        assert snap["avg_think_seconds"] == 3.0

    def test_snapshot_no_llm_calls_avg(self) -> None:
        stats = AgentStats()
        snap = stats.snapshot()
        assert snap["avg_think_seconds"] == 0


# ============================================================
# BaseAgent Lifecycle
# ============================================================


class TestBaseAgentLifecycle:
    """Tests for agent init, start, stop, and status."""

    async def test_init_status_stopped(self, db: Database, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        assert agent.status == AgentStatus.STOPPED
        assert agent.role == AgentRole.PM

    async def test_stop_sets_running_false(self, db: Database, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        agent._running = True
        await agent.stop()
        assert agent._running is False

    async def test_system_prompt_loads_from_file(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        # PM prompt file should exist in src/prompts/pm.txt
        if agent._prompt_path.exists():
            assert len(agent._system_prompt) > 10
        else:
            assert "pm" in agent._system_prompt.lower()

    async def test_reload_prompt(self, db, mock_bus, mock_ollama, tmp_path) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        original = agent._system_prompt

        # Patch prompt path to a temp file
        custom_prompt = "You are a test agent."
        prompt_file = tmp_path / "pm.txt"
        prompt_file.write_text(custom_prompt, encoding="utf-8")

        with patch.object(type(agent), "_prompt_path", new_callable=lambda: property(lambda self: prompt_file)):
            agent.reload_prompt()
            assert agent._system_prompt == custom_prompt


# ============================================================
# Per-Project Conversation Memory
# ============================================================


class TestProjectMemory:
    """Tests for per-project conversation history isolation."""

    async def test_histories_isolated_by_project(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)

        await agent.think("Hello project A", project_id="proj-a")
        await agent.think("Hello project B", project_id="proj-b")

        assert "proj-a" in agent._project_histories
        assert "proj-b" in agent._project_histories
        # Each project should have 2 entries (user + assistant)
        assert len(agent._project_histories["proj-a"]) == 2
        assert len(agent._project_histories["proj-b"]) == 2

    async def test_history_auto_prune(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        agent.MAX_HISTORY_PER_PROJECT = 4  # Tiny limit for testing

        # Add 5 turns (10 entries: user + assistant each)
        for i in range(5):
            await agent.think(f"Turn {i}", project_id="proj")

        history = agent._project_histories["proj"]
        assert len(history) <= 4

    async def test_clear_history_single_project(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        await agent.think("A", project_id="proj-a")
        await agent.think("B", project_id="proj-b")

        agent.clear_history("proj-a")
        assert "proj-a" not in agent._project_histories
        assert "proj-b" in agent._project_histories

    async def test_clear_history_all(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        await agent.think("A", project_id="proj-a")
        await agent.think("B", project_id="proj-b")

        agent.clear_history()
        assert len(agent._project_histories) == 0

    async def test_think_without_project_id_uses_global(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        await agent.think("Global question")
        assert "_global" in agent._project_histories


# ============================================================
# LLM Interaction (think)
# ============================================================


class TestThink:
    """Tests for the think() method."""

    async def test_think_calls_ollama_chat(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        result = await agent.think("What is 2+2?", project_id="p1")

        assert result == "Mock LLM response."
        assert len(mock_ollama.chat_calls) == 1
        call = mock_ollama.chat_calls[0]
        # Should include system prompt and user message
        roles = [m["role"] for m in call["messages"]]
        assert "system" in roles
        assert "user" in roles

    async def test_think_records_stats(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        await agent.think("Question", project_id="p1")
        assert agent.stats.llm_calls == 1
        assert agent.stats.total_think_seconds > 0

    async def test_think_error_increments_llm_errors(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        mock_ollama.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

        with pytest.raises(RuntimeError):
            await agent.think("Fail", project_id="p1")

        assert agent.stats.llm_errors == 1
        assert agent.status == AgentStatus.ERROR


# ============================================================
# Message Handling
# ============================================================


class TestMessageHandling:
    """Tests for reliable message processing."""

    async def test_handle_message_reliable_success(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        msg = make_message(sender=AgentRole.USER, recipient=AgentRole.PM)

        await agent._handle_message_reliable("stream:pm", "entry-1", msg)

        # Should have processed the message
        assert len(agent.processed) == 1
        # Should have acknowledged
        assert ("stream:pm", "entry-1") in mock_bus.acknowledged
        # Should have published a response
        assert len(mock_bus.published) == 1
        # Stats
        assert agent.stats.messages_received == 1
        assert agent.stats.messages_sent == 1

    async def test_handle_message_reliable_failure_dead_letters(
        self, db, mock_bus, mock_ollama
    ) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        agent.raise_on_process = ValueError("Broken!")
        agent.MAX_PROCESSING_RETRIES = 0  # No retries

        msg = make_message()
        await agent._handle_message_reliable("stream:pm", "entry-2", msg)

        # Should have dead-lettered
        assert len(mock_bus.dead_lettered) == 1
        assert mock_bus.dead_lettered[0][3]  # reason is non-empty
        assert agent.stats.messages_dead_lettered == 1

    async def test_handle_message_reliable_retries_then_dead_letters(
        self, db, mock_bus, mock_ollama
    ) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        agent.raise_on_process = RuntimeError("Flaky!")
        agent.MAX_PROCESSING_RETRIES = 1  # 1 retry = 2 total attempts

        msg = make_message()
        await agent._handle_message_reliable("stream:pm", "entry-3", msg)

        # Should have failed twice then dead-lettered
        assert agent.stats.messages_failed == 2
        assert agent.stats.messages_dead_lettered == 1

    async def test_message_saved_to_db(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        msg = make_message(project_id="proj-db-test")

        await agent._handle_message_reliable("stream:pm", "entry-4", msg)

        # Both the incoming message and the response should be in DB
        messages = await db.get_project_messages("proj-db-test")
        assert len(messages) == 2  # original + response


# ============================================================
# Message Sending Helpers
# ============================================================


class TestMessageHelpers:
    """Tests for create_message and send."""

    async def test_create_message(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        msg = agent.create_message(
            recipient=AgentRole.CODER,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id="proj-1",
            content="Code this",
            metadata={"key": "val"},
        )
        assert msg.sender == AgentRole.PM
        assert msg.recipient == AgentRole.CODER
        assert msg.payload.content == "Code this"
        assert msg.payload.metadata["key"] == "val"

    async def test_send_publishes_and_saves(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        await agent.send(
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id="proj-1",
            content="Research this",
        )
        assert len(mock_bus.published) == 1
        assert agent.stats.messages_sent == 1

        # Should also be saved to DB
        messages = await db.get_project_messages("proj-1")
        assert len(messages) == 1


# ============================================================
# Agent Info (for API)
# ============================================================


class TestGetInfo:
    """Tests for the get_info() API snapshot."""

    async def test_get_info_structure(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        info = agent.get_info()

        assert info["role"] == "pm"
        assert info["status"] == "stopped"
        assert "stats" in info
        assert "active_projects" in info
        assert isinstance(info["history_sizes"], dict)

    async def test_get_info_reflects_activity(self, db, mock_bus, mock_ollama) -> None:
        agent = StubAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        await agent.think("Hello", project_id="proj-x")

        info = agent.get_info()
        assert "proj-x" in info["active_projects"]
        assert info["history_sizes"]["proj-x"] == 2
        assert info["stats"]["llm_calls"] == 1
