"""Tests for the Redis message bus.

Since we don't have Redis in the test environment, these tests focus on:
- Message serialization/deserialization round-trips
- Bus initialization and configuration
- BusStats tracking
- The MockMessageBus (used by all other test modules) itself

Integration tests with real Redis are deferred to CI with Docker.
"""

from __future__ import annotations

import json

import pytest

from src.core.message_bus import BusStats, CONSUMER_GROUP, AGENT_STREAM
from src.core.models import (
    AgentMessage,
    AgentRole,
    MessagePayload,
    MessageType,
    Priority,
)
from tests.conftest import MockMessageBus, make_message


# ============================================================
# Message Serialization Round-Trip
# ============================================================


class TestMessageSerialization:
    """Test that AgentMessage can round-trip through JSON (as the bus does)."""

    def test_serialize_deserialize(self) -> None:
        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.CODER,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id="proj-1",
            content="Build the feature",
            metadata={"artifact_type": "code"},
        )
        # Serialize
        data = msg.model_dump(mode="json")
        json_str = json.dumps(data, default=str)

        # Deserialize
        parsed = json.loads(json_str)
        restored = AgentMessage(**parsed)

        assert restored.id == msg.id
        assert restored.sender == AgentRole.PM
        assert restored.recipient == AgentRole.CODER
        assert restored.type == MessageType.TASK_ASSIGNMENT
        assert restored.payload.content == "Build the feature"
        assert restored.payload.metadata["artifact_type"] == "code"

    def test_serialize_with_artifacts(self) -> None:
        msg = make_message(content="Code output")
        msg.payload.artifacts = ["main.py", "test_main.py"]
        data = msg.model_dump(mode="json")
        restored = AgentMessage(**data)
        assert restored.payload.artifacts == ["main.py", "test_main.py"]

    def test_all_priority_levels(self) -> None:
        for priority in Priority:
            msg = make_message(priority=priority)
            data = msg.model_dump(mode="json")
            restored = AgentMessage(**data)
            assert restored.priority == priority

    def test_all_message_types(self) -> None:
        for msg_type in MessageType:
            msg = make_message(msg_type=msg_type)
            data = msg.model_dump(mode="json")
            restored = AgentMessage(**data)
            assert restored.type == msg_type

    def test_all_agent_roles(self) -> None:
        roles = [r for r in AgentRole if r not in (AgentRole.SYSTEM, AgentRole.USER)]
        for role in roles:
            msg = make_message(sender=role, recipient=AgentRole.PM)
            data = msg.model_dump(mode="json")
            restored = AgentMessage(**data)
            assert restored.sender == role


# ============================================================
# Bus Configuration Constants
# ============================================================


class TestBusConfig:
    """Test that bus configuration constants are correct."""

    def test_consumer_group_name(self) -> None:
        assert CONSUMER_GROUP == "hca-workers"

    def test_agent_stream_pattern(self) -> None:
        assert AGENT_STREAM == "hca:agents:{agent}"

    def test_stream_name_for_agent(self) -> None:
        expected = AGENT_STREAM.format(agent="pm")
        assert expected == "hca:agents:pm"


# ============================================================
# BusStats
# ============================================================


class TestBusStats:
    """Tests for BusStats tracking."""

    def test_initial_values(self) -> None:
        stats = BusStats()
        assert stats.messages_published == 0
        assert stats.messages_consumed == 0
        assert stats.messages_acknowledged == 0

    def test_record_publish(self) -> None:
        stats = BusStats()
        stats.record_publish("task_assignment")
        stats.record_publish("task_assignment")
        stats.record_publish("deliverable")

        assert stats.messages_published == 3
        assert stats.messages_by_type["task_assignment"] == 2
        assert stats.messages_by_type["deliverable"] == 1


# ============================================================
# MockMessageBus (validate the test infrastructure itself)
# ============================================================


class TestMockMessageBus:
    """Tests that our MockMessageBus works correctly for other tests."""

    async def test_publish_records_message(self) -> None:
        bus = MockMessageBus()
        msg = make_message()
        entry_id = await bus.publish(msg)
        assert len(bus.published) == 1
        assert bus.published[0] is msg
        assert entry_id.startswith("mock-")

    async def test_consume_returns_empty(self) -> None:
        bus = MockMessageBus()
        results = await bus.consume(AgentRole.PM)
        assert results == []

    async def test_acknowledge_records(self) -> None:
        bus = MockMessageBus()
        await bus.acknowledge("stream:pm", "entry-1")
        assert ("stream:pm", "entry-1") in bus.acknowledged

    async def test_dead_letter_records(self) -> None:
        bus = MockMessageBus()
        msg = make_message()
        await bus.move_to_dead_letter("stream:pm", "e-1", msg, reason="test failure")
        assert len(bus.dead_lettered) == 1
        assert bus.dead_lettered[0][3] == "test failure"

    async def test_ui_event_records(self) -> None:
        bus = MockMessageBus()
        await bus.publish_ui_event("agent_heartbeat", {"agent": "pm"})
        assert len(bus.ui_events) == 1
        assert bus.ui_events[0][0] == "agent_heartbeat"

    async def test_connect_disconnect(self) -> None:
        bus = MockMessageBus()
        assert bus._connected is True
        await bus.disconnect()
        assert bus._connected is False
        await bus.connect()
        assert bus._connected is True
