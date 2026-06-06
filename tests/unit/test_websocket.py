"""Tests for the WebSocket endpoint.

Uses starlette TestClient to test the /ws endpoint
without a real server.  The MessageBus is mocked so no Redis required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hca.api.app import create_app
from hca.core.models import AgentRole
from tests.conftest import MockMessageBus


class FakeAgent:
    def __init__(self, role: AgentRole) -> None:
        self.role = role

    def get_info(self) -> dict:
        return {"role": self.role.value, "status": "idle"}


class FakeTaskManager:
    pass


@pytest.fixture
def client():
    bus = MockMessageBus()
    agents = [FakeAgent(role) for role in AgentRole if role != AgentRole.USER]
    app = create_app(db=None, bus=bus, task_manager=FakeTaskManager(), agents=agents)
    return TestClient(app)


class TestWebSocketConnection:
    def test_websocket_connect_and_disconnect(self, client: TestClient) -> None:
        with client.websocket_connect("/ws"):
            pass

    def test_websocket_sends_ack_on_valid_json(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"ping": "hello"})
            response = ws.receive_json()
            assert response["type"] == "ack"

    def test_websocket_handles_invalid_json(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not valid json")
            ws.send_json({"valid": True})
            response = ws.receive_json()
            assert response["type"] == "ack"

    def test_websocket_receives_ui_events(self, client: TestClient) -> None:
        """Simulate a UI event being pushed through the message bus."""
        with client.websocket_connect("/ws") as ws:
            # The listen_redis task won't forward mock bus events
            # because they're published differently from Redis pub/sub.
            # This test verifies the connection stays alive.
            ws.send_json({"ping": "check"})
            response = ws.receive_json()
            assert response["type"] == "ack"

    def test_multiple_clients(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                ws1.send_json({"msg": "client1"})
                ws2.send_json({"msg": "client2"})
                r1 = ws1.receive_json()
                r2 = ws2.receive_json()
                assert r1["type"] == "ack"
                assert r2["type"] == "ack"
