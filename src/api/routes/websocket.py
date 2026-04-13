"""WebSocket endpoint for real-time UI updates."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
        """Send a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time agent activity updates."""
    await manager.connect(websocket)
    bus = websocket.app.state.bus

    try:
        # Subscribe to Redis pub/sub for real-time notifications
        pubsub = bus.redis.pubsub()
        await pubsub.subscribe("hca:notifications")

        async def listen_redis() -> None:
            """Forward Redis pub/sub messages to the WebSocket client."""
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    await websocket.send_text(data)

        # Run Redis listener and WebSocket receiver concurrently
        redis_task = asyncio.create_task(listen_redis())

        try:
            while True:
                # Receive any messages from the client (e.g., commands)
                data = await websocket.receive_text()
                # Client can send commands via WebSocket if needed
                try:
                    cmd = json.loads(data)
                    # Handle client commands here in the future
                    await websocket.send_text(json.dumps({"type": "ack", "command": cmd}))
                except json.JSONDecodeError:
                    pass
        finally:
            redis_task.cancel()
            await pubsub.unsubscribe("hca:notifications")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
