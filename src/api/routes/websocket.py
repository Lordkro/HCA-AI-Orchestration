"""WebSocket endpoint for real-time UI updates."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections.

    NOTE: This is a per-process singleton.  If you scale to multiple
    uvicorn workers you will need Redis-backed connection tracking.
    """

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass  # Already removed

    async def broadcast(self, message: str) -> None:
        """Send a message to all connected clients."""
        disconnected: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time agent activity updates."""
    await manager.connect(websocket)
    bus = websocket.app.state.bus

    pubsub = None
    redis_task = None

    try:
        # Subscribe to Redis pub/sub for real-time notifications
        pubsub = bus.redis.pubsub()
        await pubsub.subscribe("hca:notifications")

        async def listen_redis() -> None:
            """Forward Redis pub/sub messages to the WebSocket client."""
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        await websocket.send_text(data)
            except asyncio.CancelledError:
                pass

        redis_task = asyncio.create_task(listen_redis())

        while True:
            # Receive any messages from the client (e.g., commands)
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                await websocket.send_text(json.dumps({"type": "ack", "command": cmd}))
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        # Clean up in all cases (normal close, error, disconnect)
        manager.disconnect(websocket)
        if redis_task is not None:
            redis_task.cancel()
        if pubsub is not None:
            try:
                await pubsub.unsubscribe("hca:notifications")
            except Exception:
                pass
