"""WebSocket endpoint for real-time UI updates."""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from hca.core.message_bus import MessageBusError
from hca.core.models import AgentMessage, AgentRole, MessagePayload, MessageType

logger = structlog.get_logger()
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
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


async def _handle_hitl_command(cmd: dict, websocket: WebSocket) -> None:
    """Route incoming HITL commands from a WebSocket client to the bus."""
    bus = websocket.app.state.bus
    db = websocket.app.state.db
    cmd_type = cmd.get("type", "")

    if cmd_type == "inject_feedback":
        task_id = cmd.get("task_id", "")
        content = cmd.get("content", "")
        agent_role_str = cmd.get("agent_role")
        project_id = cmd.get("project_id", "")

        if not task_id or not content:
            await websocket.send_json({"type": "error", "message": "task_id and content required"})
            return

        task = await db.get_task(task_id) if task_id else None
        if task:
            project_id = project_id or task.project_id

        agent_role = AgentRole(agent_role_str) if agent_role_str else AgentRole.CODER
        msg = AgentMessage(
            sender=AgentRole.USER,
            recipient=agent_role,
            type=MessageType.FEEDBACK,
            project_id=project_id,
            task_id=task_id,
            payload=MessagePayload(
                content=content,
                metadata={"source": "human", "artifact_type": "code"},
            ),
        )
        await bus.publish(msg)
        await db.save_message(msg.model_dump(mode="json"))
        await websocket.send_json({"type": "feedback_sent", "task_id": task_id})

    elif cmd_type == "approve_task":
        task_id = cmd.get("task_id", "")
        summary = cmd.get("summary", "")
        project_id = cmd.get("project_id", "")

        task = await db.get_task(task_id) if task_id else None
        if not task:
            await websocket.send_json({"type": "error", "message": "Task not found"})
            return
        project_id = project_id or task.project_id

        msg = AgentMessage(
            sender=AgentRole.USER,
            recipient=AgentRole.PM,
            type=MessageType.DELIVERABLE,
            project_id=project_id,
            task_id=task_id,
            payload=MessagePayload(
                content=summary or "Human approved",
                metadata={"review_result": "approved", "artifact_type": task.artifact_type or "code", "source": "human"},
            ),
        )
        await bus.publish(msg)
        await db.save_message(msg.model_dump(mode="json"))
        await websocket.send_json({"type": "task_approved", "task_id": task_id})

    elif cmd_type == "reject_task":
        task_id = cmd.get("task_id", "")
        feedback = cmd.get("feedback", "")
        project_id = cmd.get("project_id", "")

        task = await db.get_task(task_id) if task_id else None
        if not task:
            await websocket.send_json({"type": "error", "message": "Task not found"})
            return
        project_id = project_id or task.project_id

        msg = AgentMessage(
            sender=AgentRole.USER,
            recipient=AgentRole.PM,
            type=MessageType.DELIVERABLE,
            project_id=project_id,
            task_id=task_id,
            payload=MessagePayload(
                content=feedback or "Human requested revision",
                metadata={"review_result": "needs_revision", "artifact_type": task.artifact_type or "code", "source": "human"},
            ),
        )
        await bus.publish(msg)
        await db.save_message(msg.model_dump(mode="json"))
        await websocket.send_json({"type": "task_rejected", "task_id": task_id})


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time agent activity updates."""
    await manager.connect(websocket)
    bus = websocket.app.state.bus

    pubsub = None
    redis_task = None

    try:
        # Subscribe to Redis pub/sub for real-time notifications
        try:
            pubsub = bus.redis.pubsub()
            await pubsub.subscribe("hca:notifications")
        except (MessageBusError, Exception) as e:
            logger.warning("websocket_pubsub_failed", error=str(e))
            await websocket.send_json({"type": "error", "message": "Redis unavailable"})
            return

        async def listen_redis() -> None:
            """Forward Redis pub/sub messages to the WebSocket client."""
            retry_delay = 1.0
            while True:
                try:
                    async for message in pubsub.listen():
                        if message["type"] == "message":
                            data = message["data"]
                            if isinstance(data, bytes):
                                data = data.decode("utf-8")
                            await websocket.send_text(data)
                        retry_delay = 1.0  # Reset on successful message
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.warning(
                        "redis_pubsub_listener_error",
                        retry_delay=retry_delay,
                        exc_info=True,
                    )
                try:
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30.0)  # Exponential backoff, max 30s
                except asyncio.CancelledError:
                    return

        redis_task = asyncio.create_task(listen_redis())

        while True:
            try:
                data = await websocket.receive_text()
            except Exception:
                break
            try:
                cmd = json.loads(data)
                await websocket.send_text(json.dumps({"type": "ack", "command_type": cmd.get("type")}))
                await _handle_hitl_command(cmd, websocket)
            except json.JSONDecodeError:
                await asyncio.sleep(0.1)

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
                await pubsub.aclose()
            except Exception:
                logger.debug(
                    "Failed to close websocket pubsub cleanly",
                    exc_info=True,
                )
