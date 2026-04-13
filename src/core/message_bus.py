"""Redis Streams message bus for agent communication."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import redis.asyncio as aioredis
import structlog

from src.core.models import AgentMessage, AgentRole

logger = structlog.get_logger()

# Stream names
AGENT_STREAM = "hca:agents:{agent}"      # Per-agent inbox
BROADCAST_STREAM = "hca:broadcast"         # Broadcast messages
EVENT_STREAM = "hca:events"                # UI event stream

CallbackType = Callable[[AgentMessage], Coroutine[Any, Any, None]]


class MessageBus:
    """Redis Streams-based message bus for inter-agent communication."""

    def __init__(self, redis_url: str = "redis://redis:6379/0") -> None:
        self.redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._subscribers: dict[str, list[CallbackType]] = {}

    async def connect(self) -> None:
        """Connect to Redis."""
        self._redis = aioredis.from_url(
            self.redis_url,
            decode_responses=True,
        )
        await self._redis.ping()
        logger.info("message_bus_connected", url=self.redis_url)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.aclose()
            logger.info("message_bus_disconnected")

    @property
    def redis(self) -> aioredis.Redis:
        """Get the Redis connection, raising if not connected."""
        if self._redis is None:
            raise RuntimeError("MessageBus not connected. Call connect() first.")
        return self._redis

    async def publish(self, message: AgentMessage) -> str:
        """Publish a message to the recipient's stream."""
        data = message.model_dump(mode="json")
        serialized = json.dumps(data)

        # Send to recipient's inbox
        if message.recipient == "*":
            stream = BROADCAST_STREAM
        else:
            recipient = message.recipient
            if isinstance(recipient, AgentRole):
                recipient = recipient.value
            stream = AGENT_STREAM.format(agent=recipient)

        msg_id = await self.redis.xadd(stream, {"data": serialized})

        # Also publish to the event stream for UI consumption
        await self.redis.xadd(EVENT_STREAM, {"data": serialized}, maxlen=1000)

        # Pub/sub notification for real-time UI updates
        await self.redis.publish("hca:notifications", serialized)

        logger.debug(
            "message_published",
            msg_id=message.id,
            sender=message.sender,
            recipient=str(message.recipient),
            type=message.type,
            stream=stream,
        )
        return str(msg_id)

    async def consume(
        self,
        agent: AgentRole,
        *,
        last_id: str = "$",
        block_ms: int = 5000,
    ) -> list[AgentMessage]:
        """Consume messages from an agent's inbox stream."""
        stream = AGENT_STREAM.format(agent=agent.value)
        messages: list[AgentMessage] = []

        # Read from agent's personal stream and broadcast
        streams = {stream: last_id, BROADCAST_STREAM: last_id}
        results = await self.redis.xread(streams, count=10, block=block_ms)

        for _stream_name, entries in results:
            for _entry_id, fields in entries:
                try:
                    data = json.loads(fields["data"])
                    msg = AgentMessage(**data)
                    messages.append(msg)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error("message_parse_error", error=str(e))

        return messages

    async def get_recent_events(self, count: int = 50) -> list[AgentMessage]:
        """Get recent events for the UI."""
        entries = await self.redis.xrevrange(EVENT_STREAM, count=count)
        messages = []
        for _entry_id, fields in entries:
            try:
                data = json.loads(fields["data"])
                messages.append(AgentMessage(**data))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return list(reversed(messages))

    async def publish_ui_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event specifically for the UI (not agent-to-agent)."""
        event = {
            "type": event_type,
            "data": json.dumps(data),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.redis.xadd(EVENT_STREAM, event, maxlen=1000)
        await self.redis.publish("hca:notifications", json.dumps(event))
