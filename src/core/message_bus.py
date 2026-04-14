"""Redis Streams message bus for agent communication.

Provides reliable inter-agent messaging with:
- Consumer groups for guaranteed delivery
- Message acknowledgment and retry for failed processing
- Dead letter queue for poison messages
- Stream size management (maxlen capping)
- Per-agent inbox streams + broadcast stream
- Real-time pub/sub forwarding for the UI
- Connection health monitoring with auto-reconnect
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import redis.asyncio as aioredis
import structlog

from src.core.models import AgentMessage, AgentRole

logger = structlog.get_logger()

# ============================================================
# Stream & Group Names
# ============================================================

AGENT_STREAM = "hca:agents:{agent}"        # Per-agent inbox
BROADCAST_STREAM = "hca:broadcast"           # Broadcast messages
EVENT_STREAM = "hca:events"                  # UI event stream
DEAD_LETTER_STREAM = "hca:deadletter"        # Failed messages

CONSUMER_GROUP = "hca-workers"               # Consumer group name
MAX_STREAM_LEN = 5000                        # Max entries per stream
MAX_EVENT_STREAM_LEN = 2000                  # Max entries in event stream
MAX_DELIVERY_ATTEMPTS = 3                    # Before moving to dead letter

CallbackType = Callable[[AgentMessage], Coroutine[Any, Any, None]]


# ============================================================
# Statistics
# ============================================================

@dataclass
class BusStats:
    """Cumulative statistics for the message bus."""
    messages_published: int = 0
    messages_consumed: int = 0
    messages_acknowledged: int = 0
    messages_dead_lettered: int = 0
    parse_errors: int = 0
    publish_errors: int = 0
    reconnections: int = 0
    messages_by_type: dict[str, int] = field(default_factory=dict)

    def record_publish(self, msg_type: str) -> None:
        self.messages_published += 1
        self.messages_by_type[msg_type] = self.messages_by_type.get(msg_type, 0) + 1


# ============================================================
# Message Bus
# ============================================================


class MessageBusError(Exception):
    """Base exception for message bus errors."""


class MessageBus:
    """Redis Streams-based message bus for inter-agent communication.

    Uses Redis consumer groups for reliable message delivery:
    - Each agent has a personal inbox stream (hca:agents:{role})
    - A broadcast stream delivers messages to all agents
    - Consumer groups ensure messages are processed exactly once
    - Failed messages are retried up to MAX_DELIVERY_ATTEMPTS
    - Poison messages go to a dead letter stream
    - The event stream feeds the UI via pub/sub
    """

    def __init__(
        self,
        redis_url: str = "redis://redis:6379/0",
        max_stream_len: int = MAX_STREAM_LEN,
    ) -> None:
        self.redis_url = redis_url
        self.max_stream_len = max_stream_len
        self._redis: aioredis.Redis | None = None
        self._connected = False
        self.stats = BusStats()

    # --------------------------------------------------------
    # Connection Management
    # --------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Redis and initialize consumer groups."""
        self._redis = aioredis.from_url(
            self.redis_url,
            decode_responses=True,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        await self._redis.ping()
        self._connected = True
        logger.info("message_bus_connected", url=self.redis_url)

    async def disconnect(self) -> None:
        """Disconnect from Redis gracefully."""
        if self._redis:
            self._connected = False
            await self._redis.aclose()
            self._redis = None
            logger.info("message_bus_disconnected")

    @property
    def redis(self) -> aioredis.Redis:
        """Get the Redis connection, raising if not connected."""
        if self._redis is None or not self._connected:
            raise MessageBusError("MessageBus not connected. Call connect() first.")
        return self._redis

    async def _ensure_connected(self) -> None:
        """Verify connection is alive, reconnect if needed."""
        try:
            await self.redis.ping()
        except (aioredis.ConnectionError, aioredis.TimeoutError, OSError):
            logger.warning("message_bus_reconnecting")
            self.stats.reconnections += 1
            await self.connect()

    async def ensure_consumer_group(self, stream: str, group: str = CONSUMER_GROUP) -> None:
        """Create a consumer group for a stream if it doesn't exist.

        Uses MKSTREAM to create the stream if it doesn't exist yet.
        """
        try:
            await self.redis.xgroup_create(stream, group, id="0", mkstream=True)
            logger.debug("consumer_group_created", stream=stream, group=group)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists — this is fine
                pass
            else:
                raise

    async def setup_agent_streams(self, agents: list[AgentRole]) -> None:
        """Initialize consumer groups for all agent streams.

        Call this during startup to ensure all streams and groups exist.
        """
        streams = [AGENT_STREAM.format(agent=a.value) for a in agents]
        streams.append(BROADCAST_STREAM)
        streams.append(EVENT_STREAM)

        for stream in streams:
            await self.ensure_consumer_group(stream)

        logger.info("agent_streams_initialized", count=len(streams))

    # --------------------------------------------------------
    # Publishing
    # --------------------------------------------------------

    async def publish(self, message: AgentMessage) -> str:
        """Publish a message to the recipient's stream.

        Messages are written to:
        1. The recipient's personal inbox stream (or broadcast)
        2. The event stream (for UI consumption, capped)
        3. The pub/sub channel (for real-time WebSocket push)

        Returns the Redis stream entry ID.
        """
        await self._ensure_connected()

        data = message.model_dump(mode="json")
        serialized = json.dumps(data, default=str)

        # Determine target stream
        if message.recipient == "*":
            stream = BROADCAST_STREAM
        else:
            recipient = message.recipient
            if isinstance(recipient, AgentRole):
                recipient = recipient.value
            stream = AGENT_STREAM.format(agent=recipient)

        try:
            # 1. Write to recipient's inbox
            msg_id = await self.redis.xadd(
                stream, {"data": serialized}, maxlen=self.max_stream_len
            )

            # 2. Write to event stream (for UI history)
            await self.redis.xadd(
                EVENT_STREAM,
                {"data": serialized},
                maxlen=MAX_EVENT_STREAM_LEN,
            )

            # 3. Pub/sub push for real-time WebSocket
            await self.redis.publish("hca:notifications", serialized)

            self.stats.record_publish(message.type.value if hasattr(message.type, 'value') else str(message.type))

            logger.debug(
                "message_published",
                msg_id=message.id,
                sender=str(message.sender),
                recipient=str(message.recipient),
                type=str(message.type),
                stream=stream,
                redis_id=msg_id,
            )
            return str(msg_id)

        except (aioredis.ConnectionError, aioredis.TimeoutError) as e:
            self.stats.publish_errors += 1
            logger.error("message_publish_failed", error=str(e), msg_id=message.id)
            raise MessageBusError(f"Failed to publish message: {e}") from e

    # --------------------------------------------------------
    # Consuming (Consumer Group based)
    # --------------------------------------------------------

    async def consume(
        self,
        agent: AgentRole,
        *,
        consumer_name: str | None = None,
        last_id: str = ">",
        block_ms: int = 5000,
        count: int = 10,
    ) -> list[tuple[str, str, AgentMessage]]:
        """Consume messages from an agent's inbox using consumer groups.

        Uses XREADGROUP for reliable, exactly-once processing.
        Each message must be acknowledged after successful processing.

        Args:
            agent: The agent role consuming messages.
            consumer_name: Unique consumer name (defaults to agent role).
            last_id: ">" for new messages, "0" to re-read pending.
            block_ms: How long to block waiting for messages.
            count: Max messages to read per call.

        Returns:
            List of (stream_name, redis_entry_id, AgentMessage) tuples.
            The redis_entry_id is needed to acknowledge the message.
        """
        await self._ensure_connected()

        consumer = consumer_name or agent.value
        inbox_stream = AGENT_STREAM.format(agent=agent.value)

        # Ensure consumer groups exist
        await self.ensure_consumer_group(inbox_stream)
        await self.ensure_consumer_group(BROADCAST_STREAM)

        messages: list[tuple[str, str, AgentMessage]] = []

        try:
            streams = {inbox_stream: last_id, BROADCAST_STREAM: last_id}
            results = await self.redis.xreadgroup(
                CONSUMER_GROUP,
                consumer,
                streams,
                count=count,
                block=block_ms,
            )

            for stream_name, entries in results:
                for entry_id, fields in entries:
                    try:
                        data = json.loads(fields["data"])
                        msg = AgentMessage(**data)
                        messages.append((stream_name, entry_id, msg))
                        self.stats.messages_consumed += 1
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        self.stats.parse_errors += 1
                        logger.error(
                            "message_parse_error",
                            stream=stream_name,
                            entry_id=entry_id,
                            error=str(e),
                        )
                        # Acknowledge bad messages so they don't block the queue
                        await self.acknowledge(stream_name, entry_id)

        except (aioredis.ConnectionError, aioredis.TimeoutError) as e:
            logger.warning("consume_connection_error", agent=agent.value, error=str(e))
            await asyncio.sleep(1)

        return messages

    async def acknowledge(self, stream: str, entry_id: str) -> None:
        """Acknowledge a message as successfully processed.

        This removes it from the consumer's pending entries list (PEL).
        """
        try:
            await self.redis.xack(stream, CONSUMER_GROUP, entry_id)
            self.stats.messages_acknowledged += 1
        except (aioredis.ConnectionError, aioredis.TimeoutError) as e:
            logger.error("acknowledge_failed", stream=stream, entry_id=entry_id, error=str(e))

    async def claim_stale_messages(
        self,
        agent: AgentRole,
        min_idle_ms: int = 60_000,
        count: int = 10,
    ) -> list[tuple[str, str, AgentMessage]]:
        """Claim messages that have been pending too long (from crashed consumers).

        Uses XAUTOCLAIM to take ownership of stale messages that another
        consumer failed to acknowledge.
        """
        inbox_stream = AGENT_STREAM.format(agent=agent.value)
        consumer = agent.value
        messages: list[tuple[str, str, AgentMessage]] = []

        try:
            # XAUTOCLAIM returns (next_start_id, claimed_entries, deleted_ids)
            result = await self.redis.xautoclaim(
                inbox_stream, CONSUMER_GROUP, consumer,
                min_idle_time=min_idle_ms, count=count,
            )
            if result and len(result) >= 2:
                claimed_entries = result[1]
                for entry_id, fields in claimed_entries:
                    if fields is None:
                        continue
                    try:
                        data = json.loads(fields["data"])
                        msg = AgentMessage(**data)
                        messages.append((inbox_stream, entry_id, msg))
                        logger.info(
                            "stale_message_claimed",
                            stream=inbox_stream,
                            entry_id=entry_id,
                            msg_id=msg.id,
                        )
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.error("stale_message_parse_error", error=str(e))
                        await self.acknowledge(inbox_stream, entry_id)
        except (aioredis.ResponseError, aioredis.ConnectionError) as e:
            logger.debug("xautoclaim_error", error=str(e))

        return messages

    async def move_to_dead_letter(
        self,
        original_stream: str,
        entry_id: str,
        message: AgentMessage,
        *,
        reason: str,
    ) -> None:
        """Move a failed message to the dead letter stream."""
        dead_letter_data = {
            "original_stream": original_stream,
            "original_entry_id": entry_id,
            "reason": reason,
            "data": json.dumps(message.model_dump(mode="json"), default=str),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.redis.xadd(DEAD_LETTER_STREAM, dead_letter_data, maxlen=1000)
        await self.acknowledge(original_stream, entry_id)
        self.stats.messages_dead_lettered += 1
        logger.warning(
            "message_dead_lettered",
            msg_id=message.id,
            stream=original_stream,
            reason=reason,
        )

    # --------------------------------------------------------
    # Pending Messages Inspection
    # --------------------------------------------------------

    async def get_pending_count(self, agent: AgentRole) -> int:
        """Get the number of pending (unacknowledged) messages for an agent."""
        inbox_stream = AGENT_STREAM.format(agent=agent.value)
        try:
            info = await self.redis.xpending(inbox_stream, CONSUMER_GROUP)
            return info.get("pending", 0) if isinstance(info, dict) else 0
        except (aioredis.ResponseError, aioredis.ConnectionError):
            return 0

    async def get_stream_length(self, stream: str) -> int:
        """Get the number of entries in a stream."""
        try:
            return await self.redis.xlen(stream)
        except (aioredis.ResponseError, aioredis.ConnectionError):
            return 0

    # --------------------------------------------------------
    # UI / Event Methods
    # --------------------------------------------------------

    async def get_recent_events(self, count: int = 50) -> list[AgentMessage]:
        """Get recent events for the UI."""
        await self._ensure_connected()
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
        await self._ensure_connected()
        event = {
            "type": event_type,
            "data": json.dumps(data, default=str),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.redis.xadd(EVENT_STREAM, event, maxlen=MAX_EVENT_STREAM_LEN)
        await self.redis.publish("hca:notifications", json.dumps(event, default=str))

    # --------------------------------------------------------
    # Stream Maintenance
    # --------------------------------------------------------

    async def trim_streams(self) -> None:
        """Trim all streams to their maximum length.

        Call this periodically to prevent unbounded memory growth.
        """
        try:
            agents = [role for role in AgentRole if role not in (AgentRole.SYSTEM, AgentRole.USER)]
            for agent in agents:
                stream = AGENT_STREAM.format(agent=agent.value)
                await self.redis.xtrim(stream, maxlen=self.max_stream_len)
            await self.redis.xtrim(BROADCAST_STREAM, maxlen=self.max_stream_len)
            await self.redis.xtrim(EVENT_STREAM, maxlen=MAX_EVENT_STREAM_LEN)
            logger.debug("streams_trimmed")
        except (aioredis.ConnectionError, aioredis.TimeoutError) as e:
            logger.warning("stream_trim_failed", error=str(e))

    async def flush_all(self) -> None:
        """Delete all streams and data. USE WITH CAUTION."""
        agents = [role for role in AgentRole if role not in (AgentRole.SYSTEM, AgentRole.USER)]
        for agent in agents:
            stream = AGENT_STREAM.format(agent=agent.value)
            await self.redis.delete(stream)
        await self.redis.delete(BROADCAST_STREAM)
        await self.redis.delete(EVENT_STREAM)
        await self.redis.delete(DEAD_LETTER_STREAM)
        logger.warning("all_streams_flushed")

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get message bus statistics."""
        return {
            "messages_published": self.stats.messages_published,
            "messages_consumed": self.stats.messages_consumed,
            "messages_acknowledged": self.stats.messages_acknowledged,
            "messages_dead_lettered": self.stats.messages_dead_lettered,
            "parse_errors": self.stats.parse_errors,
            "publish_errors": self.stats.publish_errors,
            "reconnections": self.stats.reconnections,
            "messages_by_type": self.stats.messages_by_type,
            "connected": self._connected,
        }
