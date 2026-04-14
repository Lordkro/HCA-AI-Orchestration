"""Base agent class — shared behavior for all HCA agents.

Provides:
- Reliable message consumption with consumer groups
- Per-project conversation memory isolation
- LLM interaction with automatic context management
- Heartbeat / status broadcasting for the dashboard
- Graceful shutdown with in-flight message draining
- Hot-reloadable system prompts
- Agent statistics for monitoring
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.core.config import settings
from src.core.database import Database
from src.core.message_bus import MessageBus
from src.core.models import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    ConversationEntry,
    MessagePayload,
    MessageType,
    Priority,
    TaskState,
)
from src.core.ollama_client import OllamaClient

# Avoid circular import — TaskManager is optional and injected at runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.orchestrator.task_manager import TaskManager

logger = structlog.get_logger()


# ============================================================
# Agent Statistics
# ============================================================


@dataclass
class AgentStats:
    """Cumulative statistics for a single agent."""

    messages_received: int = 0
    messages_sent: int = 0
    messages_failed: int = 0
    messages_dead_lettered: int = 0
    llm_calls: int = 0
    llm_errors: int = 0
    total_think_seconds: float = 0.0
    projects_touched: set[str] = field(default_factory=set)
    started_at: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of current stats."""
        uptime = time.monotonic() - self.started_at if self.started_at else 0
        return {
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "messages_dead_lettered": self.messages_dead_lettered,
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
            "total_think_seconds": round(self.total_think_seconds, 2),
            "avg_think_seconds": (
                round(self.total_think_seconds / self.llm_calls, 2)
                if self.llm_calls
                else 0
            ),
            "projects_touched": len(self.projects_touched),
            "uptime_seconds": round(uptime, 1),
        }


# ============================================================
# Base Agent
# ============================================================


class BaseAgent(ABC):
    """Abstract base class for all HCA agents.

    Provides:
    - Reliable message loop with consumer groups (claim stale → consume new)
    - Per-project conversation memory (agents don't mix project contexts)
    - LLM interaction with automatic context-window trimming
    - Heartbeat events for the dashboard
    - Graceful shutdown (drain in-flight, then stop)
    - Hot-reloadable system prompts
    - Agent stats for the monitoring API
    """

    # How long to wait between heartbeat emissions (seconds)
    HEARTBEAT_INTERVAL: float = 30.0
    # How many conversation turns to keep per project before auto-pruning
    MAX_HISTORY_PER_PROJECT: int = 40
    # Retry processing once before dead-lettering
    MAX_PROCESSING_RETRIES: int = 1
    # Stale message reclaim threshold (ms)
    STALE_CLAIM_MS: int = 120_000

    def __init__(
        self,
        *,
        role: AgentRole,
        bus: MessageBus,
        ollama: OllamaClient,
        db: Database,
        task_manager: TaskManager | None = None,
    ) -> None:
        self.role = role
        self.bus = bus
        self.ollama = ollama
        self.db = db
        self.task_manager = task_manager
        self.status = AgentStatus.STOPPED
        self.stats = AgentStats()

        self._running = False
        self._processing = False  # True while handling a message (for drain)

        # Per-project conversation histories
        self._project_histories: dict[str, list[ConversationEntry]] = {}

        self._system_prompt: str = ""
        self._model: str = settings.get_agent_model(role.value)
        self._last_heartbeat: float = 0.0

        # Load system prompt from file
        self._load_system_prompt()

    # --------------------------------------------------------
    # System Prompt Management
    # --------------------------------------------------------

    @property
    def _prompt_path(self) -> Path:
        return Path(__file__).parent.parent / "prompts" / f"{self.role.value}.txt"

    def _load_system_prompt(self) -> None:
        """Load the system prompt from the prompts directory."""
        prompt_file = self._prompt_path
        if prompt_file.exists():
            self._system_prompt = prompt_file.read_text(encoding="utf-8").strip()
            logger.info("prompt_loaded", agent=self.role.value, file=str(prompt_file))
        else:
            self._system_prompt = (
                f"You are the {self.role.value} agent in an AI development team."
            )
            logger.warning("prompt_default", agent=self.role.value)

    def reload_prompt(self) -> None:
        """Hot-reload the system prompt from disk (e.g. via API call)."""
        self._load_system_prompt()

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    async def start(self) -> None:
        """Start the agent's message processing loop.

        Loop strategy (per iteration):
        1. Emit heartbeat if interval has elapsed
        2. Reclaim any stale/orphaned messages from previous crashes
        3. Consume new messages from the inbox
        4. Acknowledge each message after successful processing
        5. Retry once on failure, then dead-letter
        """
        self._running = True
        self.status = AgentStatus.IDLE
        self.stats.started_at = time.monotonic()
        logger.info("agent_started", agent=self.role.value, model=self._model)

        await self._emit_heartbeat(force=True)

        while self._running:
            try:
                # Heartbeat
                await self._emit_heartbeat()

                # 1. Reclaim stale messages
                stale = await self.bus.claim_stale_messages(
                    self.role, min_idle_ms=self.STALE_CLAIM_MS
                )
                for stream_name, entry_id, msg in stale:
                    if msg.sender != self.role:
                        await self._handle_message_reliable(stream_name, entry_id, msg)

                # 2. Consume new messages
                results = await self.bus.consume(
                    self.role, last_id=">", block_ms=2000
                )
                for stream_name, entry_id, msg in results:
                    if msg.sender == self.role:
                        await self.bus.acknowledge(stream_name, entry_id)
                        continue
                    await self._handle_message_reliable(stream_name, entry_id, msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("agent_loop_error", agent=self.role.value, error=str(e))
                self.status = AgentStatus.ERROR
                await asyncio.sleep(5)

        self.status = AgentStatus.STOPPED
        logger.info("agent_stopped", agent=self.role.value, stats=self.stats.snapshot())

    async def stop(self) -> None:
        """Signal the agent to stop. Waits for any in-flight message to finish."""
        logger.info("agent_stopping", agent=self.role.value)
        self._running = False

        # Drain: wait up to 60s for in-flight message to complete
        for _ in range(120):
            if not self._processing:
                break
            await asyncio.sleep(0.5)

        await self._emit_heartbeat(force=True, stopping=True)

    # --------------------------------------------------------
    # Heartbeat
    # --------------------------------------------------------

    async def _emit_heartbeat(
        self, *, force: bool = False, stopping: bool = False
    ) -> None:
        """Publish a heartbeat event so the dashboard can show agent status."""
        now = time.monotonic()
        if not force and (now - self._last_heartbeat) < self.HEARTBEAT_INTERVAL:
            return
        self._last_heartbeat = now

        await self.bus.publish_ui_event(
            "agent_heartbeat",
            {
                "agent": self.role.value,
                "status": "stopping" if stopping else self.status.value,
                "model": self._model,
                "stats": self.stats.snapshot(),
            },
        )

    # --------------------------------------------------------
    # Reliable Message Processing
    # --------------------------------------------------------

    async def _handle_message_reliable(
        self, stream_name: str, entry_id: str, message: AgentMessage
    ) -> None:
        """Process a message with retry, ack, and dead-letter support."""
        self.stats.messages_received += 1
        self.stats.projects_touched.add(message.project_id)

        last_error: Exception | None = None
        for attempt in range(1 + self.MAX_PROCESSING_RETRIES):
            try:
                await self._handle_message(message)
                # Success
                await self.bus.acknowledge(stream_name, entry_id)
                return
            except Exception as e:
                last_error = e
                self.stats.messages_failed += 1
                logger.warning(
                    "message_processing_failed",
                    agent=self.role.value,
                    msg_id=message.id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.MAX_PROCESSING_RETRIES:
                    await asyncio.sleep(2)  # Brief pause before retry

        # All retries exhausted — dead-letter
        self.stats.messages_dead_lettered += 1
        await self.bus.move_to_dead_letter(
            stream_name,
            entry_id,
            message,
            reason=f"Agent {self.role.value} failed after {1 + self.MAX_PROCESSING_RETRIES} attempts: {last_error}",
        )

    async def _handle_message(self, message: AgentMessage) -> None:
        """Route an incoming message to the agent-specific handler."""
        self._processing = True
        self.status = AgentStatus.THINKING
        logger.info(
            "message_received",
            agent=self.role.value,
            sender=message.sender.value,
            type=message.type.value,
            project_id=message.project_id,
        )

        try:
            # Persist for history
            await self.db.save_message(message.model_dump(mode="json"))

            # Dispatch to subclass
            response = await self.process_message(message)

            if response:
                await self.bus.publish(response)
                await self.db.save_message(response.model_dump(mode="json"))
                self.stats.messages_sent += 1

        except Exception as e:
            logger.error(
                "message_handling_error",
                agent=self.role.value,
                error=str(e),
                message_id=message.id,
            )
            raise  # Let _handle_message_reliable decide retry/dead-letter
        finally:
            self._processing = False
            self.status = AgentStatus.IDLE

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process an incoming message. Must be implemented by each agent.

        Returns an AgentMessage to send as a response, or None if no response.
        """
        ...

    # --------------------------------------------------------
    # Per-Project Conversation Memory
    # --------------------------------------------------------

    def _get_history(self, project_id: str) -> list[ConversationEntry]:
        """Get (or create) the conversation history for a project."""
        if project_id not in self._project_histories:
            self._project_histories[project_id] = []
        return self._project_histories[project_id]

    def _append_history(
        self, project_id: str, role: str, content: str
    ) -> None:
        """Append a turn to a project's history, auto-pruning old turns."""
        history = self._get_history(project_id)
        history.append(ConversationEntry(role=role, content=content))

        # Auto-prune: keep the most recent turns only
        if len(history) > self.MAX_HISTORY_PER_PROJECT:
            # Keep system-prompt-relevant early turns and recent ones
            overflow = len(history) - self.MAX_HISTORY_PER_PROJECT
            del history[:overflow]

    def clear_history(self, project_id: str | None = None) -> None:
        """Clear conversation history for one project or all projects."""
        if project_id:
            self._project_histories.pop(project_id, None)
        else:
            self._project_histories.clear()

    # --------------------------------------------------------
    # LLM Interaction
    # --------------------------------------------------------

    async def think(
        self,
        prompt: str,
        *,
        project_id: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a prompt to the LLM with system prompt and per-project history.

        Args:
            prompt: The user-role content to send.
            project_id: Isolates conversation history by project.  If empty,
                        uses a shared "_global" bucket (for non-project queries).
            temperature: Sampling temperature.
            max_tokens: Max tokens in the response.

        Returns:
            The assistant's response text.

        Context window management is handled automatically by OllamaClient
        (auto_trim=True).  Older history entries are dropped first.
        """
        pid = project_id or "_global"
        self.status = AgentStatus.WORKING

        # Build messages list
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt}
        ]

        # Per-project history (OllamaClient auto-trims if too long)
        for entry in self._get_history(pid):
            messages.append({"role": entry.role, "content": entry.content})

        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        try:
            response = await self.ollama.chat(
                messages,
                model=self._model,
                temperature=temperature,
                max_tokens=max_tokens,
                auto_trim=True,
            )
        except Exception as e:
            self.stats.llm_errors += 1
            logger.error("agent_think_error", agent=self.role.value, error=str(e))
            self.status = AgentStatus.ERROR
            raise

        elapsed = time.monotonic() - t0
        self.stats.llm_calls += 1
        self.stats.total_think_seconds += elapsed

        # Record in per-project history
        self._append_history(pid, "user", prompt)
        self._append_history(pid, "assistant", response)

        self.status = AgentStatus.IDLE
        return response

    # --------------------------------------------------------
    # Message Sending Helpers
    # --------------------------------------------------------

    async def _transition_task(
        self, task_id: str, new_state: TaskState
    ) -> bool:
        """Attempt a task state transition via the TaskManager.

        Returns True on success, False if no TaskManager is set or the
        transition is invalid.  Logs warnings on failure rather than
        raising, because a failed transition should not crash an agent.
        """
        if not self.task_manager or not task_id:
            return False
        try:
            await self.task_manager.transition(task_id, new_state)
            return True
        except ValueError as exc:
            logger.warning(
                "agent_transition_failed",
                agent=self.role.value,
                task_id=task_id,
                target_state=new_state.value,
                reason=str(exc),
            )
            return False

    def create_message(
        self,
        *,
        recipient: AgentRole,
        msg_type: MessageType,
        project_id: str,
        content: str,
        task_id: str = "",
        artifacts: list[str] | None = None,
        metadata: dict[str, str] | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> AgentMessage:
        """Create a new outbound message."""
        return AgentMessage(
            sender=self.role,
            recipient=recipient,
            type=msg_type,
            project_id=project_id,
            task_id=task_id,
            payload=MessagePayload(
                content=content,
                artifacts=artifacts or [],
                metadata=metadata or {},
            ),
            priority=priority,
        )

    async def send(
        self,
        *,
        recipient: AgentRole,
        msg_type: MessageType,
        project_id: str,
        content: str,
        task_id: str = "",
        artifacts: list[str] | None = None,
        metadata: dict[str, str] | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        """Create and publish a message in one step."""
        msg = self.create_message(
            recipient=recipient,
            msg_type=msg_type,
            project_id=project_id,
            content=content,
            task_id=task_id,
            artifacts=artifacts,
            metadata=metadata,
            priority=priority,
        )
        await self.bus.publish(msg)
        await self.db.save_message(msg.model_dump(mode="json"))
        self.stats.messages_sent += 1

    # --------------------------------------------------------
    # Agent Info (for API / Dashboard)
    # --------------------------------------------------------

    def get_info(self) -> dict[str, Any]:
        """Return a snapshot of agent state for the monitoring API."""
        return {
            "role": self.role.value,
            "status": self.status.value,
            "model": self._model,
            "active_projects": list(self._project_histories.keys()),
            "history_sizes": {
                pid: len(h) for pid, h in self._project_histories.items()
            },
            "stats": self.stats.snapshot(),
        }
