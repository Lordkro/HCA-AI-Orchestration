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

# Avoid circular import — TaskManager is optional and injected at runtime.
from typing import TYPE_CHECKING, Any

import structlog

from hca.core.config import settings
from hca.core.database import Database
from hca.core.message_bus import MessageBus
from hca.core.metrics import (
    agent_status as agent_status_metric,
)
from hca.core.metrics import (
    agent_uptime_seconds,
    record_agent_dead_lettered,
    record_agent_llm_call,
    record_agent_llm_duration,
    record_agent_llm_error,
    record_agent_message_failed,
    record_agent_message_received,
    record_agent_message_sent,
)
from hca.core.models import (
    AgentMessage,
    AgentRole,
    AgentStatus,
    ConversationEntry,
    MessagePayload,
    MessageType,
    Priority,
    TaskState,
)
from hca.core.ollama_client import (
    OllamaCircuitBreakerOpenError,
    OllamaClient,
    estimate_messages_tokens,
    estimate_tokens,
)

if TYPE_CHECKING:
    from hca.orchestrator.task_manager import TaskManager

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
                round(self.total_think_seconds / self.llm_calls, 2) if self.llm_calls else 0
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
    # Retry backoff: base delay in seconds (doubles each retry)
    RETRY_BASE_DELAY: float = 2.0
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

        # Activity tracking for UI feedback
        self._current_activity: str = ""
        self._activity_since: float = 0.0

        # Per-project conversation histories
        self._project_histories: dict[str, list[ConversationEntry]] = {}
        self._history_lock = asyncio.Lock()

        self._system_prompt: str = ""
        self._model: str = settings.get_agent_model(role.value)
        self._temperature: float = settings.get_agent_temperature(role.value)
        self._top_p: float = settings.get_agent_top_p(role.value)
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
            self._system_prompt = f"You are the {self.role.value} agent in an AI development team."
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
        agent_status_metric.labels(agent=self.role.value).set(0)
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
                results = await self.bus.consume(self.role, last_id=">", block_ms=2000)
                if not results:
                    # No messages, pause briefly to avoid tight loop
                    await asyncio.sleep(0.5)
                else:
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
                agent_status_metric.labels(agent=self.role.value).set(2)
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

    async def _emit_heartbeat(self, *, force: bool = False, stopping: bool = False) -> None:
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
        uptime = time.monotonic() - self.stats.started_at
        agent_uptime_seconds.labels(agent=self.role.value).set(uptime)

    # --------------------------------------------------------
    # Reliable Message Processing
    # --------------------------------------------------------

    async def _handle_message_reliable(
        self, stream_name: str, entry_id: str, message: AgentMessage
    ) -> None:
        """Process a message with retry, ack, and dead-letter support."""
        self.stats.messages_received += 1
        record_agent_message_received(self.role.value)
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
                record_agent_message_failed(self.role.value)
                logger.warning(
                    "message_processing_failed",
                    agent=self.role.value,
                    msg_id=message.id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.MAX_PROCESSING_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "msg_retry_backoff",
                        agent=self.role.value,
                        msg_id=message.id,
                        attempt=attempt + 1,
                        delay_seconds=delay,
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        # If the circuit breaker is open, don't dead-letter — leave unacked
        # so the stale claim mechanism retries when Ollama recovers.
        if isinstance(last_error, OllamaCircuitBreakerOpenError):
            logger.warning(
                "message_deferred_circuit_breaker_open",
                agent=self.role.value,
                msg_id=message.id,
            )
            return

        self.stats.messages_dead_lettered += 1
        record_agent_dead_lettered(self.role.value)
        reason = (
            f"Agent {self.role.value} failed after "
            f"{1 + self.MAX_PROCESSING_RETRIES} attempts: {last_error}"
        )
        await self.bus.move_to_dead_letter(
            stream_name,
            entry_id,
            message,
            reason=reason,
        )

    def _set_activity(self, activity: str) -> None:
        """Update the current activity description for UI feedback."""
        self._current_activity = activity
        self._activity_since = time.monotonic()

    def _clear_activity(self) -> None:
        """Clear the current activity."""
        self._current_activity = ""
        self._activity_since = 0.0

    async def _handle_message(self, message: AgentMessage) -> None:
        """Route an incoming message to the agent-specific handler."""
        self._processing = True
        self.status = AgentStatus.THINKING
        self._set_activity(f"Processing {message.type.value} from {message.sender.value}")
        logger.info(
            "message_received",
            agent=self.role.value,
            sender=message.sender.value,
            type=message.type.value,
            project_id=message.project_id,
        )

        try:
            # Check if the project is paused — skip processing if so
            project = await self.db.get_project(message.project_id)
            if project and project.status == "paused":
                logger.info(
                    "message_skipped_project_paused",
                    agent=self.role.value,
                    project_id=message.project_id,
                )
                return

            # Persist for history
            await self.db.save_message(message.model_dump(mode="json"))

            # Dispatch to subclass
            response = await self.process_message(message)

            if response:
                await self.bus.publish(response)
                await self.db.save_message(response.model_dump(mode="json"))
                self.stats.messages_sent += 1
                record_agent_message_sent(self.role.value)

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
            if self.status is not AgentStatus.ERROR:
                self.status = AgentStatus.IDLE
            self._clear_activity()

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

    async def _append_history(self, project_id: str, role: str, content: str) -> None:
        """Append a turn to a project's history, auto-pruning old turns."""
        async with self._history_lock:
            history = self._project_histories.get(project_id)
            if history is None:
                history = []
                self._project_histories[project_id] = history
            history.append(ConversationEntry(role=role, content=content))

            # Auto-prune: keep first 2 turns (early context) + most recent turns
            if len(history) > self.MAX_HISTORY_PER_PROJECT:
                keep_early = 2
                overflow = len(history) - self.MAX_HISTORY_PER_PROJECT
                del history[keep_early : keep_early + overflow]

    async def clear_history(self, project_id: str | None = None) -> None:
        """Clear conversation history for one project or all projects."""
        async with self._history_lock:
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
        task_id: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Send a prompt to the LLM with system prompt and per-project history.

        Args:
            prompt: The user-role content to send.
            project_id: Isolates conversation history by project.  If empty,
                        uses a shared "_global" bucket (for non-project queries).
            task_id: Optional task ID for per-task token tracking.
            temperature: Sampling temperature (None = use per-agent default).
            top_p: Nucleus sampling parameter (None = use per-agent default).
            max_tokens: Max tokens in the response.

        Returns:
            The assistant's response text.

        Context window management is handled automatically by OllamaClient
        (auto_trim=True).  Older history entries are dropped first.
        """
        pid = project_id or "_global"
        self.status = AgentStatus.WORKING
        self._set_activity(f"Waiting for LLM response ({self._model})")

        # Build messages list
        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt}]

        # Per-project history (OllamaClient auto-trims if too long)
        for entry in self._get_history(pid):
            messages.append({"role": entry.role, "content": entry.content})

        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self.ollama.chat(
                    messages,
                    model=self._model,
                    temperature=temperature if temperature is not None else self._temperature,
                    top_p=top_p if top_p is not None else self._top_p,
                    max_tokens=max_tokens,
                    auto_trim=True,
                ),
                timeout=300,
            )
        except TimeoutError:
            self.stats.llm_errors += 1
            record_agent_llm_error(self.role.value)
            logger.error("agent_think_timeout", agent=self.role.value, timeout=300)
            self.status = AgentStatus.ERROR
            agent_status_metric.labels(agent=self.role.value).set(2)
            raise
        except Exception as e:
            self.stats.llm_errors += 1
            record_agent_llm_error(self.role.value)
            logger.error("agent_think_error", agent=self.role.value, error=str(e))
            self.status = AgentStatus.ERROR
            agent_status_metric.labels(agent=self.role.value).set(2)
            raise

        self.status = AgentStatus.IDLE
        agent_status_metric.labels(agent=self.role.value).set(0)
        elapsed = time.monotonic() - t0
        self.stats.llm_calls += 1
        self.stats.total_think_seconds += elapsed
        record_agent_llm_call(self.role.value)
        record_agent_llm_duration(self.role.value, elapsed)

        # Record in per-project history
        await self._append_history(pid, "user", prompt)
        await self._append_history(pid, "assistant", response)

        # Track token usage at the project level (via TaskManager)
        # Use real GenerationStats from the Ollama client if available.
        if self.task_manager and project_id:
            stats = self.ollama.last_stats
            if stats and stats.total_tokens > 0:
                total_tokens = stats.total_tokens
                cost = stats.cost_estimate_usd
            else:
                # Fallback: rough estimate when stats are unavailable (cache hit etc.)
                total_tokens = estimate_messages_tokens(messages) + estimate_tokens(response)
                cost = 0.0
            try:
                await self.task_manager.record_tokens(
                    project_id, task_id, total_tokens, cost_estimate=cost
                )
            except Exception as exc:
                logger.warning(
                    "token_tracking_failed",
                    agent=self.role.value,
                    project_id=project_id,
                    task_id=task_id,
                    error=str(exc),
                )

        return response

    async def think_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        *,
        project_id: str = "",
        task_id: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 4096,
    ) -> tuple[str, list[dict]]:
        """Send a prompt with tool definitions to the LLM."""
        pid = project_id or "_global"
        self.status = AgentStatus.WORKING
        self._set_activity(f"Waiting for LLM response ({self._model})")

        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt}]

        for entry in self._get_history(pid):
            messages.append({"role": entry.role, "content": entry.content})

        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        try:
            text_response, tool_calls = await asyncio.wait_for(
                self.ollama.chat_with_tools(
                    messages,
                    tools,
                    model=self._model,
                    temperature=temperature if temperature is not None else self._temperature,
                    top_p=top_p if top_p is not None else self._top_p,
                    max_tokens=max_tokens,
                    auto_trim=True,
                ),
                timeout=300,
            )
        except TimeoutError:
            self.stats.llm_errors += 1
            record_agent_llm_error(self.role.value)
            logger.error("agent_think_timeout", agent=self.role.value, timeout=300)
            self.status = AgentStatus.ERROR
            agent_status_metric.labels(agent=self.role.value).set(2)
            raise
        except Exception as e:
            self.stats.llm_errors += 1
            record_agent_llm_error(self.role.value)
            logger.error("agent_think_error", agent=self.role.value, error=str(e))
            self.status = AgentStatus.ERROR
            agent_status_metric.labels(agent=self.role.value).set(2)
            raise

        elapsed = time.monotonic() - t0
        self.stats.llm_calls += 1
        self.stats.total_think_seconds += elapsed
        record_agent_llm_call(self.role.value)
        record_agent_llm_duration(self.role.value, elapsed)

        await self._append_history(pid, "user", prompt)
        if text_response:
            await self._append_history(pid, "assistant", text_response)

        # Track token usage using real GenerationStats from OllamaClient.
        if self.task_manager and project_id:
            stats = self.ollama.last_stats
            if stats and stats.total_tokens > 0:
                total_tokens = stats.total_tokens
                cost = stats.cost_estimate_usd
            else:
                total_tokens = estimate_messages_tokens(messages) + (
                    estimate_tokens(text_response) if text_response else 0
                )
                cost = 0.0
            try:
                await self.task_manager.record_tokens(
                    project_id, task_id, total_tokens, cost_estimate=cost
                )
            except Exception as exc:
                logger.warning(
                    "token_tracking_failed",
                    agent=self.role.value,
                    project_id=project_id,
                    task_id=task_id,
                    error=str(exc),
                )

        return text_response, tool_calls

    # --------------------------------------------------------
    # Message Sending Helpers
    # --------------------------------------------------------

    async def _transition_task(self, task_id: str, new_state: TaskState) -> bool:
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
        activity_duration = 0.0
        if self._activity_since > 0:
            activity_duration = time.monotonic() - self._activity_since
        return {
            "role": self.role.value,
            "status": self.status.value,
            "model": self._model,
            "current_activity": self._current_activity,
            "activity_duration_seconds": round(activity_duration, 1),
            "active_projects": list(self._project_histories.keys()),
            "history_sizes": {pid: len(h) for pid, h in self._project_histories.items()},
            "stats": self.stats.snapshot(),
        }
