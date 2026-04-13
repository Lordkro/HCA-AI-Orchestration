"""Base agent class — shared behavior for all HCA agents."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

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
)
from src.core.ollama_client import OllamaClient

logger = structlog.get_logger()


class BaseAgent(ABC):
    """Abstract base class for all HCA agents.

    Provides:
    - Message listening loop (inbox consumption)
    - LLM interaction via Ollama
    - Conversation memory management
    - Message sending helpers
    - Lifecycle management (start/stop)
    """

    def __init__(
        self,
        *,
        role: AgentRole,
        bus: MessageBus,
        ollama: OllamaClient,
        db: Database,
    ) -> None:
        self.role = role
        self.bus = bus
        self.ollama = ollama
        self.db = db
        self.status = AgentStatus.STOPPED
        self._running = False
        self._conversation_history: list[ConversationEntry] = []
        self._system_prompt: str = ""
        self._model: str = settings.get_agent_model(role.value)

        # Load system prompt from file
        self._load_system_prompt()

    def _load_system_prompt(self) -> None:
        """Load the system prompt from the prompts directory."""
        prompt_file = Path(__file__).parent.parent / "prompts" / f"{self.role.value}.txt"
        if prompt_file.exists():
            self._system_prompt = prompt_file.read_text(encoding="utf-8").strip()
            logger.info("prompt_loaded", agent=self.role.value, file=str(prompt_file))
        else:
            self._system_prompt = f"You are the {self.role.value} agent in an AI development team."
            logger.warning("prompt_default", agent=self.role.value, file=str(prompt_file))

    async def start(self) -> None:
        """Start the agent's message processing loop."""
        self._running = True
        self.status = AgentStatus.IDLE
        logger.info("agent_started", agent=self.role.value, model=self._model)

        last_id = "$"  # Only read new messages
        while self._running:
            try:
                messages = await self.bus.consume(
                    self.role, last_id=last_id, block_ms=2000
                )
                for msg in messages:
                    # Skip messages sent by self
                    if msg.sender == self.role:
                        continue
                    await self._handle_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("agent_error", agent=self.role.value, error=str(e))
                self.status = AgentStatus.ERROR
                await asyncio.sleep(5)  # Back off on error

        self.status = AgentStatus.STOPPED
        logger.info("agent_stopped", agent=self.role.value)

    async def stop(self) -> None:
        """Signal the agent to stop processing."""
        self._running = False

    async def _handle_message(self, message: AgentMessage) -> None:
        """Route an incoming message to the appropriate handler."""
        self.status = AgentStatus.THINKING
        logger.info(
            "message_received",
            agent=self.role.value,
            sender=message.sender,
            type=message.type,
            project_id=message.project_id,
        )

        try:
            # Save message to DB for history
            await self.db.save_message(message.model_dump(mode="json"))

            # Dispatch to the agent's specific handler
            response = await self.process_message(message)

            if response:
                await self.bus.publish(response)
                await self.db.save_message(response.model_dump(mode="json"))

        except Exception as e:
            logger.error(
                "message_handling_error",
                agent=self.role.value,
                error=str(e),
                message_id=message.id,
            )
        finally:
            self.status = AgentStatus.IDLE

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process an incoming message. Must be implemented by each agent.

        Returns an AgentMessage to send as a response, or None if no response.
        """
        ...

    # --------------------------------------------------------
    # LLM Interaction Helpers
    # --------------------------------------------------------

    async def think(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Send a prompt to the LLM with the agent's system prompt and history.

        Context window management is handled automatically by OllamaClient.
        If conversation history is too long, older messages are trimmed.
        """
        self.status = AgentStatus.WORKING

        # Build the chat messages
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add conversation history (OllamaClient will auto-trim if needed)
        for entry in self._conversation_history:
            messages.append({"role": entry.role, "content": entry.content})

        # Add the current prompt
        messages.append({"role": "user", "content": prompt})

        # Call Ollama (auto_trim=True handles context overflow)
        try:
            response = await self.ollama.chat(
                messages,
                model=self._model,
                temperature=temperature,
                max_tokens=max_tokens,
                auto_trim=True,
            )
        except Exception as e:
            logger.error("agent_think_error", agent=self.role.value, error=str(e))
            self.status = AgentStatus.ERROR
            raise

        # Update conversation history
        self._conversation_history.append(
            ConversationEntry(role="user", content=prompt)
        )
        self._conversation_history.append(
            ConversationEntry(role="assistant", content=response)
        )

        self.status = AgentStatus.IDLE
        return response

    def clear_history(self) -> None:
        """Clear the agent's conversation history."""
        self._conversation_history.clear()

    # --------------------------------------------------------
    # Message Sending Helpers
    # --------------------------------------------------------

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
            priority=priority,
        )
        await self.bus.publish(msg)
        await self.db.save_message(msg.model_dump(mode="json"))
