"""Specification Agent — writes detailed technical specifications."""

from __future__ import annotations

import structlog

from src.agents.base_agent import BaseAgent
from src.core.database import Database
from src.core.message_bus import MessageBus
from src.core.models import (
    AgentMessage,
    AgentRole,
    MessageType,
)
from src.core.ollama_client import OllamaClient

logger = structlog.get_logger()


class SpecAgent(BaseAgent):
    """The Specification agent.

    Responsibilities:
    - Translate research and ideas into detailed technical specifications
    - Define API contracts, data models, and architecture
    - Produce structured documents the Coder agent can implement from
    - Revise specs based on feedback
    """

    def __init__(
        self,
        *,
        bus: MessageBus,
        ollama: OllamaClient,
        db: Database,
    ) -> None:
        super().__init__(role=AgentRole.SPEC, bus=bus, ollama=ollama, db=db)

    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        """Handle incoming messages."""
        match message.type:
            case MessageType.TASK_ASSIGNMENT:
                return await self._handle_spec_task(message)
            case MessageType.FEEDBACK:
                return await self._handle_feedback(message)
            case MessageType.QUESTION:
                return await self._handle_question(message)
            case _:
                logger.debug("spec_skipping_message", type=message.type)
                return None

    async def _handle_spec_task(self, message: AgentMessage) -> AgentMessage | None:
        """Write technical specifications based on research and PM direction."""
        prompt = f"""You have been assigned a specification task. Use the following context to write a detailed technical specification.

CONTEXT FROM PM / RESEARCH:
{message.payload.content}

Write a comprehensive technical specification document that includes:

1. **Overview**: Brief summary of what is being built and why.

2. **Architecture**:
   - System architecture diagram (describe in text/ASCII)
   - Component breakdown
   - Technology stack with versions

3. **Data Models**:
   - All entities with their fields, types, and relationships
   - Database schema (SQL or equivalent)

4. **API Specification**:
   - All endpoints with HTTP method, path, request/response schemas
   - Authentication/authorization approach
   - Error handling patterns

5. **File Structure**:
   - Complete project directory layout
   - Purpose of each file/module

6. **Implementation Notes**:
   - Key algorithms or logic flows
   - Third-party integrations
   - Configuration requirements
   - Environment variables needed

7. **Testing Strategy**:
   - What should be tested
   - Test categories (unit, integration, e2e)

Be extremely specific. The Coder agent will implement directly from this spec.
Use code blocks for schemas, API definitions, and file structures.
Do NOT write the actual implementation code — only the specification."""

        response = await self.think(prompt, project_id=message.project_id, temperature=0.5)

        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
            metadata={"artifact_type": "specification"},
        )

    async def _handle_feedback(self, message: AgentMessage) -> AgentMessage | None:
        """Revise specifications based on feedback."""
        prompt = f"""Your specification received feedback that needs to be addressed:

FEEDBACK:
{message.payload.content}

Please revise the specification to address all feedback points. Output the complete updated specification."""

        response = await self.think(prompt, project_id=message.project_id, temperature=0.5)

        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
            metadata={"artifact_type": "specification", "revision": "true"},
        )

    async def _handle_question(self, message: AgentMessage) -> AgentMessage | None:
        """Answer questions about the specification."""
        prompt = f"""The {message.sender.value} agent has a question about the specification:

{message.payload.content}

Provide a clear, detailed answer referencing the relevant parts of the spec."""

        response = await self.think(prompt, project_id=message.project_id, temperature=0.5)

        return self.create_message(
            recipient=message.sender,
            msg_type=MessageType.ANSWER,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )
