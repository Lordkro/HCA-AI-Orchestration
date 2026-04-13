"""Research Agent — investigates technologies and provides context."""

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


class ResearchAgent(BaseAgent):
    """The Research agent.

    Responsibilities:
    - Investigate technologies, libraries, and patterns
    - Analyze feasibility of requested features
    - Provide context and recommendations
    - Synthesize findings into actionable reports
    """

    def __init__(
        self,
        *,
        bus: MessageBus,
        ollama: OllamaClient,
        db: Database,
    ) -> None:
        super().__init__(role=AgentRole.RESEARCH, bus=bus, ollama=ollama, db=db)

    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        """Handle incoming messages."""
        match message.type:
            case MessageType.TASK_ASSIGNMENT:
                return await self._handle_research_task(message)
            case MessageType.QUESTION:
                return await self._handle_question(message)
            case MessageType.FEEDBACK:
                return await self._handle_feedback(message)
            case _:
                logger.debug("research_skipping_message", type=message.type)
                return None

    async def _handle_research_task(self, message: AgentMessage) -> AgentMessage | None:
        """Conduct research based on the assigned task."""
        prompt = f"""You have been assigned a research task by the Project Manager.

TASK/CONTEXT:
{message.payload.content}

Please conduct thorough research and provide a detailed report covering:

1. **Technology Analysis**: What technologies, frameworks, and libraries are best suited for this project? Explain why.
2. **Architecture Recommendations**: What architecture patterns should be used? (e.g., monolith, microservices, event-driven)
3. **Data Model Considerations**: What are the key data entities and relationships?
4. **Potential Challenges**: What are the likely technical challenges and how to address them?
5. **Best Practices**: What best practices should the team follow?
6. **Estimated Complexity**: How complex is this project? What are the main risk areas?

Be specific and actionable. The Specification Agent will use your report to write detailed technical specs.
Format your output clearly with sections and bullet points."""

        response = await self.think(prompt, temperature=0.6)

        # Send research report to PM (who will route to Spec agent)
        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
            metadata={"artifact_type": "research_report"},
        )

    async def _handle_question(self, message: AgentMessage) -> AgentMessage | None:
        """Answer a specific research question."""
        prompt = f"""Another agent has a research question:

FROM: {message.sender.value}
QUESTION: {message.payload.content}

Provide a thorough, well-reasoned answer with specific recommendations."""

        response = await self.think(prompt, temperature=0.5)

        return self.create_message(
            recipient=message.sender,
            msg_type=MessageType.ANSWER,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )

    async def _handle_feedback(self, message: AgentMessage) -> AgentMessage | None:
        """Revise research based on feedback."""
        prompt = f"""Your previous research report received feedback:

FEEDBACK:
{message.payload.content}

Please revise and improve your research based on this feedback. Address all points raised."""

        response = await self.think(prompt, temperature=0.6)

        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
            metadata={"artifact_type": "research_report", "revision": "true"},
        )
