"""Project Manager Agent — orchestrates the AI development team."""

from __future__ import annotations

import structlog

from src.agents.base_agent import BaseAgent
from src.core.database import Database
from src.core.message_bus import MessageBus
from src.core.models import (
    AgentMessage,
    AgentRole,
    MessageType,
    Priority,
    Project,
    Task,
    TaskState,
)
from src.core.ollama_client import OllamaClient

logger = structlog.get_logger()


class PMAgent(BaseAgent):
    """The Project Manager agent.

    Responsibilities:
    - Receive product ideas and decompose them into tasks
    - Assign tasks to appropriate agents
    - Track progress and make workflow decisions
    - Manage the project lifecycle
    """

    def __init__(
        self,
        *,
        bus: MessageBus,
        ollama: OllamaClient,
        db: Database,
    ) -> None:
        super().__init__(role=AgentRole.PM, bus=bus, ollama=ollama, db=db)

    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        """Handle incoming messages based on type."""
        match message.type:
            case MessageType.SYSTEM:
                # New project idea from user
                return await self._handle_new_project(message)
            case MessageType.DELIVERABLE:
                # An agent completed a task
                return await self._handle_deliverable(message)
            case MessageType.STATUS_UPDATE:
                return await self._handle_status_update(message)
            case MessageType.FEEDBACK:
                return await self._handle_feedback(message)
            case MessageType.QUESTION:
                return await self._handle_question(message)
            case _:
                logger.warning(
                    "pm_unhandled_message_type",
                    type=message.type,
                )
                return None

    async def _handle_new_project(self, message: AgentMessage) -> AgentMessage | None:
        """Break down a new product idea into a project plan."""
        idea = message.payload.content

        # Create the project in the database
        project = Project(
            name=f"Project from idea",
            description=idea,
            idea=idea,
        )
        await self.db.create_project(project)

        # Ask the LLM to decompose the idea into tasks
        prompt = f"""A user has submitted a new product idea for our AI development team to build.

PRODUCT IDEA:
{idea}

Please analyze this idea and create a detailed project plan. Break it down into specific, actionable tasks that our team can execute. For each task, specify:

1. Task title (concise)
2. Task description (detailed enough for the assigned agent to work independently)
3. Which agent should handle it: research, spec, coder, or critic
4. Dependencies (which tasks must complete first)
5. Priority: low, normal, high, or critical

Output your plan as a structured list. Start with research tasks, then specification tasks, then coding tasks. The critic will review deliverables automatically.

Remember our team:
- Research Agent: investigates technologies, patterns, feasibility
- Spec Agent: writes detailed technical specifications, API contracts, data models
- Coder Agent: implements the code based on specifications
- Critic Agent: reviews all outputs for quality

Think step by step about what needs to be built and in what order."""

        response = await self.think(prompt, project_id=project.id)

        # Send the plan to the research agent to begin
        return self.create_message(
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            content=response,
            priority=Priority.HIGH,
            metadata={"original_idea": idea},
        )

    async def _handle_deliverable(self, message: AgentMessage) -> AgentMessage | None:
        """Handle a completed deliverable from an agent."""
        prompt = f"""An agent has submitted a deliverable for review.

FROM: {message.sender.value} agent
TASK ID: {message.task_id}
DELIVERABLE:
{message.payload.content}

As the Project Manager, decide the next step:
1. If this is from the Research or Spec agent, route the output to the next agent in the pipeline.
2. If this is from the Coder agent, send it to the Critic for review.
3. If this is from the Critic agent with approval, mark the task as done.
4. If this is from the Critic agent with rejection, send feedback back to the appropriate agent.

What should happen next? Specify the recipient agent and your instructions."""

        response = await self.think(prompt, project_id=message.project_id)

        # Route to the next agent in the pipeline
        next_agent = self._determine_next_agent(message.sender)
        if next_agent:
            return self.create_message(
                recipient=next_agent,
                msg_type=MessageType.TASK_ASSIGNMENT,
                project_id=message.project_id,
                task_id=message.task_id,
                content=response,
            )
        return None

    async def _handle_status_update(self, message: AgentMessage) -> AgentMessage | None:
        """Handle status updates from agents."""
        logger.info(
            "pm_status_update",
            sender=message.sender.value,
            content=message.payload.content[:100],
        )
        return None

    async def _handle_feedback(self, message: AgentMessage) -> AgentMessage | None:
        """Handle feedback (usually from Critic)."""
        prompt = f"""The Critic agent has provided feedback:

{message.payload.content}

Based on this feedback, decide what action to take and who should address it.
Provide clear instructions for the next agent."""

        response = await self.think(prompt, project_id=message.project_id)

        # Route feedback to the appropriate agent
        return self.create_message(
            recipient=AgentRole.CODER,  # Usually goes back to coder
            msg_type=MessageType.FEEDBACK,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )

    async def _handle_question(self, message: AgentMessage) -> AgentMessage | None:
        """Answer questions from other agents."""
        prompt = f"""The {message.sender.value} agent has a question:

{message.payload.content}

Please provide a clear, decisive answer to help them proceed."""

        response = await self.think(prompt, project_id=message.project_id)

        return self.create_message(
            recipient=message.sender,
            msg_type=MessageType.ANSWER,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )

    def _determine_next_agent(self, current: AgentRole) -> AgentRole | None:
        """Determine the next agent in the pipeline."""
        pipeline = {
            AgentRole.RESEARCH: AgentRole.SPEC,
            AgentRole.SPEC: AgentRole.CODER,
            AgentRole.CODER: AgentRole.CRITIC,
            AgentRole.CRITIC: AgentRole.PM,
        }
        return pipeline.get(current)
