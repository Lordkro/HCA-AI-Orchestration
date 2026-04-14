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
        """Break down a new product idea into a project plan.

        The project has already been created by the API route and its ID
        is carried in ``message.project_id``.  The PM agent's job is to
        decompose the idea into a plan and kick off the pipeline.
        """
        idea = message.payload.content
        project_id = message.project_id

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

        response = await self.think(prompt, project_id=project_id)

        # Send the plan to the research agent to begin
        return self.create_message(
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project_id,
            content=response,
            priority=Priority.HIGH,
            metadata={"original_idea": idea},
        )

    async def _handle_deliverable(self, message: AgentMessage) -> AgentMessage | None:
        """Handle a completed deliverable from an agent.

        Uses metadata from the Critic to decide whether to approve or
        route back for revision.  For other agents the standard pipeline
        order is followed: Research → Spec → Coder → Critic.
        """
        review_result = message.payload.metadata.get("review_result", "")

        # --- Critic approved → mark done / advance ---
        if message.sender == AgentRole.CRITIC and review_result == "approved":
            logger.info(
                "pm_deliverable_approved",
                project_id=message.project_id,
                task_id=message.task_id,
            )
            # Notify all agents that the task is complete
            return self.create_message(
                recipient=AgentRole.RESEARCH,
                msg_type=MessageType.STATUS_UPDATE,
                project_id=message.project_id,
                task_id=message.task_id,
                content=f"Task approved by Critic. Output:\n\n{message.payload.content}",
                metadata={"status": "approved"},
            )

        # --- Critic rejected → route feedback to the appropriate agent ---
        if message.sender == AgentRole.CRITIC and review_result == "needs_revision":
            return await self._handle_feedback(message)

        # --- Standard pipeline progression (non-critic deliverables) ---
        next_agent = self._determine_next_agent(message.sender)
        if next_agent is None:
            return None

        prompt = f"""An agent has submitted a deliverable.

FROM: {message.sender.value} agent
DELIVERABLE (summary):
{message.payload.content[:2000]}

Provide any additional context or instructions for the {next_agent.value} agent
who will work on this next.  Be concise."""

        response = await self.think(prompt, project_id=message.project_id)

        # Choose the right message type for the next hop
        msg_type = MessageType.TASK_ASSIGNMENT

        return self.create_message(
            recipient=next_agent,
            msg_type=msg_type,
            project_id=message.project_id,
            task_id=message.task_id,
            content=f"{response}\n\n--- PREVIOUS DELIVERABLE ---\n{message.payload.content}",
        )

    async def _handle_status_update(self, message: AgentMessage) -> AgentMessage | None:
        """Handle status updates from agents."""
        logger.info(
            "pm_status_update",
            sender=message.sender.value,
            content=message.payload.content[:100],
        )
        return None

    async def _handle_feedback(self, message: AgentMessage) -> AgentMessage | None:
        """Handle feedback (usually from Critic) and route to the right agent.

        Determines which agent should receive the revision request based on
        the artifact type in the metadata.  Falls back to CODER if unknown.
        """
        artifact_type = message.payload.metadata.get("artifact_type", "")

        # Determine who should address the feedback
        revision_target = self._feedback_target(artifact_type, message.sender)

        prompt = f"""The Critic agent has provided feedback:

{message.payload.content}

Based on this feedback, provide clear instructions for the {revision_target.value} agent
who will make the revisions. Be specific about what needs to change."""

        response = await self.think(prompt, project_id=message.project_id)

        return self.create_message(
            recipient=revision_target,
            msg_type=MessageType.FEEDBACK,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )

    @staticmethod
    def _feedback_target(artifact_type: str, sender: AgentRole) -> AgentRole:
        """Decide which agent should receive revision feedback."""
        type_to_agent = {
            "code": AgentRole.CODER,
            "specification": AgentRole.SPEC,
            "research_report": AgentRole.RESEARCH,
        }
        return type_to_agent.get(artifact_type, AgentRole.CODER)

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
