"""Project Manager Agent — orchestrates the AI development team.

Phase 2 responsibilities:
- Parse LLM project plans into persisted Task records
- Drive tasks through the state machine (assign → in_progress → review → done)
- Route Critic feedback to the correct agent for revision
- Kick off the next pending task when one completes
"""

from __future__ import annotations

import re

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
        task_manager: object | None = None,
    ) -> None:
        super().__init__(role=AgentRole.PM, bus=bus, ollama=ollama, db=db, task_manager=task_manager)

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
        is carried in ``message.project_id``.  The PM:

        1. Asks the LLM to decompose the idea into structured tasks.
        2. Parses the LLM output into ``Task`` records and persists them
           via ``TaskManager``.
        3. Assigns the first pending task to kick off the pipeline.
        """
        idea = message.payload.content
        project_id = message.project_id

        # Ask the LLM to decompose the idea into tasks
        prompt = f"""A user has submitted a new product idea for our AI development team to build.

PRODUCT IDEA:
{idea}

Please analyze this idea and create a detailed project plan.

OUTPUT FORMAT — you MUST use this exact format for each task:

TASK: <concise title>
AGENT: <research|spec|coder|critic>
PRIORITY: <low|normal|high|critical>
DEPENDS_ON: <comma-separated titles of tasks this depends on, or "none">
DESCRIPTION: <detailed description on one or more lines, until the next TASK: or end>

Example:
TASK: Investigate auth libraries
AGENT: research
PRIORITY: high
DEPENDS_ON: none
DESCRIPTION: Research the best authentication libraries for a Python REST API.
Compare JWT, session-based, and OAuth2 approaches.

TASK: Write API specification
AGENT: spec
PRIORITY: normal
DEPENDS_ON: Investigate auth libraries
DESCRIPTION: Based on the research results, write a detailed API specification.

Important rules:
- Start with research tasks, then specification, then coding tasks.
- The critic will review deliverables automatically — you do NOT need to create critic tasks.
- Each task must be self-contained and detailed enough for the assigned agent.
- Order tasks by dependency (earlier tasks should be done first).
- Typically a project needs 1–3 research tasks, 1–2 spec tasks, and 1–3 coding tasks.

Think step by step about what needs to be built and in what order."""

        response = await self.think(prompt, project_id=project_id)

        # Parse the LLM output into Task records
        tasks = self._parse_tasks(response, project_id)

        if not tasks and self.task_manager:
            # Fallback: create a single research task from the raw plan
            logger.warning(
                "pm_task_parse_fallback",
                project_id=project_id,
            )
            tasks = [
                await self.task_manager.create_task(
                    project_id=project_id,
                    title="Research and plan",
                    description=f"Research the following idea and produce a report:\n\n{idea}",
                    assigned_to=AgentRole.RESEARCH,
                ),
            ]
        elif self.task_manager:
            # Two-pass creation: first create all tasks, then resolve deps
            title_to_id: dict[str, str] = {}
            persisted: list[tuple[dict, Task]] = []
            for t in tasks:
                try:
                    created = await self.task_manager.create_task(
                        project_id=project_id,
                        title=str(t["title"]),
                        description=str(t["description"]),
                        assigned_to=AgentRole(str(t["agent"])),
                    )
                    title_to_id[str(t["title"]).lower()] = created.id
                    persisted.append((t, created))
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "pm_task_create_skipped",
                        title=t.get("title", "?"),
                        reason=str(exc),
                    )

            # Second pass: resolve depends_on titles → IDs
            for task_dict, task_obj in persisted:
                dep_titles = task_dict.get("depends_on_titles", [])
                if dep_titles and isinstance(dep_titles, list):
                    dep_ids = []
                    for dep_title in dep_titles:
                        dep_id = title_to_id.get(dep_title.lower())
                        if dep_id:
                            dep_ids.append(dep_id)
                    if dep_ids:
                        task_obj.depends_on = dep_ids
                        await self.db.update_task(task_obj)

            tasks = [task_obj for _, task_obj in persisted]  # type: ignore[assignment]

        # Kick off the first pending task
        return await self._assign_next_task(project_id)

    # ------------------------------------------------------------------
    # Task Parsing
    # ------------------------------------------------------------------

    # Regex for the structured TASK: blocks from the LLM
    _TASK_HEADER_RE = re.compile(
        r"^TASK:\s*(.+)$", re.IGNORECASE | re.MULTILINE
    )
    _AGENT_RE = re.compile(r"^AGENT:\s*(\w+)", re.IGNORECASE | re.MULTILINE)
    _PRIORITY_RE = re.compile(
        r"^PRIORITY:\s*(\w+)", re.IGNORECASE | re.MULTILINE
    )
    _DEPENDS_ON_RE = re.compile(
        r"^DEPENDS_ON:\s*(.+)$", re.IGNORECASE | re.MULTILINE
    )
    _DESCRIPTION_RE = re.compile(
        r"^DESCRIPTION:\s*(.*)", re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    _VALID_AGENTS = {"research", "spec", "coder"}

    def _parse_tasks(
        self, response: str, project_id: str
    ) -> list[dict[str, str | list[str]]]:
        """Parse the LLM's structured output into task dicts.

        Returns a list of ``{"title", "agent", "priority", "description",
        "depends_on"}`` dicts.  Skips malformed blocks.
        """
        # Split the response on TASK: headers
        parts = re.split(r"(?=^TASK:)", response, flags=re.IGNORECASE | re.MULTILINE)
        tasks: list[dict[str, str | list[str]]] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            title_m = self._TASK_HEADER_RE.search(part)
            agent_m = self._AGENT_RE.search(part)
            if not title_m or not agent_m:
                continue

            agent = agent_m.group(1).strip().lower()
            if agent not in self._VALID_AGENTS:
                continue

            priority_m = self._PRIORITY_RE.search(part)
            priority = (
                priority_m.group(1).strip().lower()
                if priority_m
                else "normal"
            )
            if priority not in {"low", "normal", "high", "critical"}:
                priority = "normal"

            # Optional DEPENDS_ON: comma-separated task titles
            depends_on_m = self._DEPENDS_ON_RE.search(part)
            depends_on_titles: list[str] = []
            if depends_on_m:
                raw = depends_on_m.group(1).strip()
                if raw.lower() != "none":
                    depends_on_titles = [
                        t.strip() for t in raw.split(",") if t.strip()
                    ]

            # Everything after DESCRIPTION: (up to the next TASK: which
            # we already split on)
            desc_m = self._DESCRIPTION_RE.search(part)
            description = desc_m.group(1).strip() if desc_m else title_m.group(1).strip()

            tasks.append({
                "title": title_m.group(1).strip(),
                "agent": agent,
                "priority": priority,
                "description": description,
                "depends_on_titles": depends_on_titles,
            })

        logger.info(
            "pm_tasks_parsed",
            project_id=project_id,
            count=len(tasks),
        )
        return tasks

    async def _handle_deliverable(self, message: AgentMessage) -> AgentMessage | None:
        """Handle a completed deliverable from an agent.

        State machine transitions:
        - Critic approved  → REVIEW → APPROVED → DONE, then assign next task
        - Critic rejected  → REVIEW → REVISION, then route feedback
        - Other agents     → advance via standard pipeline
        """
        review_result = message.payload.metadata.get("review_result", "")
        task_id = message.task_id

        # --- Critic approved → close the task and start the next one ---
        if message.sender == AgentRole.CRITIC and review_result == "approved":
            logger.info(
                "pm_deliverable_approved",
                project_id=message.project_id,
                task_id=task_id,
            )
            # REVIEW → APPROVED → DONE
            await self._transition_task(task_id, TaskState.APPROVED)
            await self._transition_task(task_id, TaskState.DONE)

            # Kick off the next pending task for this project
            return await self._assign_next_task(message.project_id)

        # --- Critic rejected → revision cycle ---
        if message.sender == AgentRole.CRITIC and review_result == "needs_revision":
            return await self._handle_feedback(message)

        # --- Standard pipeline progression (non-critic deliverables) ---
        # The agent just submitted work → transition to REVIEW
        await self._transition_task(task_id, TaskState.REVIEW)

        next_agent = self._determine_next_agent(message.sender)
        if next_agent is None:
            return None

        prompt = f"""An agent has submitted a deliverable.

FROM: {message.sender.value} agent
DELIVERABLE (summary):
{message.payload.content[:2000]}

Provide any additional context or instructions for the {next_agent.value} agent
who will work on this next.  Be concise."""

        response = await self.think(prompt, project_id=message.project_id, task_id=message.task_id)

        return self.create_message(
            recipient=next_agent,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=message.project_id,
            task_id=task_id,
            content=f"{response}\n\n--- PREVIOUS DELIVERABLE ---\n{message.payload.content}",
            metadata=message.payload.metadata,
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
        Also transitions the task to REVISION if it's currently in REVIEW.
        """
        artifact_type = message.payload.metadata.get("artifact_type", "")
        task_id = message.task_id

        # Transition task to REVISION (safe — returns False if invalid)
        await self._transition_task(task_id, TaskState.REVISION)

        # Determine who should address the feedback
        revision_target = self._feedback_target(artifact_type, message.sender)

        prompt = f"""The Critic agent has provided feedback:

{message.payload.content}

Based on this feedback, provide clear instructions for the {revision_target.value} agent
who will make the revisions. Be specific about what needs to change."""

        response = await self.think(prompt, project_id=message.project_id, task_id=message.task_id)

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

        response = await self.think(prompt, project_id=message.project_id, task_id=message.task_id)

        return self.create_message(
            recipient=message.sender,
            msg_type=MessageType.ANSWER,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )

    # ------------------------------------------------------------------
    # Task Assignment
    # ------------------------------------------------------------------

    async def _assign_next_task(
        self, project_id: str
    ) -> AgentMessage | None:
        """Find assignable tasks and dispatch them.

        Uses dependency-aware ordering via ``TaskManager.get_assignable_tasks``.
        Can dispatch multiple independent tasks in parallel.

        Returns the first assignment message (additional assignments are
        published directly via ``self.send``).  Returns ``None`` when there
        are no more assignable tasks.
        """
        if not self.task_manager:
            return None

        assignable = await self.task_manager.get_assignable_tasks(project_id)
        if not assignable:
            logger.info("pm_no_assignable_tasks", project_id=project_id)

            # Check if all tasks are done → mark project complete
            progress = await self.task_manager.get_project_progress(project_id)
            if progress["total_tasks"] > 0 and progress["completed"] == progress["total_tasks"]:
                await self.db.update_project(project_id, status="completed")
                logger.info("pm_project_completed", project_id=project_id)

            return None

        first_msg: AgentMessage | None = None

        for task in assignable:
            assigned_to = task.assigned_to or AgentRole.RESEARCH

            # Transition PENDING → ASSIGNED
            await self._transition_task(task.id, TaskState.ASSIGNED)

            msg = self.create_message(
                recipient=assigned_to,
                msg_type=MessageType.TASK_ASSIGNMENT,
                project_id=project_id,
                task_id=task.id,
                content=task.description,
                priority=task.priority,
                metadata={"task_title": task.title},
            )

            if first_msg is None:
                first_msg = msg
            else:
                # Publish additional parallel assignments directly
                await self.bus.publish(msg)
                await self.db.save_message(msg.model_dump(mode="json"))
                self.stats.messages_sent += 1

        return first_msg

    # ------------------------------------------------------------------
    # Pipeline Helpers
    # ------------------------------------------------------------------

    def _determine_next_agent(self, current: AgentRole) -> AgentRole | None:
        """Determine the next agent in the pipeline."""
        pipeline = {
            AgentRole.RESEARCH: AgentRole.SPEC,
            AgentRole.SPEC: AgentRole.CODER,
            AgentRole.CODER: AgentRole.CRITIC,
            AgentRole.CRITIC: AgentRole.PM,
        }
        return pipeline.get(current)
