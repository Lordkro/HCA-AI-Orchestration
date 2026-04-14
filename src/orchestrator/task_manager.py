"""Task state machine and management.

Integrates with Guardrails to enforce safety limits on iteration
counts, task counts per project, task timeouts, and token budgets.

Phase 3 additions:
- Dependency-aware task ordering (``depends_on``)
- Parallel task dispatch (get multiple assignable tasks)
- Per-project token budget tracking
- Escalation: PM is notified when a task fails due to guardrails
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from src.core.database import Database
from src.core.message_bus import MessageBus
from src.core.models import (
    AgentMessage,
    AgentRole,
    MessagePayload,
    MessageType,
    Task,
    TaskState,
)
from src.orchestrator.guardrails import Guardrails

logger = structlog.get_logger()

# Valid state transitions
VALID_TRANSITIONS: dict[TaskState, list[TaskState]] = {
    TaskState.PENDING: [TaskState.ASSIGNED],
    TaskState.ASSIGNED: [TaskState.IN_PROGRESS, TaskState.FAILED],
    TaskState.IN_PROGRESS: [TaskState.REVIEW, TaskState.FAILED],
    TaskState.REVIEW: [TaskState.APPROVED, TaskState.REVISION],
    TaskState.REVISION: [TaskState.IN_PROGRESS, TaskState.FAILED],
    TaskState.APPROVED: [TaskState.DONE],
    TaskState.DONE: [],
    TaskState.FAILED: [TaskState.PENDING],  # Allow retry
}


class TaskManager:
    """Manages task lifecycle and state transitions.

    Uses :class:`Guardrails` to enforce per-task iteration limits,
    per-project task count caps, task timeouts, and token budgets.
    """

    def __init__(
        self,
        *,
        db: Database,
        bus: MessageBus,
        guardrails: Guardrails | None = None,
    ) -> None:
        self.db = db
        self.bus = bus
        self.guardrails = guardrails or Guardrails()

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    async def create_task(
        self,
        *,
        project_id: str,
        title: str,
        description: str,
        assigned_to: AgentRole | None = None,
        max_iterations: int = 5,
        depends_on: list[str] | None = None,
    ) -> Task:
        """Create a new task in the pending state.

        Raises ValueError if the project has reached its task limit.
        """
        # Guardrail: check task count
        current_count = await self.db.count_tasks(project_id)
        if not self.guardrails.check_task_limit(current_count):
            raise ValueError(
                f"Project {project_id} has reached the maximum task limit "
                f"({self.guardrails.max_tasks})"
            )

        task = Task(
            project_id=project_id,
            title=title,
            description=description,
            assigned_to=assigned_to,
            max_iterations=max_iterations,
            depends_on=depends_on or [],
        )
        await self.db.create_task(task)
        logger.info("task_created", task_id=task.id, title=title)
        return task

    # ------------------------------------------------------------------
    # State Transitions
    # ------------------------------------------------------------------

    async def transition(self, task_id: str, new_state: TaskState) -> Task:
        """Transition a task to a new state with validation."""
        task = await self.db.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        valid_next = VALID_TRANSITIONS.get(task.state, [])
        if new_state not in valid_next:
            raise ValueError(
                f"Invalid transition: {task.state.value} → {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid_next]}"
            )

        old_state = task.state
        task.state = new_state
        task.updated_at = datetime.now(timezone.utc)

        # Handle iteration counting for revision cycles
        if new_state == TaskState.REVISION:
            task.iteration += 1
            if not self.guardrails.should_allow_revision(task):
                logger.warning(
                    "task_guardrail_failed",
                    task_id=task.id,
                    iterations=task.iteration,
                )
                task.state = TaskState.FAILED
                task.feedback = "Guardrail triggered: maximum iterations or timeout reached"
                # Escalate to PM
                await self._escalate_to_pm(task, reason="max_iterations")

        await self.db.update_task(task)

        logger.info(
            "task_transitioned",
            task_id=task.id,
            from_state=old_state.value,
            to_state=task.state.value,
            iteration=task.iteration,
        )

        # Publish state change event for UI
        await self.bus.publish_ui_event("task_state_changed", {
            "task_id": task.id,
            "project_id": task.project_id,
            "old_state": old_state.value,
            "new_state": task.state.value,
            "iteration": task.iteration,
        })

        return task

    # ------------------------------------------------------------------
    # Dependency-Aware Task Queries
    # ------------------------------------------------------------------

    async def get_assignable_tasks(
        self, project_id: str, *, limit: int | None = None
    ) -> list[Task]:
        """Return PENDING tasks whose dependencies are all satisfied.

        A dependency is satisfied when the depended-on task has reached
        a terminal state (DONE).  FAILED dependencies block the task.

        Args:
            project_id: The project to query.
            limit: Max tasks to return (defaults to config max_parallel_tasks).

        Returns:
            List of PENDING tasks that can be assigned right now.
        """
        from src.core.config import settings as _settings
        cap = limit if limit is not None else _settings.max_parallel_tasks

        pending = await self.db.list_tasks(project_id, state=TaskState.PENDING)
        if not pending:
            return []

        # Build set of completed task IDs (only DONE counts as satisfied)
        all_tasks = await self.db.list_tasks(project_id)
        done_ids = {t.id for t in all_tasks if t.state == TaskState.DONE}

        assignable: list[Task] = []
        for task in pending:
            if not task.depends_on:
                assignable.append(task)
            elif all(dep_id in done_ids for dep_id in task.depends_on):
                assignable.append(task)
            if len(assignable) >= cap:
                break

        return assignable

    # ------------------------------------------------------------------
    # Token Budget
    # ------------------------------------------------------------------

    async def record_tokens(
        self, project_id: str, task_id: str, tokens: int
    ) -> bool:
        """Record tokens consumed by an LLM call.

        Updates both the task-level and project-level counters.
        Returns True if still within budget, False if budget is exceeded.
        """
        if tokens <= 0:
            return True

        # Update task-level counter
        task = await self.db.get_task(task_id)
        if task:
            task.tokens_used += tokens
            await self.db.update_task(task)

        # Update project-level counter
        new_total = await self.db.add_project_tokens(project_id, tokens)

        within_budget = self.guardrails.check_token_budget(new_total)
        if not within_budget:
            logger.warning(
                "project_token_budget_exceeded",
                project_id=project_id,
                tokens_used=new_total,
                budget=self.guardrails.project_token_budget,
            )
        return within_budget

    async def get_project_token_usage(self, project_id: str) -> dict:
        """Get token usage summary for a project."""
        project_tokens = await self.db.get_project_tokens(project_id)
        return {
            "tokens_used": project_tokens,
            "budget": self.guardrails.project_token_budget,
            "remaining": max(0, self.guardrails.project_token_budget - project_tokens),
            "pct_used": round(
                project_tokens / self.guardrails.project_token_budget * 100, 1
            )
            if self.guardrails.project_token_budget > 0
            else 0,
        }

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    async def _escalate_to_pm(self, task: Task, *, reason: str) -> None:
        """Send the PM a notification about a task that hit a guardrail.

        This allows the PM to decide whether to retry, skip, or re-plan.
        """
        content = (
            f"⚠️ Task guardrail triggered on '{task.title}' (ID: {task.id}).\n"
            f"Reason: {reason}\n"
            f"Iterations: {task.iteration}/{task.max_iterations}\n"
            f"The task has been marked FAILED.  You may choose to:\n"
            f"1. Retry the task (transition FAILED → PENDING)\n"
            f"2. Skip it and continue with the next task\n"
            f"3. Re-plan the project with a different approach"
        )
        msg = AgentMessage(
            sender=AgentRole.SYSTEM,
            recipient=AgentRole.PM,
            type=MessageType.STATUS_UPDATE,
            project_id=task.project_id,
            task_id=task.id,
            payload=MessagePayload(
                content=content,
                metadata={"escalation_reason": reason},
            ),
        )
        await self.bus.publish(msg)
        logger.info(
            "task_escalated_to_pm",
            task_id=task.id,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    async def get_project_progress(self, project_id: str) -> dict:
        """Get a summary of task progress for a project."""
        tasks = await self.db.list_tasks(project_id)
        total = len(tasks)
        by_state: dict[str, int] = {}
        for task in tasks:
            by_state[task.state.value] = by_state.get(task.state.value, 0) + 1

        done = by_state.get("done", 0)
        token_usage = await self.get_project_token_usage(project_id)

        return {
            "total_tasks": total,
            "completed": done,
            "progress_pct": round(done / total * 100, 1) if total > 0 else 0,
            "by_state": by_state,
            "token_usage": token_usage,
        }
