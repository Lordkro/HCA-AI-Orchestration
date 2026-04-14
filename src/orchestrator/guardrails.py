"""Guardrails — safety limits and controls for agent behavior.

Provides configurable limits to prevent:
- Infinite revision loops (per-task iteration cap)
- Unbounded task creation (per-project task cap)
- Stuck / timed-out tasks
- Stuck / timed-out projects
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog

from src.core.config import settings
from src.core.models import Task, TaskState

logger = structlog.get_logger()


class Guardrails:
    """Safety controls to prevent runaway agent behavior.

    All checks return ``True`` when the operation is **allowed** and
    ``False`` when the limit has been reached or exceeded.
    """

    def __init__(
        self,
        *,
        max_iterations: int | None = None,
        max_tasks: int | None = None,
        task_timeout: int | None = None,
        project_timeout: int | None = None,
    ) -> None:
        self.max_iterations = max_iterations or settings.max_iterations_per_task
        self.max_tasks = max_tasks or settings.max_tasks_per_project
        self.task_timeout_minutes = task_timeout or settings.task_timeout_minutes
        self.project_timeout_minutes = project_timeout or settings.project_timeout_minutes

    # --------------------------------------------------------
    # Iteration Limit
    # --------------------------------------------------------

    def check_iteration_limit(self, current: int, max_override: int | None = None) -> bool:
        """Return True if the current iteration count is within limits."""
        limit = max_override or self.max_iterations
        if current >= limit:
            logger.warning(
                "guardrail_iteration_limit_reached",
                current=current,
                max=limit,
            )
            return False
        return True

    # --------------------------------------------------------
    # Task Count Limit
    # --------------------------------------------------------

    def check_task_limit(self, current_task_count: int) -> bool:
        """Return True if we can still create more tasks for the project."""
        if current_task_count >= self.max_tasks:
            logger.warning(
                "guardrail_task_limit_reached",
                current=current_task_count,
                max=self.max_tasks,
            )
            return False
        return True

    # --------------------------------------------------------
    # Task Timeout
    # --------------------------------------------------------

    def check_task_timeout(self, task: Task) -> bool:
        """Return True if the task has NOT timed out.

        A task is considered timed-out if it has been in a non-terminal
        state (not done/failed) for longer than ``task_timeout_minutes``.
        """
        if task.state in (TaskState.DONE, TaskState.FAILED):
            return True  # Terminal states never time out

        now = datetime.now(timezone.utc)
        # task.updated_at may be a string from the DB or a datetime object
        updated = task.updated_at
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)

        deadline = updated + timedelta(minutes=self.task_timeout_minutes)
        if now > deadline:
            logger.warning(
                "guardrail_task_timeout",
                task_id=task.id,
                state=task.state.value,
                minutes_elapsed=(now - updated).total_seconds() / 60,
                timeout_minutes=self.task_timeout_minutes,
            )
            return False
        return True

    # --------------------------------------------------------
    # Composite Check
    # --------------------------------------------------------

    def should_allow_revision(self, task: Task) -> bool:
        """Check all relevant guards before allowing a revision cycle."""
        if not self.check_iteration_limit(task.iteration, task.max_iterations):
            return False
        if not self.check_task_timeout(task):
            return False
        return True
