"""Guardrails — safety limits and controls for agent behavior.

Provides configurable limits to prevent:
- Infinite revision loops (per-task iteration cap)
- Unbounded task creation (per-project task cap)
- Stuck / timed-out tasks
- Stuck / timed-out projects
- Runaway token consumption (per-project token budget)
- Deadlocked projects (no progress detection)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import structlog

from src.core.config import settings
from src.core.models import Task, TaskState

logger = structlog.get_logger()

# Terminal states — a task in one of these is "finished" and never times out.
_TERMINAL_STATES = frozenset({TaskState.DONE, TaskState.FAILED})

# Active (non-terminal) states — tasks we should watch for stuck-ness.
_ACTIVE_STATES = frozenset({
    TaskState.PENDING,
    TaskState.ASSIGNED,
    TaskState.IN_PROGRESS,
    TaskState.REVIEW,
    TaskState.REVISION,
})


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
        project_token_budget: int | None = None,
        activity_timeout: int | None = None,
    ) -> None:
        self.max_iterations = max_iterations or settings.max_iterations_per_task
        self.max_tasks = max_tasks or settings.max_tasks_per_project
        self.task_timeout_minutes = task_timeout or settings.task_timeout_minutes
        self.project_timeout_minutes = project_timeout or settings.project_timeout_minutes
        self.project_token_budget = project_token_budget or settings.project_token_budget
        self.activity_timeout_minutes = activity_timeout or settings.activity_timeout_minutes

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
        if task.state in _TERMINAL_STATES:
            return True  # Terminal states never time out

        now = datetime.now(timezone.utc)
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
    # Token Budget
    # --------------------------------------------------------

    def check_token_budget(self, current_tokens: int) -> bool:
        """Return True if the project has NOT exceeded its token budget."""
        if current_tokens >= self.project_token_budget:
            logger.warning(
                "guardrail_token_budget_exceeded",
                current=current_tokens,
                budget=self.project_token_budget,
            )
            return False
        return True

    # --------------------------------------------------------
    # Activity Timeout
    # --------------------------------------------------------

    def check_activity_timeout(self, last_activity: datetime | str) -> bool:
        """Return True if activity is recent enough (NOT timed out).

        ``last_activity`` is the most-recent ``updated_at`` across all
        non-terminal tasks in a project.
        """
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)

        now = datetime.now(timezone.utc)
        deadline = last_activity + timedelta(minutes=self.activity_timeout_minutes)
        if now > deadline:
            logger.warning(
                "guardrail_activity_timeout",
                minutes_since_activity=(now - last_activity).total_seconds() / 60,
                timeout_minutes=self.activity_timeout_minutes,
            )
            return False
        return True

    # --------------------------------------------------------
    # Deadlock Detection
    # --------------------------------------------------------

    def detect_deadlock(self, tasks: Sequence[Task]) -> bool:
        """Return True if the project appears deadlocked.

        A project is deadlocked when:
        - There is at least one non-DONE task, AND
        - No non-DONE task can make further progress.

        Specifically, every non-DONE task is either:
        - FAILED (stuck without manual retry), or
        - PENDING with unmet dependencies (a dep is not DONE).
        """
        not_done = [t for t in tasks if t.state != TaskState.DONE]
        if not not_done:
            return False  # Everything is done → not deadlocked

        done_ids = {t.id for t in tasks if t.state == TaskState.DONE}

        for task in not_done:
            if task.state == TaskState.FAILED:
                continue  # Failed tasks can't make progress on their own
            if task.state == TaskState.PENDING and task.depends_on:
                unmet = [d for d in task.depends_on if d not in done_ids]
                if unmet:
                    continue  # Blocked on unmet deps
            # This task is in a state where it can still make progress
            return False

        # Every non-DONE task is either FAILED or blocked on unmet deps
        logger.warning(
            "guardrail_deadlock_detected",
            not_done_count=len(not_done),
            failed_count=sum(1 for t in not_done if t.state == TaskState.FAILED),
        )
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
