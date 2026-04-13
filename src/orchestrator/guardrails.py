"""Guardrails — safety limits and controls for agent behavior."""

from __future__ import annotations

import structlog

from src.core.config import settings

logger = structlog.get_logger()


class Guardrails:
    """Safety controls to prevent runaway agent behavior."""

    def __init__(self) -> None:
        self.max_iterations = settings.max_iterations_per_task
        self.max_tasks = settings.max_tasks_per_project
        self.task_timeout = settings.task_timeout_minutes
        self.project_timeout = settings.project_timeout_minutes

    def check_iteration_limit(self, current: int) -> bool:
        """Check if the iteration count is within limits."""
        if current >= self.max_iterations:
            logger.warning("guardrail_iteration_limit", current=current, max=self.max_iterations)
            return False
        return True

    def check_task_limit(self, current: int) -> bool:
        """Check if the task count is within limits."""
        if current >= self.max_tasks:
            logger.warning("guardrail_task_limit", current=current, max=self.max_tasks)
            return False
        return True
