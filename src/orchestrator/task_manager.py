"""Task state machine and management."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from src.core.database import Database
from src.core.message_bus import MessageBus
from src.core.models import Task, TaskState, AgentRole

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
    """Manages task lifecycle and state transitions."""

    def __init__(self, *, db: Database, bus: MessageBus) -> None:
        self.db = db
        self.bus = bus

    async def create_task(
        self,
        *,
        project_id: str,
        title: str,
        description: str,
        assigned_to: AgentRole | None = None,
        max_iterations: int = 5,
    ) -> Task:
        """Create a new task in the pending state."""
        task = Task(
            project_id=project_id,
            title=title,
            description=description,
            assigned_to=assigned_to,
            max_iterations=max_iterations,
        )
        await self.db.create_task(task)
        logger.info("task_created", task_id=task.id, title=title)
        return task

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
            if task.iteration >= task.max_iterations:
                logger.warning(
                    "task_max_iterations",
                    task_id=task.id,
                    iterations=task.iteration,
                )
                task.state = TaskState.FAILED
                task.feedback = "Maximum revision iterations reached"

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

    async def get_project_progress(self, project_id: str) -> dict:
        """Get a summary of task progress for a project."""
        tasks = await self.db.list_tasks(project_id)
        total = len(tasks)
        by_state: dict[str, int] = {}
        for task in tasks:
            by_state[task.state.value] = by_state.get(task.state.value, 0) + 1

        done = by_state.get("done", 0)
        return {
            "total_tasks": total,
            "completed": done,
            "progress_pct": round(done / total * 100, 1) if total > 0 else 0,
            "by_state": by_state,
        }
