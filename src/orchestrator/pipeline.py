"""Pipeline definitions — standard agent workflow orchestration.

Phase 3 responsibilities:
- Periodic health checks (task timeout, activity timeout, deadlock)
- Token budget enforcement at the project level
- Stream trimming to prevent unbounded Redis memory growth
"""

from __future__ import annotations

import asyncio

import structlog

from src.core.message_bus import MessageBus
from src.core.models import TaskState
from src.orchestrator.guardrails import Guardrails
from src.orchestrator.task_manager import TaskManager

logger = structlog.get_logger()

# How often to run maintenance tasks (seconds)
HEALTH_CHECK_INTERVAL = 30
STREAM_TRIM_INTERVAL = 300  # 5 minutes


class Pipeline:
    """Manages the overall agent workflow pipeline.

    Standard flow: PM → Research → Spec → Code → Critic → (iterate) → Done

    Periodic maintenance:
    - Stream trimming to prevent unbounded memory growth
    - Health checks for stuck / timed-out tasks
    - Activity timeout detection (no progress for N minutes)
    - Deadlock detection (all active tasks are stuck/failed)
    - Token budget enforcement
    """

    def __init__(self, *, task_manager: TaskManager, bus: MessageBus) -> None:
        self.task_manager = task_manager
        self.bus = bus
        self.guardrails = Guardrails()
        self._running = False
        self._tick_count = 0

    async def start(self) -> None:
        """Start the pipeline monitor."""
        self._running = True
        logger.info("pipeline_started")

        while self._running:
            try:
                self._tick_count += 1

                # Health check every tick
                await self._check_health()

                # Stream maintenance less frequently
                if self._tick_count % (STREAM_TRIM_INTERVAL // HEALTH_CHECK_INTERVAL) == 0:
                    await self.bus.trim_streams()

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("pipeline_error", error=str(e))
                await asyncio.sleep(10)

        logger.info("pipeline_stopped")

    def stop(self) -> None:
        """Stop the pipeline monitor."""
        self._running = False

    async def _check_health(self) -> None:
        """Check all active projects for stuck, timed-out, or deadlocked tasks.

        Walks through every active project and:
        1. Fails any individual task that exceeded its timeout.
        2. Detects project-level activity timeout (no task updated recently).
        3. Detects deadlocks (no task can make progress).
        4. Enforces the project token budget.
        """
        try:
            db = self.task_manager.db

            projects = await db.list_projects(status="active")

            for project in projects:
                all_tasks = await db.list_tasks(project.id)

                # 1. Token budget check (runs even with no tasks)
                project_tokens = await db.get_project_tokens(project.id)
                if not self.guardrails.check_token_budget(project_tokens):
                    logger.warning(
                        "pipeline_token_budget_exceeded",
                        project_id=project.id,
                        tokens_used=project_tokens,
                    )
                    await db.update_project(project.id, status="failed")
                    await self.bus.publish_ui_event(
                        "project_status_changed",
                        {
                            "project_id": project.id,
                            "new_status": "failed",
                            "reason": "token_budget_exceeded",
                        },
                    )
                    continue

                if not all_tasks:
                    continue

                # 2. Per-task timeout check
                active_states = [
                    TaskState.PENDING,
                    TaskState.ASSIGNED,
                    TaskState.IN_PROGRESS,
                    TaskState.REVIEW,
                    TaskState.REVISION,
                ]
                for task in all_tasks:
                    if task.state in active_states and not self.guardrails.check_task_timeout(task):
                        logger.warning(
                            "pipeline_failing_timed_out_task",
                            task_id=task.id,
                            state=task.state.value,
                            project_id=project.id,
                        )
                        task.state = TaskState.FAILED
                        task.feedback = (
                            f"Task timed out after "
                            f"{self.guardrails.task_timeout_minutes} minutes "
                            f"in state '{task.state.value}'"
                        )
                        await db.update_task(task)
                        await self.bus.publish_ui_event(
                            "task_state_changed",
                            {
                                "task_id": task.id,
                                "project_id": project.id,
                                "old_state": task.state.value,
                                "new_state": TaskState.FAILED.value,
                                "reason": "timeout",
                            },
                        )

                # Refresh task list after potential state changes
                all_tasks = await db.list_tasks(project.id)

                # 3. Activity timeout — find the most-recent update across
                #    all non-terminal tasks.
                active_tasks = [
                    t for t in all_tasks
                    if t.state not in (TaskState.DONE, TaskState.FAILED)
                ]
                if active_tasks:
                    from datetime import datetime
                    def _updated(t: object) -> datetime:
                        u = t.updated_at  # type: ignore[attr-defined]
                        if isinstance(u, str):
                            return datetime.fromisoformat(u)
                        return u  # type: ignore[return-value]

                    most_recent = max(active_tasks, key=_updated)
                    if not self.guardrails.check_activity_timeout(most_recent.updated_at):
                        logger.warning(
                            "pipeline_activity_timeout",
                            project_id=project.id,
                        )
                        await db.update_project(project.id, status="failed")
                        await self.bus.publish_ui_event(
                            "project_status_changed",
                            {
                                "project_id": project.id,
                                "new_status": "failed",
                                "reason": "activity_timeout",
                            },
                        )
                        continue  # Skip further checks for this project

                # 4. Deadlock detection
                if self.guardrails.detect_deadlock(all_tasks):
                    logger.warning(
                        "pipeline_deadlock_detected",
                        project_id=project.id,
                    )
                    await self.bus.publish_ui_event(
                        "project_deadlock",
                        {
                            "project_id": project.id,
                            "reason": "All active tasks are blocked or failed",
                        },
                    )

        except Exception as e:
            logger.error("pipeline_health_check_error", error=str(e))
