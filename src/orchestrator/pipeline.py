"""Pipeline definitions — standard agent workflow orchestration."""

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

    Also handles periodic maintenance:
    - Stream trimming to prevent unbounded memory growth
    - Health checks for stuck / timed-out tasks
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
        """Check all active projects for stuck or timed-out tasks.

        Walks through every active project's non-terminal tasks and
        fails any that have exceeded the task timeout.
        """
        try:
            db = self.task_manager.db
            projects = await db.list_projects(status="active")

            for project in projects:
                # Non-terminal states to check
                active_states = [
                    TaskState.PENDING,
                    TaskState.ASSIGNED,
                    TaskState.IN_PROGRESS,
                    TaskState.REVIEW,
                    TaskState.REVISION,
                ]
                for state in active_states:
                    tasks = await db.list_tasks(project.id, state=state)
                    for task in tasks:
                        if not self.guardrails.check_task_timeout(task):
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
                                f"in state '{state.value}'"
                            )
                            await db.update_task(task)
                            await self.bus.publish_ui_event(
                                "task_state_changed",
                                {
                                    "task_id": task.id,
                                    "project_id": project.id,
                                    "old_state": state.value,
                                    "new_state": TaskState.FAILED.value,
                                    "reason": "timeout",
                                },
                            )
        except Exception as e:
            logger.error("pipeline_health_check_error", error=str(e))
