"""Pipeline definitions — standard agent workflow orchestration.

Phase 3 responsibilities:
- Periodic health checks (task timeout, activity timeout, deadlock)
- Token budget enforcement at the project level
- Stream trimming to prevent unbounded Redis memory growth
"""

from __future__ import annotations

import asyncio

import structlog

from hca.core.message_bus import MessageBus
from hca.core.models import (
    AgentMessage,
    AgentRole,
    MessagePayload,
    MessageType,
    TaskState,
)
from hca.orchestrator.guardrails import Guardrails
from hca.orchestrator.task_manager import TaskManager
from hca.orchestrator.workspace_manager import WorkspaceManager

logger = structlog.get_logger()

# How often to run maintenance tasks (seconds)
HEALTH_CHECK_INTERVAL = 30
STREAM_TRIM_INTERVAL = 300  # 5 minutes
WORKSPACE_CLEANUP_INTERVAL = 3600  # 1 hour


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
            # ensure graceful stop after each await
            if not self._running:
                break
            try:
                self._tick_count += 1

                # Health check every tick
                await self._check_health()

                # Stream maintenance less frequently
                if self._tick_count % (STREAM_TRIM_INTERVAL // HEALTH_CHECK_INTERVAL) == 0:
                    await self.bus.trim_streams()

                # Workspace cleanup even less frequently
                if self._tick_count % (WORKSPACE_CLEANUP_INTERVAL // HEALTH_CHECK_INTERVAL) == 0:
                    await WorkspaceManager.cleanup_old_workspaces()

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
                        old_state = task.state.value
                        logger.warning(
                            "pipeline_failing_timed_out_task",
                            task_id=task.id,
                            state=old_state,
                            project_id=project.id,
                        )
                        task.state = TaskState.FAILED
                        task.feedback = (
                            f"Task timed out after "
                            f"{self.guardrails.task_timeout_minutes} minutes "
                            f"in state '{old_state}'"
                        )
                        await db.update_task(task)
                        await self.bus.publish_ui_event(
                            "task_state_changed",
                            {
                                "task_id": task.id,
                                "project_id": project.id,
                                "old_state": old_state,
                                "new_state": TaskState.FAILED.value,
                                "reason": "timeout",
                            },
                        )
                        # Notify PM so it can retry the failed task
                        await self.bus.publish(
                            AgentMessage(
                                sender=AgentRole.SYSTEM,
                                recipient=AgentRole.PM,
                                type=MessageType.STATUS_UPDATE,
                                project_id=project.id,
                                task_id=task.id,
                                payload=MessagePayload(
                                    content=(
                                        f"Task '{task.title}' (ID: {task.id}) timed out in "
                                        f"state '{old_state}'.  Marked FAILED."
                                    ),
                                    metadata={"reason": "timeout"},
                                ),
                            )
                        )

                # Refresh task list after potential state changes
                all_tasks = await db.list_tasks(project.id)

                # 3. Activity timeout — find the most-recent update across
                #    all non-terminal tasks.
                active_tasks = [
                    t for t in all_tasks if t.state not in (TaskState.DONE, TaskState.FAILED)
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
                    # Wake the PM to retry FAILED tasks and break the deadlock
                    await self.bus.publish(
                        AgentMessage(
                            sender=AgentRole.SYSTEM,
                            recipient=AgentRole.PM,
                            type=MessageType.STATUS_UPDATE,
                            project_id=project.id,
                            task_id="",
                            payload=MessagePayload(
                                content=(
                                    f"Deadlock detected: all remaining tasks are blocked or "
                                    f"failed.  Attempting to retry failed tasks."
                                ),
                                metadata={"reason": "deadlock"},
                            ),
                        )
                    )

                # 5. Nudge the PM about unassigned PENDING tasks
                pending = [t for t in all_tasks if t.state == TaskState.PENDING]
                if pending:
                    await self.bus.publish(
                        AgentMessage(
                            sender=AgentRole.SYSTEM,
                            recipient=AgentRole.PM,
                            type=MessageType.STATUS_UPDATE,
                            project_id=project.id,
                            task_id="",
                            payload=MessagePayload(
                                content=f"{len(pending)} PENDING task(s) need assignment.",
                                metadata={"reason": "pending_tasks"},
                            ),
                        )
                    )

        except Exception as e:
            logger.error("pipeline_health_check_error", error=str(e))

    async def resume_projects(self) -> None:
        """On orchestrator restart, resume in-progress projects.

        Scans active projects for tasks stuck in ASSIGNED or IN_PROGRESS
        states and resets them to PENDING so the PM re-assigns and agents
        re-process them.
        """
        try:
            db = self.task_manager.db
            projects = await db.list_projects(status="active")

            for project in projects:
                all_tasks = await db.list_tasks(project.id)

                # Reset tasks stuck in transient states back to PENDING
                resume_states = {TaskState.ASSIGNED, TaskState.IN_PROGRESS}
                stuck = [t for t in all_tasks if t.state in resume_states]
                for task in stuck:
                    old_state = task.state.value
                    task.state = TaskState.PENDING
                    task.assigned_to = None
                    await db.update_task(task)
                    await self.bus.publish_ui_event(
                        "task_state_changed",
                        {
                            "task_id": task.id,
                            "project_id": project.id,
                            "old_state": old_state,
                            "new_state": TaskState.PENDING.value,
                            "reason": "resume_after_restart",
                        },
                    )
                    logger.info(
                        "task_reset_to_pending",
                        task_id=task.id,
                        old_state=old_state,
                    )

                # Reset FAILED tasks to PENDING so the PM can retry them
                failed = [t for t in all_tasks if t.state == TaskState.FAILED]
                for task in failed:
                    logger.info(
                        "task_retrying_on_resume",
                        task_id=task.id,
                        title=task.title,
                    )
                    task.state = TaskState.PENDING
                    task.feedback = ""
                    await db.update_task(task)
                    await self.bus.publish_ui_event(
                        "task_state_changed",
                        {
                            "task_id": task.id,
                            "project_id": project.id,
                            "old_state": TaskState.FAILED.value,
                            "new_state": TaskState.PENDING.value,
                            "reason": "resume_retry",
                        },
                    )

                # Re-trigger the PM if there are PENDING or FAILED tasks
                has_pending = any(t.state == TaskState.PENDING for t in all_tasks)
                has_failed = any(t.state == TaskState.FAILED for t in all_tasks)

                if stuck or has_pending or has_failed:
                    if stuck or has_failed:
                        logger.info(
                            "project_resume_reset",
                            project_id=project.id,
                            stuck_count=len(stuck),
                            failed_retried=len(failed),
                        )
                    # Wake the PM to assign tasks
                    await self.bus.publish(
                        AgentMessage(
                            sender=AgentRole.SYSTEM,
                            recipient=AgentRole.PM,
                            type=MessageType.STATUS_UPDATE,
                            project_id=project.id,
                            task_id="",
                            payload=MessagePayload(
                                content=f"Project resumed after restart. Tasks need assignment.",
                                metadata={"reason": "resume"},
                            ),
                        )
                    )

                # No tasks at all — re-send the original project idea as a
                # SYSTEM message so the PM decomposes it into tasks.
                if not all_tasks and project.idea:
                    logger.info(
                        "project_resume_no_tasks",
                        project_id=project.id,
                        project_name=project.name,
                    )
                    await self.bus.publish(
                        AgentMessage(
                            sender=AgentRole.USER,
                            recipient=AgentRole.PM,
                            type=MessageType.SYSTEM,
                            project_id=project.id,
                            task_id="",
                            payload=MessagePayload(
                                content=project.idea,
                                metadata={"reason": "resume_no_tasks"},
                            ),
                        )
                    )

            logger.info("project_resume_complete")
        except Exception as e:
            logger.error("project_resume_error", error=str(e))
