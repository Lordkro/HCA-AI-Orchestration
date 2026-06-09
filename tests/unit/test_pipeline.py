"""Tests for the Pipeline health-check loop.

Uses the real database (temp file), real TaskManager, real Guardrails,
and a mock message bus so no external services are needed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from hca.core.database import Database
from hca.core.models import AgentRole, Project, Task, TaskState
from hca.orchestrator.guardrails import Guardrails
from hca.orchestrator.pipeline import Pipeline
from hca.orchestrator.task_manager import TaskManager
from tests.conftest import MockMessageBus

# ============================================================
# Helpers
# ============================================================


async def _force_updated_at(db: Database, task_id: str, dt: datetime) -> None:
    """Bypass update_task's auto-timestamp to set a past updated_at."""
    await db._execute(
        "UPDATE tasks SET updated_at = ? WHERE id = ?",
        (dt.isoformat(), task_id),
    )
    await db.db.commit()


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
async def pipeline(db) -> Pipeline:
    """Provide a Pipeline with real TaskManager + mock bus."""
    bus = MockMessageBus()
    tm = TaskManager(db=db, bus=bus)
    return Pipeline(task_manager=tm, bus=bus)


@pytest.fixture
async def project(db) -> str:
    """Create and return an active project ID."""
    p = Project(name="test", description="pipeline test", idea="test")
    created = await db.create_project(p)
    return created.id


@pytest.fixture
async def task(project: str, db) -> str:
    """Create and return a PENDING task ID."""
    t = Task(project_id=project, title="test task", description="do it")
    created = await db.create_task(t)
    return created.id


# ============================================================
# Lifecycle
# ============================================================


class TestPipelineLifecycle:
    async def test_start_stop(self, pipeline: Pipeline) -> None:
        assert pipeline._running is False
        pipeline.stop()  # no-op before start
        # start in a task that we cancel quickly
        async def run() -> None:
            await pipeline.start()

        t = asyncio.create_task(run())
        await asyncio.sleep(0.05)
        assert pipeline._running is True
        assert pipeline._tick_count >= 0
        pipeline.stop()
        await t
        assert pipeline._running is False

    async def test_multiple_stops_safe(self, pipeline: Pipeline) -> None:
        pipeline.stop()
        pipeline.stop()  # second stop should not raise
        assert pipeline._running is False


# ============================================================
# Health Check — Token Budget
# ============================================================


class TestTokenBudget:
    async def test_token_budget_ok(self, pipeline: Pipeline, project: str, db) -> None:
        await pipeline._check_health()
        proj = await db.get_project(project)
        assert proj.status == "active"

    async def test_token_budget_exceeded(self, pipeline: Pipeline, project: str, db) -> None:
        await db.update_project(project, tokens_used=900_000)
        await pipeline._check_health()
        proj = await db.get_project(project)
        assert proj.status == "failed"

    async def test_token_budget_exceeded_emits_event(self, pipeline: Pipeline, project: str, db) -> None:
        await db.update_project(project, tokens_used=900_000)
        await pipeline._check_health()
        assert any(
            evt[0] == "project_status_changed" and evt[1]["reason"] == "token_budget_exceeded"
            for evt in pipeline.bus.ui_events
        )


# ============================================================
# Health Check — Task Timeout
# ============================================================


class TestTaskTimeout:
    async def test_task_not_timed_out(self, pipeline: Pipeline, project: str, task: str, db) -> None:
        await pipeline._check_health()
        t = await db.get_task(task)
        assert t.state == TaskState.PENDING

    async def test_task_timed_out(self, pipeline: Pipeline, project: str, task: str, db) -> None:
        t = await db.get_task(task)
        t.state = TaskState.IN_PROGRESS
        await db.update_task(t)
        await _force_updated_at(db, task, datetime.now(UTC) - timedelta(minutes=60))
        await pipeline._check_health()
        t = await db.get_task(task)
        assert t.state == TaskState.FAILED
        assert "timed out" in (t.feedback or "")

    async def test_terminal_task_not_timed_out(
        self, pipeline: Pipeline, project: str, task: str, db
    ) -> None:
        t = await db.get_task(task)
        t.state = TaskState.DONE
        await db.update_task(t)
        await _force_updated_at(db, task, datetime.now(UTC) - timedelta(minutes=60))
        await pipeline._check_health()
        t = await db.get_task(task)
        assert t.state == TaskState.DONE  # unchanged


# ============================================================
# Health Check — Activity Timeout
# ============================================================


class TestActivityTimeout:
    async def test_recent_activity(self, pipeline: Pipeline, project: str, task: str, db) -> None:
        t = await db.get_task(task)
        t.state = TaskState.IN_PROGRESS
        await db.update_task(t)
        await pipeline._check_health()
        proj = await db.get_project(project)
        assert proj.status == "active"

    async def test_activity_timed_out(self, pipeline: Pipeline, project: str, db, task) -> None:
        # Use a guardrails with long task timeout so task doesn't time out first
        pipeline.guardrails = Guardrails(
            task_timeout=120,  # 2 hour task timeout
            activity_timeout=1,  # 1 minute activity timeout
        )
        t = await db.get_task(task)
        t.state = TaskState.IN_PROGRESS
        await db.update_task(t)
        await _force_updated_at(db, task, datetime.now(UTC) - timedelta(minutes=5))
        await pipeline._check_health()
        proj = await db.get_project(project)
        assert proj.status == "failed"


# ============================================================
# Health Check — Deadlock Detection
# ============================================================


class TestDeadlock:
    async def test_no_deadlock_when_progress_possible(
        self, pipeline: Pipeline, project: str, task: str, db
    ) -> None:
        await pipeline._check_health()
        events = [e for e in pipeline.bus.ui_events if e[0] == "project_deadlock"]
        assert len(events) == 0

    async def test_deadlock_detected(self, pipeline: Pipeline, project: str, db) -> None:
        t1 = Task(project_id=project, title="t1", description="", state=TaskState.FAILED)
        t2 = Task(
            project_id=project,
            title="t2",
            description="",
            state=TaskState.PENDING,
            depends_on=["t1"],
        )
        await db.create_task(t1)
        await db.create_task(t2)
        await pipeline._check_health()
        events = [e for e in pipeline.bus.ui_events if e[0] == "project_deadlock"]
        assert len(events) == 1


# ============================================================
# Resume Projects
# ============================================================


class TestResume:
    async def test_resume_resets_in_progress_tasks(
        self, pipeline: Pipeline, project: str, task: str, db
    ) -> None:
        t = await db.get_task(task)
        t.state = TaskState.IN_PROGRESS
        t.assigned_to = AgentRole.CODER
        await db.update_task(t)

        await pipeline.resume_projects()

        t = await db.get_task(task)
        assert t.state == TaskState.PENDING
        assert t.assigned_to is None

    async def test_resume_resets_assigned_tasks(
        self, pipeline: Pipeline, project: str, task: str, db
    ) -> None:
        t = await db.get_task(task)
        t.state = TaskState.ASSIGNED
        t.assigned_to = AgentRole.PM
        await db.update_task(t)

        await pipeline.resume_projects()

        t = await db.get_task(task)
        assert t.state == TaskState.PENDING
        assert t.assigned_to is None

    async def test_resume_skips_done_tasks(
        self, pipeline: Pipeline, project: str, task: str, db
    ) -> None:
        t = await db.get_task(task)
        t.state = TaskState.DONE
        t.assigned_to = AgentRole.CODER
        await db.update_task(t)

        await pipeline.resume_projects()

        t = await db.get_task(task)
        assert t.state == TaskState.DONE  # unchanged
        assert t.assigned_to == "coder"

    async def test_resume_with_no_stuck_tasks(
        self, pipeline: Pipeline, project: str, task: str, db
    ) -> None:
        t = await db.get_task(task)
        t.state = TaskState.PENDING
        await db.update_task(t)

        await pipeline.resume_projects()
        t = await db.get_task(task)
        assert t.state == TaskState.PENDING  # unchanged

    async def test_resume_emits_events(
        self, pipeline: Pipeline, project: str, task: str, db
    ) -> None:
        t = await db.get_task(task)
        t.state = TaskState.IN_PROGRESS
        await db.update_task(t)

        await pipeline.resume_projects()
        assert any(
            evt[0] == "task_state_changed" and evt[1]["reason"] == "resume_after_restart"
            for evt in pipeline.bus.ui_events
        )
