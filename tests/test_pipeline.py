"""Tests for the TaskManager state machine and Guardrails integration.

Validates: state transitions, iteration counting, max-iteration failure,
invalid transitions, progress reporting, task-count guardrails,
and timeout detection.
"""

from __future__ import annotations

import pytest

from src.core.database import Database
from src.core.models import AgentRole, Project, Task, TaskState
from src.orchestrator.guardrails import Guardrails
from src.orchestrator.task_manager import TaskManager, VALID_TRANSITIONS
from tests.conftest import MockMessageBus


# ============================================================
# Helpers
# ============================================================


async def _make_project_and_task(
    db: Database,
    state: TaskState = TaskState.PENDING,
    max_iterations: int = 5,
) -> tuple[Project, Task]:
    """Create a project and a task in the given state."""
    p = Project(name="TM Project", description="", idea="x")
    await db.create_project(p)
    t = Task(
        project_id=p.id,
        title="Test Task",
        description="",
        state=state,
        max_iterations=max_iterations,
    )
    await db.create_task(t)
    return p, t


# ============================================================
# Valid Transitions Map
# ============================================================


class TestTransitionMap:
    """Tests for the VALID_TRANSITIONS constant."""

    def test_all_states_have_entry(self) -> None:
        for state in TaskState:
            assert state in VALID_TRANSITIONS

    def test_done_has_no_transitions(self) -> None:
        assert VALID_TRANSITIONS[TaskState.DONE] == []

    def test_pending_can_only_go_to_assigned(self) -> None:
        assert VALID_TRANSITIONS[TaskState.PENDING] == [TaskState.ASSIGNED]


# ============================================================
# TaskManager.create_task
# ============================================================


class TestCreateTask:
    """Tests for TaskManager.create_task."""

    async def test_create_task(self, db: Database, mock_bus: MockMessageBus) -> None:
        p = Project(name="P", description="", idea="x")
        await db.create_project(p)
        tm = TaskManager(db=db, bus=mock_bus)

        task = await tm.create_task(
            project_id=p.id,
            title="Research API",
            description="Research the best API",
            assigned_to=AgentRole.RESEARCH,
        )
        assert task.state == TaskState.PENDING
        assert task.assigned_to == AgentRole.RESEARCH

        # Should be persisted
        fetched = await db.get_task(task.id)
        assert fetched is not None


# ============================================================
# TaskManager.transition
# ============================================================


class TestTransition:
    """Tests for state transitions."""

    async def test_valid_transition_pending_to_assigned(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        _, task = await _make_project_and_task(db, TaskState.PENDING)
        tm = TaskManager(db=db, bus=mock_bus)

        result = await tm.transition(task.id, TaskState.ASSIGNED)
        assert result.state == TaskState.ASSIGNED

        # Should publish a UI event
        assert any(e[0] == "task_state_changed" for e in mock_bus.ui_events)

    async def test_valid_transition_assigned_to_in_progress(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        _, task = await _make_project_and_task(db, TaskState.ASSIGNED)
        tm = TaskManager(db=db, bus=mock_bus)

        result = await tm.transition(task.id, TaskState.IN_PROGRESS)
        assert result.state == TaskState.IN_PROGRESS

    async def test_valid_transition_in_progress_to_review(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        _, task = await _make_project_and_task(db, TaskState.IN_PROGRESS)
        tm = TaskManager(db=db, bus=mock_bus)

        result = await tm.transition(task.id, TaskState.REVIEW)
        assert result.state == TaskState.REVIEW

    async def test_invalid_transition_raises(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        _, task = await _make_project_and_task(db, TaskState.PENDING)
        tm = TaskManager(db=db, bus=mock_bus)

        with pytest.raises(ValueError, match="Invalid transition"):
            await tm.transition(task.id, TaskState.DONE)

    async def test_transition_nonexistent_task_raises(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        tm = TaskManager(db=db, bus=mock_bus)
        with pytest.raises(ValueError, match="not found"):
            await tm.transition("nonexistent", TaskState.ASSIGNED)

    async def test_revision_increments_iteration(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        _, task = await _make_project_and_task(db, TaskState.REVIEW)
        tm = TaskManager(db=db, bus=mock_bus)

        result = await tm.transition(task.id, TaskState.REVISION)
        assert result.iteration == 1

    async def test_max_iterations_forces_failure(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """When iteration count reaches max_iterations, task should be FAILED."""
        _, task = await _make_project_and_task(
            db, TaskState.REVIEW, max_iterations=1
        )
        # Set iteration to max_iterations - 1 so the next revision hits the limit
        task.iteration = 0
        await db.update_task(task)

        tm = TaskManager(db=db, bus=mock_bus)
        result = await tm.transition(task.id, TaskState.REVISION)
        # iteration becomes 1, which equals max_iterations (1), so FAILED
        assert result.state == TaskState.FAILED
        assert "Guardrail" in (result.feedback or "") or "maximum" in (result.feedback or "").lower()

    async def test_full_happy_path(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """Walk through the full pipeline: pending → assigned → in_progress → review → approved → done."""
        _, task = await _make_project_and_task(db, TaskState.PENDING)
        tm = TaskManager(db=db, bus=mock_bus)

        task = await tm.transition(task.id, TaskState.ASSIGNED)
        assert task.state == TaskState.ASSIGNED

        task = await tm.transition(task.id, TaskState.IN_PROGRESS)
        assert task.state == TaskState.IN_PROGRESS

        task = await tm.transition(task.id, TaskState.REVIEW)
        assert task.state == TaskState.REVIEW

        task = await tm.transition(task.id, TaskState.APPROVED)
        assert task.state == TaskState.APPROVED

        task = await tm.transition(task.id, TaskState.DONE)
        assert task.state == TaskState.DONE

    async def test_revision_loop_path(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """Test the revision loop: review → revision → in_progress → review."""
        _, task = await _make_project_and_task(db, TaskState.REVIEW, max_iterations=5)
        tm = TaskManager(db=db, bus=mock_bus)

        task = await tm.transition(task.id, TaskState.REVISION)
        assert task.state == TaskState.REVISION
        assert task.iteration == 1

        task = await tm.transition(task.id, TaskState.IN_PROGRESS)
        assert task.state == TaskState.IN_PROGRESS

        task = await tm.transition(task.id, TaskState.REVIEW)
        assert task.state == TaskState.REVIEW

    async def test_failed_can_retry(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """A failed task can be retried (failed → pending)."""
        _, task = await _make_project_and_task(db, TaskState.FAILED)
        tm = TaskManager(db=db, bus=mock_bus)

        task = await tm.transition(task.id, TaskState.PENDING)
        assert task.state == TaskState.PENDING


# ============================================================
# TaskManager.get_project_progress
# ============================================================


class TestProjectProgress:
    """Tests for progress reporting."""

    async def test_progress_empty_project(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = Project(name="Empty", description="", idea="x")
        await db.create_project(p)
        tm = TaskManager(db=db, bus=mock_bus)

        progress = await tm.get_project_progress(p.id)
        assert progress["total_tasks"] == 0
        assert progress["progress_pct"] == 0

    async def test_progress_with_tasks(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = Project(name="Prog", description="", idea="x")
        await db.create_project(p)

        t1 = Task(project_id=p.id, title="T1", description="", state=TaskState.DONE)
        t2 = Task(project_id=p.id, title="T2", description="", state=TaskState.IN_PROGRESS)
        t3 = Task(project_id=p.id, title="T3", description="", state=TaskState.DONE)
        await db.create_task(t1)
        await db.create_task(t2)
        await db.create_task(t3)

        tm = TaskManager(db=db, bus=mock_bus)
        progress = await tm.get_project_progress(p.id)

        assert progress["total_tasks"] == 3
        assert progress["completed"] == 2
        assert progress["progress_pct"] == pytest.approx(66.7, abs=0.1)
        assert progress["by_state"]["done"] == 2
        assert progress["by_state"]["in_progress"] == 1


# ============================================================
# Guardrails
# ============================================================


class TestGuardrails:
    """Tests for the Guardrails safety controls."""

    def test_iteration_within_limit(self) -> None:
        g = Guardrails(max_iterations=5)
        assert g.check_iteration_limit(0) is True
        assert g.check_iteration_limit(4) is True

    def test_iteration_at_limit(self) -> None:
        g = Guardrails(max_iterations=5)
        assert g.check_iteration_limit(5) is False
        assert g.check_iteration_limit(10) is False

    def test_iteration_with_override(self) -> None:
        g = Guardrails(max_iterations=5)
        assert g.check_iteration_limit(2, max_override=3) is True
        assert g.check_iteration_limit(3, max_override=3) is False

    def test_task_limit_within(self) -> None:
        g = Guardrails(max_tasks=10)
        assert g.check_task_limit(0) is True
        assert g.check_task_limit(9) is True

    def test_task_limit_reached(self) -> None:
        g = Guardrails(max_tasks=10)
        assert g.check_task_limit(10) is False
        assert g.check_task_limit(11) is False

    def test_should_allow_revision_yes(self) -> None:
        g = Guardrails(max_iterations=5, task_timeout=999)
        task = Task(
            project_id="p", title="t", description="",
            iteration=2, max_iterations=5,
        )
        assert g.should_allow_revision(task) is True

    def test_should_allow_revision_no(self) -> None:
        g = Guardrails(max_iterations=5, task_timeout=999)
        task = Task(
            project_id="p", title="t", description="",
            iteration=5, max_iterations=5,
        )
        assert g.should_allow_revision(task) is False


# ============================================================
# TaskManager — Task Count Guardrail
# ============================================================


class TestTaskCountGuardrail:
    """Tests that create_task enforces the per-project task limit."""

    async def test_create_task_within_limit(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = Project(name="GL", description="", idea="x")
        await db.create_project(p)
        g = Guardrails(max_tasks=3)
        tm = TaskManager(db=db, bus=mock_bus, guardrails=g)

        # Should succeed
        await tm.create_task(project_id=p.id, title="T1", description="")
        await tm.create_task(project_id=p.id, title="T2", description="")
        await tm.create_task(project_id=p.id, title="T3", description="")

    async def test_create_task_exceeds_limit(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = Project(name="GL", description="", idea="x")
        await db.create_project(p)
        g = Guardrails(max_tasks=2)
        tm = TaskManager(db=db, bus=mock_bus, guardrails=g)

        await tm.create_task(project_id=p.id, title="T1", description="")
        await tm.create_task(project_id=p.id, title="T2", description="")

        with pytest.raises(ValueError, match="maximum task limit"):
            await tm.create_task(project_id=p.id, title="T3", description="")
