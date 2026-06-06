"""Tests for Guardrails — safety limits and controls.

Covers all guardrail methods in isolation:
- Iteration limit, task count, task timeout, token budget, activity timeout
- Deadlock detection
- Composite should_allow_revision check
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hca.core.models import Task, TaskState
from hca.orchestrator.guardrails import Guardrails


class TestIterationLimit:
    def test_within_limit(self) -> None:
        g = Guardrails(max_iterations=5)
        assert g.check_iteration_limit(3) is True

    def test_at_limit(self) -> None:
        g = Guardrails(max_iterations=5)
        assert g.check_iteration_limit(5) is False

    def test_exceeds_limit(self) -> None:
        g = Guardrails(max_iterations=5)
        assert g.check_iteration_limit(10) is False

    def test_zero_iterations(self) -> None:
        g = Guardrails(max_iterations=0)
        assert g.check_iteration_limit(0) is False

    def test_max_override_lower(self) -> None:
        g = Guardrails(max_iterations=5)
        assert g.check_iteration_limit(3, max_override=2) is False

    def test_max_override_higher(self) -> None:
        g = Guardrails(max_iterations=5)
        assert g.check_iteration_limit(8, max_override=10) is True


class TestTaskLimit:
    def test_within_limit(self) -> None:
        g = Guardrails(max_tasks=50)
        assert g.check_task_limit(10) is True

    def test_at_limit(self) -> None:
        g = Guardrails(max_tasks=50)
        assert g.check_task_limit(50) is False

    def test_exceeds_limit(self) -> None:
        g = Guardrails(max_tasks=50)
        assert g.check_task_limit(100) is False

    def test_zero_limit(self) -> None:
        g = Guardrails(max_tasks=0)
        assert g.check_task_limit(0) is False

    def test_empty_project(self) -> None:
        g = Guardrails(max_tasks=50)
        assert g.check_task_limit(0) is True


class TestTaskTimeout:
    def make_task(
        self,
        state: TaskState = TaskState.IN_PROGRESS,
        minutes_ago: int = 1,
        iteration: int = 1,
    ) -> Task:
        updated = datetime.now(UTC) - timedelta(minutes=minutes_ago)
        return Task(
            id="test-task",
            project_id="test-project",
            title="Test",
            description="",
            state=state,
            iteration=iteration,
            agent_type="coder",
            created_at=updated,
            updated_at=updated,
        )

    def test_recently_updated(self) -> None:
        g = Guardrails(task_timeout=30)
        task = self.make_task(minutes_ago=5)
        assert g.check_task_timeout(task) is True

    def test_timed_out(self) -> None:
        g = Guardrails(task_timeout=30)
        task = self.make_task(minutes_ago=60)
        assert g.check_task_timeout(task) is False

    def test_terminal_state_never_times_out(self) -> None:
        g = Guardrails(task_timeout=30)
        for state in (TaskState.DONE, TaskState.FAILED):
            task = self.make_task(state=state, minutes_ago=999)
            assert g.check_task_timeout(task) is True

    def test_string_datetime(self) -> None:
        g = Guardrails(task_timeout=30)
        updated = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        task = Task(
            id="test-task",
            project_id="test-project",
            title="Test",
            description="",
            state=TaskState.IN_PROGRESS,
            iteration=1,
            agent_type="coder",
            created_at=updated,
            updated_at=updated,
        )
        assert g.check_task_timeout(task) is True


class TestTokenBudget:
    def test_under_budget(self) -> None:
        g = Guardrails(project_token_budget=500_000)
        assert g.check_token_budget(100_000) is True

    def test_at_budget(self) -> None:
        g = Guardrails(project_token_budget=500_000)
        assert g.check_token_budget(500_000) is False

    def test_exceeds_budget(self) -> None:
        g = Guardrails(project_token_budget=500_000)
        assert g.check_token_budget(600_000) is False

    def test_zero_budget(self) -> None:
        g = Guardrails(project_token_budget=0)
        assert g.check_token_budget(0) is False

    def test_no_tokens_used(self) -> None:
        g = Guardrails(project_token_budget=500_000)
        assert g.check_token_budget(0) is True


class TestActivityTimeout:
    def test_recent_activity(self) -> None:
        g = Guardrails(activity_timeout=60)
        recent = datetime.now(UTC) - timedelta(minutes=10)
        assert g.check_activity_timeout(recent) is True

    def test_timed_out(self) -> None:
        g = Guardrails(activity_timeout=60)
        old = datetime.now(UTC) - timedelta(minutes=120)
        assert g.check_activity_timeout(old) is False

    def test_string_datetime(self) -> None:
        g = Guardrails(activity_timeout=60)
        recent = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        assert g.check_activity_timeout(recent) is True

    def test_exactly_at_timeout(self) -> None:
        g = Guardrails(activity_timeout=60)
        exact = datetime.now(UTC) - timedelta(minutes=60)
        assert g.check_activity_timeout(exact) is False


class TestDeadlockDetection:
    def make_task(
        self,
        task_id: str,
        state: TaskState = TaskState.DONE,
        depends_on: list[str] | None = None,
    ) -> Task:
        now = datetime.now(UTC)
        return Task(
            id=task_id,
            project_id="test-project",
            title=f"Task {task_id}",
            description="",
            state=state,
            iteration=1,
            agent_type="coder",
            created_at=now,
            updated_at=now,
            depends_on=depends_on or [],
        )

    def test_all_done(self) -> None:
        g = Guardrails()
        tasks = [
            self.make_task("t1", TaskState.DONE),
            self.make_task("t2", TaskState.DONE),
        ]
        assert g.detect_deadlock(tasks) is False

    def test_in_progress_working(self) -> None:
        g = Guardrails()
        tasks = [
            self.make_task("t1", TaskState.DONE),
            self.make_task("t2", TaskState.IN_PROGRESS),
        ]
        assert g.detect_deadlock(tasks) is False

    def test_all_failed(self) -> None:
        g = Guardrails()
        tasks = [
            self.make_task("t1", TaskState.FAILED),
            self.make_task("t2", TaskState.FAILED),
        ]
        assert g.detect_deadlock(tasks) is True

    def test_pending_with_unmet_dep(self) -> None:
        g = Guardrails()
        tasks = [
            self.make_task("t1", TaskState.PENDING, depends_on=["t2"]),
            self.make_task("t2", TaskState.FAILED),
        ]
        assert g.detect_deadlock(tasks) is True

    def test_pending_with_met_dep(self) -> None:
        g = Guardrails()
        tasks = [
            self.make_task("t1", TaskState.PENDING, depends_on=["t2"]),
            self.make_task("t2", TaskState.DONE),
        ]
        assert g.detect_deadlock(tasks) is False

    def test_mixed_deadlocked(self) -> None:
        g = Guardrails()
        tasks = [
            self.make_task("t1", TaskState.FAILED),
            self.make_task("t2", TaskState.FAILED),
            self.make_task("t3", TaskState.DONE),
        ]
        assert g.detect_deadlock(tasks) is True

    def test_empty_tasks(self) -> None:
        g = Guardrails()
        assert g.detect_deadlock([]) is False


class TestShouldAllowRevision:
    def make_task(self, iteration: int = 2, max_iterations: int = 5) -> Task:
        now = datetime.now(UTC)
        return Task(
            id="test-task",
            project_id="test-project",
            title="Test",
            description="",
            state=TaskState.REVISION,
            iteration=iteration,
            max_iterations=max_iterations,
            agent_type="coder",
            created_at=now,
            updated_at=now,
        )

    def test_allows_when_within_limits(self) -> None:
        g = Guardrails(max_iterations=5, task_timeout=30)
        task = self.make_task(iteration=2)
        assert g.should_allow_revision(task) is True

    def test_blocks_when_iteration_exceeded(self) -> None:
        """Guardrails' global limit of 3 blocks when task hits its own limit (5)."""
        g = Guardrails(max_iterations=3, task_timeout=30)
        task = self.make_task(iteration=5)
        assert g.should_allow_revision(task) is False

    def test_blocks_when_timed_out(self) -> None:
        g = Guardrails(max_iterations=5, task_timeout=30)
        task = Task(
            id="test-task",
            project_id="test-project",
            title="Test",
            description="",
            state=TaskState.REVISION,
            iteration=2,
            agent_type="coder",
            created_at=datetime.now(UTC) - timedelta(minutes=60),
            updated_at=datetime.now(UTC) - timedelta(minutes=60),
        )
        assert g.should_allow_revision(task) is False

    def test_custom_max_iterations_on_task(self) -> None:
        g = Guardrails(max_iterations=5, task_timeout=30)
        task = self.make_task(iteration=2, max_iterations=1)
        assert g.should_allow_revision(task) is False
