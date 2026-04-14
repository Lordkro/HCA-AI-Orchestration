"""Tests for orchestration, workflow engine, and guardrails.

Covers:
- Task dependencies and dependency-aware ordering
- Parallel task dispatch
- Token budget tracking and enforcement
- Deadlock detection
- Activity timeout detection
- Pause / resume (project-level)
- Escalation on guardrail failure
- Task retry (FAILED → PENDING)
- Pipeline health checks
- Database schema v3 fields (depends_on, tokens_used)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from src.core.database import Database
from src.core.models import (
    AgentMessage,
    AgentRole,
    MessagePayload,
    MessageType,
    Priority,
    Project,
    Task,
    TaskState,
)
from src.orchestrator.guardrails import Guardrails
from src.orchestrator.task_manager import TaskManager, VALID_TRANSITIONS
from src.orchestrator.pipeline import Pipeline
from src.agents.pm_agent import PMAgent
from src.agents.research_agent import ResearchAgent
from tests.conftest import MockMessageBus, MockOllamaClient, make_message


# ============================================================
# Helpers
# ============================================================


async def _make_project(db: Database) -> Project:
    p = Project(name="Phase3 Project", description="Test idea", idea="Build a chat app")
    await db.create_project(p)
    return p


async def _make_tm(
    db: Database,
    bus: MockMessageBus,
    *,
    max_iterations: int = 5,
    max_tasks: int = 50,
    token_budget: int = 500_000,
) -> TaskManager:
    g = Guardrails(
        max_iterations=max_iterations,
        max_tasks=max_tasks,
        project_token_budget=token_budget,
    )
    return TaskManager(db=db, bus=bus, guardrails=g)


# ============================================================
# Task Dependencies
# ============================================================


class TestTaskDependencies:
    """Tests for dependency-aware task creation and ordering."""

    async def test_create_task_with_dependencies(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus)

        t1 = await tm.create_task(
            project_id=p.id, title="Research", description="Do research",
            assigned_to=AgentRole.RESEARCH,
        )
        t2 = await tm.create_task(
            project_id=p.id, title="Spec", description="Write spec",
            assigned_to=AgentRole.SPEC, depends_on=[t1.id],
        )

        assert t2.depends_on == [t1.id]

        # Verify persisted correctly
        fetched = await db.get_task(t2.id)
        assert fetched is not None
        assert fetched.depends_on == [t1.id]

    async def test_assignable_tasks_no_deps(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """Tasks with no dependencies are always assignable."""
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus)

        t1 = await tm.create_task(
            project_id=p.id, title="A", description="", assigned_to=AgentRole.RESEARCH,
        )
        t2 = await tm.create_task(
            project_id=p.id, title="B", description="", assigned_to=AgentRole.RESEARCH,
        )

        assignable = await tm.get_assignable_tasks(p.id)
        assert len(assignable) == 2

    async def test_assignable_tasks_blocked_by_deps(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """Tasks whose deps are not DONE are blocked."""
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus)

        t1 = await tm.create_task(
            project_id=p.id, title="Research", description="",
            assigned_to=AgentRole.RESEARCH,
        )
        t2 = await tm.create_task(
            project_id=p.id, title="Spec", description="",
            assigned_to=AgentRole.SPEC, depends_on=[t1.id],
        )

        assignable = await tm.get_assignable_tasks(p.id)
        ids = [t.id for t in assignable]
        assert t1.id in ids
        assert t2.id not in ids

    async def test_assignable_tasks_unblocked_after_dep_done(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """Once dependency is DONE, dependent task becomes assignable."""
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus)

        t1 = await tm.create_task(
            project_id=p.id, title="Research", description="",
            assigned_to=AgentRole.RESEARCH,
        )
        t2 = await tm.create_task(
            project_id=p.id, title="Spec", description="",
            assigned_to=AgentRole.SPEC, depends_on=[t1.id],
        )

        # Walk t1 through its lifecycle to DONE
        await tm.transition(t1.id, TaskState.ASSIGNED)
        await tm.transition(t1.id, TaskState.IN_PROGRESS)
        await tm.transition(t1.id, TaskState.REVIEW)
        await tm.transition(t1.id, TaskState.APPROVED)
        await tm.transition(t1.id, TaskState.DONE)

        assignable = await tm.get_assignable_tasks(p.id)
        ids = [t.id for t in assignable]
        assert t2.id in ids

    async def test_assignable_tasks_respects_limit(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """get_assignable_tasks respects the limit parameter."""
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus)

        for i in range(5):
            await tm.create_task(
                project_id=p.id, title=f"Task {i}", description="",
                assigned_to=AgentRole.RESEARCH,
            )

        assignable = await tm.get_assignable_tasks(p.id, limit=2)
        assert len(assignable) == 2


# ============================================================
# Token Budget
# ============================================================


class TestTokenBudget:
    """Tests for project token budget tracking."""

    async def test_record_tokens(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus, token_budget=10_000)

        t1 = await tm.create_task(
            project_id=p.id, title="Task", description="",
        )

        result = await tm.record_tokens(p.id, t1.id, 500)
        assert result is True

        usage = await tm.get_project_token_usage(p.id)
        assert usage["tokens_used"] == 500
        assert usage["remaining"] == 9_500

    async def test_token_budget_exceeded(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus, token_budget=1_000)

        t1 = await tm.create_task(
            project_id=p.id, title="Task", description="",
        )

        result = await tm.record_tokens(p.id, t1.id, 1_500)
        assert result is False

    async def test_record_tokens_zero_is_noop(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus, token_budget=10_000)

        result = await tm.record_tokens(p.id, "", 0)
        assert result is True

        tokens = await db.get_project_tokens(p.id)
        assert tokens == 0

    async def test_progress_includes_token_usage(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus, token_budget=100_000)

        t1 = await tm.create_task(
            project_id=p.id, title="Task", description="",
        )
        await tm.record_tokens(p.id, t1.id, 2_000)

        progress = await tm.get_project_progress(p.id)
        assert "token_usage" in progress
        assert progress["token_usage"]["tokens_used"] == 2_000


# ============================================================
# Deadlock Detection
# ============================================================


class TestDeadlockDetection:
    """Tests for Guardrails.detect_deadlock()."""

    def test_no_deadlock_with_assignable_tasks(self) -> None:
        g = Guardrails()
        tasks = [
            Task(project_id="p", title="A", description="", state=TaskState.PENDING),
            Task(project_id="p", title="B", description="", state=TaskState.IN_PROGRESS),
        ]
        assert g.detect_deadlock(tasks) is False

    def test_deadlock_all_failed(self) -> None:
        g = Guardrails()
        tasks = [
            Task(project_id="p", title="A", description="", state=TaskState.FAILED),
            Task(project_id="p", title="B", description="", state=TaskState.FAILED),
        ]
        assert g.detect_deadlock(tasks) is True

    def test_deadlock_pending_with_unmet_deps(self) -> None:
        g = Guardrails()
        t_failed = Task(
            id="t1", project_id="p", title="A", description="",
            state=TaskState.FAILED,
        )
        t_blocked = Task(
            id="t2", project_id="p", title="B", description="",
            state=TaskState.PENDING, depends_on=["t1"],
        )
        assert g.detect_deadlock([t_failed, t_blocked]) is True

    def test_no_deadlock_all_done(self) -> None:
        g = Guardrails()
        tasks = [
            Task(project_id="p", title="A", description="", state=TaskState.DONE),
        ]
        assert g.detect_deadlock(tasks) is False

    def test_no_deadlock_empty_tasks(self) -> None:
        g = Guardrails()
        assert g.detect_deadlock([]) is False

    def test_no_deadlock_pending_no_deps(self) -> None:
        """A PENDING task with no dependencies is not deadlocked."""
        g = Guardrails()
        tasks = [
            Task(project_id="p", title="A", description="", state=TaskState.PENDING),
        ]
        assert g.detect_deadlock(tasks) is False

    def test_deadlock_mixed_failed_and_blocked(self) -> None:
        g = Guardrails()
        t1 = Task(id="dep", project_id="p", title="Dep", description="", state=TaskState.FAILED)
        t2 = Task(
            id="blocked", project_id="p", title="Blocked", description="",
            state=TaskState.PENDING, depends_on=["dep"],
        )
        t3 = Task(id="also_failed", project_id="p", title="Also Failed", description="", state=TaskState.FAILED)
        assert g.detect_deadlock([t1, t2, t3]) is True


# ============================================================
# Activity Timeout
# ============================================================


class TestActivityTimeout:
    """Tests for Guardrails.check_activity_timeout()."""

    def test_recent_activity(self) -> None:
        g = Guardrails(activity_timeout=60)
        recent = datetime.now(timezone.utc)
        assert g.check_activity_timeout(recent) is True

    def test_stale_activity(self) -> None:
        g = Guardrails(activity_timeout=60)
        stale = datetime.now(timezone.utc) - timedelta(minutes=120)
        assert g.check_activity_timeout(stale) is False

    def test_activity_timeout_with_string_datetime(self) -> None:
        g = Guardrails(activity_timeout=60)
        stale = (datetime.now(timezone.utc) - timedelta(minutes=120)).isoformat()
        assert g.check_activity_timeout(stale) is False


# ============================================================
# Escalation
# ============================================================


class TestEscalation:
    """Tests that guardrail failures escalate to PM."""

    async def test_max_iterations_escalates(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """When max iterations are reached, PM gets a STATUS_UPDATE."""
        p = await _make_project(db)
        g = Guardrails(max_iterations=2)
        tm = TaskManager(db=db, bus=mock_bus, guardrails=g)

        task = await tm.create_task(
            project_id=p.id, title="Coding", description="Write code",
            assigned_to=AgentRole.CODER, max_iterations=2,
        )

        # Walk to REVISION twice
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.IN_PROGRESS)
        await tm.transition(task.id, TaskState.REVIEW)
        await tm.transition(task.id, TaskState.REVISION)  # iteration 1
        await tm.transition(task.id, TaskState.IN_PROGRESS)
        await tm.transition(task.id, TaskState.REVIEW)
        result = await tm.transition(task.id, TaskState.REVISION)  # iteration 2 → FAILED

        assert result.state == TaskState.FAILED

        # Check that an escalation message was published
        escalation_msgs = [
            m for m in mock_bus.published
            if m.recipient == AgentRole.PM
        ]
        assert len(escalation_msgs) == 1
        assert "guardrail" in escalation_msgs[0].payload.content.lower()


# ============================================================
# Pause / Resume
# ============================================================


class TestPauseResume:
    """Tests for project pause/resume and agent skip behavior."""

    async def test_paused_project_skips_processing(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """Messages for paused projects are skipped."""
        p = await _make_project(db)
        await db.update_project(p.id, status="paused")

        agent = ResearchAgent(bus=mock_bus, ollama=mock_ollama, db=db)

        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=p.id,
            content="Research something",
        )

        # _handle_message should return without calling process_message
        await agent._handle_message(msg)

        # No LLM calls should have been made
        assert len(mock_ollama.chat_calls) == 0

    async def test_active_project_processes_normally(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """Active project messages are processed normally."""
        p = await _make_project(db)

        agent = ResearchAgent(bus=mock_bus, ollama=mock_ollama, db=db)

        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=p.id,
            content="Research something",
        )

        await agent._handle_message(msg)

        # LLM should have been called
        assert len(mock_ollama.chat_calls) > 0


# ============================================================
# Task Retry (FAILED → PENDING)
# ============================================================


class TestTaskRetry:
    """Tests for manual task retry."""

    async def test_retry_failed_task(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus)

        task = await tm.create_task(
            project_id=p.id, title="Coding", description="",
            assigned_to=AgentRole.CODER,
        )

        # Walk to FAILED
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.FAILED)

        # Retry: FAILED → PENDING
        retried = await tm.transition(task.id, TaskState.PENDING)
        assert retried.state == TaskState.PENDING

    async def test_cannot_retry_non_failed_task(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus)

        task = await tm.create_task(
            project_id=p.id, title="Coding", description="",
        )

        with pytest.raises(ValueError, match="Invalid transition"):
            await tm.transition(task.id, TaskState.IN_PROGRESS)


# ============================================================
# PM Parallel Dispatch
# ============================================================


class TestPMParallelDispatch:
    """Tests that PM dispatches multiple independent tasks."""

    async def test_parallel_dispatch(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """PM should dispatch multiple assignable tasks at once."""
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus)
        pm = PMAgent(bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm)

        # Create two independent tasks
        await tm.create_task(
            project_id=p.id, title="Research A", description="Research topic A",
            assigned_to=AgentRole.RESEARCH,
        )
        await tm.create_task(
            project_id=p.id, title="Research B", description="Research topic B",
            assigned_to=AgentRole.RESEARCH,
        )

        # The first message is returned, the second published directly
        result = await pm._assign_next_task(p.id)
        assert result is not None
        assert result.type == MessageType.TASK_ASSIGNMENT

        # Check that additional messages were published for parallel tasks
        task_assignments = [
            m for m in mock_bus.published
            if m.type == MessageType.TASK_ASSIGNMENT
        ]
        # We expect 1 published (the second task), the first is returned
        assert len(task_assignments) >= 1

    async def test_parallel_dispatch_respects_deps(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """PM should not dispatch tasks with unmet dependencies."""
        p = await _make_project(db)
        tm = await _make_tm(db, mock_bus)
        pm = PMAgent(bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm)

        t1 = await tm.create_task(
            project_id=p.id, title="Research", description="Research topic",
            assigned_to=AgentRole.RESEARCH,
        )
        await tm.create_task(
            project_id=p.id, title="Spec", description="Write spec",
            assigned_to=AgentRole.SPEC, depends_on=[t1.id],
        )

        result = await pm._assign_next_task(p.id)
        assert result is not None

        # Only one task should be dispatched (the independent one)
        all_assignments = [
            m for m in mock_bus.published
            if m.type == MessageType.TASK_ASSIGNMENT
        ]
        # Plus the returned result — total should be 1 independent task
        assert len(all_assignments) == 0  # Only the returned message, nothing extra published


# ============================================================
# PM Task Parsing with Dependencies
# ============================================================


class TestPMDependencyParsing:
    """Tests for PM parsing of DEPENDS_ON: in LLM output."""

    def test_parse_tasks_with_depends_on(self) -> None:
        pm = PMAgent.__new__(PMAgent)
        response = """TASK: Research auth
AGENT: research
PRIORITY: high
DEPENDS_ON: none
DESCRIPTION: Research authentication approaches.

TASK: Write spec
AGENT: spec
PRIORITY: normal
DEPENDS_ON: Research auth
DESCRIPTION: Write the API specification based on research.
"""
        tasks = pm._parse_tasks(response, "proj-1")
        assert len(tasks) == 2
        assert tasks[0]["depends_on_titles"] == []
        assert tasks[1]["depends_on_titles"] == ["Research auth"]

    def test_parse_tasks_without_depends_on(self) -> None:
        pm = PMAgent.__new__(PMAgent)
        response = """TASK: Research
AGENT: research
PRIORITY: normal
DESCRIPTION: Do some research.
"""
        tasks = pm._parse_tasks(response, "proj-1")
        assert len(tasks) == 1
        assert tasks[0]["depends_on_titles"] == []

    def test_parse_tasks_multiple_deps(self) -> None:
        pm = PMAgent.__new__(PMAgent)
        response = """TASK: Implement backend
AGENT: coder
PRIORITY: high
DEPENDS_ON: Research auth, Write spec
DESCRIPTION: Build the backend.
"""
        tasks = pm._parse_tasks(response, "proj-1")
        assert len(tasks) == 1
        assert tasks[0]["depends_on_titles"] == ["Research auth", "Write spec"]


# ============================================================
# Guardrails — Token Budget
# ============================================================


class TestGuardrailsTokenBudget:
    """Tests for Guardrails.check_token_budget()."""

    def test_within_budget(self) -> None:
        g = Guardrails(project_token_budget=100_000)
        assert g.check_token_budget(50_000) is True

    def test_at_budget(self) -> None:
        g = Guardrails(project_token_budget=100_000)
        assert g.check_token_budget(100_000) is False

    def test_over_budget(self) -> None:
        g = Guardrails(project_token_budget=100_000)
        assert g.check_token_budget(150_000) is False


# ============================================================
# Pipeline Health Check
# ============================================================


class TestPipelineHealthCheck:
    """Tests for Pipeline._check_health()."""

    async def test_health_check_detects_deadlock(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """Pipeline emits a deadlock event when detected."""
        p = await _make_project(db)
        tm = TaskManager(db=db, bus=mock_bus)

        task = await tm.create_task(
            project_id=p.id, title="Stuck", description="",
            assigned_to=AgentRole.CODER,
        )
        # Set to FAILED directly
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.FAILED)

        pipeline = Pipeline(task_manager=tm, bus=mock_bus)
        await pipeline._check_health()

        deadlock_events = [
            (et, data) for et, data in mock_bus.ui_events
            if et == "project_deadlock"
        ]
        assert len(deadlock_events) == 1

    async def test_health_check_token_budget_exceeded(
        self, db: Database, mock_bus: MockMessageBus
    ) -> None:
        """Pipeline fails a project when token budget is exceeded."""
        p = await _make_project(db)
        g = Guardrails(project_token_budget=100)
        tm = TaskManager(db=db, bus=mock_bus, guardrails=g)

        # Artificially exceed the budget
        await db.add_project_tokens(p.id, 200)

        pipeline = Pipeline(task_manager=tm, bus=mock_bus)
        pipeline.guardrails = g
        await pipeline._check_health()

        # Project should be failed
        project = await db.get_project(p.id)
        assert project is not None
        assert project.status == "failed"


# ============================================================
# Database Migration V3
# ============================================================


class TestDatabaseV3:
    """Tests that the V3 schema fields are persisted correctly."""

    async def test_depends_on_round_trip(self, db: Database) -> None:
        p = await _make_project(db)
        task = Task(
            project_id=p.id, title="T", description="",
            depends_on=["dep-1", "dep-2"],
        )
        await db.create_task(task)

        fetched = await db.get_task(task.id)
        assert fetched is not None
        assert fetched.depends_on == ["dep-1", "dep-2"]

    async def test_tokens_used_round_trip(self, db: Database) -> None:
        p = await _make_project(db)
        task = Task(
            project_id=p.id, title="T", description="",
            tokens_used=42,
        )
        await db.create_task(task)

        fetched = await db.get_task(task.id)
        assert fetched is not None
        assert fetched.tokens_used == 42

    async def test_add_project_tokens(self, db: Database) -> None:
        p = await _make_project(db)
        new_total = await db.add_project_tokens(p.id, 100)
        assert new_total == 100

        new_total = await db.add_project_tokens(p.id, 200)
        assert new_total == 300

    async def test_project_tokens_used_default(self, db: Database) -> None:
        p = await _make_project(db)
        tokens = await db.get_project_tokens(p.id)
        assert tokens == 0
