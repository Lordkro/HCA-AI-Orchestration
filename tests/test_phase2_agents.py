"""Phase 2 tests — Agent ↔ TaskManager integration.

Tests that agents correctly:
- Parse LLM plans into Task records (PM)
- Transition task state on pickup and delivery
- Route Critic results through the state machine
- Forward artifact_type through the Critic
- Assign the next pending task automatically
"""

from __future__ import annotations

import pytest

from src.agents.coder_agent import CoderAgent
from src.agents.critic_agent import CriticAgent
from src.agents.pm_agent import PMAgent
from src.agents.research_agent import ResearchAgent
from src.agents.spec_agent import SpecAgent
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
from src.orchestrator.task_manager import TaskManager
from tests.conftest import MockMessageBus, MockOllamaClient, make_message


# ============================================================
# Helpers
# ============================================================


async def _setup_pm(
    db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
) -> PMAgent:
    """Create a PM agent wired to a real TaskManager."""
    tm = TaskManager(db=db, bus=mock_bus)
    return PMAgent(bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm)


async def _setup_project(db: Database) -> Project:
    p = Project(name="Test Project", description="Test idea", idea="Build a TODO app")
    await db.create_project(p)
    return p


# ============================================================
# PM Task Parsing
# ============================================================


class TestPMTaskParsing:
    """Tests for PMAgent._parse_tasks()."""

    def test_parse_well_formed_tasks(self) -> None:
        pm = PMAgent.__new__(PMAgent)  # skip __init__ — just test the parser
        response = """Here is the project plan:

TASK: Research auth libraries
AGENT: research
PRIORITY: high
DESCRIPTION: Investigate JWT vs session-based authentication for Python REST APIs.

TASK: Write API specification
AGENT: spec
PRIORITY: normal
DESCRIPTION: Create a full OpenAPI spec for the TODO app endpoints.

TASK: Implement backend
AGENT: coder
PRIORITY: high
DESCRIPTION: Build the FastAPI backend following the specification.
Include CRUD endpoints and authentication middleware.
"""
        tasks = pm._parse_tasks(response, "proj-1")
        assert len(tasks) == 3
        assert tasks[0]["title"] == "Research auth libraries"
        assert tasks[0]["agent"] == "research"
        assert tasks[0]["priority"] == "high"
        assert "JWT" in tasks[0]["description"]
        assert tasks[1]["agent"] == "spec"
        assert tasks[2]["agent"] == "coder"

    def test_parse_skips_critic_tasks(self) -> None:
        pm = PMAgent.__new__(PMAgent)
        response = """TASK: Review code quality
AGENT: critic
PRIORITY: normal
DESCRIPTION: Review all code for quality.

TASK: Research databases
AGENT: research
PRIORITY: normal
DESCRIPTION: Research the best database for this project.
"""
        tasks = pm._parse_tasks(response, "proj-1")
        assert len(tasks) == 1
        assert tasks[0]["agent"] == "research"

    def test_parse_empty_response(self) -> None:
        pm = PMAgent.__new__(PMAgent)
        tasks = pm._parse_tasks("No structured tasks here.", "proj-1")
        assert len(tasks) == 0

    def test_parse_invalid_agent_skipped(self) -> None:
        pm = PMAgent.__new__(PMAgent)
        response = """TASK: Do magic
AGENT: wizard
PRIORITY: normal
DESCRIPTION: Cast spells.
"""
        tasks = pm._parse_tasks(response, "proj-1")
        assert len(tasks) == 0

    def test_parse_default_priority(self) -> None:
        pm = PMAgent.__new__(PMAgent)
        response = """TASK: Quick research
AGENT: research
DESCRIPTION: Look into caching options.
"""
        tasks = pm._parse_tasks(response, "proj-1")
        assert len(tasks) == 1
        assert tasks[0]["priority"] == "normal"

    def test_parse_invalid_priority_defaults_to_normal(self) -> None:
        pm = PMAgent.__new__(PMAgent)
        response = """TASK: Quick research
AGENT: research
PRIORITY: ultra
DESCRIPTION: Look into caching options.
"""
        tasks = pm._parse_tasks(response, "proj-1")
        assert len(tasks) == 1
        assert tasks[0]["priority"] == "normal"


# ============================================================
# PM — New Project Creates Tasks
# ============================================================


class TestPMNewProject:
    """Tests that _handle_new_project persists tasks and assigns the first."""

    async def test_new_project_creates_tasks(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """When the LLM returns a parseable plan, tasks are persisted."""
        mock_ollama.default_response = """TASK: Research tech stack
AGENT: research
PRIORITY: high
DESCRIPTION: Research the best tech stack for a TODO app.

TASK: Write specification
AGENT: spec
PRIORITY: normal
DESCRIPTION: Write the API specification.
"""
        pm = await _setup_pm(db, mock_bus, mock_ollama)
        project = await _setup_project(db)

        msg = make_message(
            sender=AgentRole.USER,
            recipient=AgentRole.PM,
            msg_type=MessageType.SYSTEM,
            project_id=project.id,
            content="Build a TODO app",
        )
        result = await pm.process_message(msg)

        # Should have created 2 tasks
        tasks = await db.list_tasks(project.id)
        assert len(tasks) == 2
        assert tasks[0].title == "Research tech stack"
        assert tasks[0].assigned_to == AgentRole.RESEARCH

        # First task should be assigned (PENDING → ASSIGNED)
        first_task = await db.get_task(tasks[0].id)
        assert first_task.state == TaskState.ASSIGNED

        # Should return a TASK_ASSIGNMENT message for the first task
        assert result is not None
        assert result.type == MessageType.TASK_ASSIGNMENT
        assert result.recipient == AgentRole.RESEARCH

    async def test_new_project_fallback_on_parse_failure(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """When the LLM returns unparseable text, a single fallback task is created."""
        mock_ollama.default_response = "Here's a vague plan without any structure."
        pm = await _setup_pm(db, mock_bus, mock_ollama)
        project = await _setup_project(db)

        msg = make_message(
            sender=AgentRole.USER,
            recipient=AgentRole.PM,
            msg_type=MessageType.SYSTEM,
            project_id=project.id,
            content="Build something cool",
        )
        result = await pm.process_message(msg)

        tasks = await db.list_tasks(project.id)
        assert len(tasks) == 1
        assert tasks[0].assigned_to == AgentRole.RESEARCH
        assert result is not None


# ============================================================
# Agent State Transitions on Task Pickup
# ============================================================


class TestAgentTaskTransitions:
    """Tests that agents transition task state when they pick up work."""

    async def test_research_transitions_to_in_progress(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        task = await tm.create_task(
            project_id=project.id,
            title="Research task",
            description="Research something",
            assigned_to=AgentRole.RESEARCH,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)

        agent = ResearchAgent(
            bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm
        )
        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            task_id=task.id,
            content="Research auth",
        )
        await agent.process_message(msg)

        updated = await db.get_task(task.id)
        assert updated.state == TaskState.IN_PROGRESS

    async def test_spec_transitions_to_in_progress(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        task = await tm.create_task(
            project_id=project.id,
            title="Spec task",
            description="Write spec",
            assigned_to=AgentRole.SPEC,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)

        agent = SpecAgent(
            bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm
        )
        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.SPEC,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            task_id=task.id,
            content="Write the spec",
        )
        await agent.process_message(msg)

        updated = await db.get_task(task.id)
        assert updated.state == TaskState.IN_PROGRESS

    async def test_coder_transitions_to_in_progress(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        task = await tm.create_task(
            project_id=project.id,
            title="Code task",
            description="Write code",
            assigned_to=AgentRole.CODER,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)

        agent = CoderAgent(
            bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm
        )
        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.CODER,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            task_id=task.id,
            content="Implement the backend",
        )
        await agent.process_message(msg)

        updated = await db.get_task(task.id)
        assert updated.state == TaskState.IN_PROGRESS


# ============================================================
# Critic — artifact_type Forwarding
# ============================================================


class TestCriticMetadata:
    """Tests that the Critic forwards the original artifact_type."""

    async def test_approved_carries_original_artifact_type(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        mock_ollama.default_response = "**APPROVED**\n\nGreat work!"
        agent = CriticAgent(bus=mock_bus, ollama=mock_ollama, db=db)

        msg = make_message(
            sender=AgentRole.CODER,
            recipient=AgentRole.CRITIC,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id="proj-1",
            content="Some code output",
            metadata={"artifact_type": "code"},
        )
        result = await agent.process_message(msg)

        assert result is not None
        assert result.payload.metadata["review_result"] == "approved"
        assert result.payload.metadata["artifact_type"] == "code"

    async def test_rejected_carries_original_artifact_type(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        mock_ollama.default_response = "**NEEDS REVISION**\n\nFix the SQL injection."
        agent = CriticAgent(bus=mock_bus, ollama=mock_ollama, db=db)

        msg = make_message(
            sender=AgentRole.SPEC,
            recipient=AgentRole.CRITIC,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id="proj-1",
            content="A specification",
            metadata={"artifact_type": "specification"},
        )
        result = await agent.process_message(msg)

        assert result is not None
        assert result.payload.metadata["review_result"] == "needs_revision"
        assert result.payload.metadata["artifact_type"] == "specification"


# ============================================================
# PM — Deliverable Handling with State Machine
# ============================================================


class TestPMDeliverable:
    """Tests that PM drives the state machine on deliverables."""

    async def test_critic_approved_closes_task(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        pm = PMAgent(bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm)

        # Create and advance a task to REVIEW
        task = await tm.create_task(
            project_id=project.id,
            title="Code task",
            description="Write code",
            assigned_to=AgentRole.CODER,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.IN_PROGRESS)
        await tm.transition(task.id, TaskState.REVIEW)

        # Simulate Critic approval
        msg = make_message(
            sender=AgentRole.CRITIC,
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=project.id,
            task_id=task.id,
            content="**APPROVED** — looks great",
            metadata={"review_result": "approved", "artifact_type": "code"},
        )
        await pm.process_message(msg)

        updated = await db.get_task(task.id)
        assert updated.state == TaskState.DONE

    async def test_critic_rejected_transitions_to_revision(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        pm = PMAgent(bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm)

        task = await tm.create_task(
            project_id=project.id,
            title="Code task",
            description="Write code",
            assigned_to=AgentRole.CODER,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.IN_PROGRESS)
        await tm.transition(task.id, TaskState.REVIEW)

        msg = make_message(
            sender=AgentRole.CRITIC,
            recipient=AgentRole.PM,
            msg_type=MessageType.FEEDBACK,
            project_id=project.id,
            task_id=task.id,
            content="**NEEDS REVISION** — fix the SQL injection",
            metadata={"review_result": "needs_revision", "artifact_type": "code"},
        )
        result = await pm.process_message(msg)

        updated = await db.get_task(task.id)
        assert updated.state == TaskState.REVISION

        # PM should route feedback to the coder (artifact_type = code)
        assert result is not None
        assert result.recipient == AgentRole.CODER
        assert result.type == MessageType.FEEDBACK

    async def test_approved_task_triggers_next_assignment(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """After approving a task, PM should assign the next pending one."""
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        pm = PMAgent(bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm)

        # Create two tasks
        task1 = await tm.create_task(
            project_id=project.id,
            title="Task 1",
            description="First task",
            assigned_to=AgentRole.RESEARCH,
        )
        task2 = await tm.create_task(
            project_id=project.id,
            title="Task 2",
            description="Second task",
            assigned_to=AgentRole.SPEC,
        )

        # Advance task1 to REVIEW
        await tm.transition(task1.id, TaskState.ASSIGNED)
        await tm.transition(task1.id, TaskState.IN_PROGRESS)
        await tm.transition(task1.id, TaskState.REVIEW)

        # Approve task1
        msg = make_message(
            sender=AgentRole.CRITIC,
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=project.id,
            task_id=task1.id,
            content="Approved",
            metadata={"review_result": "approved", "artifact_type": "research_report"},
        )
        result = await pm.process_message(msg)

        # task1 should be DONE
        t1 = await db.get_task(task1.id)
        assert t1.state == TaskState.DONE

        # task2 should now be ASSIGNED (via _assign_next_task)
        t2 = await db.get_task(task2.id)
        assert t2.state == TaskState.ASSIGNED

        # Should have returned a TASK_ASSIGNMENT for task2
        assert result is not None
        assert result.task_id == task2.id
        assert result.recipient == AgentRole.SPEC

    async def test_all_tasks_done_completes_project(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """When all tasks are done, project status → completed."""
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        pm = PMAgent(bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm)

        task = await tm.create_task(
            project_id=project.id,
            title="Only task",
            description="The only task",
            assigned_to=AgentRole.CODER,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.IN_PROGRESS)
        await tm.transition(task.id, TaskState.REVIEW)

        msg = make_message(
            sender=AgentRole.CRITIC,
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=project.id,
            task_id=task.id,
            content="Approved",
            metadata={"review_result": "approved", "artifact_type": "code"},
        )
        result = await pm.process_message(msg)

        # No more tasks → result is None
        assert result is None

        # Project should be completed
        proj = await db.get_project(project.id)
        assert proj.status == "completed"


# ============================================================
# BaseAgent._transition_task
# ============================================================


class TestBaseTransitionHelper:
    """Tests for the BaseAgent._transition_task helper."""

    async def test_transition_success(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        task = await tm.create_task(
            project_id=project.id,
            title="Test",
            description="",
            assigned_to=AgentRole.RESEARCH,
        )
        agent = ResearchAgent(
            bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm
        )
        ok = await agent._transition_task(task.id, TaskState.ASSIGNED)
        assert ok is True
        updated = await db.get_task(task.id)
        assert updated.state == TaskState.ASSIGNED

    async def test_transition_invalid_returns_false(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        task = await tm.create_task(
            project_id=project.id,
            title="Test",
            description="",
        )
        agent = ResearchAgent(
            bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm
        )
        # PENDING → IN_PROGRESS is invalid (must go through ASSIGNED first)
        ok = await agent._transition_task(task.id, TaskState.IN_PROGRESS)
        assert ok is False

    async def test_transition_without_task_manager(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        agent = ResearchAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        ok = await agent._transition_task("some-id", TaskState.ASSIGNED)
        assert ok is False

    async def test_transition_empty_task_id(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        tm = TaskManager(db=db, bus=mock_bus)
        agent = ResearchAgent(
            bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm
        )
        ok = await agent._transition_task("", TaskState.ASSIGNED)
        assert ok is False
