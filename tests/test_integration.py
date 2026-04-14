"""Full end-to-end integration tests.

These tests exercise the *complete* lifecycle of a project:

    User idea → PM decomposes → Research → Spec → Coder → Critic review
    → PM approves / routes revision → Project completes

Every layer is involved:
- Database (real SQLite in temp dir)
- TaskManager + Guardrails (real)
- All five agents (PM, Research, Spec, Coder, Critic)
- Message routing (mock bus — messages are delivered manually so we can
  assert intermediate states without needing real Redis or async loops)
- OllamaClient (mock — returns structured responses that agents can parse)

No real LLM or Redis required.  Tests are deterministic and fast.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

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
from src.orchestrator.guardrails import Guardrails
from src.orchestrator.pipeline import Pipeline
from src.orchestrator.task_manager import TaskManager
from tests.conftest import MockMessageBus, MockOllamaClient, make_message


# ============================================================
# Helpers
# ============================================================

# A structured LLM response that PMAgent._parse_tasks can parse
# into three tasks with dependencies.
PM_PLAN_RESPONSE = """
TASK: Research authentication approaches
AGENT: research
PRIORITY: high
DEPENDS_ON: none
DESCRIPTION: Investigate OAuth2, JWT, and session-based auth for a Python REST API.

TASK: Write API specification
AGENT: spec
PRIORITY: normal
DEPENDS_ON: Research authentication approaches
DESCRIPTION: Based on the research, write a detailed OpenAPI specification for the REST API.

TASK: Implement API server
AGENT: coder
PRIORITY: normal
DEPENDS_ON: Write API specification
DESCRIPTION: Implement the full API server with authentication endpoints based on the specification.
""".strip()

RESEARCH_RESPONSE = "Research report: Use JWT with RS256 for stateless auth. Recommended library: PyJWT."

SPEC_RESPONSE = "API Specification: POST /auth/login, POST /auth/register, GET /auth/me. Use Bearer tokens."

CODER_RESPONSE = """Here is the implementation:

=== FILE: src/auth/routes.py ===
```python
from fastapi import APIRouter
router = APIRouter()

@router.post("/auth/login")
async def login():
    return {"token": "jwt-token"}
```

=== FILE: tests/test_auth.py ===
```python
def test_login():
    assert True
```
"""

CRITIC_APPROVED_RESPONSE = """**APPROVED**

**Summary**: The implementation follows the specification correctly.

**Issues Found**: None critical.

**Strengths**: Clean code, proper type hints.

**Recommendations**: Consider adding rate limiting.
"""

CRITIC_REVISION_RESPONSE = """**NEEDS REVISION**

**Summary**: The implementation has issues that need to be addressed.

**Issues Found**:
1. (major) Missing input validation on login endpoint
2. (minor) No error handling for invalid credentials

**Strengths**: Good project structure.

**Recommendations**: Add Pydantic models for request validation.
"""


async def _create_project(db: Database) -> Project:
    """Create and persist a test project."""
    project = Project(
        name="Auth API",
        description="Build a REST API with authentication",
        idea="Build a REST API with JWT authentication for a web app",
    )
    await db.create_project(project)
    return project


def _build_agents(
    *,
    db: Database,
    bus: MockMessageBus,
    ollama: MockOllamaClient,
    task_manager: TaskManager,
) -> dict[AgentRole, PMAgent | ResearchAgent | SpecAgent | CoderAgent | CriticAgent]:
    """Instantiate all five agents with shared dependencies."""
    return {
        AgentRole.PM: PMAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
        AgentRole.RESEARCH: ResearchAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
        AgentRole.SPEC: SpecAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
        AgentRole.CODER: CoderAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
        AgentRole.CRITIC: CriticAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
    }


def _find_message(bus: MockMessageBus, recipient: AgentRole) -> AgentMessage | None:
    """Return the most recent published message addressed to *recipient*."""
    for msg in reversed(bus.published):
        if msg.recipient == recipient:
            return msg
    return None


def _find_messages(bus: MockMessageBus, recipient: AgentRole) -> list[AgentMessage]:
    """Return all published messages addressed to *recipient* in order."""
    return [m for m in bus.published if m.recipient == recipient]


def _drain_messages_to(
    bus: MockMessageBus, recipient: AgentRole
) -> list[AgentMessage]:
    """Pop and return all messages addressed to *recipient*, leaving the rest."""
    matched = [m for m in bus.published if m.recipient == recipient]
    bus.published = [m for m in bus.published if m.recipient != recipient]
    return matched


# ============================================================
# Integration Tests
# ============================================================


class TestHappyPath:
    """Full happy-path: idea → plan → research → spec → code → approve → done."""

    @pytest.fixture
    async def env(self, db: Database):
        """Set up the full environment: project, agents, task manager."""
        bus = MockMessageBus()
        ollama = MockOllamaClient()
        guardrails = Guardrails(max_iterations=5, max_tasks=20)
        tm = TaskManager(db=db, bus=bus, guardrails=guardrails)
        project = await _create_project(db)
        agents = _build_agents(db=db, bus=bus, ollama=ollama, task_manager=tm)
        return {
            "db": db,
            "bus": bus,
            "ollama": ollama,
            "tm": tm,
            "project": project,
            "agents": agents,
        }

    @pytest.mark.asyncio
    async def test_full_project_lifecycle(self, env):
        """Walk through the entire project lifecycle step by step.

        Flow:
        1. User sends idea → PM
        2. PM decomposes into 3 tasks (research → spec → coder) with deps
        3. PM assigns first assignable task (research)
        4. Research agent processes → sends deliverable to PM
        5. PM receives deliverable → transitions to REVIEW → routes to Spec
        6. PM assigns spec task (dep on research now met)
        7. Spec processes → sends deliverable to PM
        8. PM routes to Coder, assigns coder task
        9. Coder processes → sends deliverable to PM
        10. PM routes to Critic
        11. Critic approves → PM marks DONE, project complete
        """
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        ollama: MockOllamaClient = env["ollama"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]
        agents = env["agents"]
        pm: PMAgent = agents[AgentRole.PM]
        research: ResearchAgent = agents[AgentRole.RESEARCH]
        spec: SpecAgent = agents[AgentRole.SPEC]
        coder: CoderAgent = agents[AgentRole.CODER]
        critic: CriticAgent = agents[AgentRole.CRITIC]

        # ── Step 1: User submits idea → PM ──────────────────────
        ollama.default_response = PM_PLAN_RESPONSE

        user_msg = make_message(
            sender=AgentRole.USER,
            recipient=AgentRole.PM,
            msg_type=MessageType.SYSTEM,
            project_id=project.id,
            content=project.idea,
        )

        # Deliver the message through the full _handle_message path
        # (checks pause, saves to DB, calls process_message)
        await pm._handle_message(user_msg)

        # Verify: 3 tasks were created
        tasks = await db.list_tasks(project.id)
        assert len(tasks) == 3
        titles = {t.title for t in tasks}
        assert "Research authentication approaches" in titles
        assert "Write API specification" in titles
        assert "Implement API server" in titles

        # Verify: dependencies were wired correctly
        task_by_title = {t.title: t for t in tasks}
        research_task = task_by_title["Research authentication approaches"]
        spec_task = task_by_title["Write API specification"]
        coder_task = task_by_title["Implement API server"]

        assert research_task.depends_on == []
        assert spec_task.depends_on == [research_task.id]
        assert coder_task.depends_on == [spec_task.id]

        # Verify: research task was assigned (PENDING → ASSIGNED)
        research_task = await db.get_task(research_task.id)
        assert research_task.state == TaskState.ASSIGNED

        # Verify: spec/coder still pending (deps not met)
        spec_task = await db.get_task(spec_task.id)
        assert spec_task.state == TaskState.PENDING
        coder_task = await db.get_task(coder_task.id)
        assert coder_task.state == TaskState.PENDING

        # Verify: an assignment message was sent to the research agent
        research_msgs = _find_messages(bus, AgentRole.RESEARCH)
        assert len(research_msgs) >= 1
        assert research_msgs[-1].type == MessageType.TASK_ASSIGNMENT

        # ── Step 2: Research agent processes the task ────────────
        bus.published.clear()
        ollama.default_response = RESEARCH_RESPONSE

        research_assignment = research_msgs[-1]
        await research._handle_message(research_assignment)

        # Verify: task moved to IN_PROGRESS during processing
        # (it may have been transitioned already; check the deliverable was sent)
        pm_msgs = _find_messages(bus, AgentRole.PM)
        assert len(pm_msgs) >= 1
        deliverable_msg = pm_msgs[-1]
        assert deliverable_msg.type == MessageType.DELIVERABLE
        assert deliverable_msg.payload.metadata.get("artifact_type") == "research_report"

        # ── Step 3: PM receives research deliverable ─────────────
        # PM routes it through the standard pipeline (research → spec agent)
        bus.published.clear()
        ollama.default_response = "Please proceed with the API spec based on the research."

        await pm._handle_message(deliverable_msg)

        # The PM should:
        # 1. Transition research task: IN_PROGRESS → REVIEW
        research_task = await db.get_task(research_task.id)
        assert research_task.state == TaskState.REVIEW

        # 2. Send a TASK_ASSIGNMENT to the spec agent (next in pipeline)
        spec_msgs = _find_messages(bus, AgentRole.SPEC)
        assert len(spec_msgs) >= 1

        # ── Step 4: Simulate Critic approving the research ────────
        # In real flow, PM routes to next agent, but let's also close
        # the research task so spec task's dependency is met.
        bus.published.clear()

        critic_approval = make_message(
            sender=AgentRole.CRITIC,
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=project.id,
            task_id=research_task.id,
            content="Research looks thorough and well-structured.",
            metadata={"review_result": "approved", "artifact_type": "research_report"},
        )

        ollama.default_response = "Moving on to the specification phase."
        await pm._handle_message(critic_approval)

        # Research task should be DONE now
        research_task = await db.get_task(research_task.id)
        assert research_task.state == TaskState.DONE

        # PM should have assigned the spec task (dep on research is now DONE)
        spec_task = await db.get_task(spec_task.id)
        assert spec_task.state == TaskState.ASSIGNED

        # ── Step 5: Spec agent processes the task ────────────────
        bus.published.clear()
        ollama.default_response = SPEC_RESPONSE

        spec_assignment = _find_messages(bus, AgentRole.SPEC)
        # The PM should have published a task assignment for spec
        # Check bus for the message sent via _assign_next_task
        # (it was published during critic_approval handling)
        # Get latest from all bus messages
        all_spec_msgs = [m for m in bus.published if m.recipient == AgentRole.SPEC]

        # The PM sent the assignment as the return value of _handle_message
        # but it was also published. Let's find it in the full bus history.
        # Since we cleared bus.published, we need to re-send the assignment.
        spec_assignment_msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.SPEC,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            task_id=spec_task.id,
            content="Write the API specification based on research findings.",
        )

        await spec._handle_message(spec_assignment_msg)

        # Spec should be IN_PROGRESS then send deliverable to PM
        pm_msgs = _find_messages(bus, AgentRole.PM)
        assert len(pm_msgs) >= 1
        spec_deliverable = pm_msgs[-1]
        assert spec_deliverable.type == MessageType.DELIVERABLE
        assert spec_deliverable.payload.metadata.get("artifact_type") == "specification"

        # ── Step 6a: PM routes spec deliverable → Critic ─────────
        bus.published.clear()
        ollama.default_response = "Spec looks good, routing to next agent."
        await pm._handle_message(spec_deliverable)

        # Spec task should be in REVIEW now
        spec_task = await db.get_task(spec_task.id)
        assert spec_task.state == TaskState.REVIEW

        # ── Step 6b: Critic approves spec, PM assigns coder task ──
        bus.published.clear()

        critic_approval_spec = make_message(
            sender=AgentRole.CRITIC,
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=project.id,
            task_id=spec_task.id,
            content="Specification is complete and well-structured.",
            metadata={"review_result": "approved", "artifact_type": "specification"},
        )

        ollama.default_response = "Spec approved. Moving to implementation."
        await pm._handle_message(critic_approval_spec)

        spec_task = await db.get_task(spec_task.id)
        assert spec_task.state == TaskState.DONE

        coder_task = await db.get_task(coder_task.id)
        assert coder_task.state == TaskState.ASSIGNED

        # ── Step 7: Coder implements ─────────────────────────────
        bus.published.clear()
        ollama.default_response = CODER_RESPONSE

        coder_assignment_msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.CODER,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            task_id=coder_task.id,
            content="Implement the API server per the specification.",
        )

        await coder._handle_message(coder_assignment_msg)

        # Coder should produce artifacts and send deliverable
        pm_msgs = _find_messages(bus, AgentRole.PM)
        assert len(pm_msgs) >= 1
        coder_deliverable = pm_msgs[-1]
        assert coder_deliverable.type == MessageType.DELIVERABLE
        assert coder_deliverable.payload.metadata.get("artifact_type") == "code"

        # Verify artifacts were saved to DB
        artifacts = await db.list_artifacts(project.id)
        assert len(artifacts) >= 1
        filenames = {a.filename for a in artifacts}
        assert "src/auth/routes.py" in filenames

        # ── Step 8: PM routes coder deliverable → Critic ─────────
        bus.published.clear()
        ollama.default_response = "Code submitted, routing to Critic for review."
        await pm._handle_message(coder_deliverable)

        # Task should be in REVIEW
        coder_task = await db.get_task(coder_task.id)
        assert coder_task.state == TaskState.REVIEW

        # A message should be sent to the Critic
        critic_msgs = _find_messages(bus, AgentRole.CRITIC)
        assert len(critic_msgs) >= 1

        # ── Step 9: Critic approves code → PM marks DONE ────────
        bus.published.clear()
        ollama.default_response = CRITIC_APPROVED_RESPONSE

        critic_review_msg = make_message(
            sender=AgentRole.CRITIC,
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=project.id,
            task_id=coder_task.id,
            content=CRITIC_APPROVED_RESPONSE,
            metadata={"review_result": "approved", "artifact_type": "code"},
        )

        await pm._handle_message(critic_review_msg)

        # All tasks should be DONE
        coder_task = await db.get_task(coder_task.id)
        assert coder_task.state == TaskState.DONE

        # Project should be marked completed
        project_final = await db.get_project(project.id)
        assert project_final.status == "completed"

        # Verify progress is 100%
        progress = await tm.get_project_progress(project.id)
        assert progress["completed"] == 3
        assert progress["total_tasks"] == 3
        assert progress["progress_pct"] == 100.0

    @pytest.mark.asyncio
    async def test_task_state_machine_integrity(self, env):
        """Verify every task passes through the expected state sequence."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        ollama: MockOllamaClient = env["ollama"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]
        agents = env["agents"]
        pm: PMAgent = agents[AgentRole.PM]

        ollama.default_response = PM_PLAN_RESPONSE

        user_msg = make_message(
            sender=AgentRole.USER,
            recipient=AgentRole.PM,
            msg_type=MessageType.SYSTEM,
            project_id=project.id,
            content=project.idea,
        )
        await pm._handle_message(user_msg)

        tasks = await db.list_tasks(project.id)
        research_task = next(t for t in tasks if t.assigned_to == AgentRole.RESEARCH)

        # Valid transitions check
        assert research_task.state == TaskState.ASSIGNED

        # ASSIGNED → IN_PROGRESS
        await tm.transition(research_task.id, TaskState.IN_PROGRESS)
        research_task = await db.get_task(research_task.id)
        assert research_task.state == TaskState.IN_PROGRESS

        # IN_PROGRESS → REVIEW
        await tm.transition(research_task.id, TaskState.REVIEW)
        research_task = await db.get_task(research_task.id)
        assert research_task.state == TaskState.REVIEW

        # REVIEW → APPROVED
        await tm.transition(research_task.id, TaskState.APPROVED)
        research_task = await db.get_task(research_task.id)
        assert research_task.state == TaskState.APPROVED

        # APPROVED → DONE
        await tm.transition(research_task.id, TaskState.DONE)
        research_task = await db.get_task(research_task.id)
        assert research_task.state == TaskState.DONE

        # DONE → anything is invalid
        with pytest.raises(ValueError, match="Invalid transition"):
            await tm.transition(research_task.id, TaskState.PENDING)


class TestRevisionCycle:
    """Critic rejects → PM routes feedback → agent revises → re-review."""

    @pytest.fixture
    async def env(self, db: Database):
        bus = MockMessageBus()
        ollama = MockOllamaClient()
        guardrails = Guardrails(max_iterations=5, max_tasks=20)
        tm = TaskManager(db=db, bus=bus, guardrails=guardrails)
        project = await _create_project(db)
        agents = _build_agents(db=db, bus=bus, ollama=ollama, task_manager=tm)
        return {
            "db": db, "bus": bus, "ollama": ollama, "tm": tm,
            "project": project, "agents": agents,
        }

    @pytest.mark.asyncio
    async def test_revision_then_approval(self, env):
        """Critic rejects code → PM routes revision → Coder revises → approved."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        ollama: MockOllamaClient = env["ollama"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]
        pm: PMAgent = env["agents"][AgentRole.PM]
        coder: CoderAgent = env["agents"][AgentRole.CODER]

        # Create a single coder task directly (skip PM planning)
        task = await tm.create_task(
            project_id=project.id,
            title="Implement login",
            description="Implement login endpoint",
            assigned_to=AgentRole.CODER,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.IN_PROGRESS)
        await tm.transition(task.id, TaskState.REVIEW)

        # ── Critic rejects ──
        bus.published.clear()
        rejection = make_message(
            sender=AgentRole.CRITIC,
            recipient=AgentRole.PM,
            msg_type=MessageType.FEEDBACK,
            project_id=project.id,
            task_id=task.id,
            content=CRITIC_REVISION_RESPONSE,
            metadata={"review_result": "needs_revision", "artifact_type": "code"},
        )

        ollama.default_response = "The Coder needs to add input validation. Fix the login endpoint."
        await pm._handle_message(rejection)

        # Task should be in REVISION
        task = await db.get_task(task.id)
        assert task.state == TaskState.REVISION
        assert task.iteration == 1

        # PM should have routed feedback to CODER
        coder_msgs = _find_messages(bus, AgentRole.CODER)
        assert len(coder_msgs) >= 1
        assert coder_msgs[-1].type == MessageType.FEEDBACK

        # ── Coder revises ──
        bus.published.clear()
        ollama.default_response = CODER_RESPONSE  # revised code

        # The coder should move task from REVISION → IN_PROGRESS before
        # working on the revision.  Currently the coder's _handle_feedback
        # doesn't do this transition itself, so we simulate it here.
        await tm.transition(task.id, TaskState.IN_PROGRESS)

        coder_feedback_msg = coder_msgs[-1]
        await coder._handle_message(coder_feedback_msg)

        # Coder sends revised deliverable to PM
        pm_msgs = _find_messages(bus, AgentRole.PM)
        assert len(pm_msgs) >= 1
        revised_deliverable = pm_msgs[-1]
        assert revised_deliverable.type == MessageType.DELIVERABLE

        # ── PM routes revised deliverable → Critic again ──
        bus.published.clear()
        ollama.default_response = "Revised code submitted for review."
        await pm._handle_message(revised_deliverable)

        # ── Critic approves the revision ──
        bus.published.clear()
        critic_final_approval = make_message(
            sender=AgentRole.CRITIC,
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=project.id,
            task_id=task.id,
            content=CRITIC_APPROVED_RESPONSE,
            metadata={"review_result": "approved", "artifact_type": "code"},
        )
        await pm._handle_message(critic_final_approval)

        task = await db.get_task(task.id)
        assert task.state == TaskState.DONE


class TestGuardrailIntegration:
    """Guardrails fire during real agent workflows."""

    @pytest.fixture
    async def env(self, db: Database):
        bus = MockMessageBus()
        ollama = MockOllamaClient()
        # Tight limits for testing
        guardrails = Guardrails(
            max_iterations=2,
            max_tasks=5,
            project_token_budget=100,
        )
        tm = TaskManager(db=db, bus=bus, guardrails=guardrails)
        project = await _create_project(db)
        agents = _build_agents(db=db, bus=bus, ollama=ollama, task_manager=tm)
        return {
            "db": db, "bus": bus, "ollama": ollama, "tm": tm,
            "project": project, "agents": agents,
        }

    @pytest.mark.asyncio
    async def test_max_iterations_triggers_failure(self, env):
        """Task exceeds max iterations → guardrail fails the task and escalates."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]

        task = await tm.create_task(
            project_id=project.id,
            title="Coding task",
            description="Write code",
            assigned_to=AgentRole.CODER,
            max_iterations=2,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.IN_PROGRESS)
        await tm.transition(task.id, TaskState.REVIEW)

        # First revision — allowed
        task = await tm.transition(task.id, TaskState.REVISION)
        assert task.state == TaskState.REVISION
        assert task.iteration == 1

        await tm.transition(task.id, TaskState.IN_PROGRESS)
        await tm.transition(task.id, TaskState.REVIEW)

        # Second revision — hits max (2), task should be FAILED
        task = await tm.transition(task.id, TaskState.REVISION)
        assert task.state == TaskState.FAILED

        # Escalation message should have been sent to PM
        pm_msgs = [m for m in bus.published if m.recipient == AgentRole.PM]
        assert len(pm_msgs) >= 1
        assert "guardrail" in pm_msgs[-1].payload.content.lower() or "escalation" in pm_msgs[-1].payload.metadata.get("escalation_reason", "")

    @pytest.mark.asyncio
    async def test_task_limit_prevents_creation(self, env):
        """Cannot create tasks beyond the guardrail limit."""
        db = env["db"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]

        # Create up to the limit (5)
        for i in range(5):
            await tm.create_task(
                project_id=project.id,
                title=f"Task {i}",
                description=f"Task {i} description",
                assigned_to=AgentRole.RESEARCH,
            )

        # The 6th task should be rejected
        with pytest.raises(ValueError, match="maximum task limit"):
            await tm.create_task(
                project_id=project.id,
                title="Task 6",
                description="One too many",
                assigned_to=AgentRole.RESEARCH,
            )

    @pytest.mark.asyncio
    async def test_token_budget_tracking(self, env):
        """Token usage is tracked across the project lifecycle."""
        db = env["db"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]

        task = await tm.create_task(
            project_id=project.id,
            title="Token test task",
            description="Test token tracking",
            assigned_to=AgentRole.RESEARCH,
        )

        # Record some tokens — within budget
        within = await tm.record_tokens(project.id, task.id, 50)
        assert within is True

        usage = await tm.get_project_token_usage(project.id)
        assert usage["tokens_used"] == 50
        assert usage["remaining"] == 50

        # Record more — exceeds budget (budget is 100)
        within = await tm.record_tokens(project.id, task.id, 60)
        assert within is False

        usage = await tm.get_project_token_usage(project.id)
        assert usage["tokens_used"] == 110
        assert usage["remaining"] == 0


class TestDependencyChains:
    """Tasks with dependencies are dispatched in the correct order."""

    @pytest.fixture
    async def env(self, db: Database):
        bus = MockMessageBus()
        ollama = MockOllamaClient()
        guardrails = Guardrails(max_tasks=20)
        tm = TaskManager(db=db, bus=bus, guardrails=guardrails)
        project = await _create_project(db)
        return {"db": db, "bus": bus, "tm": tm, "project": project, "ollama": ollama}

    @pytest.mark.asyncio
    async def test_dependency_blocks_until_satisfied(self, env):
        """A task with an unmet dep is not assignable; becomes assignable when dep is DONE."""
        tm: TaskManager = env["tm"]
        project: Project = env["project"]

        t1 = await tm.create_task(
            project_id=project.id,
            title="First",
            description="Do first",
            assigned_to=AgentRole.RESEARCH,
        )
        t2 = await tm.create_task(
            project_id=project.id,
            title="Second",
            description="Do second",
            assigned_to=AgentRole.SPEC,
            depends_on=[t1.id],
        )

        # Only t1 is assignable
        assignable = await tm.get_assignable_tasks(project.id)
        assert [t.id for t in assignable] == [t1.id]

        # Complete t1
        await tm.transition(t1.id, TaskState.ASSIGNED)
        await tm.transition(t1.id, TaskState.IN_PROGRESS)
        await tm.transition(t1.id, TaskState.REVIEW)
        await tm.transition(t1.id, TaskState.APPROVED)
        await tm.transition(t1.id, TaskState.DONE)

        # Now t2 is assignable
        assignable = await tm.get_assignable_tasks(project.id)
        assert [t.id for t in assignable] == [t2.id]

    @pytest.mark.asyncio
    async def test_parallel_independent_tasks(self, env):
        """Independent tasks (no deps) are all assignable at once."""
        tm: TaskManager = env["tm"]
        project: Project = env["project"]

        t1 = await tm.create_task(
            project_id=project.id,
            title="Independent A",
            description="No deps",
            assigned_to=AgentRole.RESEARCH,
        )
        t2 = await tm.create_task(
            project_id=project.id,
            title="Independent B",
            description="No deps",
            assigned_to=AgentRole.SPEC,
        )
        t3 = await tm.create_task(
            project_id=project.id,
            title="Depends on A+B",
            description="Depends on both",
            assigned_to=AgentRole.CODER,
            depends_on=[t1.id, t2.id],
        )

        assignable = await tm.get_assignable_tasks(project.id)
        assignable_ids = {t.id for t in assignable}
        assert t1.id in assignable_ids
        assert t2.id in assignable_ids
        assert t3.id not in assignable_ids  # deps not met

    @pytest.mark.asyncio
    async def test_diamond_dependency(self, env):
        """Diamond dependency: A → B, A → C, B+C → D."""
        tm: TaskManager = env["tm"]
        project: Project = env["project"]

        a = await tm.create_task(project_id=project.id, title="A", description="root", assigned_to=AgentRole.RESEARCH)
        b = await tm.create_task(project_id=project.id, title="B", description="b", assigned_to=AgentRole.SPEC, depends_on=[a.id])
        c = await tm.create_task(project_id=project.id, title="C", description="c", assigned_to=AgentRole.SPEC, depends_on=[a.id])
        d = await tm.create_task(project_id=project.id, title="D", description="d", assigned_to=AgentRole.CODER, depends_on=[b.id, c.id])

        # Only A is assignable
        assignable = await tm.get_assignable_tasks(project.id, limit=10)
        assert [t.id for t in assignable] == [a.id]

        # Complete A → B and C become assignable
        for state in [TaskState.ASSIGNED, TaskState.IN_PROGRESS, TaskState.REVIEW, TaskState.APPROVED, TaskState.DONE]:
            await tm.transition(a.id, state)

        assignable = await tm.get_assignable_tasks(project.id, limit=10)
        assignable_ids = {t.id for t in assignable}
        assert b.id in assignable_ids
        assert c.id in assignable_ids
        assert d.id not in assignable_ids

        # Complete B → D still blocked (C not done)
        for state in [TaskState.ASSIGNED, TaskState.IN_PROGRESS, TaskState.REVIEW, TaskState.APPROVED, TaskState.DONE]:
            await tm.transition(b.id, state)

        assignable = await tm.get_assignable_tasks(project.id, limit=10)
        assert d.id not in {t.id for t in assignable}

        # Complete C → D finally assignable
        for state in [TaskState.ASSIGNED, TaskState.IN_PROGRESS, TaskState.REVIEW, TaskState.APPROVED, TaskState.DONE]:
            await tm.transition(c.id, state)

        assignable = await tm.get_assignable_tasks(project.id, limit=10)
        assert [t.id for t in assignable] == [d.id]


class TestPauseResumeIntegration:
    """Project pause/resume affects agent message processing."""

    @pytest.fixture
    async def env(self, db: Database):
        bus = MockMessageBus()
        ollama = MockOllamaClient()
        tm = TaskManager(db=db, bus=bus)
        project = await _create_project(db)
        agents = _build_agents(db=db, bus=bus, ollama=ollama, task_manager=tm)
        return {
            "db": db, "bus": bus, "ollama": ollama, "tm": tm,
            "project": project, "agents": agents,
        }

    @pytest.mark.asyncio
    async def test_paused_project_skips_messages(self, env):
        """When a project is paused, agents skip message processing."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        ollama: MockOllamaClient = env["ollama"]
        project: Project = env["project"]
        research: ResearchAgent = env["agents"][AgentRole.RESEARCH]

        # Pause the project
        await db.update_project(project.id, status="paused")

        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            content="Do some research",
        )

        await research._handle_message(msg)

        # No LLM calls should have been made (message was skipped)
        assert len(ollama.chat_calls) == 0
        # No response published
        assert len(bus.published) == 0

    @pytest.mark.asyncio
    async def test_resume_allows_processing(self, env):
        """After resuming, agents process messages normally."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        ollama: MockOllamaClient = env["ollama"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]
        research: ResearchAgent = env["agents"][AgentRole.RESEARCH]

        # Create a task for the research agent
        task = await tm.create_task(
            project_id=project.id,
            title="Research task",
            description="Research something",
            assigned_to=AgentRole.RESEARCH,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)

        # Pause then resume
        await db.update_project(project.id, status="paused")
        await db.update_project(project.id, status="active")

        ollama.default_response = RESEARCH_RESPONSE

        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            task_id=task.id,
            content="Research authentication approaches",
        )

        await research._handle_message(msg)

        # LLM was called and a response was published
        assert len(ollama.chat_calls) >= 1
        assert len(bus.published) >= 1


class TestPipelineHealthIntegration:
    """Pipeline health checks interact with real DB and guardrails."""

    @pytest.fixture
    async def env(self, db: Database):
        bus = MockMessageBus()
        guardrails = Guardrails(
            task_timeout=1,  # 1 minute timeout
            activity_timeout=1,  # 1 minute timeout
            project_token_budget=100,
        )
        tm = TaskManager(db=db, bus=bus, guardrails=guardrails)
        pipeline = Pipeline(task_manager=tm, bus=bus)
        pipeline.guardrails = guardrails
        project = await _create_project(db)
        return {"db": db, "bus": bus, "tm": tm, "pipeline": pipeline, "project": project}

    @pytest.mark.asyncio
    async def test_pipeline_fails_timed_out_tasks(self, env):
        """Pipeline health check auto-fails tasks that have timed out."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        tm: TaskManager = env["tm"]
        pipeline: Pipeline = env["pipeline"]
        project: Project = env["project"]

        task = await tm.create_task(
            project_id=project.id,
            title="Slow task",
            description="This will time out",
            assigned_to=AgentRole.RESEARCH,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.IN_PROGRESS)

        # Push updated_at far into the past via raw SQL (update_task always
        # sets updated_at=now, so we must bypass it).
        old_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        await db.db.execute(
            "UPDATE tasks SET updated_at = ? WHERE id = ?",
            (old_time, task.id),
        )
        await db.db.commit()

        # Run health check — with task_timeout=0, it should fail immediately
        await pipeline._check_health()

        task = await db.get_task(task.id)
        assert task.state == TaskState.FAILED

        # UI event should have been emitted
        state_events = [e for e in bus.ui_events if e[0] == "task_state_changed"]
        assert len(state_events) >= 1

    @pytest.mark.asyncio
    async def test_pipeline_detects_token_budget_exceeded(self, env):
        """Pipeline health check fails a project that exceeded its token budget."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        tm: TaskManager = env["tm"]
        pipeline: Pipeline = env["pipeline"]
        project: Project = env["project"]

        # Record tokens beyond the budget
        await db.add_project_tokens(project.id, 200)

        await pipeline._check_health()

        project_final = await db.get_project(project.id)
        assert project_final.status == "failed"

        status_events = [e for e in bus.ui_events if e[0] == "project_status_changed"]
        assert any(e[1].get("reason") == "token_budget_exceeded" for e in status_events)


class TestRetryIntegration:
    """Failed task can be retried and re-enter the pipeline."""

    @pytest.fixture
    async def env(self, db: Database):
        bus = MockMessageBus()
        ollama = MockOllamaClient()
        tm = TaskManager(db=db, bus=bus)
        project = await _create_project(db)
        agents = _build_agents(db=db, bus=bus, ollama=ollama, task_manager=tm)
        return {
            "db": db, "bus": bus, "ollama": ollama, "tm": tm,
            "project": project, "agents": agents,
        }

    @pytest.mark.asyncio
    async def test_retry_failed_task(self, env):
        """A FAILED task can be retried back to PENDING and re-processed."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        ollama: MockOllamaClient = env["ollama"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]
        pm: PMAgent = env["agents"][AgentRole.PM]
        research: ResearchAgent = env["agents"][AgentRole.RESEARCH]

        task = await tm.create_task(
            project_id=project.id,
            title="Retryable research",
            description="Will fail then succeed",
            assigned_to=AgentRole.RESEARCH,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)
        await tm.transition(task.id, TaskState.IN_PROGRESS)
        await tm.transition(task.id, TaskState.FAILED)

        # Task is now FAILED
        task = await db.get_task(task.id)
        assert task.state == TaskState.FAILED

        # Retry: FAILED → PENDING
        task = await tm.transition(task.id, TaskState.PENDING)
        assert task.state == TaskState.PENDING

        # Now the task should be assignable again
        assignable = await tm.get_assignable_tasks(project.id)
        assert any(t.id == task.id for t in assignable)

        # Assign and process successfully this time
        await tm.transition(task.id, TaskState.ASSIGNED)
        bus.published.clear()
        ollama.default_response = RESEARCH_RESPONSE

        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            task_id=task.id,
            content="Retry this research task",
        )
        await research._handle_message(msg)

        # Should produce a deliverable
        pm_msgs = _find_messages(bus, AgentRole.PM)
        assert len(pm_msgs) >= 1
        assert pm_msgs[-1].type == MessageType.DELIVERABLE


class TestMultiAgentConversationIsolation:
    """Per-project conversation histories are properly isolated."""

    @pytest.fixture
    async def env(self, db: Database):
        bus = MockMessageBus()
        ollama = MockOllamaClient()
        tm = TaskManager(db=db, bus=bus)
        return {"db": db, "bus": bus, "ollama": ollama, "tm": tm}

    @pytest.mark.asyncio
    async def test_agents_isolate_project_histories(self, env):
        """Two projects running concurrently don't leak context."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        ollama: MockOllamaClient = env["ollama"]
        tm: TaskManager = env["tm"]

        # Create two projects
        p1 = Project(name="Project A", description="A", idea="Build a chat app")
        p2 = Project(name="Project B", description="B", idea="Build a todo app")
        await db.create_project(p1)
        await db.create_project(p2)

        research = ResearchAgent(bus=bus, ollama=ollama, db=db, task_manager=tm)

        # Create tasks for both projects
        t1 = await tm.create_task(project_id=p1.id, title="Research A", description="Chat app research", assigned_to=AgentRole.RESEARCH)
        t2 = await tm.create_task(project_id=p2.id, title="Research B", description="Todo app research", assigned_to=AgentRole.RESEARCH)
        await tm.transition(t1.id, TaskState.ASSIGNED)
        await tm.transition(t2.id, TaskState.ASSIGNED)

        # Process task for project A
        ollama.default_response = "Chat app research results"
        msg_a = make_message(
            sender=AgentRole.PM, recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=p1.id, task_id=t1.id,
            content="Research chat app technologies",
        )
        await research._handle_message(msg_a)

        # Process task for project B
        ollama.default_response = "Todo app research results"
        msg_b = make_message(
            sender=AgentRole.PM, recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=p2.id, task_id=t2.id,
            content="Research todo app technologies",
        )
        await research._handle_message(msg_b)

        # Verify histories are isolated
        history_a = research._get_history(p1.id)
        history_b = research._get_history(p2.id)

        assert len(history_a) > 0
        assert len(history_b) > 0

        # Content should be different
        a_content = " ".join(e.content for e in history_a)
        b_content = " ".join(e.content for e in history_b)
        assert "chat" in a_content.lower()
        assert "todo" in b_content.lower()


class TestMessagePersistence:
    """Messages flowing through agents are persisted to the database."""

    @pytest.fixture
    async def env(self, db: Database):
        bus = MockMessageBus()
        ollama = MockOllamaClient()
        tm = TaskManager(db=db, bus=bus)
        project = await _create_project(db)
        agents = _build_agents(db=db, bus=bus, ollama=ollama, task_manager=tm)
        return {
            "db": db, "bus": bus, "ollama": ollama, "tm": tm,
            "project": project, "agents": agents,
        }

    @pytest.mark.asyncio
    async def test_messages_are_persisted(self, env):
        """Both incoming and outgoing messages are saved to the database."""
        db = env["db"]
        bus: MockMessageBus = env["bus"]
        ollama: MockOllamaClient = env["ollama"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]
        research: ResearchAgent = env["agents"][AgentRole.RESEARCH]

        task = await tm.create_task(
            project_id=project.id,
            title="Persisted research",
            description="Test persistence",
            assigned_to=AgentRole.RESEARCH,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)

        ollama.default_response = RESEARCH_RESPONSE

        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.RESEARCH,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            task_id=task.id,
            content="Research something",
        )

        await research._handle_message(msg)

        # Both the incoming message and the agent's response should be in DB
        messages = await db.get_project_messages(project.id)
        assert len(messages) >= 2  # incoming + response

        senders = {m["sender"] for m in messages}
        assert AgentRole.PM.value in senders
        assert AgentRole.RESEARCH.value in senders


class TestUIEventEmission:
    """Key state changes emit UI events via the message bus."""

    @pytest.fixture
    async def env(self, db: Database):
        bus = MockMessageBus()
        tm = TaskManager(db=db, bus=bus)
        project = await _create_project(db)
        return {"db": db, "bus": bus, "tm": tm, "project": project}

    @pytest.mark.asyncio
    async def test_task_transitions_emit_ui_events(self, env):
        """Every task state transition emits a task_state_changed UI event."""
        bus: MockMessageBus = env["bus"]
        tm: TaskManager = env["tm"]
        project: Project = env["project"]

        task = await tm.create_task(
            project_id=project.id,
            title="UI test",
            description="Test UI events",
            assigned_to=AgentRole.RESEARCH,
        )

        transitions = [
            TaskState.ASSIGNED,
            TaskState.IN_PROGRESS,
            TaskState.REVIEW,
            TaskState.APPROVED,
            TaskState.DONE,
        ]

        for state in transitions:
            await tm.transition(task.id, state)

        # Each transition should have emitted a UI event
        state_events = [e for e in bus.ui_events if e[0] == "task_state_changed"]
        assert len(state_events) == len(transitions)

        # Verify the last event contains the right data
        last_event = state_events[-1][1]
        assert last_event["task_id"] == task.id
        assert last_event["new_state"] == "done"
