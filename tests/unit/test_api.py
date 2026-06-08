"""Tests for the FastAPI REST API endpoints.

Uses httpx AsyncClient with the ASGI transport so no real server is needed.
All dependencies (DB, bus, agents, task_manager) are mocked or use temp storage.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from hca.api.app import create_app
from hca.core.models import AgentRole, AgentStatus, Task, TaskState
from tests.conftest import MockMessageBus, MockOllamaClient, make_message

# ============================================================
# Helpers
# ============================================================


class FakeAgent:
    """Minimal agent stub exposing get_info() and an ollama attribute."""

    def __init__(self, role: AgentRole) -> None:
        self.role = role
        self.ollama = MockOllamaClient()
        self._busy = False
        self.status = AgentStatus.IDLE
        self._current_activity = ""

        class _FakeStats:
            @staticmethod
            def snapshot() -> dict:
                return {"uptime_seconds": 0.0}
        self.stats = _FakeStats()

    def get_info(self) -> dict:
        return {
            "role": self.role.value,
            "status": "busy" if self._busy else "idle",
            "model": "mock-model",
            "tasks_completed": 0,
            "tokens_used": 0,
        }


class FakeTaskManager:
    """Stub that provides get_project_progress and transition."""

    def __init__(self, db):
        self.db = db

    async def get_project_progress(self, project_id: str) -> dict:
        tasks = await self.db.list_tasks(project_id)
        total = len(tasks)
        done = sum(1 for t in tasks if t.state == TaskState.DONE)
        return {"total": total, "done": done, "percent": (done / total * 100) if total else 0}

    async def transition(self, task_id: str, new_state: TaskState) -> Task:
        task = await self.db.get_task(task_id)
        await self.db.update_task_state(task_id, new_state)
        task.state = new_state
        return task


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
async def client(db):
    """Provide an httpx AsyncClient wired to the real app."""
    bus = MockMessageBus()
    agents = [FakeAgent(role) for role in AgentRole if role != AgentRole.USER]
    tm = FakeTaskManager(db)

    app = create_app(db=db, bus=bus, task_manager=tm, agents=agents)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac, bus


# ============================================================
# Health
# ============================================================


class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, client):
        ac, _ = client
        r = await ac.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "db" in body
        assert "redis" in body
        assert "ollama" in body
        assert "agents" in body

    @pytest.mark.asyncio
    async def test_liveness(self, client):
        ac, _ = client
        r = await ac.get("/api/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    @pytest.mark.asyncio
    async def test_readiness(self, client):
        ac, _ = client
        r = await ac.get("/api/health/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["issues"] == []

    @pytest.mark.asyncio
    async def test_db_health(self, client):
        ac, _ = client
        r = await ac.get("/api/health/db")
        assert r.status_code == 200
        body = r.json()
        assert "schema_version" in body
        assert "total_projects" in body


# ============================================================
# Projects
# ============================================================


class TestProjects:
    @pytest.mark.asyncio
    async def test_list_projects_empty(self, client):
        ac, _ = client
        r = await ac.get("/api/projects/")
        assert r.status_code == 200
        assert r.json() == []

    @pytest.mark.asyncio
    async def test_create_project(self, client):
        ac, bus = client
        r = await ac.post("/api/projects/", json={"idea": "Build a chat app"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "created"
        assert "project_id" in body

        # Bus should have received the kick-off message
        assert len(bus.published) == 1
        assert bus.published[0].recipient == AgentRole.PM

    @pytest.mark.asyncio
    async def test_create_project_with_name(self, client):
        ac, _ = client
        r = await ac.post("/api/projects/", json={"idea": "Test", "name": "My Project"})
        assert r.status_code == 200
        pid = r.json()["project_id"]

        r2 = await ac.get(f"/api/projects/{pid}")
        assert r2.status_code == 200
        assert r2.json()["project"]["name"] == "My Project"

    @pytest.mark.asyncio
    async def test_list_projects_after_create(self, client):
        ac, _ = client
        await ac.post("/api/projects/", json={"idea": "Idea 1"})
        await ac.post("/api/projects/", json={"idea": "Idea 2"})
        r = await ac.get("/api/projects/")
        assert len(r.json()) == 2

    @pytest.mark.asyncio
    async def test_get_project_not_found(self, client):
        ac, _ = client
        r = await ac.get("/api/projects/nonexistent")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_get_project_detail(self, client):
        ac, _ = client
        r = await ac.post("/api/projects/", json={"idea": "Detail test"})
        pid = r.json()["project_id"]

        r2 = await ac.get(f"/api/projects/{pid}")
        assert r2.status_code == 200
        body = r2.json()
        assert "project" in body
        assert "progress" in body
        assert body["message_count"] >= 0
        assert body["artifact_count"] >= 0

    @pytest.mark.asyncio
    async def test_get_project_messages_empty(self, client):
        ac, _ = client
        r = await ac.post("/api/projects/", json={"idea": "Msg test"})
        pid = r.json()["project_id"]
        r2 = await ac.get(f"/api/projects/{pid}/messages")
        assert r2.status_code == 200
        assert r2.json() == []

    @pytest.mark.asyncio
    async def test_get_project_artifacts_empty(self, client):
        ac, _ = client
        r = await ac.post("/api/projects/", json={"idea": "Art test"})
        pid = r.json()["project_id"]
        r2 = await ac.get(f"/api/projects/{pid}/artifacts")
        assert r2.status_code == 200
        assert r2.json() == []


# ============================================================
# Pause / Resume
# ============================================================


class TestPauseResume:
    @pytest.mark.asyncio
    async def test_pause_active_project(self, client):
        ac, _ = client
        r = await ac.post("/api/projects/", json={"idea": "Pause me"})
        pid = r.json()["project_id"]

        r2 = await ac.post(f"/api/projects/{pid}/pause")
        assert r2.status_code == 200
        assert r2.json()["status"] == "paused"

    @pytest.mark.asyncio
    async def test_pause_already_paused(self, client):
        ac, _ = client
        r = await ac.post("/api/projects/", json={"idea": "Double pause"})
        pid = r.json()["project_id"]
        await ac.post(f"/api/projects/{pid}/pause")

        r2 = await ac.post(f"/api/projects/{pid}/pause")
        assert r2.status_code == 400

    @pytest.mark.asyncio
    async def test_resume_paused_project(self, client):
        ac, _ = client
        r = await ac.post("/api/projects/", json={"idea": "Resume me"})
        pid = r.json()["project_id"]
        await ac.post(f"/api/projects/{pid}/pause")

        r2 = await ac.post(f"/api/projects/{pid}/resume")
        assert r2.status_code == 200
        assert r2.json()["status"] == "active"

    @pytest.mark.asyncio
    async def test_resume_active_project_fails(self, client):
        ac, _ = client
        r = await ac.post("/api/projects/", json={"idea": "Bad resume"})
        pid = r.json()["project_id"]

        r2 = await ac.post(f"/api/projects/{pid}/resume")
        assert r2.status_code == 400

    @pytest.mark.asyncio
    async def test_pause_nonexistent(self, client):
        ac, _ = client
        r = await ac.post("/api/projects/no-such-id/pause")
        assert r.status_code == 404


# ============================================================
# Agents
# ============================================================


class TestAgents:
    @pytest.mark.asyncio
    async def test_list_agents(self, client):
        ac, _ = client
        r = await ac.get("/api/agents/")
        assert r.status_code == 200
        roles = {a["role"] for a in r.json()}
        assert "pm" in roles
        assert "coder" in roles

    @pytest.mark.asyncio
    async def test_get_agent_by_role(self, client):
        ac, _ = client
        r = await ac.get("/api/agents/pm")
        assert r.status_code == 200
        assert r.json()["role"] == "pm"

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, client):
        ac, _ = client
        r = await ac.get("/api/agents/unknown")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_ollama_stats(self, client):
        ac, _ = client
        r = await ac.get("/api/agents/stats")
        assert r.status_code == 200
        assert "total_requests" in r.json()


# ============================================================
# Tasks
# ============================================================


class TestTasks:
    async def _setup_project_with_task(self, ac, db):
        """Helper: create a project and a task, return (project_id, task_id)."""
        r = await ac.post("/api/projects/", json={"idea": "Task test"})
        pid = r.json()["project_id"]
        task = Task(
            project_id=pid,
            title="Implement login",
            description="Build the login page",
        )
        await db.create_task(task)
        return pid, task.id

    @pytest.mark.asyncio
    async def test_list_tasks_empty(self, client, db):
        ac, _ = client
        r = await ac.post("/api/projects/", json={"idea": "No tasks"})
        pid = r.json()["project_id"]
        r2 = await ac.get(f"/api/tasks/{pid}")
        assert r2.status_code == 200
        assert r2.json() == []

    @pytest.mark.asyncio
    async def test_list_tasks_with_task(self, client, db):
        ac, _ = client
        pid, tid = await self._setup_project_with_task(ac, db)
        r = await ac.get(f"/api/tasks/{pid}")
        assert r.status_code == 200
        tasks = r.json()
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Implement login"

    @pytest.mark.asyncio
    async def test_list_tasks_filter_by_state(self, client, db):
        ac, _ = client
        pid, tid = await self._setup_project_with_task(ac, db)
        # Task starts as PENDING
        r = await ac.get(f"/api/tasks/{pid}?state=pending")
        assert len(r.json()) == 1
        r2 = await ac.get(f"/api/tasks/{pid}?state=done")
        assert len(r2.json()) == 0

    @pytest.mark.asyncio
    async def test_list_tasks_invalid_state(self, client, db):
        ac, _ = client
        pid, _ = await self._setup_project_with_task(ac, db)

        r = await ac.get(f"/api/tasks/{pid}?state=definitely-not-a-state")

        assert r.status_code == 400
        assert "Invalid task state" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_task_detail(self, client, db):
        ac, _ = client
        _, tid = await self._setup_project_with_task(ac, db)
        r = await ac.get(f"/api/tasks/detail/{tid}")
        assert r.status_code == 200
        assert r.json()["id"] == tid

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, client):
        ac, _ = client
        r = await ac.get("/api/tasks/detail/nonexistent")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_retry_failed_task(self, client, db):
        ac, _ = client
        pid, tid = await self._setup_project_with_task(ac, db)
        # Move task to FAILED
        await db.update_task_state(tid, TaskState.FAILED)

        r = await ac.post(f"/api/tasks/detail/{tid}/retry")
        assert r.status_code == 200
        assert r.json()["state"] == "pending"

    @pytest.mark.asyncio
    async def test_retry_non_failed_task_rejected(self, client, db):
        ac, _ = client
        pid, tid = await self._setup_project_with_task(ac, db)
        # Task is PENDING, not FAILED
        r = await ac.post(f"/api/tasks/detail/{tid}/retry")
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_retry_nonexistent_task(self, client):
        ac, _ = client
        r = await ac.post("/api/tasks/detail/no-such-id/retry")
        assert r.status_code == 404


# ============================================================
# Dead-Letter API
# ============================================================


class TestDeadLetterAPI:
    @pytest.mark.asyncio
    async def test_list_dead_letter_empty(self, client):
        ac, _ = client
        r = await ac.get("/api/admin/dead-letter")
        assert r.status_code == 200
        assert r.json() == []

    @pytest.mark.asyncio
    async def test_list_dead_letter_with_entries(self, client):
        ac, bus = client
        msg = make_message()
        await bus.move_to_dead_letter("stream:pm", "entry-1", msg, reason="test_failure")

        r = await ac.get("/api/admin/dead-letter")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["reason"] == "test_failure"
        assert data[0]["data"]["id"] == msg.id

    @pytest.mark.asyncio
    async def test_replay_dead_letter(self, client):
        ac, bus = client
        msg = make_message(content="replay me")
        await bus.move_to_dead_letter("stream:pm", "entry-1", msg, reason="test")

        entries = await bus.list_dead_letter_messages()
        assert len(entries) == 1
        entry_id = entries[0]["id"]

        r = await ac.post(f"/api/admin/dead-letter/{entry_id}/replay")
        assert r.status_code == 200
        result = r.json()
        assert result["status"] == "replayed"
        assert result["message_id"] == msg.id

    @pytest.mark.asyncio
    async def test_replay_dead_letter_not_found(self, client):
        ac, _ = client
        r = await ac.post("/api/admin/dead-letter/no-such-id/replay")
        assert r.status_code == 404


# ============================================================
# HITL — Human-in-the-Loop
# ============================================================


class TestHITL:
    """Tests for human-in-the-loop API endpoints."""

    async def _setup(self, ac, db):
        """Create a project and a task, return (project_id, task_id)."""
        r = await ac.post("/api/projects/", json={"idea": "HITL test"})
        pid = r.json()["project_id"]
        task = Task(project_id=pid, title="Test task", description="do it")
        await db.create_task(task)
        return pid, task.id

    @pytest.mark.asyncio
    async def test_inject_feedback(self, client, db):
        ac, _ = client
        pid, tid = await self._setup(ac, db)

        r = await ac.post(
            f"/api/projects/{pid}/feedback",
            json={"task_id": tid, "content": "Please add error handling"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["task_id"] == tid

    @pytest.mark.asyncio
    async def test_inject_feedback_missing_fields(self, client, db):
        ac, _ = client
        r = await ac.post("/api/projects/p1/feedback", json={})
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_inject_feedback_invalid_agent_role(self, client, db):
        ac, _ = client
        pid, tid = await self._setup(ac, db)
        r = await ac.post(
            f"/api/projects/{pid}/feedback",
            json={"task_id": tid, "content": "fix it", "agent_role": "invalid_role"},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_human_review_approve(self, client, db):
        ac, _ = client
        pid, tid = await self._setup(ac, db)
        # Task must be in REVIEW state
        await db.update_task_state(tid, TaskState.REVIEW)

        r = await ac.post(
            f"/api/projects/{pid}/review",
            json={"task_id": tid, "verdict": "approved", "summary": "Looks good!"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["verdict"] == "approved"

    @pytest.mark.asyncio
    async def test_human_review_reject(self, client, db):
        ac, _ = client
        pid, tid = await self._setup(ac, db)
        await db.update_task_state(tid, TaskState.REVIEW)

        r = await ac.post(
            f"/api/projects/{pid}/review",
            json={"task_id": tid, "verdict": "needs_revision", "summary": "Fix the bugs"},
        )
        assert r.status_code == 200
        assert r.json()["verdict"] == "needs_revision"

    @pytest.mark.asyncio
    async def test_human_review_non_review_task(self, client, db):
        ac, _ = client
        pid, tid = await self._setup(ac, db)
        # Task is PENDING, not REVIEW
        r = await ac.post(
            f"/api/projects/{pid}/review",
            json={"task_id": tid, "verdict": "approved"},
        )
        assert r.status_code == 400
        assert "Only REVIEW tasks" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_human_review_invalid_verdict(self, client, db):
        ac, _ = client
        pid, tid = await self._setup(ac, db)
        r = await ac.post(
            f"/api/projects/{pid}/review",
            json={"task_id": tid, "verdict": "maybe"},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_steer_task_priority(self, client, db):
        ac, _ = client
        pid, tid = await self._setup(ac, db)

        r = await ac.patch(
            f"/api/tasks/{tid}/steer",
            json={"priority": "high"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["task"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_steer_task_title(self, client, db):
        ac, _ = client
        pid, tid = await self._setup(ac, db)

        r = await ac.patch(
            f"/api/tasks/{tid}/steer",
            json={"title": "Updated title"},
        )
        assert r.status_code == 200
        assert r.json()["task"]["title"] == "Updated title"

    @pytest.mark.asyncio
    async def test_steer_task_not_found(self, client):
        ac, _ = client
        r = await ac.patch("/api/tasks/no-such-id/steer", json={"priority": "high"})
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_steer_task_invalid_priority(self, client, db):
        ac, _ = client
        pid, tid = await self._setup(ac, db)
        r = await ac.patch(f"/api/tasks/{tid}/steer", json={"priority": "urgent"})
        assert r.status_code == 400
