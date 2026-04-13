"""Tests for the SQLite database layer.

Covers: migrations, project/task/artifact/message CRUD,
pagination, search, events timeline, and diagnostics.
All tests use a real SQLite database in a temp directory.
"""

from __future__ import annotations

import pytest

from src.core.database import Database, DatabaseError, CURRENT_SCHEMA_VERSION
from src.core.models import (
    AgentRole,
    Artifact,
    Priority,
    Project,
    Task,
    TaskState,
)


# ============================================================
# Migrations & Init
# ============================================================


class TestDatabaseInit:
    """Tests for database initialization and migration system."""

    async def test_initialize_creates_tables(self, db: Database) -> None:
        """Schema version should match CURRENT_SCHEMA_VERSION after init."""
        version = await db._get_current_version()
        assert version == CURRENT_SCHEMA_VERSION

    async def test_double_initialize_is_safe(self, tmp_path) -> None:
        """Calling initialize() twice should not error or duplicate data."""
        db_path = str(tmp_path / "double.db")
        database = Database(f"sqlite:///{db_path}")
        await database.initialize()
        await database.initialize()  # Should be a no-op
        version = await database._get_current_version()
        assert version == CURRENT_SCHEMA_VERSION
        await database.close()

    async def test_db_property_raises_before_init(self) -> None:
        """Accessing .db before initialize() should raise DatabaseError."""
        database = Database("sqlite:///nonexistent.db")
        with pytest.raises(DatabaseError, match="not initialized"):
            _ = database.db


# ============================================================
# Projects CRUD
# ============================================================


class TestProjects:
    """Tests for project create, read, update, delete."""

    async def test_create_and_get_project(self, db: Database) -> None:
        project = Project(name="Test", description="A test project", idea="Build a thing")
        created = await db.create_project(project)
        assert created.id == project.id

        fetched = await db.get_project(project.id)
        assert fetched is not None
        assert fetched.name == "Test"
        assert fetched.idea == "Build a thing"

    async def test_get_nonexistent_project(self, db: Database) -> None:
        result = await db.get_project("nonexistent-id")
        assert result is None

    async def test_create_duplicate_project_raises(self, db: Database) -> None:
        project = Project(name="Dup", description="", idea="idea")
        await db.create_project(project)
        with pytest.raises(DatabaseError, match="already exists"):
            await db.create_project(project)

    async def test_list_projects_empty(self, db: Database) -> None:
        projects = await db.list_projects()
        assert projects == []

    async def test_list_projects_with_status_filter(self, db: Database) -> None:
        p1 = Project(name="Active", description="", idea="a", status="active")
        p2 = Project(name="Done", description="", idea="b", status="completed")
        await db.create_project(p1)
        await db.create_project(p2)

        active = await db.list_projects(status="active")
        assert len(active) == 1
        assert active[0].name == "Active"

        completed = await db.list_projects(status="completed")
        assert len(completed) == 1
        assert completed[0].name == "Done"

    async def test_list_projects_pagination(self, db: Database) -> None:
        for i in range(5):
            await db.create_project(
                Project(name=f"P{i}", description="", idea=f"idea {i}")
            )
        page1 = await db.list_projects(limit=2, offset=0)
        page2 = await db.list_projects(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # No overlap
        ids1 = {p.id for p in page1}
        ids2 = {p.id for p in page2}
        assert ids1.isdisjoint(ids2)

    async def test_update_project(self, db: Database) -> None:
        project = Project(name="Old", description="old desc", idea="idea")
        await db.create_project(project)
        await db.update_project(project.id, name="New", description="new desc")
        fetched = await db.get_project(project.id)
        assert fetched is not None
        assert fetched.name == "New"
        assert fetched.description == "new desc"

    async def test_update_project_status(self, db: Database) -> None:
        project = Project(name="P", description="", idea="idea")
        await db.create_project(project)
        await db.update_project(project.id, status="completed")
        fetched = await db.get_project(project.id)
        assert fetched is not None
        assert fetched.status == "completed"

    async def test_update_project_ignores_invalid_fields(self, db: Database) -> None:
        project = Project(name="P", description="", idea="idea")
        await db.create_project(project)
        # "idea" is not in the allowed set, should be silently ignored
        await db.update_project(project.id, idea="new idea")
        fetched = await db.get_project(project.id)
        assert fetched is not None
        assert fetched.idea == "idea"  # unchanged

    async def test_delete_project(self, db: Database) -> None:
        project = Project(name="Del", description="", idea="idea")
        await db.create_project(project)
        deleted = await db.delete_project(project.id)
        assert deleted is True
        assert await db.get_project(project.id) is None

    async def test_delete_nonexistent_project(self, db: Database) -> None:
        deleted = await db.delete_project("nope")
        assert deleted is False

    async def test_count_projects(self, db: Database) -> None:
        assert await db.count_projects() == 0
        await db.create_project(Project(name="A", description="", idea="a"))
        await db.create_project(
            Project(name="B", description="", idea="b", status="completed")
        )
        assert await db.count_projects() == 2
        assert await db.count_projects(status="active") == 1
        assert await db.count_projects(status="completed") == 1


# ============================================================
# Tasks CRUD
# ============================================================


class TestTasks:
    """Tests for task create, read, update, list."""

    async def _make_project(self, db: Database) -> Project:
        p = Project(name="TaskProject", description="", idea="idea")
        return await db.create_project(p)

    async def test_create_and_get_task(self, db: Database) -> None:
        project = await self._make_project(db)
        task = Task(
            project_id=project.id,
            title="Research phase",
            description="Do research",
            assigned_to=AgentRole.RESEARCH,
        )
        await db.create_task(task)
        fetched = await db.get_task(task.id)
        assert fetched is not None
        assert fetched.title == "Research phase"
        assert fetched.state == TaskState.PENDING

    async def test_get_nonexistent_task(self, db: Database) -> None:
        assert await db.get_task("nope") is None

    async def test_list_tasks_by_state(self, db: Database) -> None:
        project = await self._make_project(db)
        t1 = Task(project_id=project.id, title="T1", description="", state=TaskState.PENDING)
        t2 = Task(project_id=project.id, title="T2", description="", state=TaskState.IN_PROGRESS)
        await db.create_task(t1)
        await db.create_task(t2)

        pending = await db.list_tasks(project.id, state=TaskState.PENDING)
        assert len(pending) == 1
        assert pending[0].title == "T1"

    async def test_list_tasks_by_assigned_to(self, db: Database) -> None:
        project = await self._make_project(db)
        t1 = Task(
            project_id=project.id, title="T1", description="",
            assigned_to=AgentRole.CODER,
        )
        t2 = Task(
            project_id=project.id, title="T2", description="",
            assigned_to=AgentRole.RESEARCH,
        )
        await db.create_task(t1)
        await db.create_task(t2)

        coder_tasks = await db.list_tasks(project.id, assigned_to="coder")
        assert len(coder_tasks) == 1
        assert coder_tasks[0].title == "T1"

    async def test_update_task_full(self, db: Database) -> None:
        project = await self._make_project(db)
        task = Task(project_id=project.id, title="T", description="")
        await db.create_task(task)

        task.state = TaskState.IN_PROGRESS
        task.deliverable = "some output"
        task.iteration = 2
        await db.update_task(task)

        fetched = await db.get_task(task.id)
        assert fetched is not None
        assert fetched.state == TaskState.IN_PROGRESS
        assert fetched.deliverable == "some output"
        assert fetched.iteration == 2

    async def test_update_task_state(self, db: Database) -> None:
        project = await self._make_project(db)
        task = Task(project_id=project.id, title="T", description="")
        await db.create_task(task)

        await db.update_task_state(task.id, TaskState.ASSIGNED)
        fetched = await db.get_task(task.id)
        assert fetched is not None
        assert fetched.state == TaskState.ASSIGNED

    async def test_count_tasks(self, db: Database) -> None:
        project = await self._make_project(db)
        await db.create_task(Task(project_id=project.id, title="A", description=""))
        await db.create_task(
            Task(project_id=project.id, title="B", description="", state=TaskState.DONE)
        )
        assert await db.count_tasks(project.id) == 2
        assert await db.count_tasks(project.id, state=TaskState.DONE) == 1


# ============================================================
# Artifacts CRUD
# ============================================================


class TestArtifacts:
    """Tests for artifact create, read, list, search."""

    async def _setup(self, db: Database) -> tuple[Project, Task]:
        p = await db.create_project(Project(name="ArtProj", description="", idea="x"))
        t = Task(project_id=p.id, title="ArtTask", description="")
        await db.create_task(t)
        return p, t

    async def test_create_and_get_artifact(self, db: Database) -> None:
        p, t = await self._setup(db)
        artifact = Artifact(
            project_id=p.id, task_id=t.id, agent=AgentRole.CODER,
            filename="main.py", content="print('hello')",
            artifact_type="code",
        )
        await db.create_artifact(artifact)
        fetched = await db.get_artifact(artifact.id)
        assert fetched is not None
        assert fetched.filename == "main.py"
        assert fetched.content == "print('hello')"

    async def test_get_nonexistent_artifact(self, db: Database) -> None:
        assert await db.get_artifact("nope") is None

    async def test_list_artifacts_by_type(self, db: Database) -> None:
        p, t = await self._setup(db)
        a1 = Artifact(
            project_id=p.id, task_id=t.id, agent=AgentRole.CODER,
            filename="app.py", content="code", artifact_type="code",
        )
        a2 = Artifact(
            project_id=p.id, task_id=t.id, agent=AgentRole.SPEC,
            filename="spec.md", content="spec", artifact_type="doc",
        )
        await db.create_artifact(a1)
        await db.create_artifact(a2)

        code_arts = await db.list_artifacts(p.id, artifact_type="code")
        assert len(code_arts) == 1
        assert code_arts[0].filename == "app.py"

    async def test_get_latest_artifact(self, db: Database) -> None:
        p, t = await self._setup(db)
        v1 = Artifact(
            project_id=p.id, task_id=t.id, agent=AgentRole.CODER,
            filename="main.py", content="v1", artifact_type="code", version=1,
        )
        v2 = Artifact(
            project_id=p.id, task_id=t.id, agent=AgentRole.CODER,
            filename="main.py", content="v2", artifact_type="code", version=2,
        )
        await db.create_artifact(v1)
        await db.create_artifact(v2)

        latest = await db.get_latest_artifact(p.id, "main.py")
        assert latest is not None
        assert latest.content == "v2"
        assert latest.version == 2

    async def test_count_artifacts(self, db: Database) -> None:
        p, t = await self._setup(db)
        assert await db.count_artifacts(p.id) == 0
        await db.create_artifact(Artifact(
            project_id=p.id, task_id=t.id, agent=AgentRole.CODER,
            filename="f.py", content="x", artifact_type="code",
        ))
        assert await db.count_artifacts(p.id) == 1


# ============================================================
# Messages
# ============================================================


class TestMessages:
    """Tests for message saving and retrieval."""

    async def test_save_and_get_messages(self, db: Database) -> None:
        msg = {
            "id": "msg-1",
            "timestamp": "2024-01-01T00:00:00Z",
            "sender": "pm",
            "recipient": "research",
            "type": "task_assignment",
            "project_id": "proj-1",
            "task_id": "task-1",
            "payload": {"content": "Do research", "artifacts": []},
            "priority": "normal",
        }
        await db.save_message(msg)
        messages = await db.get_project_messages("proj-1")
        assert len(messages) == 1
        assert messages[0]["sender"] == "pm"
        assert messages[0]["payload"]["content"] == "Do research"

    async def test_save_duplicate_message_is_ignored(self, db: Database) -> None:
        msg = {
            "id": "dup-msg",
            "timestamp": "2024-01-01T00:00:00Z",
            "sender": "pm",
            "recipient": "coder",
            "type": "task_assignment",
            "project_id": "proj-1",
            "payload": {"content": "First"},
        }
        await db.save_message(msg)
        await db.save_message(msg)  # Duplicate — should be ignored
        messages = await db.get_project_messages("proj-1")
        assert len(messages) == 1

    async def test_get_messages_with_sender_filter(self, db: Database) -> None:
        for sender in ["pm", "coder", "pm"]:
            await db.save_message({
                "id": f"msg-{sender}-{id(sender)}",
                "timestamp": "2024-01-01T00:00:00Z",
                "sender": sender,
                "recipient": "critic",
                "type": "deliverable",
                "project_id": "proj-1",
                "payload": "content",
            })
        pm_msgs = await db.get_project_messages("proj-1", sender="pm")
        assert len(pm_msgs) == 2

    async def test_count_messages(self, db: Database) -> None:
        await db.save_message({
            "id": "m1", "timestamp": "2024-01-01T00:00:00Z",
            "sender": "pm", "recipient": "coder", "type": "task_assignment",
            "project_id": "proj-1", "payload": "x",
        })
        assert await db.count_messages("proj-1") == 1
        assert await db.count_messages("proj-999") == 0


# ============================================================
# Search
# ============================================================


class TestSearch:
    """Tests for search functionality."""

    async def test_search_projects_by_name(self, db: Database) -> None:
        await db.create_project(Project(name="Weather App", description="", idea="weather"))
        await db.create_project(Project(name="Todo List", description="", idea="todo"))
        results = await db.search_projects("Weather")
        assert len(results) == 1
        assert results[0].name == "Weather App"

    async def test_search_projects_by_idea(self, db: Database) -> None:
        await db.create_project(Project(name="P1", description="", idea="build a chatbot"))
        results = await db.search_projects("chatbot")
        assert len(results) == 1

    async def test_search_projects_no_match(self, db: Database) -> None:
        await db.create_project(Project(name="P1", description="", idea="something"))
        results = await db.search_projects("nonexistent")
        assert len(results) == 0

    async def test_search_artifacts(self, db: Database) -> None:
        p = await db.create_project(Project(name="P", description="", idea="x"))
        t = Task(project_id=p.id, title="T", description="")
        await db.create_task(t)
        await db.create_artifact(Artifact(
            project_id=p.id, task_id=t.id, agent=AgentRole.CODER,
            filename="utils.py", content="def helper(): pass",
            artifact_type="code",
        ))
        results = await db.search_artifacts(p.id, "helper")
        assert len(results) == 1
        assert results[0].filename == "utils.py"


# ============================================================
# Project Events / Timeline
# ============================================================


class TestProjectEvents:
    """Tests for the project events timeline."""

    async def test_create_project_records_event(self, db: Database) -> None:
        p = await db.create_project(Project(name="EP", description="", idea="x"))
        timeline = await db.get_project_timeline(p.id)
        assert len(timeline) >= 1
        assert any(e["event_type"] == "project_created" for e in timeline)

    async def test_status_update_records_event(self, db: Database) -> None:
        p = await db.create_project(Project(name="EP2", description="", idea="x"))
        await db.update_project(p.id, status="completed")
        timeline = await db.get_project_timeline(p.id)
        assert any(e["event_type"] == "status_changed" for e in timeline)


# ============================================================
# Diagnostics
# ============================================================


class TestDiagnostics:
    """Tests for get_stats()."""

    async def test_stats_empty_db(self, db: Database) -> None:
        stats = await db.get_stats()
        assert stats["schema_version"] == CURRENT_SCHEMA_VERSION
        assert stats["total_projects"] == 0
        assert stats["total_tasks"] == 0
        assert stats["total_artifacts"] == 0
        assert stats["total_messages"] == 0

    async def test_stats_with_data(self, db: Database) -> None:
        p = await db.create_project(Project(name="S", description="", idea="x"))
        t = Task(project_id=p.id, title="T", description="")
        await db.create_task(t)
        await db.create_artifact(Artifact(
            project_id=p.id, task_id=t.id, agent=AgentRole.CODER,
            filename="f.py", content="x", artifact_type="code",
        ))
        stats = await db.get_stats()
        assert stats["total_projects"] == 1
        assert stats["total_tasks"] == 1
        assert stats["total_artifacts"] == 1
