"""SQLite database layer for persistent state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from src.core.models import Artifact, Project, Task, TaskState

logger = structlog.get_logger()

# SQL schema for all tables
SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    idea TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'pending',
    assigned_to TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    iteration INTEGER NOT NULL DEFAULT 0,
    max_iterations INTEGER NOT NULL DEFAULT 5,
    parent_task_id TEXT DEFAULT '',
    deliverable TEXT DEFAULT '',
    feedback TEXT DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'normal',
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    sender TEXT NOT NULL,
    recipient TEXT NOT NULL,
    type TEXT NOT NULL,
    project_id TEXT NOT NULL,
    task_id TEXT DEFAULT '',
    payload TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal'
);

CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id);
CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project_id);
"""


class Database:
    """Async SQLite database for HCA persistent state."""

    def __init__(self, database_url: str = "sqlite:///data/hca.db") -> None:
        # Extract file path from URL
        self.db_path = database_url.replace("sqlite:///", "")
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the database and tables."""
        # Ensure the directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info("database_initialized", path=self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            logger.info("database_closed")

    @property
    def db(self) -> aiosqlite.Connection:
        """Get the database connection."""
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._db

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    async def create_project(self, project: Project) -> Project:
        """Insert a new project."""
        await self.db.execute(
            """INSERT INTO projects (id, name, description, created_at, updated_at, status, idea)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                project.id, project.name, project.description,
                project.created_at.isoformat(), project.updated_at.isoformat(),
                project.status, project.idea,
            ),
        )
        await self.db.commit()
        logger.info("project_created", project_id=project.id, name=project.name)
        return project

    async def get_project(self, project_id: str) -> Project | None:
        """Get a project by ID."""
        async with self.db.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Project(**dict(row))
        return None

    async def list_projects(self, status: str | None = None) -> list[Project]:
        """List all projects, optionally filtered by status."""
        if status:
            query = "SELECT * FROM projects WHERE status = ? ORDER BY created_at DESC"
            params: tuple = (status,)
        else:
            query = "SELECT * FROM projects ORDER BY created_at DESC"
            params = ()
        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [Project(**dict(row)) for row in rows]

    async def update_project_status(self, project_id: str, status: str) -> None:
        """Update a project's status."""
        await self.db.execute(
            "UPDATE projects SET status = ?, updated_at = datetime('now') WHERE id = ?",
            (status, project_id),
        )
        await self.db.commit()

    # --------------------------------------------------------
    # Tasks
    # --------------------------------------------------------

    async def create_task(self, task: Task) -> Task:
        """Insert a new task."""
        await self.db.execute(
            """INSERT INTO tasks
               (id, project_id, title, description, state, assigned_to,
                created_at, updated_at, iteration, max_iterations,
                parent_task_id, deliverable, feedback, priority)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.id, task.project_id, task.title, task.description,
                task.state.value, task.assigned_to.value if task.assigned_to else None,
                task.created_at.isoformat(), task.updated_at.isoformat(),
                task.iteration, task.max_iterations, task.parent_task_id,
                task.deliverable, task.feedback, task.priority.value,
            ),
        )
        await self.db.commit()
        logger.info("task_created", task_id=task.id, title=task.title)
        return task

    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        async with self.db.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Task(**dict(row))
        return None

    async def list_tasks(
        self, project_id: str, state: TaskState | None = None
    ) -> list[Task]:
        """List tasks for a project, optionally filtered by state."""
        if state:
            query = "SELECT * FROM tasks WHERE project_id = ? AND state = ? ORDER BY created_at"
            params: tuple = (project_id, state.value)
        else:
            query = "SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at"
            params = (project_id,)
        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [Task(**dict(row)) for row in rows]

    async def update_task(self, task: Task) -> None:
        """Update a task's full state."""
        await self.db.execute(
            """UPDATE tasks SET
               state = ?, assigned_to = ?, updated_at = datetime('now'),
               iteration = ?, deliverable = ?, feedback = ?
               WHERE id = ?""",
            (
                task.state.value,
                task.assigned_to.value if task.assigned_to else None,
                task.iteration, task.deliverable, task.feedback,
                task.id,
            ),
        )
        await self.db.commit()

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    async def create_artifact(self, artifact: Artifact) -> Artifact:
        """Insert a new artifact."""
        await self.db.execute(
            """INSERT INTO artifacts
               (id, project_id, task_id, agent, filename, content,
                artifact_type, created_at, version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact.id, artifact.project_id, artifact.task_id,
                artifact.agent.value, artifact.filename, artifact.content,
                artifact.artifact_type, artifact.created_at.isoformat(),
                artifact.version,
            ),
        )
        await self.db.commit()
        logger.info("artifact_created", artifact_id=artifact.id, filename=artifact.filename)
        return artifact

    async def list_artifacts(self, project_id: str) -> list[Artifact]:
        """List all artifacts for a project."""
        async with self.db.execute(
            "SELECT * FROM artifacts WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [Artifact(**dict(row)) for row in rows]

    # --------------------------------------------------------
    # Messages (for history / UI)
    # --------------------------------------------------------

    async def save_message(self, msg: dict[str, Any]) -> None:
        """Save a message to the database for history."""
        payload = msg.get("payload", {})
        if isinstance(payload, dict):
            payload = json.dumps(payload)

        await self.db.execute(
            """INSERT INTO messages (id, timestamp, sender, recipient, type,
               project_id, task_id, payload, priority)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg["id"], msg["timestamp"], msg["sender"], str(msg["recipient"]),
                msg["type"], msg["project_id"], msg.get("task_id", ""),
                payload, msg.get("priority", "normal"),
            ),
        )
        await self.db.commit()

    async def get_project_messages(self, project_id: str) -> list[dict[str, Any]]:
        """Get all messages for a project."""
        async with self.db.execute(
            "SELECT * FROM messages WHERE project_id = ? ORDER BY timestamp",
            (project_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                try:
                    d["payload"] = json.loads(d["payload"])
                except (json.JSONDecodeError, TypeError):
                    pass
                results.append(d)
            return results
