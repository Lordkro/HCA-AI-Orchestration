"""SQLite database layer for persistent state.

Provides async database access with:
- WAL mode for concurrent read/write performance
- Schema versioning and migration support
- Full CRUD for projects, tasks, artifacts, and messages
- Pagination support for large result sets
- Search across projects and artifacts
- Database statistics and diagnostics
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from src.core.models import Artifact, Project, Task, TaskState

logger = structlog.get_logger()

# ============================================================
# Schema Versioning
# ============================================================

CURRENT_SCHEMA_VERSION = 2

MIGRATIONS: dict[int, str] = {
    # Version 1: Initial schema
    1: """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );

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
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
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
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
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
    CREATE INDEX IF NOT EXISTS idx_tasks_assigned ON tasks(assigned_to);
    CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id);
    CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);
    CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
    CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project_id);
    CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender);
    CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
    """,

    # Version 2: Add project_events table for timeline tracking
    2: """
    CREATE TABLE IF NOT EXISTS project_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        agent TEXT,
        description TEXT NOT NULL DEFAULT '',
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_events_project ON project_events(project_id);
    CREATE INDEX IF NOT EXISTS idx_events_type ON project_events(event_type);
    """,
}


# ============================================================
# Database Error
# ============================================================


class DatabaseError(Exception):
    """Base exception for database errors."""


# ============================================================
# Database
# ============================================================


class Database:
    """Async SQLite database for HCA persistent state.

    Features:
    - WAL mode for better concurrent performance
    - Foreign keys enforced
    - Schema migration system
    - Comprehensive CRUD for all entities
    - Pagination and search support
    """

    def __init__(self, database_url: str = "sqlite:///data/hca.db") -> None:
        self.db_path = database_url.replace("sqlite:///", "")
        self._db: aiosqlite.Connection | None = None

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    async def initialize(self) -> None:
        """Create the database, apply migrations, and configure settings."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrent read/write
        await self._db.execute("PRAGMA journal_mode=WAL")
        # Enable foreign key enforcement
        await self._db.execute("PRAGMA foreign_keys=ON")
        # Reasonable busy timeout (5 seconds)
        await self._db.execute("PRAGMA busy_timeout=5000")

        await self._run_migrations()
        logger.info("database_initialized", path=self.db_path, version=CURRENT_SCHEMA_VERSION)

    async def close(self) -> None:
        """Close the database connection gracefully."""
        if self._db:
            # Checkpoint WAL before closing for clean state
            try:
                await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            await self._db.close()
            self._db = None
            logger.info("database_closed")

    @property
    def db(self) -> aiosqlite.Connection:
        """Get the database connection."""
        if self._db is None:
            raise DatabaseError("Database not initialized. Call initialize() first.")
        return self._db

    # --------------------------------------------------------
    # Migration System
    # --------------------------------------------------------

    async def _get_current_version(self) -> int:
        """Get the current schema version from the database."""
        try:
            async with self.db.execute(
                "SELECT MAX(version) as v FROM schema_version"
            ) as cursor:
                row = await cursor.fetchone()
                return row["v"] if row and row["v"] else 0
        except aiosqlite.OperationalError:
            # schema_version table doesn't exist yet
            return 0

    async def _run_migrations(self) -> None:
        """Apply any pending migrations."""
        current = await self._get_current_version()

        if current >= CURRENT_SCHEMA_VERSION:
            logger.debug("database_up_to_date", version=current)
            return

        for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
            sql = MIGRATIONS.get(version)
            if sql is None:
                raise DatabaseError(f"Missing migration for version {version}")

            logger.info("applying_migration", version=version)
            await self.db.executescript(sql)
            await self.db.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat()),
            )
            await self.db.commit()
            logger.info("migration_applied", version=version)

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    async def create_project(self, project: Project) -> Project:
        """Insert a new project."""
        try:
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

            # Record event
            await self._record_event(
                project.id, "project_created", description=f"Project '{project.name}' created"
            )

            logger.info("project_created", project_id=project.id, name=project.name)
            return project
        except aiosqlite.IntegrityError as e:
            raise DatabaseError(f"Project with ID {project.id} already exists") from e

    async def get_project(self, project_id: str) -> Project | None:
        """Get a project by ID."""
        async with self.db.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Project(**dict(row))
        return None

    async def list_projects(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[Project]:
        """List projects with optional status filter and pagination."""
        if status:
            query = "SELECT * FROM projects WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params: tuple = (status, limit, offset)
        else:
            query = "SELECT * FROM projects ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params = (limit, offset)
        async with self.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [Project(**dict(row)) for row in rows]

    async def update_project(self, project_id: str, **fields: Any) -> None:
        """Update specific fields on a project."""
        allowed = {"name", "description", "status"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return

        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [project_id]

        await self.db.execute(
            f"UPDATE projects SET {set_clause} WHERE id = ?", values
        )
        await self.db.commit()

        if "status" in updates:
            await self._record_event(
                project_id, "status_changed",
                description=f"Status changed to '{updates['status']}'"
            )

    async def delete_project(self, project_id: str) -> bool:
        """Delete a project and all associated data (cascades)."""
        cursor = await self.db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        await self.db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("project_deleted", project_id=project_id)
        return deleted

    async def count_projects(self, status: str | None = None) -> int:
        """Count projects, optionally filtered by status."""
        if status:
            query = "SELECT COUNT(*) as c FROM projects WHERE status = ?"
            params: tuple = (status,)
        else:
            query = "SELECT COUNT(*) as c FROM projects"
            params = ()
        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row["c"] if row else 0

    # --------------------------------------------------------
    # Tasks
    # --------------------------------------------------------

    async def create_task(self, task: Task) -> Task:
        """Insert a new task."""
        try:
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

            await self._record_event(
                task.project_id, "task_created",
                agent=task.assigned_to.value if task.assigned_to else None,
                description=f"Task '{task.title}' created",
            )

            logger.info("task_created", task_id=task.id, title=task.title)
            return task
        except aiosqlite.IntegrityError as e:
            raise DatabaseError(f"Failed to create task: {e}") from e

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
        self,
        project_id: str,
        state: TaskState | None = None,
        assigned_to: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Task]:
        """List tasks with filters and pagination."""
        conditions = ["project_id = ?"]
        params: list = [project_id]

        if state:
            conditions.append("state = ?")
            params.append(state.value)
        if assigned_to:
            conditions.append("assigned_to = ?")
            params.append(assigned_to)

        where = " AND ".join(conditions)
        params.extend([limit, offset])

        async with self.db.execute(
            f"SELECT * FROM tasks WHERE {where} ORDER BY created_at LIMIT ? OFFSET ?",
            params,
        ) as cursor:
            rows = await cursor.fetchall()
            return [Task(**dict(row)) for row in rows]

    async def update_task(self, task: Task) -> None:
        """Update a task's full state."""
        await self.db.execute(
            """UPDATE tasks SET
               state = ?, assigned_to = ?, updated_at = ?,
               iteration = ?, deliverable = ?, feedback = ?,
               title = ?, description = ?, priority = ?
               WHERE id = ?""",
            (
                task.state.value,
                task.assigned_to.value if task.assigned_to else None,
                datetime.now(timezone.utc).isoformat(),
                task.iteration, task.deliverable, task.feedback,
                task.title, task.description, task.priority.value,
                task.id,
            ),
        )
        await self.db.commit()

    async def update_task_state(self, task_id: str, state: TaskState) -> None:
        """Update only the task state (lightweight update)."""
        await self.db.execute(
            "UPDATE tasks SET state = ?, updated_at = ? WHERE id = ?",
            (state.value, datetime.now(timezone.utc).isoformat(), task_id),
        )
        await self.db.commit()

    async def count_tasks(self, project_id: str, state: TaskState | None = None) -> int:
        """Count tasks for a project."""
        if state:
            query = "SELECT COUNT(*) as c FROM tasks WHERE project_id = ? AND state = ?"
            params: tuple = (project_id, state.value)
        else:
            query = "SELECT COUNT(*) as c FROM tasks WHERE project_id = ?"
            params = (project_id,)
        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row["c"] if row else 0

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    async def create_artifact(self, artifact: Artifact) -> Artifact:
        """Insert a new artifact."""
        try:
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

            await self._record_event(
                artifact.project_id, "artifact_created",
                agent=artifact.agent.value,
                description=f"Artifact '{artifact.filename}' ({artifact.artifact_type}) created",
            )

            logger.info("artifact_created", artifact_id=artifact.id, filename=artifact.filename)
            return artifact
        except aiosqlite.IntegrityError as e:
            raise DatabaseError(f"Failed to create artifact: {e}") from e

    async def get_artifact(self, artifact_id: str) -> Artifact | None:
        """Get a single artifact by ID."""
        async with self.db.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Artifact(**dict(row))
        return None

    async def list_artifacts(
        self,
        project_id: str,
        artifact_type: str | None = None,
        task_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Artifact]:
        """List artifacts with filters and pagination."""
        conditions = ["project_id = ?"]
        params: list = [project_id]

        if artifact_type:
            conditions.append("artifact_type = ?")
            params.append(artifact_type)
        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)

        where = " AND ".join(conditions)
        params.extend([limit, offset])

        async with self.db.execute(
            f"SELECT * FROM artifacts WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ) as cursor:
            rows = await cursor.fetchall()
            return [Artifact(**dict(row)) for row in rows]

    async def get_latest_artifact(
        self, project_id: str, filename: str
    ) -> Artifact | None:
        """Get the latest version of an artifact by filename."""
        async with self.db.execute(
            """SELECT * FROM artifacts
               WHERE project_id = ? AND filename = ?
               ORDER BY version DESC LIMIT 1""",
            (project_id, filename),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Artifact(**dict(row))
        return None

    async def count_artifacts(self, project_id: str) -> int:
        """Count artifacts for a project."""
        async with self.db.execute(
            "SELECT COUNT(*) as c FROM artifacts WHERE project_id = ?",
            (project_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row["c"] if row else 0

    # --------------------------------------------------------
    # Messages (for history / UI)
    # --------------------------------------------------------

    async def save_message(self, msg: dict[str, Any]) -> None:
        """Save a message to the database for history.

        Uses INSERT OR IGNORE to handle duplicate message IDs gracefully
        (e.g., if the same message is processed after a retry).
        """
        payload = msg.get("payload", {})
        if isinstance(payload, dict):
            payload = json.dumps(payload, default=str)

        try:
            await self.db.execute(
                """INSERT OR IGNORE INTO messages
                   (id, timestamp, sender, recipient, type,
                    project_id, task_id, payload, priority)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    msg["id"], msg["timestamp"], str(msg["sender"]),
                    str(msg["recipient"]), str(msg["type"]),
                    msg["project_id"], msg.get("task_id", ""),
                    payload, msg.get("priority", "normal"),
                ),
            )
            await self.db.commit()
        except aiosqlite.Error as e:
            logger.error("save_message_failed", error=str(e), msg_id=msg.get("id"))

    async def get_project_messages(
        self,
        project_id: str,
        limit: int = 200,
        offset: int = 0,
        sender: str | None = None,
        msg_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages for a project with optional filters and pagination."""
        conditions = ["project_id = ?"]
        params: list = [project_id]

        if sender:
            conditions.append("sender = ?")
            params.append(sender)
        if msg_type:
            conditions.append("type = ?")
            params.append(msg_type)

        where = " AND ".join(conditions)
        params.extend([limit, offset])

        async with self.db.execute(
            f"SELECT * FROM messages WHERE {where} ORDER BY timestamp LIMIT ? OFFSET ?",
            params,
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

    async def count_messages(self, project_id: str) -> int:
        """Count messages for a project."""
        async with self.db.execute(
            "SELECT COUNT(*) as c FROM messages WHERE project_id = ?",
            (project_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row["c"] if row else 0

    # --------------------------------------------------------
    # Project Events (Timeline)
    # --------------------------------------------------------

    async def _record_event(
        self,
        project_id: str,
        event_type: str,
        *,
        agent: str | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a project event for the timeline."""
        try:
            await self.db.execute(
                """INSERT INTO project_events
                   (project_id, event_type, agent, description, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    project_id, event_type, agent, description,
                    json.dumps(metadata or {}, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await self.db.commit()
        except aiosqlite.Error as e:
            logger.debug("event_recording_failed", error=str(e))

    async def get_project_timeline(
        self, project_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get the event timeline for a project."""
        async with self.db.execute(
            """SELECT * FROM project_events
               WHERE project_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (project_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except (json.JSONDecodeError, TypeError):
                    d["metadata"] = {}
                results.append(d)
            return list(reversed(results))

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    async def search_projects(self, query: str, limit: int = 20) -> list[Project]:
        """Search projects by name, description, or idea content."""
        pattern = f"%{query}%"
        async with self.db.execute(
            """SELECT * FROM projects
               WHERE name LIKE ? OR description LIKE ? OR idea LIKE ?
               ORDER BY created_at DESC LIMIT ?""",
            (pattern, pattern, pattern, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [Project(**dict(row)) for row in rows]

    async def search_artifacts(
        self, project_id: str, query: str, limit: int = 20
    ) -> list[Artifact]:
        """Search artifacts by filename or content."""
        pattern = f"%{query}%"
        async with self.db.execute(
            """SELECT * FROM artifacts
               WHERE project_id = ? AND (filename LIKE ? OR content LIKE ?)
               ORDER BY created_at DESC LIMIT ?""",
            (project_id, pattern, pattern, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [Artifact(**dict(row)) for row in rows]

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        projects = await self.count_projects()
        active_projects = await self.count_projects(status="active")

        # Get total tasks by state
        task_states: dict[str, int] = {}
        async with self.db.execute(
            "SELECT state, COUNT(*) as c FROM tasks GROUP BY state"
        ) as cursor:
            async for row in cursor:
                task_states[row["state"]] = row["c"]

        # Get total artifacts by type
        artifact_types: dict[str, int] = {}
        async with self.db.execute(
            "SELECT artifact_type, COUNT(*) as c FROM artifacts GROUP BY artifact_type"
        ) as cursor:
            async for row in cursor:
                artifact_types[row["artifact_type"]] = row["c"]

        # Get total messages
        async with self.db.execute("SELECT COUNT(*) as c FROM messages") as cursor:
            row = await cursor.fetchone()
            total_messages = row["c"] if row else 0

        # Database file size
        db_size_bytes = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0

        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "total_projects": projects,
            "active_projects": active_projects,
            "task_states": task_states,
            "total_tasks": sum(task_states.values()),
            "artifact_types": artifact_types,
            "total_artifacts": sum(artifact_types.values()),
            "total_messages": total_messages,
            "database_size_mb": round(db_size_bytes / (1024 * 1024), 2),
        }
