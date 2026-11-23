//! Projects, tasks and task history.

use chrono::{DateTime, Utc};
use hca_core::{
    validate_plan, Limits, PlanError, Project, ProjectEvent, ProjectId, ProjectStatus, Task, TaskEvent,
    TaskId, TaskState, Transition,
};
use sqlx::{Sqlite, SqliteConnection, Transaction};

use crate::{time, wire_name, Result, Store, StoreError};

/// One accepted task event, as recorded in the history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskEventRecord {
    /// Position in the global history; increases with every event.
    pub seq: i64,
    /// The task.
    pub task_id: TaskId,
    /// State before the event.
    pub from: TaskState,
    /// State after the event.
    pub to: TaskState,
    /// The event itself.
    pub event: TaskEvent,
    /// When it was applied.
    pub at: DateTime<Utc>,
}

impl Store {
    /// Saves a new project.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] if the insert fails, for example
    /// because the id already exists.
    pub async fn create_project(&self, project: &Project) -> Result<()> {
        sqlx::query("INSERT INTO projects (id, status, created_at, updated_at, data) VALUES (?, ?, ?, ?, ?)")
            .bind(project.id.to_string())
            .bind(wire_name(&project.status())?)
            .bind(time::encode(project.created_at()))
            .bind(time::encode(project.updated_at()))
            .bind(serde_json::to_string(project)?)
            .execute(&self.writer)
            .await?;
        Ok(())
    }

    /// Loads a project.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] if there is no such project.
    pub async fn project(&self, id: ProjectId) -> Result<Project> {
        let data: Option<String> = sqlx::query_scalar("SELECT data FROM projects WHERE id = ?")
            .bind(id.to_string())
            .fetch_optional(&self.reader)
            .await?;
        let data = data.ok_or_else(|| not_found("project", &id))?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Lists projects, newest first, optionally only those with `status`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] or [`StoreError::Json`] on failure.
    pub async fn projects(&self, status: Option<ProjectStatus>) -> Result<Vec<Project>> {
        let rows: Vec<String> = match status {
            Some(status) => {
                sqlx::query_scalar("SELECT data FROM projects WHERE status = ? ORDER BY created_at DESC, id")
                    .bind(wire_name(&status)?)
                    .fetch_all(&self.reader)
                    .await?
            }
            None => {
                sqlx::query_scalar("SELECT data FROM projects ORDER BY created_at DESC, id")
                    .fetch_all(&self.reader)
                    .await?
            }
        };
        rows.iter().map(|data| Ok(serde_json::from_str(data)?)).collect()
    }

    /// Applies a project event atomically. When the project reaches a final
    /// status, its queued and running jobs are cancelled in the same
    /// transaction.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] if there is no such project, or
    /// [`StoreError::Project`] if the event isn't allowed. Nothing is
    /// written in either case.
    pub async fn apply_project_event(
        &self,
        id: ProjectId,
        event: ProjectEvent,
        now: DateTime<Utc>,
    ) -> Result<ProjectStatus> {
        let mut tx = self.writer.begin().await?;
        let mut project = load_project(&mut tx, id).await?;
        let status = project.apply(event, now)?;
        save_project(&mut tx, &project).await?;
        if status.is_terminal() {
            crate::jobs::cancel_for_project(&mut tx, id, now).await?;
        }
        tx.commit().await?;
        Ok(status)
    }

    /// Charges tokens to a project. If that spends the budget, the project
    /// fails and its jobs are cancelled, all in one transaction.
    ///
    /// Returns `true` while the project is still within budget.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] if there is no such project.
    pub async fn charge_tokens(
        &self,
        id: ProjectId,
        tokens: u64,
        limits: &Limits,
        now: DateTime<Utc>,
    ) -> Result<bool> {
        let mut tx = self.writer.begin().await?;
        let mut project = load_project(&mut tx, id).await?;
        let was_terminal = project.status().is_terminal();
        let within = project.charge_tokens(tokens, limits, now);
        save_project(&mut tx, &project).await?;
        if !was_terminal && project.status().is_terminal() {
            crate::jobs::cancel_for_project(&mut tx, id, now).await?;
        }
        tx.commit().await?;
        Ok(within)
    }

    /// Validates a plan with the core rules and saves all of its tasks, or
    /// none of them.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::ProjectNotActive`] if the project isn't active,
    /// [`StoreError::PlanExists`] if it already has tasks, or
    /// [`StoreError::Plan`] if the plan is invalid or contains tasks of
    /// another project.
    pub async fn create_plan(&self, project_id: ProjectId, tasks: &[Task], limits: &Limits) -> Result<()> {
        if let Some(stranger) = tasks.iter().find(|t| t.project_id != project_id) {
            return Err(PlanError::MixedProjects(stranger.id).into());
        }
        validate_plan(tasks, limits)?;

        let mut tx = self.writer.begin().await?;
        let project = load_project(&mut tx, project_id).await?;
        if !project.accepts_work() {
            return Err(StoreError::ProjectNotActive(project.status()));
        }
        let existing: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tasks WHERE project_id = ?")
            .bind(project_id.to_string())
            .fetch_one(&mut *tx)
            .await?;
        if existing > 0 {
            return Err(StoreError::PlanExists);
        }
        for task in tasks {
            sqlx::query(
                "INSERT INTO tasks (id, project_id, state, created_at, updated_at, data) VALUES (?, ?, ?, ?, ?, ?)",
            )
            .bind(task.id.to_string())
            .bind(project_id.to_string())
            .bind(wire_name(&task.state())?)
            .bind(time::encode(task.created_at()))
            .bind(time::encode(task.updated_at()))
            .bind(serde_json::to_string(task)?)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Loads a task.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] if there is no such task.
    pub async fn task(&self, id: TaskId) -> Result<Task> {
        let data: Option<String> = sqlx::query_scalar("SELECT data FROM tasks WHERE id = ?")
            .bind(id.to_string())
            .fetch_optional(&self.reader)
            .await?;
        let data = data.ok_or_else(|| not_found("task", &id))?;
        Ok(serde_json::from_str(&data)?)
    }

    /// All tasks of a project, oldest first.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] or [`StoreError::Json`] on failure.
    pub async fn tasks(&self, project_id: ProjectId) -> Result<Vec<Task>> {
        let rows: Vec<String> =
            sqlx::query_scalar("SELECT data FROM tasks WHERE project_id = ? ORDER BY created_at, id")
                .bind(project_id.to_string())
                .fetch_all(&self.reader)
                .await?;
        rows.iter().map(|data| Ok(serde_json::from_str(data)?)).collect()
    }

    /// Applies a task event atomically and records it in the history.
    ///
    /// Only active projects accept task events, except `Fail`, which is
    /// always allowed so work can be stopped on a paused or finished
    /// project.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] for an unknown task,
    /// [`StoreError::ProjectNotActive`] if the project doesn't accept work,
    /// or [`StoreError::Task`] if the state machine refuses the event.
    /// Nothing is written in any error case.
    pub async fn apply_task_event(
        &self,
        id: TaskId,
        event: TaskEvent,
        limits: &Limits,
        now: DateTime<Utc>,
    ) -> Result<Transition> {
        let mut tx = self.writer.begin().await?;
        let mut task = load_task(&mut tx, id).await?;
        let project = load_project(&mut tx, task.project_id).await?;
        if !project.accepts_work() && !matches!(event, TaskEvent::Fail { .. }) {
            return Err(StoreError::ProjectNotActive(project.status()));
        }

        let event_json = serde_json::to_string(&event)?;
        let transition = task.apply(event, limits, now)?;

        sqlx::query("UPDATE tasks SET state = ?, updated_at = ?, data = ? WHERE id = ?")
            .bind(wire_name(&task.state())?)
            .bind(time::encode(task.updated_at()))
            .bind(serde_json::to_string(&task)?)
            .bind(id.to_string())
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "INSERT INTO task_events (task_id, project_id, from_state, to_state, event, at) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(id.to_string())
        .bind(task.project_id.to_string())
        .bind(wire_name(&transition.from)?)
        .bind(wire_name(&transition.to)?)
        .bind(event_json)
        .bind(time::encode(now))
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(transition)
    }

    /// A task's history, oldest first.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] or a decoding error on failure.
    pub async fn task_history(&self, id: TaskId) -> Result<Vec<TaskEventRecord>> {
        let rows: Vec<(i64, String, String, String, String)> = sqlx::query_as(
            "SELECT seq, from_state, to_state, event, at FROM task_events WHERE task_id = ? ORDER BY seq",
        )
        .bind(id.to_string())
        .fetch_all(&self.reader)
        .await?;
        rows.into_iter()
            .map(|(seq, from, to, event, at)| {
                Ok(TaskEventRecord {
                    seq,
                    task_id: id,
                    from: serde_json::from_value(serde_json::Value::String(from))?,
                    to: serde_json::from_value(serde_json::Value::String(to))?,
                    event: serde_json::from_str(&event)?,
                    at: time::decode(&at)?,
                })
            })
            .collect()
    }
}

pub(crate) async fn load_project(conn: &mut SqliteConnection, id: ProjectId) -> Result<Project> {
    let data: Option<String> = sqlx::query_scalar("SELECT data FROM projects WHERE id = ?")
        .bind(id.to_string())
        .fetch_optional(&mut *conn)
        .await?;
    let data = data.ok_or_else(|| not_found("project", &id))?;
    Ok(serde_json::from_str(&data)?)
}

async fn save_project(tx: &mut Transaction<'_, Sqlite>, project: &Project) -> Result<()> {
    sqlx::query("UPDATE projects SET status = ?, updated_at = ?, data = ? WHERE id = ?")
        .bind(wire_name(&project.status())?)
        .bind(time::encode(project.updated_at()))
        .bind(serde_json::to_string(project)?)
        .bind(project.id.to_string())
        .execute(&mut **tx)
        .await?;
    Ok(())
}

async fn load_task(conn: &mut SqliteConnection, id: TaskId) -> Result<Task> {
    let data: Option<String> = sqlx::query_scalar("SELECT data FROM tasks WHERE id = ?")
        .bind(id.to_string())
        .fetch_optional(&mut *conn)
        .await?;
    let data = data.ok_or_else(|| not_found("task", &id))?;
    Ok(serde_json::from_str(&data)?)
}

fn not_found(kind: &'static str, id: &impl ToString) -> StoreError {
    StoreError::NotFound { kind, id: id.to_string() }
}
