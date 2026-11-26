//! Projects, tasks and task history.

use chrono::{DateTime, Utc};
use hca_core::{
    validate_plan, Limits, PlanError, Project, ProjectEvent, ProjectId, ProjectStatus, Reservation, Task,
    TaskEvent, TaskId, TaskState, Transition,
};
use sqlx::{Sqlite, SqliteConnection, Transaction};

use crate::{time, wire_name, JobId, NewJob, Result, Store, StoreError};

/// An activity log entry to add.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewActivity {
    /// The project.
    pub project_id: ProjectId,
    /// The task, if the entry is about one.
    pub task_id: Option<TaskId>,
    /// Who acted, e.g. `"coder"`, `"critic"`, `"user"`, `"system"`.
    pub sender: String,
    /// Who it was for.
    pub recipient: String,
    /// What happened, e.g. `"deliverable"`, `"feedback"`, `"plan"`.
    pub kind: String,
    /// The text.
    pub content: String,
    /// When.
    pub at: DateTime<Utc>,
}

/// A stored activity log entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Activity {
    /// Position in the log.
    pub seq: i64,
    /// The entry.
    pub entry: NewActivity,
}

type ActivityRow = (i64, String, Option<String>, String, String, String, String, String);

impl TryFrom<ActivityRow> for Activity {
    type Error = StoreError;

    fn try_from((seq, project, task, sender, recipient, kind, content, at): ActivityRow) -> Result<Self> {
        let bad_id = |e: uuid::Error| StoreError::Corrupt(format!("bad id in activity: {e}"));
        Ok(Self {
            seq,
            entry: NewActivity {
                project_id: project.parse().map_err(bad_id)?,
                task_id: task.as_deref().map(str::parse).transpose().map_err(bad_id)?,
                sender,
                recipient,
                kind,
                content,
                at: time::decode(&at)?,
            },
        })
    }
}

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
    /// status, its queued jobs are cancelled in the same transaction, and
    /// its running jobs too unless it completed.
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
            // A completed project's running job is the one that completed
            // it, so let it finish; a failed or cancelled project stops all
            // of its work.
            let include_running = status != ProjectStatus::Completed;
            crate::jobs::cancel_for_project(&mut tx, id, include_running, now).await?;
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
        self.charge_usage(id, None, tokens, limits, now).await
    }

    /// Charges one model call's tokens to a project and, if given, the task
    /// it was for, in one transaction, so the two totals always agree. If
    /// that spends the budget, the project fails and its jobs are cancelled
    /// in the same transaction.
    ///
    /// The task's `updated_at` is left alone: the task timeout is measured
    /// from it, so recording spend must not reset it.
    ///
    /// Returns `true` while the project is still within budget.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] if the project or task doesn't exist.
    pub async fn charge_usage(
        &self,
        id: ProjectId,
        task: Option<TaskId>,
        tokens: u64,
        limits: &Limits,
        now: DateTime<Utc>,
    ) -> Result<bool> {
        let mut tx = self.writer.begin().await?;
        let mut project = load_project(&mut tx, id).await?;
        if let Some(task_id) = task {
            let mut task = load_task(&mut tx, task_id).await?;
            task.record_tokens(tokens);
            sqlx::query("UPDATE tasks SET data = ? WHERE id = ?")
                .bind(serde_json::to_string(&task)?)
                .bind(task_id.to_string())
                .execute(&mut *tx)
                .await?;
        }
        let was_terminal = project.status().is_terminal();
        let within = project.charge_tokens(tokens, limits, now);
        save_project(&mut tx, &project).await?;
        if !was_terminal && project.status().is_terminal() {
            crate::jobs::cancel_for_project(&mut tx, id, true, now).await?;
        }
        tx.commit().await?;
        Ok(within)
    }

    /// Reserves a model call's worst-case size against a project's budget,
    /// before the call is made. See [`Project::reserve_tokens`].
    ///
    /// A [`Reservation::Refused`] call can't fit even once every other call
    /// settles. The caller may make it smaller, or give up with
    /// [`exhaust_budget`](Self::exhaust_budget).
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::ProjectNotActive`] if the project doesn't
    /// accept work (paused, or ended), or [`StoreError::NotFound`].
    pub async fn reserve_usage(&self, id: ProjectId, tokens: u64, limits: &Limits) -> Result<Reservation> {
        let mut tx = self.writer.begin().await?;
        let mut project = load_project(&mut tx, id).await?;
        if !project.accepts_work() {
            return Err(StoreError::ProjectNotActive(project.status()));
        }
        let reservation = project.reserve_tokens(tokens, limits);
        if reservation == Reservation::Granted {
            save_project(&mut tx, &project).await?;
            tx.commit().await?;
        }
        Ok(reservation)
    }

    /// Fails a project because its remaining budget can't cover even the
    /// smallest next model call, and cancels its jobs, in one transaction.
    /// Does nothing to a project that already ended.
    ///
    /// Returns whether the project was failed by this call.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] if the project doesn't exist.
    pub async fn exhaust_budget(&self, id: ProjectId, now: DateTime<Utc>) -> Result<bool> {
        let mut tx = self.writer.begin().await?;
        let mut project = load_project(&mut tx, id).await?;
        if project.status().is_terminal() {
            return Ok(false);
        }
        project.exhaust_budget(now);
        save_project(&mut tx, &project).await?;
        crate::jobs::cancel_for_project(&mut tx, id, true, now).await?;
        tx.commit().await?;
        Ok(true)
    }

    /// Settles a reservation made by [`reserve_usage`](Self::reserve_usage):
    /// releases `reserved` and charges `charged` to the project and, if
    /// given, the task, in one transaction. If that spends the budget, the
    /// project fails and its jobs are cancelled.
    ///
    /// Returns `true` while the project is still within budget.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] if the project or task doesn't exist.
    pub async fn settle_usage(
        &self,
        id: ProjectId,
        task: Option<TaskId>,
        reserved: u64,
        charged: u64,
        limits: &Limits,
        now: DateTime<Utc>,
    ) -> Result<bool> {
        let mut tx = self.writer.begin().await?;
        let mut project = load_project(&mut tx, id).await?;
        if let Some(task_id) = task {
            let mut task = load_task(&mut tx, task_id).await?;
            task.record_tokens(charged);
            sqlx::query("UPDATE tasks SET data = ? WHERE id = ?")
                .bind(serde_json::to_string(&task)?)
                .bind(task_id.to_string())
                .execute(&mut *tx)
                .await?;
        }
        let was_terminal = project.status().is_terminal();
        let within = project.settle_tokens(reserved, charged, limits, now);
        save_project(&mut tx, &project).await?;
        if !was_terminal && project.status().is_terminal() {
            crate::jobs::cancel_for_project(&mut tx, id, true, now).await?;
        }
        tx.commit().await?;
        Ok(within)
    }

    /// Charges in full every reservation left open by a process that
    /// stopped mid-call. Run at startup, before any worker. A project this
    /// pushes out of budget fails, and its jobs are cancelled.
    ///
    /// Returns how many projects had open reservations.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] on failure.
    pub async fn charge_open_reservations(&self, limits: &Limits, now: DateTime<Utc>) -> Result<usize> {
        let mut charged = 0;
        for listed in self.projects(None).await? {
            if listed.tokens_reserved() == 0 {
                continue;
            }
            let mut tx = self.writer.begin().await?;
            let mut project = load_project(&mut tx, listed.id).await?;
            let was_terminal = project.status().is_terminal();
            project.charge_reservations(limits, now);
            save_project(&mut tx, &project).await?;
            if !was_terminal && project.status().is_terminal() {
                crate::jobs::cancel_for_project(&mut tx, project.id, true, now).await?;
            }
            tx.commit().await?;
            charged += 1;
        }
        Ok(charged)
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
        self.apply_task_event_with_job(id, event, limits, now, |_, _| None)
            .await
            .map(|(transition, _)| transition)
    }

    /// Like [`apply_task_event`](Self::apply_task_event), and also queues the
    /// job that `next_job` returns for the resulting state, in the same
    /// transaction. A task can then never end up assigned with no job to
    /// work on it, even if the process crashes.
    ///
    /// # Errors
    ///
    /// Same as [`apply_task_event`](Self::apply_task_event).
    pub async fn apply_task_event_with_job(
        &self,
        id: TaskId,
        event: TaskEvent,
        limits: &Limits,
        now: DateTime<Utc>,
        next_job: impl FnOnce(&Transition, &Task) -> Option<NewJob> + Send,
    ) -> Result<(Transition, Option<JobId>)> {
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
        let job = match next_job(&transition, &task) {
            Some(job) => Some(crate::jobs::insert_job(&mut tx, &job, now).await?),
            None => None,
        };
        tx.commit().await?;
        Ok((transition, job))
    }

    /// Adds an entry to a project's activity log.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] if the insert fails.
    pub async fn log_activity(&self, entry: &NewActivity) -> Result<i64> {
        let seq: i64 = sqlx::query_scalar(
            "INSERT INTO activity (project_id, task_id, sender, recipient, kind, content, at) \
             VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING seq",
        )
        .bind(entry.project_id.to_string())
        .bind(entry.task_id.map(|t| t.to_string()))
        .bind(&entry.sender)
        .bind(&entry.recipient)
        .bind(&entry.kind)
        .bind(&entry.content)
        .bind(time::encode(entry.at))
        .fetch_one(&self.writer)
        .await?;
        Ok(seq)
    }

    /// A project's activity log, oldest first.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] or a decoding error on failure.
    pub async fn activity(&self, project_id: ProjectId) -> Result<Vec<Activity>> {
        let rows: Vec<ActivityRow> = sqlx::query_as(
            "SELECT seq, project_id, task_id, sender, recipient, kind, content, at FROM activity \
             WHERE project_id = ? ORDER BY seq",
        )
        .bind(project_id.to_string())
        .fetch_all(&self.reader)
        .await?;
        rows.into_iter().map(Activity::try_from).collect()
    }

    /// The most recent activity of `kind` for a task.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] or a decoding error on failure.
    pub async fn latest_activity(&self, task_id: TaskId, kind: &str) -> Result<Option<Activity>> {
        let row: Option<ActivityRow> = sqlx::query_as(
            "SELECT seq, project_id, task_id, sender, recipient, kind, content, at FROM activity \
             WHERE task_id = ? AND kind = ? ORDER BY seq DESC LIMIT 1",
        )
        .bind(task_id.to_string())
        .bind(kind)
        .fetch_optional(&self.reader)
        .await?;
        row.map(Activity::try_from).transpose()
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
