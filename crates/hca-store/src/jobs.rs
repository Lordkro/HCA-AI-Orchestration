//! The durable job queue that replaces Redis Streams.
//!
//! - **At-least-once with leases.** A worker claims a job for a lease
//!   period. If it crashes, the lease expires and another worker can claim
//!   the job. A worker whose lease was taken over can't complete the job.
//! - **Bounded attempts.** Every claim counts as an attempt. After
//!   `max_attempts` the job is dead instead of retrying forever.
//! - **Project-aware.** Jobs of paused projects wait in the queue instead of
//!   being dropped. Jobs of finished projects are cancelled.

use std::fmt;

use chrono::{DateTime, Duration, Utc};
use hca_core::{ProjectId, TaskId};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::SqliteRow;
use sqlx::{Row, Sqlite, Transaction};

use crate::projects::load_project;
use crate::{time, wire_name, Result, Store, StoreError};

/// Identifies a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JobId(pub i64);

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Where a job is in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Waiting to be claimed.
    Queued,
    /// Claimed by a worker that holds a lease.
    Running,
    /// Finished successfully.
    Done,
    /// Failed on its last allowed attempt.
    Dead,
    /// Its project finished before it ran.
    Cancelled,
}

/// A job to enqueue.
#[derive(Clone, Debug, PartialEq)]
pub struct NewJob {
    /// The project the job belongs to.
    pub project_id: ProjectId,
    /// The task, if the job works on one.
    pub task_id: Option<TaskId>,
    /// What kind of work this is; workers claim by kind.
    pub kind: String,
    /// Kind-specific input.
    pub payload: serde_json::Value,
    /// Claims allowed before the job is dead. At least 1.
    pub max_attempts: u32,
    /// Don't run before this time.
    pub run_after: DateTime<Utc>,
}

impl NewJob {
    /// A job that can run now, with 3 attempts.
    pub fn new(
        project_id: ProjectId,
        kind: impl Into<String>,
        payload: serde_json::Value,
        now: DateTime<Utc>,
    ) -> Self {
        Self { project_id, task_id: None, kind: kind.into(), payload, max_attempts: 3, run_after: now }
    }

    /// Attaches the job to a task.
    #[must_use]
    pub fn for_task(mut self, task_id: TaskId) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// Sets the attempt limit (minimum 1).
    #[must_use]
    pub fn max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }
}

/// A job as stored.
#[derive(Clone, Debug, PartialEq)]
pub struct Job {
    /// The job's id.
    pub id: JobId,
    /// The project it belongs to.
    pub project_id: ProjectId,
    /// The task, if any.
    pub task_id: Option<TaskId>,
    /// What kind of work.
    pub kind: String,
    /// Kind-specific input.
    pub payload: serde_json::Value,
    /// Current status.
    pub status: JobStatus,
    /// Claims so far.
    pub attempts: u32,
    /// Claims allowed.
    pub max_attempts: u32,
    /// Earliest time it may run.
    pub run_after: DateTime<Utc>,
    /// The worker holding the lease, while running.
    pub locked_by: Option<String>,
    /// When the lease ends, while running.
    pub locked_until: Option<DateTime<Utc>>,
    /// The most recent failure message.
    pub last_error: Option<String>,
}

const JOB_COLUMNS: &str = "id, project_id, task_id, kind, payload, status, attempts, max_attempts, \
                           run_after, locked_by, locked_until, last_error";

impl Store {
    /// Adds a job to the queue.
    ///
    /// Jobs can be queued for active or paused projects; paused ones wait.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::ProjectNotActive`] if the project is finished,
    /// or [`StoreError::NotFound`] if it doesn't exist.
    pub async fn enqueue(&self, job: &NewJob, now: DateTime<Utc>) -> Result<JobId> {
        let mut tx = self.writer.begin().await?;
        let project = load_project(&mut tx, job.project_id).await?;
        if project.status().is_terminal() {
            return Err(StoreError::ProjectNotActive(project.status()));
        }
        let id = insert_job(&mut tx, job, now).await?;
        tx.commit().await?;
        Ok(id)
    }

    /// Claims the oldest runnable job of one of `kinds` (any kind if empty)
    /// for `lease`.
    ///
    /// Runnable means: the project is active, and the job is queued and due,
    /// or running with an expired lease. A job whose lease expired on its
    /// last attempt is marked dead instead of claimed.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] on failure.
    pub async fn claim(
        &self,
        worker: &str,
        kinds: &[&str],
        lease: Duration,
        now: DateTime<Utc>,
    ) -> Result<Option<Job>> {
        let now_text = time::encode(now);
        let kind_filter = if kinds.is_empty() {
            String::new()
        } else {
            format!(" AND j.kind IN ({})", vec!["?"; kinds.len()].join(", "))
        };
        let select = format!(
            "SELECT j.id, j.attempts, j.max_attempts FROM jobs j JOIN projects p ON p.id = j.project_id \
             WHERE p.status = 'active' \
             AND ((j.status = 'queued' AND j.run_after <= ?) OR (j.status = 'running' AND j.locked_until <= ?))\
             {kind_filter} ORDER BY j.run_after, j.id LIMIT 1"
        );

        let mut tx = self.writer.begin().await?;
        loop {
            let mut query = sqlx::query_as::<_, (i64, i64, i64)>(&select).bind(&now_text).bind(&now_text);
            for kind in kinds {
                query = query.bind(*kind);
            }
            let Some((id, attempts, max_attempts)) = query.fetch_optional(&mut *tx).await? else {
                tx.commit().await?;
                return Ok(None);
            };

            if attempts >= max_attempts {
                sqlx::query(
                    "UPDATE jobs SET status = 'dead', locked_by = NULL, locked_until = NULL, \
                     last_error = COALESCE(last_error, 'lease expired on the last attempt'), updated_at = ? \
                     WHERE id = ?",
                )
                .bind(&now_text)
                .bind(id)
                .execute(&mut *tx)
                .await?;
                continue;
            }

            let row = sqlx::query(&format!(
                "UPDATE jobs SET status = 'running', attempts = attempts + 1, locked_by = ?, \
                 locked_until = ?, updated_at = ? WHERE id = ? RETURNING {JOB_COLUMNS}"
            ))
            .bind(worker)
            .bind(time::encode(now + lease))
            .bind(&now_text)
            .bind(id)
            .fetch_one(&mut *tx)
            .await?;
            let job = job_from_row(&row)?;
            tx.commit().await?;
            return Ok(Some(job));
        }
    }

    /// Extends the lease on a job the worker holds.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::LeaseLost`] if the worker no longer holds it.
    pub async fn extend_lease(
        &self,
        id: JobId,
        worker: &str,
        lease: Duration,
        now: DateTime<Utc>,
    ) -> Result<()> {
        let result = sqlx::query(
            "UPDATE jobs SET locked_until = ?, updated_at = ? \
             WHERE id = ? AND status = 'running' AND locked_by = ? AND locked_until > ?",
        )
        .bind(time::encode(now + lease))
        .bind(time::encode(now))
        .bind(id.0)
        .bind(worker)
        .bind(time::encode(now))
        .execute(&self.writer)
        .await?;
        if result.rows_affected() == 0 {
            return Err(StoreError::LeaseLost(id));
        }
        Ok(())
    }

    /// Marks a job done.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::LeaseLost`] if the worker no longer holds it,
    /// for example because the lease expired and another worker claimed it,
    /// or the project finished and the job was cancelled.
    pub async fn complete(&self, id: JobId, worker: &str, now: DateTime<Utc>) -> Result<()> {
        let result = sqlx::query(
            "UPDATE jobs SET status = 'done', locked_by = NULL, locked_until = NULL, updated_at = ? \
             WHERE id = ? AND status = 'running' AND locked_by = ?",
        )
        .bind(time::encode(now))
        .bind(id.0)
        .bind(worker)
        .execute(&self.writer)
        .await?;
        if result.rows_affected() == 0 {
            return Err(StoreError::LeaseLost(id));
        }
        Ok(())
    }

    /// Records a failed attempt. The job is queued again after
    /// `retry_after`, or marked dead if this was its last attempt.
    ///
    /// Returns the job's new status.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::LeaseLost`] if the worker no longer holds it.
    pub async fn fail(
        &self,
        id: JobId,
        worker: &str,
        error: &str,
        retry_after: Duration,
        now: DateTime<Utc>,
    ) -> Result<JobStatus> {
        let row = sqlx::query(
            "UPDATE jobs SET \
               status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'queued' END, \
               run_after = ?, locked_by = NULL, locked_until = NULL, last_error = ?, updated_at = ? \
             WHERE id = ? AND status = 'running' AND locked_by = ? RETURNING status",
        )
        .bind(time::encode(now + retry_after))
        .bind(error)
        .bind(time::encode(now))
        .bind(id.0)
        .bind(worker)
        .fetch_optional(&self.writer)
        .await?;
        let status: String = row.ok_or(StoreError::LeaseLost(id))?.try_get("status")?;
        parse_status(&status)
    }

    /// Gives a dead job a fresh set of attempts.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] if there is no dead job with that id.
    pub async fn requeue_dead(&self, id: JobId, now: DateTime<Utc>) -> Result<()> {
        let result = sqlx::query(
            "UPDATE jobs SET status = 'queued', attempts = 0, run_after = ?, updated_at = ? \
             WHERE id = ? AND status = 'dead'",
        )
        .bind(time::encode(now))
        .bind(time::encode(now))
        .bind(id.0)
        .execute(&self.writer)
        .await?;
        if result.rows_affected() == 0 {
            return Err(StoreError::NotFound { kind: "dead job", id: id.to_string() });
        }
        Ok(())
    }

    /// Loads a job.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::NotFound`] if there is no such job.
    pub async fn job(&self, id: JobId) -> Result<Job> {
        let row = sqlx::query(&format!("SELECT {JOB_COLUMNS} FROM jobs WHERE id = ?"))
            .bind(id.0)
            .fetch_optional(&self.reader)
            .await?
            .ok_or_else(|| StoreError::NotFound { kind: "job", id: id.to_string() })?;
        job_from_row(&row)
    }

    /// A project's jobs, oldest first, optionally only those with `status`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] or a decoding error on failure.
    pub async fn jobs(&self, project_id: ProjectId, status: Option<JobStatus>) -> Result<Vec<Job>> {
        let rows = match status {
            Some(status) => {
                sqlx::query(&format!(
                    "SELECT {JOB_COLUMNS} FROM jobs WHERE project_id = ? AND status = ? ORDER BY id"
                ))
                .bind(project_id.to_string())
                .bind(wire_name(&status)?)
                .fetch_all(&self.reader)
                .await?
            }
            None => {
                sqlx::query(&format!("SELECT {JOB_COLUMNS} FROM jobs WHERE project_id = ? ORDER BY id"))
                    .bind(project_id.to_string())
                    .fetch_all(&self.reader)
                    .await?
            }
        };
        rows.iter().map(job_from_row).collect()
    }
}

/// Inserts a queued job inside the caller's transaction.
pub(crate) async fn insert_job(
    tx: &mut Transaction<'_, Sqlite>,
    job: &NewJob,
    now: DateTime<Utc>,
) -> Result<JobId> {
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO jobs (project_id, task_id, kind, payload, status, attempts, max_attempts, \
         run_after, created_at, updated_at) VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?) RETURNING id",
    )
    .bind(job.project_id.to_string())
    .bind(job.task_id.map(|t| t.to_string()))
    .bind(&job.kind)
    .bind(serde_json::to_string(&job.payload)?)
    .bind(i64::from(job.max_attempts.max(1)))
    .bind(time::encode(job.run_after))
    .bind(time::encode(now))
    .bind(time::encode(now))
    .fetch_one(&mut **tx)
    .await?;
    Ok(JobId(id))
}

/// Cancels a project's queued jobs, and its running jobs too if
/// `include_running`. Runs inside the caller's transaction so it commits
/// together with the project's final status.
pub(crate) async fn cancel_for_project(
    tx: &mut Transaction<'_, Sqlite>,
    project_id: ProjectId,
    include_running: bool,
    now: DateTime<Utc>,
) -> Result<u64> {
    let statuses = if include_running { "('queued', 'running')" } else { "('queued')" };
    let result = sqlx::query(&format!(
        "UPDATE jobs SET status = 'cancelled', locked_by = NULL, locked_until = NULL, updated_at = ? \
         WHERE project_id = ? AND status IN {statuses}"
    ))
    .bind(time::encode(now))
    .bind(project_id.to_string())
    .execute(&mut **tx)
    .await?;
    Ok(result.rows_affected())
}

fn job_from_row(row: &SqliteRow) -> Result<Job> {
    let count =
        |value: i64| u32::try_from(value).map_err(|_| StoreError::Corrupt(format!("bad count {value}")));

    Ok(Job {
        id: JobId(row.try_get("id")?),
        project_id: parse_id(row.try_get("project_id")?)?,
        task_id: row.try_get::<Option<String>, _>("task_id")?.as_deref().map(parse_id).transpose()?,
        kind: row.try_get("kind")?,
        payload: serde_json::from_str(row.try_get("payload")?)?,
        status: parse_status(row.try_get("status")?)?,
        attempts: count(row.try_get("attempts")?)?,
        max_attempts: count(row.try_get("max_attempts")?)?,
        run_after: time::decode(row.try_get("run_after")?)?,
        locked_by: row.try_get("locked_by")?,
        locked_until: row
            .try_get::<Option<String>, _>("locked_until")?
            .as_deref()
            .map(time::decode)
            .transpose()?,
        last_error: row.try_get("last_error")?,
    })
}

fn parse_id<T>(text: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    text.parse().map_err(|e| StoreError::Corrupt(format!("bad id {text:?}: {e}")))
}

fn parse_status(text: &str) -> Result<JobStatus> {
    Ok(serde_json::from_value(serde_json::Value::String(text.to_owned()))?)
}
