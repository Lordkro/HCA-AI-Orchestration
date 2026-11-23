//! SQLite storage for HCA v2.
//!
//! One file holds everything: projects, tasks, the task history and the job
//! queue that replaces Redis Streams. Every state change runs in a single
//! transaction through a **single writer connection**, so two coroutines
//! can never interleave half-finished writes (a problem in the Python
//! version, which shared one connection with no transactions). Reads go
//! through a separate pool and see the last committed state.
//!
//! The store never decides a state change itself: it loads the
//! [`hca_core`] type, applies the event with the core's rules, and saves
//! the result together with a history row.

mod jobs;
mod projects;
mod time;

use std::path::Path;
use std::time::Duration;

use hca_core::{PlanError, ProjectStatus, ProjectTransitionError, TransitionError};
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteSynchronous,
};

pub use jobs::{Job, JobId, JobStatus, NewJob};
pub use projects::TaskEventRecord;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Everything that can go wrong in the store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// SQLite or the connection pool failed.
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    /// A schema migration failed.
    #[error("migration failed: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    /// A stored JSON document couldn't be read or written.
    #[error("stored data is invalid: {0}")]
    Json(#[from] serde_json::Error),
    /// A stored value is out of range or malformed.
    #[error("stored data is invalid: {0}")]
    Corrupt(String),
    /// The project, task or job doesn't exist.
    #[error("{kind} {id} not found")]
    NotFound {
        /// `"project"`, `"task"` or `"job"`.
        kind: &'static str,
        /// The id that was looked up.
        id: String,
    },
    /// The task's state machine refused the event.
    #[error(transparent)]
    Task(#[from] TransitionError),
    /// The project's state machine refused the event.
    #[error(transparent)]
    Project(#[from] ProjectTransitionError),
    /// The proposed plan is invalid.
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// Work was attempted on a project that isn't active.
    #[error("the project is {0:?} and accepts no work")]
    ProjectNotActive(ProjectStatus),
    /// A plan was submitted for a project that already has tasks.
    #[error("the project already has a plan")]
    PlanExists,
    /// A worker tried to finish a job it no longer holds, usually because
    /// its lease expired and another worker took the job.
    #[error("job {0} is not held by this worker")]
    LeaseLost(JobId),
}

/// Result type for store operations.
pub type Result<T, E = StoreError> = std::result::Result<T, E>;

/// Handle to the HCA database. Cheap to clone.
#[derive(Clone, Debug)]
pub struct Store {
    writer: SqlitePool,
    reader: SqlitePool,
}

impl Store {
    /// Opens (creating if needed) the database at `path` and applies any
    /// pending migrations.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Database`] if the file can't be opened, or
    /// [`StoreError::Migrate`] if a migration fails.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let options = SqliteConnectOptions::new()
            .filename(path.as_ref())
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true)
            .busy_timeout(Duration::from_secs(5));

        // One writer connection: writes are serialized by the pool, so a
        // transaction can never fail halfway with SQLITE_BUSY from another
        // writer in this process.
        let writer = SqlitePoolOptions::new().max_connections(1).connect_with(options.clone()).await?;
        MIGRATOR.run(&writer).await?;
        let reader = SqlitePoolOptions::new().max_connections(4).connect_with(options).await?;
        Ok(Self { writer, reader })
    }

    /// Closes both pools, waiting for in-flight queries to finish.
    pub async fn close(&self) {
        self.writer.close().await;
        self.reader.close().await;
    }
}

/// The JSON string form of a serializable enum, e.g. `"in_progress"`.
fn wire_name<T: serde::Serialize>(value: &T) -> Result<String> {
    match serde_json::to_value(value)? {
        serde_json::Value::String(s) => Ok(s),
        other => Err(StoreError::Corrupt(format!("expected a string tag, got {other}"))),
    }
}
