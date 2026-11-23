//! The HCA v2 orchestrator.
//!
//! Routing is plain code, not a model. A project moves through three kinds
//! of job on the durable queue:
//!
//! 1. **plan**: the PM agent turns the idea into a validated task graph.
//! 2. **work**: the task's agent (research, spec or coder) does the task
//!    and submits it.
//! 3. **review**: the critic approves it or requests changes; changes send
//!    the task back to step 2.
//!
//! After every job the orchestrator calls [`assess`] and does what it says:
//! dispatch ready tasks, retry failed ones that still have a retry, or
//! finish the project as completed or stuck. Every state change and the job
//! that follows it commit in one transaction, and every limit from
//! `hca-core` applies, so a project always ends.

mod status;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use chrono::{DateTime, Utc};
use hca_agents::{
    Agent, AgentConfig, AgentError, AgentRun, CommandRunner, Submission, Toolbox, Verdict, Workspace,
};
use hca_core::{
    assess, ready_tasks, AgentRole, FailureReason, Limits, Project, ProjectEvent, ProjectFailure, ProjectId,
    ProjectProgress, ProjectStatus, Task, TaskEvent, TaskId, TaskState, Transition,
};
use hca_llm::{ContextBuilder, Llm, LlmError, Priority, Truncate};
use hca_store::{Job, JobStatus, NewActivity, NewJob, Store, StoreError};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;

pub use status::{AgentStats, AgentStatus};

/// Job kind: plan a project.
pub const JOB_PLAN: &str = "plan";
/// Job kind: do a task.
pub const JOB_WORK: &str = "work";
/// Job kind: review a task.
pub const JOB_REVIEW: &str = "review";

/// Orchestrator settings.
#[derive(Clone, Debug)]
pub struct OrchestratorConfig {
    /// Project and task limits.
    pub limits: Limits,
    /// Limits on each agent run.
    pub agent: AgentConfig,
    /// Jobs processed at the same time.
    pub workers: usize,
    /// How long a worker holds a job before another may take it over. The
    /// lease is renewed while the agent runs.
    pub lease: chrono::Duration,
    /// How often idle workers check for jobs.
    pub poll_interval: Duration,
    /// Where project workspaces live; each project gets a folder named
    /// after its id.
    pub workspaces_dir: PathBuf,
    /// Attempts per job for retryable failures (network, rate limits).
    pub job_attempts: u32,
    /// Wait before retrying a job.
    pub retry_delay: chrono::Duration,
    /// Token budget for the brief given to each agent run.
    pub brief_tokens: u64,
}

impl OrchestratorConfig {
    /// Defaults, with workspaces under `workspaces_dir`.
    pub fn new(workspaces_dir: impl Into<PathBuf>) -> Self {
        Self {
            limits: Limits::default(),
            agent: AgentConfig::default(),
            workers: 2,
            lease: chrono::Duration::minutes(15),
            poll_interval: Duration::from_secs(1),
            workspaces_dir: workspaces_dir.into(),
            job_attempts: 3,
            retry_delay: chrono::Duration::seconds(30),
            brief_tokens: 8_000,
        }
    }
}

/// Errors from orchestrator operations.
#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    /// The store failed or refused the change.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A workspace couldn't be opened.
    #[error("workspace error: {0}")]
    Workspace(#[from] std::io::Error),
    /// The request doesn't make sense in the current state.
    #[error("{0}")]
    Conflict(String),
}

/// Progress of a project, shaped for the dashboard.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Progress {
    /// Tasks in the plan.
    pub total_tasks: usize,
    /// Tasks done.
    pub completed: usize,
    /// Done, as a percentage.
    pub progress_pct: f64,
    /// Task count per state.
    pub by_state: std::collections::BTreeMap<TaskState, usize>,
    /// Token use against the budget.
    pub token_usage: TokenUsage,
}

/// Token use against the budget.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TokenUsage {
    /// Tokens used.
    pub tokens_used: u64,
    /// The budget.
    pub budget: u64,
    /// Tokens left.
    pub remaining: u64,
    /// Used, as a percentage.
    pub pct_used: f64,
}

enum JobError {
    Retry(String),
    LeaseLost,
    Store(StoreError),
}

impl From<StoreError> for JobError {
    fn from(error: StoreError) -> Self {
        match error {
            StoreError::LeaseLost(_) => Self::LeaseLost,
            other => Self::Store(other),
        }
    }
}

impl From<std::io::Error> for JobError {
    fn from(error: std::io::Error) -> Self {
        Self::Retry(format!("workspace error: {error}"))
    }
}

enum RunOutcome {
    Done(Box<AgentRun>),
    Failed(FailureReason, String),
}

/// Runs projects. Share it with `Arc`.
#[derive(Debug)]
pub struct Orchestrator {
    store: Store,
    llm: Llm,
    runner: Option<Arc<dyn CommandRunner>>,
    config: OrchestratorConfig,
    events: broadcast::Sender<Value>,
    board: Mutex<status::StatusBoard>,
}

impl Orchestrator {
    /// An orchestrator using `store`, `llm` and, if given, a sandboxed
    /// command runner.
    pub fn new(
        store: Store,
        llm: Llm,
        runner: Option<Arc<dyn CommandRunner>>,
        config: OrchestratorConfig,
    ) -> Self {
        let (events, _) = broadcast::channel(1024);
        Self { store, llm, runner, config, events, board: Mutex::default() }
    }

    /// The store.
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The settings.
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Whether agents can run commands.
    pub fn has_sandbox(&self) -> bool {
        self.runner.is_some()
    }

    /// Live events for the dashboard, as JSON.
    pub fn subscribe(&self) -> broadcast::Receiver<Value> {
        self.events.subscribe()
    }

    /// What each role is doing.
    pub fn agent_statuses(&self) -> Vec<AgentStatus> {
        let board = self.board.lock().unwrap_or_else(PoisonError::into_inner);
        board
            .snapshot(|role| self.llm.role(role).map_or_else(|| "not configured".into(), |m| m.model.clone()))
    }

    /// A project's workspace folder.
    pub fn workspace_path(&self, project: ProjectId) -> PathBuf {
        self.config.workspaces_dir.join(project.to_string())
    }

    /// Opens a project's workspace.
    ///
    /// # Errors
    ///
    /// Returns the I/O error if the folder can't be created.
    pub fn workspace(&self, project: ProjectId) -> std::io::Result<Workspace> {
        Workspace::open(self.workspace_path(project))
    }

    /// A project's progress.
    ///
    /// # Errors
    ///
    /// Returns [`OrchestratorError::Store`] if the project can't be loaded.
    pub async fn progress(&self, project_id: ProjectId) -> Result<Progress, OrchestratorError> {
        let project = self.store.project(project_id).await?;
        let tasks = self.store.tasks(project_id).await?;
        let mut by_state = std::collections::BTreeMap::new();
        for task in &tasks {
            *by_state.entry(task.state()).or_insert(0) += 1;
        }
        let completed = by_state.get(&TaskState::Done).copied().unwrap_or(0);
        let budget = project.budget(&self.config.limits);
        Ok(Progress {
            total_tasks: tasks.len(),
            completed,
            progress_pct: percent(count(completed), count(tasks.len())),
            by_state,
            token_usage: TokenUsage {
                tokens_used: budget.used(),
                budget: budget.limit(),
                remaining: budget.remaining(),
                pct_used: percent(budget.used(), budget.limit()),
            },
        })
    }

    // ------------------------------------------------------------------
    // Human actions
    // ------------------------------------------------------------------

    /// Creates a project and queues its planning job.
    ///
    /// # Errors
    ///
    /// Returns [`OrchestratorError::Conflict`] for a blank idea, or a store
    /// error.
    pub async fn create_project(&self, name: &str, idea: &str) -> Result<Project, OrchestratorError> {
        if idea.trim().is_empty() {
            return Err(OrchestratorError::Conflict("the idea is empty".into()));
        }
        let now = Utc::now();
        let name = if name.trim().is_empty() { "New project" } else { name.trim() };
        let project = Project::new(name, idea.trim(), now);
        self.store.create_project(&project).await?;
        self.store.enqueue(&self.job(project.id, None, JOB_PLAN, json!({}), now), now).await?;
        self.log(project.id, None, "user", "pm", "system", idea.trim()).await;
        self.emit_project(&project);
        Ok(project)
    }

    /// Pauses an active project. Queued work waits.
    ///
    /// # Errors
    ///
    /// Returns a store error if the project isn't active.
    pub async fn pause(&self, id: ProjectId) -> Result<ProjectStatus, OrchestratorError> {
        let status = self.store.apply_project_event(id, ProjectEvent::Pause, Utc::now()).await?;
        self.after_project_change(id, "Project paused").await;
        Ok(status)
    }

    /// Resumes a paused project and dispatches whatever is ready.
    ///
    /// # Errors
    ///
    /// Returns a store error if the project isn't paused.
    pub async fn resume(&self, id: ProjectId) -> Result<ProjectStatus, OrchestratorError> {
        let status = self.store.apply_project_event(id, ProjectEvent::Resume, Utc::now()).await?;
        self.after_project_change(id, "Project resumed").await;
        self.advance(id).await?;
        Ok(status)
    }

    /// Cancels a project and its queued work.
    ///
    /// # Errors
    ///
    /// Returns a store error if the project already finished.
    pub async fn cancel(&self, id: ProjectId) -> Result<ProjectStatus, OrchestratorError> {
        let status = self.store.apply_project_event(id, ProjectEvent::Cancel, Utc::now()).await?;
        self.after_project_change(id, "Project cancelled").await;
        Ok(status)
    }

    /// Retries a failed task, if it has a retry left.
    ///
    /// # Errors
    ///
    /// Returns a store error if the task can't be retried.
    pub async fn retry_task(&self, id: TaskId) -> Result<(), OrchestratorError> {
        let transition =
            self.store.apply_task_event(id, TaskEvent::Retry, &self.config.limits, Utc::now()).await?;
        let task = self.store.task(id).await?;
        self.emit_task(&task, transition);
        self.log(
            task.project_id,
            Some(id),
            "user",
            "pm",
            "status_update",
            &format!("Retry requested for '{}'", task.title),
        )
        .await;
        self.advance(task.project_id).await?;
        Ok(())
    }

    /// A person's verdict on a task in review, overriding the critic.
    ///
    /// # Errors
    ///
    /// Returns [`OrchestratorError::Conflict`] if changes are requested
    /// without feedback, or a store error if the task isn't in review.
    pub async fn human_review(
        &self,
        id: TaskId,
        approve: bool,
        feedback: &str,
    ) -> Result<(), OrchestratorError> {
        let now = Utc::now();
        let task = self.store.task(id).await?;
        if approve {
            let transition =
                self.store.apply_task_event(id, TaskEvent::Approve, &self.config.limits, now).await?;
            self.emit_task(&task, transition);
            self.log(
                task.project_id,
                Some(id),
                "user",
                &role_name(&task),
                "feedback",
                &format!("Approved by a person. {feedback}"),
            )
            .await;
        } else {
            if feedback.trim().is_empty() {
                return Err(OrchestratorError::Conflict("feedback is required to request changes".into()));
            }
            let event = TaskEvent::RequestRevision { feedback: feedback.trim().to_owned() };
            let (transition, _) = self
                .store
                .apply_task_event_with_job(id, event, &self.config.limits, now, |t, task| {
                    self.revision_job(*t, task, now)
                })
                .await?;
            self.emit_task(&task, transition);
            self.log(task.project_id, Some(id), "user", &role_name(&task), "feedback", feedback.trim()).await;
        }
        self.advance(task.project_id).await?;
        Ok(())
    }

    /// Adds a person's note to a task. It is included in the brief the next
    /// time an agent works on the task.
    ///
    /// # Errors
    ///
    /// Returns a store error if the task doesn't exist.
    pub async fn add_feedback(&self, id: TaskId, content: &str) -> Result<(), OrchestratorError> {
        if content.trim().is_empty() {
            return Err(OrchestratorError::Conflict("the feedback is empty".into()));
        }
        let task = self.store.task(id).await?;
        self.log(task.project_id, Some(id), "user", &role_name(&task), "user_feedback", content.trim()).await;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Running
    // ------------------------------------------------------------------

    /// Starts `config.workers` workers that process jobs until `shutdown`
    /// becomes `true`.
    pub fn spawn_workers(self: &Arc<Self>, shutdown: &watch::Receiver<bool>) -> Vec<JoinHandle<()>> {
        (0..self.config.workers.max(1))
            .map(|i| {
                let this = Arc::clone(self);
                let mut shutdown = shutdown.clone();
                tokio::spawn(async move {
                    let name = format!("worker-{i}-{}", std::process::id());
                    while !*shutdown.borrow() {
                        match this.step(&name).await {
                            Ok(true) => {}
                            Ok(false) => {
                                tokio::select! {
                                    () = tokio::time::sleep(this.config.poll_interval) => {}
                                    _ = shutdown.changed() => {}
                                }
                            }
                            Err(error) => {
                                tracing::warn!(%error, worker = %name, "job processing failed");
                                tokio::time::sleep(this.config.poll_interval).await;
                            }
                        }
                    }
                })
            })
            .collect()
    }

    /// Claims and processes one job. Returns `false` if there was none.
    ///
    /// # Errors
    ///
    /// Returns a store error if the job couldn't be claimed or settled.
    pub async fn step(&self, worker: &str) -> Result<bool, OrchestratorError> {
        let now = Utc::now();
        let Some(job) = self.store.claim(worker, &[], self.config.lease, now).await? else {
            return Ok(false);
        };
        tracing::info!(job = %job.id, kind = %job.kind, project = %job.project_id, "job claimed");
        match self.with_lease(&job, worker).await {
            Ok(()) => match self.store.complete(job.id, worker, Utc::now()).await {
                Ok(()) | Err(StoreError::LeaseLost(_)) => {}
                Err(other) => return Err(other.into()),
            },
            Err(JobError::LeaseLost) => {
                tracing::info!(job = %job.id, "job dropped: lease lost or project finished");
            }
            Err(JobError::Retry(message)) => self.retry_or_bury(&job, worker, &message).await?,
            Err(JobError::Store(error)) => self.retry_or_bury(&job, worker, &error.to_string()).await?,
        }
        Ok(true)
    }

    async fn retry_or_bury(&self, job: &Job, worker: &str, message: &str) -> Result<(), OrchestratorError> {
        tracing::warn!(job = %job.id, %message, "job failed");
        match self.store.fail(job.id, worker, message, self.config.retry_delay, Utc::now()).await {
            Ok(JobStatus::Dead) => self.job_dead(job, message).await,
            Ok(_) | Err(StoreError::LeaseLost(_)) => Ok(()),
            Err(other) => Err(other.into()),
        }
    }

    /// Processes jobs until none is left. For tests and one-shot runs.
    ///
    /// # Errors
    ///
    /// Returns the first error from [`step`](Self::step).
    pub async fn run_until_idle(&self, worker: &str) -> Result<usize, OrchestratorError> {
        let mut processed = 0;
        while self.step(worker).await? {
            processed += 1;
        }
        Ok(processed)
    }

    /// Repairs state after a restart: queues planning for projects without
    /// a plan, queues work for in-flight tasks that lost their job, and
    /// dispatches what's ready.
    ///
    /// # Errors
    ///
    /// Returns a store error.
    pub async fn recover(&self) -> Result<(), OrchestratorError> {
        let now = Utc::now();
        for project in self.store.projects(Some(ProjectStatus::Active)).await? {
            let open: Vec<Job> = self
                .store
                .jobs(project.id, None)
                .await?
                .into_iter()
                .filter(|j| matches!(j.status, JobStatus::Queued | JobStatus::Running))
                .collect();
            let tasks = self.store.tasks(project.id).await?;
            if tasks.is_empty() && !open.iter().any(|j| j.kind == JOB_PLAN) {
                self.store.enqueue(&self.job(project.id, None, JOB_PLAN, json!({}), now), now).await?;
            }
            for task in tasks.iter().filter(|t| t.state().is_in_flight()) {
                if open.iter().any(|j| j.task_id == Some(task.id)) {
                    continue;
                }
                let job = if task.state() == TaskState::Review {
                    let summary = self
                        .store
                        .latest_activity(task.id, "deliverable")
                        .await?
                        .map_or_else(|| "(summary lost in a restart)".into(), |a| a.entry.content);
                    self.job(
                        project.id,
                        Some(task.id),
                        JOB_REVIEW,
                        json!({ "summary": summary, "files": [] }),
                        now,
                    )
                } else {
                    self.job(project.id, Some(task.id), JOB_WORK, json!({}), now)
                };
                self.store.enqueue(&job, now).await?;
            }
            self.advance(project.id).await?;
        }
        Ok(())
    }

    async fn with_lease(&self, job: &Job, worker: &str) -> Result<(), JobError> {
        let lease = self.config.lease;
        let every = (lease / 3).to_std().unwrap_or(Duration::from_mins(1)).max(Duration::from_millis(100));
        let heartbeat = async {
            let mut tick = tokio::time::interval(every);
            tick.tick().await;
            loop {
                tick.tick().await;
                if let Err(StoreError::LeaseLost(_)) =
                    self.store.extend_lease(job.id, worker, lease, Utc::now()).await
                {
                    return;
                }
            }
        };
        tokio::select! {
            result = self.handle(job) => result,
            () = heartbeat => Err(JobError::LeaseLost),
        }
    }

    async fn handle(&self, job: &Job) -> Result<(), JobError> {
        match job.kind.as_str() {
            JOB_PLAN => self.plan(job.project_id).await,
            JOB_WORK => match job.task_id {
                Some(task) => self.work(task).await,
                None => Ok(()),
            },
            JOB_REVIEW => match job.task_id {
                Some(task) => self.review(task, &job.payload).await,
                None => Ok(()),
            },
            other => {
                tracing::warn!(kind = other, "unknown job kind, dropping it");
                Ok(())
            }
        }
    }

    async fn job_dead(&self, job: &Job, message: &str) -> Result<(), OrchestratorError> {
        let now = Utc::now();
        if let Some(task_id) = job.task_id {
            let reason =
                FailureReason::AgentError(format!("gave up after {} attempts: {message}", job.attempts));
            if let Ok(transition) = self
                .store
                .apply_task_event(task_id, TaskEvent::Fail { reason }, &self.config.limits, now)
                .await
            {
                let task = self.store.task(task_id).await?;
                self.emit_task(&task, transition);
                let note = format!("'{}' failed: {message}", task.title);
                self.log(job.project_id, Some(task_id), "system", "pm", "status_update", &note).await;
            }
        } else {
            let reason = ProjectFailure::InvalidPlan(format!("planning failed: {message}"));
            if self
                .store
                .apply_project_event(job.project_id, ProjectEvent::Fail { reason }, now)
                .await
                .is_ok()
            {
                self.after_project_change(job.project_id, &format!("Planning failed: {message}")).await;
            }
        }
        self.advance(job.project_id).await
    }

    // ------------------------------------------------------------------
    // Jobs
    // ------------------------------------------------------------------

    async fn plan(&self, project_id: ProjectId) -> Result<(), JobError> {
        let project = self.store.project(project_id).await?;
        if !project.accepts_work() {
            return Ok(());
        }
        if !self.store.tasks(project_id).await?.is_empty() {
            self.advance(project_id).await.map_err(|e| JobError::Retry(e.to_string()))?;
            return Ok(());
        }
        let workspace = self.workspace(project_id)?;
        let files = workspace.list(".").await.unwrap_or_default();
        let brief = ContextBuilder::new(self.config.brief_tokens)
            .required("Product idea", &project.idea)
            .section("Project name", &project.name, Priority::High, Truncate::Never)
            .section("Existing files", listing(&files), Priority::Normal, Truncate::KeepStart)
            .build()
            .text;

        let outcome =
            self.run_agent(AgentRole::Pm, &project, "Planning the project".into(), workspace, brief).await?;
        if !self.store.project(project_id).await?.accepts_work() {
            return Ok(());
        }
        let now = Utc::now();
        let failure = match outcome {
            RunOutcome::Done(run) => match run.submission {
                Submission::Plan(plan) => match plan.into_tasks(project_id, &self.config.limits, now) {
                    Ok(tasks) => {
                        self.store.create_plan(project_id, &tasks, &self.config.limits).await?;
                        let lines: Vec<String> = tasks
                            .iter()
                            .enumerate()
                            .map(|(i, t)| {
                                format!(
                                    "{}. [{}] {}",
                                    i + 1,
                                    t.role.map_or("coder", AgentRole::as_str),
                                    t.title
                                )
                            })
                            .collect();
                        self.log(
                            project_id,
                            None,
                            "pm",
                            "team",
                            "system",
                            &format!("Plan:\n{}", lines.join("\n")),
                        )
                        .await;
                        self.emit_project(&self.store.project(project_id).await?);
                        self.advance(project_id).await.map_err(|e| JobError::Retry(e.to_string()))?;
                        return Ok(());
                    }
                    Err(message) => message,
                },
                other => format!("the planner submitted {other:?} instead of a plan"),
            },
            RunOutcome::Failed(_, message) => message,
        };
        let reason = ProjectFailure::InvalidPlan(failure.clone());
        if self.store.apply_project_event(project_id, ProjectEvent::Fail { reason }, now).await.is_ok() {
            self.after_project_change(project_id, &format!("Planning failed: {failure}")).await;
        }
        Ok(())
    }

    async fn work(&self, task_id: TaskId) -> Result<(), JobError> {
        let task = self.store.task(task_id).await?;
        let project = self.store.project(task.project_id).await?;
        if !project.accepts_work() {
            return Ok(());
        }
        match task.state() {
            TaskState::Assigned | TaskState::Revision => {
                match self
                    .store
                    .apply_task_event(task_id, TaskEvent::Start, &self.config.limits, Utc::now())
                    .await
                {
                    Ok(transition) => self.emit_task(&task, transition),
                    Err(StoreError::Task(_) | StoreError::ProjectNotActive(_)) => return Ok(()),
                    Err(other) => return Err(other.into()),
                }
            }
            // A previous attempt crashed mid-run: carry on.
            TaskState::InProgress => {}
            _ => return Ok(()),
        }
        let role = task.assignee().or(task.role).unwrap_or(AgentRole::Coder);
        let workspace = self.workspace(project.id)?;
        let brief = self.work_brief(&project, &task, &workspace).await?;

        let outcome =
            self.run_agent(role, &project, format!("Working on '{}'", task.title), workspace, brief).await?;
        if !self.store.project(project.id).await?.accepts_work() {
            return Ok(());
        }
        let now = Utc::now();
        match outcome {
            RunOutcome::Done(run) => match run.submission {
                Submission::Work { summary } => {
                    let files = run.files_written.clone();
                    let payload = json!({ "summary": summary, "files": files });
                    let (transition, _) = self
                        .store
                        .apply_task_event_with_job(
                            task_id,
                            TaskEvent::Submit,
                            &self.config.limits,
                            now,
                            |_, t| Some(self.job(t.project_id, Some(t.id), JOB_REVIEW, payload, now)),
                        )
                        .await?;
                    self.emit_task(&task, transition);
                    let files_line = if files.is_empty() {
                        String::new()
                    } else {
                        format!("\n\nFiles: {}", files.join(", "))
                    };
                    self.log(
                        project.id,
                        Some(task_id),
                        role.as_str(),
                        "critic",
                        "deliverable",
                        &format!("{summary}{files_line}"),
                    )
                    .await;
                }
                other => {
                    self.fail_task(
                        &task,
                        FailureReason::AgentError(format!("submitted {other:?} instead of work")),
                        now,
                    )
                    .await?;
                }
            },
            RunOutcome::Failed(reason, _) => self.fail_task(&task, reason, now).await?,
        }
        self.advance(project.id).await.map_err(|e| JobError::Retry(e.to_string()))
    }

    async fn review(&self, task_id: TaskId, payload: &Value) -> Result<(), JobError> {
        let task = self.store.task(task_id).await?;
        let project = self.store.project(task.project_id).await?;
        if !project.accepts_work() || task.state() != TaskState::Review {
            return Ok(());
        }
        let summary = payload.get("summary").and_then(Value::as_str).unwrap_or("(no summary)");
        let files: Vec<String> = payload
            .get("files")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
            .unwrap_or_default();

        let brief = self.review_brief(&task, summary, &files);
        let workspace = self.workspace(project.id)?;
        let label = format!("Reviewing '{}'", task.title);
        let outcome = self.run_agent(AgentRole::Critic, &project, label, workspace, brief).await?;
        if !self.store.project(project.id).await?.accepts_work() {
            return Ok(());
        }
        let now = Utc::now();
        let owner = role_name(&task);
        match outcome {
            RunOutcome::Done(run) => match run.submission {
                Submission::Review(review) if review.verdict == Verdict::Approve => {
                    let transition = self
                        .store
                        .apply_task_event(task_id, TaskEvent::Approve, &self.config.limits, now)
                        .await?;
                    self.emit_task(&task, transition);
                    let note = if review.feedback.trim().is_empty() {
                        "Approved.".to_owned()
                    } else {
                        format!("Approved. {}", review.feedback.trim())
                    };
                    self.log(project.id, Some(task_id), "critic", &owner, "feedback", &note).await;
                }
                Submission::Review(review) => {
                    let event = TaskEvent::RequestRevision { feedback: review.feedback.clone() };
                    let (transition, _) = self
                        .store
                        .apply_task_event_with_job(task_id, event, &self.config.limits, now, |t, task| {
                            self.revision_job(*t, task, now)
                        })
                        .await?;
                    self.emit_task(&task, transition);
                    self.log(project.id, Some(task_id), "critic", &owner, "feedback", &review.feedback).await;
                    if transition.to == TaskState::Failed {
                        self.log(
                            project.id,
                            Some(task_id),
                            "system",
                            "pm",
                            "status_update",
                            &format!("'{}' failed: revision limit reached", task.title),
                        )
                        .await;
                    }
                }
                other => {
                    let reason = FailureReason::AgentError(format!(
                        "the reviewer submitted {other:?} instead of a review"
                    ));
                    self.fail_task(&task, reason, now).await?;
                }
            },
            RunOutcome::Failed(reason, message) => {
                let reason = match reason {
                    FailureReason::AgentError(_) => {
                        FailureReason::AgentError(format!("review failed: {message}"))
                    }
                    other => other,
                };
                self.fail_task(&task, reason, now).await?;
            }
        }
        self.advance(project.id).await.map_err(|e| JobError::Retry(e.to_string()))
    }

    /// Acts on [`assess`] until there's nothing more to do right now.
    async fn advance(&self, project_id: ProjectId) -> Result<(), OrchestratorError> {
        let limits = &self.config.limits;
        // Each pass either finishes, dispatches, or uses up retries, so this
        // bound is never reached in practice; it only guards against bugs.
        for _ in 0..(limits.max_tasks * 2 + 4) {
            let project = self.store.project(project_id).await?;
            if project.status() != ProjectStatus::Active {
                return Ok(());
            }
            let tasks = self.store.tasks(project_id).await?;
            let now = Utc::now();
            match assess(&tasks, limits) {
                ProjectProgress::AwaitingPlan | ProjectProgress::Running { ready: 0, .. } => return Ok(()),
                ProjectProgress::Completed => {
                    if self.store.apply_project_event(project_id, ProjectEvent::Complete, now).await.is_ok() {
                        self.after_project_change(project_id, "All tasks are done. Project complete.").await;
                    }
                    return Ok(());
                }
                ProjectProgress::Stuck { exhausted } => {
                    let titles: Vec<String> = tasks
                        .iter()
                        .filter(|t| exhausted.contains(&t.id))
                        .map(|t| format!("'{}'", t.title))
                        .collect();
                    let reason = ProjectFailure::Stuck;
                    if self
                        .store
                        .apply_project_event(project_id, ProjectEvent::Fail { reason }, now)
                        .await
                        .is_ok()
                    {
                        self.after_project_change(
                            project_id,
                            &format!("Project stopped: {} failed with no retries left.", titles.join(", ")),
                        )
                        .await;
                    }
                    return Ok(());
                }
                ProjectProgress::NeedsRetry { retriable } => {
                    for id in retriable {
                        if let Ok(transition) =
                            self.store.apply_task_event(id, TaskEvent::Retry, limits, now).await
                        {
                            let task = self.store.task(id).await?;
                            self.emit_task(&task, transition);
                            self.log(
                                project_id,
                                Some(id),
                                "system",
                                "pm",
                                "status_update",
                                &format!("Retrying '{}'", task.title),
                            )
                            .await;
                        }
                    }
                }
                ProjectProgress::Running { .. } => {
                    for id in ready_tasks(&tasks, limits) {
                        let Some(task) = tasks.iter().find(|t| t.id == id) else { continue };
                        let role = task.role.unwrap_or(AgentRole::Coder);
                        let result = self
                            .store
                            .apply_task_event_with_job(
                                id,
                                TaskEvent::Assign { to: role },
                                limits,
                                now,
                                |_, t| Some(self.job(t.project_id, Some(t.id), JOB_WORK, json!({}), now)),
                            )
                            .await;
                        match result {
                            Ok((transition, _)) => {
                                self.emit_task(task, transition);
                                self.log(
                                    project_id,
                                    Some(id),
                                    "pm",
                                    role.as_str(),
                                    "task_assignment",
                                    &task.title,
                                )
                                .await;
                            }
                            // Another worker got there first, or the project
                            // was paused in between.
                            Err(StoreError::Task(_) | StoreError::ProjectNotActive(_)) => {}
                            Err(other) => return Err(other.into()),
                        }
                    }
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    async fn run_agent(
        &self,
        role: AgentRole,
        project: &Project,
        label: String,
        workspace: Workspace,
        brief: String,
    ) -> Result<RunOutcome, JobError> {
        let toolbox = Toolbox::new(workspace, self.runner.clone(), self.config.limits);
        let agent = Agent::new(&self.llm, &toolbox, self.config.agent);
        let mut tokens = 0u64;
        let mut calls = 0u64;
        let run_id = self.board.lock().unwrap_or_else(PoisonError::into_inner).start(role, label);
        let result = agent
            .run(role, brief, project.budget(&self.config.limits), |usage| {
                tokens += usage.total();
                calls += 1;
            })
            .await;
        self.board.lock().unwrap_or_else(PoisonError::into_inner).finish(role, run_id, calls, tokens);

        if tokens > 0 {
            let within =
                self.store.charge_tokens(project.id, tokens, &self.config.limits, Utc::now()).await?;
            if !within {
                self.after_project_change(project.id, "Token budget spent. Project stopped.").await;
            }
        }
        match result {
            Ok(run) => Ok(RunOutcome::Done(Box::new(run))),
            Err(AgentError::Llm(LlmError::Budget(e))) => {
                Ok(RunOutcome::Failed(FailureReason::BudgetExhausted, e.to_string()))
            }
            Err(AgentError::Llm(e)) if e.is_retryable() => Err(JobError::Retry(e.to_string())),
            Err(e) => Ok(RunOutcome::Failed(FailureReason::AgentError(e.to_string()), e.to_string())),
        }
    }

    fn review_brief(&self, task: &Task, summary: &str, files: &[String]) -> String {
        let mut brief = ContextBuilder::new(self.config.brief_tokens)
            .required("Task under review", format!("{}\n\n{}", task.title, task.description))
            .required("What the agent says it did", summary)
            .section("Files written or changed", listing(files), Priority::High, Truncate::KeepStart);
        if let Some(previous) = task.last_feedback() {
            brief = brief.section(
                "Your previous feedback (check that it was addressed)",
                previous,
                Priority::High,
                Truncate::KeepStart,
            );
        }
        if !self.has_sandbox() {
            brief = brief.section(
                "Note",
                "Commands can't be run in this setup, so judge by reading the code and tests.",
                Priority::High,
                Truncate::Never,
            );
        }
        brief.build().text
    }

    async fn work_brief(
        &self,
        project: &Project,
        task: &Task,
        workspace: &Workspace,
    ) -> Result<String, JobError> {
        let mut brief = ContextBuilder::new(self.config.brief_tokens)
            .required("Your task", format!("{}\n\n{}", task.title, task.description));
        if let Some(feedback) = task.last_feedback() {
            brief = brief.required("Reviewer feedback you must address", feedback);
        }
        let notes: Vec<String> = self
            .store
            .activity(project.id)
            .await?
            .into_iter()
            .filter(|a| a.entry.task_id == Some(task.id) && a.entry.kind == "user_feedback")
            .map(|a| a.entry.content)
            .collect();
        if !notes.is_empty() {
            brief = brief.required("Notes from a person on the team", notes.join("\n\n"));
        }
        brief = brief.section("Project goal", &project.idea, Priority::High, Truncate::KeepStart);
        for dependency in &task.depends_on {
            let dep = self.store.task(*dependency).await?;
            if let Some(output) = self.store.latest_activity(dep.id, "deliverable").await? {
                brief = brief.section(
                    format!("Output of the earlier task '{}'", dep.title),
                    output.entry.content,
                    Priority::High,
                    Truncate::KeepStart,
                );
            }
        }
        let files = workspace.list(".").await.unwrap_or_default();
        brief = brief.section("Project files", listing(&files), Priority::Normal, Truncate::KeepStart);
        if !self.has_sandbox() && matches!(task.role, Some(AgentRole::Coder) | None) {
            brief = brief.section(
                "Note",
                "Commands can't be run in this setup. Write the tests anyway, and say in your summary that they weren't run.",
                Priority::High,
                Truncate::Never,
            );
        }
        Ok(brief.build().text)
    }

    async fn fail_task(
        &self,
        task: &Task,
        reason: FailureReason,
        now: DateTime<Utc>,
    ) -> Result<(), JobError> {
        let message = match &reason {
            FailureReason::AgentError(m) => m.clone(),
            other => format!("{other:?}"),
        };
        match self.store.apply_task_event(task.id, TaskEvent::Fail { reason }, &self.config.limits, now).await
        {
            Ok(transition) => {
                self.emit_task(task, transition);
                self.log(
                    task.project_id,
                    Some(task.id),
                    &role_name(task),
                    "pm",
                    "status_update",
                    &format!("'{}' failed: {message}", task.title),
                )
                .await;
                Ok(())
            }
            Err(StoreError::Task(_)) => Ok(()),
            Err(other) => Err(other.into()),
        }
    }

    fn revision_job(&self, transition: Transition, task: &Task, now: DateTime<Utc>) -> Option<NewJob> {
        (transition.to == TaskState::Revision)
            .then(|| self.job(task.project_id, Some(task.id), JOB_WORK, json!({}), now))
    }

    fn job(
        &self,
        project: ProjectId,
        task: Option<TaskId>,
        kind: &str,
        payload: Value,
        now: DateTime<Utc>,
    ) -> NewJob {
        let job = NewJob::new(project, kind, payload, now).max_attempts(self.config.job_attempts);
        match task {
            Some(task) => job.for_task(task),
            None => job,
        }
    }

    async fn after_project_change(&self, id: ProjectId, message: &str) {
        self.log(id, None, "system", "user", "status_update", message).await;
        if let Ok(project) = self.store.project(id).await {
            self.emit_project(&project);
        }
    }

    async fn log(
        &self,
        project: ProjectId,
        task: Option<TaskId>,
        sender: &str,
        recipient: &str,
        kind: &str,
        content: &str,
    ) {
        let at = Utc::now();
        let entry = NewActivity {
            project_id: project,
            task_id: task,
            sender: sender.into(),
            recipient: recipient.into(),
            kind: kind.into(),
            content: content.into(),
            at,
        };
        if let Err(error) = self.store.log_activity(&entry).await {
            tracing::warn!(%error, "could not record activity");
        }
        let _ = self.events.send(json!({
            "sender": sender,
            "recipient": recipient,
            "type": kind,
            "project_id": project,
            "task_id": task,
            "payload": { "content": content },
            "timestamp": at,
        }));
    }

    fn emit_task(&self, task: &Task, transition: Transition) {
        let _ = self.events.send(json!({
            "type": "task_state_changed",
            "data": {
                "task_id": task.id,
                "project_id": task.project_id,
                "old_state": transition.from,
                "new_state": transition.to,
            },
            "timestamp": Utc::now(),
        }));
    }

    fn emit_project(&self, project: &Project) {
        let _ = self.events.send(json!({
            "type": "project_status_changed",
            "data": { "project_id": project.id, "new_status": project.status() },
            "timestamp": Utc::now(),
        }));
    }
}

fn role_name(task: &Task) -> String {
    task.assignee().or(task.role).unwrap_or(AgentRole::Coder).as_str().to_owned()
}

fn listing(files: &[String]) -> String {
    if files.is_empty() {
        "(none)".into()
    } else {
        files.join("\n")
    }
}

fn count(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// `part` as a percentage of `whole`, to one decimal. Precision loss above
/// 2^52 doesn't matter for a progress display.
#[allow(clippy::cast_precision_loss)]
fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        (part as f64 / whole as f64 * 1000.0).round() / 10.0
    }
}
