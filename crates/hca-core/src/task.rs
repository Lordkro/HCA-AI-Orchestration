//! The task state machine.
//!
//! ```text
//!  Pending ──Assign──▶ Assigned ──Start──▶ InProgress ──Submit──▶ Review ──Approve──▶ Done
//!     ▲                                ▲                            │
//!     │                                └──Start── Revision ◀────────┤ RequestRevision
//!     │                                                             │ (cap reached)
//!     └────────────Retry (bounded)──────────── Failed ◀─────────────┘
//!                                                ▲
//!                         any working state ─Fail┘
//! ```
//!
//! State only changes through [`Task::apply`]. Counters are private, so the
//! revision and retry limits can't be skipped by a caller that forgets to
//! increment them, which is how the Python version looped forever.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::{AgentRole, Limits, ProjectId, TaskId};

/// Where a task is in its lifecycle.
///
/// Serialized in `snake_case` to match the existing dashboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    /// Waiting for its dependencies or for a free slot.
    Pending,
    /// Given to an agent, not started yet.
    Assigned,
    /// An agent is working on it.
    InProgress,
    /// Submitted and waiting for a verdict.
    Review,
    /// Sent back with feedback, waiting for the agent to start again.
    Revision,
    /// Approved. Terminal.
    Done,
    /// Failed. Terminal unless a retry is still allowed.
    Failed,
}

impl TaskState {
    /// `Done` or `Failed`.
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Failed)
    }

    /// Assigned to an agent and not yet finished. These count against
    /// [`Limits::max_parallel_tasks`] and can time out.
    pub const fn is_in_flight(self) -> bool {
        matches!(self, Self::Assigned | Self::InProgress | Self::Review | Self::Revision)
    }
}

/// Why a task failed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum FailureReason {
    /// The reviewer asked for more revisions than [`Limits::max_revisions`].
    RevisionLimit,
    /// The task sat in a working state longer than the timeout.
    Timeout,
    /// The project's token budget ran out.
    BudgetExhausted,
    /// The agent or one of its tools failed.
    AgentError(String),
    /// A person or the orchestrator cancelled it.
    Cancelled,
}

/// Something that happens to a task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TaskEvent {
    /// Give the task to an agent. Only from `Pending`.
    Assign {
        /// The agent role that will own the task.
        to: AgentRole,
    },
    /// The agent starts work. From `Assigned` or `Revision`.
    Start,
    /// The agent hands in its work. From `InProgress`.
    Submit,
    /// The reviewer accepts the work. From `Review`.
    Approve,
    /// The reviewer sends the work back. From `Review`. Feedback is required.
    RequestRevision {
        /// What has to change. Must not be blank.
        feedback: String,
    },
    /// Stop the task. From any state except `Done` and `Failed`.
    Fail {
        /// Why it failed.
        reason: FailureReason,
    },
    /// Try a failed task again from the start. Only while retries remain.
    Retry,
}

impl TaskEvent {
    /// A short name for logs and error messages.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Assign { .. } => "assign",
            Self::Start => "start",
            Self::Submit => "submit",
            Self::Approve => "approve",
            Self::RequestRevision { .. } => "request_revision",
            Self::Fail { .. } => "fail",
            Self::Retry => "retry",
        }
    }
}

/// A successful state change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    /// The state before the event.
    pub from: TaskState,
    /// The state after the event.
    pub to: TaskState,
}

/// Why an event was refused. A refused event changes nothing.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TransitionError {
    /// The event isn't allowed in the task's current state.
    #[error("cannot {event} a task that is {from:?}")]
    InvalidTransition {
        /// The task's state.
        from: TaskState,
        /// The refused event's name.
        event: &'static str,
    },
    /// The role can't own tasks (the PM and the critic).
    #[error("{0} cannot own tasks")]
    RoleCannotOwnTasks(AgentRole),
    /// A revision request had no feedback, so the agent would have nothing to fix.
    #[error("a revision request needs feedback")]
    EmptyFeedback,
    /// The task has used all of its retries.
    #[error("no retries left ({used} of {max} used)")]
    RetriesExhausted {
        /// Retries already used.
        used: u32,
        /// The limit.
        max: u32,
    },
}

/// A unit of work owned by one agent role.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    /// The task's id.
    pub id: TaskId,
    /// The project it belongs to.
    pub project_id: ProjectId,
    /// Short title.
    pub title: String,
    /// What has to be done.
    pub description: String,
    /// Tasks that must be `Done` before this one can start.
    pub depends_on: Vec<TaskId>,
    /// The role the planner intended the task for. The orchestrator assigns
    /// it to this role unless a person steers it elsewhere.
    #[serde(default)]
    pub role: Option<AgentRole>,
    assignee: Option<AgentRole>,
    state: TaskState,
    revisions: u32,
    retries: u32,
    last_feedback: Option<String>,
    failure: Option<FailureReason>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl Task {
    /// Creates a pending task.
    pub fn new(
        project_id: ProjectId,
        title: impl Into<String>,
        description: impl Into<String>,
        depends_on: Vec<TaskId>,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            id: TaskId::new(),
            project_id,
            title: title.into(),
            description: description.into(),
            depends_on,
            role: None,
            assignee: None,
            state: TaskState::Pending,
            revisions: 0,
            retries: 0,
            last_feedback: None,
            failure: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Sets the planned role.
    #[must_use]
    pub fn with_role(mut self, role: AgentRole) -> Self {
        self.role = Some(role);
        self
    }

    /// The current state.
    pub const fn state(&self) -> TaskState {
        self.state
    }

    /// The role that owns the task, once assigned.
    pub const fn assignee(&self) -> Option<AgentRole> {
        self.assignee
    }

    /// Revision rounds used in the current attempt.
    pub const fn revisions(&self) -> u32 {
        self.revisions
    }

    /// Retries used so far.
    pub const fn retries(&self) -> u32 {
        self.retries
    }

    /// The most recent reviewer feedback. Kept across retries so the next
    /// attempt knows what went wrong.
    pub fn last_feedback(&self) -> Option<&str> {
        self.last_feedback.as_deref()
    }

    /// Why the task failed, while it is `Failed`.
    pub const fn failure(&self) -> Option<&FailureReason> {
        self.failure.as_ref()
    }

    /// When the task was created.
    pub const fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// When the task last changed state.
    pub const fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    /// Whether a failed task may still be retried.
    pub fn can_retry(&self, limits: &Limits) -> bool {
        self.state == TaskState::Failed
            && self.retries < limits.max_retries
            && self.failure != Some(FailureReason::Cancelled)
    }

    /// Whether an in-flight task has been in its state for longer than the
    /// timeout. Pending tasks never time out: waiting for dependencies is
    /// not being stuck.
    pub fn is_timed_out(&self, limits: &Limits, now: DateTime<Utc>) -> bool {
        self.state.is_in_flight()
            && now - self.updated_at > Duration::minutes(i64::from(limits.task_timeout_minutes))
    }

    /// Applies an event, changing state only if the event is allowed.
    ///
    /// # Errors
    ///
    /// Returns a [`TransitionError`] if the event isn't allowed in the current
    /// state, if a revision request has blank feedback, if a role that can't
    /// own tasks is assigned, or if no retries remain. The task is left
    /// unchanged in every error case.
    pub fn apply(
        &mut self,
        event: TaskEvent,
        limits: &Limits,
        now: DateTime<Utc>,
    ) -> Result<Transition, TransitionError> {
        use TaskState as S;

        let from = self.state;
        let invalid = |event: &TaskEvent| TransitionError::InvalidTransition { from, event: event.name() };

        let to = match (from, event) {
            (S::Pending, TaskEvent::Assign { to }) => {
                if !to.can_own_tasks() {
                    return Err(TransitionError::RoleCannotOwnTasks(to));
                }
                self.assignee = Some(to);
                S::Assigned
            }
            (S::Assigned | S::Revision, TaskEvent::Start) => S::InProgress,
            (S::InProgress, TaskEvent::Submit) => S::Review,
            (S::Review, TaskEvent::Approve) => {
                self.last_feedback = None;
                S::Done
            }
            (S::Review, TaskEvent::RequestRevision { feedback }) => {
                if feedback.trim().is_empty() {
                    return Err(TransitionError::EmptyFeedback);
                }
                self.last_feedback = Some(feedback);
                if self.revisions >= limits.max_revisions {
                    self.failure = Some(FailureReason::RevisionLimit);
                    S::Failed
                } else {
                    self.revisions += 1;
                    S::Revision
                }
            }
            (state, TaskEvent::Fail { reason }) if !state.is_terminal() => {
                self.failure = Some(reason);
                S::Failed
            }
            (S::Failed, TaskEvent::Retry) => {
                if !self.can_retry(limits) {
                    return Err(TransitionError::RetriesExhausted {
                        used: self.retries,
                        max: limits.max_retries,
                    });
                }
                self.retries += 1;
                self.revisions = 0;
                self.failure = None;
                S::Pending
            }
            (_, event) => return Err(invalid(&event)),
        };

        self.state = to;
        self.updated_at = now;
        Ok(Transition { from, to })
    }

    /// An upper bound on how many events this task can ever accept under
    /// `limits`. Used by tests to prove termination; also handy as a sanity
    /// cap in the orchestrator.
    ///
    /// Per attempt: one assign, at most `max_revisions + 1` rounds of
    /// start, submit and verdict, and one fail. Plus one retry event between
    /// attempts.
    pub fn max_events(limits: &Limits) -> u64 {
        let attempts = u64::from(limits.max_retries) + 1;
        let rounds = u64::from(limits.max_revisions) + 1;
        let per_attempt = 1 + 3 * rounds + 1;
        attempts * per_attempt + u64::from(limits.max_retries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }

    fn task() -> Task {
        Task::new(ProjectId::new(), "Write fib.py", "Print the Fibonacci sequence", vec![], now())
    }

    fn drive(task: &mut Task, limits: &Limits, events: impl IntoIterator<Item = TaskEvent>) {
        for event in events {
            task.apply(event, limits, now()).unwrap();
        }
    }

    fn revise(feedback: &str) -> TaskEvent {
        TaskEvent::RequestRevision { feedback: feedback.into() }
    }

    #[test]
    fn happy_path_reaches_done() {
        let limits = Limits::default();
        let mut t = task();
        drive(
            &mut t,
            &limits,
            [
                TaskEvent::Assign { to: AgentRole::Coder },
                TaskEvent::Start,
                TaskEvent::Submit,
                TaskEvent::Approve,
            ],
        );
        assert_eq!(t.state(), TaskState::Done);
        assert_eq!(t.assignee(), Some(AgentRole::Coder));
    }

    #[test]
    fn revision_goes_back_through_in_progress() {
        // Bug 1 in the review: the Python version had no way out of REVISION.
        let limits = Limits::default();
        let mut t = task();
        drive(
            &mut t,
            &limits,
            [
                TaskEvent::Assign { to: AgentRole::Coder },
                TaskEvent::Start,
                TaskEvent::Submit,
                revise("missing tests"),
            ],
        );
        assert_eq!(t.state(), TaskState::Revision);
        assert_eq!(t.revisions(), 1);
        drive(&mut t, &limits, [TaskEvent::Start, TaskEvent::Submit, TaskEvent::Approve]);
        assert_eq!(t.state(), TaskState::Done);
    }

    #[test]
    fn revision_limit_fails_the_task() {
        let limits = Limits { max_revisions: 2, ..Limits::default() };
        let mut t = task();
        drive(&mut t, &limits, [TaskEvent::Assign { to: AgentRole::Coder }]);
        for _ in 0..2 {
            drive(&mut t, &limits, [TaskEvent::Start, TaskEvent::Submit, revise("again")]);
        }
        drive(&mut t, &limits, [TaskEvent::Start, TaskEvent::Submit, revise("again")]);
        assert_eq!(t.state(), TaskState::Failed);
        assert_eq!(t.failure(), Some(&FailureReason::RevisionLimit));
        assert_eq!(t.revisions(), 2);
    }

    #[test]
    fn blank_feedback_is_refused() {
        // The recorded Python loop passed empty feedback back and forth.
        let limits = Limits::default();
        let mut t = task();
        drive(
            &mut t,
            &limits,
            [TaskEvent::Assign { to: AgentRole::Coder }, TaskEvent::Start, TaskEvent::Submit],
        );
        let err = t.apply(revise("   "), &limits, now()).unwrap_err();
        assert_eq!(err, TransitionError::EmptyFeedback);
        assert_eq!(t.state(), TaskState::Review);
    }

    #[test]
    fn retries_are_bounded_and_reset_revisions() {
        let limits = Limits { max_retries: 1, ..Limits::default() };
        let mut t = task();
        drive(
            &mut t,
            &limits,
            [
                TaskEvent::Assign { to: AgentRole::Coder },
                TaskEvent::Start,
                TaskEvent::Submit,
                revise("x"),
                TaskEvent::Start,
                TaskEvent::Fail { reason: FailureReason::Timeout },
            ],
        );
        assert!(t.can_retry(&limits));
        drive(&mut t, &limits, [TaskEvent::Retry]);
        assert_eq!(t.state(), TaskState::Pending);
        assert_eq!(t.revisions(), 0);
        assert_eq!(t.last_feedback(), Some("x"), "feedback survives a retry");

        drive(&mut t, &limits, [TaskEvent::Fail { reason: FailureReason::Timeout }]);
        let err = t.apply(TaskEvent::Retry, &limits, now()).unwrap_err();
        assert_eq!(err, TransitionError::RetriesExhausted { used: 1, max: 1 });
    }

    #[test]
    fn cancelled_tasks_are_never_retried() {
        let limits = Limits { max_retries: 3, ..Limits::default() };
        let mut t = task();
        drive(&mut t, &limits, [TaskEvent::Fail { reason: FailureReason::Cancelled }]);
        assert!(!t.can_retry(&limits));
    }

    #[test]
    fn done_is_final() {
        let limits = Limits::default();
        let mut t = task();
        drive(
            &mut t,
            &limits,
            [
                TaskEvent::Assign { to: AgentRole::Spec },
                TaskEvent::Start,
                TaskEvent::Submit,
                TaskEvent::Approve,
            ],
        );
        for event in [TaskEvent::Start, TaskEvent::Retry, TaskEvent::Fail { reason: FailureReason::Timeout }]
        {
            assert!(t.apply(event, &limits, now()).is_err());
        }
        assert_eq!(t.state(), TaskState::Done);
    }

    #[test]
    fn pm_and_critic_cannot_own_tasks() {
        let limits = Limits::default();
        let mut t = task();
        let err = t.apply(TaskEvent::Assign { to: AgentRole::Critic }, &limits, now()).unwrap_err();
        assert_eq!(err, TransitionError::RoleCannotOwnTasks(AgentRole::Critic));
        assert_eq!(t.state(), TaskState::Pending);
    }

    #[test]
    fn pending_tasks_do_not_time_out() {
        let limits = Limits::default();
        let t = task();
        assert!(!t.is_timed_out(&limits, now() + Duration::days(1)));
    }

    #[test]
    fn in_flight_tasks_time_out() {
        let limits = Limits::default();
        let mut t = task();
        drive(&mut t, &limits, [TaskEvent::Assign { to: AgentRole::Coder }]);
        assert!(!t.is_timed_out(&limits, now() + Duration::minutes(30)));
        assert!(t.is_timed_out(&limits, now() + Duration::minutes(31)));
    }

    #[test]
    fn states_serialize_like_the_python_api() {
        let json = serde_json::to_string(&TaskState::InProgress).unwrap();
        assert_eq!(json, "\"in_progress\"");
    }
}
