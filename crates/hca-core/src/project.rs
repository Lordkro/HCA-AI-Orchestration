//! The project state machine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{Limits, ProjectId, TokenBudget};

/// Where a project is in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectStatus {
    /// Agents may work on it.
    Active,
    /// Work is suspended; queued work waits rather than being dropped.
    Paused,
    /// Every task is done. Terminal.
    Completed,
    /// Stopped by a guardrail. Terminal.
    Failed,
    /// Stopped by a person. Terminal.
    Cancelled,
}

impl ProjectStatus {
    /// `Completed`, `Failed` or `Cancelled`.
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Why a project failed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum ProjectFailure {
    /// The token budget ran out.
    BudgetExhausted,
    /// No task can make progress and no retries are left.
    Stuck,
    /// Planning produced an invalid task graph.
    InvalidPlan(String),
}

/// Something that happens to a project.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProjectEvent {
    /// Suspend work. Only from `Active`.
    Pause,
    /// Continue work. Only from `Paused`.
    Resume,
    /// Every task is done. Only from `Active`.
    Complete,
    /// A guardrail stopped the project. From `Active` or `Paused`.
    Fail {
        /// Why.
        reason: ProjectFailure,
    },
    /// A person stopped the project. From `Active` or `Paused`.
    Cancel,
}

/// Returned when an event isn't allowed in the project's current status.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("cannot apply {event} to a project that is {from:?}")]
pub struct ProjectTransitionError {
    /// The project's status.
    pub from: ProjectStatus,
    /// The refused event's name.
    pub event: &'static str,
}

/// A product idea being built by the team.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Project {
    /// The project's id.
    pub id: ProjectId,
    /// Display name.
    pub name: String,
    /// The user's original idea.
    pub idea: String,
    status: ProjectStatus,
    failure: Option<ProjectFailure>,
    tokens_used: u64,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl Project {
    /// Creates an active project.
    pub fn new(name: impl Into<String>, idea: impl Into<String>, now: DateTime<Utc>) -> Self {
        Self {
            id: ProjectId::new(),
            name: name.into(),
            idea: idea.into(),
            status: ProjectStatus::Active,
            failure: None,
            tokens_used: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// The current status.
    pub const fn status(&self) -> ProjectStatus {
        self.status
    }

    /// Why the project failed, while it is `Failed`.
    pub const fn failure(&self) -> Option<&ProjectFailure> {
        self.failure.as_ref()
    }

    /// Tokens charged to the project so far.
    pub const fn tokens_used(&self) -> u64 {
        self.tokens_used
    }

    /// When the project was created.
    pub const fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// When the project last changed.
    pub const fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    /// Whether agents may do work for this project right now.
    ///
    /// Only `Active` projects accept work: not paused ones, and not failed,
    /// completed or cancelled ones.
    pub const fn accepts_work(&self) -> bool {
        matches!(self.status, ProjectStatus::Active)
    }

    /// The project's token budget under `limits`.
    pub fn budget(&self, limits: &Limits) -> TokenBudget {
        TokenBudget::with_used(limits.project_token_budget, self.tokens_used)
    }

    /// Charges tokens to the project. If that exhausts the budget, an active
    /// or paused project fails with [`ProjectFailure::BudgetExhausted`].
    ///
    /// Returns `true` if the project is still within budget.
    pub fn charge_tokens(&mut self, tokens: u64, limits: &Limits, now: DateTime<Utc>) -> bool {
        self.tokens_used = self.tokens_used.saturating_add(tokens);
        self.updated_at = now;
        let within = !self.budget(limits).is_exhausted();
        if !within && !self.status.is_terminal() {
            self.status = ProjectStatus::Failed;
            self.failure = Some(ProjectFailure::BudgetExhausted);
        }
        within
    }

    /// Applies an event, changing status only if the event is allowed.
    ///
    /// # Errors
    ///
    /// Returns a [`ProjectTransitionError`] if the event isn't allowed in the
    /// current status. The project is unchanged in that case.
    pub fn apply(
        &mut self,
        event: ProjectEvent,
        now: DateTime<Utc>,
    ) -> Result<ProjectStatus, ProjectTransitionError> {
        use ProjectStatus as S;

        let from = self.status;
        let name = match &event {
            ProjectEvent::Pause => "pause",
            ProjectEvent::Resume => "resume",
            ProjectEvent::Complete => "complete",
            ProjectEvent::Fail { .. } => "fail",
            ProjectEvent::Cancel => "cancel",
        };

        let to = match (from, event) {
            (S::Active, ProjectEvent::Pause) => S::Paused,
            (S::Paused, ProjectEvent::Resume) => S::Active,
            (S::Active, ProjectEvent::Complete) => S::Completed,
            (S::Active | S::Paused, ProjectEvent::Fail { reason }) => {
                self.failure = Some(reason);
                S::Failed
            }
            (S::Active | S::Paused, ProjectEvent::Cancel) => S::Cancelled,
            _ => return Err(ProjectTransitionError { from, event: name }),
        };

        self.status = to;
        self.updated_at = now;
        Ok(to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }

    fn project() -> Project {
        Project::new("Fibonacci", "Print the Fibonacci sequence", now())
    }

    #[test]
    fn pause_and_resume() {
        let mut p = project();
        assert!(p.accepts_work());
        p.apply(ProjectEvent::Pause, now()).unwrap();
        assert!(!p.accepts_work());
        p.apply(ProjectEvent::Resume, now()).unwrap();
        assert!(p.accepts_work());
    }

    #[test]
    fn failed_projects_accept_no_work() {
        let mut p = project();
        p.apply(ProjectEvent::Fail { reason: ProjectFailure::Stuck }, now()).unwrap();
        assert!(!p.accepts_work());
        assert_eq!(p.failure(), Some(&ProjectFailure::Stuck));
    }

    #[test]
    fn terminal_statuses_are_final() {
        let mut p = project();
        p.apply(ProjectEvent::Complete, now()).unwrap();
        for event in [ProjectEvent::Resume, ProjectEvent::Pause, ProjectEvent::Cancel] {
            assert!(p.apply(event, now()).is_err());
        }
        assert_eq!(p.status(), ProjectStatus::Completed);
    }

    #[test]
    fn cannot_complete_while_paused() {
        let mut p = project();
        p.apply(ProjectEvent::Pause, now()).unwrap();
        assert!(p.apply(ProjectEvent::Complete, now()).is_err());
    }

    #[test]
    fn spending_the_budget_fails_the_project() {
        let limits = Limits { project_token_budget: 1_000, ..Limits::default() };
        let mut p = project();
        assert!(p.charge_tokens(999, &limits, now()));
        assert!(p.accepts_work());
        assert!(!p.charge_tokens(1, &limits, now()));
        assert_eq!(p.status(), ProjectStatus::Failed);
        assert_eq!(p.failure(), Some(&ProjectFailure::BudgetExhausted));
        assert!(!p.accepts_work());
    }

    #[test]
    fn charging_a_finished_project_keeps_its_status() {
        let limits = Limits { project_token_budget: 10, ..Limits::default() };
        let mut p = project();
        p.apply(ProjectEvent::Complete, now()).unwrap();
        p.charge_tokens(100, &limits, now());
        assert_eq!(p.status(), ProjectStatus::Completed);
    }
}
