//! The project state machine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{BudgetError, Limits, ProjectId, TokenBudget};

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
    /// A person restarts a failed project: back to `Active`. Only a person
    /// sends this, never the automatic routing, so every project still ends
    /// on its own.
    Reopen,
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

/// Why [`Project::reserve_tokens`] didn't reserve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reservation {
    /// The tokens are reserved.
    Granted,
    /// The call would fit what is spent, but not with the reservations of
    /// calls still in flight. Wait for those to settle and ask again.
    Busy,
    /// The call can't fit even once every other call is settled.
    Refused(BudgetError),
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
    /// Worst-case sizes of model calls in flight, not yet settled.
    #[serde(default)]
    tokens_reserved: u64,
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
            tokens_reserved: 0,
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

    /// Tokens reserved for model calls still in flight.
    pub const fn tokens_reserved(&self) -> u64 {
        self.tokens_reserved
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

    /// Reserves `tokens`, a call's worst-case size, before the call is made.
    ///
    /// The call must fit the budget together with every reservation still
    /// in flight, so parallel calls can never spend more than the budget
    /// between them. Every reservation is later settled with
    /// [`settle_tokens`](Self::settle_tokens).
    pub fn reserve_tokens(&mut self, tokens: u64, limits: &Limits) -> Reservation {
        let committed = self.tokens_used.saturating_add(self.tokens_reserved);
        if let Err(error) = self.budget(limits).authorize(tokens) {
            return Reservation::Refused(error);
        }
        if TokenBudget::with_used(limits.project_token_budget, committed).authorize(tokens).is_err() {
            return Reservation::Busy;
        }
        self.tokens_reserved = self.tokens_reserved.saturating_add(tokens);
        Reservation::Granted
    }

    /// Settles a reservation of `reserved` tokens: releases it and charges
    /// `charged`, what the call actually cost (the full reservation when
    /// that is unknown). Settles in any status, since a call can finish
    /// after its project ended. Exhausting the budget fails an active or
    /// paused project, as [`charge_tokens`](Self::charge_tokens) does.
    ///
    /// Returns `true` if the project is still within budget.
    pub fn settle_tokens(
        &mut self,
        reserved: u64,
        charged: u64,
        limits: &Limits,
        now: DateTime<Utc>,
    ) -> bool {
        self.tokens_reserved = self.tokens_reserved.saturating_sub(reserved);
        self.charge_tokens(charged, limits, now)
    }

    /// Charges every open reservation in full. Run at startup: a call that
    /// was in flight when the process stopped may have been billed, and
    /// nobody is left to settle it.
    ///
    /// Returns `true` if the project is still within budget.
    pub fn charge_reservations(&mut self, limits: &Limits, now: DateTime<Utc>) -> bool {
        let reserved = std::mem::take(&mut self.tokens_reserved);
        self.charge_tokens(reserved, limits, now)
    }

    /// Fails the project because the remaining budget can't cover the next
    /// model call, though it isn't spent to the last token. Does nothing to
    /// a project that already ended.
    pub fn exhaust_budget(&mut self, now: DateTime<Utc>) {
        if !self.status.is_terminal() {
            self.status = ProjectStatus::Failed;
            self.failure = Some(ProjectFailure::BudgetExhausted);
            self.updated_at = now;
        }
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
            ProjectEvent::Reopen => "reopen",
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
            (S::Failed, ProjectEvent::Reopen) => {
                self.failure = None;
                S::Active
            }
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

    #[test]
    fn only_failed_projects_reopen() {
        let mut p = project();
        assert!(p.apply(ProjectEvent::Reopen, now()).is_err(), "an active project can't reopen");
        p.apply(ProjectEvent::Fail { reason: ProjectFailure::Stuck }, now()).unwrap();
        assert_eq!(p.apply(ProjectEvent::Reopen, now()), Ok(ProjectStatus::Active));
        assert_eq!(p.failure(), None);

        let mut done = project();
        done.apply(ProjectEvent::Complete, now()).unwrap();
        assert!(done.apply(ProjectEvent::Reopen, now()).is_err(), "completed stays completed");
        let mut cancelled = project();
        cancelled.apply(ProjectEvent::Cancel, now()).unwrap();
        assert!(cancelled.apply(ProjectEvent::Reopen, now()).is_err(), "cancelled stays cancelled");
    }

    #[test]
    fn reservations_keep_parallel_calls_within_the_budget() {
        let limits = Limits { project_token_budget: 1_000, ..Limits::default() };
        let mut p = project();
        assert_eq!(p.reserve_tokens(700, &limits), Reservation::Granted);
        assert_eq!(p.reserve_tokens(700, &limits), Reservation::Busy, "fits once the first call settles");
        assert!(matches!(p.reserve_tokens(1_001, &limits), Reservation::Refused(_)), "never fits");

        assert!(p.settle_tokens(700, 250, &limits, now()));
        assert_eq!((p.tokens_used(), p.tokens_reserved()), (250, 0), "the real cost, not the worst case");
        assert_eq!(p.reserve_tokens(700, &limits), Reservation::Granted);

        // A call nobody settled (the process died) counts in full.
        assert!(p.charge_reservations(&limits, now()));
        assert_eq!((p.tokens_used(), p.tokens_reserved()), (950, 0));
        assert!(matches!(p.reserve_tokens(100, &limits), Reservation::Refused(_)));
        p.exhaust_budget(now());
        assert_eq!(p.failure(), Some(&ProjectFailure::BudgetExhausted));
    }

    #[test]
    fn projects_saved_before_reservations_load() {
        let mut json = serde_json::to_value(project()).unwrap();
        json.as_object_mut().unwrap().remove("tokens_reserved");
        let p: Project = serde_json::from_value(json).unwrap();
        assert_eq!(p.tokens_reserved(), 0);
    }
}
