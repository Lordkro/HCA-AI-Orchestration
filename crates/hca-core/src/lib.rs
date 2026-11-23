//! Core domain logic for HCA v2.
//!
//! This crate holds the rules of the system and nothing else: no database,
//! no network, no clock. Every function that needs the current time takes it
//! as an argument, so the whole crate can be tested deterministically.
//!
//! The design goal is that **every project terminates**:
//!
//! - A task can be revised at most [`Limits::max_revisions`] times per attempt,
//!   and retried at most [`Limits::max_retries`] times. Both counters live in
//!   the [`Task`] state machine and cannot be bypassed.
//! - A project only accepts work while it is [`ProjectStatus::Active`].
//! - The [`TokenBudget`] refuses a model call *before* it is made once the
//!   budget is spent.
//! - [`assess`] tells the orchestrator when a project is complete or stuck,
//!   instead of retrying failed tasks forever.
//!
//! These rules close the loop that let the Python version spend 95 million
//! tokens on 1,940 revision cycles of one task.

mod budget;
mod ids;
mod limits;
mod plan;
mod project;
mod role;
mod task;

pub use budget::{BudgetError, BudgetStatus, TokenBudget};
pub use ids::{ProjectId, TaskId};
pub use limits::{Limits, LimitsError};
pub use plan::{assess, ready_tasks, validate_plan, PlanError, ProjectProgress};
pub use project::{Project, ProjectEvent, ProjectFailure, ProjectStatus, ProjectTransitionError};
pub use role::{AgentRole, ParseRoleError};
pub use task::{FailureReason, Task, TaskEvent, TaskState, Transition, TransitionError};
