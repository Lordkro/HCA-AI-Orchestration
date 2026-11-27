//! Token accounting for agent runs.
//!
//! The agent loop asks a [`UsageMeter`] before every model call and reports
//! to it after every call. That one seam gives three guarantees:
//!
//! - **The budget is checked against everything in flight.** A meter
//!   reserves each call's worst case ([`hca_llm::max_request_tokens`])
//!   before it is sent, so parallel runs sharing a project budget can't
//!   spend more than it between them.
//! - **Usage is recorded per call**, so tokens are counted even if the run
//!   later fails, is cancelled, or the process dies. A call whose cost is
//!   unknown (a timeout, say) is charged its full worst case.
//! - **A run can be stopped between steps**, for example when its project
//!   is paused or HCA is shutting down.

use std::sync::{Mutex, PoisonError};

use hca_core::{BudgetError, TokenBudget};
use hca_llm::{BoxFuture, Usage};

/// Why a meter refused a call or couldn't record one.
#[derive(Debug, thiserror::Error)]
pub enum MeterError {
    /// The budget can't cover the call.
    #[error(transparent)]
    Budget(#[from] BudgetError),
    /// The run should stop, for the given reason. Not a failure of the
    /// work: the caller decides what happens to it.
    #[error("stopped: {0}")]
    Stopped(String),
    /// Usage couldn't be read or saved.
    #[error("could not account for tokens: {0}")]
    Failed(String),
}

/// Checks and records the tokens of an agent run's model calls.
pub trait UsageMeter: Send + Sync {
    /// Called before each model call with its estimated worst-case size.
    /// Reserve it, or refuse the call.
    fn reserve(&self, estimate: u64) -> BoxFuture<'_, Result<(), MeterError>>;

    /// Called after each call, whether it succeeded or not, with the same
    /// estimate and the tokens actually used: `None` when that is unknown
    /// because the call failed in a way that may still have been billed.
    /// Release the reservation and record the usage, or the full estimate
    /// for `None`.
    fn settle(&self, estimate: u64, usage: Option<Usage>) -> BoxFuture<'_, Result<(), MeterError>>;
}

/// A meter over a budget held in memory. For tests and one-off runs.
#[derive(Debug)]
pub struct LocalMeter {
    state: Mutex<LocalState>,
}

#[derive(Debug)]
struct LocalState {
    budget: TokenBudget,
    reserved: u64,
    calls: Vec<Usage>,
}

impl LocalMeter {
    /// A meter that starts from `budget`.
    pub fn new(budget: TokenBudget) -> Self {
        Self { state: Mutex::new(LocalState { budget, reserved: 0, calls: Vec::new() }) }
    }

    /// The budget, with every settled call recorded.
    pub fn budget(&self) -> TokenBudget {
        self.lock().budget
    }

    /// The usage of each settled call, in order.
    pub fn calls(&self) -> Vec<Usage> {
        self.lock().calls.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LocalState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl UsageMeter for LocalMeter {
    fn reserve(&self, estimate: u64) -> BoxFuture<'_, Result<(), MeterError>> {
        let result = {
            let mut state = self.lock();
            let committed = state.budget.used().saturating_add(state.reserved);
            TokenBudget::with_used(state.budget.limit(), committed).authorize(estimate).map(|()| {
                state.reserved = state.reserved.saturating_add(estimate);
            })
        };
        Box::pin(async move { result.map_err(MeterError::from) })
    }

    fn settle(&self, estimate: u64, usage: Option<Usage>) -> BoxFuture<'_, Result<(), MeterError>> {
        {
            let mut state = self.lock();
            state.reserved = state.reserved.saturating_sub(estimate);
            let usage = usage.unwrap_or(Usage { input_tokens: estimate, output_tokens: 0 });
            state.budget.record(usage.total());
            state.calls.push(usage);
        }
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reservations_count_against_the_budget_until_settled() {
        let meter = LocalMeter::new(TokenBudget::new(1_000));
        meter.reserve(600).await.unwrap();
        let refused = meter.reserve(600).await.unwrap_err();
        assert!(matches!(refused, MeterError::Budget(BudgetError::Insufficient { .. })), "{refused:?}");

        meter.settle(600, Some(Usage { input_tokens: 100, output_tokens: 50 })).await.unwrap();
        assert_eq!(meter.budget().used(), 150, "the real usage is charged, not the estimate");
        meter.reserve(600).await.unwrap();
        meter.settle(600, None).await.unwrap();
        assert_eq!(meter.budget().used(), 750, "an unknown cost counts in full");
    }
}
