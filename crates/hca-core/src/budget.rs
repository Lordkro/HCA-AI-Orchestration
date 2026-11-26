//! Token budgets that are checked *before* a model call, not after.

use serde::{Deserialize, Serialize};

/// A project's token allowance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBudget {
    limit: u64,
    used: u64,
}

/// Whether a budget still has room after recording usage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetStatus {
    /// Tokens remain.
    Within,
    /// The budget is spent. No further calls will be authorized.
    Exhausted,
}

/// Returned when a model call is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BudgetError {
    /// Nothing is left.
    #[error("token budget exhausted ({used} of {limit} used)")]
    Exhausted {
        /// Tokens used.
        used: u64,
        /// The limit.
        limit: u64,
    },
    /// The call's estimate is larger than what is left.
    #[error("call needs about {needed} tokens but only {remaining} remain")]
    Insufficient {
        /// The call's estimated size.
        needed: u64,
        /// Tokens left.
        remaining: u64,
    },
}

impl TokenBudget {
    /// A fresh budget.
    pub const fn new(limit: u64) -> Self {
        Self { limit, used: 0 }
    }

    /// A budget with tokens already spent, for example loaded from storage.
    pub const fn with_used(limit: u64, used: u64) -> Self {
        Self { limit, used }
    }

    /// The limit.
    pub const fn limit(&self) -> u64 {
        self.limit
    }

    /// Tokens used.
    pub const fn used(&self) -> u64 {
        self.used
    }

    /// Tokens left.
    pub const fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.used)
    }

    /// Whether nothing is left.
    pub const fn is_exhausted(&self) -> bool {
        self.used >= self.limit
    }

    /// Checks whether a call estimated at `estimated_tokens` may be made.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError::Exhausted`] if the budget is spent, or
    /// [`BudgetError::Insufficient`] if the estimate exceeds what remains.
    pub const fn authorize(&self, estimated_tokens: u64) -> Result<(), BudgetError> {
        if self.is_exhausted() {
            return Err(BudgetError::Exhausted { used: self.used, limit: self.limit });
        }
        let remaining = self.remaining();
        if estimated_tokens > remaining {
            return Err(BudgetError::Insufficient { needed: estimated_tokens, remaining });
        }
        Ok(())
    }

    /// Records the tokens a call actually used.
    pub fn record(&mut self, tokens: u64) -> BudgetStatus {
        self.used = self.used.saturating_add(tokens);
        if self.is_exhausted() {
            BudgetStatus::Exhausted
        } else {
            BudgetStatus::Within
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorizes_within_budget() {
        let budget = TokenBudget::new(1_000);
        assert_eq!(budget.authorize(1_000), Ok(()));
    }

    #[test]
    fn refuses_calls_that_do_not_fit() {
        let budget = TokenBudget::with_used(1_000, 900);
        assert_eq!(budget.authorize(200), Err(BudgetError::Insufficient { needed: 200, remaining: 100 }));
    }

    #[test]
    fn refuses_everything_once_spent() {
        let mut budget = TokenBudget::new(100);
        assert_eq!(budget.record(150), BudgetStatus::Exhausted);
        assert_eq!(budget.authorize(0), Err(BudgetError::Exhausted { used: 150, limit: 100 }));
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn recording_saturates_instead_of_overflowing() {
        let mut budget = TokenBudget::with_used(10, u64::MAX - 1);
        budget.record(10);
        assert_eq!(budget.used(), u64::MAX);
    }
}
