//! Errors from model calls.

use hca_core::{AgentRole, BudgetError};

/// Everything that can go wrong when talking to a model.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// The request never got a response (connection, DNS, timeout).
    #[error("could not reach {provider}: {source}")]
    Transport {
        /// The provider's name.
        provider: String,
        /// The underlying error.
        #[source]
        source: reqwest::Error,
    },
    /// The provider answered with an error status.
    #[error("{provider} returned HTTP {status}: {body}")]
    Status {
        /// The provider's name.
        provider: String,
        /// The HTTP status code.
        status: u16,
        /// The response body, truncated.
        body: String,
    },
    /// The provider's response wasn't in the expected format.
    #[error("unexpected response from {provider}: {message}")]
    Decode {
        /// The provider's name.
        provider: String,
        /// What was wrong.
        message: String,
    },
    /// The call was refused before it was made because the project's token
    /// budget can't cover it.
    #[error(transparent)]
    Budget(#[from] BudgetError),
    /// No model is configured for this role.
    #[error("no model is configured for the {0} role")]
    NoModelForRole(AgentRole),
    /// The configuration is invalid, for example a missing API key.
    #[error("invalid model configuration: {0}")]
    Config(String),
}

impl LlmError {
    /// Whether the provider certainly didn't bill the call: it was never
    /// sent (no connection, a configuration problem, refused by the budget)
    /// or the provider rejected it with an error status. A timeout, a cut
    /// connection or an unreadable response may still have been billed.
    pub fn is_unbilled(&self) -> bool {
        match self {
            Self::Transport { source, .. } => source.is_connect() || source.is_builder(),
            Self::Status { .. } | Self::Budget(_) | Self::NoModelForRole(_) | Self::Config(_) => true,
            Self::Decode { .. } => false,
        }
    }

    /// Whether trying again later could succeed: network failures, rate
    /// limits (429) and server errors (5xx).
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Transport { .. } => true,
            Self::Status { status, .. } => *status == 429 || *status >= 500,
            _ => false,
        }
    }
}
