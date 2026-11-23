//! Model providers for HCA v2.
//!
//! - One provider-neutral conversation format ([`ChatRequest`],
//!   [`Message`], [`ToolSpec`], [`ChatResponse`]) and one trait,
//!   [`LlmProvider`].
//! - Clients for **Ollama**, **Anthropic**, **OpenAI**, **Grok (xAI)** and
//!   any other OpenAI-compatible server (vLLM, LM Studio, OpenRouter...).
//! - [`LlmConfig`]: a provider and model per agent role, with API keys read
//!   from environment variables, never from the config file.
//! - [`Llm::chat`]: checks the project's [`TokenBudget`](hca_core::TokenBudget)
//!   *before* calling the model.
//! - [`ContextBuilder`]: assembles a prompt from the task, feedback, test
//!   output and files within a token budget, instead of replaying raw chat
//!   history.

mod config;
mod context;
mod error;
mod http;
mod llm;
mod providers;
pub mod testing;
mod types;

pub use config::{LlmConfig, MaxTokensField, ProviderConfig, RetryConfig, RoleConfig};
pub use context::{BuiltContext, ContextBuilder, Priority, Truncate};
pub use error::LlmError;
pub use llm::{Conversation, Llm, RoleModel};
pub use providers::{AnthropicProvider, OllamaProvider, OpenAiCompatibleProvider};
pub use types::{
    estimate_request_tokens, estimate_tokens, BoxFuture, ChatRequest, ChatResponse, LlmProvider, Message,
    StopReason, ToolCall, ToolSpec, Usage,
};
