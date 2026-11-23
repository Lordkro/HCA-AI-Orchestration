//! Model configuration: named providers, and a model per agent role.
//!
//! ```toml
//! [providers.local]
//! kind = "ollama"
//! num_ctx = 16384
//!
//! [providers.claude]
//! kind = "anthropic"            # key read from ANTHROPIC_API_KEY
//!
//! [providers.grok]
//! kind = "grok"                 # key read from XAI_API_KEY
//!
//! [roles.pm]
//! provider = "claude"
//! model = "claude-opus-5-5"
//!
//! [roles.coder]
//! provider = "grok"
//! model = "grok-4"
//! max_tokens = 8192
//!
//! [roles.research]
//! provider = "local"
//! model = "qwen3:8b"
//! ```
//!
//! API keys are only ever read from environment variables. The config names
//! the variable, so it can be committed without leaking secrets.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use hca_core::AgentRole;
use serde::{Deserialize, Serialize};

use crate::{
    AnthropicProvider, Llm, LlmError, LlmProvider, OllamaProvider, OpenAiCompatibleProvider, RoleModel,
};

/// Which request field carries the output token limit on an
/// OpenAI-compatible server.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
    /// `max_tokens`: understood by most compatible servers and by xAI.
    #[default]
    MaxTokens,
    /// `max_completion_tokens`: what OpenAI's current models expect.
    MaxCompletionTokens,
}

impl MaxTokensField {
    /// The JSON field name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

/// One model backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderConfig {
    /// A local or remote Ollama server.
    Ollama {
        /// Defaults to `http://localhost:11434`.
        #[serde(default = "defaults::ollama_url")]
        base_url: String,
        /// Context window in tokens. Unset uses Ollama's default, which is
        /// usually too small for coding.
        #[serde(default)]
        num_ctx: Option<u32>,
        /// Reasoning on or off for models that support it. Unset leaves the
        /// model's default.
        #[serde(default)]
        think: Option<bool>,
    },
    /// Anthropic's Messages API.
    Anthropic {
        /// Defaults to `https://api.anthropic.com/v1`.
        #[serde(default = "defaults::anthropic_url")]
        base_url: String,
        /// Environment variable holding the key. Defaults to `ANTHROPIC_API_KEY`.
        #[serde(default = "defaults::anthropic_key")]
        api_key_env: String,
    },
    /// OpenAI's API.
    Openai {
        /// Defaults to `https://api.openai.com/v1`.
        #[serde(default = "defaults::openai_url")]
        base_url: String,
        /// Defaults to `OPENAI_API_KEY`.
        #[serde(default = "defaults::openai_key")]
        api_key_env: String,
    },
    /// xAI's Grok models, through xAI's OpenAI-compatible API.
    Grok {
        /// Defaults to `https://api.x.ai/v1`.
        #[serde(default = "defaults::grok_url")]
        base_url: String,
        /// Defaults to `XAI_API_KEY`.
        #[serde(default = "defaults::grok_key")]
        api_key_env: String,
    },
    /// Any other server that speaks OpenAI Chat Completions.
    OpenaiCompatible {
        /// The server's base URL, including `/v1` if it needs it.
        base_url: String,
        /// Environment variable holding the key, if the server needs one.
        #[serde(default)]
        api_key_env: Option<String>,
        /// Which field carries the output token limit.
        #[serde(default)]
        max_tokens_field: MaxTokensField,
    },
}

/// Retry policy for transport errors, 429s and 5xx responses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RetryConfig {
    /// Retries after the first attempt.
    pub max_retries: u32,
    /// First backoff delay; doubles with each retry.
    pub base_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self { max_retries: 2, base_delay_ms: 1_000 }
    }
}

/// The model an agent role uses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoleConfig {
    /// A key of [`LlmConfig::providers`].
    pub provider: String,
    /// The provider's model name.
    pub model: String,
    /// Output token limit per call. Defaults to 4096.
    #[serde(default = "defaults::max_tokens")]
    pub max_tokens: u32,
    /// Sampling temperature. Unset uses the provider's default; `0.0` is a
    /// real zero.
    #[serde(default)]
    pub temperature: Option<f32>,
}

/// All model settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Named providers.
    pub providers: BTreeMap<String, ProviderConfig>,
    /// The model for each role. Roles without an entry can't call a model.
    pub roles: BTreeMap<AgentRole, RoleConfig>,
    /// Request timeout. Defaults to 600 seconds, since local models can be
    /// slow.
    #[serde(default = "defaults::timeout_secs")]
    pub timeout_secs: u64,
    /// Retry policy for every provider.
    #[serde(default)]
    pub retry: RetryConfig,
}

impl LlmConfig {
    /// Builds the clients, reading API keys from the process environment.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::Config`] if a role names an unknown provider, an
    /// API key variable is unset or empty, or a client can't be built.
    pub fn build(&self) -> Result<Llm, LlmError> {
        self.build_with_env(|name| std::env::var(name).ok())
    }

    /// Like [`build`](Self::build), with a custom environment lookup.
    ///
    /// # Errors
    ///
    /// Same as [`build`](Self::build).
    pub fn build_with_env(&self, env: impl Fn(&str) -> Option<String>) -> Result<Llm, LlmError> {
        let key = |var: &str| {
            env(var)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| LlmError::Config(format!("environment variable {var} is not set")))
        };

        // Only build providers that some role uses, so an unused provider's
        // missing key isn't an error.
        let used: std::collections::BTreeSet<&str> =
            self.roles.values().map(|r| r.provider.as_str()).collect();
        let mut providers: HashMap<&str, Arc<dyn LlmProvider>> = HashMap::new();
        for name in used {
            let config = self.providers.get(name).ok_or_else(|| {
                LlmError::Config(format!("a role uses provider {name:?}, which isn't defined"))
            })?;
            let (timeout, retry) = (self.timeout_secs, self.retry);
            let provider: Arc<dyn LlmProvider> = match config {
                ProviderConfig::Ollama { base_url, num_ctx, think } => {
                    Arc::new(OllamaProvider::new(base_url, *num_ctx, *think, retry, timeout)?)
                }
                ProviderConfig::Anthropic { base_url, api_key_env } => {
                    Arc::new(AnthropicProvider::new(base_url, key(api_key_env)?, retry, timeout)?)
                }
                ProviderConfig::Openai { base_url, api_key_env } => Arc::new(OpenAiCompatibleProvider::new(
                    name,
                    base_url,
                    Some(key(api_key_env)?),
                    MaxTokensField::MaxCompletionTokens,
                    retry,
                    timeout,
                )?),
                ProviderConfig::Grok { base_url, api_key_env } => Arc::new(OpenAiCompatibleProvider::new(
                    name,
                    base_url,
                    Some(key(api_key_env)?),
                    MaxTokensField::MaxTokens,
                    retry,
                    timeout,
                )?),
                ProviderConfig::OpenaiCompatible { base_url, api_key_env, max_tokens_field } => {
                    let api_key = api_key_env.as_deref().map(key).transpose()?;
                    Arc::new(OpenAiCompatibleProvider::new(
                        name,
                        base_url,
                        api_key,
                        *max_tokens_field,
                        retry,
                        timeout,
                    )?)
                }
            };
            providers.insert(name, provider);
        }

        let roles = self
            .roles
            .iter()
            .map(|(role, config)| {
                let provider = Arc::clone(&providers[config.provider.as_str()]);
                let model = RoleModel {
                    provider,
                    model: config.model.clone(),
                    max_tokens: config.max_tokens,
                    temperature: config.temperature,
                };
                (*role, model)
            })
            .collect();
        Ok(Llm::new(roles))
    }
}

mod defaults {
    pub fn ollama_url() -> String {
        "http://localhost:11434".into()
    }
    pub fn anthropic_url() -> String {
        "https://api.anthropic.com/v1".into()
    }
    pub fn anthropic_key() -> String {
        "ANTHROPIC_API_KEY".into()
    }
    pub fn openai_url() -> String {
        "https://api.openai.com/v1".into()
    }
    pub fn openai_key() -> String {
        "OPENAI_API_KEY".into()
    }
    pub fn grok_url() -> String {
        "https://api.x.ai/v1".into()
    }
    pub fn grok_key() -> String {
        "XAI_API_KEY".into()
    }
    pub const fn max_tokens() -> u32 {
        4096
    }
    pub const fn timeout_secs() -> u64 {
        600
    }
}
