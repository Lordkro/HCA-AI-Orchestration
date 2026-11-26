//! Provider clients.

mod anthropic;
mod ollama;
mod openai;

use std::time::Duration;

pub use anthropic::AnthropicProvider;
pub use ollama::OllamaProvider;
pub use openai::OpenAiCompatibleProvider;

use crate::LlmError;

pub(crate) fn http_client(timeout_secs: u64) -> Result<reqwest::Client, LlmError> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .connect_timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| LlmError::Config(format!("could not build HTTP client: {e}")))
}

pub(crate) fn join_url(base: &str, path: &str) -> String {
    format!("{}/{}", base.trim_end_matches('/'), path.trim_start_matches('/'))
}
