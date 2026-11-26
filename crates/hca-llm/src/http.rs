//! Shared HTTP plumbing: send JSON, retry what's worth retrying.

use std::time::Duration;

use serde::de::DeserializeOwned;

use crate::{LlmError, RetryConfig};

/// Most characters of an error body kept in [`LlmError::Status`].
const MAX_ERROR_BODY: usize = 2_000;

/// POSTs `body` and decodes the JSON response, retrying transport errors,
/// 429s and 5xx responses with exponential backoff.
pub(crate) async fn post_json<T: DeserializeOwned>(
    provider: &str,
    retry: &RetryConfig,
    build: impl Fn() -> reqwest::RequestBuilder,
    body: &serde_json::Value,
) -> Result<T, LlmError> {
    let mut attempt = 0;
    loop {
        let result = send_once(provider, build().json(body)).await;
        match result {
            Err(err) if err.is_retryable() && attempt < retry.max_retries => {
                let delay = retry.base_delay_ms.saturating_mul(1 << attempt.min(16));
                tokio::time::sleep(Duration::from_millis(delay)).await;
                attempt += 1;
            }
            other => return other,
        }
    }
}

async fn send_once<T: DeserializeOwned>(
    provider: &str,
    request: reqwest::RequestBuilder,
) -> Result<T, LlmError> {
    let transport = |source| LlmError::Transport { provider: provider.to_owned(), source };
    let response = request.send().await.map_err(transport)?;
    let status = response.status();
    let text = response.text().await.map_err(transport)?;
    if !status.is_success() {
        let body: String = text.chars().take(MAX_ERROR_BODY).collect();
        return Err(LlmError::Status { provider: provider.to_owned(), status: status.as_u16(), body });
    }
    serde_json::from_str(&text)
        .map_err(|e| LlmError::Decode { provider: provider.to_owned(), message: e.to_string() })
}

/// Parses tool-call arguments that arrive as a JSON string. Invalid JSON is
/// kept as a raw string so the agent can tell the model what went wrong.
pub(crate) fn parse_arguments(raw: &str) -> serde_json::Value {
    if raw.trim().is_empty() {
        return serde_json::Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_owned()))
}
