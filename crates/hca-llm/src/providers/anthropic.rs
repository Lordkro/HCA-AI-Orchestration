//! Anthropic Messages API.

use serde::Deserialize;
use serde_json::{json, Value};

use super::{http_client, join_url};
use crate::http::post_json;
use crate::{
    BoxFuture, ChatRequest, ChatResponse, LlmError, LlmProvider, Message, RetryConfig, StopReason, ToolCall,
    Usage,
};

const API_VERSION: &str = "2023-06-01";

/// A client for the Anthropic Messages API.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    base_url: String,
    api_key: String,
    retry: RetryConfig,
    client: reqwest::Client,
}

impl AnthropicProvider {
    /// A client for `base_url` (normally `https://api.anthropic.com/v1`).
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::Config`] if the HTTP client can't be built.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        retry: RetryConfig,
        timeout_secs: u64,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            retry,
            client: http_client(timeout_secs)?,
        })
    }

    fn body(request: &ChatRequest) -> Value {
        // The Messages API needs alternating user/assistant turns. Tool
        // results are user-side content blocks, so consecutive results (and
        // a user message right after them) merge into one user turn.
        let mut turns: Vec<(&str, Vec<Value>)> = Vec::new();
        let mut push = |role: &'static str, block: Value| match turns.last_mut() {
            Some((last, blocks)) if *last == role => blocks.push(block),
            _ => turns.push((role, vec![block])),
        };
        for message in &request.messages {
            match message {
                Message::User { content } => push("user", json!({ "type": "text", "text": content })),
                Message::Assistant { content, tool_calls } => {
                    if !content.is_empty() {
                        push("assistant", json!({ "type": "text", "text": content }));
                    }
                    for call in tool_calls {
                        push(
                            "assistant",
                            json!({ "type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments }),
                        );
                    }
                }
                Message::ToolResult { call_id, content, is_error, .. } => push(
                    "user",
                    json!({ "type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error }),
                ),
            }
        }
        let messages: Vec<Value> =
            turns.into_iter().map(|(role, content)| json!({ "role": role, "content": content })).collect();

        let mut body =
            json!({ "model": request.model, "max_tokens": request.max_tokens, "messages": messages });
        if let Some(system) = &request.system {
            body["system"] = json!(system);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if !request.tools.is_empty() {
            body["tools"] = request
                .tools
                .iter()
                .map(|tool| json!({ "name": tool.name, "description": tool.description, "input_schema": tool.parameters }))
                .collect();
        }
        body
    }

    async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let url = join_url(&self.base_url, "messages");
        let body = Self::body(request);
        let response: WireResponse = post_json(
            "anthropic",
            &self.retry,
            || {
                self.client
                    .post(&url)
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", API_VERSION)
            },
            &body,
        )
        .await?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        for block in response.content {
            match block {
                WireBlock::Text { text } => content.push_str(&text),
                WireBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall { id, name, arguments: input });
                }
                WireBlock::Other => {}
            }
        }
        let stop = match response.stop_reason.as_deref() {
            Some("end_turn" | "stop_sequence") | None => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_owned()),
        };
        Ok(ChatResponse {
            content,
            tool_calls,
            stop,
            usage: Usage {
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
            },
        })
    }
}

impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn chat<'a>(&'a self, request: &'a ChatRequest) -> BoxFuture<'a, Result<ChatResponse, LlmError>> {
        Box::pin(self.send(request))
    }
}

#[derive(Deserialize)]
struct WireResponse {
    content: Vec<WireBlock>,
    stop_reason: Option<String>,
    #[serde(default)]
    usage: WireUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}
