//! Ollama's native chat API.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;
use serde_json::{json, Value};

use super::{http_client, join_url};
use crate::http::{null_as_default, parse_arguments, post_json};
use crate::{
    BoxFuture, ChatRequest, ChatResponse, LlmError, LlmProvider, Message, RetryConfig, StopReason, ToolCall,
    Usage,
};

/// A client for a local or remote Ollama server.
#[derive(Debug)]
pub struct OllamaProvider {
    base_url: String,
    num_ctx: Option<u32>,
    think: Option<bool>,
    retry: RetryConfig,
    client: reqwest::Client,
    next_call_id: AtomicU64,
}

impl OllamaProvider {
    /// A client for `base_url`, e.g. `http://localhost:11434`.
    ///
    /// `num_ctx` sets the context window (Ollama's own default is small).
    /// `think` turns reasoning on or off for models that support it; `None`
    /// leaves the model's default.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::Config`] if the HTTP client can't be built.
    pub fn new(
        base_url: impl Into<String>,
        num_ctx: Option<u32>,
        think: Option<bool>,
        retry: RetryConfig,
        timeout_secs: u64,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            base_url: base_url.into(),
            num_ctx,
            think,
            retry,
            client: http_client(timeout_secs)?,
            next_call_id: AtomicU64::new(1),
        })
    }

    fn body(&self, request: &ChatRequest) -> Value {
        let mut messages = Vec::with_capacity(request.messages.len() + 1);
        if let Some(system) = &request.system {
            messages.push(json!({ "role": "system", "content": system }));
        }
        for message in &request.messages {
            messages.push(match message {
                Message::User { content } => json!({ "role": "user", "content": content }),
                Message::Assistant { content, tool_calls } => {
                    let mut turn = json!({ "role": "assistant", "content": content });
                    if !tool_calls.is_empty() {
                        turn["tool_calls"] = tool_calls
                            .iter()
                            .map(|call| json!({ "function": { "name": call.name, "arguments": call.arguments } }))
                            .collect();
                    }
                    turn
                }
                Message::ToolResult { name, content, is_error, .. } => json!({
                    "role": "tool",
                    "tool_name": name,
                    "content": if *is_error { format!("ERROR: {content}") } else { content.clone() },
                }),
            });
        }

        let mut options = json!({ "num_predict": request.max_tokens });
        if let Some(temperature) = request.temperature {
            options["temperature"] = json!(temperature);
        }
        if let Some(num_ctx) = self.num_ctx {
            options["num_ctx"] = json!(num_ctx);
        }
        let mut body =
            json!({ "model": request.model, "messages": messages, "stream": false, "options": options });
        if let Some(think) = self.think {
            body["think"] = json!(think);
        }
        if !request.tools.is_empty() {
            body["tools"] = request
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": { "name": tool.name, "description": tool.description, "parameters": tool.parameters },
                    })
                })
                .collect();
        }
        body
    }

    async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let url = join_url(&self.base_url, "api/chat");
        let body = self.body(request);
        let response: WireResponse =
            post_json("ollama", &self.retry, || self.client.post(&url), &body).await?;

        // Ollama doesn't give tool calls ids; make unique ones so results
        // can still be matched to calls.
        let tool_calls: Vec<ToolCall> = response
            .message
            .tool_calls
            .into_iter()
            .map(|call| ToolCall {
                id: format!("ollama-{}", self.next_call_id.fetch_add(1, Ordering::Relaxed)),
                name: call.function.name,
                arguments: normalize_arguments(call.function.arguments),
            })
            .collect();
        let stop = match response.done_reason.as_deref() {
            _ if !tool_calls.is_empty() => StopReason::ToolUse,
            Some("stop") | None => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_owned()),
        };
        Ok(ChatResponse {
            content: response.message.content,
            tool_calls,
            stop,
            usage: Usage { input_tokens: response.prompt_eval_count, output_tokens: response.eval_count },
        })
    }
}

impl LlmProvider for OllamaProvider {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn chat<'a>(&'a self, request: &'a ChatRequest) -> BoxFuture<'a, Result<ChatResponse, LlmError>> {
        Box::pin(self.send(request))
    }
    /// `num_ctx`, which Ollama honours by cutting the start of longer
    /// prompts. Unknown when unset: Ollama then picks its own size.
    fn context_window(&self) -> Option<u64> {
        self.num_ctx.map(u64::from)
    }
}

#[derive(Deserialize)]
struct WireResponse {
    message: WireMessage,
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    eval_count: u64,
}

#[derive(Deserialize)]
struct WireMessage {
    #[serde(default, deserialize_with = "null_as_default")]
    content: String,
    #[serde(default, deserialize_with = "null_as_default")]
    tool_calls: Vec<WireToolCall>,
}

#[derive(Deserialize)]
struct WireToolCall {
    function: WireFunction,
}

#[derive(Deserialize)]
struct WireFunction {
    name: String,
    #[serde(default, deserialize_with = "null_as_default")]
    arguments: Value,
}

fn normalize_arguments(value: Value) -> Value {
    match value {
        Value::String(raw) => parse_arguments(&raw),
        Value::Null => serde_json::json!({}),
        other => other,
    }
}
