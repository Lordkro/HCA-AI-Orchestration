//! OpenAI Chat Completions and everything compatible with it: OpenAI, Grok
//! (xAI), vLLM, LM Studio, OpenRouter and others.
//!
//! xAI documents Chat Completions as its OpenAI-compatible, stateless API
//! and points new integrations at its Responses API; a Responses client can
//! be added later without changing the [`LlmProvider`] trait.

use serde::Deserialize;
use serde_json::{json, Value};

use super::{http_client, join_url};
use crate::http::{parse_arguments, post_json};
use crate::{
    BoxFuture, ChatRequest, ChatResponse, LlmError, LlmProvider, MaxTokensField, Message, RetryConfig,
    StopReason, ToolCall, Usage,
};

/// A client for any server that speaks OpenAI Chat Completions.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleProvider {
    name: String,
    base_url: String,
    api_key: Option<String>,
    max_tokens_field: MaxTokensField,
    retry: RetryConfig,
    client: reqwest::Client,
    context_window: Option<u64>,
}

impl OpenAiCompatibleProvider {
    /// A client for `base_url` (which includes `/v1` where the server needs
    /// it), authenticating with `api_key` as a bearer token if given.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::Config`] if the HTTP client can't be built.
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
        max_tokens_field: MaxTokensField,
        retry: RetryConfig,
        timeout_secs: u64,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            name: name.into(),
            base_url: base_url.into(),
            api_key,
            max_tokens_field,
            retry,
            client: http_client(timeout_secs)?,
            context_window: None,
        })
    }

    /// Sets the context window of the server's models, in tokens.
    #[must_use]
    pub fn with_context_window(mut self, tokens: Option<u64>) -> Self {
        self.context_window = tokens;
        self
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
                    // Chat Completions requires content or tool_calls. An
                    // empty turn with neither is sent as "" so the next call
                    // is not rejected. Null content is valid alongside tool
                    // calls.
                    let content = if content.is_empty() && tool_calls.is_empty() {
                        json!("")
                    } else if content.is_empty() {
                        Value::Null
                    } else {
                        json!(content)
                    };
                    let mut turn = json!({ "role": "assistant", "content": content });
                    if !tool_calls.is_empty() {
                        turn["tool_calls"] = tool_calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": { "name": call.name, "arguments": call.arguments.to_string() },
                                })
                            })
                            .collect();
                    }
                    turn
                }
                Message::ToolResult { call_id, content, is_error, .. } => json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": if *is_error { format!("ERROR: {content}") } else { content.clone() },
                }),
            });
        }

        let mut body = json!({ "model": request.model, "messages": messages });
        body[self.max_tokens_field.as_str()] = json!(request.max_tokens);
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if !request.tools.is_empty() {
            body["tools"] = request
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    })
                })
                .collect();
        }
        body
    }

    async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let url = join_url(&self.base_url, "chat/completions");
        let body = self.body(request);
        let response: WireResponse = post_json(
            &self.name,
            &self.retry,
            || {
                let builder = self.client.post(&url);
                match &self.api_key {
                    Some(key) => builder.bearer_auth(key),
                    None => builder,
                }
            },
            &body,
        )
        .await?;

        let choice = response.choices.into_iter().next().ok_or_else(|| LlmError::Decode {
            provider: self.name.clone(),
            message: "response has no choices".into(),
        })?;
        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
            .into_iter()
            .map(|call| ToolCall {
                id: call.id,
                name: call.function.name,
                arguments: parse_arguments(&call.function.arguments),
            })
            .collect();
        let stop = match choice.finish_reason.as_deref() {
            _ if !tool_calls.is_empty() => StopReason::ToolUse,
            Some("stop") | None => StopReason::EndTurn,
            Some("tool_calls") => StopReason::ToolUse,
            Some("length") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_owned()),
        };
        let usage = response.usage.unwrap_or_default();
        Ok(ChatResponse {
            content: choice.message.content.unwrap_or_default(),
            tool_calls,
            stop,
            usage: Usage { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
        })
    }
}

impl LlmProvider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn chat<'a>(&'a self, request: &'a ChatRequest) -> BoxFuture<'a, Result<ChatResponse, LlmError>> {
        Box::pin(self.send(request))
    }
    fn context_window(&self) -> Option<u64> {
        self.context_window
    }
}

#[derive(Deserialize)]
struct WireResponse {
    choices: Vec<WireChoice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireChoice {
    message: WireMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireMessage {
    content: Option<String>,
    #[serde(default, deserialize_with = "crate::http::null_as_default")]
    tool_calls: Vec<WireToolCall>,
}

#[derive(Deserialize)]
struct WireToolCall {
    id: String,
    function: WireFunction,
}

#[derive(Deserialize)]
struct WireFunction {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}
