//! The provider-neutral conversation format.

use std::future::Future;
use std::pin::Pin;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::LlmError;

/// A boxed, sendable future, so [`LlmProvider`] can be used as a trait object.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A model backend.
pub trait LlmProvider: Send + Sync + std::fmt::Debug {
    /// A short name for logs and errors, e.g. `"anthropic"`.
    fn name(&self) -> &str;

    /// Sends one request and waits for the full response.
    fn chat<'a>(&'a self, request: &'a ChatRequest) -> BoxFuture<'a, Result<ChatResponse, LlmError>>;
}

/// One turn in a conversation. The system prompt is separate, on
/// [`ChatRequest::system`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// Text from the user or the orchestrator.
    User {
        /// The text.
        content: String,
    },
    /// A model turn: text, tool calls, or both.
    Assistant {
        /// The text. May be empty when the model only called tools.
        content: String,
        /// Tools the model asked to run.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    /// The result of running one tool call.
    ToolResult {
        /// The [`ToolCall::id`] this answers.
        call_id: String,
        /// The tool's name.
        name: String,
        /// The tool's output, or the error message.
        content: String,
        /// Whether the tool failed.
        #[serde(default)]
        is_error: bool,
    },
}

impl Message {
    /// A user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::User { content: content.into() }
    }

    /// A successful tool result for `call`.
    pub fn tool_result(call: &ToolCall, content: impl Into<String>) -> Self {
        Self::ToolResult {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: content.into(),
            is_error: false,
        }
    }

    /// A failed tool result for `call`.
    pub fn tool_error(call: &ToolCall, error: impl Into<String>) -> Self {
        Self::ToolResult {
            call_id: call.id.clone(),
            name: call.name.clone(),
            content: error.into(),
            is_error: true,
        }
    }
}

/// A tool the model may call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// The tool's name, e.g. `"write_file"`.
    pub name: String,
    /// What the tool does, written for the model.
    pub description: String,
    /// JSON Schema of the arguments object.
    pub parameters: serde_json::Value,
}

impl ToolSpec {
    /// A tool whose arguments are the Rust type `T`. The JSON Schema is
    /// derived from `T`, so the schema the model sees and the type the
    /// arguments are parsed into can't drift apart.
    pub fn of<T: JsonSchema>(name: impl Into<String>, description: impl Into<String>) -> Self {
        let mut parameters = serde_json::to_value(schemars::schema_for!(T)).unwrap_or_default();
        if let Some(object) = parameters.as_object_mut() {
            object.remove("$schema");
            object.remove("title");
        }
        Self { name: name.into(), description: description.into(), parameters }
    }
}

/// A tool call requested by the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Identifies the call, so its result can be matched to it.
    pub id: String,
    /// The tool's name.
    pub name: String,
    /// The arguments. If the model sent arguments that aren't valid JSON,
    /// this holds the raw text as a JSON string.
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Parses the arguments into `T`.
    ///
    /// # Errors
    ///
    /// Returns the deserialization error, worded so it can be sent back to
    /// the model as a tool error.
    pub fn parse<T: DeserializeOwned>(&self) -> Result<T, String> {
        serde_json::from_value(self.arguments.clone())
            .map_err(|e| format!("invalid arguments for {}: {e}", self.name))
    }
}

/// Why the model stopped.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// It finished its turn.
    EndTurn,
    /// It wants tools run.
    ToolUse,
    /// It hit the output token limit.
    MaxTokens,
    /// Anything else the provider reported.
    Other(String),
}

/// Tokens used by one call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens.
    pub input_tokens: u64,
    /// Completion tokens, including reasoning tokens where reported.
    pub output_tokens: u64,
}

impl Usage {
    /// Input plus output.
    pub const fn total(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

/// One model call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// The provider's model name.
    pub model: String,
    /// The system prompt.
    pub system: Option<String>,
    /// The conversation so far.
    pub messages: Vec<Message>,
    /// Tools the model may call.
    pub tools: Vec<ToolSpec>,
    /// Maximum output tokens.
    pub max_tokens: u32,
    /// Sampling temperature. `None` uses the provider's default; `0.0`
    /// really means zero.
    pub temperature: Option<f32>,
}

/// A model's answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The text part of the answer.
    pub content: String,
    /// Tools the model wants run.
    pub tool_calls: Vec<ToolCall>,
    /// Why it stopped.
    pub stop: StopReason,
    /// Tokens used.
    pub usage: Usage,
}

impl ChatResponse {
    /// The answer as an assistant message, to append to the conversation.
    pub fn to_message(&self) -> Message {
        Message::Assistant { content: self.content.clone(), tool_calls: self.tool_calls.clone() }
    }
}

/// A conservative token estimate for text: about 3 characters per token,
/// rounded up. Errs high so budget checks refuse too early rather than too
/// late.
pub fn estimate_tokens(text: &str) -> u64 {
    let chars = u64::try_from(text.chars().count()).unwrap_or(u64::MAX);
    chars.div_ceil(3)
}

/// A conservative estimate of the most tokens a request can use: everything
/// sent, plus the full output allowance.
pub fn estimate_request_tokens(request: &ChatRequest) -> u64 {
    let mut total = request.system.as_deref().map_or(0, estimate_tokens);
    for message in &request.messages {
        total += 4 + match message {
            Message::User { content } | Message::ToolResult { content, .. } => estimate_tokens(content),
            Message::Assistant { content, tool_calls } => {
                estimate_tokens(content)
                    + tool_calls
                        .iter()
                        .map(|c| estimate_tokens(&c.name) + estimate_tokens(&c.arguments.to_string()))
                        .sum::<u64>()
            }
        };
    }
    for tool in &request.tools {
        total += estimate_tokens(&tool.name)
            + estimate_tokens(&tool.description)
            + estimate_tokens(&tool.parameters.to_string());
    }
    total + u64::from(request.max_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
    struct WriteFile {
        /// Path relative to the project root.
        path: String,
        /// Full file contents.
        content: String,
    }

    #[test]
    fn tool_schemas_come_from_types() {
        let spec = ToolSpec::of::<WriteFile>("write_file", "Write a file");
        assert_eq!(spec.parameters["type"], "object");
        assert_eq!(spec.parameters["required"], serde_json::json!(["path", "content"]));
        assert_eq!(
            spec.parameters["properties"]["path"]["description"],
            "Path relative to the project root."
        );
        assert!(spec.parameters.get("$schema").is_none());
    }

    #[test]
    fn tool_calls_parse_into_types() {
        let call = ToolCall {
            id: "1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({"path": "a.py", "content": "print(1)"}),
        };
        assert_eq!(
            call.parse::<WriteFile>().unwrap(),
            WriteFile { path: "a.py".into(), content: "print(1)".into() }
        );

        let bad = ToolCall { arguments: serde_json::json!({"path": "a.py"}), ..call };
        let err = bad.parse::<WriteFile>().unwrap_err();
        assert!(err.contains("missing field `content`"), "{err}");
    }

    #[test]
    fn estimates_err_high() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 2);
        let request = ChatRequest {
            model: "m".into(),
            system: Some("x".repeat(300)),
            messages: vec![Message::user("y".repeat(300))],
            tools: vec![],
            max_tokens: 1000,
            temperature: None,
        };
        assert_eq!(estimate_request_tokens(&request), 100 + 104 + 1000);
    }
}
