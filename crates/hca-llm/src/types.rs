//! The provider-neutral conversation format.

use std::future::Future;
use std::pin::Pin;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::LlmError;

/// A boxed, sendable future, so [`LlmProvider`] can be used as a trait object.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A model backend.
pub trait LlmProvider: Send + Sync + std::fmt::Debug {
    /// A short name for logs and errors, e.g. `"anthropic"`.
    fn name(&self) -> &str;

    /// Sends one request and waits for the full response.
    fn chat<'a>(&'a self, request: &'a ChatRequest) -> BoxFuture<'a, Result<ChatResponse, LlmError>>;

    /// The model's context window in tokens (input and output together), if
    /// known. The agent loop keeps conversations within it, so a server
    /// never has to cut the start of a prompt.
    fn context_window(&self) -> Option<u64> {
        None
    }
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
        let schema = schemars::generate::SchemaSettings::default()
            .with(|settings| settings.inline_subschemas = true)
            .into_generator()
            .into_root_schema_for::<T>();
        let mut parameters = serde_json::to_value(schema).unwrap_or_default();
        if let Some(object) = parameters.as_object_mut() {
            scrub_schema(object);
        }
        Self { name: name.into(), description: description.into(), parameters }
    }
}

/// Drops schema bits Anthropic's `input_schema` rejects, and formats it
/// doesn't understand. Nested types are inlined before this runs.
fn scrub_schema(value: &mut Map<String, Value>) {
    value.remove("$schema");
    value.remove("$defs");
    value.remove("definitions");
    value.remove("title");
    if let Some(format) = value.get("format").and_then(Value::as_str) {
        const KEPT: &[&str] = &["date-time", "date", "time", "email", "uri", "uuid"];
        if !KEPT.contains(&format) {
            value.remove("format");
        }
    }
    for child in value.values_mut() {
        scrub_value(child);
    }
}

fn scrub_value(value: &mut Value) {
    match value {
        Value::Object(map) => scrub_schema(map),
        Value::Array(items) => {
            for item in items {
                scrub_value(item);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
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

/// A token estimate: one token per three bytes of UTF-8, rounded up.
///
/// Tokenizers average about four characters per token for English and
/// three for code, so this errs high for both. Scripts where each character
/// is its own token (Chinese, Japanese) take three bytes per character in
/// UTF-8, so they come out about right too. The old estimate, one token per
/// character, ran three to four times high, which refused calls the budget
/// could easily afford.
pub fn estimate_tokens(text: &str) -> u64 {
    u64::try_from(text.len().div_ceil(3)).unwrap_or(u64::MAX)
}

/// Tokens a provider may add per message: role markers and separators of
/// the chat template.
const MESSAGE_OVERHEAD: u64 = 16;
/// Tokens a provider may add per tool when it renders the definition into
/// the prompt.
const TOOL_OVERHEAD: u64 = 64;
/// Tokens a provider may add per request, such as the hidden system prompt
/// Anthropic adds when tools are present (several hundred tokens).
const REQUEST_OVERHEAD: u64 = 1_024;

/// The most tokens `request` can cost: one token per byte of everything
/// sent, the providers' template overhead, and the full output allowance.
///
/// This is a bound, not an estimate, for byte-level tokenizers (BPE as in
/// tiktoken, or `SentencePiece` with byte fallback), which every supported
/// provider uses: no token is shorter than a byte. Budgets reserve this
/// before each call, so a project can't spend more than its budget. It is
/// about three times what English text or code really costs, so near the
/// end of a budget calls are refused while some tokens remain.
pub fn max_request_tokens(request: &ChatRequest) -> u64 {
    let bytes = |text: &str| u64::try_from(text.len()).unwrap_or(u64::MAX);
    let mut total = REQUEST_OVERHEAD + request.system.as_deref().map_or(0, bytes);
    for message in &request.messages {
        total += MESSAGE_OVERHEAD
            + match message {
                Message::User { content } => bytes(content),
                Message::ToolResult { name, content, .. } => bytes(name) + bytes(content),
                Message::Assistant { content, tool_calls } => {
                    bytes(content)
                        + tool_calls
                            .iter()
                            .map(|call| {
                                MESSAGE_OVERHEAD
                                    + bytes(&call.id)
                                    + bytes(&call.name)
                                    + bytes(&call.arguments.to_string())
                            })
                            .sum::<u64>()
                }
            };
    }
    for tool in &request.tools {
        total += TOOL_OVERHEAD
            + bytes(&tool.name)
            + bytes(&tool.description)
            + bytes(&tool.parameters.to_string());
    }
    total.saturating_add(u64::from(request.max_tokens))
}

/// A conservative estimate of the most tokens a request can use: everything
/// sent, plus a small envelope per message, plus the full output allowance.
pub fn estimate_request_tokens(request: &ChatRequest) -> u64 {
    let mut total = request.system.as_deref().map_or(0, estimate_tokens);
    for message in &request.messages {
        // Role, separators, and the JSON envelope around the message.
        total += 8 + match message {
            Message::User { content } => estimate_tokens(content),
            Message::ToolResult { name, content, .. } => estimate_tokens(name) + estimate_tokens(content),
            Message::Assistant { content, tool_calls } => {
                estimate_tokens(content)
                    + tool_calls
                        .iter()
                        .map(|call| {
                            estimate_tokens(&call.id)
                                + estimate_tokens(&call.name)
                                + estimate_tokens(&call.arguments.to_string())
                        })
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

    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Nested {
        /// A nested object.
        item: WriteFile,
        /// How many.
        count: u64,
    }

    #[test]
    fn nested_tool_schemas_have_no_refs() {
        let spec = ToolSpec::of::<Nested>("nested", "A nested tool");
        let text = spec.parameters.to_string();
        assert!(!text.contains("$ref"), "{text}");
        assert!(!text.contains("$defs"), "{text}");
        assert!(!text.contains("uint"), "{text}");
        assert_eq!(spec.parameters["properties"]["item"]["type"], "object");
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
        assert_eq!(estimate_tokens("abcd"), 2, "rounds up");
        assert_eq!(estimate_tokens("abc"), 1);
        assert_eq!(estimate_tokens("日本語"), 3, "a token per CJK character");
        let request = ChatRequest {
            model: "m".into(),
            system: Some("x".repeat(300)),
            messages: vec![Message::user("y".repeat(300))],
            tools: vec![],
            max_tokens: 1000,
            temperature: None,
        };
        assert_eq!(estimate_request_tokens(&request), 100 + 8 + 100 + 1000);
        assert_eq!(max_request_tokens(&request), 1_024 + 300 + 16 + 300 + 1000, "a token per byte");
        let cjk = ChatRequest { system: None, messages: vec![Message::user("日本語")], ..request };
        assert_eq!(max_request_tokens(&cjk), 1_024 + 16 + 9 + 1000, "3 bytes per character");
    }
}
