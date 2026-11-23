//! A scripted provider for tests of code that calls models.

use std::collections::VecDeque;
use std::sync::{Mutex, PoisonError};

use crate::{BoxFuture, ChatRequest, ChatResponse, LlmError, LlmProvider, StopReason, ToolCall, Usage};

/// Returns queued responses in order and records every request.
#[derive(Debug, Default)]
pub struct ScriptedProvider {
    responses: Mutex<VecDeque<Result<ChatResponse, LlmError>>>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl ScriptedProvider {
    /// An empty script.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues a plain text answer.
    #[must_use]
    pub fn then_text(self, text: impl Into<String>) -> Self {
        self.then(Ok(ChatResponse {
            content: text.into(),
            tool_calls: vec![],
            stop: StopReason::EndTurn,
            usage: Usage { input_tokens: 10, output_tokens: 10 },
        }))
    }

    /// Queues one tool call.
    #[must_use]
    pub fn then_tool(self, name: impl Into<String>, arguments: serde_json::Value) -> Self {
        let id = format!("call-{}", self.requests_len() + self.queued_len() + 1);
        self.then(Ok(ChatResponse {
            content: String::new(),
            tool_calls: vec![ToolCall { id, name: name.into(), arguments }],
            stop: StopReason::ToolUse,
            usage: Usage { input_tokens: 10, output_tokens: 10 },
        }))
    }

    /// Queues any result, including an error.
    #[must_use]
    pub fn then(self, result: Result<ChatResponse, LlmError>) -> Self {
        self.responses.lock().unwrap_or_else(PoisonError::into_inner).push_back(result);
        self
    }

    /// Every request received so far.
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn requests_len(&self) -> usize {
        self.requests.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    fn queued_len(&self) -> usize {
        self.responses.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

impl LlmProvider for ScriptedProvider {
    fn name(&self) -> &'static str {
        "scripted"
    }

    fn chat<'a>(&'a self, request: &'a ChatRequest) -> BoxFuture<'a, Result<ChatResponse, LlmError>> {
        self.requests.lock().unwrap_or_else(PoisonError::into_inner).push(request.clone());
        let next = self.responses.lock().unwrap_or_else(PoisonError::into_inner).pop_front();
        Box::pin(async move {
            next.unwrap_or_else(|| {
                Err(LlmError::Decode {
                    provider: "scripted".into(),
                    message: "script ran out of responses".into(),
                })
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    #[tokio::test]
    async fn replays_in_order_and_records_requests() {
        let provider =
            ScriptedProvider::new().then_tool("list_files", serde_json::json!({})).then_text("done");
        let request = ChatRequest {
            model: "m".into(),
            system: None,
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
        };
        assert_eq!(provider.chat(&request).await.unwrap().stop, StopReason::ToolUse);
        assert_eq!(provider.chat(&request).await.unwrap().content, "done");
        assert!(provider.chat(&request).await.is_err(), "an exhausted script errors instead of hanging");
        assert_eq!(provider.requests().len(), 3);
    }
}
