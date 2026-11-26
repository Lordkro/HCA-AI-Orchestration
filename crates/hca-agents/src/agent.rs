//! The agent loop: call the model, run the tools it asks for, repeat until
//! it calls its finishing tool.
//!
//! Every run is bounded three ways: a step limit, the project's token
//! budget (reserved through a [`UsageMeter`] before each call), and a limit
//! on how often the model may stop without finishing.
//!
//! The conversation is kept within the model's context window. Old tool
//! output and the bodies of old file writes are replaced by a short note
//! when space runs out; messages are never removed, so every tool call
//! keeps its result, as the providers require.

use std::collections::BTreeSet;

use hca_core::{AgentRole, BudgetError};
use hca_llm::{
    estimate_request_tokens, max_request_tokens, Conversation, Llm, LlmError, Message, StopReason, Usage,
};

use crate::{system_prompt, MeterError, Submission, Toolbox, UsageMeter};

/// What replaces old tool output when the context fills up.
const ELIDED_OUTPUT: &str =
    "[older output removed to fit the context window; run the tool again if you need it]";

/// What replaces the text arguments of old tool calls.
const ELIDED_ARGUMENT: &str = "[removed to fit the context window]";

/// Messages at the end of the conversation that are never trimmed: the
/// model's latest turns and what it just saw.
const KEEP_RECENT: usize = 6;

/// Tool results shorter than this aren't worth trimming.
const MIN_ELIDE_BYTES: usize = 300;

/// Limits on one agent run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentConfig {
    /// Model calls allowed in one run.
    pub max_steps: u32,
    /// Times the model may end its turn without calling its finishing tool
    /// before the run fails. Each time, it is reminded to finish.
    pub max_reminders: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self { max_steps: 40, max_reminders: 2 }
    }
}

/// Why a run ended without a submission.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// The model call failed, or was refused by the budget.
    #[error(transparent)]
    Llm(#[from] LlmError),
    /// The step limit was reached.
    #[error("the agent used all {0} steps without finishing")]
    StepLimit(u32),
    /// The model kept stopping without calling its finishing tool.
    #[error("the agent stopped {0} times without finishing")]
    NoSubmission(u32),
    /// The meter stopped the run between steps, e.g. because the project
    /// was paused. The work isn't finished and didn't fail.
    #[error("{0}")]
    Stopped(String),
    /// Token usage couldn't be checked or saved.
    #[error("{0}")]
    Meter(String),
    /// Even with old output trimmed, the conversation doesn't fit the
    /// model's context window.
    #[error(
        "the conversation needs about {needed} tokens but the model's context window is {window}; \
         use a model with a bigger window (for Ollama, raise num_ctx) or lower max_tokens"
    )]
    ContextFull {
        /// Estimated tokens of the smallest request that could be built.
        needed: u64,
        /// The model's context window.
        window: u64,
    },
}

impl From<MeterError> for AgentError {
    fn from(error: MeterError) -> Self {
        match error {
            MeterError::Budget(e) => Self::Llm(LlmError::Budget(e)),
            MeterError::Stopped(reason) => Self::Stopped(reason),
            MeterError::Failed(message) => Self::Meter(message),
        }
    }
}

/// A finished run.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentRun {
    /// What the agent handed in.
    pub submission: Submission,
    /// Model calls made.
    pub steps: u32,
    /// Tokens used across all calls.
    pub usage: Usage,
    /// Files written or edited, sorted.
    pub files_written: Vec<String>,
    /// The whole conversation, for logs and debugging.
    pub transcript: Vec<Message>,
}

/// Runs agents for one project.
#[derive(Debug)]
pub struct Agent<'a> {
    llm: &'a Llm,
    toolbox: &'a Toolbox,
    config: AgentConfig,
}

impl<'a> Agent<'a> {
    /// An agent using `llm` for model calls and `toolbox` for tools.
    pub fn new(llm: &'a Llm, toolbox: &'a Toolbox, config: AgentConfig) -> Self {
        Self { llm, toolbox, config }
    }

    /// Runs `role` on `brief` until it hands in its work.
    ///
    /// Each call's worst-case size is reserved with `meter` before it is
    /// sent, and its usage is settled with `meter` right after, so the
    /// tokens are counted even if the run fails later.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Llm`] if a model call fails or the budget
    /// refuses it, [`AgentError::Stopped`] if the meter stops the run,
    /// [`AgentError::StepLimit`] if the step limit is reached,
    /// [`AgentError::NoSubmission`] if the model keeps stopping without
    /// finishing, or [`AgentError::ContextFull`] if the conversation can't
    /// be made to fit the model.
    pub async fn run(
        &self,
        role: AgentRole,
        brief: impl Into<String>,
        meter: &dyn UsageMeter,
    ) -> Result<AgentRun, AgentError> {
        let finish = Toolbox::finishing_tool(role);
        let mut conversation = Conversation {
            system: Some(system_prompt(role).to_owned()),
            messages: vec![Message::user(brief)],
            tools: self.toolbox.specs(role),
        };
        let window = self.llm.context_window(role);
        let mut usage = Usage::default();
        let mut files = BTreeSet::new();
        let mut reminders = 0;

        for step in 1..=self.config.max_steps {
            let mut request = self.llm.request(role, &conversation)?;
            if let Some(window) = window {
                let needed = estimate_request_tokens(&request);
                if needed > window {
                    fit(&mut conversation.messages, needed - window);
                    request = self.llm.request(role, &conversation)?;
                    let needed = estimate_request_tokens(&request);
                    if needed > window {
                        return Err(AgentError::ContextFull { needed, window });
                    }
                }
            }
            let response = self.call(role, &mut conversation, &mut request, meter).await?;
            usage.input_tokens += response.usage.input_tokens;
            usage.output_tokens += response.usage.output_tokens;
            conversation.messages.push(response.to_message());

            if response.tool_calls.is_empty() {
                let nudge = if response.stop == StopReason::MaxTokens {
                    format!("Your reply was cut off at the output limit. Continue, and finish by calling `{finish}`.")
                } else {
                    reminders += 1;
                    if reminders > self.config.max_reminders {
                        return Err(AgentError::NoSubmission(reminders));
                    }
                    format!(
                        "You haven't finished. Continue the task, and finish by calling the `{finish}` tool."
                    )
                };
                conversation.messages.push(Message::user(nudge));
                continue;
            }

            let mut submission = None;
            for call in &response.tool_calls {
                if submission.is_some() {
                    // Every call needs a result, but nothing runs after the
                    // agent has handed in its work.
                    conversation
                        .messages
                        .push(Message::tool_error(call, "not run: the task was already submitted"));
                    continue;
                }
                let mut outcome = self.toolbox.execute(role, call).await;
                if let Some(window) = window {
                    clip_result(&mut outcome.message, window);
                }
                if let Some(path) = outcome.wrote {
                    files.insert(path);
                }
                submission = outcome.submission;
                conversation.messages.push(outcome.message);
            }

            if let Some(submission) = submission {
                return Ok(AgentRun {
                    submission,
                    steps: step,
                    usage,
                    files_written: files.into_iter().collect(),
                    transcript: conversation.messages,
                });
            }
        }
        Err(AgentError::StepLimit(self.config.max_steps))
    }
}

/// Trims the oldest tool output and tool-call text until about `excess`
/// tokens are freed. The brief (the first message) and the last
/// [`KEEP_RECENT`] messages are kept whole. Nothing is removed outright:
/// results keep their call ids and arguments stay JSON objects.
///
/// Returns the tokens freed: 0 when nothing is left to trim.
/// Retries of a call that failed in a way that may have been billed (a
/// timeout, a cut connection), each charged in full. The HTTP layer
/// doesn't retry these, so that no attempt goes uncounted.
const MAX_BILLED_RETRIES: u32 = 2;

/// Wait before the first such retry; it doubles after.
const BILLED_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

impl Agent<'_> {
    /// Makes one model call: reserves its worst case, sends it, settles
    /// what it cost.
    ///
    /// - If the budget can't cover it, the oldest tool output is trimmed
    ///   until it can. Only when nothing is left to trim does the call fail
    ///   with the budget error.
    /// - A call that failed in a way that may have been billed is charged
    ///   in full and tried again, up to [`MAX_BILLED_RETRIES`] times, so a
    ///   timeout late in a long run doesn't throw the run away.
    async fn call(
        &self,
        role: AgentRole,
        conversation: &mut Conversation,
        request: &mut hca_llm::ChatRequest,
        meter: &dyn UsageMeter,
    ) -> Result<hca_llm::ChatResponse, AgentError> {
        let mut retries = 0;
        loop {
            // Reserve the worst case: whatever happens next, the call can't
            // cost more.
            let bound = max_request_tokens(request);
            match meter.reserve(bound).await {
                Ok(()) => {}
                Err(MeterError::Budget(BudgetError::Insufficient { needed, remaining })) => {
                    // The bound counts a token per byte, the trimming a
                    // token per three bytes.
                    if fit(&mut conversation.messages, (needed - remaining).div_ceil(3)) == 0 {
                        return Err(
                            MeterError::Budget(BudgetError::Insufficient { needed, remaining }).into()
                        );
                    }
                    *request = self.llm.request(role, conversation)?;
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
            let result = self.llm.send(role, request).await;
            let spent = match &result {
                Ok(response) => Some(response.usage),
                Err(error) if error.is_unbilled() => Some(Usage::default()),
                // Maybe billed, at an unknown cost: count the worst case.
                Err(_) => None,
            };
            meter.settle(bound, spent).await?;
            match result {
                Ok(response) => return Ok(response),
                Err(error)
                    if error.is_retryable() && !error.is_unbilled() && retries < MAX_BILLED_RETRIES =>
                {
                    tracing::warn!(%error, "model call failed after it was sent; trying again");
                    tokio::time::sleep(BILLED_RETRY_DELAY * 2_u32.pow(retries)).await;
                    retries += 1;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

fn fit(messages: &mut [Message], excess: u64) -> u64 {
    let end = messages.len().saturating_sub(KEEP_RECENT);
    let mut freed = 0_u64;
    for message in messages.iter_mut().take(end).skip(1) {
        if freed >= excess {
            return freed;
        }
        match message {
            Message::ToolResult { content, .. } if content.len() > MIN_ELIDE_BYTES => {
                freed += hca_llm::estimate_tokens(content);
                ELIDED_OUTPUT.clone_into(content);
            }
            Message::Assistant { tool_calls, .. } => {
                for call in tool_calls {
                    if let Some(arguments) = call.arguments.as_object_mut() {
                        for value in arguments.values_mut() {
                            if value.as_str().is_some_and(|text| text.len() > MIN_ELIDE_BYTES) {
                                freed += hca_llm::estimate_tokens(value.as_str().unwrap_or_default());
                                *value = serde_json::Value::String(ELIDED_ARGUMENT.to_owned());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    freed
}

/// Shortens a single tool result that would take more than a quarter of
/// the context window. Command output keeps its end, where errors are;
/// anything else keeps its start.
fn clip_result(message: &mut Message, window: u64) {
    let Message::ToolResult { name, content, .. } = message else { return };
    let limit = usize::try_from((window / 4).saturating_mul(3)).unwrap_or(usize::MAX);
    if content.len() <= limit {
        return;
    }
    let hidden = content.len() - limit;
    if name == "run_command" {
        let mut start = hidden;
        while !content.is_char_boundary(start) {
            start += 1;
        }
        *content = format!("[... {start} bytes cut to fit the context window ...]\n{}", &content[start..]);
    } else {
        let mut end = limit;
        while !content.is_char_boundary(end) {
            end -= 1;
        }
        let rest = content.len() - end;
        *content = format!(
            "{}\n[... {rest} more bytes not shown, to fit the context window. Read the rest in parts with \
             start_line and end_line.]",
            &content[..end]
        );
    }
}

#[cfg(test)]
mod tests {
    use hca_llm::ToolCall;
    use serde_json::json;

    use super::*;

    fn call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall { id: id.into(), name: name.into(), arguments }
    }

    #[test]
    fn fitting_keeps_every_call_paired_with_its_result() {
        let big = "x".repeat(3_000);
        let mut messages = vec![Message::user("the brief")];
        for i in 0..6 {
            let write = call(&format!("w{i}"), "write_file", json!({"path": "a.py", "content": big}));
            let read = call(&format!("r{i}"), "read_file", json!({"path": "a.py"}));
            messages.push(Message::Assistant {
                content: String::new(),
                tool_calls: vec![write.clone(), read.clone()],
            });
            messages.push(Message::tool_result(&write, "wrote a.py"));
            messages.push(Message::tool_result(&read, big.clone()));
        }
        let before = messages.clone();
        fit(&mut messages, u64::MAX);

        assert_eq!(messages.len(), before.len(), "no message is removed");
        assert_eq!(messages[0], before[0], "the brief is kept");
        let recent = before.len() - KEEP_RECENT;
        assert_eq!(&messages[recent..], &before[recent..], "recent turns are kept");
        for (now, was) in messages.iter().zip(&before) {
            match (now, was) {
                (Message::ToolResult { call_id: a, .. }, Message::ToolResult { call_id: b, .. }) => {
                    assert_eq!(a, b);
                }
                (Message::Assistant { tool_calls: a, .. }, Message::Assistant { tool_calls: b, .. }) => {
                    for (x, y) in a.iter().zip(b) {
                        assert_eq!((&x.id, &x.name), (&y.id, &y.name));
                        assert!(x.arguments.is_object());
                        assert_eq!(x.arguments["path"], "a.py", "short arguments stay");
                    }
                }
                _ => {}
            }
        }
        let Message::ToolResult { content, .. } = &messages[3] else { panic!("expected a result") };
        assert_eq!(content, ELIDED_OUTPUT);
        let Message::Assistant { tool_calls, .. } = &messages[1] else { panic!("expected a turn") };
        assert_eq!(tool_calls[0].arguments["content"], ELIDED_ARGUMENT);
    }

    #[test]
    fn fitting_stops_once_enough_is_freed() {
        let big = "x".repeat(3_000);
        let calls: Vec<_> =
            (0..10).map(|i| call(&format!("c{i}"), "read_file", json!({"path": "a"}))).collect();
        let mut messages = vec![Message::user("brief")];
        messages.extend(calls.iter().map(|c| Message::tool_result(c, big.clone())));
        fit(&mut messages, 1);
        let elided = messages
            .iter()
            .filter(|m| matches!(m, Message::ToolResult { content, .. } if content == ELIDED_OUTPUT))
            .count();
        assert_eq!(elided, 1, "only as much as needed");
    }

    #[test]
    fn huge_results_are_clipped_at_the_useful_end() {
        let read = call("r", "read_file", json!({"path": "big.txt"}));
        let mut message = Message::tool_result(&read, format!("START{}", "x".repeat(10_000)));
        clip_result(&mut message, 1_000);
        let Message::ToolResult { content, .. } = &message else { unreachable!() };
        assert!(content.starts_with("START") && content.contains("start_line"), "{content}");
        assert!(content.len() < 1_000);

        let run = call("c", "run_command", json!({"command": "pytest"}));
        let mut message = Message::tool_result(&run, format!("{}FAILED test_x", "y".repeat(10_000)));
        clip_result(&mut message, 1_000);
        let Message::ToolResult { content, .. } = &message else { unreachable!() };
        assert!(content.ends_with("FAILED test_x"), "{content}");
    }
}
