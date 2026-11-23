//! The agent loop: call the model, run the tools it asks for, repeat until
//! it calls its finishing tool.
//!
//! Every run is bounded three ways: a step limit, the project's token
//! budget (checked before each call), and a limit on how often the model
//! may stop without finishing.

use std::collections::BTreeSet;

use hca_core::{AgentRole, TokenBudget};
use hca_llm::{Conversation, Llm, LlmError, Message, StopReason, Usage};

use crate::{system_prompt, Submission, Toolbox};

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
    /// `budget` is the project's remaining budget; each call's usage is
    /// recorded against it, and also passed to `on_usage` straight away so
    /// the caller can persist it even if the run later fails.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Llm`] if a model call fails or the budget
    /// refuses it, [`AgentError::StepLimit`] if the step limit is reached,
    /// or [`AgentError::NoSubmission`] if the model keeps stopping without
    /// finishing.
    pub async fn run(
        &self,
        role: AgentRole,
        brief: impl Into<String>,
        mut budget: TokenBudget,
        mut on_usage: impl FnMut(Usage) + Send,
    ) -> Result<AgentRun, AgentError> {
        let finish = Toolbox::finishing_tool(role);
        let mut conversation = Conversation {
            system: Some(system_prompt(role).to_owned()),
            messages: vec![Message::user(brief)],
            tools: self.toolbox.specs(role),
        };
        let mut usage = Usage::default();
        let mut files = BTreeSet::new();
        let mut reminders = 0;

        for step in 1..=self.config.max_steps {
            let response = self.llm.chat(role, &conversation, &budget).await?;
            budget.record(response.usage.total());
            on_usage(response.usage);
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
                let outcome = self.toolbox.execute(role, call).await;
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
