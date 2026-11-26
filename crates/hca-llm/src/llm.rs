//! Per-role model access with a budget gate.

use std::collections::HashMap;
use std::sync::Arc;

use hca_core::{AgentRole, TokenBudget};

use crate::{estimate_request_tokens, ChatRequest, ChatResponse, LlmError, LlmProvider, Message, ToolSpec};

/// The model settings for one role.
#[derive(Clone, Debug)]
pub struct RoleModel {
    /// The backend.
    pub provider: Arc<dyn LlmProvider>,
    /// The provider's model name.
    pub model: String,
    /// Output token limit per call.
    pub max_tokens: u32,
    /// Sampling temperature, if set.
    pub temperature: Option<f32>,
}

/// What an agent sends: everything except the model settings, which come
/// from its role.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Conversation {
    /// The system prompt.
    pub system: Option<String>,
    /// The turns so far.
    pub messages: Vec<Message>,
    /// Tools the model may call.
    pub tools: Vec<ToolSpec>,
}

/// Model access for every role.
#[derive(Clone, Debug)]
pub struct Llm {
    roles: HashMap<AgentRole, RoleModel>,
}

impl Llm {
    /// Builds from explicit role models. Usually made by
    /// [`LlmConfig::build`](crate::LlmConfig::build).
    pub fn new(roles: HashMap<AgentRole, RoleModel>) -> Self {
        Self { roles }
    }

    /// The model settings for `role`, if configured.
    ///
    /// Research and spec work use the coder's model when they have none of
    /// their own, so a three-role config (pm, coder, critic) runs every task
    /// the PM can plan.
    pub fn role(&self, role: AgentRole) -> Option<&RoleModel> {
        self.roles.get(&role).or_else(|| match role {
            AgentRole::Research | AgentRole::Spec => self.roles.get(&AgentRole::Coder),
            AgentRole::Pm | AgentRole::Coder | AgentRole::Critic => None,
        })
    }

    /// The context window of `role`'s model in tokens, if known.
    pub fn context_window(&self, role: AgentRole) -> Option<u64> {
        self.role(role).and_then(|model| model.provider.context_window())
    }

    /// The request that [`chat`](Self::chat) would send for `role`.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::NoModelForRole`] if the role isn't configured.
    pub fn request(&self, role: AgentRole, conversation: &Conversation) -> Result<ChatRequest, LlmError> {
        let settings = self.role(role).ok_or(LlmError::NoModelForRole(role))?;
        Ok(ChatRequest {
            model: settings.model.clone(),
            system: conversation.system.clone(),
            messages: conversation.messages.clone(),
            tools: conversation.tools.clone(),
            max_tokens: settings.max_tokens,
            temperature: settings.temperature,
        })
    }

    /// Sends the conversation to `role`'s model, if the budget allows it.
    ///
    /// The worst-case size of the call (everything sent plus the full output
    /// allowance) is checked against `budget` first, and the call is refused
    /// if it might not fit. The caller charges the response's
    /// [`usage`](ChatResponse::usage) to the project afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::NoModelForRole`], [`LlmError::Budget`] if the call
    /// was refused, or the provider's error.
    pub async fn chat(
        &self,
        role: AgentRole,
        conversation: &Conversation,
        budget: &TokenBudget,
    ) -> Result<ChatResponse, LlmError> {
        let request = self.request(role, conversation)?;
        budget.authorize(estimate_request_tokens(&request))?;
        self.send(role, &request).await
    }

    /// Sends a request built by [`request`](Self::request) without a budget
    /// check. For callers that authorize the call themselves, such as the
    /// agent loop, which reserves the estimate against the project's shared
    /// budget.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::NoModelForRole`] or the provider's error.
    pub async fn send(&self, role: AgentRole, request: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let settings = self.role(role).ok_or(LlmError::NoModelForRole(role))?;
        settings.provider.chat(request).await
    }
}
