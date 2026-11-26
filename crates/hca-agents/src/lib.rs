//! The HCA agent loop, workspace tools and role prompts.
//!
//! An agent run is: a role's system prompt, a brief built by the
//! orchestrator, and a loop of model calls and tool calls until the model
//! calls its finishing tool (`submit`, `submit_review` or `submit_plan`).
//! Agents see the results of their tools and can take as many steps as the
//! task needs, within a step limit and the project's token budget.

mod agent;
mod runner;
mod submission;
mod tools;
mod workspace;

pub use agent::{Agent, AgentConfig, AgentError, AgentRun};
pub use runner::{CommandOutput, CommandRunner, CRITIC_USER};
pub use submission::{Plan, PlanRole, PlannedTask, Review, Submission, SubmitArgs, Verdict};
pub use tools::{ToolOutcome, Toolbox};
pub use workspace::{Workspace, WorkspaceError};

use hca_core::AgentRole;

/// The system prompt for `role`. Prompts live in `prompts/*.txt` and are
/// compiled into the binary.
pub const fn system_prompt(role: AgentRole) -> &'static str {
    match role {
        AgentRole::Pm => include_str!("../prompts/pm.txt"),
        AgentRole::Research => include_str!("../prompts/research.txt"),
        AgentRole::Spec => include_str!("../prompts/spec.txt"),
        AgentRole::Coder => include_str!("../prompts/coder.txt"),
        AgentRole::Critic => include_str!("../prompts/critic.txt"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_prompt_names_its_finishing_tool() {
        for role in AgentRole::ALL {
            let finish = Toolbox::finishing_tool(role);
            assert!(system_prompt(role).contains(finish), "the {role} prompt never mentions {finish}");
        }
    }
}
