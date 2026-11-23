//! Agent roles.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The roles in the agent team.
///
/// Serialized in `snake_case` (`"pm"`, `"research"`, ...) to stay compatible
/// with the existing dashboard and API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// Plans the project. Never routes work: routing is deterministic code.
    Pm,
    /// Investigates technologies and writes research reports.
    Research,
    /// Writes technical specifications.
    Spec,
    /// Writes and fixes code.
    Coder,
    /// Reviews deliverables.
    Critic,
}

impl AgentRole {
    /// Every role, in pipeline order.
    pub const ALL: [Self; 5] = [Self::Pm, Self::Research, Self::Spec, Self::Coder, Self::Critic];

    /// The role's wire name, as used in the API and database.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pm => "pm",
            Self::Research => "research",
            Self::Spec => "spec",
            Self::Coder => "coder",
            Self::Critic => "critic",
        }
    }

    /// Whether tasks can be assigned to this role.
    ///
    /// The PM plans and the critic reviews; neither owns tasks.
    pub const fn can_own_tasks(self) -> bool {
        matches!(self, Self::Research | Self::Spec | Self::Coder)
    }
}

impl fmt::Display for AgentRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Returned when a string is not a known role name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown agent role: {0:?}")]
pub struct ParseRoleError(pub String);

impl FromStr for AgentRole {
    type Err = ParseRoleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL.into_iter().find(|role| role.as_str() == s).ok_or_else(|| ParseRoleError(s.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_names_match_the_python_api() {
        for role in AgentRole::ALL {
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, format!("\"{}\"", role.as_str()));
            assert_eq!(role.as_str().parse::<AgentRole>().unwrap(), role);
        }
    }

    #[test]
    fn only_workers_own_tasks() {
        assert!(!AgentRole::Pm.can_own_tasks());
        assert!(!AgentRole::Critic.can_own_tasks());
        assert!(AgentRole::Coder.can_own_tasks());
    }

    #[test]
    fn rejects_unknown_names() {
        assert!("system".parse::<AgentRole>().is_err());
    }
}
