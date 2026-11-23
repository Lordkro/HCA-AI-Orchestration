//! What each agent role is doing, for the dashboard.

use std::collections::HashMap;
use std::time::Instant;

use hca_core::AgentRole;
use serde::Serialize;

/// One role's current state and totals, shaped for the dashboard.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AgentStatus {
    /// The role.
    pub role: AgentRole,
    /// `"working"` or `"idle"`.
    pub status: &'static str,
    /// The configured model, or `"not configured"`.
    pub model: String,
    /// What it's doing now.
    pub current_activity: String,
    /// How long it has been doing it.
    pub activity_duration_seconds: f64,
    /// Totals since start.
    pub stats: AgentStats,
}

/// Totals for one role since the orchestrator started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct AgentStats {
    /// Runs started.
    pub messages_received: u64,
    /// Model calls made.
    pub llm_calls: u64,
    /// Tokens used.
    pub tokens: u64,
    /// Seconds spent in runs.
    pub total_think_seconds: f64,
}

#[derive(Debug, Default)]
pub(crate) struct StatusBoard {
    running: HashMap<AgentRole, Vec<(u64, String, Instant)>>,
    totals: HashMap<AgentRole, AgentStats>,
    next_id: u64,
}

impl StatusBoard {
    pub(crate) fn start(&mut self, role: AgentRole, activity: String) -> u64 {
        self.next_id += 1;
        self.running.entry(role).or_default().push((self.next_id, activity, Instant::now()));
        self.totals.entry(role).or_default().messages_received += 1;
        self.next_id
    }

    pub(crate) fn finish(&mut self, role: AgentRole, id: u64, calls: u64, tokens: u64) {
        let runs = self.running.entry(role).or_default();
        if let Some(i) = runs.iter().position(|(run, ..)| *run == id) {
            let (.., started) = runs.remove(i);
            let totals = self.totals.entry(role).or_default();
            totals.llm_calls += calls;
            totals.tokens += tokens;
            totals.total_think_seconds += started.elapsed().as_secs_f64();
        }
    }

    pub(crate) fn snapshot(&self, model: impl Fn(AgentRole) -> String) -> Vec<AgentStatus> {
        AgentRole::ALL
            .into_iter()
            .map(|role| {
                let current = self.running.get(&role).and_then(|runs| runs.first());
                let extra = self.running.get(&role).map_or(0, |runs| runs.len().saturating_sub(1));
                AgentStatus {
                    role,
                    status: if current.is_some() { "working" } else { "idle" },
                    model: model(role),
                    current_activity: current.map_or_else(String::new, |(_, what, _)| {
                        if extra > 0 {
                            format!("{what} (+{extra} more)")
                        } else {
                            what.clone()
                        }
                    }),
                    activity_duration_seconds: current
                        .map_or(0.0, |(.., started)| started.elapsed().as_secs_f64()),
                    stats: self.totals.get(&role).copied().unwrap_or_default(),
                }
            })
            .collect()
    }
}
