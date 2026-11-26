//! What an agent hands in when it finishes, and the tools it uses to do so.

use chrono::{DateTime, Utc};
use hca_core::{validate_plan, AgentRole, Limits, ProjectId, Task};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The result of a finished agent run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Submission {
    /// Research, spec or code work is ready for review.
    Work {
        /// What was done, for the reviewer.
        summary: String,
    },
    /// The critic's verdict.
    Review(Review),
    /// The PM's plan.
    Plan(Plan),
}

/// Arguments of the `submit` tool.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct SubmitArgs {
    /// What you did and how you checked it, for the reviewer. Mention every
    /// file you created or changed.
    pub summary: String,
}

/// The critic's decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The work meets the task.
    Approve,
    /// The work needs changes; the feedback says which.
    RequestChanges,
}

/// Arguments of the `submit_review` tool, and the critic's submission.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Review {
    /// `approve` or `request_changes`.
    pub verdict: Verdict,
    /// What must change, specific enough to act on: file, problem, fix.
    /// Required when requesting changes.
    #[serde(default)]
    pub feedback: String,
}

/// A role the PM can plan work for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanRole {
    /// Investigation; produces notes in `docs/research/`.
    Research,
    /// A technical specification.
    Spec,
    /// Code and tests.
    Coder,
}

impl From<PlanRole> for AgentRole {
    fn from(role: PlanRole) -> Self {
        match role {
            PlanRole::Research => Self::Research,
            PlanRole::Spec => Self::Spec,
            PlanRole::Coder => Self::Coder,
        }
    }
}

/// One task in a plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PlannedTask {
    /// Short title.
    pub title: String,
    /// Everything the assigned agent needs: what to build, where, and how
    /// to know it's done.
    pub description: String,
    /// Who does it.
    pub role: PlanRole,
    /// Positions (0-based) of earlier tasks in this list that must finish
    /// first.
    #[serde(default)]
    pub depends_on: Vec<usize>,
}

/// Arguments of the `submit_plan` tool, and the PM's submission.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Plan {
    /// The tasks, in a sensible order.
    pub tasks: Vec<PlannedTask>,
}

impl Plan {
    /// Turns the plan into tasks for `project_id`, with dependencies linked.
    ///
    /// # Errors
    ///
    /// Returns a message for the model if a title is blank, a dependency
    /// points at itself or past the end of the list, or the plan breaks a
    /// rule in [`validate_plan`] (too many tasks, a cycle).
    pub fn into_tasks(
        self,
        project_id: ProjectId,
        limits: &Limits,
        now: DateTime<Utc>,
    ) -> Result<Vec<Task>, String> {
        if self.tasks.is_empty() {
            return Err("the plan has no tasks".into());
        }
        let count = self.tasks.len();
        for (i, task) in self.tasks.iter().enumerate() {
            if task.title.trim().is_empty() {
                return Err(format!("task {i} has no title"));
            }
            if task.description.trim().is_empty() {
                return Err(format!("task {i} ({}) has no description", task.title));
            }
            if let Some(&bad) = task.depends_on.iter().find(|&&d| d >= count || d == i) {
                return Err(format!(
                    "task {i} ({}) depends on {bad}, which is not another task in the list (valid: 0 to {})",
                    task.title,
                    count - 1
                ));
            }
        }

        let mut tasks: Vec<Task> = self
            .tasks
            .iter()
            .map(|t| {
                Task::new(project_id, t.title.trim(), t.description.trim(), vec![], now)
                    .with_role(t.role.into())
            })
            .collect();
        let ids: Vec<_> = tasks.iter().map(|t| t.id).collect();
        for (task, planned) in tasks.iter_mut().zip(&self.tasks) {
            let mut deps: Vec<_> = planned.depends_on.iter().map(|&d| ids[d]).collect();
            deps.sort();
            deps.dedup();
            task.depends_on = deps;
        }
        validate_plan(&tasks, limits).map_err(|e| format!("the plan is invalid: {e}"))?;
        Ok(tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planned(title: &str, role: PlanRole, depends_on: Vec<usize>) -> PlannedTask {
        PlannedTask { title: title.into(), description: format!("Do {title}"), role, depends_on }
    }

    fn now() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }

    #[test]
    fn plans_become_linked_tasks() {
        let plan = Plan {
            tasks: vec![
                planned("Spec", PlanRole::Spec, vec![]),
                planned("Code", PlanRole::Coder, vec![0]),
                planned("Tests", PlanRole::Coder, vec![0, 1, 1]),
            ],
        };
        let tasks = plan.into_tasks(ProjectId::new(), &Limits::default(), now()).unwrap();
        assert_eq!(tasks[1].depends_on, vec![tasks[0].id]);
        assert_eq!(tasks[2].depends_on.len(), 2, "duplicate dependencies are merged");
        assert_eq!(tasks[1].role, Some(AgentRole::Coder));
    }

    #[test]
    fn bad_dependencies_are_explained() {
        let plan = Plan { tasks: vec![planned("A", PlanRole::Coder, vec![1])] };
        let err = plan.into_tasks(ProjectId::new(), &Limits::default(), now()).unwrap_err();
        assert!(err.contains("depends on 1"), "{err}");
    }

    #[test]
    fn cycles_are_rejected() {
        let plan = Plan {
            tasks: vec![planned("A", PlanRole::Coder, vec![1]), planned("B", PlanRole::Coder, vec![0])],
        };
        let err = plan.into_tasks(ProjectId::new(), &Limits::default(), now()).unwrap_err();
        assert!(err.contains("cycle"), "{err}");
    }

    #[test]
    fn oversized_plans_are_rejected() {
        let limits = Limits { max_tasks: 2, ..Limits::default() };
        let plan =
            Plan { tasks: (0..3).map(|i| planned(&format!("T{i}"), PlanRole::Coder, vec![])).collect() };
        let err = plan.into_tasks(ProjectId::new(), &limits, now()).unwrap_err();
        assert!(err.contains("exceeds the limit"), "{err}");
    }
}
