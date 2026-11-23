//! Configurable safety limits.

use serde::{Deserialize, Serialize};

/// Safety limits that bound how much work a project can ever do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Limits {
    /// Revision rounds allowed per attempt at a task. The next revision
    /// request after this fails the task.
    pub max_revisions: u32,
    /// How many times a failed task may be retried. `0` means never.
    pub max_retries: u32,
    /// Maximum number of tasks in one project.
    pub max_tasks: usize,
    /// Maximum number of tasks running at the same time in one project.
    pub max_parallel_tasks: usize,
    /// Token budget for a whole project, prompt and completion combined.
    pub project_token_budget: u64,
    /// Minutes a task may sit in a working state before it is timed out.
    pub task_timeout_minutes: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_revisions: 5,
            max_retries: 1,
            max_tasks: 50,
            max_parallel_tasks: 3,
            project_token_budget: 500_000,
            task_timeout_minutes: 30,
        }
    }
}

/// Returned when a [`Limits`] value would make the system unusable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LimitsError {
    /// A limit that must be at least 1 was 0.
    #[error("{0} must be at least 1")]
    MustBePositive(&'static str),
}

impl Limits {
    /// Checks that every limit is usable.
    ///
    /// # Errors
    ///
    /// Returns [`LimitsError::MustBePositive`] naming the first limit that is
    /// zero but must be at least one. `max_revisions` and `max_retries` may be
    /// zero.
    pub fn validate(&self) -> Result<(), LimitsError> {
        if self.max_tasks == 0 {
            return Err(LimitsError::MustBePositive("max_tasks"));
        }
        if self.max_parallel_tasks == 0 {
            return Err(LimitsError::MustBePositive("max_parallel_tasks"));
        }
        if self.project_token_budget == 0 {
            return Err(LimitsError::MustBePositive("project_token_budget"));
        }
        if self.task_timeout_minutes == 0 {
            return Err(LimitsError::MustBePositive("task_timeout_minutes"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert_eq!(Limits::default().validate(), Ok(()));
    }

    #[test]
    fn rejects_zero_parallelism() {
        let limits = Limits { max_parallel_tasks: 0, ..Limits::default() };
        assert_eq!(limits.validate(), Err(LimitsError::MustBePositive("max_parallel_tasks")));
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        let limits: Limits = serde_json::from_str(r#"{"max_retries": 3}"#).unwrap();
        assert_eq!(limits.max_retries, 3);
        assert_eq!(limits.max_revisions, 5);
    }
}
