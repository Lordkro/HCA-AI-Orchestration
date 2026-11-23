//! Task graphs: validation, scheduling and progress.
//!
//! The orchestrator loop is meant to be:
//!
//! 1. [`validate_plan`] once, after the PM proposes tasks.
//! 2. Repeatedly: [`assess`] the project, then act on the answer.
//!    `Running` → dispatch [`ready_tasks`]; `NeedsRetry` → retry those tasks
//!    once; `Completed` / `Stuck` → finish the project.
//!
//! Because every task accepts a bounded number of events
//! ([`Task::max_events`]) and `NeedsRetry` only lists tasks with retries
//! left, this loop always ends.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{Limits, Task, TaskId, TaskState};

/// Why a proposed set of tasks can't be run.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    /// The plan has no tasks.
    #[error("the plan has no tasks")]
    Empty,
    /// More tasks than [`Limits::max_tasks`].
    #[error("{count} tasks exceeds the limit of {max}")]
    TooManyTasks {
        /// Tasks in the plan.
        count: usize,
        /// The limit.
        max: usize,
    },
    /// Two tasks share an id.
    #[error("task {0} appears twice")]
    DuplicateTask(TaskId),
    /// Tasks from different projects were mixed.
    #[error("task {0} belongs to a different project")]
    MixedProjects(TaskId),
    /// A task depends on itself.
    #[error("task {0} depends on itself")]
    SelfDependency(TaskId),
    /// A task depends on a task that isn't in the plan.
    #[error("task {task} depends on unknown task {missing}")]
    UnknownDependency {
        /// The task with the bad dependency.
        task: TaskId,
        /// The id that wasn't found.
        missing: TaskId,
    },
    /// The dependencies form a cycle, so none of these tasks could ever start.
    #[error("dependency cycle among {} tasks", .0.len())]
    Cycle(Vec<TaskId>),
}

/// What the orchestrator should do next with a project.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectProgress {
    /// No tasks yet: the PM still has to plan.
    AwaitingPlan,
    /// Work is happening or can be dispatched.
    Running {
        /// Tasks that could be dispatched now, ignoring the parallel limit.
        ready: usize,
        /// Tasks currently with an agent.
        in_flight: usize,
    },
    /// Nothing can move until these failed tasks are retried. Each one still
    /// has a retry left.
    NeedsRetry {
        /// Failed tasks that may be retried, oldest first.
        retriable: Vec<TaskId>,
    },
    /// Every task is done.
    Completed,
    /// Nothing can move and no retries are left. The project should fail.
    Stuck {
        /// Failed tasks with no retries left, oldest first.
        exhausted: Vec<TaskId>,
    },
}

/// Checks a proposed plan before any of it runs.
///
/// # Errors
///
/// Returns the first [`PlanError`] found: empty plan, too many tasks,
/// duplicate ids, mixed projects, self or unknown dependencies, or a cycle.
pub fn validate_plan(tasks: &[Task], limits: &Limits) -> Result<(), PlanError> {
    let first = tasks.first().ok_or(PlanError::Empty)?;
    if tasks.len() > limits.max_tasks {
        return Err(PlanError::TooManyTasks { count: tasks.len(), max: limits.max_tasks });
    }

    let mut ids = HashSet::with_capacity(tasks.len());
    for task in tasks {
        if !ids.insert(task.id) {
            return Err(PlanError::DuplicateTask(task.id));
        }
        if task.project_id != first.project_id {
            return Err(PlanError::MixedProjects(task.id));
        }
    }

    for task in tasks {
        for dep in &task.depends_on {
            if *dep == task.id {
                return Err(PlanError::SelfDependency(task.id));
            }
            if !ids.contains(dep) {
                return Err(PlanError::UnknownDependency { task: task.id, missing: *dep });
            }
        }
    }

    // Kahn's algorithm: whatever can't be ordered is on or behind a cycle.
    let mut indegree: HashMap<TaskId, usize> =
        tasks.iter().map(|t| (t.id, unique(&t.depends_on).len())).collect();
    let mut dependents: HashMap<TaskId, Vec<TaskId>> = HashMap::new();
    for task in tasks {
        for dep in unique(&task.depends_on) {
            dependents.entry(dep).or_default().push(task.id);
        }
    }
    let mut queue: VecDeque<TaskId> = indegree.iter().filter(|(_, n)| **n == 0).map(|(id, _)| *id).collect();
    let mut ordered = 0;
    while let Some(id) = queue.pop_front() {
        ordered += 1;
        for next in dependents.get(&id).into_iter().flatten() {
            if let Some(n) = indegree.get_mut(next) {
                *n -= 1;
                if *n == 0 {
                    queue.push_back(*next);
                }
            }
        }
    }
    if ordered < tasks.len() {
        let mut stuck: Vec<TaskId> = indegree.into_iter().filter(|(_, n)| *n > 0).map(|(id, _)| id).collect();
        stuck.sort();
        return Err(PlanError::Cycle(stuck));
    }
    Ok(())
}

/// Pending tasks that can be dispatched now, oldest first, limited to the
/// free parallel slots.
pub fn ready_tasks(tasks: &[Task], limits: &Limits) -> Vec<TaskId> {
    let in_flight = tasks.iter().filter(|t| t.state().is_in_flight()).count();
    let slots = limits.max_parallel_tasks.saturating_sub(in_flight);
    let mut ready = unblocked_pending(tasks);
    ready.truncate(slots);
    ready.into_iter().map(|t| t.id).collect()
}

/// Decides what the orchestrator should do next with a project.
pub fn assess(tasks: &[Task], limits: &Limits) -> ProjectProgress {
    if tasks.is_empty() {
        return ProjectProgress::AwaitingPlan;
    }
    if tasks.iter().all(|t| t.state() == TaskState::Done) {
        return ProjectProgress::Completed;
    }

    let in_flight = tasks.iter().filter(|t| t.state().is_in_flight()).count();
    let ready = unblocked_pending(tasks).len();
    if in_flight > 0 || ready > 0 {
        return ProjectProgress::Running { ready, in_flight };
    }

    let mut failed: Vec<&Task> = tasks.iter().filter(|t| t.state() == TaskState::Failed).collect();
    failed.sort_by_key(|t| (t.created_at(), t.id));
    let (retriable, exhausted): (Vec<&Task>, Vec<&Task>) =
        failed.into_iter().partition(|t| t.can_retry(limits));

    if retriable.is_empty() {
        ProjectProgress::Stuck { exhausted: exhausted.into_iter().map(|t| t.id).collect() }
    } else {
        ProjectProgress::NeedsRetry { retriable: retriable.into_iter().map(|t| t.id).collect() }
    }
}

fn unblocked_pending(tasks: &[Task]) -> Vec<&Task> {
    let done: HashSet<TaskId> = tasks.iter().filter(|t| t.state() == TaskState::Done).map(|t| t.id).collect();
    let mut ready: Vec<&Task> = tasks
        .iter()
        .filter(|t| t.state() == TaskState::Pending && t.depends_on.iter().all(|d| done.contains(d)))
        .collect();
    ready.sort_by_key(|t| (t.created_at(), t.id));
    ready
}

fn unique(ids: &[TaskId]) -> HashSet<TaskId> {
    ids.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Duration, Utc};

    use super::*;
    use crate::{AgentRole, FailureReason, ProjectId, TaskEvent};

    fn at(minute: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap() + Duration::minutes(minute)
    }

    fn plan(project: ProjectId, deps: &[&[usize]]) -> Vec<Task> {
        let mut tasks: Vec<Task> = (0..deps.len())
            .map(|i| Task::new(project, format!("t{i}"), "", vec![], at(i64::try_from(i).unwrap())))
            .collect();
        let ids: Vec<TaskId> = tasks.iter().map(|t| t.id).collect();
        for (task, deps) in tasks.iter_mut().zip(deps) {
            task.depends_on = deps.iter().map(|&d| ids[d]).collect();
        }
        tasks
    }

    fn finish(task: &mut Task, limits: &Limits) {
        for event in [
            TaskEvent::Assign { to: AgentRole::Coder },
            TaskEvent::Start,
            TaskEvent::Submit,
            TaskEvent::Approve,
        ] {
            task.apply(event, limits, at(100)).unwrap();
        }
    }

    #[test]
    fn accepts_a_valid_dag() {
        let tasks = plan(ProjectId::new(), &[&[], &[0], &[0], &[1, 2]]);
        assert_eq!(validate_plan(&tasks, &Limits::default()), Ok(()));
    }

    #[test]
    fn rejects_cycles() {
        let tasks = plan(ProjectId::new(), &[&[], &[2], &[1]]);
        let err = validate_plan(&tasks, &Limits::default()).unwrap_err();
        let PlanError::Cycle(ids) = err else { panic!("expected a cycle, got {err:?}") };
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn rejects_unknown_and_self_dependencies() {
        let limits = Limits::default();
        let mut tasks = plan(ProjectId::new(), &[&[]]);
        let id = tasks[0].id;
        tasks[0].depends_on = vec![id];
        assert_eq!(validate_plan(&tasks, &limits), Err(PlanError::SelfDependency(id)));

        let stranger = TaskId::new();
        tasks[0].depends_on = vec![stranger];
        assert_eq!(
            validate_plan(&tasks, &limits),
            Err(PlanError::UnknownDependency { task: id, missing: stranger })
        );
    }

    #[test]
    fn rejects_oversized_and_empty_plans() {
        let limits = Limits { max_tasks: 2, ..Limits::default() };
        let tasks = plan(ProjectId::new(), &[&[], &[], &[]]);
        assert_eq!(validate_plan(&tasks, &limits), Err(PlanError::TooManyTasks { count: 3, max: 2 }));
        assert_eq!(validate_plan(&[], &limits), Err(PlanError::Empty));
    }

    #[test]
    fn rejects_mixed_projects() {
        let mut tasks = plan(ProjectId::new(), &[&[], &[]]);
        tasks[1].project_id = ProjectId::new();
        let id = tasks[1].id;
        assert_eq!(validate_plan(&tasks, &Limits::default()), Err(PlanError::MixedProjects(id)));
    }

    #[test]
    fn dispatches_in_dependency_order_within_slots() {
        let limits = Limits { max_parallel_tasks: 2, ..Limits::default() };
        let mut tasks = plan(ProjectId::new(), &[&[], &[], &[], &[0]]);
        let ids: Vec<TaskId> = tasks.iter().map(|t| t.id).collect();

        assert_eq!(ready_tasks(&tasks, &limits), vec![ids[0], ids[1]]);

        finish(&mut tasks[0], &limits);
        assert_eq!(ready_tasks(&tasks, &limits), vec![ids[1], ids[2]]);

        tasks[1].apply(TaskEvent::Assign { to: AgentRole::Coder }, &limits, at(101)).unwrap();
        assert_eq!(ready_tasks(&tasks, &limits), vec![ids[2]], "one slot is taken");
    }

    #[test]
    fn assess_reports_each_phase() {
        let limits = Limits { max_retries: 1, ..Limits::default() };
        assert_eq!(assess(&[], &limits), ProjectProgress::AwaitingPlan);

        let mut tasks = plan(ProjectId::new(), &[&[], &[0]]);
        assert_eq!(assess(&tasks, &limits), ProjectProgress::Running { ready: 1, in_flight: 0 });

        tasks[0]
            .apply(TaskEvent::Fail { reason: FailureReason::AgentError("boom".into()) }, &limits, at(5))
            .unwrap();
        let first = tasks[0].id;
        assert_eq!(assess(&tasks, &limits), ProjectProgress::NeedsRetry { retriable: vec![first] });

        tasks[0].apply(TaskEvent::Retry, &limits, at(6)).unwrap();
        tasks[0]
            .apply(TaskEvent::Fail { reason: FailureReason::AgentError("boom".into()) }, &limits, at(7))
            .unwrap();
        assert_eq!(assess(&tasks, &limits), ProjectProgress::Stuck { exhausted: vec![first] });
    }

    #[test]
    fn assess_completes_when_everything_is_done() {
        let limits = Limits::default();
        let mut tasks = plan(ProjectId::new(), &[&[], &[0]]);
        for task in &mut tasks {
            finish(task, &limits);
        }
        assert_eq!(assess(&tasks, &limits), ProjectProgress::Completed);
    }
}
