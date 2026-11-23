//! Property tests: whatever the agents and reviewers do, work always ends.
//!
//! A reviewer that never approves, or an orchestrator that keeps retrying,
//! must not be able to keep a task running and spending tokens forever.
//! These tests pin down the guarantees that make that impossible.

use chrono::{DateTime, Duration, Utc};
use hca_core::{
    assess, ready_tasks, validate_plan, AgentRole, FailureReason, Limits, ProjectId, ProjectProgress, Task,
    TaskEvent, TaskId, TaskState,
};
use proptest::prelude::*;

fn at(step: u64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).unwrap() + Duration::seconds(i64::try_from(step).unwrap())
}

fn limits_strategy() -> impl Strategy<Value = Limits> {
    (0u32..6, 0u32..4, 1usize..5).prop_map(|(max_revisions, max_retries, max_parallel_tasks)| Limits {
        max_revisions,
        max_retries,
        max_parallel_tasks,
        ..Limits::default()
    })
}

fn event_strategy() -> impl Strategy<Value = TaskEvent> {
    prop_oneof![
        Just(TaskEvent::Assign { to: AgentRole::Coder }),
        Just(TaskEvent::Assign { to: AgentRole::Critic }),
        Just(TaskEvent::Start),
        Just(TaskEvent::Submit),
        Just(TaskEvent::Approve),
        Just(TaskEvent::RequestRevision { feedback: "add tests".into() }),
        Just(TaskEvent::RequestRevision { feedback: String::new() }),
        Just(TaskEvent::Fail { reason: FailureReason::Timeout }),
        Just(TaskEvent::Fail { reason: FailureReason::Cancelled }),
        Just(TaskEvent::Retry),
    ]
}

proptest! {
    /// Any sequence of events, in any order: the counters never pass their
    /// limits and the task never accepts more than `Task::max_events`.
    #[test]
    fn a_task_accepts_a_bounded_number_of_events(
        limits in limits_strategy(),
        events in prop::collection::vec(event_strategy(), 0..500),
    ) {
        let mut task = Task::new(ProjectId::new(), "t", "", vec![], at(0));
        let mut accepted = 0u64;
        for (step, event) in (1u64..).zip(events) {
            let before = task.clone();
            if task.apply(event, &limits, at(step)).is_ok() {
                accepted += 1;
            } else {
                prop_assert_eq!(&task, &before, "a refused event must change nothing");
            }
            prop_assert!(task.revisions() <= limits.max_revisions);
            prop_assert!(task.retries() <= limits.max_retries);
        }
        prop_assert!(accepted <= Task::max_events(&limits));
    }

    /// A random task graph, random reviewer verdicts and random agent
    /// crashes: the project always ends as Completed or Stuck, within a
    /// bounded number of orchestrator steps, and never runs more tasks at
    /// once than allowed.
    #[test]
    fn every_project_terminates(
        limits in limits_strategy(),
        graph in prop::collection::vec(prop::collection::vec(any::<prop::sample::Index>(), 0..3), 1..12),
        verdicts in prop::collection::vec(any::<bool>(), 1..40),
        crashes in prop::collection::vec(prop::bool::weighted(0.15), 1..40),
    ) {
        let project = ProjectId::new();
        let mut tasks: Vec<Task> = Vec::with_capacity(graph.len());
        for (i, deps) in graph.iter().enumerate() {
            let mut task = Task::new(project, format!("t{i}"), "", vec![], at(u64::try_from(i).unwrap()));
            if i > 0 {
                let mut ids: Vec<TaskId> = deps.iter().map(|d| tasks[d.index(i)].id).collect();
                ids.sort();
                ids.dedup();
                task.depends_on = ids;
            }
            tasks.push(task);
        }
        prop_assert_eq!(validate_plan(&tasks, &limits), Ok(()));

        let outcome = run_project(&mut tasks, &limits, &verdicts, &crashes);
        let bound: u64 = tasks.iter().map(|_| Task::max_events(&limits)).sum::<u64>() + 1;
        prop_assert!(outcome.steps <= bound, "took {} steps, bound {}", outcome.steps, bound);
        prop_assert!(outcome.max_in_flight <= limits.max_parallel_tasks);
        match outcome.progress {
            ProjectProgress::Completed => {
                prop_assert!(tasks.iter().all(|t| t.state() == TaskState::Done));
            }
            ProjectProgress::Stuck { exhausted } => {
                prop_assert!(!exhausted.is_empty());
                for id in exhausted {
                    let task = tasks.iter().find(|t| t.id == id).unwrap();
                    prop_assert!(!task.can_retry(&limits));
                }
            }
            other => prop_assert!(false, "project ended in {:?}", other),
        }
    }

    /// With a reviewer that always approves and agents that never crash,
    /// every valid plan completes.
    #[test]
    fn a_cooperative_team_always_completes(
        limits in limits_strategy(),
        graph in prop::collection::vec(prop::collection::vec(any::<prop::sample::Index>(), 0..3), 1..12),
    ) {
        let project = ProjectId::new();
        let mut tasks: Vec<Task> = Vec::new();
        for (i, deps) in graph.iter().enumerate() {
            let mut task = Task::new(project, format!("t{i}"), "", vec![], at(u64::try_from(i).unwrap()));
            if i > 0 {
                let mut ids: Vec<TaskId> = deps.iter().map(|d| tasks[d.index(i)].id).collect();
                ids.sort();
                ids.dedup();
                task.depends_on = ids;
            }
            tasks.push(task);
        }
        let outcome = run_project(&mut tasks, &limits, &[true], &[false]);
        prop_assert_eq!(outcome.progress, ProjectProgress::Completed);
    }
}

/// A reviewer that always asks for a revision. The task must fail after its
/// revisions and retries are used up, instead of looping forever.
#[test]
fn an_endlessly_unhappy_reviewer_cannot_loop_forever() {
    let limits = Limits::default();
    let mut task = Task::new(ProjectId::new(), "Fibonacci script", "", vec![], at(0));
    let mut steps = 0u64;
    loop {
        let event = match task.state() {
            TaskState::Pending => TaskEvent::Assign { to: AgentRole::Coder },
            TaskState::Assigned | TaskState::Revision => TaskEvent::Start,
            TaskState::InProgress => TaskEvent::Submit,
            TaskState::Review => TaskEvent::RequestRevision { feedback: "still wrong".into() },
            TaskState::Failed => TaskEvent::Retry,
            TaskState::Done => unreachable!("the reviewer never approves"),
        };
        if task.apply(event, &limits, at(steps)).is_err() {
            break;
        }
        steps += 1;
        assert!(steps <= Task::max_events(&limits), "looped past the limits");
    }
    assert_eq!(task.state(), TaskState::Failed);
    assert_eq!(task.failure(), Some(&FailureReason::RevisionLimit));
    assert_eq!(task.retries(), limits.max_retries);
    assert!(steps <= Task::max_events(&limits));
}

struct Outcome {
    progress: ProjectProgress,
    steps: u64,
    max_in_flight: usize,
}

/// A minimal deterministic orchestrator, the shape the real one will take.
///
/// Panics if it runs past the proven bound, so a regression fails the test
/// instead of hanging it.
fn run_project(tasks: &mut [Task], limits: &Limits, verdicts: &[bool], crashes: &[bool]) -> Outcome {
    let mut verdicts = verdicts.iter().copied().cycle();
    let mut crashes = crashes.iter().copied().cycle();
    let mut steps = 0u64;
    let mut max_in_flight = 0;
    let hard_cap: u64 = tasks.iter().map(|_| Task::max_events(limits)).sum::<u64>() + 1;

    loop {
        assert!(steps <= hard_cap, "orchestrator ran {steps} steps, past the bound of {hard_cap}");
        let progress = assess(tasks, limits);
        match progress {
            ProjectProgress::Completed | ProjectProgress::Stuck { .. } => {
                return Outcome { progress, steps, max_in_flight };
            }
            ProjectProgress::AwaitingPlan => unreachable!("the plan is never empty here"),
            ProjectProgress::NeedsRetry { retriable } => {
                for id in retriable {
                    let task = tasks.iter_mut().find(|t| t.id == id).unwrap();
                    task.apply(TaskEvent::Retry, limits, at(steps))
                        .expect("assess only lists retriable tasks");
                }
            }
            ProjectProgress::Running { .. } => {
                for id in ready_tasks(tasks, limits) {
                    let task = tasks.iter_mut().find(|t| t.id == id).unwrap();
                    task.apply(TaskEvent::Assign { to: AgentRole::Coder }, limits, at(steps)).unwrap();
                }
                let in_flight = tasks.iter().filter(|t| t.state().is_in_flight()).count();
                max_in_flight = max_in_flight.max(in_flight);

                for task in tasks.iter_mut().filter(|t| t.state().is_in_flight()) {
                    let event = match task.state() {
                        TaskState::Assigned | TaskState::Revision => TaskEvent::Start,
                        TaskState::InProgress if crashes.next().unwrap_or(false) => {
                            TaskEvent::Fail { reason: FailureReason::AgentError("tool crashed".into()) }
                        }
                        TaskState::InProgress => TaskEvent::Submit,
                        TaskState::Review if verdicts.next().unwrap_or(true) => TaskEvent::Approve,
                        TaskState::Review => TaskEvent::RequestRevision { feedback: "fix it".into() },
                        state => unreachable!("{state:?} is not in flight"),
                    };
                    task.apply(event, limits, at(steps)).unwrap();
                }
            }
        }
        steps += 1;
    }
}
