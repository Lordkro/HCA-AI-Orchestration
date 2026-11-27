//! Integration tests against a real SQLite file.

use std::collections::HashSet;

use chrono::{DateTime, Duration, Utc};
use hca_core::{
    AgentRole, FailureReason, Limits, PlanError, Project, ProjectEvent, ProjectFailure, ProjectStatus,
    Reservation, Task, TaskEvent, TaskState, TransitionError,
};
use hca_store::{JobStatus, NewJob, Store, StoreError};
use serde_json::json;
use tempfile::TempDir;

fn t0() -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).unwrap()
}

async fn store() -> (Store, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("hca.db")).await.unwrap();
    (store, dir)
}

async fn project_with_plan(store: &Store, deps: &[&[usize]]) -> (Project, Vec<Task>) {
    let project = Project::new("Fibonacci", "Print the Fibonacci sequence", t0());
    store.create_project(&project).await.unwrap();
    let mut tasks: Vec<Task> = (0..deps.len())
        .map(|i| {
            let at = t0() + Duration::seconds(i64::try_from(i).unwrap());
            Task::new(project.id, format!("task {i}"), "do it", vec![], at)
        })
        .collect();
    let ids: Vec<_> = tasks.iter().map(|t| t.id).collect();
    for (task, deps) in tasks.iter_mut().zip(deps) {
        task.depends_on = deps.iter().map(|&d| ids[d]).collect();
    }
    store.create_plan(project.id, &tasks, &Limits::default()).await.unwrap();
    (project, tasks)
}

#[tokio::test]
async fn reopening_keeps_data_and_reruns_no_migrations() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hca.db");
    let project = Project::new("p", "idea", t0());
    {
        let store = Store::open(&path).await.unwrap();
        store.create_project(&project).await.unwrap();
        store.close().await;
    }
    let store = Store::open(&path).await.unwrap();
    assert_eq!(store.project(project.id).await.unwrap(), project);
}

#[tokio::test]
async fn plans_round_trip_in_order() {
    let (store, _dir) = store().await;
    let (project, tasks) = project_with_plan(&store, &[&[], &[0], &[0, 1]]).await;
    assert_eq!(store.tasks(project.id).await.unwrap(), tasks);
    assert_eq!(store.projects(Some(ProjectStatus::Active)).await.unwrap(), vec![project]);
}

#[tokio::test]
async fn invalid_plans_write_nothing() {
    let (store, _dir) = store().await;
    let project = Project::new("p", "idea", t0());
    store.create_project(&project).await.unwrap();

    let mut a = Task::new(project.id, "a", "", vec![], t0());
    let mut b = Task::new(project.id, "b", "", vec![], t0());
    a.depends_on = vec![b.id];
    b.depends_on = vec![a.id];
    let err = store.create_plan(project.id, &[a, b], &Limits::default()).await.unwrap_err();
    assert!(matches!(err, StoreError::Plan(PlanError::Cycle(_))), "{err:?}");
    assert_eq!(store.tasks(project.id).await.unwrap().len(), 0);

    let stranger = Task::new(hca_core::ProjectId::new(), "c", "", vec![], t0());
    let err = store.create_plan(project.id, &[stranger], &Limits::default()).await.unwrap_err();
    assert!(matches!(err, StoreError::Plan(PlanError::MixedProjects(_))), "{err:?}");
}

#[tokio::test]
async fn a_project_gets_one_plan() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let again = Task::new(project.id, "again", "", vec![], t0());
    let err = store.create_plan(project.id, &[again], &Limits::default()).await.unwrap_err();
    assert!(matches!(err, StoreError::PlanExists), "{err:?}");
}

#[tokio::test]
async fn task_events_are_saved_with_history() {
    let (store, _dir) = store().await;
    let limits = Limits::default();
    let (_, tasks) = project_with_plan(&store, &[&[]]).await;
    let id = tasks[0].id;

    let events = [
        TaskEvent::Assign { to: AgentRole::Coder },
        TaskEvent::Start,
        TaskEvent::Submit,
        TaskEvent::RequestRevision { feedback: "add tests".into() },
        TaskEvent::Start,
        TaskEvent::Submit,
        TaskEvent::Approve,
    ];
    for (i, event) in events.iter().enumerate() {
        let at = t0() + Duration::minutes(i64::try_from(i).unwrap());
        store.apply_task_event(id, event.clone(), &limits, at).await.unwrap();
    }

    let task = store.task(id).await.unwrap();
    assert_eq!(task.state(), TaskState::Done);
    assert_eq!(task.revisions(), 1);

    let history = store.task_history(id).await.unwrap();
    assert_eq!(history.len(), events.len());
    assert_eq!(history.iter().map(|h| h.event.clone()).collect::<Vec<_>>(), events);
    assert_eq!(history[3].from, TaskState::Review);
    assert_eq!(history[3].to, TaskState::Revision);
    assert!(history.windows(2).all(|w| w[0].seq < w[1].seq));
}

#[tokio::test]
async fn refused_events_change_nothing() {
    let (store, _dir) = store().await;
    let limits = Limits::default();
    let (_, tasks) = project_with_plan(&store, &[&[]]).await;
    let id = tasks[0].id;

    let err = store.apply_task_event(id, TaskEvent::Approve, &limits, t0()).await.unwrap_err();
    assert!(matches!(err, StoreError::Task(TransitionError::InvalidTransition { .. })), "{err:?}");
    assert_eq!(store.task(id).await.unwrap(), tasks[0]);
    assert_eq!(store.task_history(id).await.unwrap().len(), 0);
}

#[tokio::test]
async fn paused_projects_refuse_work_but_allow_stopping_it() {
    let (store, _dir) = store().await;
    let limits = Limits::default();
    let (project, tasks) = project_with_plan(&store, &[&[]]).await;
    store.apply_project_event(project.id, ProjectEvent::Pause, t0()).await.unwrap();

    let err = store
        .apply_task_event(tasks[0].id, TaskEvent::Assign { to: AgentRole::Coder }, &limits, t0())
        .await
        .unwrap_err();
    assert!(matches!(err, StoreError::ProjectNotActive(ProjectStatus::Paused)), "{err:?}");

    store
        .apply_task_event(tasks[0].id, TaskEvent::Fail { reason: FailureReason::Cancelled }, &limits, t0())
        .await
        .unwrap();
}

#[tokio::test]
async fn jobs_of_paused_projects_wait_and_resume() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let job = store.enqueue(&NewJob::new(project.id, "plan", json!({}), t0()), t0()).await.unwrap();

    store.apply_project_event(project.id, ProjectEvent::Pause, t0()).await.unwrap();
    assert_eq!(store.claim("w1", &[], Duration::minutes(5), t0()).await.unwrap(), None);
    assert_eq!(store.job(job).await.unwrap().status, JobStatus::Queued, "paused work is kept, not dropped");

    store.apply_project_event(project.id, ProjectEvent::Resume, t0()).await.unwrap();
    let claimed = store.claim("w1", &[], Duration::minutes(5), t0()).await.unwrap().unwrap();
    assert_eq!(claimed.id, job);
}

#[tokio::test]
async fn finishing_a_project_cancels_its_jobs() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let queued = store.enqueue(&NewJob::new(project.id, "a", json!({}), t0()), t0()).await.unwrap();
    let running = store.enqueue(&NewJob::new(project.id, "b", json!({}), t0()), t0()).await.unwrap();
    store.claim("w1", &["b"], Duration::minutes(5), t0()).await.unwrap().unwrap();

    store.apply_project_event(project.id, ProjectEvent::Cancel, t0()).await.unwrap();

    assert_eq!(store.job(queued).await.unwrap().status, JobStatus::Cancelled);
    assert_eq!(store.job(running).await.unwrap().status, JobStatus::Cancelled);
    let err = store.complete(running, "w1", t0()).await.unwrap_err();
    assert!(matches!(err, StoreError::LeaseLost(_)), "a cancelled job's result is discarded");
    let err = store.enqueue(&NewJob::new(project.id, "c", json!({}), t0()), t0()).await.unwrap_err();
    assert!(matches!(err, StoreError::ProjectNotActive(ProjectStatus::Cancelled)));
}

#[tokio::test]
async fn spending_the_budget_fails_the_project_and_stops_its_jobs() {
    let (store, _dir) = store().await;
    let limits = Limits { project_token_budget: 1_000, ..Limits::default() };
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let job = store.enqueue(&NewJob::new(project.id, "code", json!({}), t0()), t0()).await.unwrap();

    assert!(store.charge_tokens(project.id, 600, &limits, t0()).await.unwrap());
    assert!(!store.charge_tokens(project.id, 600, &limits, t0()).await.unwrap());

    let stored = store.project(project.id).await.unwrap();
    assert_eq!(stored.status(), ProjectStatus::Failed);
    assert_eq!(stored.failure(), Some(&ProjectFailure::BudgetExhausted));
    assert_eq!(stored.tokens_used(), 1_200);
    assert_eq!(store.job(job).await.unwrap().status, JobStatus::Cancelled);
}

#[tokio::test]
async fn claims_respect_kind_and_schedule() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let later = NewJob {
        run_after: t0() + Duration::minutes(10),
        ..NewJob::new(project.id, "code", json!({}), t0())
    };
    let later = store.enqueue(&later, t0()).await.unwrap();
    let review =
        store.enqueue(&NewJob::new(project.id, "review", json!({"n": 1}), t0()), t0()).await.unwrap();

    assert_eq!(store.claim("w", &["code"], Duration::minutes(5), t0()).await.unwrap(), None);
    let job = store.claim("w", &["review", "code"], Duration::minutes(5), t0()).await.unwrap().unwrap();
    assert_eq!(job.id, review);
    assert_eq!(job.payload, json!({"n": 1}));
    assert_eq!(job.attempts, 1);

    let job = store.claim("w", &["code"], Duration::minutes(5), t0() + Duration::minutes(10)).await.unwrap();
    assert_eq!(job.unwrap().id, later);
}

#[tokio::test]
async fn an_expired_lease_moves_the_job_to_another_worker() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let id = store.enqueue(&NewJob::new(project.id, "code", json!({}), t0()), t0()).await.unwrap();

    store.claim("crashed", &[], Duration::minutes(5), t0()).await.unwrap().unwrap();
    assert_eq!(
        store.claim("other", &[], Duration::minutes(5), t0() + Duration::minutes(4)).await.unwrap(),
        None
    );

    let job =
        store.claim("other", &[], Duration::minutes(5), t0() + Duration::minutes(6)).await.unwrap().unwrap();
    assert_eq!(job.id, id);
    assert_eq!(job.attempts, 2);
    assert_eq!(job.locked_by.as_deref(), Some("other"));

    let err = store.complete(id, "crashed", t0() + Duration::minutes(7)).await.unwrap_err();
    assert!(matches!(err, StoreError::LeaseLost(_)), "the old worker can't finish it");
    store.complete(id, "other", t0() + Duration::minutes(7)).await.unwrap();
    assert_eq!(store.job(id).await.unwrap().status, JobStatus::Done);
}

#[tokio::test]
async fn attempts_are_bounded_then_the_job_is_dead() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let id =
        store.enqueue(&NewJob::new(project.id, "code", json!({}), t0()).max_attempts(2), t0()).await.unwrap();

    let mut now = t0();
    store.claim("w", &[], Duration::minutes(5), now).await.unwrap().unwrap();
    let status = store.fail(id, "w", "model timed out", Duration::seconds(30), now).await.unwrap();
    assert_eq!(status, JobStatus::Queued);

    assert_eq!(store.claim("w", &[], Duration::minutes(5), now).await.unwrap(), None, "backoff applies");
    now += Duration::seconds(30);
    store.claim("w", &[], Duration::minutes(5), now).await.unwrap().unwrap();
    let status = store.fail(id, "w", "model timed out again", Duration::seconds(30), now).await.unwrap();
    assert_eq!(status, JobStatus::Dead);

    now += Duration::hours(1);
    assert_eq!(store.claim("w", &[], Duration::minutes(5), now).await.unwrap(), None);
    let job = store.job(id).await.unwrap();
    assert_eq!(job.last_error.as_deref(), Some("model timed out again"));
}

#[tokio::test]
async fn releasing_a_job_gives_its_attempt_back() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let id =
        store.enqueue(&NewJob::new(project.id, "code", json!({}), t0()).max_attempts(1), t0()).await.unwrap();
    // Paused and resumed three times mid-job: none of it counts as a failure.
    for _ in 0..3 {
        let job = store.claim("w", &[], Duration::minutes(5), t0()).await.unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Running);
        assert_eq!(job.attempts, 1);
        store.release(id, "w", t0()).await.unwrap();
        assert_eq!(store.job(id).await.unwrap().attempts, 0);
    }
    let job = store.claim("w", &[], Duration::minutes(5), t0()).await.unwrap().unwrap();
    assert_eq!(job.status, JobStatus::Running, "the job still has its one attempt");
}

#[tokio::test]
async fn orphaned_jobs_are_requeued_at_startup_and_keep_their_attempt() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let id =
        store.enqueue(&NewJob::new(project.id, "code", json!({}), t0()).max_attempts(2), t0()).await.unwrap();
    store.claim("old-process", &[], Duration::minutes(15), t0()).await.unwrap().unwrap();

    assert_eq!(store.requeue_orphans(t0()).await.unwrap(), 1);
    let job = store.claim("new-process", &[], Duration::minutes(15), t0()).await.unwrap().unwrap();
    assert_eq!(job.id, id, "no need to wait 15 minutes for the old lease");
    assert_eq!(job.attempts, 2, "the crashed attempt still counts");
    assert_eq!(store.open_jobs(project.id).await.unwrap().len(), 1);
}

#[tokio::test]
async fn jobs_interrupted_at_shutdown_are_requeued_with_their_attempt_back() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let id =
        store.enqueue(&NewJob::new(project.id, "code", json!({}), t0()).max_attempts(1), t0()).await.unwrap();
    store.claim("stopping", &[], Duration::minutes(15), t0()).await.unwrap().unwrap();

    assert_eq!(store.requeue_interrupted(t0()).await.unwrap(), 1);
    let job = store.job(id).await.unwrap();
    assert_eq!((job.status, job.attempts), (JobStatus::Queued, 0));
    let job = store.claim("next-start", &[], Duration::minutes(15), t0()).await.unwrap().unwrap();
    assert_eq!(job.id, id, "its only attempt is still there");
}

#[tokio::test]
async fn a_lease_expiring_on_the_last_attempt_kills_the_job() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let id =
        store.enqueue(&NewJob::new(project.id, "code", json!({}), t0()).max_attempts(1), t0()).await.unwrap();
    store.claim("crashed", &[], Duration::minutes(5), t0()).await.unwrap().unwrap();

    let later = t0() + Duration::minutes(10);
    let job = store.claim("other", &[], Duration::minutes(5), later).await.unwrap().unwrap();
    assert_eq!(job.id, id);
    assert_eq!(job.status, JobStatus::Dead);
    assert_eq!(job.last_error.as_deref(), Some("lease expired on the last attempt"));
    let stored = store.job(id).await.unwrap();
    assert_eq!(stored.status, JobStatus::Dead);
    assert_eq!(stored.last_error.as_deref(), Some("lease expired on the last attempt"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_workers_never_claim_the_same_job() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    let mut expected = HashSet::new();
    for i in 0..60 {
        expected.insert(
            store.enqueue(&NewJob::new(project.id, "code", json!({ "i": i }), t0()), t0()).await.unwrap(),
        );
    }

    let mut workers = Vec::new();
    for w in 0..8 {
        let store = store.clone();
        workers.push(tokio::spawn(async move {
            let name = format!("worker-{w}");
            let mut mine = Vec::new();
            while let Some(job) = store.claim(&name, &[], Duration::minutes(5), t0()).await.unwrap() {
                store.complete(job.id, &name, t0()).await.unwrap();
                mine.push(job.id);
            }
            mine
        }));
    }

    let mut seen = HashSet::new();
    for worker in workers {
        for id in worker.await.unwrap() {
            assert!(seen.insert(id), "job {id} was claimed twice");
        }
    }
    assert_eq!(seen, expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_task_events_are_all_recorded() {
    let (store, _dir) = store().await;
    let limits = Limits::default();
    let deps: Vec<&[usize]> = vec![&[]; 20];
    let (_, tasks) = project_with_plan(&store, &deps).await;

    let mut handles = Vec::new();
    for task in &tasks {
        let store = store.clone();
        let id = task.id;
        handles.push(tokio::spawn(async move {
            for event in [
                TaskEvent::Assign { to: AgentRole::Coder },
                TaskEvent::Start,
                TaskEvent::Submit,
                TaskEvent::Approve,
            ] {
                store.apply_task_event(id, event, &limits, t0()).await.unwrap();
            }
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }

    for task in &tasks {
        assert_eq!(store.task(task.id).await.unwrap().state(), TaskState::Done);
        assert_eq!(store.task_history(task.id).await.unwrap().len(), 4);
    }
}

#[tokio::test]
async fn a_transition_and_its_next_job_commit_together() {
    let (store, _dir) = store().await;
    let limits = Limits::default();
    let (project, tasks) = project_with_plan(&store, &[&[]]).await;
    let id = tasks[0].id;
    let work = |_: &hca_core::Transition, t: &Task| {
        Some(NewJob::new(t.project_id, "work", json!({}), t0()).for_task(t.id))
    };

    let (transition, job) = store
        .apply_task_event_with_job(id, TaskEvent::Assign { to: AgentRole::Coder }, &limits, t0(), work)
        .await
        .unwrap();
    assert_eq!(transition.to, TaskState::Assigned);
    let job = store.job(job.unwrap()).await.unwrap();
    assert_eq!((job.kind.as_str(), job.task_id), ("work", Some(id)));

    // A refused transition queues nothing.
    let err = store
        .apply_task_event_with_job(id, TaskEvent::Assign { to: AgentRole::Coder }, &limits, t0(), work)
        .await
        .unwrap_err();
    assert!(matches!(err, StoreError::Task(_)));
    assert_eq!(store.jobs(project.id, None).await.unwrap().len(), 1);
}

#[tokio::test]
async fn activity_is_logged_in_order() {
    let (store, _dir) = store().await;
    let (project, tasks) = project_with_plan(&store, &[&[]]).await;
    let entry = |kind: &str, content: &str| hca_store::NewActivity {
        project_id: project.id,
        task_id: Some(tasks[0].id),
        sender: "coder".into(),
        recipient: "critic".into(),
        kind: kind.into(),
        content: content.into(),
        at: t0(),
    };
    store.log_activity(&entry("deliverable", "first")).await.unwrap();
    store.log_activity(&entry("feedback", "fix it")).await.unwrap();
    store.log_activity(&entry("deliverable", "second")).await.unwrap();

    let log = store.activity(project.id).await.unwrap();
    assert_eq!(
        log.iter().map(|a| a.entry.content.as_str()).collect::<Vec<_>>(),
        ["first", "fix it", "second"]
    );
    let latest = store.latest_activity(tasks[0].id, "deliverable").await.unwrap().unwrap();
    assert_eq!(latest.entry.content, "second");
    assert!(store.latest_activity(tasks[0].id, "plan").await.unwrap().is_none());
}

#[tokio::test]
async fn reservations_are_saved_settled_and_charged_after_a_crash() {
    let (store, _dir) = store().await;
    let (project, tasks) = project_with_plan(&store, &[&[]]).await;
    let limits = Limits { project_token_budget: 1_000, ..Limits::default() };

    assert_eq!(store.reserve_usage(project.id, 600, &limits).await.unwrap(), Reservation::Granted);
    assert_eq!(store.reserve_usage(project.id, 600, &limits).await.unwrap(), Reservation::Busy);
    assert!(store.settle_usage(project.id, Some(tasks[0].id), 600, 200, &limits, t0()).await.unwrap());
    let saved = store.project(project.id).await.unwrap();
    assert_eq!((saved.tokens_used(), saved.tokens_reserved()), (200, 0));
    assert_eq!(store.task(tasks[0].id).await.unwrap().tokens_used(), 200);

    // The process dies mid-call: the reservation stays saved and counts in
    // full at the next start.
    store.reserve_usage(project.id, 700, &limits).await.unwrap();
    assert_eq!(store.charge_open_reservations(&limits, t0()).await.unwrap(), 1);
    let saved = store.project(project.id).await.unwrap();
    assert_eq!((saved.tokens_used(), saved.tokens_reserved()), (900, 0));
    assert_eq!(saved.status(), ProjectStatus::Active);
}

#[tokio::test]
async fn a_call_that_can_never_fit_ends_the_project_as_out_of_budget() {
    let (store, _dir) = store().await;
    let (project, _) = project_with_plan(&store, &[&[]]).await;
    store.enqueue(&NewJob::new(project.id, "code", json!({}), t0()), t0()).await.unwrap();
    let limits = Limits { project_token_budget: 1_000, ..Limits::default() };

    let refused = store.reserve_usage(project.id, 1_500, &limits).await.unwrap();
    assert!(matches!(refused, Reservation::Refused(_)), "{refused:?}");
    assert_eq!(
        store.project(project.id).await.unwrap().status(),
        ProjectStatus::Active,
        "a smaller call may fit"
    );
    assert!(store.exhaust_budget(project.id, t0()).await.unwrap());
    assert!(!store.exhaust_budget(project.id, t0()).await.unwrap(), "only once");
    let saved = store.project(project.id).await.unwrap();
    assert_eq!(saved.failure(), Some(&ProjectFailure::BudgetExhausted));
    assert!(store.open_jobs(project.id).await.unwrap().is_empty(), "its jobs are cancelled");
    let paused = store.reserve_usage(project.id, 1, &limits).await.unwrap_err();
    assert!(matches!(paused, StoreError::ProjectNotActive(ProjectStatus::Failed)), "{paused:?}");
}
