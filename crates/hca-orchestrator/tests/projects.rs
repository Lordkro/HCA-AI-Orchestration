//! Whole projects, from idea to finish, with scripted models.

use std::collections::HashMap;
use std::sync::Arc;

use hca_core::{AgentRole, Limits, ProjectFailure, ProjectStatus, Task, TaskState};
use hca_llm::testing::ScriptedProvider;
use hca_llm::{ChatResponse, Llm, Message, RoleModel, StopReason, ToolCall, Usage};
use hca_orchestrator::{Orchestrator, OrchestratorConfig};
use hca_store::{JobStatus, Store};
use serde_json::json;
use tempfile::TempDir;

struct Team {
    pm: ScriptedProvider,
    spec: ScriptedProvider,
    coder: ScriptedProvider,
    critic: ScriptedProvider,
}

impl Default for Team {
    fn default() -> Self {
        Self {
            pm: ScriptedProvider::new(),
            spec: ScriptedProvider::new(),
            coder: ScriptedProvider::new(),
            critic: ScriptedProvider::new(),
        }
    }
}

struct Setup {
    orchestrator: Orchestrator,
    models: HashMap<AgentRole, Arc<ScriptedProvider>>,
    _dir: TempDir,
}

async fn setup(team: Team, limits: Limits) -> Setup {
    setup_with(team, limits, 1).await
}

async fn setup_with(team: Team, limits: Limits, workers: usize) -> Setup {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("hca.db")).await.unwrap();
    let models: HashMap<AgentRole, Arc<ScriptedProvider>> = HashMap::from([
        (AgentRole::Pm, Arc::new(team.pm)),
        (AgentRole::Spec, Arc::new(team.spec)),
        (AgentRole::Coder, Arc::new(team.coder)),
        (AgentRole::Critic, Arc::new(team.critic)),
    ]);
    let roles = models
        .iter()
        .map(|(role, provider)| {
            let model = RoleModel {
                provider: provider.clone(),
                model: format!("{role}-model"),
                max_tokens: 200,
                temperature: None,
            };
            (*role, model)
        })
        .collect();
    let mut config = OrchestratorConfig::new(dir.path().join("workspaces"));
    config.limits = limits;
    config.workers = workers;
    config.poll_interval = std::time::Duration::from_millis(20);
    let orchestrator = Orchestrator::new(store, Llm::new(roles), None, config);
    Setup { orchestrator, models, _dir: dir }
}

fn plan(tasks: &serde_json::Value) -> ScriptedProvider {
    ScriptedProvider::new().then_tool("submit_plan", json!({ "tasks": tasks }))
}

fn approve(p: ScriptedProvider) -> ScriptedProvider {
    p.then_tool("submit_review", json!({"verdict": "approve"}))
}

fn reject(p: ScriptedProvider, feedback: &str) -> ScriptedProvider {
    p.then_tool("submit_review", json!({"verdict": "request_changes", "feedback": feedback}))
}

fn last_user_text(request: &hca_llm::ChatRequest) -> String {
    request
        .messages
        .iter()
        .find_map(|m| match m {
            Message::User { content } => Some(content.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

async fn tasks_by_title(o: &Orchestrator, project: hca_core::ProjectId) -> HashMap<String, Task> {
    o.store().tasks(project).await.unwrap().into_iter().map(|t| (t.title.clone(), t)).collect()
}

#[tokio::test]
async fn an_idea_goes_from_plan_to_done() {
    let team = Team {
        pm: plan(&json!([
            {"title": "Write the spec", "description": "Specify fib(n) and its tests", "role": "spec"},
            {"title": "Implement fib", "description": "Implement the spec with tests", "role": "coder", "depends_on": [0]}
        ])),
        spec: ScriptedProvider::new()
            .then_tool("write_file", json!({"path": "docs/specs/fib.md", "content": "fib(0)=0, fib(1)=1"}))
            .then_tool("submit", json!({"summary": "Spec in docs/specs/fib.md"})),
        coder: ScriptedProvider::new()
            .then_tool("write_file", json!({"path": "fib.py", "content": "def fib(n): return 1"}))
            .then_tool("submit", json!({"summary": "fib.py written"}))
            .then_tool("write_file", json!({"path": "fib.py", "content": "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"}))
            .then_tool("submit", json!({"summary": "Fixed fib(0)"})),
        critic: reject(approve(ScriptedProvider::new()), "fib.py: fib(0) returns 1; it must return 0.")
            .then_tool("submit_review", json!({"verdict": "approve", "feedback": "Good."})),
    };
    let s = setup(team, Limits::default()).await;
    let o = &s.orchestrator;

    let project = o.create_project("Fibonacci", "A Python fib function with tests").await.unwrap();
    o.run_until_idle("w").await.unwrap();

    let project = o.store().project(project.id).await.unwrap();
    assert_eq!(project.status(), ProjectStatus::Completed);
    let tasks = tasks_by_title(o, project.id).await;
    assert!(tasks.values().all(|t| t.state() == TaskState::Done));
    assert_eq!(tasks["Implement fib"].revisions(), 1);
    assert_eq!(tasks["Implement fib"].assignee(), Some(AgentRole::Coder));
    assert_eq!(tasks["Write the spec"].assignee(), Some(AgentRole::Spec));

    let code = std::fs::read_to_string(o.workspace_path(project.id).join("fib.py")).unwrap();
    assert!(code.contains("fib(n-1)"));

    // Every scripted call costs 20 tokens: 1 plan + 2 spec + 4 coder + 3 critic.
    assert_eq!(project.tokens_used(), 10 * 20);

    // The coder's first brief carries the spec's output; its second carries the feedback.
    let coder = s.models[&AgentRole::Coder].requests();
    assert!(last_user_text(&coder[0]).contains("Output of the earlier task 'Write the spec'"));
    let revision_brief = last_user_text(&coder[2]);
    assert!(revision_brief.contains("Reviewer feedback you must address"));
    assert!(revision_brief.contains("fib(0) returns 1"));

    // The second review checks that the feedback was addressed.
    let critic = s.models[&AgentRole::Critic].requests();
    assert!(last_user_text(&critic[2]).contains("Your previous feedback"));

    let log: Vec<String> =
        o.store().activity(project.id).await.unwrap().into_iter().map(|a| a.entry.kind).collect();
    assert!(log.contains(&"task_assignment".to_string()));
    assert!(log.contains(&"deliverable".to_string()));
    assert!(log.contains(&"feedback".to_string()));
    assert_eq!(log.last().unwrap(), "status_update");

    let jobs = o.store().jobs(project.id, None).await.unwrap();
    assert!(jobs.iter().all(|j| j.status == JobStatus::Done), "{jobs:?}");
}

#[tokio::test]
async fn an_endlessly_unhappy_critic_cannot_burn_the_budget() {
    // The task fails after its revisions and retry, and the project stops
    // instead of looping and spending tokens forever.
    let mut coder = ScriptedProvider::new();
    let mut critic = ScriptedProvider::new();
    for _ in 0..50 {
        coder = coder.then_tool("submit", json!({"summary": "Done, I think."}));
        critic = reject(critic, "Still wrong.");
    }
    let team = Team {
        pm: plan(
            &json!([{"title": "Fibonacci script", "description": "Print fib numbers", "role": "coder"}]),
        ),
        coder,
        critic,
        ..Team::default()
    };
    let limits = Limits { max_revisions: 2, max_retries: 1, ..Limits::default() };
    let s = setup(team, limits).await;
    let o = &s.orchestrator;

    let project = o.create_project("Fibonacci", "Print the Fibonacci sequence").await.unwrap();
    o.run_until_idle("w").await.unwrap();

    let project = o.store().project(project.id).await.unwrap();
    assert_eq!(project.status(), ProjectStatus::Failed);
    assert_eq!(project.failure(), Some(&ProjectFailure::Stuck));
    // Two attempts, each with the first try plus two revisions.
    assert_eq!(s.models[&AgentRole::Coder].requests().len(), 6);
    assert_eq!(s.models[&AgentRole::Critic].requests().len(), 6);
    assert_eq!(project.tokens_used(), 13 * 20);
}

#[tokio::test]
async fn spending_the_budget_stops_the_project_mid_run() {
    let expensive_plan = Ok(ChatResponse {
        content: String::new(),
        tool_calls: vec![ToolCall {
            id: "1".into(),
            name: "submit_plan".into(),
            arguments: json!({"tasks": [{"title": "T", "description": "D", "role": "coder"}]}),
        }],
        stop: StopReason::ToolUse,
        usage: Usage { input_tokens: 60_000, output_tokens: 0 },
    });
    let team = Team { pm: ScriptedProvider::new().then(expensive_plan), ..Team::default() };
    let limits = Limits { project_token_budget: 50_000, ..Limits::default() };
    let s = setup(team, limits).await;
    let o = &s.orchestrator;

    let project = o.create_project("p", "idea").await.unwrap();
    o.run_until_idle("w").await.unwrap();

    let project = o.store().project(project.id).await.unwrap();
    assert_eq!(project.status(), ProjectStatus::Failed);
    assert_eq!(project.failure(), Some(&ProjectFailure::BudgetExhausted));
    assert!(
        o.store().tasks(project.id).await.unwrap().is_empty(),
        "no plan is saved after the budget ran out"
    );
    assert_eq!(s.models[&AgentRole::Coder].requests().len(), 0);
}

#[tokio::test]
async fn paused_projects_wait_and_resume() {
    let team = Team {
        pm: plan(&json!([{"title": "T", "description": "Do it", "role": "coder"}])),
        coder: ScriptedProvider::new().then_tool("submit", json!({"summary": "done"})),
        critic: approve(ScriptedProvider::new()),
        ..Team::default()
    };
    let s = setup(team, Limits::default()).await;
    let o = &s.orchestrator;

    let project = o.create_project("p", "idea").await.unwrap();
    o.pause(project.id).await.unwrap();
    assert_eq!(o.run_until_idle("w").await.unwrap(), 0, "nothing runs while paused");

    o.resume(project.id).await.unwrap();
    o.run_until_idle("w").await.unwrap();
    assert_eq!(o.store().project(project.id).await.unwrap().status(), ProjectStatus::Completed);
}

#[tokio::test]
async fn a_person_can_approve_instead_of_the_critic() {
    let team = Team {
        pm: plan(&json!([{"title": "T", "description": "Do it", "role": "coder"}])),
        coder: ScriptedProvider::new().then_tool("submit", json!({"summary": "done"})),
        ..Team::default()
    };
    let s = setup(team, Limits::default()).await;
    let o = &s.orchestrator;

    let project = o.create_project("p", "idea").await.unwrap();
    assert!(o.step("w").await.unwrap(), "plan");
    assert!(o.step("w").await.unwrap(), "work");
    let task = o.store().tasks(project.id).await.unwrap().remove(0);
    assert_eq!(task.state(), TaskState::Review);

    o.human_review(task.id, true, "Looks right to me").await.unwrap();
    o.run_until_idle("w").await.unwrap();

    assert_eq!(o.store().project(project.id).await.unwrap().status(), ProjectStatus::Completed);
    assert_eq!(s.models[&AgentRole::Critic].requests().len(), 0, "the queued review saw the task was done");
}

#[tokio::test]
async fn recovery_queues_missing_work_after_a_restart() {
    let team = Team {
        coder: ScriptedProvider::new().then_tool("submit", json!({"summary": "done"})),
        critic: approve(ScriptedProvider::new()),
        ..Team::default()
    };
    let s = setup(team, Limits::default()).await;
    let o = &s.orchestrator;

    // A project whose plan was saved but whose jobs never were, as after a
    // crash at the wrong moment.
    let project = hca_core::Project::new("p", "idea", chrono::Utc::now());
    o.store().create_project(&project).await.unwrap();
    let task = Task::new(project.id, "T", "Do it", vec![], chrono::Utc::now()).with_role(AgentRole::Coder);
    o.store().create_plan(project.id, &[task], &Limits::default()).await.unwrap();
    assert_eq!(o.run_until_idle("w").await.unwrap(), 0);

    o.recover().await.unwrap();
    o.run_until_idle("w").await.unwrap();
    assert_eq!(o.store().project(project.id).await.unwrap().status(), ProjectStatus::Completed);
}

#[tokio::test]
async fn an_invalid_plan_fails_the_project_with_the_reason() {
    let team = Team {
        pm: ScriptedProvider::new()
            .then_text("I'd rather chat.")
            .then_text("Still chatting.")
            .then_text("Bye."),
        ..Team::default()
    };
    let s = setup(team, Limits::default()).await;
    let o = &s.orchestrator;

    let project = o.create_project("p", "idea").await.unwrap();
    o.run_until_idle("w").await.unwrap();

    let project = o.store().project(project.id).await.unwrap();
    assert_eq!(project.status(), ProjectStatus::Failed);
    let Some(ProjectFailure::InvalidPlan(reason)) = project.failure() else {
        panic!("{:?}", project.failure())
    };
    assert!(reason.contains("without finishing"), "{reason}");
}

#[tokio::test]
async fn progress_reports_counts_and_budget() {
    let team = Team {
        pm: plan(&json!([{"title": "T", "description": "Do it", "role": "coder"}])),
        coder: ScriptedProvider::new().then_tool("submit", json!({"summary": "done"})),
        critic: approve(ScriptedProvider::new()),
        ..Team::default()
    };
    let s = setup(team, Limits::default()).await;
    let o = &s.orchestrator;
    let project = o.create_project("p", "idea").await.unwrap();
    o.run_until_idle("w").await.unwrap();

    let progress = o.progress(project.id).await.unwrap();
    assert_eq!((progress.total_tasks, progress.completed), (1, 1));
    assert!((progress.progress_pct - 100.0).abs() < f64::EPSILON);
    assert_eq!(progress.token_usage.tokens_used, 60);
    assert_eq!(progress.token_usage.budget, 500_000);

    let statuses = o.agent_statuses();
    let coder = statuses.iter().find(|a| a.role == AgentRole::Coder).unwrap();
    assert_eq!((coder.status, coder.stats.llm_calls), ("idle", 1));
    assert_eq!(coder.model, "coder-model");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_workers_do_each_task_once() {
    let mut coder = ScriptedProvider::new();
    let mut critic = ScriptedProvider::new();
    for _ in 0..4 {
        coder = coder.then_tool("submit", json!({"summary": "done"}));
        critic = approve(critic);
    }
    let team = Team {
        pm: plan(&json!([
            {"title": "A", "description": "a", "role": "coder"},
            {"title": "B", "description": "b", "role": "coder"},
            {"title": "C", "description": "c", "role": "coder"},
            {"title": "D", "description": "d", "role": "coder", "depends_on": [0, 1, 2]}
        ])),
        coder,
        critic,
        ..Team::default()
    };
    let s = setup_with(team, Limits { max_parallel_tasks: 3, ..Limits::default() }, 3).await;
    let o = Arc::new(s.orchestrator);
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let workers = o.spawn_workers(&stopped);
    let project = o.create_project("p", "idea").await.unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while o.store().project(project.id).await.unwrap().status() == ProjectStatus::Active {
        assert!(std::time::Instant::now() < deadline, "project did not finish");
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    stop.send(true).unwrap();
    for worker in workers {
        worker.await.unwrap();
    }

    assert_eq!(o.store().project(project.id).await.unwrap().status(), ProjectStatus::Completed);
    assert_eq!(s.models[&AgentRole::Coder].requests().len(), 4, "each task ran exactly once");
    assert_eq!(s.models[&AgentRole::Critic].requests().len(), 4);
}
