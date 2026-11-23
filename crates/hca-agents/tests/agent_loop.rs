//! The agent loop, driven by a scripted model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use hca_agents::{
    Agent, AgentConfig, AgentError, CommandOutput, CommandRunner, Submission, Toolbox, Verdict, Workspace,
};
use hca_core::{AgentRole, BudgetError, Limits, ProjectId, TokenBudget};
use hca_llm::testing::ScriptedProvider;
use hca_llm::{BoxFuture, ChatResponse, Llm, LlmError, Message, RoleModel, StopReason, ToolCall, Usage};
use serde_json::json;
use tempfile::TempDir;

#[derive(Debug, Default)]
struct FakeRunner {
    calls: Mutex<Vec<(PathBuf, String)>>,
}

impl CommandRunner for FakeRunner {
    fn run<'a>(
        &'a self,
        workspace: &'a Path,
        command: &'a str,
        _timeout: Duration,
    ) -> BoxFuture<'a, Result<CommandOutput, String>> {
        self.calls.lock().unwrap().push((workspace.to_path_buf(), command.to_owned()));
        Box::pin(async {
            Ok(CommandOutput {
                exit_code: Some(0),
                stdout: "2 passed".into(),
                stderr: String::new(),
                timed_out: false,
            })
        })
    }
}

struct Setup {
    llm: Llm,
    model: Arc<ScriptedProvider>,
    toolbox: Toolbox,
    runner: Arc<FakeRunner>,
    dir: TempDir,
}

fn setup(role: AgentRole, script: ScriptedProvider, with_runner: bool) -> Setup {
    let model = Arc::new(script);
    let llm = Llm::new(HashMap::from([(
        role,
        RoleModel { provider: model.clone(), model: "scripted".into(), max_tokens: 100, temperature: None },
    )]));
    let dir = tempfile::tempdir().unwrap();
    let workspace = Workspace::open(dir.path().join("project")).unwrap();
    let runner = Arc::new(FakeRunner::default());
    let dyn_runner: Option<Arc<dyn CommandRunner>> = if with_runner { Some(runner.clone()) } else { None };
    let toolbox = Toolbox::new(workspace, dyn_runner, Limits::default());
    Setup { llm, model, toolbox, runner, dir }
}

fn budget() -> TokenBudget {
    TokenBudget::new(1_000_000)
}

fn tool_calls(calls: &[(&str, serde_json::Value)]) -> ChatResponse {
    ChatResponse {
        content: String::new(),
        tool_calls: calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| ToolCall {
                id: format!("batch-{i}"),
                name: (*name).into(),
                arguments: args.clone(),
            })
            .collect(),
        stop: StopReason::ToolUse,
        usage: Usage { input_tokens: 10, output_tokens: 10 },
    }
}

fn errors(transcript: &[Message]) -> Vec<String> {
    transcript
        .iter()
        .filter_map(|m| match m {
            Message::ToolResult { content, is_error: true, .. } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn the_coder_writes_tests_runs_them_and_submits() {
    let script = ScriptedProvider::new()
        .then_tool("write_file", json!({"path": "fib.py", "content": "def fib(n):\n    return n\n"}))
        .then_tool("write_file", json!({"path": "test_fib.py", "content": "from fib import fib\n"}))
        .then_tool("run_command", json!({"command": "python -m pytest -q"}))
        .then_tool("submit", json!({"summary": "Added fib.py and test_fib.py; pytest: 2 passed."}));
    let s = setup(AgentRole::Coder, script, true);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig::default());

    let mut charged = Vec::new();
    let run = agent.run(AgentRole::Coder, "Write fib.py", budget(), |u| charged.push(u)).await.unwrap();

    assert_eq!(
        run.submission,
        Submission::Work { summary: "Added fib.py and test_fib.py; pytest: 2 passed.".into() }
    );
    assert_eq!(run.steps, 4);
    assert_eq!(run.files_written, vec!["fib.py", "test_fib.py"]);
    assert_eq!(run.usage.total(), 80);
    assert_eq!(charged.len(), 4, "usage is reported after every call");
    assert!(errors(&run.transcript).is_empty());
    assert!(std::fs::read_to_string(s.toolbox.workspace().root().join("fib.py"))
        .unwrap()
        .contains("def fib"));

    let commands = s.runner.calls.lock().unwrap().clone();
    assert_eq!(
        commands,
        vec![(s.toolbox.workspace().root().to_path_buf(), "python -m pytest -q".to_string())]
    );

    let requests = s.model.requests();
    assert!(requests[0].system.as_deref().unwrap().contains("You are the coder"));
    let tools: Vec<&str> = requests[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tools, ["list_files", "read_file", "write_file", "edit_file", "run_command", "submit"]);
    let Message::ToolResult { content, .. } = requests[3].messages.last().unwrap() else { panic!() };
    assert!(content.contains("2 passed"), "the model sees the test output: {content}");
}

#[tokio::test]
async fn refused_paths_are_reported_to_the_model_not_fatal() {
    let script = ScriptedProvider::new()
        .then_tool("write_file", json!({"path": "../escape.py", "content": "x"}))
        .then_tool("read_file", json!({"path": ".git/config"}))
        .then_tool("write_file", json!({"path": "ok.py"}))
        .then_tool("submit", json!({"summary": "done"}));
    let s = setup(AgentRole::Coder, script, false);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig::default());
    let run = agent.run(AgentRole::Coder, "task", budget(), |_| {}).await.unwrap();

    let errors = errors(&run.transcript);
    assert_eq!(errors.len(), 3);
    assert!(errors[0].contains("outside the project"), "{}", errors[0]);
    assert!(errors[1].contains(".git"), "{}", errors[1]);
    assert!(errors[2].contains("missing field `content`"), "{}", errors[2]);
    assert!(run.files_written.is_empty());
    assert!(!s.dir.path().join("escape.py").exists());
}

#[tokio::test]
async fn the_critic_cannot_write_and_must_explain_rejections() {
    let script = ScriptedProvider::new()
        .then_tool("write_file", json!({"path": "fix.py", "content": "x"}))
        .then_tool("submit_review", json!({"verdict": "request_changes", "feedback": "  "}))
        .then_tool(
            "submit_review",
            json!({"verdict": "request_changes", "feedback": "fib.py: fib(0) returns 1; it should return 0."}),
        );
    let s = setup(AgentRole::Critic, script, true);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig::default());
    let run = agent.run(AgentRole::Critic, "Review fib.py", budget(), |_| {}).await.unwrap();

    let Submission::Review(review) = run.submission else { panic!("expected a review") };
    assert_eq!(review.verdict, Verdict::RequestChanges);
    assert!(review.feedback.contains("fib(0)"));
    let errors = errors(&run.transcript);
    assert!(errors[0].contains("no tool named \"write_file\""), "{}", errors[0]);
    assert!(errors[1].contains("feedback is required"), "{}", errors[1]);
    assert!(!s.toolbox.workspace().root().join("fix.py").exists());

    let tools: Vec<String> = s.model.requests()[0].tools.iter().map(|t| t.name.clone()).collect();
    assert_eq!(tools, ["list_files", "read_file", "run_command", "submit_review"]);
}

#[tokio::test]
async fn without_a_sandbox_there_is_no_run_command() {
    let script = ScriptedProvider::new()
        .then_tool("run_command", json!({"command": "rm -rf /"}))
        .then_tool("submit", json!({"summary": "done"}));
    let s = setup(AgentRole::Coder, script, false);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig::default());
    let run = agent.run(AgentRole::Coder, "task", budget(), |_| {}).await.unwrap();

    assert!(errors(&run.transcript)[0].contains("no tool named \"run_command\""));
    assert!(s.model.requests()[0].tools.iter().all(|t| t.name != "run_command"));
    assert!(s.runner.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_pm_fixes_an_invalid_plan_before_it_is_accepted() {
    let script = ScriptedProvider::new()
        .then_tool(
            "submit_plan",
            json!({"tasks": [{"title": "Code", "description": "Write it", "role": "coder", "depends_on": [3]}]}),
        )
        .then_tool(
            "submit_plan",
            json!({"tasks": [
                {"title": "Spec", "description": "Specify the CLI", "role": "spec"},
                {"title": "Code", "description": "Implement the spec with tests", "role": "coder", "depends_on": [0]}
            ]}),
        );
    let s = setup(AgentRole::Pm, script, false);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig::default());
    let run = agent.run(AgentRole::Pm, "Build a Fibonacci CLI", budget(), |_| {}).await.unwrap();

    assert!(errors(&run.transcript)[0].contains("depends on 3"));
    let Submission::Plan(plan) = run.submission else { panic!("expected a plan") };
    let tasks = plan.into_tasks(ProjectId::new(), &Limits::default(), Utc::now()).unwrap();
    assert_eq!(tasks[1].depends_on, vec![tasks[0].id]);
    assert_eq!(tasks[0].role, Some(AgentRole::Spec));
}

#[tokio::test]
async fn nothing_runs_after_the_submission() {
    let script = ScriptedProvider::new().then(Ok(tool_calls(&[
        ("submit", json!({"summary": "done"})),
        ("write_file", json!({"path": "late.py", "content": "x"})),
    ])));
    let s = setup(AgentRole::Coder, script, false);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig::default());
    let run = agent.run(AgentRole::Coder, "task", budget(), |_| {}).await.unwrap();

    assert!(!s.toolbox.workspace().root().join("late.py").exists());
    assert!(errors(&run.transcript)[0].contains("already submitted"));
    assert_eq!(run.transcript.iter().filter(|m| matches!(m, Message::ToolResult { .. })).count(), 2);
}

#[tokio::test]
async fn runs_stop_at_the_step_limit() {
    let script = (0..5).fold(ScriptedProvider::new(), |s, _| s.then_tool("list_files", json!({})));
    let s = setup(AgentRole::Coder, script, false);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig { max_steps: 3, max_reminders: 2 });
    let err = agent.run(AgentRole::Coder, "task", budget(), |_| {}).await.unwrap_err();
    assert!(matches!(err, AgentError::StepLimit(3)), "{err:?}");
    assert_eq!(s.model.requests().len(), 3);
}

#[tokio::test]
async fn an_agent_that_never_submits_is_reminded_then_stopped() {
    let script =
        ScriptedProvider::new().then_text("I think it's done.").then_text("Done!").then_text("Really done.");
    let s = setup(AgentRole::Coder, script, false);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig { max_steps: 10, max_reminders: 2 });
    let err = agent.run(AgentRole::Coder, "task", budget(), |_| {}).await.unwrap_err();
    assert!(matches!(err, AgentError::NoSubmission(3)), "{err:?}");

    let reminder = s.model.requests()[1].messages.last().unwrap().clone();
    assert!(matches!(reminder, Message::User { content } if content.contains("`submit`")));
}

#[tokio::test]
async fn a_cut_off_reply_is_continued_without_using_a_reminder() {
    let cut_off = Ok(ChatResponse {
        content: "Here is the first half".into(),
        tool_calls: vec![],
        stop: StopReason::MaxTokens,
        usage: Usage { input_tokens: 10, output_tokens: 100 },
    });
    let script = ScriptedProvider::new().then(cut_off).then_tool("submit", json!({"summary": "done"}));
    let s = setup(AgentRole::Research, script, false);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig { max_steps: 10, max_reminders: 0 });
    let run = agent.run(AgentRole::Research, "task", budget(), |_| {}).await.unwrap();
    assert_eq!(run.steps, 2);
}

#[tokio::test]
async fn the_budget_stops_the_run_between_calls() {
    let expensive = Ok(ChatResponse {
        content: String::new(),
        tool_calls: vec![ToolCall { id: "1".into(), name: "list_files".into(), arguments: json!({}) }],
        stop: StopReason::ToolUse,
        usage: Usage { input_tokens: 40_000, output_tokens: 1_000 },
    });
    let script = ScriptedProvider::new().then(expensive).then_tool("submit", json!({"summary": "done"}));
    let s = setup(AgentRole::Coder, script, false);
    let agent = Agent::new(&s.llm, &s.toolbox, AgentConfig::default());

    let mut charged = Vec::new();
    let err =
        agent.run(AgentRole::Coder, "task", TokenBudget::new(40_000), |u| charged.push(u)).await.unwrap_err();
    assert!(matches!(err, AgentError::Llm(LlmError::Budget(BudgetError::Exhausted { .. }))), "{err:?}");
    assert_eq!(s.model.requests().len(), 1, "the second call is never sent");
    assert_eq!(charged.iter().map(Usage::total).sum::<u64>(), 41_000, "the spent tokens were still reported");
}
