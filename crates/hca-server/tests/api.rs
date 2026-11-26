//! The HTTP API, end to end, with scripted models.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use hca_core::AgentRole;
use hca_llm::testing::ScriptedProvider;
use hca_llm::{Llm, RoleModel};
use hca_orchestrator::{Orchestrator, OrchestratorConfig};
use hca_server::{router, AppState};
use hca_store::Store;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

const TOKEN: &str = "test-token-123";

async fn app(models: Vec<(AgentRole, ScriptedProvider)>) -> (Router, Arc<Orchestrator>, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("hca.db")).await.unwrap();
    let roles: HashMap<AgentRole, RoleModel> = models
        .into_iter()
        .map(|(role, provider)| {
            (
                role,
                RoleModel {
                    provider: Arc::new(provider),
                    model: format!("{role}-model"),
                    max_tokens: 200,
                    temperature: None,
                },
            )
        })
        .collect();
    let orchestrator = Arc::new(Orchestrator::new(
        store,
        Llm::new(roles),
        None,
        OrchestratorConfig::new(dir.path().join("ws")),
    ));
    let state = AppState { orchestrator: orchestrator.clone(), token: Arc::from(TOKEN) };
    (router(state), orchestrator, dir)
}

async fn call(
    app: &Router,
    method: Method,
    uri: &str,
    token: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut request = Request::builder().method(method).uri(uri);
    if let Some(token) = token {
        request = request.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    let request = match body {
        Some(body) => {
            request.header(header::CONTENT_TYPE, "application/json").body(Body::from(body.to_string()))
        }
        None => request.body(Body::empty()),
    }
    .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test]
async fn the_api_requires_the_token_but_the_dashboard_is_public() {
    let (app, _, _dir) = app(vec![]).await;
    for (uri, expected) in
        [("/", StatusCode::OK), ("/js/app.js", StatusCode::OK), ("/api/health/live", StatusCode::OK)]
    {
        assert_eq!(call(&app, Method::GET, uri, None, None).await.0, expected, "{uri}");
    }
    for uri in ["/api/projects/", "/api/health", "/api/agents/", "/ws"] {
        assert_eq!(call(&app, Method::GET, uri, None, None).await.0, StatusCode::UNAUTHORIZED, "{uri}");
        assert_eq!(
            call(&app, Method::GET, uri, Some("wrong"), None).await.0,
            StatusCode::UNAUTHORIZED,
            "{uri}"
        );
    }
    let (status, _) =
        call(&app, Method::POST, "/api/projects/", None, Some(json!({"idea": "steal compute"}))).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(call(&app, Method::GET, "/api/projects/", Some(TOKEN), None).await.0, StatusCode::OK);
}

#[tokio::test]
async fn other_websites_get_no_cors_permission() {
    let (app, _, _dir) = app(vec![]).await;
    let preflight = Request::builder()
        .method(Method::OPTIONS)
        .uri("/api/projects/")
        .header(header::ORIGIN, "https://evil.example")
        .header("access-control-request-method", "POST")
        .header("access-control-request-headers", "authorization,content-type")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(preflight).await.unwrap();
    assert!(response.headers().get("access-control-allow-origin").is_none());
}

#[tokio::test]
async fn the_token_can_come_from_the_query_for_websockets() {
    let (app, _, _dir) = app(vec![]).await;
    let (status, _) = call(&app, Method::GET, &format!("/api/health?token={TOKEN}"), None, None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn a_project_runs_through_the_api() {
    let models = vec![
        (
            AgentRole::Pm,
            ScriptedProvider::new().then_tool(
                "submit_plan",
                json!({"tasks": [{"title": "Write fib", "description": "fib.py with tests", "role": "coder"}]}),
            ),
        ),
        (
            AgentRole::Coder,
            ScriptedProvider::new()
                .then_tool("write_file", json!({"path": "fib.py", "content": "def fib(n): return n"}))
                .then_tool("submit", json!({"summary": "Wrote fib.py"})),
        ),
        (AgentRole::Critic, ScriptedProvider::new().then_tool("submit_review", json!({"verdict": "approve"}))),
    ];
    let (app, orchestrator, _dir) = app(models).await;

    let (status, created) = call(
        &app,
        Method::POST,
        "/api/projects/",
        Some(TOKEN),
        Some(json!({"idea": "A fib function", "name": "Fib"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let id = created["project_id"].as_str().unwrap().to_owned();

    orchestrator.run_until_idle("test").await.unwrap();

    let (_, detail) = call(&app, Method::GET, &format!("/api/projects/{id}"), Some(TOKEN), None).await;
    assert_eq!(detail["project"]["status"], "completed");
    assert_eq!(detail["progress"]["completed"], 1);
    assert_eq!(detail["progress"]["token_usage"]["budget"], 500_000);
    assert_eq!(detail["artifact_count"], 1);

    let (_, tasks) = call(&app, Method::GET, &format!("/api/tasks/{id}"), Some(TOKEN), None).await;
    assert_eq!(tasks[0]["state"], "done");
    assert_eq!(tasks[0]["assigned_to"], "coder");
    assert!(tasks[0]["deliverable"].as_str().unwrap().contains("Wrote fib.py"));

    let (_, artifacts) =
        call(&app, Method::GET, &format!("/api/projects/{id}/artifacts"), Some(TOKEN), None).await;
    assert_eq!(artifacts[0]["filename"], "fib.py");
    assert_eq!(artifacts[0]["content"], "def fib(n): return n");

    let (_, messages) =
        call(&app, Method::GET, &format!("/api/projects/{id}/messages"), Some(TOKEN), None).await;
    let kinds: Vec<&str> = messages.as_array().unwrap().iter().map(|m| m["type"].as_str().unwrap()).collect();
    assert_eq!(kinds.first(), Some(&"system"));
    assert!(kinds.contains(&"deliverable"));
    assert_eq!(messages[0]["payload"]["content"], "A fib function");

    let task_id = tasks[0]["id"].as_str().unwrap();
    let (_, detail) =
        call(&app, Method::GET, &format!("/api/tasks/detail/{task_id}"), Some(TOKEN), None).await;
    assert_eq!(detail["history"].as_array().unwrap().len(), 4);

    let (_, agents) = call(&app, Method::GET, "/api/agents/", Some(TOKEN), None).await;
    assert_eq!(agents.as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn bad_requests_get_clear_errors() {
    let (app, orchestrator, _dir) = app(vec![]).await;
    let (status, body) = call(&app, Method::GET, "/api/projects/not-a-uuid", Some(TOKEN), None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["detail"].as_str().unwrap().contains("not a valid project id"));

    let missing = hca_core::ProjectId::new();
    let (status, _) = call(&app, Method::GET, &format!("/api/projects/{missing}"), Some(TOKEN), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) =
        call(&app, Method::POST, "/api/projects/", Some(TOKEN), Some(json!({"idea": "  "}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let project = orchestrator.create_project("p", "idea").await.unwrap();
    let (status, _) =
        call(&app, Method::POST, &format!("/api/projects/{}/resume", project.id), Some(TOKEN), None).await;
    assert_eq!(status, StatusCode::CONFLICT, "an active project can't be resumed");

    let (status, body) = call(
        &app,
        Method::POST,
        &format!("/api/projects/{}/review", project.id),
        Some(TOKEN),
        Some(json!({"task_id": hca_core::TaskId::new().to_string(), "verdict": "maybe"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn pausing_and_cancelling_work_through_the_api() {
    let (app, orchestrator, _dir) = app(vec![]).await;
    let project = orchestrator.create_project("p", "idea").await.unwrap();
    let base = format!("/api/projects/{}", project.id);

    let (status, body) = call(&app, Method::POST, &format!("{base}/pause"), Some(TOKEN), None).await;
    assert_eq!((status, body["status"].as_str()), (StatusCode::OK, Some("paused")));
    let (status, body) = call(&app, Method::POST, &format!("{base}/cancel"), Some(TOKEN), None).await;
    assert_eq!((status, body["status"].as_str()), (StatusCode::OK, Some("cancelled")));
    assert_eq!(orchestrator.run_until_idle("w").await.unwrap(), 0, "cancelled work never runs");
}
