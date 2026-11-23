//! Request handlers. Response shapes follow what the dashboard reads.

use std::str::FromStr;

use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::{DateTime, Utc};
use hca_core::{Limits, Project, ProjectId, ProjectStatus, Task, TaskId};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast::error::RecvError;

use crate::{ApiError, AppState};

type ApiResult<T = Json<Value>> = Result<T, ApiError>;

/// Most files returned by the artifacts endpoint.
const MAX_ARTIFACTS: usize = 200;

fn parse<T: FromStr>(text: &str, what: &str) -> Result<T, ApiError> {
    text.parse()
        .map_err(|_| ApiError::new(StatusCode::BAD_REQUEST, format!("{text:?} is not a valid {what} id")))
}

fn project_json(project: &Project) -> Value {
    json!({
        "id": project.id,
        "name": project.name,
        "idea": project.idea,
        "description": project.idea,
        "status": project.status(),
        "failure": project.failure(),
        "tokens_used": project.tokens_used(),
        "created_at": project.created_at(),
        "updated_at": project.updated_at(),
    })
}

fn task_json(task: &Task, limits: &Limits, deliverable: Option<&str>) -> Value {
    let feedback =
        task.last_feedback().map(str::to_owned).or_else(|| task.failure().map(|f| format!("{f:?}")));
    json!({
        "id": task.id,
        "project_id": task.project_id,
        "title": task.title,
        "description": task.description,
        "state": task.state(),
        "role": task.role,
        "assigned_to": task.assignee().or(task.role),
        "iteration": task.revisions(),
        "max_iterations": limits.max_revisions,
        "retries": task.retries(),
        "max_retries": limits.max_retries,
        "priority": "normal",
        "tokens_used": 0,
        "depends_on": task.depends_on,
        "created_at": task.created_at(),
        "updated_at": task.updated_at(),
        "feedback": feedback,
        "deliverable": deliverable,
    })
}

pub(crate) async fn health(State(state): State<AppState>) -> ApiResult {
    let projects = state.orchestrator.store().projects(None).await?;
    let active = projects.iter().filter(|p| p.status() == ProjectStatus::Active).count();
    let tokens: u64 = projects.iter().map(Project::tokens_used).sum();
    Ok(Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "sandbox": state.orchestrator.has_sandbox(),
        "projects": projects.len(),
        "active_projects": active,
        "tokens_used": tokens,
    })))
}

pub(crate) async fn agents(State(state): State<AppState>) -> Json<Value> {
    Json(json!(state.orchestrator.agent_statuses()))
}

pub(crate) async fn list_projects(State(state): State<AppState>) -> ApiResult {
    let projects = state.orchestrator.store().projects(None).await?;
    Ok(Json(Value::Array(projects.iter().map(project_json).collect())))
}

#[derive(Deserialize)]
pub(crate) struct CreateProject {
    idea: String,
    #[serde(default)]
    name: String,
}

pub(crate) async fn create_project(
    State(state): State<AppState>,
    Json(body): Json<CreateProject>,
) -> ApiResult {
    let project = state.orchestrator.create_project(&body.name, &body.idea).await?;
    Ok(Json(
        json!({ "project_id": project.id, "status": "created", "message": "Project submitted to the planner" }),
    ))
}

pub(crate) async fn project_detail(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let id: ProjectId = parse(&id, "project")?;
    let orchestrator = &state.orchestrator;
    let project = orchestrator.store().project(id).await?;
    let progress = orchestrator.progress(id).await?;
    let messages = orchestrator.store().activity(id).await?.len();
    let files =
        orchestrator.workspace(id).map_err(hca_orchestrator::OrchestratorError::from)?.list(".").await;
    let artifacts = files.map_or(0, |f| f.iter().filter(|p| !p.ends_with('/')).count());
    Ok(Json(json!({
        "project": project_json(&project),
        "progress": progress,
        "message_count": messages,
        "artifact_count": artifacts,
    })))
}

pub(crate) async fn pause(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let id = parse(&id, "project")?;
    let status = state.orchestrator.pause(id).await?;
    Ok(Json(json!({ "project_id": id, "status": status })))
}

pub(crate) async fn resume(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let id = parse(&id, "project")?;
    let status = state.orchestrator.resume(id).await?;
    Ok(Json(json!({ "project_id": id, "status": status })))
}

pub(crate) async fn cancel(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let id = parse(&id, "project")?;
    let status = state.orchestrator.cancel(id).await?;
    Ok(Json(json!({ "project_id": id, "status": status })))
}

pub(crate) async fn messages(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let id: ProjectId = parse(&id, "project")?;
    state.orchestrator.store().project(id).await?;
    let log = state.orchestrator.store().activity(id).await?;
    Ok(Json(Value::Array(
        log.into_iter()
            .map(|a| {
                json!({
                    "id": a.seq,
                    "sender": a.entry.sender,
                    "recipient": a.entry.recipient,
                    "type": a.entry.kind,
                    "task_id": a.entry.task_id,
                    "payload": { "content": a.entry.content },
                    "timestamp": a.entry.at,
                })
            })
            .collect(),
    )))
}

pub(crate) async fn artifacts(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let id: ProjectId = parse(&id, "project")?;
    state.orchestrator.store().project(id).await?;
    let workspace = state.orchestrator.workspace(id).map_err(hca_orchestrator::OrchestratorError::from)?;
    let files = workspace.list(".").await.unwrap_or_default();
    let mut out = Vec::new();
    for path in files.iter().filter(|p| !p.ends_with('/') && !p.starts_with("...")).take(MAX_ARTIFACTS) {
        let Ok(content) = workspace.read(path).await else { continue };
        let modified: Option<DateTime<Utc>> = workspace
            .resolve(path)
            .ok()
            .and_then(|full| std::fs::metadata(full).ok())
            .and_then(|m| m.modified().ok())
            .map(DateTime::<Utc>::from);
        out.push(json!({
            "id": path,
            "filename": path,
            "agent": "",
            "version": 1,
            "artifact_type": artifact_type(path),
            "content": content,
            "created_at": modified,
        }));
    }
    Ok(Json(Value::Array(out)))
}

fn artifact_type(path: &str) -> &'static str {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if path.to_ascii_lowercase().contains("test") {
        "test"
    } else if matches!(extension.as_str(), "md" | "txt" | "rst") {
        "doc"
    } else if matches!(extension.as_str(), "json" | "yaml" | "yml" | "toml" | "ini" | "cfg") {
        "config"
    } else {
        "code"
    }
}

#[derive(Deserialize)]
pub(crate) struct Feedback {
    task_id: String,
    content: String,
}

pub(crate) async fn feedback(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<Feedback>,
) -> ApiResult {
    let project: ProjectId = parse(&id, "project")?;
    let task_id: TaskId = parse(&body.task_id, "task")?;
    let task = state.orchestrator.store().task(task_id).await?;
    if task.project_id != project {
        return Err(ApiError::new(StatusCode::NOT_FOUND, "task not found in this project"));
    }
    state.orchestrator.add_feedback(task_id, &body.content).await?;
    Ok(Json(json!({ "status": "ok", "task_id": task_id })))
}

#[derive(Deserialize)]
pub(crate) struct HumanReview {
    task_id: String,
    verdict: String,
    #[serde(default)]
    summary: String,
}

pub(crate) async fn review(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<HumanReview>,
) -> ApiResult {
    let project: ProjectId = parse(&id, "project")?;
    let task_id: TaskId = parse(&body.task_id, "task")?;
    let approve = match body.verdict.as_str() {
        "approved" | "approve" => true,
        "needs_revision" | "request_changes" => false,
        other => {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("verdict must be 'approved' or 'needs_revision', not {other:?}"),
            ))
        }
    };
    let task = state.orchestrator.store().task(task_id).await?;
    if task.project_id != project {
        return Err(ApiError::new(StatusCode::NOT_FOUND, "task not found in this project"));
    }
    state.orchestrator.human_review(task_id, approve, &body.summary).await?;
    Ok(Json(json!({ "status": "ok", "verdict": body.verdict, "task_id": task_id })))
}

pub(crate) async fn tasks(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let id: ProjectId = parse(&id, "project")?;
    let store = state.orchestrator.store();
    store.project(id).await?;
    let limits = state.orchestrator.config().limits;
    let mut out = Vec::new();
    for task in store.tasks(id).await? {
        let deliverable = store.latest_activity(task.id, "deliverable").await?.map(|a| a.entry.content);
        out.push(task_json(&task, &limits, deliverable.as_deref()));
    }
    Ok(Json(Value::Array(out)))
}

pub(crate) async fn task_detail(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let id: TaskId = parse(&id, "task")?;
    let store = state.orchestrator.store();
    let task = store.task(id).await?;
    let deliverable = store.latest_activity(id, "deliverable").await?.map(|a| a.entry.content);
    let history = store.task_history(id).await?;
    let mut body = task_json(&task, &state.orchestrator.config().limits, deliverable.as_deref());
    body["history"] = history
        .into_iter()
        .map(|h| json!({ "from": h.from, "to": h.to, "event": h.event, "at": h.at }))
        .collect();
    Ok(Json(body))
}

pub(crate) async fn retry_task(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let id: TaskId = parse(&id, "task")?;
    state.orchestrator.retry_task(id).await?;
    let task = state.orchestrator.store().task(id).await?;
    Ok(Json(json!({ "task_id": id, "state": task.state() })))
}

pub(crate) async fn websocket(
    State(state): State<AppState>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    if !same_origin(&headers) {
        return ApiError::new(StatusCode::FORBIDDEN, "cross-origin WebSocket connections are not allowed")
            .into_response();
    }
    let events = state.orchestrator.subscribe();
    upgrade.on_upgrade(move |socket| forward_events(socket, events))
}

/// Browsers always send `Origin` on WebSocket handshakes. Accept it only if
/// it names the same host and port the request was sent to. Clients that
/// aren't browsers (no `Origin`) still need the token.
fn same_origin(headers: &HeaderMap) -> bool {
    let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) else {
        return true;
    };
    let host = headers.get(header::HOST).and_then(|v| v.to_str().ok()).unwrap_or_default();
    origin.split_once("://").is_some_and(|(_, rest)| rest.trim_end_matches('/') == host)
}

async fn forward_events(mut socket: WebSocket, mut events: tokio::sync::broadcast::Receiver<Value>) {
    loop {
        tokio::select! {
            event = events.recv() => match event {
                Ok(event) => {
                    if socket.send(WsMessage::Text(event.to_string().into())).await.is_err() {
                        return;
                    }
                }
                Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => return,
            },
            incoming = socket.recv() => match incoming {
                Some(Ok(WsMessage::Close(_)) | Err(_)) | None => return,
                Some(Ok(_)) => {}
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    fn headers(origin: Option<&str>, host: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_str(host).unwrap());
        if let Some(origin) = origin {
            headers.insert(header::ORIGIN, HeaderValue::from_str(origin).unwrap());
        }
        headers
    }

    #[test]
    fn websocket_origin_must_match_the_host() {
        assert!(same_origin(&headers(Some("http://127.0.0.1:8080"), "127.0.0.1:8080")));
        assert!(!same_origin(&headers(Some("https://evil.example"), "127.0.0.1:8080")));
        assert!(!same_origin(&headers(Some("http://127.0.0.1:8080.evil.example"), "127.0.0.1:8080")));
        assert!(same_origin(&headers(None, "127.0.0.1:8080")));
    }

    #[test]
    fn artifact_types_follow_the_file_name() {
        assert_eq!(artifact_type("tests/test_fib.py"), "test");
        assert_eq!(artifact_type("README.md"), "doc");
        assert_eq!(artifact_type("pyproject.toml"), "config");
        assert_eq!(artifact_type("fib.py"), "code");
    }
}
