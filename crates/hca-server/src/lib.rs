//! HTTP API, WebSocket and dashboard for HCA.
//!
//! Security defaults:
//!
//! - **Every API call needs the access token** (`Authorization: Bearer`),
//!   and so does the WebSocket (`?token=`). Only the dashboard's static
//!   files and `/api/health/live` are public.
//! - **No CORS headers**, so other websites can't call the API from your
//!   browser. WebSocket connections from another origin are refused.
//! - The binary binds to `127.0.0.1` by default.

mod routes;

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hca_orchestrator::Orchestrator;
use serde_json::json;

/// Shared state for every request.
#[derive(Clone, Debug)]
pub struct AppState {
    /// The orchestrator.
    pub orchestrator: Arc<Orchestrator>,
    /// The access token.
    pub token: Arc<str>,
}

/// The full application: dashboard, API and WebSocket.
pub fn router(state: AppState) -> Router {
    let api = Router::new()
        .route("/api/health", get(routes::health))
        .route("/api/agents/", get(routes::agents))
        .route("/api/projects/", get(routes::list_projects).post(routes::create_project))
        .route("/api/projects/{id}", get(routes::project_detail))
        .route("/api/projects/{id}/pause", post(routes::pause))
        .route("/api/projects/{id}/resume", post(routes::resume))
        .route("/api/projects/{id}/cancel", post(routes::cancel))
        .route("/api/projects/{id}/restart", post(routes::restart))
        .route("/api/projects/{id}/messages", get(routes::messages))
        .route("/api/projects/{id}/artifacts", get(routes::artifacts))
        .route("/api/projects/{id}/feedback", post(routes::feedback))
        .route("/api/projects/{id}/review", post(routes::review))
        .route("/api/tasks/{project_id}", get(routes::tasks))
        .route("/api/tasks/detail/{task_id}", get(routes::task_detail))
        .route("/api/tasks/detail/{task_id}/retry", post(routes::retry_task))
        .route("/ws", get(routes::websocket))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_token));

    Router::new()
        .route(
            "/",
            get(|| async { static_file("text/html; charset=utf-8", include_str!("../static/index.html")) }),
        )
        .route(
            "/js/app.js",
            get(|| async {
                static_file("text/javascript; charset=utf-8", include_str!("../static/js/app.js"))
            }),
        )
        .route(
            "/css/styles.css",
            get(|| async {
                static_file("text/css; charset=utf-8", include_str!("../static/css/styles.css"))
            }),
        )
        .route("/api/health/live", get(|| async { Json(json!({ "status": "alive" })) }))
        .merge(api)
        .with_state(state)
}

fn static_file(content_type: &'static str, body: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (header::X_FRAME_OPTIONS, "DENY"),
            (header::CONTENT_SECURITY_POLICY, "frame-ancestors 'none'"),
        ],
        body,
    )
        .into_response()
}

async fn require_token(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let from_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_owned);
    let from_query = request
        .uri()
        .query()
        .and_then(|q| q.split('&').find_map(|pair| pair.strip_prefix("token=")))
        .map(percent_decode);
    let presented = from_header.or(from_query).unwrap_or_default();
    if constant_time_eq(presented.as_bytes(), state.token.as_bytes()) {
        next.run(request).await
    } else {
        ApiError::new(StatusCode::UNAUTHORIZED, "missing or invalid access token").into_response()
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn percent_decode(text: &str) -> String {
    fn hex(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let decoded = match (
            bytes[i],
            bytes.get(i + 1).copied().and_then(hex),
            bytes.get(i + 2).copied().and_then(hex),
        ) {
            (b'%', Some(high), Some(low)) => Some(high * 16 + low),
            _ => None,
        };
        if let Some(byte) = decoded {
            out.push(byte);
            i += 3;
        } else {
            out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// An error response: `{"detail": "..."}` with a status code.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    detail: String,
}

impl ApiError {
    fn new(status: StatusCode, detail: impl Into<String>) -> Self {
        Self { status, detail: detail.into() }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "detail": self.detail }))).into_response()
    }
}

impl From<hca_store::StoreError> for ApiError {
    fn from(error: hca_store::StoreError) -> Self {
        use hca_store::StoreError as E;
        let status = match &error {
            E::NotFound { .. } => StatusCode::NOT_FOUND,
            E::Task(_)
            | E::Project(_)
            | E::Plan(_)
            | E::ProjectNotActive(_)
            | E::PlanExists
            | E::LeaseLost(_) => StatusCode::CONFLICT,
            _ => {
                tracing::error!(%error, "store error");
                return Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal error");
            }
        };
        Self::new(status, error.to_string())
    }
}

impl From<hca_orchestrator::OrchestratorError> for ApiError {
    fn from(error: hca_orchestrator::OrchestratorError) -> Self {
        use hca_orchestrator::OrchestratorError as E;
        match error {
            E::Store(e) => e.into(),
            E::Conflict(message) => Self::new(StatusCode::BAD_REQUEST, message),
            E::Workspace(e) => {
                tracing::error!(error = %e, "workspace error");
                Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_percent_escapes() {
        assert_eq!(percent_decode("a%2Bb%3D"), "a+b=");
        assert_eq!(percent_decode("plain"), "plain");
        assert_eq!(percent_decode("bad%zz"), "bad%zz");
        assert_eq!(percent_decode("end%2"), "end%2");
        assert_eq!(percent_decode("é%"), "é%", "never splits a multibyte character");
    }

    #[test]
    fn compares_tokens_exactly() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret2"));
    }
}
