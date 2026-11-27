//! A deny-by-default proxy between HCA and the Docker socket.
//!
//! Access to the Docker API is root on the host: whoever can create a
//! container can mount `/`, run it privileged, or join the host's network.
//! A generic socket proxy only filters API sections (containers yes, images
//! no), not what a container may be. This guard checks every call against
//! exactly what [`DockerSandbox`](crate::DockerSandbox) sends, so a
//! compromised HCA process can still only start sandbox containers:
//!
//! - **Only the calls the sandbox makes:** ping, version, listing by HCA's
//!   labels, and create, inspect, start, kill, remove and exec of containers
//!   named `hca-sb-*`. Everything else (images, volumes, networks, `archive`,
//!   `attach`, the daemon's `info`) gets 403.
//! - **Only sandbox containers:** the configured image, a non-root user, no
//!   network (unless allowed), all capabilities dropped,
//!   `no-new-privileges`, a read-only root filesystem, resource limits, and
//!   as the only mount one project folder: a single-component subpath of
//!   the configured workspaces volume. A field the guard doesn't know is
//!   refused, so new Docker options can't slip through.
//! - **Exec only as the container's user:** exec may not set a user or
//!   privileges, and only execs the guard created can be started.
//!
//! Workspaces are mounted as volume subpaths, never host paths. Docker
//! resolves a subpath inside the volume and refuses one that leads out of
//! it through a symlink, so swapping a project folder for a symlink to `/`
//! mounts nothing. (The volume's own folder must be one HCA can't swap
//! either; the Compose setup makes it a mount point of its own.)

use std::collections::{HashSet, VecDeque};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use bytes::Bytes;
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Incoming;
use hyper::header::{HeaderValue, CONNECTION, CONTENT_TYPE, HOST, UPGRADE};
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{Map, Value};
use tokio::net::{TcpListener, UnixStream};

use crate::{CONTAINER_WORKSPACE, LABEL, SESSION_LABEL, WORKSPACE_LABEL};

/// Largest request body accepted. Sandbox requests are a few KiB.
const MAX_REQUEST: usize = 1024 * 1024;
/// Largest response body passed back. Inspect and list answers are small.
const MAX_RESPONSE: usize = 16 * 1024 * 1024;
/// How many created exec ids are remembered for starting.
const MAX_EXECS: usize = 10_000;

/// What the guard lets through.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardConfig {
    /// The only image containers may use, e.g. `hca-sandbox:latest`.
    pub image: String,
    /// The only volume containers may mount: the workspaces volume.
    pub volume: String,
    /// Whether containers may have a network (`bridge`). Off by default.
    pub allow_network: bool,
}

/// Why a call was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct Denied(pub String);

/// What kind of call was allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Allowed {
    /// Forward and return the answer.
    Plain,
    /// An exec was created: remember its id.
    ExecCreate,
    /// An exec starts: the connection is upgraded to a raw stream.
    ExecStart,
}

/// The guard. Cheap to share; one serves all connections.
#[derive(Debug)]
pub struct Guard {
    config: GuardConfig,
    socket: PathBuf,
    execs: Mutex<(VecDeque<String>, HashSet<String>)>,
}

fn deny<T>(reason: impl Into<String>) -> Result<T, Denied> {
    Err(Denied(reason.into()))
}

impl Guard {
    /// A guard for the Docker daemon at `socket`, e.g.
    /// `/var/run/docker.sock`.
    pub fn new(config: GuardConfig, socket: impl Into<PathBuf>) -> Arc<Self> {
        Arc::new(Self { config, socket: socket.into(), execs: Mutex::default() })
    }

    /// Accepts connections on `listener` until the task is dropped.
    ///
    /// # Errors
    ///
    /// Returns the error if accepting a connection fails.
    pub async fn serve(self: Arc<Self>, listener: TcpListener) -> std::io::Result<()> {
        loop {
            let (stream, _) = listener.accept().await?;
            let guard = Arc::clone(&self);
            tokio::spawn(async move {
                let service = hyper::service::service_fn(move |request| Arc::clone(&guard).handle(request));
                let connection = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .with_upgrades();
                if let Err(error) = connection.await {
                    tracing::debug!(%error, "guard connection ended");
                }
            });
        }
    }

    /// Checks one call. `path` is the request path, with or without the
    /// `/v1.xx` API version prefix.
    ///
    /// # Errors
    ///
    /// Returns [`Denied`] with the reason for anything the sandbox doesn't
    /// do.
    pub fn check(
        &self,
        method: &Method,
        path: &str,
        query: Option<&str>,
        body: &[u8],
    ) -> Result<Allowed, Denied> {
        let path = strip_version(path);
        let segments: Vec<&str> = path.strip_prefix('/').unwrap_or(path).split('/').collect();
        let params = parse_query(query)?;
        let only = |allowed: &[&str]| -> Result<(), Denied> {
            match params.iter().find(|(key, _)| !allowed.contains(&key.as_str())) {
                Some((key, _)) => deny(format!("query parameter {key:?} is not allowed here")),
                None => Ok(()),
            }
        };
        let name = |name: &str| -> Result<(), Denied> {
            if is_sandbox_name(name) {
                Ok(())
            } else {
                deny(format!("{name:?} is not a sandbox container (hca-sb-*)"))
            }
        };
        match (method, segments.as_slice()) {
            (&Method::GET | &Method::HEAD, ["_ping"]) | (&Method::GET, ["version"]) => {
                only(&[])?;
                Ok(Allowed::Plain)
            }
            (&Method::GET, ["containers", "json"]) => {
                only(&["all", "filters", "size"])?;
                if params.iter().any(|(key, value)| key == "size" && value != "false") {
                    return deny("container sizes are not needed");
                }
                check_list_filters(&params)?;
                Ok(Allowed::Plain)
            }
            (&Method::POST, ["containers", "create"]) => {
                only(&["name", "platform"])?;
                // bollard always sends `platform`, empty: the daemon's own.
                if params.iter().any(|(key, value)| key == "platform" && !value.is_empty()) {
                    return deny("containers can't pick a platform");
                }
                match params.iter().find(|(key, _)| key == "name") {
                    Some((_, value)) => name(value)?,
                    None => return deny("containers must be created with a sandbox name"),
                }
                self.check_container(&parse_object(body)?)?;
                Ok(Allowed::Plain)
            }
            (&Method::GET, ["containers", id, "json"]) | (&Method::POST, ["containers", id, "start"]) => {
                only(&[])?;
                name(id)?;
                Ok(Allowed::Plain)
            }
            (&Method::POST, ["containers", id, "kill"]) => {
                only(&["signal"])?;
                name(id)?;
                Ok(Allowed::Plain)
            }
            (&Method::DELETE, ["containers", id]) => {
                only(&["force", "v", "link"])?;
                // `link=true` would remove a link, not the container.
                if params.iter().any(|(key, value)| key == "link" && value != "false") {
                    return deny("removing links is not allowed");
                }
                name(id)?;
                Ok(Allowed::Plain)
            }
            (&Method::POST, ["containers", id, "exec"]) => {
                only(&[])?;
                name(id)?;
                check_exec(&parse_object(body)?)?;
                Ok(Allowed::ExecCreate)
            }
            (&Method::POST, ["exec", id, "start"]) => {
                only(&[])?;
                self.known_exec(id)?;
                let start = parse_object(body)?;
                // `OutputCapacity` is bollard's own buffer size, which it
                // sends along; Docker ignores it.
                allow_keys(&start, &["Detach", "Tty", "ConsoleSize", "OutputCapacity"], "exec start")?;
                if start.get("Detach").and_then(Value::as_bool) == Some(true) {
                    return deny("detached execs are not allowed: nothing would wait for them");
                }
                Ok(Allowed::ExecStart)
            }
            (&Method::GET, ["exec", id, "json"]) => {
                only(&[])?;
                self.known_exec(id)?;
                Ok(Allowed::Plain)
            }
            _ => deny(format!("{method} {path} is not a call the sandbox makes")),
        }
    }

    fn check_container(&self, body: &Map<String, Value>) -> Result<(), Denied> {
        allow_keys(
            body,
            &["Image", "Cmd", "WorkingDir", "User", "Env", "NetworkDisabled", "Labels", "HostConfig"],
            "container",
        )?;
        if body.get("Image").and_then(Value::as_str) != Some(self.config.image.as_str()) {
            return deny(format!("containers must use the image {:?}", self.config.image));
        }
        strings(body.get("Cmd"), "Cmd")?;
        strings(body.get("Env"), "Env")?;
        if body.get("WorkingDir").is_some_and(|dir| dir.as_str() != Some(CONTAINER_WORKSPACE)) {
            return deny(format!("WorkingDir must be {CONTAINER_WORKSPACE}"));
        }
        check_user(body.get("User"))?;
        let network_disabled = body.get("NetworkDisabled").and_then(Value::as_bool);
        if !self.config.allow_network && network_disabled != Some(true) {
            return deny("containers may not have a network (set HCA_GUARD_ALLOW_NETWORK to allow one)");
        }
        let labels = body.get("Labels").and_then(Value::as_object);
        if labels.and_then(|l| l.get(LABEL)).and_then(Value::as_str) != Some("1") {
            return deny(format!("containers must carry the {LABEL}=1 label"));
        }
        if labels.is_some_and(|l| l.values().any(|v| !v.is_string())) {
            return deny("labels must be strings");
        }
        let Some(host) = body.get("HostConfig").and_then(Value::as_object) else {
            return deny("HostConfig is required");
        };
        self.check_host(host)
    }

    fn check_host(&self, host: &Map<String, Value>) -> Result<(), Denied> {
        allow_keys(
            host,
            &[
                "Mounts",
                "Memory",
                "MemorySwap",
                "NanoCpus",
                "PidsLimit",
                "NetworkMode",
                "CapDrop",
                "SecurityOpt",
                "ReadonlyRootfs",
                "Tmpfs",
                "Init",
            ],
            "HostConfig",
        )?;
        for limit in ["Memory", "MemorySwap", "NanoCpus", "PidsLimit"] {
            if host.get(limit).and_then(Value::as_i64).is_none_or(|n| n <= 0) {
                return deny(format!("HostConfig.{limit} must be a positive limit"));
            }
        }
        match host.get("NetworkMode").and_then(Value::as_str) {
            Some("none") => {}
            Some("bridge") if self.config.allow_network => {}
            _ => return deny("HostConfig.NetworkMode must be \"none\""),
        }
        if !strings(host.get("CapDrop"), "CapDrop")?.contains(&"ALL") {
            return deny("HostConfig.CapDrop must drop ALL capabilities");
        }
        if strings(host.get("SecurityOpt"), "SecurityOpt")? != ["no-new-privileges"] {
            return deny("HostConfig.SecurityOpt must be exactly [\"no-new-privileges\"]");
        }
        if host.get("ReadonlyRootfs").and_then(Value::as_bool) != Some(true) {
            return deny("HostConfig.ReadonlyRootfs must be true");
        }
        if host.get("Init").is_some_and(|init| !init.is_boolean()) {
            return deny("HostConfig.Init must be a boolean");
        }
        if let Some(tmpfs) = host.get("Tmpfs") {
            let Some(tmpfs) = tmpfs.as_object() else { return deny("HostConfig.Tmpfs must be an object") };
            for (target, options) in tmpfs {
                if !["/tmp", "/workspace/.git"].contains(&target.as_str()) || !options.is_string() {
                    return deny(format!("tmpfs at {target:?} is not allowed"));
                }
            }
        }
        let mounts = host.get("Mounts").and_then(Value::as_array).map_or(&[][..], Vec::as_slice);
        let [mount] = mounts else {
            return deny("exactly one mount is allowed: the project folder");
        };
        self.check_mount(mount)
    }

    fn check_mount(&self, mount: &Value) -> Result<(), Denied> {
        let Some(mount) = mount.as_object() else { return deny("a mount must be an object") };
        allow_keys(mount, &["Type", "Source", "Target", "ReadOnly", "VolumeOptions"], "mount")?;
        if mount.get("Type").and_then(Value::as_str) != Some("volume") {
            return deny("only volume mounts are allowed, never host paths (set workspace_volume)");
        }
        if mount.get("Source").and_then(Value::as_str) != Some(self.config.volume.as_str()) {
            return deny(format!("only the volume {:?} may be mounted", self.config.volume));
        }
        if mount.get("Target").and_then(Value::as_str) != Some(CONTAINER_WORKSPACE) {
            return deny(format!("the project must be mounted at {CONTAINER_WORKSPACE}"));
        }
        if mount.get("ReadOnly").is_some_and(|r| !r.is_boolean()) {
            return deny("ReadOnly must be a boolean");
        }
        let Some(options) = mount.get("VolumeOptions").and_then(Value::as_object) else {
            return deny("the mount must name a project folder (VolumeOptions.Subpath)");
        };
        // No DriverConfig: it could create the volume over any host path.
        allow_keys(options, &["Subpath", "NoCopy"], "VolumeOptions")?;
        if options.get("NoCopy").is_some_and(|n| !n.is_boolean()) {
            return deny("NoCopy must be a boolean");
        }
        match options.get("Subpath").and_then(Value::as_str) {
            Some(folder) if is_project_folder(folder) => Ok(()),
            _ => deny("the subpath must be a single project folder name"),
        }
    }

    fn known_exec(&self, id: &str) -> Result<(), Denied> {
        let execs = self.execs.lock().unwrap_or_else(PoisonError::into_inner);
        if execs.1.contains(id) {
            Ok(())
        } else {
            deny("only execs created through the guard can be used")
        }
    }

    fn remember_exec(&self, id: String) {
        let mut execs = self.execs.lock().unwrap_or_else(PoisonError::into_inner);
        if execs.1.insert(id.clone()) {
            execs.0.push_back(id);
        }
        while execs.0.len() > MAX_EXECS {
            if let Some(old) = execs.0.pop_front() {
                execs.1.remove(&old);
            }
        }
    }

    async fn handle(
        self: Arc<Self>,
        mut request: Request<Incoming>,
    ) -> Result<Response<Full<Bytes>>, Infallible> {
        let downstream = hyper::upgrade::on(&mut request);
        let (parts, body) = request.into_parts();
        let body = match Limited::new(body, MAX_REQUEST).collect().await {
            Ok(body) => body.to_bytes(),
            Err(_) => return Ok(refusal(StatusCode::PAYLOAD_TOO_LARGE, "request body too large")),
        };
        let allowed = match self.check(&parts.method, parts.uri.path(), parts.uri.query(), &body) {
            Ok(allowed) => allowed,
            Err(denied) => {
                tracing::warn!(method = %parts.method, path = parts.uri.path(), reason = %denied, "refused a Docker call");
                return Ok(refusal(StatusCode::FORBIDDEN, &denied.0));
            }
        };
        match self.forward(&parts, body, allowed, downstream).await {
            Ok(response) => Ok(response),
            Err(error) => {
                tracing::warn!(%error, "Docker call failed");
                Ok(refusal(StatusCode::BAD_GATEWAY, &format!("could not reach Docker: {error}")))
            }
        }
    }

    async fn forward(
        &self,
        parts: &hyper::http::request::Parts,
        body: Bytes,
        allowed: Allowed,
        downstream: hyper::upgrade::OnUpgrade,
    ) -> Result<Response<Full<Bytes>>, Box<dyn std::error::Error + Send + Sync>> {
        let stream = UnixStream::connect(&self.socket).await?;
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream)).await?;
        tokio::spawn(async move {
            if let Err(error) = connection.with_upgrades().await {
                tracing::debug!(%error, "Docker connection ended");
            }
        });

        let path = parts.uri.path_and_query().map_or("/", hyper::http::uri::PathAndQuery::as_str);
        let mut upstream = Request::builder().method(parts.method.clone()).uri(path);
        upstream = upstream.header(HOST, "docker");
        if let Some(content_type) = parts.headers.get(CONTENT_TYPE) {
            upstream = upstream.header(CONTENT_TYPE, content_type);
        }
        if allowed == Allowed::ExecStart {
            for name in [CONNECTION, UPGRADE] {
                if let Some(value) = parts.headers.get(&name) {
                    upstream = upstream.header(name, value);
                }
            }
        }
        let mut response = sender.send_request(upstream.body(Full::new(body))?).await?;

        if response.status() == StatusCode::SWITCHING_PROTOCOLS {
            let upstream_upgrade = hyper::upgrade::on(&mut response);
            tokio::spawn(async move {
                let (Ok(down), Ok(up)) = tokio::join!(downstream, upstream_upgrade) else { return };
                let (mut down, mut up) = (TokioIo::new(down), TokioIo::new(up));
                let _ = tokio::io::copy_bidirectional(&mut down, &mut up).await;
            });
            let mut answer = Response::builder().status(StatusCode::SWITCHING_PROTOCOLS);
            for name in [CONNECTION, UPGRADE, CONTENT_TYPE] {
                if let Some(value) = response.headers().get(&name) {
                    answer = answer.header(name, value);
                }
            }
            return Ok(answer.body(Full::new(Bytes::new()))?);
        }

        let status = response.status();
        let content_type = response.headers().get(CONTENT_TYPE).cloned();
        let bytes = Limited::new(response.into_body(), MAX_RESPONSE).collect().await?.to_bytes();
        if allowed == Allowed::ExecCreate && status.is_success() {
            if let Some(id) = serde_json::from_slice::<Value>(&bytes)
                .ok()
                .and_then(|v| v.get("Id").and_then(Value::as_str).map(str::to_owned))
            {
                self.remember_exec(id);
            }
        }
        let mut answer = Response::builder().status(status);
        if let Some(content_type) = content_type {
            answer = answer.header(CONTENT_TYPE, content_type);
        }
        Ok(answer.body(Full::new(bytes))?)
    }
}

fn refusal(status: StatusCode, message: &str) -> Response<Full<Bytes>> {
    let body = serde_json::json!({ "message": format!("refused by the HCA Docker guard: {message}") });
    let mut response = Response::new(Full::new(Bytes::from(body.to_string())));
    *response.status_mut() = status;
    response.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

/// Removes a leading `/v1.49`-style API version.
fn strip_version(path: &str) -> &str {
    if let Some(rest) = path.strip_prefix("/v") {
        if let Some(end) = rest.find('/') {
            let version = &rest[..end];
            if !version.is_empty() && version.chars().all(|c| c.is_ascii_digit() || c == '.') {
                return &rest[end..];
            }
        }
    }
    path
}

fn parse_query(query: Option<&str>) -> Result<Vec<(String, String)>, Denied> {
    let pairs: Vec<(String, String)> = form_urlencoded::parse(query.unwrap_or_default().as_bytes())
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    let mut seen = HashSet::new();
    for (key, _) in &pairs {
        if !seen.insert(key.as_str()) {
            return deny(format!("query parameter {key:?} is repeated"));
        }
    }
    Ok(pairs)
}

fn check_list_filters(params: &[(String, String)]) -> Result<(), Denied> {
    let Some((_, filters)) = params.iter().find(|(key, _)| key == "filters") else {
        return deny("listing containers needs a label filter");
    };
    let filters: Value = serde_json::from_str(filters).map_err(|_| Denied("filters must be JSON".into()))?;
    let Some(filters) = filters.as_object() else { return deny("filters must be an object") };
    allow_keys(filters, &["label"], "filters")?;
    let labels = strings(filters.get("label"), "label filter")?;
    let ours = |label: &&str| {
        [LABEL, WORKSPACE_LABEL, SESSION_LABEL].iter().any(|key| label.starts_with(&format!("{key}=")))
    };
    if labels.is_empty() || !labels.iter().all(ours) {
        return deny("containers can only be listed by HCA's own labels");
    }
    Ok(())
}

fn check_exec(body: &Map<String, Value>) -> Result<(), Denied> {
    // No `User` and no `Privileged`: commands run as the container's user.
    allow_keys(body, &["Cmd", "AttachStdin", "AttachStdout", "AttachStderr", "WorkingDir", "Tty"], "exec")?;
    strings(body.get("Cmd"), "Cmd")?;
    for flag in ["AttachStdin", "AttachStdout", "AttachStderr", "Tty"] {
        if body.get(flag).is_some_and(|v| !v.is_boolean()) {
            return deny(format!("{flag} must be a boolean"));
        }
    }
    if body.get("WorkingDir").is_some_and(|dir| dir.as_str() != Some(CONTAINER_WORKSPACE)) {
        return deny(format!("WorkingDir must be {CONTAINER_WORKSPACE}"));
    }
    Ok(())
}

fn check_user(user: Option<&Value>) -> Result<(), Denied> {
    let user = user.and_then(Value::as_str).unwrap_or_default();
    let ids: Vec<&str> = user.split(':').collect();
    let numeric_non_root =
        |id: &str| id.parse::<u32>().is_ok_and(|n| n > 0) && id.chars().all(|c| c.is_ascii_digit());
    match ids.as_slice() {
        [uid, gid] if numeric_non_root(uid) && numeric_non_root(gid) => Ok(()),
        _ => deny("containers must run as a numeric non-root uid:gid"),
    }
}

fn parse_object(body: &[u8]) -> Result<Map<String, Value>, Denied> {
    if body.is_empty() {
        return Ok(Map::new());
    }
    match serde_json::from_slice(body) {
        Ok(Value::Object(object)) => Ok(object),
        _ => deny("the body must be a JSON object"),
    }
}

/// Refuses any key not listed. Docker matches JSON keys without regard to
/// case, so an exact list also stops `privileged` sneaking past `Privileged`.
fn allow_keys(object: &Map<String, Value>, allowed: &[&str], what: &str) -> Result<(), Denied> {
    match object.keys().find(|key| !allowed.contains(&key.as_str())) {
        Some(key) => deny(format!("{what} field {key:?} is not allowed")),
        None => Ok(()),
    }
}

/// A list of strings, or an empty list if absent.
fn strings<'v>(value: Option<&'v Value>, what: &str) -> Result<Vec<&'v str>, Denied> {
    match value {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| item.as_str().ok_or_else(|| Denied(format!("{what} must be a list of strings"))))
            .collect(),
        Some(_) => deny(format!("{what} must be a list of strings")),
    }
}

fn is_sandbox_name(name: &str) -> bool {
    name.len() <= 200
        && name.strip_prefix("hca-sb-").is_some_and(|rest| !rest.is_empty())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn is_project_folder(folder: &str) -> bool {
    !folder.is_empty()
        && folder.len() <= 128
        && folder.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && folder.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use hca_agents::Session;
    use serde_json::json;

    use super::*;
    use crate::SandboxConfig;

    fn guard() -> Arc<Guard> {
        Guard::new(
            GuardConfig {
                image: "hca-sandbox:latest".into(),
                volume: "hca-workspaces".into(),
                allow_network: false,
            },
            "/nonexistent.sock",
        )
    }

    fn sandbox_body(session: Session<'_>) -> Value {
        let config =
            SandboxConfig { workspace_volume: Some("hca-workspaces".into()), ..SandboxConfig::default() };
        serde_json::to_value(config.container_body_for(Path::new("/data/workspaces/3f2a-9c1e"), session))
            .unwrap()
    }

    fn create(guard: &Guard, name: &str, body: &Value) -> Result<Allowed, Denied> {
        guard.check(
            &Method::POST,
            "/v1.49/containers/create",
            Some(&format!("name={name}")),
            body.to_string().as_bytes(),
        )
    }

    /// Applies `change` to a valid body and expects the guard to refuse it.
    fn refused(change: impl FnOnce(&mut Value)) -> String {
        let mut body = sandbox_body(Session::default());
        change(&mut body);
        create(&guard(), "hca-sb-x", &body).unwrap_err().0
    }

    #[test]
    fn what_the_sandbox_sends_is_allowed() {
        let g = guard();
        for session in [Session::default(), Session { id: "run1", user: Some(hca_agents::CRITIC_USER) }] {
            assert_eq!(create(&g, "hca-sb-3f2a-srun1", &sandbox_body(session)), Ok(Allowed::Plain));
        }
        let get = |path: &str, query: Option<&str>| g.check(&Method::GET, path, query, b"");
        assert_eq!(get("/_ping", None), Ok(Allowed::Plain));
        assert_eq!(get("/v1.49/containers/hca-sb-3f2a/json", None), Ok(Allowed::Plain));
        let filters =
            form_urlencoded::byte_serialize(br#"{"label":["hca.workspace=/w","hca.session=run1"]}"#)
                .collect::<String>();
        assert_eq!(
            get("/v1.49/containers/json", Some(&format!("all=true&filters={filters}"))),
            Ok(Allowed::Plain)
        );
        let exec = json!({"Cmd": ["sh", "-c", "pytest"], "AttachStdout": true, "AttachStderr": true, "WorkingDir": "/workspace"});
        assert_eq!(
            g.check(&Method::POST, "/v1.49/containers/hca-sb-3f2a/exec", None, exec.to_string().as_bytes()),
            Ok(Allowed::ExecCreate)
        );
        assert_eq!(
            g.check(&Method::DELETE, "/v1.49/containers/hca-sb-3f2a", Some("force=true"), b""),
            Ok(Allowed::Plain)
        );
    }

    #[test]
    fn dangerous_containers_are_refused() {
        type Change = Box<dyn FnOnce(&mut Value)>;
        let cases: Vec<(&str, Change)> = vec![
            ("privileged", Box::new(|b| b["HostConfig"]["Privileged"] = json!(true))),
            ("lower-case privileged", Box::new(|b| b["HostConfig"]["privileged"] = json!(true))),
            ("added capabilities", Box::new(|b| b["HostConfig"]["CapAdd"] = json!(["SYS_ADMIN"]))),
            ("capabilities kept", Box::new(|b| b["HostConfig"]["CapDrop"] = json!([]))),
            ("a host bind of /", Box::new(|b| b["HostConfig"]["Binds"] = json!(["/:/host"]))),
            ("a bind mount", Box::new(|b| b["HostConfig"]["Mounts"][0]["Type"] = json!("bind"))),
            ("another volume", Box::new(|b| b["HostConfig"]["Mounts"][0]["Source"] = json!("other"))),
            (
                "the whole volume",
                Box::new(|b| b["HostConfig"]["Mounts"][0]["VolumeOptions"]["Subpath"] = json!("")),
            ),
            (
                "a parent subpath",
                Box::new(|b| b["HostConfig"]["Mounts"][0]["VolumeOptions"]["Subpath"] = json!("..")),
            ),
            (
                "a nested subpath",
                Box::new(|b| b["HostConfig"]["Mounts"][0]["VolumeOptions"]["Subpath"] = json!("a/b")),
            ),
            (
                "a volume driver",
                Box::new(|b| {
                    b["HostConfig"]["Mounts"][0]["VolumeOptions"]["DriverConfig"] =
                        json!({"Name": "local", "Options": {"device": "/", "o": "bind"}});
                }),
            ),
            (
                "two mounts",
                Box::new(|b| {
                    let mount = b["HostConfig"]["Mounts"][0].clone();
                    b["HostConfig"]["Mounts"].as_array_mut().unwrap().push(mount);
                }),
            ),
            ("another image", Box::new(|b| b["Image"] = json!("alpine"))),
            ("root", Box::new(|b| b["User"] = json!("0:0"))),
            ("a named user", Box::new(|b| b["User"] = json!("root"))),
            ("the host network", Box::new(|b| b["HostConfig"]["NetworkMode"] = json!("host"))),
            ("a network", Box::new(|b| b["NetworkDisabled"] = json!(false))),
            ("the host pid namespace", Box::new(|b| b["HostConfig"]["PidMode"] = json!("host"))),
            ("devices", Box::new(|b| b["HostConfig"]["Devices"] = json!([{"PathOnHost": "/dev/sda"}]))),
            ("seccomp off", Box::new(|b| b["HostConfig"]["SecurityOpt"] = json!(["seccomp=unconfined"]))),
            ("a writable root", Box::new(|b| b["HostConfig"]["ReadonlyRootfs"] = json!(false))),
            ("no memory limit", Box::new(|b| b["HostConfig"]["Memory"] = json!(0))),
            ("an odd tmpfs", Box::new(|b| b["HostConfig"]["Tmpfs"]["/etc"] = json!("rw"))),
            ("anonymous volumes", Box::new(|b| b["Volumes"] = json!({"/x": {}}))),
            ("no sandbox label", Box::new(|b| b["Labels"] = json!({}))),
        ];
        for (what, change) in cases {
            let reason = refused(change);
            assert!(!reason.is_empty(), "{what} was refused without a reason");
        }
        let body = sandbox_body(Session::default());
        assert!(create(&guard(), "evil", &body).is_err(), "names must be sandbox names");
        assert!(guard()
            .check(&Method::POST, "/containers/create", None, body.to_string().as_bytes())
            .is_err());
    }

    #[test]
    fn exec_runs_only_as_the_container_user_and_only_known_execs_start() {
        let g = guard();
        let exec = |body: Value| {
            g.check(&Method::POST, "/containers/hca-sb-x/exec", None, body.to_string().as_bytes())
        };
        assert!(exec(json!({"Cmd": ["id"], "User": "root"})).is_err(), "no exec as root");
        assert!(exec(json!({"Cmd": ["id"], "Privileged": true})).is_err());
        assert!(g.check(&Method::POST, "/containers/docker-host/exec", None, br#"{"Cmd":["id"]}"#).is_err());

        let start = br#"{"Detach":false,"Tty":false}"#;
        assert!(g.check(&Method::POST, "/v1.49/exec/abc/start", None, start).is_err(), "unknown exec");
        g.remember_exec("abc".into());
        assert_eq!(g.check(&Method::POST, "/v1.49/exec/abc/start", None, start), Ok(Allowed::ExecStart));
        assert!(g.check(&Method::POST, "/v1.49/exec/abc/start", None, br#"{"Detach":true}"#).is_err());
    }

    #[test]
    fn other_api_calls_are_refused() {
        let g = guard();
        for (method, path, query) in [
            (Method::GET, "/v1.49/images/json", None),
            (Method::POST, "/v1.49/images/create", Some("fromImage=alpine")),
            (Method::GET, "/v1.49/info", None),
            (Method::GET, "/v1.49/containers/hca-sb-x/archive", Some("path=/")),
            (Method::PUT, "/v1.49/containers/hca-sb-x/archive", Some("path=/")),
            (Method::POST, "/v1.49/containers/hca-sb-x/attach", None),
            (Method::POST, "/v1.49/containers/hca-sb-x/update", None),
            (Method::POST, "/v1.49/volumes/create", None),
            (Method::POST, "/v1.49/networks/create", None),
            (Method::GET, "/v1.49/containers/json", None),
            (Method::GET, "/v1.49/containers/json", Some("filters=%7B%22name%22%3A%5B%22x%22%5D%7D")),
            (Method::GET, "/v1.49/containers/../images/json", None),
            (Method::GET, "/v1.49/containers/hca-sb-x%2F..%2F..%2Fimages/json", None),
            (Method::DELETE, "/v1.49/containers/hca-sb-x", Some("force=true&force=false")),
            (Method::POST, "/v1.49/containers/hca-sb-x/kill", Some("signal=KILL&x=1")),
        ] {
            assert!(g.check(&method, path, query, b"").is_err(), "{method} {path} {query:?} was allowed");
        }
    }

    #[test]
    fn version_prefixes_are_stripped_strictly() {
        assert_eq!(strip_version("/v1.49/containers/json"), "/containers/json");
        assert_eq!(strip_version("/containers/json"), "/containers/json");
        assert_eq!(strip_version("/vx/containers/json"), "/vx/containers/json");
    }
}
