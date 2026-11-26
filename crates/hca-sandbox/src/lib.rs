//! Isolated, resource-limited Docker containers for agent commands.
//!
//! Every agent run (a [`Session`]) gets a container of its own with only its
//! project's workspace mounted, and its commands run inside it through
//! `docker exec`. Processes a run starts never share a container with
//! another run's, and [`DockerSandbox::end_session`] removes the container
//! when the run ends, so nothing it left running can change files after it
//! handed its work on. Commands that run as a different user (the critic)
//! get the workspace mounted **read-only**, so a reviewer can neither change
//! the project nor leave anything behind (a Python startup hook, say) for
//! the next agent to run.
//!
//! By default every container has:
//!
//! - **no network**, so generated code can't download or exfiltrate
//!   anything;
//! - **memory, CPU and process limits**, so a fork bomb or a memory leak
//!   can't take the host down;
//! - **all Linux capabilities dropped**, `no-new-privileges`, a read-only
//!   root filesystem with a small writable `/tmp`, and a non-root user.
//!
//! A command that runs past its timeout is stopped by killing the
//! container; the next command starts a fresh one. Containers are also
//! removed when their project ends ([`DockerSandbox::remove`]) and swept at
//! startup and shutdown ([`DockerSandbox::sweep`]). This crate talks to the Docker API directly,
//! so the orchestrator image doesn't need the `docker` CLI.

pub mod guard;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use bollard::container::LogOutput;
use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};
use bollard::models::{ContainerCreateBody, HostConfig, Mount, MountTypeEnum, MountVolumeOptions};
use bollard::query_parameters::{
    CreateContainerOptionsBuilder, KillContainerOptions, ListContainersOptionsBuilder,
    RemoveContainerOptionsBuilder, StartContainerOptions,
};
use bollard::Docker;
use futures_util::StreamExt;
use hca_agents::{CommandOutput, CommandRunner, Session};
use hca_llm::BoxFuture;
use serde::{Deserialize, Serialize};

/// Translates a path as the orchestrator sees it into the path the Docker
/// daemon sees, for when the orchestrator itself runs in a container and
/// talks to the host's daemon.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathMap {
    /// Prefix inside the orchestrator, e.g. `/data/workspaces`.
    pub from: PathBuf,
    /// The same folder on the Docker host, e.g. `/srv/hca/workspaces`.
    pub to: PathBuf,
}

/// Sandbox settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SandboxConfig {
    /// Image for project containers. It must include the tools agents run,
    /// such as Python and pytest. See `docker/sandbox.Dockerfile`.
    pub image: String,
    /// Memory limit in MiB (swap is disabled).
    pub memory_mb: u64,
    /// CPU limit, in cores.
    pub cpus: f64,
    /// Maximum number of processes.
    pub pids_limit: i64,
    /// Allow network access. Off by default.
    pub network: bool,
    /// User to run as, `uid:gid`. Should own the workspace files.
    pub user: String,
    /// Size of the writable `/tmp`, in MiB.
    pub tmp_mb: u64,
    /// Most bytes of output kept per stream.
    pub max_output_bytes: usize,
    /// Path translation for Docker-in-Docker setups. Ignored with
    /// [`workspace_volume`](Self::workspace_volume).
    pub path_map: Option<PathMap>,
    /// A Docker volume holding the workspaces folder. When set, each
    /// container mounts its project's folder as a subpath of this volume
    /// instead of a host path. Docker resolves a subpath without following
    /// symlinks out of the volume, so a folder swapped for a symlink can't
    /// mount anything else. The Compose setup and [`guard`] use this.
    pub workspace_volume: Option<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            image: "hca-sandbox:latest".into(),
            memory_mb: 2048,
            cpus: 2.0,
            pids_limit: 256,
            network: false,
            user: "1000:1000".into(),
            tmp_mb: 512,
            max_output_bytes: 1024 * 1024,
            path_map: None,
            workspace_volume: None,
        }
    }
}

/// Where the workspace is mounted inside the container.
pub const CONTAINER_WORKSPACE: &str = "/workspace";

/// Label put on every sandbox container, so they can be found and cleaned up.
pub const LABEL: &str = "hca.sandbox";

/// Label holding the workspace path (as the orchestrator sees it).
pub const WORKSPACE_LABEL: &str = "hca.workspace";

/// Label holding the [`Session`] id.
pub const SESSION_LABEL: &str = "hca.session";

/// Label holding the settings a container was created with.
///
/// A running container is reused only when this matches. Anything else is
/// removed and created again, so turning the network off actually applies.
const CONFIG_LABEL: &str = "hca.config";

impl SandboxConfig {
    /// Settings that decide whether an existing container can be reused.
    ///
    /// Includes markers for the `/workspace/.git` tmpfs and the per-user
    /// layout, so containers created before those are replaced.
    fn fingerprint(&self, user: &str, read_only: bool) -> String {
        format!(
            "image={};memory={};cpus={};pids={};network={};user={user};ro={read_only};tmp={};volume={};git-tmpfs=1",
            self.image,
            self.memory_mb,
            self.cpus,
            self.pids_limit,
            self.network,
            self.tmp_mb,
            self.workspace_volume.as_deref().unwrap_or("-")
        )
    }

    /// The Docker path of `workspace`, after [`path_map`](Self::path_map).
    pub fn host_path(&self, workspace: &Path) -> PathBuf {
        match &self.path_map {
            Some(map) => match workspace.strip_prefix(&map.from) {
                Ok(rest) => map.to.join(rest),
                Err(_) => workspace.to_path_buf(),
            },
            None => workspace.to_path_buf(),
        }
    }

    /// The container definition for `workspace`'s default session and
    /// user, with the workspace writable.
    pub fn container_body(&self, workspace: &Path) -> ContainerCreateBody {
        self.container_body_for(workspace, Session::default())
    }

    /// The container definition for `session`. A session with a user other
    /// than the default gets the workspace read-only: reviewers read the
    /// project, they don't change it.
    pub fn container_body_for(&self, workspace: &Path, session: Session<'_>) -> ContainerCreateBody {
        let mib = |n: u64| i64::try_from(n.saturating_mul(1024 * 1024)).unwrap_or(i64::MAX);
        let host = self.host_path(workspace);
        let read_only = session.user.is_some();
        let user = session.user.unwrap_or(&self.user);
        let mode = if read_only { "ro" } else { "rw" };
        #[allow(clippy::cast_possible_truncation)]
        let nano_cpus = (self.cpus.max(0.01) * 1e9).round() as i64;
        let mut env = vec!["HOME=/tmp".to_owned(), "PYTHONDONTWRITEBYTECODE=1".to_owned()];
        if read_only {
            // pytest's cache would only produce warnings on a read-only
            // project.
            env.push("PYTEST_ADDOPTS=-p no:cacheprovider".to_owned());
        }
        ContainerCreateBody {
            image: Some(self.image.clone()),
            cmd: Some(vec!["sleep".into(), "infinity".into()]),
            working_dir: Some(CONTAINER_WORKSPACE.into()),
            user: Some(user.to_owned()),
            env: Some(env),
            network_disabled: Some(!self.network),
            labels: Some(HashMap::from([
                (LABEL.to_owned(), "1".to_owned()),
                (CONFIG_LABEL.to_owned(), self.fingerprint(user, read_only)),
                (WORKSPACE_LABEL.to_owned(), workspace.display().to_string()),
                (SESSION_LABEL.to_owned(), session.id.to_owned()),
            ])),
            host_config: Some(HostConfig {
                binds: self
                    .workspace_volume
                    .is_none()
                    .then(|| vec![format!("{}:{CONTAINER_WORKSPACE}:{mode}", host.display())]),
                mounts: self.workspace_volume.as_ref().map(|volume| {
                    vec![Mount {
                        typ: Some(MountTypeEnum::VOLUME),
                        source: Some(volume.clone()),
                        target: Some(CONTAINER_WORKSPACE.into()),
                        read_only: Some(read_only),
                        volume_options: Some(MountVolumeOptions {
                            subpath: Some(workspace_folder(workspace)),
                            no_copy: Some(true),
                            ..MountVolumeOptions::default()
                        }),
                        ..Mount::default()
                    }]
                }),
                memory: Some(mib(self.memory_mb)),
                memory_swap: Some(mib(self.memory_mb)),
                nano_cpus: Some(nano_cpus),
                pids_limit: Some(self.pids_limit),
                network_mode: Some(if self.network { "bridge".into() } else { "none".into() }),
                cap_drop: Some(vec!["ALL".into()]),
                security_opt: Some(vec!["no-new-privileges".into()]),
                readonly_rootfs: Some(true),
                tmpfs: Some(HashMap::from([
                    ("/tmp".to_owned(), format!("rw,nosuid,nodev,size={}m", self.tmp_mb)),
                    // Hides the host's `.git` (hooks, config, credentials)
                    // from every command, including ones the coder runs.
                    ("/workspace/.git".to_owned(), "rw,nosuid,nodev,size=1m".to_owned()),
                ])),
                init: Some(true),
                ..HostConfig::default()
            }),
            ..ContainerCreateBody::default()
        }
    }
}

/// A workspace's folder name: its project id.
fn workspace_folder(workspace: &Path) -> String {
    workspace.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// The container name for a workspace's default session: `hca-sb-` plus
/// the folder name (the project id), restricted to characters Docker allows.
pub fn container_name(workspace: &Path) -> String {
    container_name_for(workspace, Session::default())
}

/// The container name for `session`: the default name plus `-s` and the
/// session id, and `-u` and the user if it has one, e.g.
/// `hca-sb-<id>-srun7-u10000-10000`.
pub fn container_name_for(workspace: &Path, session: Session<'_>) -> String {
    let base = workspace_folder(workspace);
    let clean: String = base.chars().filter(|c| docker_name_char(*c)).take(64).collect();
    let mut name = if clean.is_empty() { "hca-sb-workspace".into() } else { format!("hca-sb-{clean}") };
    if !session.id.is_empty() {
        name.push_str("-s");
        name.extend(session.id.chars().map(|c| if docker_name_char(c) { c } else { '-' }).take(48));
    }
    if let Some(user) = session.user {
        let user: String = user.chars().map(|c| if docker_name_char(c) { c } else { '-' }).take(32).collect();
        name.push_str("-u");
        name.push_str(&user);
    }
    name
}

fn docker_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')
}

/// Why the sandbox couldn't be used.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// The Docker daemon isn't reachable.
    #[error("cannot reach Docker: {0}")]
    Unavailable(String),
    /// A Docker API call failed.
    #[error("Docker error: {0}")]
    Docker(#[from] bollard::errors::Error),
    /// The workspace couldn't be prepared for mounting.
    #[error("workspace error: {0}")]
    Workspace(#[from] std::io::Error),
}

/// Runs agent commands in per-project containers.
#[derive(Clone, Debug)]
pub struct DockerSandbox {
    docker: Docker,
    config: SandboxConfig,
    /// One lock per container, so a timeout or a reset can't kill a sibling
    /// exec.
    locks: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl DockerSandbox {
    /// Connects to the local Docker daemon (honouring `DOCKER_HOST`) and
    /// checks that it answers.
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError::Unavailable`] if Docker can't be reached.
    pub async fn connect(config: SandboxConfig) -> Result<Self, SandboxError> {
        let docker = Docker::connect_with_defaults().map_err(|e| SandboxError::Unavailable(e.to_string()))?;
        Self::connect_with(docker, config).await
    }

    /// Uses `docker`, a client for a daemon or a [`guard`], and checks that
    /// it answers.
    ///
    /// # Errors
    ///
    /// Returns [`SandboxError::Unavailable`] if it doesn't answer.
    pub async fn connect_with(docker: Docker, config: SandboxConfig) -> Result<Self, SandboxError> {
        docker.ping().await.map_err(|e| SandboxError::Unavailable(e.to_string()))?;
        Ok(Self { docker, config, locks: Arc::new(Mutex::new(HashMap::new())) })
    }

    fn lock_for(&self, name: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock().unwrap_or_else(PoisonError::into_inner);
        locks.entry(name.to_owned()).or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))).clone()
    }

    fn forget_lock(&self, name: &str) {
        self.locks.lock().unwrap_or_else(PoisonError::into_inner).remove(name);
    }

    /// The settings in use.
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }

    /// Removes the containers of session `id` in `workspace`, with every
    /// process still running in them. Doesn't wait for a running command:
    /// the session is over, so it is killed and its caller gets an error.
    ///
    /// # Errors
    ///
    /// Returns the first Docker error, except "no such container".
    pub async fn end_session(&self, workspace: &Path, id: &str) -> Result<(), SandboxError> {
        let labels = [format!("{WORKSPACE_LABEL}={}", workspace.display()), format!("{SESSION_LABEL}={id}")];
        self.remove_all(&labels).await
    }

    /// Removes every container of a workspace, for all sessions, at once.
    /// The files stay. Like [`end_session`](Self::end_session), this
    /// doesn't wait for running commands: the project is over.
    ///
    /// # Errors
    ///
    /// Returns the first Docker error, except "no such container".
    pub async fn remove(&self, workspace: &Path) -> Result<(), SandboxError> {
        self.remove_all(&[format!("{WORKSPACE_LABEL}={}", workspace.display())]).await
    }

    async fn remove_all(&self, labels: &[String]) -> Result<(), SandboxError> {
        for (name, _) in self.labelled(labels).await? {
            self.remove_named(&name).await?;
            self.forget_lock(&name);
        }
        Ok(())
    }

    /// Removes every sandbox container whose workspace is under `root`, and
    /// returns how many there were. Run at startup: containers left by an
    /// earlier run hold processes nobody is waiting for. Containers of other
    /// HCA installations on the same daemon are left alone.
    ///
    /// # Errors
    ///
    /// Returns the first Docker error.
    pub async fn sweep(&self, root: &Path) -> Result<usize, SandboxError> {
        let mut removed = 0;
        for (name, workspace) in self.labelled(&[format!("{LABEL}=1")]).await? {
            if workspace.as_deref().is_some_and(|w| Path::new(w).starts_with(root)) {
                self.remove_named(&name).await?;
                self.forget_lock(&name);
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Names and workspace labels of the containers carrying all of
    /// `labels` (`key=value`), running or not.
    async fn labelled(&self, labels: &[String]) -> Result<Vec<(String, Option<String>)>, SandboxError> {
        let filters = HashMap::from([("label", labels.iter().map(String::as_str).collect::<Vec<_>>())]);
        let options = ListContainersOptionsBuilder::new().all(true).filters(&filters).build();
        let containers = self.docker.list_containers(Some(options)).await?;
        Ok(containers
            .into_iter()
            .filter_map(|container| {
                let name = container.names?.into_iter().next()?.trim_start_matches('/').to_owned();
                let workspace = container.labels.and_then(|mut labels| labels.remove(WORKSPACE_LABEL));
                Some((name, workspace))
            })
            .collect())
    }

    async fn remove_named(&self, name: &str) -> Result<(), SandboxError> {
        let options = RemoveContainerOptionsBuilder::new().force(true).build();
        match self.docker.remove_container(name, Some(options)).await {
            Err(e) if status(&e) == Some(404) => Ok(()),
            other => Ok(other?),
        }
    }

    async fn ensure_running(
        &self,
        name: &str,
        workspace: &Path,
        session: Session<'_>,
    ) -> Result<(), SandboxError> {
        let want = self.config.fingerprint(session.user.unwrap_or(&self.config.user), session.user.is_some());
        let info = match self
            .docker
            .inspect_container(name, None::<bollard::query_parameters::InspectContainerOptions>)
            .await
        {
            Ok(info) => Some(info),
            Err(e) if status(&e) == Some(404) => None,
            Err(e) => return Err(e.into()),
        };
        let reusable = info.as_ref().is_some_and(|info| {
            let running = info.state.as_ref().and_then(|state| state.running).unwrap_or(false);
            let have = info
                .config
                .as_ref()
                .and_then(|config| config.labels.as_ref())
                .and_then(|labels| labels.get(CONFIG_LABEL))
                .map(String::as_str);
            running && have == Some(want.as_str())
        });
        if reusable {
            return Ok(());
        }
        if info.is_some() {
            self.remove_named(name).await?;
        }
        // The `.git` tmpfs needs a mount point. On a read-only mount Docker
        // can't create it, and on a writable one it would create it owned by
        // root, so make it here, as the user that owns the workspace.
        std::fs::create_dir_all(workspace.join(".git"))?;
        let options = CreateContainerOptionsBuilder::new().name(name).build();
        match self
            .docker
            .create_container(Some(options), self.config.container_body_for(workspace, session))
            .await
        {
            // Another run created it at the same moment.
            Err(e) if status(&e) == Some(409) => {}
            other => {
                other?;
            }
        }
        match self.docker.start_container(name, None::<StartContainerOptions>).await {
            Err(e) if status(&e) == Some(304) => Ok(()),
            other => Ok(other?),
        }
    }

    async fn exec(
        &self,
        workspace: &Path,
        command: &str,
        timeout: Duration,
        session: Session<'_>,
    ) -> Result<CommandOutput, SandboxError> {
        let name = container_name_for(workspace, session);
        let lock = self.lock_for(&name);
        let _guard = lock.lock().await;
        self.ensure_running(&name, workspace, session).await?;

        let exec = self
            .docker
            .create_exec(
                &name,
                CreateExecOptions {
                    cmd: Some(vec!["sh", "-c", command]),
                    attach_stdout: Some(true),
                    attach_stderr: Some(true),
                    working_dir: Some(CONTAINER_WORKSPACE),
                    ..CreateExecOptions::default()
                },
            )
            .await?;
        let started = self
            .docker
            .start_exec(&exec.id, Some(StartExecOptions { detach: false, tty: false, output_capacity: None }))
            .await?;
        let StartExecResults::Attached { mut output, .. } = started else {
            return Ok(CommandOutput::default());
        };

        let limit = self.config.max_output_bytes;
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let collect = async {
            while let Some(chunk) = output.next().await {
                match chunk {
                    Ok(LogOutput::StdOut { message }) => append_capped(&mut stdout, &message, limit),
                    Ok(LogOutput::StdErr { message }) => append_capped(&mut stderr, &message, limit),
                    Ok(_) => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        };

        let mut failure_note = String::new();
        let timed_out = if let Ok(result) = tokio::time::timeout(timeout, collect).await {
            result?;
            false
        } else {
            // Docker can't kill a single exec, so destroy the container.
            // The per-container lock means this doesn't hit a sibling exec,
            // and the next command creates a new container.
            let mut notes = Vec::new();
            if let Err(error) = self.docker.kill_container(&name, None::<KillContainerOptions>).await {
                notes.push(format!("kill failed: {error}"));
            }
            if let Err(error) = self.remove_named(&name).await {
                notes.push(format!("remove failed: {error}"));
            }
            failure_note = notes.join("\n");
            true
        };

        let exit_code = if timed_out {
            None
        } else {
            self.docker.inspect_exec(&exec.id).await?.exit_code.and_then(|c| i32::try_from(c).ok())
        };
        let mut stderr = String::from_utf8_lossy(&stderr).into_owned();
        if !failure_note.is_empty() {
            if !stderr.is_empty() && !stderr.ends_with('\n') {
                stderr.push('\n');
            }
            stderr.push_str(&failure_note);
        }
        Ok(CommandOutput {
            exit_code,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr,
            timed_out,
        })
    }
}

impl CommandRunner for DockerSandbox {
    fn run<'a>(
        &'a self,
        workspace: &'a Path,
        command: &'a str,
        timeout: Duration,
        session: Session<'a>,
    ) -> BoxFuture<'a, Result<CommandOutput, String>> {
        Box::pin(
            async move { self.exec(workspace, command, timeout, session).await.map_err(|e| e.to_string()) },
        )
    }

    fn end_session<'a>(&'a self, workspace: &'a Path, id: &'a str) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(
            async move { DockerSandbox::end_session(self, workspace, id).await.map_err(|e| e.to_string()) },
        )
    }

    fn remove<'a>(&'a self, workspace: &'a Path) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move { DockerSandbox::remove(self, workspace).await.map_err(|e| e.to_string()) })
    }
}

fn status(error: &bollard::errors::Error) -> Option<u16> {
    match error {
        bollard::errors::Error::DockerResponseServerError { status_code, .. } => Some(*status_code),
        _ => None,
    }
}

fn append_capped(buffer: &mut Vec<u8>, bytes: &[u8], limit: usize) {
    let room = limit.saturating_sub(buffer.len());
    buffer.extend_from_slice(&bytes[..bytes.len().min(room)]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn containers_are_locked_down_by_default() {
        let body = SandboxConfig::default().container_body(Path::new("/data/ws/1234"));
        let host = body.host_config.unwrap();
        assert_eq!(host.network_mode.as_deref(), Some("none"));
        assert_eq!(body.network_disabled, Some(true));
        assert_eq!(host.memory, Some(2048 * 1024 * 1024));
        assert_eq!(host.memory_swap, host.memory, "no swap beyond the memory limit");
        assert_eq!(host.nano_cpus, Some(2_000_000_000));
        assert_eq!(host.pids_limit, Some(256));
        assert_eq!(host.cap_drop, Some(vec!["ALL".to_string()]));
        assert_eq!(host.security_opt, Some(vec!["no-new-privileges".to_string()]));
        assert_eq!(host.readonly_rootfs, Some(true));
        assert_eq!(host.binds, Some(vec!["/data/ws/1234:/workspace:rw".to_string()]));
        assert_eq!(body.user.as_deref(), Some("1000:1000"));
        let tmpfs = host.tmpfs.unwrap();
        assert!(tmpfs["/tmp"].contains("size=512m"));
        assert!(tmpfs["/workspace/.git"].contains("size=1m"), "host .git is hidden");
        assert!(body.labels.unwrap()[CONFIG_LABEL].contains("git-tmpfs=1"));
    }

    #[test]
    fn other_users_get_a_read_only_container_of_their_own() {
        let ws = Path::new("/data/ws/1234");
        let critic = Session { id: "run7", user: Some("10000:10000") };
        let body = SandboxConfig::default().container_body_for(ws, critic);
        let host = body.host_config.unwrap();
        assert_eq!(host.binds, Some(vec!["/data/ws/1234:/workspace:ro".to_string()]));
        assert_eq!(body.user.as_deref(), Some("10000:10000"));
        assert_eq!(host.cap_drop, Some(vec!["ALL".to_string()]), "just as locked down");
        let labels = body.labels.unwrap();
        assert!(labels[CONFIG_LABEL].contains("ro=true"));
        assert_eq!(labels[WORKSPACE_LABEL], "/data/ws/1234");
        assert_eq!(labels[SESSION_LABEL], "run7");
        assert_eq!(container_name_for(ws, critic), "hca-sb-1234-srun7-u10000-10000");
        assert_ne!(container_name_for(ws, critic), container_name(ws));
        let coder = Session { id: "run8", user: None };
        assert_eq!(container_name_for(ws, coder), "hca-sb-1234-srun8", "every run has its own container");
    }

    #[test]
    fn a_workspace_volume_mounts_the_project_folder_as_a_subpath() {
        let config =
            SandboxConfig { workspace_volume: Some("hca-workspaces".into()), ..SandboxConfig::default() };
        let critic = Session { id: "run1", user: Some("10000:10000") };
        let host = config.container_body_for(Path::new("/data/workspaces/1234"), critic).host_config.unwrap();
        assert_eq!(host.binds, None, "no host path is mounted");
        let mount = &host.mounts.unwrap()[0];
        assert_eq!(mount.source.as_deref(), Some("hca-workspaces"));
        assert_eq!(mount.volume_options.as_ref().unwrap().subpath.as_deref(), Some("1234"));
        assert_eq!(mount.read_only, Some(true), "still read-only for the critic");
    }

    #[test]
    fn network_can_be_enabled_explicitly() {
        let config = SandboxConfig { network: true, ..SandboxConfig::default() };
        let body = config.container_body(Path::new("/w/p"));
        assert_eq!(body.host_config.unwrap().network_mode.as_deref(), Some("bridge"));
        assert_eq!(body.network_disabled, Some(false));
    }

    #[test]
    fn paths_are_translated_for_docker_in_docker() {
        let config = SandboxConfig {
            path_map: Some(PathMap { from: "/data/workspaces".into(), to: "/srv/hca/workspaces".into() }),
            ..SandboxConfig::default()
        };
        assert_eq!(
            config.host_path(Path::new("/data/workspaces/abc")),
            PathBuf::from("/srv/hca/workspaces/abc")
        );
        assert_eq!(config.host_path(Path::new("/elsewhere/abc")), PathBuf::from("/elsewhere/abc"));
    }

    #[test]
    fn container_names_are_safe() {
        assert_eq!(container_name(Path::new("/ws/3f2a9c1e-5b7d")), "hca-sb-3f2a9c1e-5b7d");
        assert_eq!(container_name(Path::new("/ws/a b;c")), "hca-sb-abc");
        assert_eq!(container_name(Path::new("/")), "hca-sb-workspace");
    }

    #[test]
    fn output_is_capped() {
        let mut buffer = Vec::new();
        append_capped(&mut buffer, b"hello", 3);
        append_capped(&mut buffer, b"world", 3);
        assert_eq!(buffer, b"hel");
    }

    /// Needs a Docker daemon and the sandbox image:
    /// `docker build -t hca-sandbox -f docker/sandbox.Dockerfile docker`
    /// then `cargo test -p hca-sandbox -- --ignored`.
    #[tokio::test]
    #[ignore = "needs a Docker daemon"]
    async fn runs_commands_in_an_isolated_container() {
        let sandbox =
            DockerSandbox::connect(SandboxConfig { user: "0:0".into(), ..SandboxConfig::default() })
                .await
                .expect("Docker is running");
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().join("sandbox-test");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("hello.py"), "print('hi')").unwrap();
        // Create this before the container starts. Docker makes the tmpfs
        // mount point as root, so the test user cannot fill it in afterwards.
        std::fs::create_dir_all(ws.join(".git/hooks")).unwrap();
        std::fs::write(ws.join(".git/hooks/pre-commit"), "#!/bin/sh\n").unwrap();

        let out =
            sandbox.run(&ws, "python hello.py", Duration::from_mins(1), Session::default()).await.unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert_eq!(out.stdout.trim(), "hi");

        let net = sandbox
            .run(&ws, "python -c \"import urllib.request; urllib.request.urlopen('http://example.com', timeout=3)\"", Duration::from_secs(30), Session::default())
            .await
            .unwrap();
        assert_ne!(net.exit_code, Some(0), "no network by default");

        let root =
            sandbox.run(&ws, "echo x > /evil", Duration::from_secs(30), Session::default()).await.unwrap();
        assert_ne!(root.exit_code, Some(0), "the root filesystem is read-only");
        let tmp = sandbox
            .run(&ws, "echo x > /tmp/ok && cat /tmp/ok", Duration::from_secs(30), Session::default())
            .await
            .unwrap();
        assert_eq!(tmp.stdout.trim(), "x", "/tmp is writable");

        // Kills and reaps its children before exiting, so the later checks
        // aren't starved of processes.
        std::fs::write(
            ws.join("forks.py"),
            "import os, signal, time\n\
             kids = []\n\
             try:\n    for _ in range(1000):\n        pid = os.fork()\n        if pid == 0:\n            time.sleep(60)\n            os._exit(0)\n        kids.append(pid)\n\
             except OSError:\n    print('limited at', len(kids))\n\
             for pid in kids:\n    os.kill(pid, signal.SIGKILL)\n\
             for pid in kids:\n    os.waitpid(pid, 0)\n",
        )
        .unwrap();
        let forks =
            sandbox.run(&ws, "python forks.py", Duration::from_mins(1), Session::default()).await.unwrap();
        assert!(forks.stdout.contains("limited at"), "the process limit applies: {forks:?}");
        let ok = sandbox
            .run(&ws, "echo still-forking", Duration::from_secs(30), Session::default())
            .await
            .unwrap();
        assert_eq!(ok.stdout.trim(), "still-forking", "processes are free again: {ok:?}");

        let memory = sandbox
            .run(
                &ws,
                "python -c \"b = bytearray(3 * 1024**3); print('allocated')\"",
                Duration::from_mins(1),
                Session::default(),
            )
            .await
            .unwrap();
        assert!(!memory.stdout.contains("allocated"), "3 GiB doesn't fit in the 2 GiB limit: {memory:?}");
        assert!(
            memory.stderr.contains("MemoryError") || memory.exit_code == Some(137),
            "the allocation failed for lack of memory: {memory:?}"
        );

        let slow = sandbox.run(&ws, "sleep 30", Duration::from_secs(2), Session::default()).await.unwrap();
        assert!(slow.timed_out, "the command was stopped at the timeout: {slow:?}");
        let after = sandbox.run(&ws, "echo back", Duration::from_secs(30), Session::default()).await.unwrap();
        assert_eq!(after.stdout.trim(), "back", "a fresh container takes over after a timeout");

        let git = sandbox
            .run(&ws, "cat .git/hooks/pre-commit", Duration::from_secs(30), Session::default())
            .await
            .unwrap();
        assert_ne!(git.exit_code, Some(0), "the host .git directory is covered by tmpfs: {git:?}");

        sandbox.remove(&ws).await.unwrap();
    }

    /// Needs Docker and the sandbox image, like the test above.
    #[tokio::test]
    #[ignore = "needs a Docker daemon"]
    async fn the_critic_is_isolated_and_containers_are_cleaned_up() {
        let sandbox = DockerSandbox::connect(SandboxConfig::default()).await.expect("Docker is running");
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().join("sandbox-critic-test");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("hello.py"), "print('hi')").unwrap();

        let critic_user = Session { id: "critic", user: Some(hca_agents::CRITIC_USER) };
        let coder = Session { id: "coder", user: None };
        let critic =
            sandbox.run(&ws, "echo pwned > hello.py", Duration::from_secs(30), critic_user).await.unwrap();
        assert_ne!(critic.exit_code, Some(0), "the critic user cannot overwrite the project: {critic:?}");
        assert_eq!(std::fs::read_to_string(ws.join("hello.py")).unwrap(), "print('hi')");
        let tests = sandbox.run(&ws, "python hello.py", Duration::from_secs(30), critic_user).await.unwrap();
        assert_eq!(tests.stdout.trim(), "hi", "the critic can still run the project: {tests:?}");

        // A Python startup hook in the critic's home folder must never run
        // for the coder, whose commands can write the project.
        let hook = "d=$(python -m site --user-site) && mkdir -p \"$d\" && \
                    printf 'open(\"/workspace/hello.py\", \"w\").write(\"pwned\")\\n' > \"$d/usercustomize.py\"";
        let planted = sandbox.run(&ws, hook, Duration::from_secs(30), critic_user).await.unwrap();
        assert_eq!(planted.exit_code, Some(0), "the critic may write its own /tmp: {planted:?}");
        let live = sandbox.run(&ws, "python -c pass", Duration::from_secs(30), critic_user).await.unwrap();
        assert!(
            live.stderr.contains("Read-only file system"),
            "the hook is real and runs for the critic: {live:?}"
        );
        sandbox.run(&ws, "python -c pass", Duration::from_secs(30), coder).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(ws.join("hello.py")).unwrap(),
            "print('hi')",
            "the critic's /tmp is not the coder's"
        );

        // Ending a session ends its background processes and its /tmp, and
        // nothing of any other session.
        let other = Session { id: "other", user: None };
        sandbox.run(&ws, "touch /tmp/mine", Duration::from_secs(30), other).await.unwrap();
        let spawn = "touch /tmp/left-over; nohup sleep 300 >/dev/null 2>&1 &";
        sandbox.run(&ws, spawn, Duration::from_secs(30), coder).await.unwrap();
        let processes = "ls /tmp; for p in /proc/[0-9]*; do cat $p/cmdline; echo; done";
        let apart = sandbox.run(&ws, processes, Duration::from_secs(30), other).await.unwrap();
        assert!(!apart.stdout.contains("sleep300"), "sessions don't see each other's processes: {apart:?}");
        sandbox.end_session(&ws, "coder").await.unwrap();
        let fresh = sandbox.run(&ws, processes, Duration::from_secs(30), coder).await.unwrap();
        assert!(!fresh.stdout.contains("left-over"), "/tmp starts empty: {fresh:?}");
        assert!(!fresh.stdout.contains("sleep300"), "no process outlives its session: {fresh:?}");
        let kept = sandbox.run(&ws, "ls /tmp", Duration::from_secs(30), other).await.unwrap();
        assert!(kept.stdout.contains("mine"), "the other session is untouched: {kept:?}");

        sandbox.remove(&ws).await.unwrap();
        let left = sandbox.labelled(&[format!("{WORKSPACE_LABEL}={}", ws.display())]).await.unwrap();
        assert!(left.is_empty(), "every container of the workspace is gone: {left:?}");

        // Startup sweep: only containers under the given folder go.
        sandbox.run(&ws, "true", Duration::from_secs(30), Session::default()).await.unwrap();
        assert_eq!(sandbox.sweep(Path::new("/somewhere/else")).await.unwrap(), 0);
        assert_eq!(sandbox.sweep(dir.path()).await.unwrap(), 1, "one container per session");
    }

    /// Needs Docker and the sandbox image, like the tests above.
    #[tokio::test]
    #[ignore = "needs a Docker daemon"]
    async fn removing_a_workspace_does_not_wait_for_its_commands() {
        let sandbox = DockerSandbox::connect(SandboxConfig::default()).await.expect("Docker is running");
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().join("sandbox-remove-test");
        std::fs::create_dir_all(&ws).unwrap();

        let running = {
            let (sandbox, ws) = (sandbox.clone(), ws.clone());
            tokio::spawn(async move {
                sandbox.run(&ws, "sleep 60", Duration::from_mins(2), Session::default()).await
            })
        };
        let name = container_name(&ws);
        for _ in 0..100 {
            let found = sandbox
                .docker
                .inspect_container(&name, None::<bollard::query_parameters::InspectContainerOptions>)
                .await;
            if found.is_ok_and(|info| info.state.and_then(|s| s.running).unwrap_or(false)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let started = std::time::Instant::now();
        tokio::time::timeout(Duration::from_secs(20), sandbox.remove(&ws))
            .await
            .expect("remove waited")
            .unwrap();
        let result =
            tokio::time::timeout(Duration::from_secs(20), running).await.expect("the command was stopped");
        assert!(started.elapsed() < Duration::from_secs(20));
        let output = result.unwrap();
        assert!(output.is_err() || output.as_ref().is_ok_and(|o| o.exit_code != Some(0)), "{output:?}");
    }

    /// Needs Docker and the sandbox image, like the tests above. Runs the
    /// whole sandbox lifecycle through the guard, then checks what it
    /// refuses against the real daemon.
    #[tokio::test]
    #[ignore = "needs a Docker daemon"]
    async fn the_sandbox_works_through_the_guard_and_nothing_else_does() {
        use bollard::models::VolumeCreateOptions;

        let direct = Docker::connect_with_defaults().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("workspaces");
        let ws = root.join("guard-test-project");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("hello.py"), "print('hi')").unwrap();
        let volume = format!("hca-guard-test-{}", std::process::id());
        let options = HashMap::from([
            ("type".to_owned(), "none".to_owned()),
            ("o".to_owned(), "bind".to_owned()),
            ("device".to_owned(), root.display().to_string()),
        ]);
        direct
            .create_volume(VolumeCreateOptions {
                name: Some(volume.clone()),
                driver: Some("local".into()),
                driver_opts: Some(options),
                ..VolumeCreateOptions::default()
            })
            .await
            .unwrap();

        let config = guard::GuardConfig {
            image: SandboxConfig::default().image,
            volume: volume.clone(),
            allow_network: false,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let serving = tokio::spawn(guard::Guard::new(config, "/var/run/docker.sock").serve(listener));
        let client =
            Docker::connect_with_http(&format!("http://{address}"), 120, bollard::API_DEFAULT_VERSION)
                .unwrap();
        let sandbox_config =
            SandboxConfig { workspace_volume: Some(volume.clone()), ..SandboxConfig::default() };
        let sandbox =
            DockerSandbox::connect_with(client.clone(), sandbox_config).await.expect("the guard answers");
        // Containers a failed earlier run of this test left behind.
        sandbox.remove(&ws).await.unwrap();

        let secs = Duration::from_secs;
        let coder = Session { id: "coder", user: None };
        let critic = Session { id: "critic", user: Some(hca_agents::CRITIC_USER) };
        let out = sandbox.run(&ws, "python hello.py", secs(60), coder).await.unwrap();
        assert_eq!(out.stdout.trim(), "hi", "commands run through the guard: {out:?}");
        let wrote = sandbox.run(&ws, "echo x > made.txt", secs(30), coder).await.unwrap();
        assert_eq!(wrote.exit_code, Some(0));
        assert!(ws.join("made.txt").exists(), "the coder writes the project");
        let denied = sandbox.run(&ws, "echo pwned > hello.py", secs(30), critic).await.unwrap();
        assert_ne!(denied.exit_code, Some(0), "the critic's mount is read-only: {denied:?}");
        let slow = sandbox.run(&ws, "sleep 30", secs(2), coder).await.unwrap();
        assert!(slow.timed_out, "timeouts kill through the guard: {slow:?}");
        sandbox.end_session(&ws, "coder").await.unwrap();
        sandbox.remove(&ws).await.unwrap();

        // Refused by the guard, against the real daemon.
        let images =
            client.list_images(None::<bollard::query_parameters::ListImagesOptions>).await.unwrap_err();
        assert_eq!(status(&images), Some(403), "{images}");
        let mut evil = SandboxConfig::default().container_body(&ws);
        evil.host_config.as_mut().unwrap().privileged = Some(true);
        let options = CreateContainerOptionsBuilder::new().name("hca-sb-evil").build();
        let refused = client.create_container(Some(options), evil).await.unwrap_err();
        assert_eq!(status(&refused), Some(403), "{refused}");

        // A project folder swapped for a symlink to / mounts nothing.
        std::fs::remove_dir_all(&ws).unwrap();
        std::os::unix::fs::symlink("/", &ws).unwrap();
        let escaped = sandbox.run(&ws, "ls /workspace", secs(30), coder).await;
        assert!(escaped.is_err(), "Docker refused the escaping subpath: {escaped:?}");

        std::fs::remove_file(&ws).unwrap();
        sandbox.sweep(&root).await.unwrap();
        serving.abort();
        direct.remove_volume(&volume, None::<bollard::query_parameters::RemoveVolumeOptions>).await.unwrap();
    }
}
