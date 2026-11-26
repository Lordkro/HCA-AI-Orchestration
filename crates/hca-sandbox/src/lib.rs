//! Isolated, resource-limited Docker containers for agent commands.
//!
//! Each project gets one long-lived container with only its workspace
//! mounted. Commands run inside it through `docker exec`. By default the
//! container has:
//!
//! - **no network**, so generated code can't download or exfiltrate
//!   anything;
//! - **memory, CPU and process limits**, so a fork bomb or a memory leak
//!   can't take the host down;
//! - **all Linux capabilities dropped**, `no-new-privileges`, a read-only
//!   root filesystem with a small writable `/tmp`, and a non-root user.
//!
//! A command that runs past its timeout is stopped by killing the
//! container; the next command starts a fresh one. This crate talks to the
//! Docker API directly, so the orchestrator image doesn't need the `docker`
//! CLI.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use bollard::container::LogOutput;
use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};
use bollard::models::{ContainerCreateBody, HostConfig};
use bollard::query_parameters::{
    CreateContainerOptionsBuilder, KillContainerOptions, RemoveContainerOptionsBuilder, StartContainerOptions,
};
use bollard::Docker;
use futures_util::StreamExt;
use hca_agents::{CommandOutput, CommandRunner};
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
    /// Path translation for Docker-in-Docker setups.
    pub path_map: Option<PathMap>,
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
        }
    }
}

/// Where the workspace is mounted inside the container.
pub const CONTAINER_WORKSPACE: &str = "/workspace";

/// Label put on every sandbox container, so they can be found and cleaned up.
pub const LABEL: &str = "hca.sandbox";

/// Label holding the settings a container was created with.
///
/// A running container is reused only when this matches. Anything else is
/// removed and created again, so turning the network off actually applies.
const CONFIG_LABEL: &str = "hca.config";

impl SandboxConfig {
    /// Settings that decide whether an existing container can be reused.
    ///
    /// Includes a marker that `/workspace/.git` is covered by a tmpfs, so
    /// containers created before that mount are replaced.
    fn fingerprint(&self) -> String {
        format!(
            "image={};memory={};cpus={};pids={};network={};user={};tmp={};git-tmpfs=1",
            self.image, self.memory_mb, self.cpus, self.pids_limit, self.network, self.user, self.tmp_mb
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

    /// The container definition for `workspace`.
    pub fn container_body(&self, workspace: &Path) -> ContainerCreateBody {
        let mib = |n: u64| i64::try_from(n.saturating_mul(1024 * 1024)).unwrap_or(i64::MAX);
        let host = self.host_path(workspace);
        #[allow(clippy::cast_possible_truncation)]
        let nano_cpus = (self.cpus.max(0.01) * 1e9).round() as i64;
        ContainerCreateBody {
            image: Some(self.image.clone()),
            cmd: Some(vec!["sleep".into(), "infinity".into()]),
            working_dir: Some(CONTAINER_WORKSPACE.into()),
            user: Some(self.user.clone()),
            env: Some(vec!["HOME=/tmp".into(), "PYTHONDONTWRITEBYTECODE=1".into()]),
            network_disabled: Some(!self.network),
            labels: Some(HashMap::from([
                (LABEL.to_owned(), "1".to_owned()),
                (CONFIG_LABEL.to_owned(), self.fingerprint()),
                ("hca.workspace".to_owned(), workspace.display().to_string()),
            ])),
            host_config: Some(HostConfig {
                binds: Some(vec![format!("{}:{CONTAINER_WORKSPACE}:rw", host.display())]),
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

/// The container name for a workspace: `hca-sb-` plus the folder name
/// (the project id), restricted to characters Docker allows.
pub fn container_name(workspace: &Path) -> String {
    let base = workspace.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let clean: String =
        base.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')).take(64).collect();
    if clean.is_empty() {
        "hca-sb-workspace".into()
    } else {
        format!("hca-sb-{clean}")
    }
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
}

/// Runs agent commands in per-project containers.
#[derive(Clone, Debug)]
pub struct DockerSandbox {
    docker: Docker,
    config: SandboxConfig,
    /// One lock per container, so a timeout can't kill a sibling exec.
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
        docker.ping().await.map_err(|e| SandboxError::Unavailable(e.to_string()))?;
        Ok(Self { docker, config, locks: Arc::new(Mutex::new(HashMap::new())) })
    }

    fn lock_for(&self, name: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock().unwrap_or_else(PoisonError::into_inner);
        locks.entry(name.to_owned()).or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))).clone()
    }

    /// The settings in use.
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }

    /// Removes a workspace's container, if it exists.
    ///
    /// # Errors
    ///
    /// Returns the Docker error, except "no such container".
    pub async fn remove(&self, workspace: &Path) -> Result<(), SandboxError> {
        self.remove_named(&container_name(workspace)).await
    }

    async fn remove_named(&self, name: &str) -> Result<(), SandboxError> {
        let options = RemoveContainerOptionsBuilder::new().force(true).build();
        match self.docker.remove_container(name, Some(options)).await {
            Err(e) if status(&e) == Some(404) => Ok(()),
            other => Ok(other?),
        }
    }

    async fn ensure_running(&self, name: &str, workspace: &Path) -> Result<(), SandboxError> {
        let want = self.config.fingerprint();
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
        let options = CreateContainerOptionsBuilder::new().name(name).build();
        match self.docker.create_container(Some(options), self.config.container_body(workspace)).await {
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
        user: Option<&str>,
    ) -> Result<CommandOutput, SandboxError> {
        let name = container_name(workspace);
        let lock = self.lock_for(&name);
        let _guard = lock.lock().await;
        self.ensure_running(&name, workspace).await?;

        let exec = self
            .docker
            .create_exec(
                &name,
                CreateExecOptions {
                    cmd: Some(vec!["sh", "-c", command]),
                    attach_stdout: Some(true),
                    attach_stderr: Some(true),
                    working_dir: Some(CONTAINER_WORKSPACE),
                    user,
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
            // The per-workspace lock means this doesn't hit a sibling exec,
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
        user: Option<&'a str>,
    ) -> BoxFuture<'a, Result<CommandOutput, String>> {
        Box::pin(async move { self.exec(workspace, command, timeout, user).await.map_err(|e| e.to_string()) })
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

        let out = sandbox.run(&ws, "python hello.py", Duration::from_mins(1), None).await.unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert_eq!(out.stdout.trim(), "hi");

        let net = sandbox
            .run(&ws, "python -c \"import urllib.request; urllib.request.urlopen('http://example.com', timeout=3)\"", Duration::from_secs(30), None)
            .await
            .unwrap();
        assert_ne!(net.exit_code, Some(0), "no network by default");

        let root = sandbox.run(&ws, "echo x > /evil", Duration::from_secs(30), None).await.unwrap();
        assert_ne!(root.exit_code, Some(0), "the root filesystem is read-only");
        let tmp =
            sandbox.run(&ws, "echo x > /tmp/ok && cat /tmp/ok", Duration::from_secs(30), None).await.unwrap();
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
        let forks = sandbox.run(&ws, "python forks.py", Duration::from_mins(1), None).await.unwrap();
        assert!(forks.stdout.contains("limited at"), "the process limit applies: {forks:?}");
        let ok = sandbox.run(&ws, "echo still-forking", Duration::from_secs(30), None).await.unwrap();
        assert_eq!(ok.stdout.trim(), "still-forking", "processes are free again: {ok:?}");

        let memory = sandbox
            .run(
                &ws,
                "python -c \"b = bytearray(3 * 1024**3); print('allocated')\"",
                Duration::from_mins(1),
                None,
            )
            .await
            .unwrap();
        assert!(!memory.stdout.contains("allocated"), "3 GiB doesn't fit in the 2 GiB limit: {memory:?}");
        assert!(
            memory.stderr.contains("MemoryError") || memory.exit_code == Some(137),
            "the allocation failed for lack of memory: {memory:?}"
        );

        let slow = sandbox.run(&ws, "sleep 30", Duration::from_secs(2), None).await.unwrap();
        assert!(slow.timed_out, "the command was stopped at the timeout: {slow:?}");
        let after = sandbox.run(&ws, "echo back", Duration::from_secs(30), None).await.unwrap();
        assert_eq!(after.stdout.trim(), "back", "a fresh container takes over after a timeout");

        let critic = sandbox
            .run(&ws, "echo pwned > hello.py", Duration::from_secs(30), Some(hca_agents::CRITIC_USER))
            .await
            .unwrap();
        assert_ne!(critic.exit_code, Some(0), "the critic user cannot overwrite the project: {critic:?}");
        assert_eq!(std::fs::read_to_string(ws.join("hello.py")).unwrap(), "print('hi')");

        let git = sandbox.run(&ws, "cat .git/hooks/pre-commit", Duration::from_secs(30), None).await.unwrap();
        assert_ne!(git.exit_code, Some(0), "the host .git directory is covered by tmpfs: {git:?}");

        sandbox.remove(&ws).await.unwrap();
    }
}
