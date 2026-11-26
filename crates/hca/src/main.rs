//! `hca`: run the HCA agent team.
//!
//! ```text
//! hca init                 write hca.toml
//! hca serve                start the dashboard, API and workers
//! hca token                print the dashboard link with the access token
//! hca check                check the config, model keys and Docker
//! hca docker-guard         run the Docker guard the sandbox talks to
//! ```

mod config;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use hca_agents::{AgentConfig, CommandRunner};
use hca_orchestrator::{Orchestrator, OrchestratorConfig};
use hca_sandbox::guard::{Guard, GuardConfig};
use hca_sandbox::DockerSandbox;
use hca_server::AppState;
use hca_store::Store;
use tokio::sync::watch;

use crate::config::Config;

#[derive(Parser)]
#[command(version, about = "An AI development team that plans, builds and reviews software.")]
struct Cli {
    /// Path to the configuration file.
    #[arg(long, short, global = true, env = "HCA_CONFIG", default_value = "hca.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Write an example configuration file.
    Init {
        /// Overwrite an existing file.
        #[arg(long)]
        force: bool,
    },
    /// Start the dashboard, API and workers (the default).
    Serve,
    /// Print the dashboard link, including the access token.
    Token,
    /// Check the configuration, API keys and Docker, then exit.
    Check,
    /// Run the Docker guard: a proxy to the Docker socket that only lets
    /// sandbox containers through. See `docker-compose.yml`.
    DockerGuard {
        /// Address to listen on. Keep it on a network only HCA can reach.
        #[arg(long, env = "HCA_GUARD_LISTEN", default_value = "0.0.0.0:2375")]
        listen: std::net::SocketAddr,
        /// The Docker socket.
        #[arg(long, env = "HCA_GUARD_SOCKET", default_value = "/var/run/docker.sock")]
        socket: PathBuf,
        /// The only image containers may use.
        #[arg(long, env = "HCA_GUARD_IMAGE", default_value = "hca-sandbox:latest")]
        image: String,
        /// The only volume containers may mount: the workspaces volume.
        #[arg(long, env = "HCA_GUARD_VOLUME", default_value = "hca-workspaces")]
        volume: String,
        /// Let containers have a network, for `[sandbox] network = true`.
        #[arg(long, env = "HCA_GUARD_ALLOW_NETWORK")]
        allow_network: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,sqlx=warn")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Serve) {
        Command::Init { force } => init(&cli.config, force),
        Command::Serve => serve(&cli.config).await,
        Command::Token => {
            let config = Config::load(&cli.config)?;
            let token = access_token(&config.storage.data_dir)?;
            println!("{}", dashboard_link(&config, &token));
            Ok(())
        }
        Command::Check => check(&cli.config).await,
        Command::DockerGuard { listen, socket, image, volume, allow_network } => {
            let listener = tokio::net::TcpListener::bind(listen)
                .await
                .with_context(|| format!("could not listen on {listen}"))?;
            tracing::info!(%listen, socket = %socket.display(), %image, %volume, allow_network, "Docker guard ready");
            let guard = Guard::new(GuardConfig { image, volume, allow_network }, socket);
            tokio::select! {
                result = guard.serve(listener) => result.context("the Docker guard stopped"),
                () = shutdown_signal() => Ok(()),
            }
        }
    }
}

fn init(path: &Path, force: bool) -> anyhow::Result<()> {
    if path.exists() && !force {
        bail!("{} already exists; use --force to overwrite it", path.display());
    }
    std::fs::write(path, config::EXAMPLE).with_context(|| format!("could not write {}", path.display()))?;
    println!("Wrote {}. Edit the [llm] section, then run `hca serve`.", path.display());
    Ok(())
}

async fn check(path: &Path) -> anyhow::Result<()> {
    let config = Config::load(path)?;
    println!("config: ok ({})", path.display());
    config.llm.build().context("model configuration")?;
    println!("models: ok ({} roles configured)", config.llm.roles.len());
    for warning in config.warnings() {
        println!("warning: {warning}");
    }
    if config.sandbox.enabled {
        match DockerSandbox::connect(config.sandbox.container.clone()).await {
            Ok(_) => println!("sandbox: ok (image {})", config.sandbox.container.image),
            Err(e) => println!("sandbox: unavailable ({e}); agents won't be able to run commands"),
        }
    } else {
        println!("sandbox: disabled; agents won't be able to run commands");
    }
    Ok(())
}

async fn serve(path: &Path) -> anyhow::Result<()> {
    let config = Config::load(path)?;
    let data_dir = &config.storage.data_dir;
    std::fs::create_dir_all(data_dir).with_context(|| format!("could not create {}", data_dir.display()))?;

    let store = Store::open(data_dir.join("hca.db")).await.context("could not open the database")?;
    let llm = config.llm.build().context("model configuration")?;
    let sandbox = sandbox(&config).await;
    let runner = sandbox.clone().map(|s| Arc::new(s) as Arc<dyn CommandRunner>);

    let mut orchestrator_config = OrchestratorConfig::new(data_dir.join("workspaces"));
    orchestrator_config.limits = config.limits;
    orchestrator_config.workers = config.orchestrator.workers;
    orchestrator_config.brief_tokens = config.orchestrator.brief_tokens;
    orchestrator_config.agent =
        AgentConfig { max_steps: config.orchestrator.max_steps, ..AgentConfig::default() };
    orchestrator_config.run_timeout =
        std::time::Duration::from_secs(u64::from(config.orchestrator.run_timeout_minutes) * 60);
    for warning in config.warnings() {
        tracing::warn!("{warning}");
    }
    let orchestrator = Arc::new(Orchestrator::new(store.clone(), llm, runner, orchestrator_config));
    orchestrator.recover().await.context("could not recover unfinished work")?;

    let (stop_workers, workers_stopped) = watch::channel(false);
    let workers = orchestrator.spawn_workers(&workers_stopped);

    let token = access_token(data_dir)?;
    let app =
        hca_server::router(AppState { orchestrator: orchestrator.clone(), token: Arc::from(token.as_str()) });
    let listener = tokio::net::TcpListener::bind(config.server.bind)
        .await
        .with_context(|| format!("could not listen on {}", config.server.bind))?;

    if !config.server.bind.ip().is_loopback() {
        tracing::warn!(bind = %config.server.bind, "the dashboard is reachable from other machines; keep the access token secret");
    }
    println!("\nHCA is running. Open the dashboard:\n\n    {}\n", dashboard_link(&config, &token));

    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await.context("server error")?;

    tracing::info!("stopping: running agents finish their current model call, then their jobs are requeued");
    orchestrator.begin_shutdown();
    let _ = stop_workers.send(true);
    let aborts: Vec<_> = workers.iter().map(tokio::task::JoinHandle::abort_handle).collect();
    let all_stopped = async {
        for worker in workers {
            let _ = worker.await;
        }
    };
    if tokio::time::timeout(SHUTDOWN_GRACE, all_stopped).await.is_err() {
        // A model call can take minutes. Stop waiting: the call's tokens
        // are already charged (see the orchestrator's meter), and its job
        // goes back on the queue below.
        tracing::warn!(
            "an agent was still waiting on a model; stopping it, its job resumes at the next start"
        );
        for abort in &aborts {
            abort.abort();
        }
        // Let the aborted tasks unwind, so their meters and guards run.
        tokio::task::yield_now().await;
    }
    match store.requeue_interrupted(chrono::Utc::now()).await {
        Ok(0) => {}
        Ok(requeued) => tracing::info!(requeued, "interrupted jobs are back on the queue"),
        Err(error) => {
            tracing::warn!(%error, "could not requeue interrupted jobs; they resume at the next start");
        }
    }
    if let Some(sandbox) = &sandbox {
        if let Err(error) = sandbox.sweep(&workspaces_root(&config)).await {
            tracing::warn!(%error, "could not remove sandbox containers; they are removed at the next start");
        }
    }
    store.close().await;
    Ok(())
}

/// How long shutdown waits for running agents before stopping them. Docker
/// Compose gives 45 s (`stop_grace_period`) before it kills the process,
/// which leaves time to requeue jobs and remove containers.
const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(20);

/// The workspaces folder, created if missing, as an absolute path.
fn workspaces_root(config: &Config) -> std::path::PathBuf {
    let workspaces = config.storage.data_dir.join("workspaces");
    let _ = std::fs::create_dir_all(&workspaces);
    workspaces.canonicalize().unwrap_or(workspaces)
}

async fn sandbox(config: &Config) -> Option<DockerSandbox> {
    if !config.sandbox.enabled {
        tracing::warn!("sandbox disabled: agents can't run commands or tests");
        return None;
    }
    match DockerSandbox::connect(config.sandbox.container.clone()).await {
        Ok(sandbox) => {
            tracing::info!(image = %config.sandbox.container.image, "sandbox ready");
            // Containers from an earlier run hold processes nobody waits
            // for; projects get fresh ones on their next command.
            match sandbox.sweep(&workspaces_root(config)).await {
                Ok(0) => {}
                Ok(removed) => tracing::info!(removed, "removed sandbox containers left by an earlier run"),
                Err(error) => tracing::warn!(%error, "could not clean up old sandbox containers"),
            }
            Some(sandbox)
        }
        Err(error) => {
            tracing::warn!(%error, "Docker unavailable: agents can't run commands or tests");
            None
        }
    }
}

/// The access token: `HCA_TOKEN` if set, otherwise the one stored in the
/// data folder, created on first run with owner-only permissions.
fn access_token(data_dir: &Path) -> anyhow::Result<String> {
    if let Ok(token) = std::env::var("HCA_TOKEN") {
        if token.trim().len() < 16 {
            bail!("HCA_TOKEN must be at least 16 characters");
        }
        return Ok(token.trim().to_owned());
    }
    let path = data_dir.join("access_token");
    if let Ok(token) = std::fs::read_to_string(&path) {
        if !token.trim().is_empty() {
            restrict_token_file(&path)?;
            return Ok(token.trim().to_owned());
        }
    }
    std::fs::create_dir_all(data_dir)?;
    let token = format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple());
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path).with_context(|| format!("could not write {}", path.display()))?;
    file.write_all(token.as_bytes())?;
    restrict_token_file(&path)?;
    tracing::info!(path = %path.display(), "created a new access token");
    Ok(token)
}

/// `OpenOptions::mode` only applies when the file is created. A token file
/// that already exists (for example an empty file left world-readable) is
/// tightened here too.
fn restrict_token_file(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("could not restrict {}", path.display()))?;
    }
    Ok(())
}

fn dashboard_link(config: &Config, token: &str) -> String {
    let bind = config.server.bind;
    let host = if bind.ip().is_unspecified() { "127.0.0.1".to_owned() } else { bind.ip().to_string() };
    format!("http://{host}:{}/#token={}", bind.port(), percent_encode(token))
}

/// Percent-encodes `token` for a URL fragment. Unreserved characters stay as they are.
fn percent_encode(token: &str) -> String {
    let mut encoded = String::with_capacity(token.len());
    for byte in token.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                encoded.push('%');
                encoded.push(char::from(HEX[usize::from(byte >> 4)]));
                encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
            }
        }
    }
    encoded
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_created_once_and_kept_private() {
        let dir = tempfile::tempdir().unwrap();
        let first = access_token(dir.path()).unwrap();
        let second = access_token(dir.path()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.path().join("access_token")).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn the_link_points_at_a_local_address() {
        let mut config = Config::parse(config::EXAMPLE).unwrap();
        assert_eq!(dashboard_link(&config, "abc"), "http://127.0.0.1:8080/#token=abc");
        config.server.bind = "0.0.0.0:9000".parse().unwrap();
        assert_eq!(dashboard_link(&config, "abc"), "http://127.0.0.1:9000/#token=abc");
        assert_eq!(dashboard_link(&config, "a%#&"), "http://127.0.0.1:9000/#token=a%25%23%26");
    }

    #[test]
    fn an_existing_token_file_is_made_private() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("access_token");
        std::fs::write(&path, "abcdefghijklmnop").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        }
        assert_eq!(access_token(dir.path()).unwrap(), "abcdefghijklmnop");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }
}
