//! `hca`: run the HCA agent team.
//!
//! ```text
//! hca init                 write hca.toml
//! hca serve                start the dashboard, API and workers
//! hca token                print the dashboard link with the access token
//! hca check                check the config, model keys and Docker
//! ```

mod config;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use hca_agents::{AgentConfig, CommandRunner};
use hca_orchestrator::{Orchestrator, OrchestratorConfig};
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
    let runner = sandbox(&config).await;

    let mut orchestrator_config = OrchestratorConfig::new(data_dir.join("workspaces"));
    orchestrator_config.limits = config.limits;
    orchestrator_config.workers = config.orchestrator.workers;
    orchestrator_config.brief_tokens = config.orchestrator.brief_tokens;
    orchestrator_config.agent =
        AgentConfig { max_steps: config.orchestrator.max_steps, ..AgentConfig::default() };
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

    tracing::info!("stopping workers; running agents finish their current step");
    let _ = stop_workers.send(true);
    for worker in workers {
        let _ = worker.await;
    }
    store.close().await;
    Ok(())
}

async fn sandbox(config: &Config) -> Option<Arc<dyn CommandRunner>> {
    if !config.sandbox.enabled {
        tracing::warn!("sandbox disabled: agents can't run commands or tests");
        return None;
    }
    match DockerSandbox::connect(config.sandbox.container.clone()).await {
        Ok(sandbox) => {
            tracing::info!(image = %config.sandbox.container.image, "sandbox ready");
            Some(Arc::new(sandbox))
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
    tracing::info!(path = %path.display(), "created a new access token");
    Ok(token)
}

fn dashboard_link(config: &Config, token: &str) -> String {
    let bind = config.server.bind;
    let host = if bind.ip().is_unspecified() { "127.0.0.1".to_owned() } else { bind.ip().to_string() };
    format!("http://{host}:{}/#token={token}", bind.port())
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
    }
}
