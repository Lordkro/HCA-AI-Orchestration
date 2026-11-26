//! The `hca.toml` configuration file.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use hca_core::Limits;
use hca_llm::LlmConfig;
use hca_sandbox::SandboxConfig;
use serde::{Deserialize, Serialize};

/// Everything in `hca.toml`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Where the server listens.
    #[serde(default)]
    pub server: ServerConfig,
    /// Where data lives.
    #[serde(default)]
    pub storage: StorageConfig,
    /// Project and task limits.
    #[serde(default)]
    pub limits: Limits,
    /// Worker and agent settings.
    #[serde(default)]
    pub orchestrator: OrchestratorSection,
    /// The command sandbox.
    #[serde(default)]
    pub sandbox: SandboxSection,
    /// Model providers and the model for each role.
    pub llm: LlmConfig,
}

/// `[server]`
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Address and port. Defaults to `127.0.0.1:8080`, reachable only from
    /// this machine.
    pub bind: SocketAddr,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { bind: SocketAddr::from(([127, 0, 0, 1], 8080)) }
    }
}

/// `[storage]`
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageConfig {
    /// Holds the database, the access token and the project workspaces.
    pub data_dir: PathBuf,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self { data_dir: PathBuf::from("data") }
    }
}

/// `[orchestrator]`
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OrchestratorSection {
    /// Jobs processed at the same time.
    pub workers: usize,
    /// Model calls allowed per agent run.
    pub max_steps: u32,
    /// Token budget for each agent's brief.
    pub brief_tokens: u64,
}

impl Default for OrchestratorSection {
    fn default() -> Self {
        Self { workers: 2, max_steps: 40, brief_tokens: 8_000 }
    }
}

/// `[sandbox]`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SandboxSection {
    /// Use Docker to run agent commands. If Docker can't be reached, HCA
    /// starts anyway and agents can't run commands.
    pub enabled: bool,
    /// Container settings.
    #[serde(flatten)]
    pub container: SandboxConfig,
}

impl Default for SandboxSection {
    fn default() -> Self {
        Self { enabled: true, container: SandboxConfig::default() }
    }
}

impl Config {
    /// Reads and checks a config file.
    ///
    /// # Errors
    ///
    /// Returns an error naming the file if it can't be read, isn't valid
    /// TOML, or has invalid limits.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("could not read {}; run `hca init` to create one", path.display()))?;
        let mut config =
            Self::parse(&text).with_context(|| format!("invalid configuration in {}", path.display()))?;
        config.apply_overrides(|name| std::env::var(name).ok())?;
        Ok(config)
    }

    /// Applies `HCA_BIND` and `HCA_DATA_DIR` from the environment, for
    /// container setups where the file is shared but paths differ.
    ///
    /// # Errors
    ///
    /// Returns an error if `HCA_BIND` isn't an address and port.
    pub fn apply_overrides(&mut self, env: impl Fn(&str) -> Option<String>) -> anyhow::Result<()> {
        if let Some(bind) = env("HCA_BIND") {
            self.server.bind =
                bind.parse().with_context(|| format!("HCA_BIND={bind:?} is not an address:port"))?;
        }
        if let Some(dir) = env("HCA_DATA_DIR") {
            self.storage.data_dir = PathBuf::from(dir);
        }
        Ok(())
    }

    /// Parses and checks config text.
    ///
    /// # Errors
    ///
    /// Returns an error if the text isn't valid or the limits are unusable.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let config: Self = toml::from_str(text)?;
        config.limits.validate()?;
        if config.llm.roles.is_empty() {
            bail!("[llm.roles] is empty: configure a model for at least the pm, coder and critic roles");
        }
        if config.orchestrator.workers == 0 {
            bail!("[orchestrator] workers must be at least 1");
        }
        Ok(config)
    }
}

/// The config written by `hca init`.
pub const EXAMPLE: &str = include_str!("../../../hca.example.toml");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_config_is_valid() {
        let config = Config::parse(EXAMPLE).unwrap();
        assert!(config.server.bind.ip().is_loopback(), "the example only listens locally");
        assert!(config.sandbox.enabled);
        assert!(!config.sandbox.container.network, "sandboxes have no network by default");
        assert_eq!(config.limits, Limits::default());
        assert!(config.llm.roles.contains_key(&hca_core::AgentRole::Coder));
    }

    #[test]
    fn minimal_config_uses_safe_defaults() {
        let config = Config::parse(
            r#"
            [llm.providers.local]
            kind = "ollama"

            [llm.roles.coder]
            provider = "local"
            model = "qwen3:8b"
            "#,
        )
        .unwrap();
        assert_eq!(config.server.bind.to_string(), "127.0.0.1:8080");
        assert_eq!(config.storage.data_dir, PathBuf::from("data"));
        assert_eq!(config.orchestrator.workers, 2);
    }

    #[test]
    fn environment_overrides_paths_and_address() {
        let mut config = Config::parse(EXAMPLE).unwrap();
        config
            .apply_overrides(|name| match name {
                "HCA_BIND" => Some("0.0.0.0:9000".into()),
                "HCA_DATA_DIR" => Some("/srv/hca".into()),
                _ => None,
            })
            .unwrap();
        assert_eq!(config.server.bind.to_string(), "0.0.0.0:9000");
        assert_eq!(config.storage.data_dir, PathBuf::from("/srv/hca"));
        assert!(config.apply_overrides(|_| Some("nonsense".into())).is_err());
    }

    #[test]
    fn typos_are_reported() {
        let err =
            Config::parse("[servr]\nbind = \"0.0.0.0:80\"\n[llm]\nproviders = {}\nroles = {}\n").unwrap_err();
        assert!(format!("{err:#}").contains("servr"), "{err:#}");
    }

    #[test]
    fn unusable_limits_are_rejected() {
        let err = Config::parse(
            "[limits]\nmax_parallel_tasks = 0\n[llm.providers.l]\nkind = \"ollama\"\n[llm.roles.coder]\nprovider = \"l\"\nmodel = \"m\"\n",
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("max_parallel_tasks"), "{err:#}");
    }
}
