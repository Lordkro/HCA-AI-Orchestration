//! The `hca.toml` configuration file.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use hca_core::{AgentRole, Limits};
use hca_llm::{LlmConfig, ProviderConfig, ASSUMED_CONTEXT_TOKENS, DEFAULT_OLLAMA_NUM_CTX};
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
    /// Longest one agent run may take before it fails as timed out.
    pub run_timeout_minutes: u32,
}

impl Default for OrchestratorSection {
    fn default() -> Self {
        Self { workers: 2, max_steps: 40, brief_tokens: 8_000, run_timeout_minutes: 120 }
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

    /// Applies `HCA_BIND`, `HCA_DATA_DIR` and `HCA_SANDBOX_VOLUME` from the
    /// environment, for container setups where the file is shared but paths
    /// differ.
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
        if let Some(volume) = env("HCA_SANDBOX_VOLUME") {
            self.sandbox.container.workspace_volume = Some(volume).filter(|v| !v.is_empty());
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
        let missing: Vec<&str> = REQUIRED_ROLES
            .iter()
            .filter(|role| !config.llm.roles.contains_key(role))
            .map(|role| role.as_str())
            .collect();
        if !missing.is_empty() {
            bail!(
                "[llm.roles] has no model for {}: the pm, coder and critic roles are required \
                 (research and spec use the coder's model when they have none)",
                missing.join(", ")
            );
        }
        if config.orchestrator.workers == 0 {
            bail!("[orchestrator] workers must be at least 1");
        }
        if config.orchestrator.run_timeout_minutes == 0 {
            bail!("[orchestrator] run_timeout_minutes must be at least 1");
        }
        Ok(config)
    }

    /// Settings that work but will likely cause trouble, as sentences.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        for (role, settings) in &self.llm.roles {
            let Some(provider) = self.llm.providers.get(&settings.provider) else { continue };
            let window = match provider {
                ProviderConfig::Ollama { num_ctx, .. } => num_ctx.unwrap_or(DEFAULT_OLLAMA_NUM_CTX),
                ProviderConfig::OpenaiCompatible { context_tokens: Some(n), .. } => *n,
                ProviderConfig::OpenaiCompatible { context_tokens: None, .. } => {
                    warnings.push(format!(
                        "provider {:?} (used by {role}) has no context_tokens, so HCA assumes a \
                         {ASSUMED_CONTEXT_TOKENS}-token window; set context_tokens to the server's real one",
                        settings.provider
                    ));
                    ASSUMED_CONTEXT_TOKENS
                }
                _ => continue,
            };
            let window = u64::from(window);
            // The brief, the system prompt and tool definitions (~3,000),
            // the output allowance, and room to work.
            let needed = self.orchestrator.brief_tokens + 3_000 + u64::from(settings.max_tokens) + 4_000;
            if window < needed {
                warnings.push(format!(
                    "{role} has a {window}-token context window, but its brief, output limit ({}) and \
                     working room need about {needed}; raise the context window or lower max_tokens",
                    settings.max_tokens
                ));
            }
        }
        warnings
    }
}

/// Roles every project needs: planning, coding and review.
const REQUIRED_ROLES: [AgentRole; 3] = [AgentRole::Pm, AgentRole::Coder, AgentRole::Critic];

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

            [llm.roles.pm]
            provider = "local"
            model = "qwen3:8b"

            [llm.roles.coder]
            provider = "local"
            model = "qwen3:8b"

            [llm.roles.critic]
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
                "HCA_SANDBOX_VOLUME" => Some("hca-workspaces".into()),
                _ => None,
            })
            .unwrap();
        assert_eq!(config.server.bind.to_string(), "0.0.0.0:9000");
        assert_eq!(config.storage.data_dir, PathBuf::from("/srv/hca"));
        assert_eq!(config.sandbox.container.workspace_volume.as_deref(), Some("hca-workspaces"));
        assert!(config.apply_overrides(|_| Some("nonsense".into())).is_err());
    }

    #[test]
    fn small_context_windows_are_flagged() {
        let config = Config::parse(EXAMPLE).unwrap();
        assert_eq!(config.warnings(), Vec::<String>::new(), "the example fits");

        let small = EXAMPLE.replace("num_ctx = 32768", "num_ctx = 8192");
        let warnings = Config::parse(&small).unwrap().warnings();
        assert!(warnings.iter().any(|w| w.contains("coder has a 8192-token context window")), "{warnings:?}");

        let unset = EXAMPLE.replace("num_ctx = 32768\n", "");
        assert_eq!(Config::parse(&unset).unwrap().warnings(), Vec::<String>::new(), "32768 is sent anyway");

        let compatible = Config::parse(
            "[llm.providers.vllm]\nkind = \"openai_compatible\"\nbase_url = \"http://vllm:8000/v1\"\n\
             [llm.roles.pm]\nprovider = \"vllm\"\nmodel = \"m\"\n[llm.roles.coder]\nprovider = \"vllm\"\n\
             model = \"m\"\n[llm.roles.critic]\nprovider = \"vllm\"\nmodel = \"m\"\n",
        )
        .unwrap();
        let warnings = compatible.warnings();
        assert!(warnings.iter().any(|w| w.contains("assumes a 32768-token window")), "{warnings:?}");
    }

    #[test]
    fn the_core_roles_are_required() {
        let err = Config::parse(
            "[llm.providers.l]\nkind = \"ollama\"\n[llm.roles.coder]\nprovider = \"l\"\nmodel = \"m\"\n",
        )
        .unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("pm, critic"), "{message}");
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
