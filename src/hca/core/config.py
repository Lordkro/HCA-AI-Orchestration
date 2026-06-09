"""Configuration management for HCA Orchestration."""

from pydantic_settings import BaseSettings

# ──────────────────────────────────────────────
# Hardware compatibility reference
# Maps VRAM tiers to recommended model combos.
# Used by the health endpoint & hardware guide.
# ──────────────────────────────────────────────
HARDWARE_TIERS: dict[str, dict[str, str | int]] = {
    "high": {
        "vram": "≥24 GB",
        "default_model": "qwen3:14b",
        "coder_model": "qwen2.5-coder:14b",
        "quantization": "Q4_K_M",
        "num_ctx": 8192,
    },
    "medium": {
        "vram": "12-24 GB",
        "default_model": "qwen3:8b",
        "coder_model": "qwen2.5-coder:7b",
        "quantization": "Q4_K_M",
        "num_ctx": 8192,
    },
    "low": {
        "vram": "8-12 GB",
        "default_model": "llama3.2:3b",
        "coder_model": "qwen2.5-coder:3b",
        "quantization": "Q4_K_M",
        "num_ctx": 4096,
    },
    "minimal": {
        "vram": "6-8 GB",
        "default_model": "phi-4:latest",
        "coder_model": "phi-4:latest",
        "quantization": "Q4_K_M",
        "num_ctx": 4096,
    },
    "tiny": {
        "vram": "<6 GB",
        "default_model": "llama3.2:1b",
        "coder_model": "qwen2.5-coder:1.5b",
        "quantization": "Q4_K_M",
        "num_ctx": 2048,
    },
}

HARDWARE_BACKENDS: dict[str, dict[str, str]] = {
    "cpu": {
        "label": "CPU-only",
        "image": "ollama/ollama",
        "tag": "latest",
        "compose_profile": "cpu",
        "note": "Slowest but most compatible.",
    },
    "nvidia": {
        "label": "NVIDIA CUDA",
        "image": "ollama/ollama",
        "tag": "latest",
        "compose_profile": "nvidia",
        "note": "Requires nvidia-container-toolkit & --profile nvidia.",
    },
    "rocm": {
        "label": "AMD ROCm",
        "image": "ollama/ollama:rocm",
        "tag": "rocm",
        "compose_profile": "rocm",
        "note": "ROCm-only image; /dev/kfd + /dev/dri passthrough.",
    },
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Ollama ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_default_model: str = "qwen3:14b"
    ollama_default_temperature: float = 0.7
    ollama_default_top_p: float = 0.9
    ollama_coder_model: str = "qwen2.5-coder:14b"  # Default coder model (fallback)
    ollama_timeout: int = 120
    ollama_num_ctx: int = 8192
    ollama_max_retries: int = 3
    ollama_retry_base_delay: float = 2.0
    ollama_max_concurrent: int = 1  # Max parallel LLM calls
    ollama_circuit_breaker_failure_threshold: int = 5  # Failures before circuit opens
    ollama_circuit_breaker_recovery_timeout: int = 60  # Seconds before retry after open

    # Per-agent model overrides (empty string = use default)
    ollama_pm_model: str = ""
    ollama_research_model: str = ""
    ollama_spec_model: str = ""
    ollama_coder_model_override: str = ""  # If set, overrides ollama_coder_model
    ollama_critic_model: str = ""

    # Per-agent temperature overrides (0 = use default)
    ollama_pm_temperature: float = 0.0
    ollama_research_temperature: float = 0.0
    ollama_spec_temperature: float = 0.0
    ollama_coder_temperature: float = 0.0
    ollama_critic_temperature: float = 0.0

    # Per-agent top_p overrides (0 = use default)
    ollama_pm_top_p: float = 0.0
    ollama_research_top_p: float = 0.0
    ollama_spec_top_p: float = 0.0
    ollama_coder_top_p: float = 0.0
    ollama_critic_top_p: float = 0.0

    # --- Redis ---
    redis_url: str = "redis://redis:6379/0"

    # --- Database ---
    database_url: str = "sqlite:///data/hca.db"

    # --- API Security ---
    hca_api_key: str = ""
    cors_origins: str = "*"

    # --- Web UI ---
    web_host: str = "0.0.0.0"  # noqa: S104 - Docker service must bind externally.
    web_port: int = 8080

    # --- Orchestration Limits ---
    max_iterations_per_task: int = 5
    max_tasks_per_project: int = 50
    task_timeout_minutes: int = 30
    project_timeout_minutes: int = 480
    project_token_budget: int = 500_000
    activity_timeout_minutes: int = 60
    max_parallel_tasks: int = 3

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "json"

    # --- Workspace ---
    workspace_dir: str = ".data/workspaces"
    workspace_retention_days: int = 7  # Clean up workspaces older than 7 days
    workspace_max_count: int = 100  # Keep only the 100 most recent workspaces

    # --- GitHub Integration ---
    github_token: str = ""  # Personal access token for pushing to GitHub repos

    # --- Model Puller ---
    ollama_models_to_pull: str = "qwen3:14b qwen2.5-coder:14b"

    def get_agent_model(self, agent_name: str) -> str:
        """Get the model for a specific agent, falling back to default."""
        overrides = {
            "pm": self.ollama_pm_model,
            "research": self.ollama_research_model,
            "spec": self.ollama_spec_model,
            "coder": self.ollama_coder_model_override or self.ollama_coder_model,
            "critic": self.ollama_critic_model,
        }
        model = overrides.get(agent_name, "")
        return model if model else self.ollama_default_model

    def get_agent_temperature(self, agent_name: str) -> float:
        """Get the temperature for a specific agent, falling back to default."""
        overrides = {
            "pm": self.ollama_pm_temperature,
            "research": self.ollama_research_temperature,
            "spec": self.ollama_spec_temperature,
            "coder": self.ollama_coder_temperature,
            "critic": self.ollama_critic_temperature,
        }
        val = overrides.get(agent_name, 0.0)
        return val if val != 0.0 else self.ollama_default_temperature

    def get_recommended_models(self, tier: str = "medium") -> dict[str, str]:
        """Return recommended models for a hardware tier.

        Tier must be one of: high, medium, low, minimal, tiny.
        Returns a dict with keys: default_model, coder_model, num_ctx.
        """
        info = HARDWARE_TIERS.get(tier, HARDWARE_TIERS["medium"])
        return {
            "default_model": str(info["default_model"]),
            "coder_model": str(info["coder_model"]),
            "quantization": str(info.get("quantization", "")),
            "num_ctx": str(info.get("num_ctx", 8192)),
        }

    def get_agent_top_p(self, agent_name: str) -> float:
        """Get the top_p for a specific agent, falling back to default."""
        overrides = {
            "pm": self.ollama_pm_top_p,
            "research": self.ollama_research_top_p,
            "spec": self.ollama_spec_top_p,
            "coder": self.ollama_coder_top_p,
            "critic": self.ollama_critic_top_p,
        }
        val = overrides.get(agent_name, 0.0)
        return val if val != 0.0 else self.ollama_default_top_p

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
