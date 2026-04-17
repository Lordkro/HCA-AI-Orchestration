"""Configuration management for HCA Orchestration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Ollama ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_default_model: str = "qwen3.5:27b"
    ollama_coder_model: str = "qwen3-coder:30b"  # Default coder model (fallback)
    ollama_timeout: int = 120
    ollama_num_ctx: int = 8192
    ollama_max_retries: int = 3
    ollama_retry_base_delay: float = 2.0

    # Per-agent model overrides (empty string = use default)
    ollama_pm_model: str = ""
    ollama_research_model: str = ""
    ollama_spec_model: str = ""
    ollama_coder_model_override: str = ""  # If set, overrides ollama_coder_model
    ollama_critic_model: str = ""

    # --- Redis ---
    redis_url: str = "redis://redis:6379/0"

    # --- Database ---
    database_url: str = "sqlite:///data/hca.db"

    # --- Web UI ---
    web_host: str = "0.0.0.0"
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
    workspace_dir: str = "/workspace"

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
