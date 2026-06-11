"""Configuration management for HCA Orchestration."""

import re
from pathlib import Path

import structlog
from pydantic_settings import BaseSettings

logger = structlog.get_logger("hca.config")

# Regex to extract parameter count from model names like "qwen3:8b" or "qwen2.5-coder:7b"
_PARAM_PATTERN = re.compile(r"(\d+\.?\d*)(?:b|B)")

# Estimated VRAM needed per billion parameters at Q4_K_M (GB)
_VRAM_PER_BILLION_PARAMS = 0.6

_KNOWN_CPU_FALLBACK_MODELS: dict[str, float] = {
    "phi-4": 4.0,
    "phi3": 3.8,
    "tinyllama": 1.1,
}

# Paths for AMD VRAM detection
_AMD_VRAM_PATHS = [
    "/sys/class/drm/card0/device/mem_info_vram_total",
    "/sys/class/drm/card1/device/mem_info_vram_total",
]


def _detect_vram_bytes() -> int | None:
    """Detect available GPU VRAM in bytes.

    Tries AMD sysfs paths first, then falls back to rocm-smi.
    Returns None if detection fails.
    """
    # AMD sysfs
    for path_str in _AMD_VRAM_PATHS:
        path = Path(path_str)
        if path.exists():
            try:
                return int(path.read_text().strip())
            except (ValueError, OSError):
                continue

    # rocm-smi fallback
    import subprocess  # noqa: S404
    try:
        result = subprocess.run(  # noqa: S603
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "VRAM Total" in line or "Total Memory" in line:
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            return int(part) * 1024 * 1024
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _detect_system_ram_bytes() -> int | None:
    """Detect total system RAM in bytes."""
    try:
        import os
        # Python 3.11+: os.sysconf can return None
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages and page_size:
            return pages * page_size
        return None
    except (ValueError, OSError, AttributeError):
        return None


def _estimate_model_vram(model_name: str) -> float | None:
    """Estimate VRAM needed (GB) for a given model at Q4_K_M.

    Strips quantization suffix (e.g. ``:Q4_K_M``) and parses the
    parameter count from well-known naming patterns.

    Returns None if the model name is unrecognised.
    """
    name = model_name.rsplit(":", 1)[0] if ":Q" in model_name or ":q" in model_name else model_name
    m = _PARAM_PATTERN.search(name)
    if m:
        try:
            params_b = float(m.group(1))
            return params_b * _VRAM_PER_BILLION_PARAMS
        except ValueError:
            return None

    # Check known CPU-friendly models that lack numeric tags
    for prefix, known_params in _KNOWN_CPU_FALLBACK_MODELS.items():
        if name.lower().startswith(prefix):
            return known_params * _VRAM_PER_BILLION_PARAMS

    return None


def check_hardware_fit(config_model: str, coder_model: str) -> list[str]:
    """Compare configured models against available VRAM / RAM.

    Returns a list of warning messages (empty = all good).
    """
    warnings: list[str] = []

    vram_bytes = _detect_vram_bytes()
    ram_bytes = _detect_system_ram_bytes()

    if vram_bytes is None and ram_bytes is None:
        return []

    for label, model in [("default", config_model), ("coder", coder_model)]:
        est = _estimate_model_vram(model)
        if est is None:
            continue

        fits_vram = vram_bytes and (est * 1024**3) <= vram_bytes
        fits_ram = ram_bytes and (est * 1024**3) <= ram_bytes * 0.5  # don't use >50% of RAM

        if vram_bytes and not fits_vram:
            warnings.append(
                f"Model '{model}' ({label}) needs ~{est:.0f}GB VRAM but only "
                f"{vram_bytes / 1024**3:.0f}GB available — expect slowdown or OOM"
            )
        elif not vram_bytes and ram_bytes and not fits_ram:
            warnings.append(
                f"Model '{model}' ({label}) needs ~{est:.0f}GB (50% of your "
                f"{ram_bytes / 1024**3:.0f}GB RAM) — system may become unresponsive"
            )

    return warnings

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
    "metal": {
        "label": "Apple Metal",
        "image": "ollama/ollama",
        "tag": "latest",
        "compose_profile": "metal",
        "note": "Apple Silicon (M-series); uses Metal GPU via arm64 image.",
    },
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Ollama ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_default_model: str = "qwen3:8b"
    ollama_default_temperature: float = 0.7
    ollama_default_top_p: float = 0.9
    ollama_coder_model: str = "qwen2.5-coder:7b"  # Default coder model (fallback)
    ollama_timeout: int = 600
    ollama_num_ctx: int = 8192
    ollama_max_retries: int = 3
    ollama_retry_base_delay: float = 2.0
    ollama_max_concurrent: int = 1  # Max parallel LLM calls
    ollama_keep_alive: str = "1m"  # How long to keep model in memory after use ("0" = unload immediately)
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
    ollama_models_to_pull: str = "qwen3:8b qwen2.5-coder:7b"

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
