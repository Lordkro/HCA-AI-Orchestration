# HCA Orchestration

**Hybrid Cognitive Architecture — An Autonomous AI Development Team**

An AI agent team that takes your product ideas and builds them into working applications, powered by local LLMs via Ollama.

## 🧠 The Team

| Agent | Role | What It Does |
|-------|------|-------------|
| 📋 **Project Manager** | Orchestrator | Breaks down ideas into tasks, assigns work, tracks progress |
| 🔍 **Research Agent** | Analyst | Investigates technologies, patterns, and feasibility |
| 📐 **Specification Agent** | Architect | Writes detailed technical specs, API contracts, data models |
| 💻 **Coder Agent** | Engineer | Implements code based on specifications |
| 🔎 **Critic Agent** | Reviewer | Reviews all outputs for quality and correctness |

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2+
- At least 8 GB system RAM (16 GB+ recommended)
- **No GPU required** — CPU mode works on any system; NVIDIA and AMD GPU profiles available

### 1. Choose your hardware profile

Pick the profile that matches your setup:

| Profile | Command | Requirements |
|---------|---------|--------------|
| **CPU** (default) | `docker compose up` | No GPU needed — slowest but most compatible |
| **NVIDIA GPU** | `docker compose --profile nvidia up` | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| **AMD ROCm** | `docker compose --profile rocm up` | AMD GPU with ROCm driver, `/dev/kfd` + `/dev/dri` |

For NVIDIA, first install the container toolkit:
```bash
# Ubuntu / Debian
sudo apt install nvidia-container-toolkit && sudo systemctl restart docker
# Validate
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 2. Pick the right model for your VRAM

| VRAM | Recommended default | Recommended coder | Models |
|------|-------------------|-------------------|--------|
| ≥24 GB | `qwen3:14b` | `qwen2.5-coder:14b` | ~9 GB each |
| 12-24 GB | `qwen3:8b` | `qwen2.5-coder:7b` | ~5-6 GB each |
| 8-12 GB | `llama3.2:3b` | `qwen2.5-coder:3b` | ~2-3 GB each |
| 6-8 GB | `phi-4:latest` | `phi-4:latest` | ~2.5 GB |
| <6 GB | `llama3.2:1b` | `qwen2.5-coder:1.5b` | <1 GB each |

Edit `.env` to set `OLLAMA_DEFAULT_MODEL` and `OLLAMA_CODER_MODEL` for your VRAM tier.

### 3. Clone and configure

```bash
git clone <repo-url> && cd HCA-Orchestration
cp .env.example .env
# Edit .env for your hardware profile (models, GPU)
```

### 4. Pull LLM models (first time only)

```bash
docker compose --profile setup run --rm model-puller
```

The puller fetches the models listed in `OLLAMA_MODELS_TO_PULL` (default: `qwen3:14b qwen2.5-coder:14b`).
Each model is 1-9 GB depending on size.

To pull only specific models instead:
```bash
OLLAMA_MODELS_TO_PULL="llama3.2:3b qwen2.5-coder:3b" docker compose --profile setup run --rm model-puller
```

### 5. Start the system

```bash
# CPU
docker compose up

# NVIDIA
docker compose --profile nvidia up

# AMD ROCm
docker compose --profile rocm up
```

### 6. Open the dashboard

Navigate to [http://localhost:8080](http://localhost:8080) and submit your first product idea!

## 📚 Documentation

| Guide | Description |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | System architecture, data flow, state machine, schema, security |
| [User Guide](docs/USER_GUIDE.md) | Step-by-step setup and usage guide |
| [Contributing](CONTRIBUTING.md) | Development setup, tests, code style, adding agents |
| [Prompt Cookbook](docs/PROMPT_COOKBOOK.md) | Prompt engineering reference and best practices |

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Web Dashboard (UI)            │
│        FastAPI + WebSocket + HTML       │
└──────────────────┬──────────────────────┘
                   │ REST / WebSocket
┌──────────────────▼──────────────────────┐
│          Orchestrator Service           │
│     (Agents + Pipeline + Task Mgmt)     │
└──────────────────┬──────────────────────┘
                   │ Redis Streams
          ┌────────┼────────┐
          ▼        ▼        ▼
     ┌────────┐┌────────┐┌────────┐
     │Agent 1 ││Agent 2 ││Agent N │
     └───┬────┘└───┬────┘└───┬────┘
         └─────────┼─────────┘
                   ▼
            ┌────────────┐
            │   Ollama   │
            │ (LLM API)  │
            └────────────┘
```

## 📁 Project Structure

```
HCA-Orchestration/
├── config/                 # Configuration files (YAML/JSON)
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # Full system architecture
│   ├── USER_GUIDE.md       # End-user guide
│   └── PROMPT_COOKBOOK.md  # Prompt engineering reference
├── src/
│   └── hca/                # Main package (hca namespace)
│       ├── main.py         # Application entrypoint
│       ├── core/           # Shared infrastructure
│       │   ├── config.py       # Settings from env vars
│       │   ├── ollama_client.py # Ollama API wrapper
│       │   ├── message_bus.py   # Redis Streams
│       │   ├── database.py      # SQLite persistence
│       │   ├── models.py        # Pydantic data models
│       │   └── logger.py        # Structured logging
│       ├── agents/         # Agent implementations
│       │   ├── base_agent.py   # Abstract base class
│       │   ├── pm_agent.py     # Project Manager
│       │   ├── research_agent.py
│       │   ├── spec_agent.py
│       │   ├── coder_agent.py
│       │   └── critic_agent.py
│       ├── orchestrator/   # Workflow engine
│       │   ├── pipeline.py
│       │   ├── task_manager.py
│       │   └── guardrails.py
│       ├── api/            # Web API + UI
│       │   ├── app.py
│       │   ├── routes/
│       │   └── static/
│       └── prompts/        # System prompts per agent
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Shared test fixtures
├── scripts/                # Utility scripts
├── .data/                  # Runtime data (git-ignored)
│   ├── workspaces/         # Generated project files
│   ├── logs/               # Application logs
│   └── cache/              # Runtime cache
├── docker-compose.yml      # All services
├── Dockerfile              # Python app image
├── pyproject.toml          # Dependencies & build config
└── .env.example            # Configuration template
```

## ⚙️ Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_DEFAULT_MODEL` | `qwen3:14b` | Model for PM, Research, Spec, Critic agents |
| `OLLAMA_CODER_MODEL` | `qwen2.5-coder:14b` | Model for the Coder agent |
| `MAX_ITERATIONS_PER_TASK` | `5` | Max revision cycles |
| `TASK_TIMEOUT_MINUTES` | `30` | Timeout per task |
| `WEB_PORT` | `8080` | Dashboard port |

## 📄 License

MIT
