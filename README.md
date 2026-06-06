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

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- An AMD GPU with ROCm support (or CPU-only mode)

### 1. Clone and configure

```bash
git clone <repo-url> && cd HCA-Orchestration
cp .env.example .env
# Edit .env if you want to change models or settings
```

### 2. Pull LLM models (first time only)

```bash
# Recommended: pull directly inside the Ollama container
docker compose up ollama -d
docker compose exec ollama ollama pull qwen3:14b
docker compose exec ollama ollama pull qwen2.5-coder:14b
```

Each model is ~9GB. Alternatively, use the model-puller service:
```bash
docker compose --profile setup run --rm model-puller
```

> **GPU VRAM note:** The default 14B models each need ~10GB VRAM. With 16GB GPUs,
> only one model can be loaded at a time (`OLLAMA_MAX_LOADED_MODELS=1`).
> Ollama will automatically swap models as needed.

### 3. Start the system

```bash
docker compose up
```

### 4. Open the dashboard

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
