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
docker compose --profile setup run --rm model-puller
```

This pulls `qwen3.5:27b` and `qwen3-coder:30b` (~17-19GB each). Be patient!

### 3. Start the system

```bash
docker compose up
```

### 4. Open the dashboard

Navigate to [http://localhost:8080](http://localhost:8080) and submit your first product idea!

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
├── docker-compose.yml      # All services
├── Dockerfile              # Python app image
├── pyproject.toml          # Dependencies
├── .env.example            # Configuration template
├── src/
│   ├── main.py             # Application entrypoint
│   ├── core/               # Shared infrastructure
│   │   ├── config.py       # Settings from env vars
│   │   ├── ollama_client.py # Ollama API wrapper
│   │   ├── message_bus.py  # Redis Streams
│   │   ├── database.py     # SQLite persistence
│   │   ├── models.py       # Pydantic data models
│   │   └── logger.py       # Structured logging
│   ├── agents/             # Agent implementations
│   │   ├── base_agent.py   # Abstract base class
│   │   ├── pm_agent.py     # Project Manager
│   │   ├── research_agent.py
│   │   ├── spec_agent.py
│   │   ├── coder_agent.py
│   │   └── critic_agent.py
│   ├── orchestrator/       # Workflow engine
│   │   ├── pipeline.py
│   │   ├── task_manager.py
│   │   └── guardrails.py
│   ├── api/                # Web API + UI
│   │   ├── app.py
│   │   ├── routes/
│   │   └── static/
│   └── prompts/            # System prompts per agent
├── workspace/              # Generated project files
├── tests/                  # Test suite
└── scripts/                # Utility scripts
```

## ⚙️ Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_DEFAULT_MODEL` | `qwen3.5:27b` | Model for all agents |
| `OLLAMA_CODER_MODEL` | `qwen3-coder:30b` | Model for the Coder agent |
| `MAX_ITERATIONS_PER_TASK` | `5` | Max revision cycles |
| `TASK_TIMEOUT_MINUTES` | `30` | Timeout per task |
| `WEB_PORT` | `8080` | Dashboard port |

## 📄 License

MIT
