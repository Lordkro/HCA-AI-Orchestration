# HCA Orchestration

**Hybrid Cognitive Architecture — An Autonomous AI Development Team**

[![Version](https://img.shields.io/github/v/tag/Lordkro/HCA-AI-Orchestration?label=version&logo=github)](https://github.com/Lordkro/HCA-AI-Orchestration/releases)
[![CI](https://github.com/Lordkro/HCA-AI-Orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/Lordkro/HCA-AI-Orchestration/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue?logo=docker)](https://github.com/Lordkro/HCA-AI-Orchestration/pkgs/container/hca-orchestration)
[![License](https://img.shields.io/github/license/Lordkro/HCA-AI-Orchestration)](LICENSE)

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
- **No GPU required** — CPU mode works on any system; NVIDIA, AMD, Vulkan (any GPU), and Apple Silicon GPU profiles available

### One-command setup (recommended)

```bash
git clone <repo-url> && cd HCA-Orchestration
bash setup.sh
```

This auto-detects your GPU, copies `.env`, starts services, and pulls the recommended models.
For NVIDIA GPUs, install the [container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first.

**Options:**

| Flag | Purpose |
|------|---------|
| `--profile nvidia` | Force NVIDIA profile (skip auto-detect) |
| `--profile rocm` | Force AMD ROCm profile |
| `--profile vulkan` | Force Vulkan GPU profile (AMD/Intel/any) |
| `--profile metal` | Force Apple Silicon (Metal) profile |
| `--models "llama3.2:3b qwen2.5-coder:3b"` | Pull different models |
| `--skip-pull` | Skip model downloading |

### Manual setup

<details>
<summary>Click to expand manual steps</summary>

#### 1. Choose your hardware profile

| Profile | Command | Requirements |
|---------|---------|--------------|
| **CPU** (default) | `docker compose up` | No GPU needed |
| **NVIDIA GPU** | `docker compose --profile nvidia up` | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| **AMD ROCm** | `docker compose --profile rocm up` | AMD GPU with ROCm driver |
| **Vulkan (any GPU)** | `docker compose --profile vulkan up` | Any GPU with Vulkan driver (recommended fallback) |
| **Apple Metal** | `docker compose --profile metal up` | Apple Silicon (M-series) Mac |

#### 2. Pick the right model for your VRAM

| VRAM | Recommended default | Recommended coder |
|------|-------------------|-------------------|
| ≥24 GB | `qwen3:14b` | `qwen2.5-coder:14b` |
| 12-24 GB | `qwen3:8b` | `qwen2.5-coder:7b` |
| 8-12 GB | `llama3.2:3b` | `qwen2.5-coder:3b` |
| 6-8 GB | `phi-4:latest` | `phi-4:latest` |
| <6 GB | `llama3.2:1b` | `qwen2.5-coder:1.5b` |

#### 3. Configure

```bash
cp .env.example .env
# Edit OLLAMA_DEFAULT_MODEL and OLLAMA_CODER_MODEL for your VRAM
```

#### 4. Start and pull models

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:14b
docker compose exec ollama ollama pull qwen2.5-coder:14b
docker compose up -d
```

#### 5. Open dashboard

Navigate to [http://localhost:8080](http://localhost:8080).
</details>

## 📚 Documentation

| Guide | Description |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | System architecture, data flow, state machine, schema, security |
| [User Guide](docs/USER_GUIDE.md) | Step-by-step setup and usage guide |
| [Contributing](CONTRIBUTING.md) | Development setup, tests, code style, adding agents |
| [Prompt Cookbook](docs/PROMPT_COOKBOOK.md) | Prompt engineering reference and best practices |
| [Workspace Management](WORKSPACE_MANAGEMENT.md) | Workspace lifecycle, retention, git integration |

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
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # Full system architecture
│   ├── USER_GUIDE.md       # End-user guide
│   └── PROMPT_COOKBOOK.md  # Prompt engineering reference
├── src/
│   └── hca/                # Main package (hca namespace)
│       ├── main.py         # Application entrypoint
│       ├── core/           # Shared infrastructure
│       │   ├── config.py       # Settings from env vars
│       │   ├── database.py     # SQLite persistence
│       │   ├── logger.py       # Structured logging
│       │   ├── message_bus.py  # Redis Streams
│       │   ├── metrics.py      # Prometheus metrics
│       │   ├── models.py       # Pydantic data models
│       │   ├── ollama_client.py# Ollama API wrapper
│       │   └── tools.py        # Tool definitions + validation
│       ├── agents/         # Agent implementations
│       │   ├── base_agent.py   # Abstract base class
│       │   ├── pm_agent.py     # Project Manager
│       │   ├── research_agent.py
│       │   ├── spec_agent.py
│       │   ├── coder_agent.py
│       │   └── critic_agent.py
│       ├── orchestrator/   # Workflow engine
│       │   ├── guardrails.py
│       │   ├── pipeline.py
│       │   ├── sandbox.py
│       │   ├── task_manager.py
│       │   └── workspace_manager.py
│       ├── api/            # Web API + UI
│       │   ├── app.py
│       │   ├── routes/
│       │   │   ├── agents.py
│       │   │   ├── dead_letter.py
│       │   │   ├── hitl.py
│       │   │   ├── projects.py
│       │   │   ├── tasks.py
│       │   │   └── websocket.py
│       │   └── static/
│       └── prompts/        # System prompts per agent
├── tests/                  # Test suite
│   ├── conftest.py         # Shared mock fixtures
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── scripts/                # Utility scripts
├── .data/                  # Runtime data (git-ignored)
│   ├── workspaces/         # Generated project files
│   ├── logs/               # Application logs
│   └── cache/              # Runtime cache
├── CONTRIBUTING.md
├── WORKSPACE_MANAGEMENT.md
├── LICENSE                 # MIT License
├── docker-compose.yml      # All services
├── Dockerfile              # Python app image
├── pyproject.toml          # Dependencies & build config
├── setup.sh                # One-command setup script
├── .dockerignore
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
