# User Guide

## Overview

HCA is an autonomous AI development team. You give it a product idea, and it researches, designs, implements, reviews, and delivers working code — all without manual intervention.

The team consists of five AI agents:
- **Project Manager** — plans and coordinates the work
- **Research Agent** — investigates technologies and approaches
- **Specification Agent** — writes detailed technical specs
- **Coder Agent** — writes the actual code
- **Critic Agent** — reviews everything for quality

---

## Prerequisites

- **Docker** and **Docker Compose** (v2+)
- An AMD GPU with ROCm support (optional — CPU-only works, but is slower)
- Apple Silicon (M-series) with Metal GPU acceleration (optional — CPU-only works)
- At least 20GB free disk space for model downloads and workspaces
- At least 16GB RAM (32GB recommended with GPU)

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd HCA-Orchestration
cp .env.example .env
```

Edit `.env` if you want to change models or other defaults. The defaults work out of the box.

### 2. Pull the LLM models

```bash
docker compose pull ollama redis
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:14b
docker compose exec ollama ollama pull qwen2.5-coder:14b
```

Or use the puller service (one-shot):
```bash
docker compose --profile setup run --rm model-puller
```

**GPU VRAM note**: Each 14B model needs ~10GB VRAM. If you have a 16GB GPU, only one model can be loaded at a time. Ollama swaps models automatically as needed.

### 3. Start the system

```bash
docker compose up
```

The first startup downloads the Docker images. Subsequent starts are fast.

### 4. Open the dashboard

Navigate to [http://localhost:8080](http://localhost:8080). You should see the HCA dashboard.

---

## Your First Project

### Submitting an Idea

1. Open the dashboard at `http://localhost:8080`
2. Type a product idea into the input box, for example:
   > "Build a REST API for a personal finance tracker with user authentication, expense categories, and monthly budget reports"
3. Click **Submit**

The Project Manager agent immediately starts working. You'll see the agent status indicators change as work progresses.

### What Happens Next

The system follows this pipeline automatically:

```
PM breaks down idea → Research investigates → Spec writes architecture
  → Coder implements → Critic reviews → (if rejected) revise → done
```

1. **PM Agent** (seconds): Decomposes the idea into 3–7 tasks
2. **Research Agent** (1–2 min): Investigates technologies and approaches
3. **Spec Agent** (2–3 min): Writes detailed technical specifications
4. **Coder Agent** (3–10 min): Implements the code
5. **Critic Agent** (1–2 min): Reviews the output
6. **If revisions needed**: The Critic sends feedback, the Coder fixes issues, and the Critic re-reviews
7. **Done**: All tasks complete and the project is marked `completed`

Each step may take 1–10 minutes depending on LLM response time and complexity.

### Monitoring Progress

- **Agent status lights**: Green (idle), Yellow (thinking/working), Red (error)
- **Project list**: Shows all projects with their current status
- **Task list**: Shows individual tasks with their state
- **Live log stream**: Real-time agent activity via WebSocket

### Viewing Results

When the project is completed:
- Navigate to the project detail page
- Browse generated **artifacts** (files the Coder agent created)
- Read the **messages** to see the full conversation between agents
- Download the workspace containing all generated files

### Git History

Each project workspace is a git repository. After every coding iteration and revision, changes are automatically committed:
- View the commit log: `cd .data/workspaces/{project_id} && git log --oneline`
- See what changed in the last iteration: `git diff HEAD~1`
- Roll back to a previous state: `git checkout {commit_hash}`

---

## Sandbox Validation

When the Coder agent finishes writing code, the **SandboxExecutor** automatically validates it inside an isolated Docker container:

1. **Syntax check** — all generated files are compiled to check for syntax errors
2. **Import check** — entrypoint modules (main.py, app.py, etc.) are imported
3. **Smoke test** — the entrypoint is run briefly to detect startup errors

The container runs with:
- No network access (`--network none`)
- Read-only filesystem (`--read-only`)
- 60-second timeout

If Docker is unavailable, sandbox validation is skipped and the system continues normally.

---

## Configuration

### Essential Settings

| Variable | Default | When to Change |
|---|---|---|
| `OLLAMA_DEFAULT_MODEL` | `qwen3:14b` | Use a different model for non-coding agents |
| `OLLAMA_CODER_MODEL` | `qwen2.5-coder:14b` | Use a code-specific model for the Coder |
| `TASK_TIMEOUT_MINUTES` | `30` | Increase for complex projects with long LLM responses |
| `MAX_ITERATIONS_PER_TASK` | `5` | Increase if you want more revision cycles |
| `HCA_API_KEY` | (empty) | Set a key to enable authentication |

See the full reference in `.env.example`.

### Per-Agent Tuning

If one agent consistently produces poor results, you can:
- Assign a different model to that agent (`OLLAMA_*_MODEL`)
- Adjust its temperature (`OLLAMA_*_TEMPERATURE`) — lower for more deterministic output, higher for more creative
- Adjust its top_p (`OLLAMA_*_TOP_P`) — lower for more focused sampling

### Resource Settings

For systems with limited resources:
- Set `OLLAMA_MAX_CONCURRENT=1` (default) to avoid overloading Ollama
- If using CPU-only, expect 5–20x slower responses
- Reduce `OLLAMA_NUM_CTX=4096` to use less memory (at the cost of shorter conversation history)

---

## Projects

### Creating a Project

```bash
curl -X POST http://localhost:8080/api/projects/ \
  -H "Content-Type: application/json" \
  -d '{"idea": "Build a CLI tool for managing Docker containers"}'
```

### Project Statuses

| Status | Meaning |
|---|---|
| `active` | Project is running, tasks are being processed |
| `paused` | Processing is suspended (can be resumed) |
| `completed` | All tasks are done |
| `failed` | A task timed out or hit a guardrail limit |

### Pausing and Resuming

You can pause an active project and resume it later. Use the dashboard pause/resume buttons or the API:

```bash
curl -X POST http://localhost:8080/api/projects/{id}/pause
curl -X POST http://localhost:8080/api/projects/{id}/resume
```

### Retrying Failed Tasks

If a task fails (timeout, error), you can retry it:

```bash
curl -X POST http://localhost:8080/api/tasks/detail/{task_id}/retry
```

This resets the task from `FAILED` → `PENDING`, and the PM will reassign it.

---

## Workspaces

Each project has a workspace directory where the Coder agent writes generated files.

You can:
- **Browse workspaces** via the dashboard
- **Clean up old workspaces** manually: `POST /api/projects/workspaces/cleanup`
- **View workspace stats**: `GET /api/projects/workspaces/stats`

Workspaces older than 7 days (configurable via `WORKSPACE_RETENTION_DAYS`) are automatically cleaned up.

---

## API Authentication

If you set `HCA_API_KEY` in `.env`, all API requests except health checks and metrics require authentication:

```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8080/api/projects/
```

---

## Troubleshooting

### "Ollama is unreachable" on startup

Make sure Ollama is running and the model is downloaded:

```bash
docker compose logs ollama
docker compose exec ollama ollama list
```

### Agents are stuck in "thinking" state

Check Ollama logs — the model may be loading or a request may have timed out:

```bash
docker compose logs orchestrator
```

### "Permission denied" errors

The workspace directory may have incorrect permissions. Check that the non-root container user can write to `WORKSPACE_DIR`.

### Slow responses

- CPU-only mode is significantly slower than GPU
- Large context windows increase response time
- Try reducing `OLLAMA_NUM_CTX` or using a smaller model

### Model-specific issues

- The default models are `qwen3:14b` and `qwen2.5-coder:14b`
- If you switch models, ensure they support tool calling (required for PM, Coder, and Critic agents)
- `<think>` blocks are automatically stripped from responses

---

## API Reference

For a complete API reference, visit `/docs` on your running instance (e.g., `http://localhost:8080/docs`).

### Key Endpoints

| Endpoint | Description |
|---|---|
| `POST /api/projects/` | Create a project |
| `GET /api/projects/` | List projects |
| `GET /api/agents/` | Agent status |
| `GET /api/health` | System health |
| `WS /ws` | Real-time events |
