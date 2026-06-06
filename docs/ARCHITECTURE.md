# Architecture

## Overview

HCA (Hybrid Cognitive Architecture) is an autonomous AI development team. It takes product ideas and drives them through a structured pipeline — research, specification, coding, and review — with zero human intervention. Every agent is powered by a local LLM served via Ollama.

The system is built on six core abstractions:

- **Agents** — LLM-driven workers, each with a distinct role
- **Message Bus** — Redis Streams for reliable inter-agent communication
- **Database** — SQLite for persistent project, task, and artifact storage
- **Orchestrator** — Pipeline loop with guardrails, timeouts, and health checks
- **Sandbox** — Isolated Docker containers for validating generated code
- **API Layer** — FastAPI for REST endpoints, WebSocket for real-time UI updates

---

## System Diagram

```
┌─────────────────────────────────────────────────────┐
│                    User / Dashboard                 │
│              REST + WebSocket (FastAPI)              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                   Orchestrator                       │
│  ┌──────────────────────────────────────────────┐   │
│  │          Pipeline (health loop)              │   │
│  │  Token budget · Timeouts · Deadlock detection│   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────┐ ┌──────────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │  PM  │ │ Research │ │ Spec │ │Coder │ │Critic│ │
│  │Agent │ │  Agent   │ │Agent │ │Agent │ │Agent │ │
│  └──┬───┘ └────┬─────┘ └──┬───┘ └──┬───┘ └──┬───┘ │
│     │          │          │        │        │      │
└─────┼──────────┼──────────┼────────┼────────┼──────┘
      │          │          │        │        │
      └──────────┴──────────┴────────┴────────┘
                    │
         ┌──────────▼──────────┐
         │   Redis Streams     │
         │  (message bus)      │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │      Ollama         │
         │  (local LLM API)    │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   SQLite (persist)  │
         └─────────────────────┘
```

---

## Agents

### Project Manager (PM) Agent

**Role**: Orchestrator and decision-maker. The PM receives product ideas, decomposes them into tasks, assigns work to the appropriate agents, tracks progress, and routes feedback.

**Message types handled**: `SYSTEM` (new project), `DELIVERABLE`, `FEEDBACK`, `STATUS_UPDATE`, `QUESTION`

**Key behaviors**:
- Breaks down project ideas using `think_with_tools` and the `create_task` tool
- Maintains dependency ordering between tasks
- Routes Critic-approved deliverables through `APPROVED → DONE` and dispatches the next task
- Routes Critic-rejected deliverables back to the originating agent for revision
- Marks projects `completed` when all tasks reach `DONE`

### Research Agent

**Role**: Technology investigator. Analyzes project requirements, recommends technologies, architecture patterns, and data models. Produces actionable research reports.

**Message types handled**: `TASK_ASSIGNMENT`, `QUESTION`, `FEEDBACK`

**Key behaviors**:
- Produces structured reports covering technology analysis, architecture recommendations, data model considerations, potential challenges, and best practices
- Revises reports based on Critic feedback

### Specification Agent

**Role**: Technical architect. Translates research and requirements into comprehensive technical specifications that the Coder agent can implement directly.

**Message types handled**: `TASK_ASSIGNMENT`, `FEEDBACK`, `QUESTION`

**Key behaviors**:
- Writes specifications covering overview, architecture, data models, API contracts, file structure, implementation notes, and testing strategy
- Uses code blocks for schemas and definitions
- Revises specs based on Critic feedback

### Coder Agent

**Role**: Software engineer. Implements code from specifications. Creates files in project workspaces using the `write_file` tool.

**Message types handled**: `TASK_ASSIGNMENT`, `FEEDBACK`, `QUESTION`

**Key behaviors**:
- Generates complete, production-ready code (no placeholders or TODOs)
- Creates tests alongside implementation
- Validates file paths against traversal attacks
- Falls back to regex-based file extraction (`=== FILE: path ===`) if tool calling fails
- Revises code based on Critic feedback

### Critic Agent

**Role**: Quality assurance. Reviews all deliverables (research, specs, code) for correctness, security, completeness, and adherence to best practices.

**Message types handled**: `TASK_ASSIGNMENT`, `QUESTION`

**Key behaviors**:
- Uses `think_with_tools` with the `submit_review` tool for structured verdicts
- Returns `approved` or `needs_revision` with severity-graded issues (critical/major/minor)
- Falls back to string matching (`**APPROVED**` in first lines) if tool calling is unavailable
- Routes approved work back to the PM, rejected work as `FEEDBACK`

---

## Communication — Redis Streams

The message bus uses **Redis Streams** with consumer groups for reliable point-to-point and broadcast messaging.

### Stream Architecture

| Stream Pattern | Purpose |
|---|---|
| `hca:agents:{agent_role}` | Per-agent inbox (one stream per agent) |
| `hca:broadcast` | Messages sent to all agents (`recipient: "*"`) |
| `hca:events` | UI event history (capped at 2000 entries) |
| `hca:deadletter` | Failed messages that exhausted retries (capped at 1000 entries) |

### Delivery Model

- **Consumer group**: `hca-workers` — all agents belong to the same group
- **Consumption**: Agents call `XREADGROUP` with `last_id=">"` (only new messages)
- **Acknowledgement**: After successful processing, agents `XACK` the message
- **Stale claim**: `XAUTOCLAIM` reclaims orphaned messages after 120s of idle time
- **Dead letter**: After exhausting retries (default 2 attempts), messages are moved to `hca:deadletter` with a reason string
- **Fan-out**: `recipient: "*"` publishes to every agent's inbox individually
- **Notifications**: All messages are also published to a Redis pub/sub channel (`hca:notifications`) for real-time WebSocket forwarding
- **Retry backoff**: Exponential: `base_delay × 2^attempt` (2s → 4s)

### Pub/Sub for Real-Time UI

Every message published to a stream is also published to the `hca:notifications` pub/sub channel. The WebSocket endpoint subscribes to this channel and forwards messages to connected dashboard clients.

---

## State Machine

Tasks progress through a well-defined state machine. The PM agent drives transitions, and the TaskManager enforces validity.

```
                ┌──────────┐
                │ PENDING  │ ◄──── retry (FAILED → PENDING)
                └────┬─────┘
                     │ assign
                ┌────▼─────┐
                │ ASSIGNED │
                └────┬─────┘
                     │ agent starts work
                ┌────▼────────┐
                │ IN_PROGRESS │ ◄──── revision loop
                └────┬────────┘
                     │ deliverable submitted
                ┌────▼────┐
                │ REVIEW  │
                └────┬────┘
              ┌──────┴──────┐
              │             │
     ┌────────▼──────┐  ┌──▼──────────┐
     │   APPROVED    │  │  REVISION   │
     └────────┬──────┘  └──────┬───────┘
              │                │
     ┌────────▼──────┐        │ (back to IN_PROGRESS)
     │     DONE      │        │
     └───────────────┘        │
                              │
     ┌───────────────┐        │
     │    FAILED     │ ◄──────┘
     └───────────────┘
```

### Valid Transitions

| From | To |
|---|---|
| `PENDING` | `ASSIGNED` |
| `ASSIGNED` | `IN_PROGRESS`, `FAILED` |
| `IN_PROGRESS` | `REVIEW`, `FAILED` |
| `REVIEW` | `APPROVED`, `REVISION` |
| `REVISION` | `IN_PROGRESS`, `FAILED` |
| `APPROVED` | `DONE` |
| `FAILED` | `PENDING` |

### Revision Cycle

When a task enters `REVISION`, its iteration counter is incremented. If the counter exceeds `max_iterations` (default 5) or the task has timed out, the guardrail denies the revision and transitions directly to `FAILED` with an escalation message to the PM.

---

## Database Schema

SQLite via `aiosqlite` with WAL mode, foreign keys, and SQLITE_BUSY retry with exponential backoff (0.1s base, 5 attempts).

### Tables

#### `projects`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `name` | TEXT | Project name |
| `description` | TEXT | Optional description |
| `created_at` | TEXT (ISO datetime) | |
| `updated_at` | TEXT (ISO datetime) | |
| `status` | TEXT | `active`, `paused`, `completed`, `failed` |
| `idea` | TEXT | Original user idea |
| `tokens_used` | INTEGER | Added in migration v3 |

#### `tasks`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `project_id` | TEXT | FK → projects(id) |
| `title` | TEXT | |
| `description` | TEXT | |
| `state` | TEXT | TaskState enum value |
| `assigned_to` | TEXT | AgentRole or NULL |
| `created_at` | TEXT | |
| `updated_at` | TEXT | |
| `iteration` | INTEGER | Revision counter |
| `max_iterations` | INTEGER | Default 5 |
| `parent_task_id` | TEXT | Optional parent task |
| `deliverable` | TEXT | Latest deliverable content |
| `feedback` | TEXT | Latest feedback |
| `priority` | TEXT | `low`, `normal`, `high`, `critical` |
| `depends_on` | TEXT | JSON array of task IDs (added v3) |
| `tokens_used` | INTEGER | Token consumption (added v3) |

#### `artifacts`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `project_id` | TEXT | FK → projects(id) |
| `task_id` | TEXT | FK → tasks(id) |
| `agent` | TEXT | AgentRole |
| `filename` | TEXT | Relative path |
| `content` | TEXT | File contents |
| `artifact_type` | TEXT | `code`, `spec`, `research`, `doc`, `test`, `config` |
| `created_at` | TEXT | |
| `version` | INTEGER | Auto-incrementing |

#### `messages`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `timestamp` | TEXT | |
| `sender` | TEXT | AgentRole |
| `recipient` | TEXT | AgentRole or `*` |
| `type` | TEXT | MessageType |
| `project_id` | TEXT | |
| `task_id` | TEXT | |
| `payload` | TEXT | JSON (content, artifacts, metadata) |
| `priority` | TEXT | |

#### `project_events`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Auto-increment PK |
| `project_id` | TEXT | FK → projects(id) |
| `event_type` | TEXT | e.g. `created`, `status_changed`, `task_transitioned` |
| `agent` | TEXT | AgentRole |
| `description` | TEXT | |
| `metadata` | TEXT | JSON |
| `created_at` | TEXT | |

#### `schema_version`

| Column | Type | Notes |
|---|---|---|
| `version` | INTEGER | Current schema version |
| `applied_at` | TEXT | Timestamp |

### Migration History

| Version | Changes |
|---|---|
| 1 | Initial schema: projects, tasks, artifacts, messages |
| 2 | project_events table for timeline tracking |
| 3 | `depends_on` and `tokens_used` columns on tasks; `tokens_used` on projects |

### Indexes

Tasks: `(project_id)`, `(state)`, `(assigned_to)`
Artifacts: `(project_id)`, `(task_id)`, `(artifact_type)`
Messages: `(project_id)`, `(sender)`, `(timestamp)`
Events: `(project_id)`, `(event_type)`

---

## Security Model

### Authentication

Bearer-token authentication via ASGI middleware, configured by the `HCA_API_KEY` environment variable:

- If `HCA_API_KEY` is **empty** (default): auth is disabled — all requests pass through
- If `HCA_API_KEY` is **set**: every request must include `Authorization: Bearer <token>`

**Exempt paths** (no auth required):
- `/metrics` — Prometheus scrape endpoint
- `/api/health/live` — Liveness probe
- `/api/health/ready` — Readiness probe
- `/docs` — OpenAPI/Swagger UI
- `/openapi.json` — OpenAPI schema

Token comparison uses `secrets.compare_digest()` for constant-time comparison against timing attacks.

### CORS

Configured via `CORS_ORIGINS` env var (comma-separated list). Default: `*` (all origins). The middleware supports credentials, all methods, and all headers.

---

## API Endpoints

### REST API

#### Projects

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/projects/` | Create project from an idea |
| `GET` | `/api/projects/` | List all projects |
| `GET` | `/api/projects/{id}` | Get project with progress stats |
| `GET` | `/api/projects/{id}/messages` | Project message history |
| `GET` | `/api/projects/{id}/artifacts` | Generated artifacts |
| `POST` | `/api/projects/{id}/pause` | Pause active project |
| `POST` | `/api/projects/{id}/resume` | Resume paused project |
| `GET` | `/api/projects/workspaces/stats` | Workspace usage statistics |
| `POST` | `/api/projects/workspaces/cleanup` | Trigger workspace cleanup |

#### Tasks

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/tasks/{project_id}` | List tasks (optional `?state=` filter) |
| `GET` | `/api/tasks/detail/{task_id}` | Get single task |
| `POST` | `/api/tasks/detail/{task_id}/retry` | Retry a FAILED task |

#### Agents

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/agents/` | Status of all agents |
| `GET` | `/api/agents/stats` | Ollama client stats |
| `GET` | `/api/agents/{role}` | Status of a specific agent |

#### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Full health with DB, Redis, Ollama, agent stats |
| `GET` | `/api/health/live` | Liveness probe (always 200) |
| `GET` | `/api/health/ready` | Readiness probe (200 if Ollama reachable, else 503) |
| `GET` | `/metrics` | Prometheus metrics |

#### Dead Letter (admin)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/admin/dead-letter` | List dead-letter messages |
| `POST` | `/api/admin/dead-letter` | Replay a dead-letter message |

### WebSocket

| Type | Path | Description |
|---|---|---|
| `WebSocket` | `/ws` | Real-time agent activity stream via Redis pub/sub |

### Static Frontend

The dashboard UI is served from `src/hca/api/static/` at the root path `/` (if the directory exists).

---

## Configuration Reference

All settings are loaded from environment variables (via `.env` file). See `.env.example` for defaults.

### Ollama

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL |
| `OLLAMA_DEFAULT_MODEL` | `qwen3:14b` | Default LLM model |
| `OLLAMA_DEFAULT_TEMPERATURE` | `0.7` | Default sampling temperature |
| `OLLAMA_DEFAULT_TOP_P` | `0.9` | Default nucleus sampling |
| `OLLAMA_CODER_MODEL` | `qwen2.5-coder:14b` | Model for Coder agent |
| `OLLAMA_TIMEOUT` | `120` | Request timeout (seconds) |
| `OLLAMA_NUM_CTX` | `8192` | Context window size |
| `OLLAMA_MAX_RETRIES` | `3` | HTTP retry count |
| `OLLAMA_RETRY_BASE_DELAY` | `2.0` | Retry backoff base (seconds) |
| `OLLAMA_MAX_CONCURRENT` | `1` | Max parallel LLM calls |
| `OLLAMA_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Failures before circuit opens |
| `OLLAMA_CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | `60` | Recovery wait (seconds) |

### Per-Agent Tuning

Each agent can override the default model, temperature, and top_p:

| Variable | Agent |
|---|---|
| `OLLAMA_PM_MODEL` | PM |
| `OLLAMA_RESEARCH_MODEL` | Research |
| `OLLAMA_SPEC_MODEL` | Spec |
| `OLLAMA_CODER_MODEL_OVERRIDE` | Coder (overrides `OLLAMA_CODER_MODEL`) |
| `OLLAMA_CRITIC_MODEL` | Critic |
| `OLLAMA_*_TEMPERATURE` | Per-agent temperature (0 = use default) |
| `OLLAMA_*_TOP_P` | Per-agent top_p (0 = use default) |

### Limits

| Variable | Default | Description |
|---|---|---|
| `MAX_ITERATIONS_PER_TASK` | `5` | Max revision cycles per task |
| `MAX_TASKS_PER_PROJECT` | `50` | Max tasks per project |
| `TASK_TIMEOUT_MINUTES` | `30` | Per-task timeout |
| `PROJECT_TIMEOUT_MINUTES` | `480` | Project-level timeout |
| `PROJECT_TOKEN_BUDGET` | `500000` | Token limit per project |
| `ACTIVITY_TIMEOUT_MINUTES` | `60` | Inactivity timeout |
| `MAX_PARALLEL_TASKS` | `3` | Concurrent task limit |

### Infrastructure

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |
| `DATABASE_URL` | `sqlite:///data/hca.db` | SQLite path |
| `HCA_API_KEY` | (empty) | Bearer auth token (empty = disabled) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `WEB_HOST` | `0.0.0.0` | API bind address |
| `WEB_PORT` | `8080` | API port |
| `WORKSPACE_DIR` | `.data/workspaces` | Project file storage |
| `WORKSPACE_RETENTION_DAYS` | `7` | Workspace cleanup age |
| `WORKSPACE_MAX_COUNT` | `100` | Max retained workspaces |

---

## Agent Lifecycle

### Startup

1. Database initializes and runs pending migrations
2. Redis message bus connects and creates consumer groups for all 5 agents
3. OllamaClient performs a health check (aborts if unreachable)
4. Pipeline loop starts (health checks every 30s)
5. Each agent starts its own `asyncio.Task` running `agent.start()`
6. FastAPI/uvicorn starts serving HTTP and WebSocket connections

### Message Processing (per agent)

1. Emit heartbeat (every 30s)
2. Claim stale messages from crashed consumers (idle > 120s)
3. Consume new messages from inbox (block up to 2000ms)
4. For each message:
   - Validate project is not paused
   - Save message to database
   - Call agent-specific `process_message()`
   - Publish and save response
   - Acknowledge message from stream
5. On failure: retry once (exponential backoff 2s → 4s), then dead-letter
6. If `OllamaCircuitBreakerOpenError`: leave message unacked for stale-claim recovery

### Shutdown

1. Signal `SIGINT` or `SIGTERM`
2. Stop accepting new connections
3. Stop each agent (drain up to 60s for in-flight messages)
4. Stop the pipeline loop
5. Close Ollama HTTP client
6. Disconnect Redis
7. Close database

---

## Pipeline & Guardrails

The `Pipeline` class runs a background loop performing periodic maintenance on active projects.

### Health Check Loop (every 30s)

For each active project:
1. **Token budget check** — if `project_tokens >= budget`, mark project `failed`
2. **Per-task timeout** — tasks in non-terminal states older than `task_timeout_minutes` → `FAILED`
3. **Activity timeout** — if no task has been updated in `activity_timeout_minutes`, mark project `failed`
4. **Deadlock detection** — if all non-DONE tasks are FAILED or dependency-blocked, emit deadlock event

### Stream Maintenance (every 5 min)

Trim all Redis streams to their max lengths (5000 for agent streams, 2000 for events, 1000 for dead-letter).

### Workspace Cleanup (every 60 min)

Remove workspaces older than `workspace_retention_days` or beyond the `workspace_max_count` most recent.

---

## Tool Call Validation

Every tool call from an agent is validated against the JSON schema in its tool definition before execution. This ensures malformed or incomplete calls are caught early and corrected.

### Validation Flow

1. **Schema check** — `validate_tool_call(tool_call, tool_def)` in `src/hca/core/tools.py` checks:
   - Required fields are present and non-empty
   - Field types match (string, array, object)
   - Enum values are valid
   - Array item types are correct
2. **Retry on failure** — If validation fails, the agent retries the LLM with a formatted error message listing exactly which fields need fixing
3. **Fallback** — If the LLM still produces invalid calls after retry, the agent falls back to regex parsing (PM/Coder) or string matching (Critic)

### Supported Tools

| Tool | Used By | Purpose |
|---|---|---|
| `create_task` | PM | Decompose project ideas into tasks |
| `write_file` | Coder | Write generated files to workspace |
| `submit_review` | Critic | Submit structured review verdicts |

---

## Git Integration

Each project workspace is a self-contained git repository, initialized automatically when the Coder agent first writes files.

### Behavior

- **First write** — `WorkspaceManager.init_project_repo()` runs `git init` with local user config
- **After each coding iteration** — `WorkspaceManager.commit_workspace()` stages all changes and commits with a descriptive message
- **Revision context** — Before a feedback revision, the Coder includes `git diff` output in the LLM prompt so the model can see exactly what changed since the last iteration
- **Idempotent** — Safe to call multiple times; skips if `.git` already exists or no changes to commit

### Benefits

- Full revision history per project
- Rollback capability
- Diffs provide precise context for revision prompts

---

## Sandboxed Code Execution

Generated code is validated inside isolated Docker containers. The `SandboxExecutor` in `src/hca/orchestrator/sandbox.py` runs syntax, import, and smoke tests without network access.

### Validation Steps

1. **Language detection** — Scans the workspace for `.py`, `.js`, or `.ts` files to determine the project language
2. **Syntax check** — Runs `python -m py_compile` on all Python files
3. **Import check** — Attempts to import entrypoint modules (`main.py`, `app.py`, etc.)
4. **Smoke test** — Runs the entrypoint briefly (`timeout 5`) to check for startup errors

### Container Isolation

- **Image**: `python:3.11-slim` (extensible to node etc.)
- **Network**: `--network none` — no network access
- **Filesystem**: `--read-only` — read-only mount
- **Timeout**: 60 seconds max per validation
- **Cleanup**: `--rm` — container removed automatically

### Graceful Degradation

If Docker is unavailable on the host, all sandbox checks return a pass with `error: "docker_unavailable"`. The system continues to function normally without sandbox validation.

---

## Metrics & Monitoring

Prometheus metrics (all prefixed with `hca_`) are exposed at `/metrics`.

| Category | Key Metrics |
|---|---|
| LLM | Request count/duration, tokens (prompt/completion), concurrent requests, circuit breaker state |
| Message Bus | Published/consumed/dead-lettered counts, errors, reconnections |
| Database | Query/error counts, size, project/task counts by status |
| Agents | Messages received/sent/failed, LLM calls/errors/duration, status gauge, uptime |
| API | Request count/duration by method+path+status, in-flight requests |

---

## Project Structure

```
HCA-Orchestration/
├── src/hca/
│   ├── main.py                 # Application entrypoint & bootstrap
│   ├── agents/
│   │   ├── base_agent.py       # Abstract base class (common lifecycle)
│   │   ├── pm_agent.py         # Project Manager
│   │   ├── research_agent.py   # Research
│   │   ├── spec_agent.py       # Specification
│   │   ├── coder_agent.py      # Code generation
│   │   └── critic_agent.py     # Quality review
│   ├── api/
│   │   ├── app.py              # FastAPI app factory, middleware
│   │   ├── routes/
│   │   │   ├── projects.py     # Project endpoints
│   │   │   ├── tasks.py        # Task endpoints
│   │   │   ├── agents.py       # Agent info endpoints
│   │   │   ├── websocket.py    # Real-time UI stream
│   │   │   └── dead_letter.py  # Admin dead-letter queue
│   │   └── static/             # Dashboard frontend
│   ├── core/
│   │   ├── config.py           # Pydantic settings from .env
│   │   ├── message_bus.py      # Redis Streams messaging
│   │   ├── database.py         # SQLite persistence
│   │   ├── models.py           # Pydantic data models
│   │   ├── ollama_client.py    # LLM API wrapper
│   │   ├── metrics.py          # Prometheus metrics
│   │   ├── logger.py           # Structured logging
│   │   └── tools.py            # Tool definitions + validation
│   ├── orchestrator/
│   │   ├── pipeline.py         # Health check loop
│   │   ├── task_manager.py     # State machine & task CRUD
│   │   ├── guardrails.py       # Limits & validation
│   │   ├── workspace_manager.py# File storage, cleanup, git
│   │   └── sandbox.py          # Docker-based code validation
│   └── prompts/
│       ├── pm.txt
│       ├── research.txt
│       ├── spec.txt
│       ├── coder.txt
│       └── critic.txt
├── tests/
│   ├── unit/                   # Unit tests (230+)
│   ├── integration/            # Integration tests
│   └── conftest.py             # Shared mock fixtures
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```
