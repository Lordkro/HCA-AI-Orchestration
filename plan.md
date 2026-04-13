# HCA Orchestration — Project Plan

## Hybrid Cognitive Architecture: An Autonomous AI Development Team

**Goal:** Build a self-managing AI agent team powered by Ollama that can take a product idea and turn it into a working application — with minimal to no human intervention.

**Tech Stack:** Python · Ollama · Docker · Redis · FastAPI · SQLite

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard (UI)                    │
│           FastAPI + WebSocket + HTML/JS/CSS              │
└──────────────────────┬──────────────────────────────────┘
                       │ REST / WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                  Orchestrator Service                    │
│         (Task lifecycle, routing, state machine)         │
└──────────────────────┬──────────────────────────────────┘
                       │ Redis Streams (message bus)
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│  Agent 1   │  │  Agent 2   │  │  Agent N   │
│ (PM, Coder │  │ (Research, │  │ (Critic,   │
│  etc.)     │  │  Spec)     │  │  etc.)     │
└─────┬──────┘  └─────┬──────┘  └─────┬──────┘
      │               │               │
      └───────────────┼───────────────┘
                      ▼
              ┌──────────────┐
              │    Ollama    │
              │  (LLM API)  │
              └──────────────┘
```

**Shared Resources:**
- **Redis** — Message bus (agent-to-agent communication) + pub/sub for UI updates
- **SQLite** — Persistent state (projects, tasks, conversations, artifacts)
- **Shared Volume** — File workspace where generated code/docs live

---

## Agent Definitions

### 1. Project Manager (PM) Agent
- **Role:** Receives the initial product idea, breaks it into an execution plan, delegates tasks, tracks progress, and decides when the project is complete.
- **Inputs:** Product idea (from user), status updates (from other agents)
- **Outputs:** Project plan, task assignments, status decisions
- **Key Behaviors:**
  - Decompose a product idea into milestones and tasks
  - Assign tasks to the appropriate agent
  - Monitor task completion and handle blockers
  - Decide when to iterate vs. when to ship
  - Final sign-off on deliverables

### 2. Research Agent
- **Role:** Investigates technologies, patterns, and approaches relevant to the project. Provides context and recommendations.
- **Inputs:** Research questions from PM or Spec agent
- **Outputs:** Research reports, technology recommendations, reference examples
- **Key Behaviors:**
  - Analyze feasibility of requested features
  - Recommend tech stack, libraries, patterns
  - Provide context that other agents can reference
  - Summarize findings concisely

### 3. Specification Agent
- **Role:** Translates high-level ideas and research into detailed technical specifications.
- **Inputs:** Project plan from PM, research reports from Research Agent
- **Outputs:** Technical specs, API contracts, data models, architecture docs
- **Key Behaviors:**
  - Write detailed feature specifications
  - Define API endpoints, request/response schemas
  - Create data model definitions
  - Produce architecture diagrams (as text/mermaid)
  - Version and update specs based on feedback

### 4. Coder Agent
- **Role:** Writes the actual code based on specifications.
- **Inputs:** Technical specs from Spec Agent, feedback from Critic Agent
- **Outputs:** Source code files, configuration files, scripts
- **Key Behaviors:**
  - Generate code that follows the specification
  - Write unit tests alongside implementation
  - Fix issues identified by the Critic
  - Structure code with proper project layout
  - Handle multiple languages/frameworks as needed

### 5. Critic Agent
- **Role:** Reviews all outputs (specs, code, docs) for quality, correctness, and completeness.
- **Inputs:** Any artifact from any other agent
- **Outputs:** Review feedback, approval/rejection decisions, improvement suggestions
- **Key Behaviors:**
  - Review code for bugs, security issues, and best practices
  - Validate specs for completeness and consistency
  - Run/verify tests if possible
  - Provide actionable, specific feedback
  - Approve artifacts when quality bar is met

---

## Implementation Phases

### Phase 1: Foundation (Core Infrastructure)
> Get the basic plumbing working so agents can exist and communicate.

- [x] **1.1 — Project scaffolding**
  - Initialize Python project with `pyproject.toml`
  - Set up directory structure (see below)
  - Create base `Dockerfile` and `docker-compose.yml`
  - Set up `.env` for configuration

- [x] **1.2 — Ollama integration layer**
  - Create `OllamaClient` wrapper class
  - Support configurable models per agent
  - Handle prompt formatting, streaming, retries
  - Add token/context window management

- [x] **1.3 — Message bus (Redis Streams)**
  - Set up Redis in Docker Compose
  - Create `MessageBus` class for publish/subscribe
  - Define message schema (sender, recipient, type, payload, timestamp)
  - Implement message routing logic

- [x] **1.4 — Database layer (SQLite)** ✅
  - Define schema: projects, tasks, messages, artifacts, project_events
  - Create `Database` class with async support (WAL mode, foreign keys, busy timeout)
  - Implement full CRUD for all entities with pagination, filtering, search
  - Add migration support (versioned schema with `schema_version` table)
  - Project events timeline for UI history tracking
  - Database diagnostics (`get_stats()`) for monitoring
  - Duplicate-safe message saving (`INSERT OR IGNORE`)
  - `get_latest_artifact()` for version-aware file retrieval

- [x] **1.5 — Base Agent framework** ✅
  - Abstract `BaseAgent` with full lifecycle (start → heartbeat → consume → process → stop)
  - Per-project conversation memory isolation (agents don't mix project contexts)
  - LLM interaction via `think()` with automatic context-window trimming
  - `AgentStats` dataclass — messages received/sent/failed, LLM calls, think time, uptime
  - Heartbeat events every 30s for dashboard agent-status display
  - Graceful shutdown with in-flight message draining (up to 60s)
  - Hot-reloadable system prompts (`reload_prompt()`)
  - Retry-before-dead-letter (1 retry attempt before moving to DLQ)
  - `get_info()` for monitoring API — role, status, model, active projects, stats
  - All concrete agents updated to pass `project_id` to `think()`
  - Agent API routes updated to use `get_info()`

**Deliverable:** Agents can start up, connect to Redis, send/receive messages, call Ollama, and persist state.

---

### Phase 1.6: Testing Foundation
> Comprehensive unit and integration tests for all hardened foundation layers.

- [ ] **1.6 — Test suite**
  - Shared fixtures in `conftest.py` (temp DB, mock Ollama, mock Redis bus)
  - `test_ollama_client.py` — token estimation, context trimming, retry logic, stats
  - `test_message_bus.py` — publish/consume round-trip, ack, dead-letter, UI events (requires Redis mock)
  - `test_database.py` — full CRUD for projects/tasks/artifacts/messages, migrations, search, pagination, stats
  - `test_agents.py` — BaseAgent lifecycle, per-project memory, heartbeat, graceful stop, message routing
  - `test_pipeline.py` — TaskManager state machine, valid/invalid transitions, iteration limits

**Deliverable:** `pytest` passes with full coverage of Phases 1.1–1.5. All tests run offline (no Ollama, no Redis required).

---

### Phase 2: Agent Implementation
> Build out each agent with its specific behavior and prompts.

- [ ] **2.1 — Project Manager Agent**
  - System prompt engineering for PM role
  - Project decomposition logic (idea → milestones → tasks)
  - Task assignment and routing
  - Progress tracking and decision-making
  - Pipeline orchestration (who works when)

- [ ] **2.2 — Research Agent**
  - System prompt for research/analysis role
  - Structured output for research reports
  - Ability to reference previous findings
  - Knowledge synthesis across multiple queries

- [ ] **2.3 — Specification Agent**
  - System prompt for technical specification writing
  - Template-based spec generation
  - Schema/API definition output formatting
  - Spec versioning on feedback

- [ ] **2.4 — Coder Agent**
  - System prompt for code generation
  - File creation and management in shared workspace
  - Multi-file project generation
  - Test generation alongside code
  - Iterative fixing based on Critic feedback

- [ ] **2.5 — Critic Agent**
  - System prompt for code/spec review
  - Structured review output (issues, severity, suggestions)
  - Approval/rejection decision logic
  - Quality scoring system

**Deliverable:** All five agents can participate in a full product development cycle via message passing.

---

### Phase 3: Orchestration & Workflow Engine
> Make the agents work together as a cohesive team.

- [ ] **3.1 — Task state machine**
  - Define task states: `pending → assigned → in_progress → review → approved/rejected → done`
  - Implement state transitions with validation
  - Add timeout and retry logic

- [ ] **3.2 — Pipeline definitions**
  - Define the standard workflow: `PM → Research → Spec → Code → Critic → (iterate) → Done`
  - Support parallel task execution where possible
  - Handle dependencies between tasks

- [ ] **3.3 — Feedback loops**
  - Critic → Coder iteration cycle (with max iteration cap)
  - Critic → Spec revision cycle
  - PM escalation on repeated failures

- [ ] **3.4 — Guardrails & safety**
  - Maximum iterations per task (prevent infinite loops)
  - Total project token budget tracking
  - Deadlock detection
  - Human override / pause mechanism
  - Activity timeout detection

**Deliverable:** Agents autonomously execute a full project pipeline from idea to code with quality gates.

---

### Phase 4: Web Dashboard (UI)
> Build an interface to observe and interact with the agent team.

- [ ] **4.1 — FastAPI backend**
  - REST endpoints: projects, tasks, agents, messages, artifacts
  - WebSocket endpoint for real-time updates
  - Project creation endpoint (submit a new idea)

- [ ] **4.2 — Dashboard frontend**
  - Project overview page (active projects, status)
  - Agent activity feed (real-time message stream)
  - Task board (Kanban-style view of task states)
  - Conversation viewer (see agent-to-agent dialogues)
  - Artifact browser (view generated specs, code, docs)

- [ ] **4.3 — Controls**
  - Start new project (submit product idea)
  - Pause / resume project
  - Human override: inject a message into the agent conversation
  - Agent configuration (change models, adjust prompts)

- [ ] **4.4 — Real-time updates**
  - WebSocket push for new messages, task state changes
  - Live agent status indicators (idle, thinking, working)
  - Progress bars for milestones

**Deliverable:** Fully functional web UI where you can submit ideas, watch agents work, and browse outputs.

---

### Phase 5: Docker & Deployment
> Containerize everything for easy setup and reproducibility.

- [ ] **5.1 — Docker Compose finalization**
  - One container per agent (scalable)
  - Redis container
  - Ollama container (with GPU passthrough if available)
  - Web UI container
  - Shared volume for workspace files

- [ ] **5.2 — Configuration management**
  - Environment-based config (`.env` file)
  - Per-agent model selection
  - Resource limits (memory, CPU per container)

- [ ] **5.3 — Startup orchestration**
  - Health checks for all services
  - Dependency ordering (Redis → Agents → UI)
  - Graceful shutdown handling

- [ ] **5.4 — Developer experience**
  - `docker compose up` one-command startup
  - Hot-reload for development
  - Log aggregation across containers
  - README with setup instructions

**Deliverable:** `docker compose up` starts the entire system, ready to accept project ideas.

---

### Phase 6: Polish & Hardening
> Make it robust and pleasant to use.

- [ ] **6.1 — Error recovery**
  - Agent crash recovery and restart
  - Message replay on failure
  - Partial project resume

- [ ] **6.2 — Observability**
  - Structured logging across all agents
  - Metrics collection (tasks completed, tokens used, time per phase)
  - Health dashboard

- [ ] **6.3 — Prompt optimization**
  - Refine system prompts based on testing
  - Add few-shot examples for better output quality
  - Tune temperature/parameters per agent role

- [ ] **6.4 — Documentation**
  - Architecture documentation
  - User guide
  - Agent prompt cookbook
  - Contributing guide

**Deliverable:** Production-quality system with good error handling, observability, and docs.

---

## Directory Structure

```
HCA-Orchestration/
├── docker-compose.yml          # All services defined here
├── Dockerfile                  # Base image for agents
├── Dockerfile.ui               # Image for the web dashboard
├── pyproject.toml              # Python project config
├── .env.example                # Environment variable template
├── README.md                   # Project documentation
├── plan.md                     # This file
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                   # Shared infrastructure
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── ollama_client.py    # Ollama API wrapper
│   │   ├── message_bus.py      # Redis Streams wrapper
│   │   ├── database.py         # SQLite persistence layer
│   │   ├── models.py           # Data models (Pydantic)
│   │   └── logger.py           # Structured logging setup
│   │
│   ├── agents/                 # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Abstract base agent class
│   │   ├── pm_agent.py         # Project Manager
│   │   ├── research_agent.py   # Research Agent
│   │   ├── spec_agent.py       # Specification Agent
│   │   ├── coder_agent.py      # Coder Agent
│   │   └── critic_agent.py     # Critic Agent
│   │
│   ├── orchestrator/           # Workflow engine
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Pipeline definitions
│   │   ├── task_manager.py     # Task state machine
│   │   └── guardrails.py       # Safety limits and controls
│   │
│   ├── api/                    # Web API layer
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI application
│   │   ├── routes/
│   │   │   ├── projects.py     # Project endpoints
│   │   │   ├── agents.py       # Agent status endpoints
│   │   │   ├── tasks.py        # Task endpoints
│   │   │   └── websocket.py    # WebSocket handler
│   │   └── static/             # Frontend assets
│   │       ├── index.html
│   │       ├── css/
│   │       │   └── styles.css
│   │       └── js/
│   │           └── app.js
│   │
│   └── prompts/                # System prompts for each agent
│       ├── pm.txt
│       ├── research.txt
│       ├── spec.txt
│       ├── coder.txt
│       └── critic.txt
│
├── workspace/                  # Shared volume for generated projects
│   └── .gitkeep
│
├── tests/                      # Test suite
│   ├── test_ollama_client.py
│   ├── test_message_bus.py
│   ├── test_database.py
│   ├── test_agents.py
│   └── test_pipeline.py
│
└── scripts/                    # Utility scripts
    ├── setup_ollama_models.sh  # Pull required models
    └── reset_workspace.sh      # Clean workspace state
```

---

## Message Schema

All agent-to-agent communication follows this format:

```json
{
  "id": "uuid",
  "timestamp": "2026-04-13T12:00:00Z",
  "sender": "pm_agent",
  "recipient": "coder_agent",       // or "*" for broadcast
  "type": "task_assignment",         // task_assignment, deliverable, feedback, status_update, question, answer
  "project_id": "uuid",
  "task_id": "uuid",
  "payload": {
    "content": "...",                // The actual message/instruction
    "artifacts": ["file1.py"],       // Referenced files
    "metadata": {}                   // Additional context
  },
  "priority": "normal"              // low, normal, high, critical
}
```

---

## Task States

```
              ┌──────────┐
              │ pending   │
              └────┬─────┘
                   │ PM assigns
              ┌────▼─────┐
              │ assigned  │
              └────┬─────┘
                   │ Agent picks up
              ┌────▼──────┐
              │ in_progress│
              └────┬──────┘
                   │ Agent submits
              ┌────▼─────┐
              │  review   │◄────────┐
              └────┬─────┘         │
                   │               │ Critic rejects
              ┌────▼─────┐   ┌────┴──────┐
              │ approved  │   │ revision  │
              └────┬─────┘   └───────────┘
                   │
              ┌────▼─────┐
              │   done    │
              └──────────┘
```

---

## Host Hardware

| Component | Specification                    |
|-----------|----------------------------------|
| GPU       | AMD Radeon RX 6900 XT (16GB VRAM)|
| CPU       | AMD Ryzen 7 5800X (8C/16T)       |
| RAM       | 32 GiB DDR4                      |
| OS        | Linux (Bazzite)                  |

**GPU Note:** AMD GPUs use ROCm for acceleration, which is fully supported on Linux.
Ollama will run in a Docker container with ROCm passthrough for GPU acceleration.
The 32B model (~20-22GB) exceeds the 16GB VRAM, so Ollama will use **partial GPU
offload** — ~75% of layers on GPU, remainder in system RAM. This yields excellent
performance (~15-25 tok/s) with the highest quality model that fits this hardware.

## Model Strategy

**Single-model approach:** All agents use the same base model, differentiated only
by system prompts. This avoids costly model-swap overhead. With ROCm GPU acceleration
and partial offload, the 32B model runs at excellent speeds on this hardware.

| Tier       | Model                | VRAM+RAM     | Speed (est.)     | When to Use                             |
|------------|----------------------|--------------|------------------|-----------------------------------------|
| Default    | `qwen2.5:32b`       | 16GB + ~6GB  | ~15-25 tok/s     | Best quality, recommended               |
| Coder swap | `qwen2.5-coder:32b` | 16GB + ~6GB  | ~15-25 tok/s     | Optional: swap in for coding tasks only |
| Fallback   | `qwen2.5:14b`       | ~10 GB       | ~30-50 tok/s     | If 32B is too slow for iteration        |

## Configuration Defaults

| Setting                  | Default Value              |
|--------------------------|----------------------------|
| Ollama base URL          | `http://ollama:11434`      |
| Default model            | `qwen2.5:32b`              |
| PM model                 | `qwen2.5:32b`              |
| Research model           | `qwen2.5:32b`              |
| Spec model               | `qwen2.5:32b`              |
| Coder model              | `qwen2.5-coder:32b`        |
| Critic model             | `qwen2.5:32b`              |
| Redis URL                | `redis://redis:6379`       |
| Max iterations per task  | `5`                        |
| Max tasks per project    | `50`                       |
| Task timeout (minutes)   | `30`                       |
| Web UI port              | `8080`                     |
| Ollama GPU mode          | `rocm`                     |

---

## Implementation Order

We build bottom-up so each layer is testable before the next:

```
Phase 1  ████████████████░░░░░░░░░░░░░░░░  Foundation
Phase 2  ░░░░░░░░████████████████░░░░░░░░  Agents
Phase 3  ░░░░░░░░░░░░░░░░████████████░░░░  Orchestration
Phase 4  ░░░░░░░░░░░░░░░░░░░░████████████  UI
Phase 5  ░░░░░░░░░░░░░░░░░░░░░░░░████████  Docker
Phase 6  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████  Polish
```

---

## Next Step

**Begin Phase 1.1** — Scaffold the project directory, create `pyproject.toml`, `docker-compose.yml`, `.env.example`, and all empty module files so we have the skeleton ready.
