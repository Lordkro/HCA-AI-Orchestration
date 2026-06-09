# Contributing

## Development Setup

### Prerequisites

- Python 3.11+
- A running Ollama instance (or mock for tests)
- A running Redis instance (or mock for tests)

### Local Install

```bash
git clone <repo-url>
cd HCA-Orchestration
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the package in editable mode with dev dependencies (pytest, ruff, mypy, etc.).

### Environment

```bash
cp .env.example .env
# Edit .env to point to your local services:
# OLLAMA_BASE_URL=http://localhost:11434
# REDIS_URL=redis://localhost:6379/0
# DATABASE_URL=sqlite:///data/hca.db
# HCA_API_KEY=your-test-key (optional)
```

---

## Running Tests

### Unit Tests

```bash
pytest tests/unit/ -v
```

230+ unit tests covering agents, API, message bus, database, and orchestrator. These use mock fixtures and do not require running services.

### Integration Tests

```bash
pytest tests/integration/ -v
```

Integration tests exercise the full agent pipeline with mock LLM responses. They require a running Redis instance (or use the mock).

### All Tests

```bash
pytest tests/ -v
```

### Test Fixtures

Shared fixtures are in `tests/conftest.py`. Key mocks:
- `MockOllamaClient` — returns canned LLM responses, no HTTP
- `MockRedis` / `MockPubSub` — in-memory Redis mock
- `MockMessageBus` — lightweight message bus for unit tests
- `MockDatabase` — SQLite in-memory database

### Writing Tests

- Place unit tests in `tests/unit/` matching the module structure
- Place integration tests in `tests/integration/`
- Use `pytest.mark.asyncio` for async tests
- Prefer mock-based tests over integration tests where possible
- All new features should include tests

---

## Code Style

### Ruff

```bash
ruff check src/ tests/
```

We use `ruff` for linting and formatting. The config is in `pyproject.toml`.

To auto-fix issues:

```bash
ruff check --fix src/ tests/
```

### Mypy

```bash
mypy src/
```

All code must pass mypy strict mode. Type hints are required on all function signatures.

### Pre-commit Checks

Before submitting a PR, run:

```bash
ruff check src/ tests/
mypy src/
pytest tests/unit/ -x --tb=short
```

All three must pass.

---

## Project Structure

```
src/hca/
├── main.py              # Application entrypoint (bootstrap sequence)
├── agents/              # Agent implementations
│   ├── base_agent.py    # Abstract base, think()/think_with_tools()
│   ├── pm_agent.py      # Project Manager
│   ├── research_agent.py
│   ├── spec_agent.py
│   ├── coder_agent.py
│   └── critic_agent.py
├── api/                 # FastAPI layer
│   ├── app.py           # App factory, middleware, CORS
│   └── routes/
│       ├── agents.py
│       ├── dead_letter.py
│       ├── hitl.py
│       ├── projects.py
│       ├── tasks.py
│       └── websocket.py
├── core/                # Shared infrastructure
│   ├── config.py        # Pydantic-based settings
│   ├── database.py      # SQLite persistence
│   ├── logger.py        # Structured logging setup
│   ├── message_bus.py   # Redis Streams messaging
│   ├── metrics.py       # Prometheus metrics
│   ├── models.py        # Data models (TaskState, AgentMessage, etc.)
│   ├── ollama_client.py # LLM API wrapper
│   └── tools.py         # Tool definitions for agents
├── orchestrator/        # Workflow engine
│   ├── guardrails.py    # Limits and validation
│   ├── pipeline.py      # Health check loop
│   ├── sandbox.py       # Docker-based code validation
│   ├── task_manager.py  # State machine + task CRUD
│   └── workspace_manager.py
└── prompts/             # System prompts (hot-reloadable)
    ├── pm.txt
    ├── research.txt
    ├── spec.txt
    ├── coder.txt
    └── critic.txt
```

---

## Adding a New Agent

### 1. Add the role

Add your agent to the `AgentRole` enum in `src/hca/core/models.py`:

```python
class AgentRole(StrEnum):
    PM = "pm"
    RESEARCH = "research"
    SPEC = "spec"
    CODER = "coder"
    CRITIC = "critic"
    SYSTEM = "system"
    USER = "user"
    YOUR_AGENT = "your_agent"   # New
```

### 2. Create the agent class

Create `src/hca/agents/your_agent.py`:

```python
"""YourAgent — description of what this agent does."""

from hca.agents.base_agent import BaseAgent
from hca.core.database import Database
from hca.core.message_bus import MessageBus
from hca.core.models import AgentMessage, AgentRole, MessageType
from hca.core.ollama_client import OllamaClient


class YourAgent(BaseAgent):
    def __init__(self, *, bus, ollama, db, task_manager=None):
        super().__init__(
            role=AgentRole.YOUR_AGENT, bus=bus, ollama=ollama, db=db,
            task_manager=task_manager,
        )

    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        # Handle relevant message types
        match message.type:
            case MessageType.TASK_ASSIGNMENT:
                return await self._handle_task(message)
            case _:
                return None

    async def _handle_task(self, message: AgentMessage) -> AgentMessage | None:
        self._set_activity("Working on assignment")
        response = await self.think(
            f"Task: {message.payload.content}",
            project_id=message.project_id,
            task_id=message.task_id,
        )
        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
            metadata={"artifact_type": "your_agent_report"},
        )
```

### 3. Create the system prompt

Create `src/hca/prompts/your_agent.txt` with the agent's instructions.

### 4. Register the agent

In `src/hca/main.py`, add your agent to the startup sequence:

```python
from hca.agents.your_agent import YourAgent

# In main():
your_agent = YourAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager)
agents = [pm_agent, research_agent, spec_agent, coder_agent, critic_agent, your_agent]
```

Also update `bus.setup_agent_streams()` to include your agent's role.

### 5. Add config overrides

Add per-agent model/temperature/top_p fields to `src/hca/core/config.py` and the corresponding env vars.

### 6. Wire into the pipeline

Update `PMAgent._determine_next_agent()` or the pipeline routing to include your agent in the workflow.

### 7. Write tests

Add unit tests in `tests/unit/test_your_agent.py` and update `tests/conftest.py` if new mock fixtures are needed.

### 8. Update prompts if integrating with PM

If the PM should assign tasks to your agent, add your agent to the `create_task` tool's `assigned_to` enum in `src/hca/core/tools.py`.

---

## CI/CD

The CI pipeline runs on every PR to `main`:

1. **Test job** (Python 3.11, 3.12)
   - `ruff check src/ tests/`
   - `mypy src/`
   - `pytest tests/unit/ --cov=src/hca`
2. **Docker job** (`main` pushes only)
   - Build and push to `ghcr.io`

Checks must pass before merging.

---

## Making a PR

1. Create a branch: `issue-N-short-description` (e.g., `issue-42-add-notifications`)
2. Make your changes with clear commit messages
3. Run the pre-commit checks (ruff, mypy, tests)
4. Push and create a PR against `main`
5. Ensure CI passes
6. Request review

---

## Logging

Use `structlog` for all logging. The convention is lowercase_with_underscores for event names:

```python
logger.info("agent_event_name", key1="value1", key2=42)
```

Avoid f-strings in log messages — use structured fields instead.

Metrics use Prometheus client library. Register new metrics in `src/hca/core/metrics.py`.
