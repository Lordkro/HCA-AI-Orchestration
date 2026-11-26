# HCA Orchestration

**Hybrid Cognitive Architecture: an autonomous AI development team**

[![CI](https://github.com/Lordkro/HCA-AI-Orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/Lordkro/HCA-AI-Orchestration/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue?logo=docker)](https://github.com/Lordkro/HCA-AI-Orchestration/pkgs/container/hca-ai-orchestration)
[![License](https://img.shields.io/github/license/Lordkro/HCA-AI-Orchestration)](LICENSE)

A team of AI agents that turns a product idea into working code. A project manager plans the work, research, spec and coder agents carry it out, and a critic reviews every result, running the tests in a locked-down Docker sandbox. It's a single Rust binary with SQLite storage and a web dashboard, and it works with local models through Ollama or hosted ones from Anthropic, OpenAI and xAI (Grok).

## The team

| Agent | What it does |
| --- | --- |
| **Project manager** | Breaks the idea into a task graph with dependencies |
| **Research** | Investigates technologies, patterns and feasibility |
| **Spec** | Writes technical specs, API contracts and data models |
| **Coder** | Implements the code and runs it in the sandbox |
| **Critic** | Reviews each result, runs the tests and approves or asks for changes |

## Quick start

```sh
# 1. Build
cargo build --release -p hca

# 2. Build the sandbox image agents run code in (needs Docker)
docker build -t hca-sandbox -f docker/sandbox.Dockerfile docker

# 3. Configure: pick a model for each role in the [llm] section
./target/release/hca init
export XAI_API_KEY=...        # only for the providers your roles use
./target/release/hca check    # checks config, API keys and Docker

# 4. Run, then open the link it prints (it includes the access token)
./target/release/hca serve
```

`hca token` prints the dashboard link again. To run it with Docker Compose instead, follow the comments at the top of `docker-compose.yml`.

Without Docker, HCA still runs, but agents can't run commands or tests; reviewers then judge by reading the code.

## Crates

| Crate | Purpose |
| --- | --- |
| `hca-core` | Domain types, task and project state machines, guardrails, scheduling. Pure logic, no I/O. |
| `hca-store` | SQLite storage with `sqlx`: projects, tasks, history, activity log and the job queue |
| `hca-llm` | Providers: Ollama, Anthropic, OpenAI, Grok (xAI), any OpenAI-compatible server. Budget gate, context builder. |
| `hca-agents` | Agent loop, workspace tools, finishing tools, role prompts |
| `hca-sandbox` | A Docker container per agent run, with no network and memory, CPU and process limits, and the Docker guard |
| `hca-orchestrator` | Deterministic plan, work and review loop on the job queue |
| `hca-server` | `axum` API, WebSocket and the dashboard, with sign-in |
| `hca` | The binary: config, access token, startup, shutdown |

## How a project runs

1. You submit an idea. A **plan** job runs the PM agent, which submits a validated task graph.
2. Each ready task gets a **work** job. Its agent (research, spec or coder) works until it submits.
3. A **review** job runs the critic. Approval finishes the task; requested changes send it back to step 2 with the feedback.
4. After every job, the orchestrator asks `hca-core` what to do next: dispatch ready tasks, retry failed ones that still have a retry, or finish the project as completed or stuck.

Routing is plain code: no model decides where work goes. Each state change and the job that follows it are saved in one transaction, so a crash can't strand a task. On restart, `recover` re-queues anything that lost its job, including jobs the previous process was in the middle of. People can pause, resume or cancel a project, approve or reject work in review, add notes to a task, retry a failed task, and restart a failed project from the dashboard.

## What `hca-core` guarantees

- **Every project ends.** A task allows at most `max_revisions` revision rounds per attempt and `max_retries` retries. Both counters are private to the state machine, so callers can't skip them. Only a person can reopen a failed project or task, and each reopen grants one more bounded round, never an open-ended one.
- **Revisions can always continue.** A revision goes back through `InProgress` to `Review`. Review and revision can't deadlock or loop forever.
- **Only active projects accept work.** Paused, failed, completed and cancelled projects don't.
- **Budgets hold, even with parallel runs.** Before each model call its worst case is reserved in the database, against the tokens spent plus every reservation still open, so runs in parallel can't spend more than the budget between them. A call that can never fit ends the project as out of budget at once.
- **Plans are validated before they run.** Empty plans, too many tasks, unknown or self dependencies and cycles are rejected.
- **The orchestrator always knows what to do next.** `assess` returns `Running`, `NeedsRetry`, `Completed` or `Stuck`, never an open-ended retry.

`hca-core/tests/termination.rs` checks these with property tests over random task graphs, reviewer verdicts, agent crashes and people reopening tasks. A regression test replays a long review-and-revise loop and checks that it stops at the limit.

## What `hca-store` guarantees

- **One transaction per state change.** Loading a task, applying the core's rules, saving it and writing its history row commit together or not at all. Writes go through a single connection, so they never interleave.
- **A refused event writes nothing.** Invalid transitions, invalid plans and work on inactive projects leave the database untouched.
- **Jobs are leased, not lost.** A crashed worker's job is picked up again once its lease expires, or straight away when HCA restarts. A worker that lost its lease can't complete the job.
- **Retries are bounded.** Every claim is an attempt, and failed attempts wait longer each time (30 s, doubling, at most 10 minutes). After `max_attempts` (5) a job is dead and its task or project fails. A person can restart a failed project from the dashboard.
- **Paused work waits; finished work stops.** Jobs of paused projects stay queued. A running agent stops at its next model call and its job goes back on the queue with its attempt given back, so pausing never uses up attempts. When a project completes, fails or is cancelled, its jobs are cancelled in the same transaction. Spending the token budget does this too.
- **No billed token goes uncounted.** Each call's reservation is saved before the call and settled after it, to the project and the task in one transaction: with the real usage, nothing for an error the provider didn't bill (a refused connection or an error status), and the full worst case when the cost is unknown (a timeout, a lost response, a run cut off mid-call). A reservation a crash left open is charged in full at the next start. Shutdown requeues interrupted jobs with their attempt given back.

## What `hca-llm` provides

- **Five backends behind one trait:** Ollama, Anthropic, OpenAI, Grok and any OpenAI-compatible server (vLLM, LM Studio, OpenRouter). Grok goes through xAI's OpenAI-compatible Chat Completions API at `https://api.x.ai/v1`. xAI recommends its newer Responses API for new work; a client for it can be added behind the same trait.
- **A model per role.** For example, Claude for planning, Grok for coding and a local Qwen for research. The pm, coder and critic roles are required; research and spec use the coder's model when they have none of their own. API keys are read from environment variables (`ANTHROPIC_API_KEY`, `XAI_API_KEY`, `OPENAI_API_KEY` by default), never from the config file. Only providers that a role uses need a key.
- **A worst case per call, not a guess.** A call's worst case is one token per byte of everything sent, plus the providers' template overhead (hidden tool prompts included) and the full output allowance. That's a bound for byte-level tokenizers, which all supported providers use: no token is shorter than a byte. It's about three times what text and code really cost, so the last part of a budget goes unused. Conversations are trimmed to the context window with a realistic estimate (a token per three bytes) instead.
- **Retries never hide a cost.** The HTTP layer only retries what the provider didn't bill: refused connections, 429s and 5xx. A timed-out call goes back to the agent, so it's counted.
- **Known context windows.** Every role has a window conversations are trimmed to. Ollama's is `num_ctx` (32,768 if unset), and HCA always sends it, so Ollama never silently falls back to its own small default. Anthropic's is 200,000 tokens, OpenAI's and Grok's 128,000. An OpenAI-compatible server's is `context_tokens`, assumed to be 32,768 if unset, which `hca check` points out. `hca check` also warns when a window can't hold the brief, the output limit and room to work.
- **Typed tools.** `ToolSpec::of::<T>()` derives the JSON Schema from a Rust type, and `ToolCall::parse::<T>()` parses the model's arguments back into it. Arguments that aren't valid JSON are kept, so the agent can tell the model what went wrong.
- **Retries where they help.** Network errors, 429s and 5xx responses are retried with backoff; other errors are returned at once.
- **A context builder** in place of chat history. It assembles the task, the feedback, the test output (keeping the end, where errors are) and the files (keeping the start) within a token budget, and reports what it cut.
- **A scripted provider** (`hca_llm::testing`) for testing agents without a model.

```toml
[llm.providers.local]
kind = "ollama"
num_ctx = 32768

[llm.providers.claude]
kind = "anthropic"

[llm.providers.grok]
kind = "grok"

[llm.roles.pm]
provider = "claude"
model = "claude-opus-5-5"

[llm.roles.coder]
provider = "grok"
model = "grok-4"
max_tokens = 8192

[llm.roles.research]
provider = "local"
model = "qwen3:8b"
```

## What `hca-agents` provides

- **A real agent loop.** The model calls tools, sees their results and keeps going until it calls its finishing tool: `submit` (research, spec, coder), `submit_review` (critic) or `submit_plan` (PM).
- **Bounded runs.** A step limit (default 40), a time limit (`run_timeout_minutes`, default 120), the project's token budget reserved before every call, and at most two reminders when the model stops without finishing.
- **Conversations fit the model.** When a conversation nears the context window, the oldest tool output and old file contents in tool calls are replaced by a short note. Messages are never dropped, so every tool call keeps its result. A single result too big for a quarter of the window is cut, keeping the start of a file or the end of command output, and `read_file` takes `start_line` and `end_line` to read the rest.
- **Mistakes go back to the model.** A bad path, invalid arguments, a non-unique edit or an invalid plan becomes an error result the model can fix. It never crashes the run.
- **Tools per role:**

  | Tool | PM | Research | Spec | Coder | Critic |
  | --- | --- | --- | --- | --- | --- |
  | `list_files`, `read_file` | yes | yes | yes | yes | yes |
  | `write_file` | | yes | yes | yes | |
  | `edit_file` | | | yes | yes | |
  | `run_command` (sandbox only) | | | | yes | yes |

  The critic can't write files, through its tools or its commands. `run_command` only exists when a sandboxed runner is configured; nothing runs on the host.
- **A safe workspace.** Paths must be relative, can't contain `..`, can't touch `.git/` (no planted hooks, no reading tokens from `.git/config`) and can't follow a symlink out of the project. Reads and writes have size limits.
- **Checked handoffs.** A review that requests changes must say what to change. A plan is validated before it's accepted: dependencies must point at other tasks, with no cycles and within the task limit.
- **Five role prompts** in `hca-agents/prompts/`, each matching its role's tools.

## What `hca-sandbox` guarantees

Every agent run gets a container of its own with only its project's workspace mounted. The run's commands run inside it with `docker exec`, so a server started by one command can be tested by the next. Checked against a real Docker daemon:

- **No network** by default.
- **Limits:** 2 GiB of memory with no extra swap, 2 CPUs and 256 processes by default. A fork bomb hits the process limit, and a 3 GiB allocation fails.
- **Locked down:** read-only root filesystem with a small writable `/tmp`, all Linux capabilities dropped, `no-new-privileges`, non-root user.
- **Nothing outlives its run.** When a run ends its container is removed, before its work is handed on, so nothing it left running can change files under review. Parallel tasks never share processes, and ending one run doesn't touch another's. Tasks in parallel do share the project's files: that's how they build one project.
- **The critic is sealed off.** Its workspace is mounted read-only and its `/tmp` is its own. It can't change the project, or leave a startup hook for the coder's next command to run.
- **Timeouts stop runaway commands** by killing the container; the next command gets a fresh one.
- **Nothing is left running.** A project's containers are removed when it ends, and every sandbox container is removed at shutdown and at startup.
- **HCA never holds the Docker socket** in the Compose setup. It talks to the Docker guard (`hca docker-guard`), a deny-by-default proxy that lets through only what the sandbox does: containers of the sandbox image, non-root, with no network, no capabilities, a read-only root and resource limits, whose only mount is one project folder, and exec as the container's user. `--privileged`, mounting `/`, other images, the images, volumes and networks APIs, `archive` and `attach` all get 403, as does any field the guard doesn't know. So a compromised HCA can't use Docker to take over the host.
- **Folders can't be swapped for symlinks.** With the guard, workspaces are mounted as subpaths of a Docker volume, never as host paths. Docker refuses a subpath that leads out of the volume, so replacing a project folder with a symlink to `/` mounts nothing. The volume's folder is a mount point of its own in the HCA container, which can't be renamed or replaced.

## What `hca-server` guarantees

- **Sign-in required.** Every API call and the WebSocket need the access token. It's generated on first run, stored with owner-only permissions in the data folder, and printed as a dashboard link. The dashboard keeps it in the browser and removes it from the address bar.
- **Other websites can't use your session.** There are no CORS headers, and WebSocket connections from another origin are refused.
- **Local by default.** The server binds to `127.0.0.1`. The Compose file publishes the port on `127.0.0.1` only.
- **Built-in dashboard.** It's compiled into the binary, so there's nothing else to serve. Artifact names are escaped before they're shown.

## Contributing

```sh
cargo test                                   # all tests
cargo test -p hca-sandbox -- --ignored       # also the real-Docker sandbox tests
cargo clippy --all-targets -- -D warnings    # pedantic lints are on
cargo fmt --all
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the conventions.

## License

MIT, see [LICENSE](LICENSE).
