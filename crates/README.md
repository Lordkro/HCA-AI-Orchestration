# HCA v2 (Rust)

The Rust rewrite of HCA Orchestration. It is built next to the Python version, which stays in `src/` until v2 reaches parity.

## Status

| Crate | Purpose | State |
| --- | --- | --- |
| `hca-core` | Domain types, task and project state machines, guardrails, scheduling. Pure logic, no I/O. | Done |
| `hca-store` | SQLite storage with `sqlx`: projects, tasks, history and the job queue that replaces Redis | Done |
| `hca-llm` | Providers: Ollama, Anthropic, OpenAI, Grok (xAI), any OpenAI-compatible server. Budget gate, context builder. | Done |
| `hca-agents` | Agent loop, typed tools, role prompts | Next |
| `hca-sandbox` | Per-project containers through `bollard` | Planned |
| `hca-server` | `axum` API and WebSocket, compatible with today's dashboard | Planned |
| `hca` | The binary | Planned |

## What `hca-core` guarantees

- **Every project ends.** A task allows at most `max_revisions` revision rounds per attempt and `max_retries` retries. Both counters are private to the state machine, so callers can't skip them.
- **Revisions can always continue.** A revision goes back through `InProgress` to `Review`. The Python version had no valid path out of `REVISION`, which is how one task looped 1,940 times.
- **Only active projects accept work.** Paused, failed, completed and cancelled projects don't.
- **Budgets are checked before a model call.** `TokenBudget::authorize` refuses a call once the budget is spent or too small.
- **Plans are validated before they run.** Empty plans, too many tasks, unknown or self dependencies and cycles are rejected.
- **The orchestrator always knows what to do next.** `assess` returns `Running`, `NeedsRetry`, `Completed` or `Stuck`, never an open-ended retry.

`hca-core/tests/termination.rs` checks these with property tests over random task graphs, reviewer verdicts and agent crashes. It also has a regression test that replays the recorded Python loop.

## What `hca-store` guarantees

- **One transaction per state change.** Loading a task, applying the core's rules, saving it and writing its history row commit together or not at all. Writes go through a single connection, so they never interleave.
- **A refused event writes nothing.** Invalid transitions, invalid plans and work on inactive projects leave the database untouched.
- **Jobs are leased, not lost.** A crashed worker's job is picked up again once its lease expires. A worker that lost its lease can't complete the job.
- **Retries are bounded.** Every claim is an attempt. After `max_attempts` a job is dead and shows up for a person to requeue.
- **Paused work waits; finished work stops.** Jobs of paused projects stay queued. When a project completes, fails or is cancelled, its jobs are cancelled in the same transaction. Spending the token budget does this too.

## What `hca-llm` provides

- **Five backends behind one trait:** Ollama, Anthropic, OpenAI, Grok and any OpenAI-compatible server (vLLM, LM Studio, OpenRouter). Grok goes through xAI's OpenAI-compatible Chat Completions API at `https://api.x.ai/v1`. xAI recommends its newer Responses API for new work; a client for it can be added behind the same trait.
- **A model per role.** For example, Claude for planning, Grok for coding and a local Qwen for research. API keys are read from environment variables (`ANTHROPIC_API_KEY`, `XAI_API_KEY`, `OPENAI_API_KEY` by default), never from the config file. Only providers that a role uses need a key.
- **Budget checked before the call.** `Llm::chat` estimates the worst case (everything sent plus the full output allowance) and refuses the call if the project's budget can't cover it.
- **Typed tools.** `ToolSpec::of::<T>()` derives the JSON Schema from a Rust type, and `ToolCall::parse::<T>()` parses the model's arguments back into it. Arguments that aren't valid JSON are kept, so the agent can tell the model what went wrong.
- **Retries where they help.** Network errors, 429s and 5xx responses are retried with backoff; other errors are returned at once.
- **A context builder** in place of chat history. It assembles the task, the feedback, the test output (keeping the end, where errors are) and the files (keeping the start) within a token budget, and reports what it cut.
- **A scripted provider** (`hca_llm::testing`) for testing agents without a model.

```toml
[providers.local]
kind = "ollama"
num_ctx = 16384

[providers.claude]
kind = "anthropic"

[providers.grok]
kind = "grok"

[roles.pm]
provider = "claude"
model = "claude-opus-5-5"

[roles.coder]
provider = "grok"
model = "grok-4"
max_tokens = 8192

[roles.research]
provider = "local"
model = "qwen3:8b"
```

## Working on it

```sh
cargo test                                   # unit + property tests
cargo clippy --all-targets -- -D warnings    # pedantic lints are on
cargo fmt --all
```
