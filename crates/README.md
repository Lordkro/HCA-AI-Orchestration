# HCA v2 (Rust)

The Rust rewrite of HCA Orchestration. It is built next to the Python version, which stays in `src/` until v2 reaches parity.

## Status

| Crate | Purpose | State |
| --- | --- | --- |
| `hca-core` | Domain types, task and project state machines, guardrails, scheduling. Pure logic, no I/O. | Done |
| `hca-store` | SQLite storage with `sqlx`, migrations, job table | Next |
| `hca-llm` | Provider trait: Ollama, Anthropic, OpenAI-compatible. Context builder. | Planned |
| `hca-agents` | Agent loop, typed tools, role prompts | Planned |
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

`tests/termination.rs` checks these with property tests over random task graphs, reviewer verdicts and agent crashes. It also has a regression test that replays the recorded Python loop.

## Working on it

```sh
cargo test                                   # unit + property tests
cargo clippy --all-targets -- -D warnings    # pedantic lints are on
cargo fmt --all
```
