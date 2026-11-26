# Contributing

## Setup

You need:

- Rust stable (`rust-toolchain.toml` installs it with `rustfmt` and `clippy` through rustup)
- Docker, for the sandbox and its test (optional for everything else)

```sh
git clone https://github.com/Lordkro/HCA-AI-Orchestration.git
cd HCA-AI-Orchestration
cargo build
docker build -t hca-sandbox -f docker/sandbox.Dockerfile docker
```

No model or API key is needed for the tests. They use `hca_llm::testing::ScriptedProvider` and `wiremock` servers instead of real providers.

## Checks

CI runs these on every push and pull request, and all of them must pass:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
cargo doc --no-deps --locked
```

Clippy runs with the `pedantic` group on, so fix lints rather than silencing them. If an `allow` is really needed, put it on the smallest item and add a comment saying why.

The real-Docker sandbox test is ignored by default. Run it when you change `hca-sandbox`:

```sh
cargo test -p hca-sandbox -- --ignored
```

CI also builds both Docker images, runs that test and validates `docker-compose.yml`.

## Layout

| Path | What's there |
| --- | --- |
| `crates/hca-core` | Domain types and state machines. Pure logic: no I/O, no async. |
| `crates/hca-store` | SQLite storage and the job queue. Migrations in `migrations/`. |
| `crates/hca-llm` | Model providers, budget gate, context builder |
| `crates/hca-agents` | Agent loop, workspace and tools. Role prompts in `prompts/`. |
| `crates/hca-sandbox` | Docker sandbox |
| `crates/hca-orchestrator` | Plan, work and review loop |
| `crates/hca-server` | HTTP API, WebSocket and the dashboard (`static/`) |
| `crates/hca` | The binary and its config |
| `docker/` | The HCA and sandbox images |
| `hca.example.toml` | Documented example config. `hca init` copies it. |

## Conventions

- **State changes go through `hca-core`.** Task and project state only changes by applying an event. Don't set fields directly in the store or the orchestrator.
- **One transaction per state change.** If a change also queues a job, use the store method that does both, so a crash can't leave a task without its job.
- **Routing is code.** The orchestrator decides what runs next from `assess`; models never pick where work goes.
- **Agent mistakes are tool errors.** Bad arguments or paths become an error result the model can fix, not a failed run.
- **Nothing runs on the host.** Commands only run through the sandbox.
- **Secrets come from environment variables.** Never from the config file, never logged.
- **Log with `tracing`**, not `println!`, outside the CLI's own output.
- **Errors use `thiserror`** in library crates. Avoid `unwrap` and `expect` outside tests.
- **Test new behavior.** Unit tests next to the code, cross-crate tests in the crate's `tests/` folder. Changes to the state machines should keep the property tests in `hca-core/tests/termination.rs` passing, and add one when they add a rule.

## Adding an agent role

1. Add the variant to `AgentRole` in `hca-core/src/role.rs`, including `as_str`, `ALL` and `can_own_tasks`.
2. Write its prompt in `hca-agents/prompts/` and add it to `system_prompt` in `hca-agents/src/lib.rs`.
3. Give it tools and a finishing tool in `hca-agents/src/tools.rs`.
4. Add an `[llm.roles.<name>]` entry to `hca.example.toml`.
5. If the PM should assign it tasks, mention it in `prompts/pm.txt`.
6. Add tests: the agent loop with a scripted provider, and a project run in `hca-orchestrator/tests/`.

## Pull requests

- Branch from `main` and keep each pull request to one change.
- Say what changed and why, and how you tested it.
- Update `README.md`, `hca.example.toml` or the prompts when behavior they describe changes.
- Releases are tagged `v*`; the release workflow builds and publishes the image.
