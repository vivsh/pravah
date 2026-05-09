# Pravah — Agent Working Guide

Operational instructions for AI agents working in this repository.
Read `ARCHITECTURE.md` first for type contracts and API signatures.

---

## Commands

```bash
cargo test                        # full suite: unit + integration + doc tests
cargo test --test flows           # flow integration tests only (tests/flows.rs)
cargo test --test integration     # provider/tool integration tests only
cargo build --examples            # verify examples compile
cargo check                       # fast type-check without building
```

All commands run from repo root. No environment variables required for the pure-Rust test suite. Integration tests that hit real providers require `OPENAI_API_KEY` etc. and are gated with `#[ignore]` or `dotenvy`.

---

## Module Map

```
src/
  lib.rs                  — public re-exports
  commons.rs              — Agent trait, AgentConfig
  context.rs              — Context, FlowConf
  deps.rs                 — DepsError, dependency container
  utils.rs                — topo_layers helper

  flows/
    flows.rs              — FlowGraph, FlowBuilder, FlowRuntime, all node types,
                            all validation, step/resume logic
    state.rs              — FlowState (runtime state map)
    mod.rs                — re-exports

  clients/
    mod.rs                — Client trait, ClientFactory trait, ClientOptions,
                            ClientResponse, Message, ClientHistory, Provider enum
    openai.rs             — OpenAI provider
    anthropic.rs          — Anthropic provider
    gemini.rs             — Gemini provider (feature-gated: provider-gemini)
    ollama.rs             — Ollama provider
    genai.rs              — GenAI adapter (feature-gated: provider-genai)
    schema.rs             — shared schema sanitization utilities

  tools/
    base.rs               — Tool trait, ToolBox, ToolBoxBuilder, Context::resolve,
                            Context::check_command, ToolError
    fs.rs                 — ReadFile, WriteFile, ListDir tools + unit tests
    cmd.rs                — RunCommand tool
    web.rs                — FetchUrl tool (uses scraper + dom_smoothie)
    mod.rs                — re-exports

tests/
  flows.rs                — flow graph integration tests (mock client, no network)
  integration.rs          — provider + tool tests (some require live keys)
```

---

## Where to Add Things

### New LLM provider

1. Create `src/clients/{name}.rs`
2. Implement `Client` trait (`async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError>`)
3. Add `#[async_trait]` on the impl block (all existing providers use it)
4. Add a feature gate in `Cargo.toml` under `[features]` if it has an optional dep
5. Register in `src/clients/mod.rs`: add a match arm in the `ClientFactory` impl that parses the `{name}://` scheme from `model_url`

### New built-in tool

1. Create `src/tools/{name}.rs` or add to an existing file
2. Implement `Tool` trait — the struct is both the input schema and the tool receiver
3. Add `use crate::tools::{name}::MyTool;` and re-export from `src/tools/mod.rs`

### New flow node kind

This is a core change. Touch all of these in order:

1. `FlowNode` enum — add variant with an `Info` struct
2. `FlowBuilder` — add builder method
3. `validate_nodes()` — add structural validation
4. `collect_terminal_state_ids()` — add terminal successor extraction
5. `FlowGraph::step()` — add dispatch arm
6. `FlowBuilder::flow()` inliner — add rename case for the new node kind

---

## Gotchas

**`FlowGraph` does not derive `Debug`.**
Do not use `.unwrap_err()` or `.expect_err()` on `Result<FlowGraph, _>` in tests — the compiler rejects it. Use `match` instead:

```rust
let err = match SomeFlow::build() {
    Ok(_) => panic!("expected error"),
    Err(e) => e,
};
```

**Node id == schema name == Rust type name.**
If `MyAgent::Output` is also named `MyAgent`, the builder inserts a duplicate key and returns `FlowError::Invalid`. Always use distinct names for input and output types.

**`join` inserts two entries in the node map, one per parent.**
Both entries share the same closure (via `Arc`). This is intentional — the step loop encounters either parent state first and checks readiness on both. Do not treat this as a bug.

**The `submit` sentinel is auto-injected.**
`ToolBox::with_agent::<A>()` appends the exit tool automatically when the toolbox is non-empty. Adding it manually causes a duplicate tool name. Never add `submit` to a `ToolBox` in `Agent::build()`.

**Structured-output mode vs tool-calling mode.**
An agent with an empty `ToolBox` (the default) uses structured-output mode: the provider receives the output JSON schema and returns a typed object. An agent with tools uses tool-calling mode with the injected `submit` sentinel. The switch is automatic — do not pass `output_schema` manually.

**`async_trait` is still required.**
The codebase uses Rust 2024 edition, but `Client` is an object-safe trait used as `dyn Client`. `async fn` in `dyn`-compatible traits still requires `#[async_trait]`. Do not remove it from `Client` impls.

**`tokio` features are narrowed.**
`Cargo.toml` uses `["rt-multi-thread", "macros", "fs", "process"]` — not `full`. Do not add new tokio sub-modules without adding the corresponding feature.

---

## Feature Flags

| Flag                           | What it enables                   |
| ------------------------------ | --------------------------------- |
| `provider-openai` (default)    | OpenAI client                     |
| `provider-anthropic` (default) | Anthropic client                  |
| `provider-gemini` (default)    | Gemini client + `gemini-rust` dep |
| `provider-ollama` (default)    | Ollama client                     |
| `provider-genai`               | GenAI adapter + `genai` dep       |

---

## Test Conventions

- Unit tests live in a `#[cfg(test)] mod tests` block inside the source file
- Flow integration tests go in `tests/flows.rs` using the mock `ClientFactory`
- Provider/network tests go in `tests/integration.rs` and are skipped without live keys
- Unix-specific tests (symlink sandbox) use `#[cfg(unix)]`
- Use `#[tokio::test]` for async tests; plain `#[test]` for sync tests

---

## Dependencies (summary)

| Crate                  | Purpose                                                      |
| ---------------------- | ------------------------------------------------------------ |
| `tokio`                | Async runtime, fs, process                                   |
| `serde` / `serde_json` | Serialization                                                |
| `schemars`             | JSON Schema derivation for tool/agent types                  |
| `thiserror`            | Error enum macros                                            |
| `async-trait`          | Object-safe async traits                                     |
| `reqwest`              | HTTP client for provider API calls and web tool              |
| `scraper`              | HTML parsing for `FetchUrl` tool                             |
| `dom_smoothie`         | Readability extraction for `FetchUrl` tool                   |
| `tracing`              | Structured logging                                           |
| `indexmap`             | Ordered map for stable state iteration in `FlowState`        |
| `either`               | `Either<A, B>` type used in `FlowBuilder::either`            |
| `futures`              | `BoxFuture` for erased async function pointers in `WorkInfo` |
| `genai`                | Optional GenAI provider adapter                              |
| `gemini-rust`          | Optional Gemini provider client                              |
