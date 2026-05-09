# Contributing to Pravah

---

## Getting Started

```bash
git clone https://github.com/vivsh/pravah
cd pravah
cargo test          # all tests must pass before you begin
```

No environment variables are required for the pure-Rust test suite.
Integration tests that hit live providers are gated with `#[ignore]` — skip them unless you're working on a specific provider.

---

## Coding Guidelines

These apply to all `.rs` files.

### Errors

- Use `thiserror`-derived structured errors. Never use `Box<dyn std::error::Error>` as a return type.
- Never panic. Use `Result` and `Option` to propagate failures; avoid `unwrap()`, `expect()`, and `unreachable!()` in non-test code.

### Async

- Do not call blocking code inside async functions. Use `spawn_blocking` for CPU-bound or blocking I/O work.
- `tokio` features are narrowed to `["rt-multi-thread", "macros", "fs", "process"]`. Do not add new sub-modules without updating `Cargo.toml`.

### Object-safe traits

- `Client` is a `dyn`-compatible trait. Until Rust stabilises `async fn` in object-safe traits, `#[async_trait]` is required on every `impl Client`. Do not remove it.

### Types and schema

- All node input/output types must derive `Serialize`, `Deserialize`, and `JsonSchema`.
- The node key in the graph equals the schemars schema name, which equals the unqualified Rust type name. Input and output types must have distinct names — a collision produces `FlowError::Invalid` at build time.

### Functions and files

- Keep functions under 50 lines. If a function grows beyond that, break it into well-named helpers.
- Keep source files under 1000 lines. If a module grows larger, split it into a folder module where `mod.rs` contains only `mod` declarations and re-exports.

### Memory

- Avoid unnecessary cloning. Prefer references and borrowing.

### Comments and docs

- Doc comments (`///`) should be terse and written as if by a human, not auto-generated prose.
- Do not add comments that exist only to visually separate sections of code.
- Every test function must have a doc comment that describes what it verifies.

### Feature flags

- New optional dependencies must be gated behind a feature flag. Add the flag to `[features]` in `Cargo.toml` and document it in the feature table in `AGENTS.md`.

---

## Testing

- Unit tests live in a `#[cfg(test)] mod tests` block in the same source file.
- Flow integration tests go in `tests/flows.rs` using the mock `ClientFactory`.
- Provider / network tests go in `tests/integration.rs` and must be gated with `#[ignore]`.
- Use `#[tokio::test]` for async tests; plain `#[test]` for sync tests.
- Unix-specific tests (e.g. symlink sandbox) use `#[cfg(unix)]`.
- All tests must pass with `cargo test` before opening a pull request.

---

## Pull Request Checklist

- [ ] `cargo test` passes with no failures or new warnings.
- [ ] `cargo check` passes with no new warnings.
- [ ] `cargo build --examples` passes if any example was added or modified.
- [ ] New public items have a doc comment.
- [ ] New feature flags are documented in the feature table in `AGENTS.md`.
- [ ] Structural changes to the flow graph or trait contracts are reflected in `ARCHITECTURE.md`.
- [ ] Operational changes (new commands, new modules, new gotchas) are reflected in `AGENTS.md`.

---

## Documentation File Guidelines

### `ARCHITECTURE.md`

- **Purpose**: LLM-oriented specification. Its goal is to prevent hallucinations when generating or reviewing code against this codebase.
- **Audience**: LLMs and developers who need exact contracts before writing code.
- **Rules**:
  - Every public trait must appear with its exact Rust signature, including all bounds and lifetimes.
  - Every required derive macro must be listed explicitly.
  - Naming conventions that have non-obvious runtime consequences must be called out.
  - Builder methods must show the exact closure signature, not a paraphrase.
  - Error enum variants must be listed with their payload types.
  - Do not add prose explanations — prefer short code blocks and bullet points.
  - Do not describe intent; describe the contract.

### `AGENTS.md`

- **Purpose**: Operational guide for AI coding agents working in this repository.
- **Audience**: AI agents that have already read `ARCHITECTURE.md`.
- **Rules**:
  - Commands must be copy-pasteable verbatim.
  - The module map must stay in sync with `src/` — update it whenever a file is added, removed, or renamed.
  - "Where to Add Things" sections must list every file that needs to be touched, in the correct order.
  - "Gotchas" must be added whenever a non-obvious invariant exists that an LLM is likely to violate.
  - Do not duplicate information that is already in `ARCHITECTURE.md`.

### `CONTRIBUTING.md` (this file)

- **Purpose**: Human and AI contributor guide for process, coding standards, and documentation hygiene.
- **Rules**:
  - Coding guidelines here must stay consistent with `.github/instructions/rust.instructions.md`.
  - If a rule in `rust.instructions.md` changes, update the corresponding bullet here.
  - Do not duplicate exact API contracts — those belong in `ARCHITECTURE.md`.
  - Do not duplicate operational commands — those belong in `AGENTS.md`.

### `README.md`

- **Purpose**: Public-facing introduction — crate overview, quick-start example, feature list.
- **Rules**:
  - Keep the quick-start example compilable. Verify with `cargo build --examples` after any API change.
  - Do not document internal implementation details here.
  - Do not duplicate the full API — link to `ARCHITECTURE.md` for contracts.

### `.github/instructions/rust.instructions.md`

- **Purpose**: IDE and agent instruction file that applies coding rules to every `.rs` file.
- **Rules**:
  - This is the authoritative source for Rust coding style rules.
  - Any rule added here must also be summarised in the "Coding Guidelines" section of this file.
  - Use the `applyTo` front-matter to scope rules correctly.
