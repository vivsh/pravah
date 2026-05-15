# Pravah

[![Crates.io](https://img.shields.io/crates/v/pravah)](https://crates.io/crates/pravah)
[![docs.rs](https://img.shields.io/docsrs/pravah)](https://docs.rs/pravah)
[![License](https://img.shields.io/crates/l/pravah)](LICENSE-MIT)

_Pravah_ (प्रवाह, _pruh-VAH_) — Sanskrit/Hindi for "flow" or "current".

A Rust library for building **stepwise, transactional data-flow pipelines** with
first-class support for agentic programming.

Flows are typed graphs where every edge is a Rust type contract. Cycles are
supported — an `either` node can route back to any earlier type, enabling
retry loops, multi-turn conversations, and interactive pipelines within a
single `FlowRuntime`.

Each call to `next()` does one bounded unit of work — one LLM turn, one tool
batch, one deterministic transform, one branch, one fork, one join, or one
step of a nested flow.

> **Pravah executes flows one transaction-sized step at a time.**
> After every `next()` call, the entire flow state can be:
>
> - **persisted** — snapshot to a database, file, or message queue
> - **suspended** — pause at an approval gate or external event
> - **resumed** — restore the snapshot in any process and continue
> - **inspected** — examine typed state between steps for debugging or auditing
> - **retried** — replay from the last good snapshot on failure
> - **transferred** — hand the snapshot to a different machine, worker, or service
>
> Nothing is hidden in closures or thread-local state. The only things needed to
> continue a flow are the `FlowSnapshot` and the flow graph definition — both
> of which you own.

## Installation

```toml
[dependencies]
pravah = "0.3"
```

| Feature              | Default | Description                                                             |
| -------------------- | :-----: | ----------------------------------------------------------------------- |
| `provider-openai`    |    ✓    | OpenAI-compatible API client                                            |
| `provider-anthropic` |    ✓    | Anthropic Claude API client                                             |
| `provider-gemini`    |    ✓    | Google Gemini API client                                                |
| `provider-ollama`    |    ✓    | Ollama local model client                                               |
| `provider-genai`     |    —    | Extra providers via the [`genai`](https://crates.io/crates/genai) crate |

To use only specific providers, disable defaults:

```toml
pravah = { version = "0.3", default-features = false, features = ["provider-openai"] }
```

Model URLs select the backend at runtime: `openai://gpt-4o`,
`anthropic://claude-sonnet-4-5`, `gemini://gemini-2.5-flash-lite`,
`ollama://localhost:11434/qwen3:8b`. Inject a custom `ClientFactory` for
testing, recording/replay, or hosted gateways.

## Getting Started

A two-node flow: an agent produces bullet points, a work node formats them into a report.

```rust
use pravah::flows::{Agent, AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Types ──────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, JsonSchema)]
struct SummariseRequest { topic: String }

#[derive(Serialize, Deserialize, JsonSchema)]
struct BulletPoints { points: Vec<String> }

#[derive(Serialize, Deserialize, JsonSchema)]
struct Report { text: String }

// ── Agent ──────────────────────────────────────────────────────────────────

impl Agent for SummariseRequest {
    type Output = BulletPoints;

    fn build() -> AgentConfig {
        AgentConfig::new("Summarise the topic as concise bullet points.", "openai://gpt-4o-mini")
    }
}

// ── Work node ─────────────────────────────────────────────────────────────

async fn format_report(points: BulletPoints, _ctx: Context) -> Result<Report, FlowError> {
    let text = points.points.iter().map(|p| format!("• {p}")).collect::<Vec<_>>().join("\n");
    Ok(Report { text })
}

// ── Flow ───────────────────────────────────────────────────────────────────

impl Flow for SummariseRequest {
    type Output = Report;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<SummariseRequest>()
            .work(format_report)
            .build()
    }
}

// ── Run ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new(FlowConf::default());
    let input = SummariseRequest { topic: "Rust ownership model".into() };
    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(v) => {
                let report: Report = serde_json::from_value(v)?;
                println!("{}", report.text);
                break;
            }
            FlowStep::Suspend { value } => {
                eprintln!("Unexpected suspension: {value}");
                break;
            }
        }
    }
    Ok(())
}
```

See the [`examples/`](examples/) directory for runnable examples covering
linear flows, fork/join, nested flows, and snapshot-based resumption.

## Core Concepts

**Each input type identifies exactly one node within a flow graph.**

`PlanInput` can be the input for one agent, one work node, one branch, one
fork, or one join participant — never more than one. The builder rejects
duplicates. When a value of that type is present in flow state there is exactly
one node that can consume it, keeping routing unambiguous and state
checkpointable between steps.

Rust types are the contract at every boundary: input structs define the LLM
message shape, output types define the result schema, tool structs define
callable arguments. State is stored as JSON internally so it can be
serialized, but user code stays typed.

### Node Types

| Builder method      | What it does                                                              |
| ------------------- | ------------------------------------------------------------------------- |
| `agent::<A>()`      | LLM-backed node; structured output or tool loop                           |
| `work(f)`           | Effectful async transform; `async fn(I, Context) -> Result<O, FlowError>` |
| `map(f)`            | Pure sync transform; `fn(I) -> O`, infallible, no `Context`               |
| `either(f)`         | Routes to one branch; `fn(I) -> Either<A, B>`, infallible (cycles ok)     |
| `split(f)`          | Fans out to N branches; `fn(I) -> (A, B, ...)`, infallible                |
| `merge(f)`          | Collects N branches once all ready; `fn(A, B, ...) -> O`, infallible      |
| `suspend::<I, O>()` | Pauses the flow; caller resumes with a value of type `O`                  |
| `flow::<F>()`       | Embeds another `Flow` as a node                                           |

`split` and `merge` are the primary fan-out/fan-in primitives. `split` receives
one typed value and returns an N-tuple; each element becomes an independent branch
in the state map. `merge` receives an N-tuple (all branches must be present before
it fires) and produces one value. Both support arities 2–16, so a single `split`
replaces chains of binary forks, and a single `merge` replaces chains of binary
joins. Fan-out/fan-in models information shape, not parallelism — the runner is
single-threaded.

`fork`/`join` remain available as binary-only aliases for `split`/`merge`.

**Node purity.** `map`, `either`, `split`/`fork`, `merge`/`join` are pure
algebra nodes: their handlers are plain `fn(I) -> O` with no `Context` argument
and no `Result` wrapper. Effects and I/O belong in `work` or `agent` nodes.

The builder validates: duplicate node identities, entry not in graph,
unreachable nodes, no path to a terminal value, invalid split/merge definitions,
and both branches of `either` routing to the same type.

Runtime construction adds two output contract checks:

- the graph must resolve to exactly one distinct terminal state id
- that terminal id must match `Flow::Output`

## Agents

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use pravah::flows::{Agent, AgentConfig};

#[derive(Serialize, Deserialize, JsonSchema)]
struct PlannerInput { goal: String }

#[derive(Serialize, Deserialize, JsonSchema)]
struct Plan { steps: Vec<String> }

impl Agent for PlannerInput {
    type Output = Plan;

    fn build() -> AgentConfig {
        AgentConfig::new("You are a careful planning agent.", "gemini://gemini-2.5-flash-lite")
    }
}
```

To attach tools, call `.with_tools()`:

```rust
fn build() -> AgentConfig {
    AgentConfig::new("You are a careful planning agent.", "gemini://gemini-2.5-flash-lite")
        .with_tools(ToolBox::builder().tool::<ReadNote>().build())
}
```

With no tools, Pravah uses structured-output mode. With tools, it injects a
typed exit sentinel so the model can submit the final value.

Structured-output behavior is provider-specific:

- OpenAI: native JSON Schema mode
- Gemini: native JSON Schema mode (schema is sanitized for provider compatibility)
- Ollama: native JSON Schema mode when `output_schema` is present; falls back to
  generic JSON object mode otherwise
- Anthropic: schema is provided as a strict prompt contract (best effort)
- GenAI: JSON schema response format via the `genai` adapter

## Tools

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use pravah::Context;
use pravah::tools::{Tool, ToolError};

#[derive(Deserialize, JsonSchema)]
struct ReadNote { path: String }

#[derive(Serialize)]
struct ReadNoteOutput { content: String }

impl Tool for ReadNote {
    type Output = ReadNoteOutput;
    fn name() -> &'static str { "read_note" }
    fn description() -> &'static str { "Read a note from the working directory." }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        let content = tokio::fs::read_to_string(ctx.resolve(&self.path)?).await?;
        Ok(ReadNoteOutput { content })
    }
}
```

`Context` carries the working directory, command allowlist, dependency
container, and shared HTTP client.

`Context::resolve` enforces path confinement: traversal outside `working_dir`
is rejected, and symlinks are allowed only when their resolved target stays
within `working_dir`.

## Suspend And Resume

There are two ways a flow can suspend:

**Tool-level suspend** — a tool returns `Err(ToolError::Suspend)`. The runtime
surfaces the tool's input args as the `SuspendedValue`. Useful for approval
gates, missing credentials, or any external confirmation needed mid-agent-turn.

**Flow-level suspend** — `builder::suspend::<I, O>()` registers a first-class
suspend node. When a value of type `I` arrives in state the flow pauses
immediately (no agent or tool needed). Resume by supplying a value of type `O`.
Useful for human-in-the-loop steps, webhook callbacks, or any out-of-band
computation that should be a named node in the graph.

In both cases the caller's loop looks the same:

```rust
use serde_json::json;

loop {
    match runtime.next(ctx.clone()).await? {
        FlowStep::Continue => {}
        FlowStep::Suspend(sv) => {
            // Downcast sv to retrieve the suspended value, do out-of-band work,
            // then supply the result:
            runtime.resume(ctx.clone(), json!({ "approved": true })).await?;
        }
        FlowStep::Done(v) => { println!("{v}"); break; }
    }
}
```

## Persistence

Call `runtime.snapshot()` to capture an opaque `FlowSnapshot`
(serializable, no closures). Restore it with `FlowRuntime::from_snapshot(snap)`.
Conversation history is separate from the execution snapshot — save it via a
`HistoryStore` (see [History Management](#history-management)) and re-attach
with `runtime.with_history(history)` after restoring. Pravah defines the
serializable state; it does not prescribe where snapshots live.

## History Management

Every LLM turn is stored in a `FlowHistory` that the runtime owns. History is
kept separate from execution state so you can persist them independently and
restore them on a different machine or process.

### Compaction

By default (`NoopCompactor`) turns accumulate forever. Attach a
`SlidingWindowCompactor` to cap how many turns are kept per session:

```rust
use pravah::flows::{FlowRuntime, SlidingWindowCompactor};

let mut runtime = FlowRuntime::new(input)?
    .with_compactor(SlidingWindowCompactor { max_turns_per_session: 10 });
```

After every `next()` / `resume()` call the runtime runs compaction per active
session, then calls `HistoryStore::flush`. A
`Role::AssistantToolCalls` message and all its matching tool results count as
one turn; incomplete turns are never evicted.

Implement `HistoryCompactor` to supply a custom strategy (summarisation,
importance scoring, etc.):

```rust
use pravah::flows::{CompactionResult, HistoryCompactor, HistoryEntry};

struct SummarisationCompactor;

impl HistoryCompactor for SummarisationCompactor {
    async fn compact(&self, session_id: &str, entries: &[&HistoryEntry]) -> CompactionResult {
        // Decide what to evict and optionally return a summary Message.
        CompactionResult { evict_indices: vec![], summary: None }
    }
}
```

### Store

Implement `HistoryStore` to persist turns to a database, object storage, or
any backend:

```rust
use pravah::flows::{FlowHistory, HistoryEntry, HistoryStore};

struct MyStore;

impl HistoryStore for MyStore {
    type Error = std::io::Error;

    async fn flush(&self, history: &mut FlowHistory) -> Result<(), Self::Error> {
        for entry in history.entries() {
            if entry.evicted {
                // delete by entry.id
            } else {
                // upsert by entry.position
            }
        }
        history.prune_evicted(); // free evicted entries from memory
        Ok(())
    }

    async fn load(&self, session_ids: &[&str]) -> Result<Vec<HistoryEntry>, Self::Error> {
        // restore from DB
        Ok(vec![])
    }
}

let mut runtime = FlowRuntime::new(input)?
    .with_compactor(SlidingWindowCompactor { max_turns_per_session: 10 })
    .with_store(MyStore);
```

The default `NoopHistoryStore` calls `prune_evicted()` immediately so evicted
entries do not accumulate in memory. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the full snapshot vs. history separation model.

## Nested Flows

A flow has the same shape as a node: typed input, stepwise execution, typed
output. Use nested flows to keep large agent systems modular — a planning flow
can contain a research sub-flow, a coding flow can contain a review-and-fix
sub-flow. The same node-identity rule applies at each graph boundary.

### Example: Article Production Pipeline

Combines every node type — split, merge, work, either, agent, and two nested flows.
The tree below is the output of `FlowGraphDiagram::for_flow::<ArticleRequest>()?.render_tree()`:

```text
● ArticleRequest (fork)
  ├── [fork] AudienceTask (agent)
  │   └── [agent] AudienceProfile (join)
  │       └── [join] ContentBrief (work)
  │           └── [work] OutlineRequest (work)
  │               └── [work] Outline (either)
  │                   ├── [either] LongDraft (work)
  │                   │   └── [work] ReviewedDraft (agent)
  │                   │       └── [agent] FinalArticle ◉
  │                   └── [either] QuickDraft (agent)
  │                       └── [agent] FinalArticle ◉ ↩
  └── [fork] ResearchTask (agent)
      └── [agent] ResearchNotes (join)
          └── [join] ContentBrief (work) ↩
```

`↩` marks nodes that converge from multiple branches (already shown above).

![Article production pipeline](assets/nested_flow.svg)

## When To Use Pravah

**Use it** when you want agentic flows that are type-directed, inspectable,
resumable, testable with fake clients, and explicit about information movement.

**Don't use it** as a distributed workflow engine, parallel job scheduler,
queue processor, or durable storage system. Pravah can sit inside those
systems but does not try to replace them.

For a deeper look at module structure, the history/compaction model, and
extension points see [ARCHITECTURE.md](ARCHITECTURE.md).
