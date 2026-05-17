# Pravah

[![Crates.io](https://img.shields.io/crates/v/pravah)](https://crates.io/crates/pravah)
[![docs.rs](https://img.shields.io/docsrs/pravah)](https://docs.rs/pravah)
[![License](https://img.shields.io/crates/l/pravah)](LICENSE-MIT)

_Pravah_ (प्रवाह, _pruh-VAH_) — Sanskrit/Hindi for "flow" or "current".

Pravah is a Rust library for building **stepwise, resumable, transactional
data-flow systems**.

It is designed for workflows where execution must remain:

- typed
- inspectable
- replayable
- resumable
- deterministic
- portable across processes and machines

Agentic systems are one application of this model — not the model itself.

## Installation

```toml
[dependencies]
pravah = "0.3.5"
```

To enable only selected providers:

```toml
pravah = { version = "0.3.5", default-features = false, features = ["provider-openai"] }
```

Available provider features: `provider-openai`, `provider-anthropic`,
`provider-gemini`, `provider-ollama`, `provider-genai`. All are enabled by
default.

## The Core Idea

A Pravah flow advances **one bounded step at a time**.

One call to `next()` performs exactly one unit of work:

- one LLM turn
- one tool batch
- one deterministic transform
- one branch
- one merge
- one suspend point
- one nested flow step

After every step, the entire runtime can be:

- snapshotted
- persisted
- transferred
- retried
- suspended
- resumed elsewhere

Nothing is hidden inside closures, async tasks, or thread-local state.

The only things required to continue execution are:

- the `FlowSnapshot`
- the flow graph definition

Both are owned by you.

## Why Pravah Exists

Most workflow and agent frameworks optimize for:

- convenience
- implicit execution
- parallel scheduling
- opaque orchestration

Pravah optimizes for something different:

> explicit information movement through deterministic execution steps

That changes the entire runtime model.

A Pravah flow behaves more like a resumable interpreter than a background task
runner.

This makes it particularly good for:

- long-running AI systems
- human-in-the-loop pipelines
- transactional orchestration
- approval workflows
- resumable execution
- durable conversations
- replay/debugging
- nested agent systems
- state-machine-like applications

## Mental Model

A flow graph is made of typed nodes.

Each node consumes one Rust type and produces another.

```text
Input
  ↓
Agent
  ↓
Split
 ↙   ↘
A     B
 ↘   ↙
 Merge
   ↓
Suspend
   ↓
Resume
   ↓
Done
```

The runtime stores typed values internally as serializable JSON state.

Execution progresses by consuming one state value at a time.

## The Most Important Rule

Within a single flow graph:

> one Rust type can identify only one node

If `PlanInput` exists in state, there must be exactly one node capable of
consuming it.

This guarantees:

- deterministic routing
- resumable execution
- checkpoint safety
- unambiguous replay
- graph validation before runtime

The builder rejects ambiguous graphs automatically.

## Tiny Example

A two-step flow:

1. an agent generates bullet points
2. a work node formats them into a report

```rust
use pravah::flows::{
    Agent, AgentConfig, Flow, FlowError,
    FlowGraph, FlowRuntime, FlowStep,
};

use pravah::{Context, FlowConf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct SummariseRequest { topic: String }

#[derive(Serialize, Deserialize, JsonSchema)]
struct BulletPoints { points: Vec<String> }

#[derive(Serialize, Deserialize, JsonSchema)]
struct Report { text: String }

impl Agent for SummariseRequest {
    type Output = BulletPoints;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "Summarise the topic into concise bullet points.",
            "openai://gpt-4o-mini",
        )
    }
}

async fn format_report(points: BulletPoints, _ctx: Context) -> Result<Report, FlowError> {
    let text = points.points.iter().map(|p| format!("• {p}")).collect::<Vec<_>>().join("\n");
    Ok(Report { text })
}

impl Flow for SummariseRequest {
    type Output = Report;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<SummariseRequest>()
            .work(format_report)
            .build()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::default();
    let input = SummariseRequest { topic: "Rust ownership".into() };
    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(v) => {
                let report: Report = serde_json::from_value(v)?;
                println!("{}", report.text);
                break;
            }
            FlowStep::Suspend(_) => {
                eprintln!("unexpected suspension");
                break;
            }
        }
    }

    Ok(())
}
```

See the [`examples/`](examples/) directory for runnable examples.

## Tracing

Pravah emits `tracing` events for runtime steps, LLM dispatches, tool calls,
retries, rate limiting, suspension, and run limits.

Add a subscriber in your application:

```toml
[dependencies]
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

```rust
use tracing_subscriber::{EnvFilter, fmt};

fn init_tracing() {
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,pravah=debug")),
        )
        .try_init();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let ctx = Context::default();
    let input = SummariseRequest { topic: "Rust ownership".into() };
    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(v) => {
                let report: Report = serde_json::from_value(v)?;
                println!("{}", report.text);
                break;
            }
            FlowStep::Suspend(_) => break,
        }
    }

    Ok(())
}
```

Set `RUST_LOG=pravah=trace` when you need the full engine trace.

## Execution Model

Pravah is intentionally:

- stepwise
- single-runtime
- deterministic

Parallelism is modeled explicitly through graph structure rather than hidden
inside the scheduler.

A `split()` represents independent information branches.
A `merge()` represents synchronization.

The runtime itself stays predictable and replayable.

## Node Types

| Builder method      | What it does                                                              |
| ------------------- | ------------------------------------------------------------------------- |
| `agent::<A>()`      | LLM-backed node; structured output or tool loop                           |
| `work(f)`           | Effectful async transform; `async fn(I, Context) -> Result<O, FlowError>` |
| `map(f)`            | Pure sync transform; `fn(I) -> O`, infallible, no `Context`               |
| `either(f)`         | Routes to one branch; `fn(I) -> Either<A, B>`, infallible (cycles ok)     |
| `split(f)`          | Fans out to N branches; `fn(I) -> (A, B, ...)`, infallible                |
| `merge(f)`          | Collects N branches once all ready; `fn((A, B, ...)) -> O`, infallible    |
| `suspend::<I, O>()` | Pauses the flow; caller resumes with a value of type `O`                  |
| `flow::<F>()`       | Embeds another `Flow` as a node                                           |

`split` and `merge` support arities 2–16. `fork`/`join` are binary-only
aliases for `split`/`merge`.

## Pure vs Effectful Nodes

Pravah distinguishes between two categories of nodes.

### Pure algebra nodes

`map`, `either`, `split`, `merge` (and their `fork`/`join` aliases) cannot
fail and cannot perform effects. Their handlers are plain functions with no
`Context` argument and no `Result` wrapper:

```rust
fn(I) -> O
```

### Effectful nodes

`work` and `agent` may perform I/O or external interaction. They are async
and fallible:

```rust
async fn(I, Context) -> Result<O, FlowError>
```

This separation keeps graph semantics explicit and keeps pure routing logic
free of error-handling noise.

## Suspend And Resume

There are two ways a flow can suspend.

**Flow-level suspend** — `suspend::<I, O>()` registers a first-class suspend
node. When a value of type `I` arrives in state the flow pauses immediately.
Resume by supplying a value of type `O`:

```rust
builder.suspend::<ApprovalRequest, ApprovalDecision>()
```

**Tool-level suspend** — register a suspend point via `ToolBox::suspend::<T, Out>()`. The LLM
calls the tool with a value of type `T`; the flow pauses and surfaces a `SuspendedValue`
wrapping that `T` as `FlowStep::Suspend`. Useful for approval gates or missing
credentials needed mid-agent-turn.

In both cases the caller's loop looks the same:

```rust
loop {
    match runtime.next(ctx.clone()).await? {
        FlowStep::Continue => {}
        FlowStep::Done(v) => break,
        FlowStep::Suspend(sv) => {
            // do out-of-band work, then:
            runtime.resume(ctx.clone(), decision).await?;
        }
    }
}
```

This makes Pravah suitable for workflows that span minutes, hours, days,
machines, or processes without losing execution state.

## Nested Flows

A flow has the same shape as a node — typed input, stepwise execution, typed
output — so flows compose naturally:

```rust
FlowGraph::builder()
    .flow::<PlannerFlow>()
    .flow::<ResearchFlow>()
    .flow::<ReviewFlow>()
    .build()
```

Nested flows inherit the same guarantees: deterministic execution,
resumability, snapshot safety, and typed boundaries.

## Agents

Agents are typed LLM nodes defined by implementing `Agent`:

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

Model URLs follow the pattern `provider://model-id`. Examples:
`openai://gpt-4o`, `anthropic://claude-opus-4-5`,
`gemini://gemini-2.5-flash-lite`, `ollama://localhost:11434/llama3`.

To attach tools, call `.with_tools()`:

```rust
fn build() -> AgentConfig {
    AgentConfig::new("You are a careful planning agent.", "gemini://gemini-2.5-flash-lite")
        .with_tools(ToolBox::new().tool::<ReadNote>())
}
```

With no tools, Pravah uses structured-output mode. With tools, it injects a
typed exit sentinel so the model can submit the final value.

Structured-output behavior is provider-specific:

- **OpenAI**: native JSON Schema mode
- **Gemini**: native JSON Schema mode (schema is sanitized for provider compatibility)
- **Ollama**: native JSON Schema mode when `output_schema` is present; falls back to
  generic JSON object mode otherwise
- **Anthropic**: schema is provided as a strict prompt contract (best effort)
- **GenAI**: JSON schema response format via the `genai` adapter

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

## Persistence

Call `runtime.snapshot()` to capture the entire execution state. Restore
later using `FlowRuntime::from_snapshot(snapshot)`.

Snapshots contain: runtime state, pending branches, suspend points, nested
flow state, and execution progress. They do not contain closures, async tasks,
thread-local state, or executor-specific handles. This keeps snapshots
portable and durable.

## History Management

LLM history is intentionally separate from runtime execution state, allowing
different persistence policies, external storage, compaction strategies, and
summarization pipelines.

Pravah includes sliding window compaction, custom compactor hooks, and
pluggable history stores.

## Examples

| Example        | Description                        |
| -------------- | ---------------------------------- |
| `linear_flow`  | Simple agent → transform pipeline  |
| `split_merge`  | Multi-branch fan-out/fan-in        |
| `nested_flow`  | Flow composition                   |
| `debate`       | Multi-agent debate and judgement   |
| `snapshot`     | Serialize and restore execution    |
| `story`        | Interactive looping narrative flow |
| `human_input`  | Human-in-the-loop suspension       |
| `gen_diagrams` | Graph visualization and rendering  |

## When To Use Pravah

Use Pravah when you need explicit information movement, deterministic
orchestration, resumable execution, typed flow boundaries, transactional
execution steps, inspectable runtime state, nested agent systems, or
long-running interactive workflows.

## When Not To Use Pravah

Pravah is not a distributed scheduler, queue system, parallel compute engine,
durable storage system, background task runner, or Kubernetes replacement. It
can sit inside those systems, but does not attempt to replace them.

## Design Philosophy

Pravah intentionally prefers explicit state, explicit transitions, explicit
suspension, explicit typing, and explicit execution boundaries over hidden
orchestration.

The runtime should always be understandable from the flow graph, the runtime
state, and the snapshot — without requiring implicit runtime knowledge.

## License

Licensed under either:

- MIT license
- Apache License 2.0

at your option.
