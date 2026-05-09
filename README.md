# Pravah

[![Crates.io](https://img.shields.io/crates/v/pravah)](https://crates.io/crates/pravah)
[![docs.rs](https://img.shields.io/docsrs/pravah)](https://docs.rs/pravah)
[![License](https://img.shields.io/crates/l/pravah)](LICENSE-MIT)

_Pravah_ (प्रवाह, _pruh-VAH_) — Sanskrit/Hindi for "flow" or "current".

A Rust library for building typed, stepwise agentic information flows.

Each call to `next()` does one bounded unit of work — one LLM turn, one tool
batch, one deterministic transform, one branch, one fork, or one join.

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
pravah = "0.1"
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
pravah = { version = "0.1", default-features = false, features = ["provider-openai"] }
```

Model URLs select the backend at runtime: `openai://gpt-4o`,
`anthropic://claude-sonnet-4-5`, `gemini://gemini-2.5-flash-lite`,
`ollama://localhost:11434/qwen3:8b`. Inject a custom `ClientFactory` for
testing, recording/replay, or hosted gateways.

## Getting Started

A two-node flow: an agent produces bullet points, a work node formats them into a report.

```rust
use pravah::flows::{Agent, AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, RunOut};
use pravah::context::{Context, FlowConf};
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
async fn main() -> anyhow::Result<()> {
    let ctx = Context::new(FlowConf::default());
    let input = SummariseRequest { topic: "Rust ownership model".into() };
    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            RunOut::Continue => {}
            RunOut::Done(report) => { println!("{}", report.text); break; }
            RunOut::Suspend { .. } => unreachable!("no suspension tools registered"),
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

| Builder method           | What it does                                    |
| ------------------------ | ----------------------------------------------- |
| `agent::<A>()`           | LLM-backed node; structured output or tool loop |
| `work::<From, Out>()`    | Deterministic async transform                   |
| `either::<From, A, B>()` | Routes to one of two typed branches             |
| `fork::<From, A, B>()`   | Splits one value into two active branches       |
| `join::<A, B, Out>()`    | Combines two branches once both are ready       |

Fork and join model information shape, not parallelism — the runner is
single-threaded.

The builder validates: duplicate node identities, entry not in graph,
unreachable nodes, no path to a terminal value, invalid fork/join definitions,
and both branches of `either` routing to the same type.

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

## Tools

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use pravah::context::Context;
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

## Suspend And Resume

A tool returns `ToolError::suspend(value)` to pause the flow. The runtime
surfaces a suspension payload and a tool id. Persist state, show the request
to a user, wait for a webhook — then call `resume()` with the matching tool id
and a JSON response. Useful for approval gates, missing credentials, payments,
or any action needing external confirmation.

```rust
loop {
    match runtime.next(ctx.clone()).await? {
        RunOut::Continue => {}
        RunOut::Suspend { value, tool_id } => {
            // Persist state, collect external input, then:
            runtime.resume(ctx.clone(), (tool_id, json!({ "approved": true }))).await?;
        }
        RunOut::Done(output) => { println!("{}", output.text); break; }
    }
}
```

## Persistence

Call `runtime.snapshot()` to capture an opaque `FlowSnapshot`
(serializable, no closures). Restore it with `FlowRuntime::from_snapshot(snap)`.
Conversation history is managed separately — re-attach it with
`runtime.with_history(history)` after restoring. Pravah defines the
serializable state; it does not prescribe where snapshots live.

## Nested Flows

A flow has the same shape as a node: typed input, stepwise execution, typed
output. Use nested flows to keep large agent systems modular — a planning flow
can contain a research sub-flow, a coding flow can contain a review-and-fix
sub-flow. The same node-identity rule applies at each graph boundary.

### Example: Article Production Pipeline

Combines every node type — fork, join, work, either, agent, and two nested flows.
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
