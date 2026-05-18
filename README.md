# Pravah

[![Crates.io](https://img.shields.io/crates/v/pravah)](https://crates.io/crates/pravah)
[![docs.rs](https://img.shields.io/docsrs/pravah)](https://docs.rs/pravah)
[![License](https://img.shields.io/crates/l/pravah)](LICENSE-MIT)

_Pravah_ (प्रवाह, _pruh-VAH_) means "flow" or "current".

Pravah is a Rust library for building stepwise, resumable, typed flows.
It is designed for workflows that need to stay inspectable, deterministic,
portable across processes, and safe to pause or resume.

Agentic systems are one use case. The underlying model is broader: explicit
information movement through a typed graph, one bounded step at a time.

## Why Pravah

Most workflow and agent frameworks optimize for convenience through implicit
orchestration. Pravah optimizes for explicit execution.

One call to `next()` performs one unit of work:

- one LLM turn
- one tool batch
- one deterministic transform
- one branch or merge
- one suspend point
- one nested flow step

After each step, the runtime can be snapshotted, stored, transferred, retried,
or resumed elsewhere. Nothing important is hidden inside background tasks or
thread-local state.

Pravah is a good fit when you need:

- long-running AI workflows
- human-in-the-loop approvals
- resumable execution
- replayable state transitions
- typed composition across subflows
- deterministic orchestration

## Mental Model

A flow graph is a typed pipeline. Each node consumes one Rust type and produces
another.

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

The runtime stores typed values as serializable state and advances by consuming
one value at a time.

The key rule is simple:

> within one flow graph, one Rust type can identify only one node

That gives you deterministic routing, safe replay, and unambiguous resumption.

## Installation

```toml
[dependencies]
pravah = "0.3.6"
```

To enable only selected providers:

```toml
pravah = { version = "0.3.6", default-features = false, features = ["provider-openai"] }
```

Available provider features: `provider-openai`, `provider-anthropic`,
`provider-gemini`, `provider-ollama`.

## Getting Started

The smallest useful Pravah flow is often:

1. one typed agent node
2. one deterministic transform

See [examples/linear_flow.rs](examples/linear_flow.rs) for a complete runnable
example.

```rust
use pravah::flows::{Agent, AgentConfig, Flow, FlowError, FlowGraph};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct SummariseRequest {
    text: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct BulletPoints {
    points: Vec<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct Report {
    markdown: String,
}

impl Agent for SummariseRequest {
    type Output = BulletPoints;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a concise summariser.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

async fn format_bullets(bullets: BulletPoints, _ctx: pravah::Context) -> Result<Report, FlowError> {
    Ok(Report {
        markdown: bullets.points.join("\n"),
    })
}

impl Flow for SummariseRequest {
    type Output = Report;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<SummariseRequest>()
            .work(format_bullets)
            .build()
    }
}
```

## Read Next

- [docs/clients.md](docs/clients.md): agents, model URLs, providers, tools, attachments, client layers
- [docs/flows.md](docs/flows.md): node types, execution model, suspend/resume, snapshots, history
- [examples/](examples/): runnable examples
- [docs.rs](https://docs.rs/pravah): API reference

## Examples

| Example                                              | What it shows                          |
| ---------------------------------------------------- | -------------------------------------- |
| [examples/linear_flow.rs](examples/linear_flow.rs)   | Minimal agent -> work pipeline         |
| [examples/split_merge.rs](examples/split_merge.rs)   | Fan-out / fan-in flow composition      |
| [examples/nested_flow.rs](examples/nested_flow.rs)   | Nested flows as reusable nodes         |
| [examples/human_input.rs](examples/human_input.rs)   | Suspend and resume with external input |
| [examples/snapshot.rs](examples/snapshot.rs)         | Serialize and restore runtime state    |
| [examples/image_prompt.rs](examples/image_prompt.rs) | Initial user message with image upload |
| [examples/debate.rs](examples/debate.rs)             | Multi-agent reasoning across branches  |
| [examples/story.rs](examples/story.rs)               | Interactive looping flow               |
| [examples/gen_diagrams.rs](examples/gen_diagrams.rs) | Flow visualization                     |

## When To Use Pravah

Use Pravah when you need explicit execution boundaries, durable state, typed
transitions, resumable workflows, or agentic systems that must remain
inspectable over time.

Do not use Pravah as a queue system, distributed scheduler, background task
runner, or durable storage layer. It can sit inside those systems, but it does
not replace them.

## License

Licensed under either:

- MIT license
- Apache License 2.0

at your option.
