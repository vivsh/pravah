# Pravah

[![Crates.io](https://img.shields.io/crates/v/pravah)](https://crates.io/crates/pravah)
[![docs.rs](https://img.shields.io/docsrs/pravah)](https://docs.rs/pravah)
[![License](https://img.shields.io/crates/l/pravah)](LICENSE-MIT)

_Pravah_ (प्रवाह, _pruh-VAH_) means "flow" or "current".

Pravah is a stepwise transactional flow engine for Rust. It builds typed
graphs that advance one bounded step at a time, keeping execution explicit,
inspectable, and resumable across agentic and non-agentic workflows.

## Why Pravah

Many workflow and agent frameworks hide orchestration behind implicit loops,
background tasks, or framework-owned state. Pravah does not.

One call to `next()` performs exactly one bounded step:

- one LLM interaction
- one tool batch
- one async `work` transform
- one branch transition
- one merge transition
- one suspend boundary
- one nested flow step
- one each-fanout dispatch (one item per step)

After each step you can inspect the runtime, snapshot it, store it, retry it,
or resume it elsewhere. Nothing important is trapped in thread-local state or
hidden inside a scheduler-owned control loop.

Pravah is a good fit when you need:

- long-running AI workflows
- human-in-the-loop approvals
- resumable execution
- replayable state transitions
- typed composition across subflows
- deterministic orchestration
- bounded tool-using agent loops
- multi-turn agent conversations with persistent history
- sequential fan-out over typed collections (`each` node)

Runtime control hints stay separate from persisted conversation state. For
example, a final-turn reminder from an agent turn budget is injected only into
the current model request, not written back into history.

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

The core invariant is simple:

> within one flow graph, one Rust type can identify only one node

That keeps routing deterministic and resumption unambiguous.

## Installation

```toml
[dependencies]
pravah = "0.4.1"
```

To opt into providers explicitly:

```toml
[dependencies]
pravah = { version = "0.4.1", default-features = false, features = [
  "provider-openai",
  "provider-anthropic",
  "provider-gemini",
  "provider-ollama",
] }
```

Remove the providers you do not need.

## Getting Started

The smallest useful Pravah flow is usually one agent followed by one
deterministic transform. See [examples/linear_flow.rs](examples/linear_flow.rs)
for the full runnable file.

```rust
use pravah::flows::{Agent, AgentConfig, Flow, FlowBuilder, FlowError, FlowRuntime, FlowStep, Node};
use pravah::{Context, FlowConf};
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

  fn configure() -> AgentConfig {
    AgentConfig::new(
      "You are a concise summariser.",
      "gemini:///gemini-2.5-flash-lite",
    )
  }
}

async fn format_bullets(bullets: BulletPoints, _ctx: Context) -> Result<Report, FlowError> {
  Ok(Report {
    markdown: bullets
      .points
      .into_iter()
      .map(|point| format!("- {point}"))
      .collect::<Vec<_>>()
      .join("\n"),
  })
}

impl Flow for SummariseRequest {
  type Output = Report;

  fn define(root: Node<Self>) -> FlowBuilder {
    root.agent().work(format_bullets).finalize()
  }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  let ctx = Context::new(FlowConf::default());
  let mut runtime = FlowRuntime::new(SummariseRequest {
    text: "Rust is fast, memory-safe, and concurrency-friendly.".into(),
  })?;

  loop {
    match runtime.next(ctx.clone()).await? {
      FlowStep::Continue => {}
      FlowStep::Done(report) => {
        println!("{}", report.markdown);
        break;
      }
      FlowStep::Suspend(_) => unreachable!("this flow never suspends"),
    }
  }

  Ok(())
}
```

For a simple persistent conversation without graph construction, use
[`Chat`](docs/chat.md) instead. See
[examples/chat.rs](examples/chat.rs) for a minimal example.

## Read Next

- [docs/chat.md](docs/chat.md): simple multi-turn chat without a flow graph
- [docs/clients.md](docs/clients.md): agents, providers, model URLs, tools, attachments
- [docs/flows.md](docs/flows.md): node types, runtime semantics, suspension, snapshots
- [examples/](examples/): runnable end-to-end flows
- [docs.rs](https://docs.rs/pravah): API reference

## Examples

| Example                                                | What it shows                            |
| ------------------------------------------------------ | ---------------------------------------- |
| [examples/chat.rs](examples/chat.rs)                   | Single-session chat with history         |
| [examples/linear_flow.rs](examples/linear_flow.rs)     | Minimal agent -> work pipeline           |
| [examples/split_merge.rs](examples/split_merge.rs)     | Fan-out and fan-in composition           |
| [examples/nested_flow.rs](examples/nested_flow.rs)     | Embedded subflows as reusable nodes      |
| [examples/snapshot.rs](examples/snapshot.rs)           | Save and restore runtime state           |
| [examples/image_prompt.rs](examples/image_prompt.rs)   | Initial user message with an image       |
| [examples/ollama_client.rs](examples/ollama_client.rs) | Direct provider client usage             |
| [examples/debate.rs](examples/debate.rs)               | Multi-agent branching workflow           |
| [examples/story.rs](examples/story.rs)                 | Looping flow with repeated agent turns   |
| [examples/gen_diagrams.rs](examples/gen_diagrams.rs)   | Tree, Mermaid, and DOT graph output      |
| [examples/each_node.rs](examples/each_node.rs)         | Fan-out over a list with the `each` node |
| [examples/tool_flow.rs](examples/tool_flow.rs)         | Sub-flow registered as an agent tool     |

## When To Use Pravah

### Use Pravah When

Use Pravah when you need explicit execution boundaries, typed transitions,
human approvals, snapshots, or resumable workflows that must remain
inspectable over time.

### Do Not Use Pravah When

Do not use Pravah as a queue system, distributed scheduler, background task
runner, or durable storage layer. It can sit inside those systems, but it does
not replace them.

## License

Licensed under either the MIT License or Apache License 2.0, at your option.
