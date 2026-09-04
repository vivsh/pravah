# Pravah

[![Crates.io](https://img.shields.io/crates/v/pravah)](https://crates.io/crates/pravah)
[![docs.rs](https://img.shields.io/docsrs/pravah)](https://docs.rs/pravah)
[![License](https://img.shields.io/crates/l/pravah)](LICENSE-MIT)

_Pravah_ (प्रवाह, _pruh-VAH_) means “flow” or “current”.

Pravah is a workflow engine for Rust applications that need to advance work one
bounded step at a time. Workflows can be inspected, snapshotted, stored, and
resumed without hiding their progress inside a scheduler or background loop.

It supports ordinary application workflows as well as workflows involving AI,
tools, and people.

## Why Pravah

Pravah is useful when a workflow must be:

- explicit and stepwise;
- typed when authored in Rust;
- serializable when stored or invoked through JSON;
- resumable across process boundaries;
- composed from branches, subflows, loops, tools, or human input.

Pravah is not a queue, database, distributed scheduler, or application server.
Applications provide those facilities and decide when to call the runtime.

## Installation

```toml
[dependencies]
pravah = "0.4.10"
```

The repository contains the `pravah` crate for explicit, resumable agent and
application workflows.

## A Small Typed Workflow

```rust
use pravah::graph::{self, GraphError, Flow, Step};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Request {
    value: i64,
}

fn double(root: Flow<Request>) -> Flow<i64> {
    root.map_named("double", |request| request.value * 2)
}

async fn run() -> Result<i64, GraphError> {
    let flow = graph::compile(double)?;
    let mut runtime = flow.runtime(Request { value: 21 })?;
    let ctx = Context::new(FlowConf::default());

    loop {
        match runtime.next(ctx.clone()).await? {
            Step::Continue => {}
            Step::Done(value) => return flow.decode_output(value),
            Step::Suspend(_) => {
                return Err(GraphError::Invalid(
                    "this workflow does not suspend".into(),
                ));
            }
        }
    }
}
```

Each `next()` call advances at most one workflow instruction. The application
remains in control of persistence, retries, scheduling, and external effects.

## Ways To Use Pravah

Pravah provides three complementary entry points:

- `pravah::graph::UntypedGraph` is the serializable core.
- `pravah::graph::compile` and `Flow` provide typed Rust authoring with ordinary
  functions.
- `pravah::graph::JsonInvoker` provides trusted, transport-neutral JSON
  start, next, and resume operations.

All three use the same stepwise runtime. Snapshots contain execution state and
history tied to a graph fingerprint. Store the serializable graph separately,
and supply runtime services again when restoring.

## Selected Examples

| Example | What it shows |
| --- | --- |
| [`graph_typed_composition`](examples/graph_typed_composition.rs) | Reusing one typed subflow twice |
| [`graph_snapshot_resume`](examples/graph_snapshot_resume.rs) | Suspend, serialize, restore, and resume |
| [`graph_json_invocation`](examples/graph_json_invocation.rs) | Stateless JSON invocation through completion |
| [`graph_typed`](examples/graph_typed.rs) | Typed maps, branches, and collections |
| [`graph_untyped`](examples/graph_untyped.rs) | Building the untyped graph directly |
| [`graph_agent_control`](examples/graph_agent_control.rs) | Adaptive tool visibility, conclusion, suspension, and resume |
| [`graph_diagram_complex`](examples/graph_diagram_complex.rs) | Tree, Mermaid, and DOT diagrams |
| [`story`](examples/story.rs) | A larger graph-backed agent workflow |

Older examples using `pravah::legacy` demonstrate the legacy API. New
workflow features target `pravah::graph`. See the [complete example
index](examples/README.md) for prerequisites and commands for every example.

## Read Next

- [`docs/graph.md`](docs/graph.md): execution, snapshots, JSON invocation, and
  effect-safety responsibilities
- [`docs/chat.md`](docs/chat.md): simple multi-turn chat
- [`docs/clients.md`](docs/clients.md): providers, models, tools, and attachments
- [`docs/mcp.md`](docs/mcp.md): MCP text resources and dynamic agent tool filters
- [`docs/legacy.md`](docs/legacy.md): compatibility-only workflow API
- [docs.rs](https://docs.rs/pravah): complete API reference

## Versioning

The `0.4.x` line is still refining its API and serialized formats. Graph,
snapshot, agent-payload, and JSON-wire versions are checked explicitly;
incompatible data is rejected instead of guessed or silently upgraded.

## License

Licensed under either the MIT License or Apache License 2.0, at your option.
