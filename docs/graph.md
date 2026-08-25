# Graph Workflows

`pravah::graph` is the primary workflow API. Typed Rust workflows, untyped
graphs, and JSON invocations all use the same runtime.

## Author With Functions

A flow is an ordinary function from one typed flow value to another:

```rust
use pravah::graph::{self, GraphError, Flow};

fn approval(root: Flow<Request>) -> Flow<Decision> {
    root.map(prepare).agent(reviewer).suspend::<Decision>()
}

let flow = graph::compile(approval)?;
# Ok::<(), GraphError>(())
```

Use `.flow(other_flow)` to compose a subflow and `.each(item_flow)` to apply a
flow to each input item. The same function may be reused at multiple call
sites. Agent definitions use the same shape; see [clients.md](clients.md).

## Drive One Step at a Time

Create a runtime, then keep its control loop in application code:

```rust
loop {
    match runtime.next(ctx.clone()).await? {
        Step::Continue => {}
        Step::Suspend(payload) => {
            save(runtime.snapshot()?);
            present(payload);
            break;
        }
        Step::Done(output) => {
            complete(output);
            break;
        }
    }
}
```

`next()` performs at most one prepared instruction. Preparation can omit dead,
infallible shaping instructions, so exact `Step::Continue` counts are not a
stable API contract. Pravah does not spawn a background execution loop.

## Persist and Restore

`Runtime::snapshot()` captures the frame stack, suspension state, graph
fingerprint, and runtime-owned history. The graph remains a separately
serialized artifact. Store the complete snapshot as one versioned value.

Prepare the original graph and `HandlerRegistry`, then restore through that
`PreparedGraph`:

```rust
let prepared = PreparedGraph::new(graph, registry)?;
let mut runtime = prepared.restore(snapshot)?;
```

Install runtime-only client and MCP registrations on `Context` again after
restoration. Reattach history stores and compactors to the runtime when used.
Closures, credentials, live clients, and service handles are deliberately
absent from snapshots. Resolved agent configuration, memory, selected tools,
and resource text are checkpointed, so restoring does not rerun configuration
or reread MCP resources.

## Agent Activation

Agent structure and candidate tools are prepared with the graph. Its
`configure` function runs once when that agent invocation begins and may choose
the model, instructions, initial user message, memory, provider options, tool
subset, and MCP resources from the input and `Context`.

Configuration must be valid before Pravah changes VM state or history. Keep
external work performed by configuration read-only or idempotent so callers can
safely retry a failed step.

Every serialized format has an explicit version. During the `0.4.x` line,
incompatible versions are rejected and are not migrated automatically. Drain
in-flight workflows or keep the matching Pravah runtime when upgrading across
a format change.

## JSON Invocation

`JsonInvoker` binds one trusted graph and registry inside the host application.
External callers can submit only these versioned operations:

- `start` with an input value;
- `next` with the latest snapshot;
- `resume` with a suspended snapshot and resume value.

Each operation advances at most one workflow instruction and returns a fresh
snapshot. Start and resume values receive full JSON Schema validation. A
snapshot with a different graph fingerprint is rejected.

Pravah does not provide HTTP routes, authentication, snapshot storage, or
automatic retries. The host application owns those concerns.

## Diagrams

Graph diagrams show the authored workflow. Preparation may omit a small set of
dead shaping instructions from execution, but it never rewrites the serialized
graph or its diagrams.

## Effects and Retries

Pravah does not claim exactly-once delivery for arbitrary external effects. A
caller that repeats the same snapshot can repeat the next work handler or model
dispatch. Side-effecting handlers must therefore use application-level
idempotency keys or durable deduplication.

History entries have stable positions. `HistoryStore` implementations must
treat a repeated position as an idempotent replay so a partially persisted
batch can be retried safely.

## Legacy API

`pravah::legacy` remains available for compatibility. It receives fixes needed
to keep existing applications working, while new workflow capabilities target
`pravah::graph`.
