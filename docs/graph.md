# Graph Workflows

The crate root exposes Pravah's primary typed workflow API. The complete
`pravah::graph` namespace also provides untyped graphs and JSON invocation;
every authoring path uses the same runtime.

## Author With Functions

A flow is an ordinary function from one typed flow value to another:

```rust
use pravah::{Flow, GraphError, compile};

fn approval(root: Flow<Request>) -> Flow<Decision> {
    root.map(prepare).agent(reviewer).suspend::<Decision>()
}

let flow = compile(approval)?;
# Ok::<(), GraphError>(())
```

Use `.flow(other_flow)` to compose a subflow and `.each(item_flow)` to apply a
flow to each input item. The same function may be reused at multiple call
sites. Agent definitions use the same shape; see [clients.md](clients.md).

## Drive One Step at a Time

Bind runtime-only dependencies to an external executor. The VM itself is
synchronous; keep its control loop in application code:

```rust
let executor = flow.prepared().executor(ctx);
let mut runtime = flow.start(input, execution_id)?;

loop {
    match runtime.next()? {
        Step::Continue => {}
        Step::Fetch(fetch) => {
            save(runtime.snapshot()?);
            let response = executor.execute(&fetch).await?;
            runtime.resume_fetch(fetch.id(), Ok(response))?;
        }
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

Typed workflows restore through the same compiled flow. Attach fresh
runtime dependencies to the executor:

```rust
let mut runtime = flow.restore(snapshot)?;
let executor = flow.prepared().executor(ctx);
```

Install runtime-only client and MCP registrations on `Context` again during
restoration. Reattach a fresh `HistoryManager` for stores and compactors when used.
Live clients and service handles are absent from snapshots, but supplied request
data can contain credentials or other sensitive content; secure its storage.
Resolved agent configuration, memory, selected tools,
and resource text are checkpointed, so restoring does not rerun configuration
or reread MCP resources.

## Agent Activation

Agent structure and candidate tools are prepared with the graph. Its
`configure` function runs once when that agent invocation begins and may choose
the model, instructions, initial user message, memory, provider options, tool
subset, and MCP resources from the input and `Context`.

Configuration is checkpointed as a pending request before the callback executes.
Its successful result must be valid before resolved settings or history are committed. Keep
external work performed by configuration read-only or idempotent so callers can
safely retry a failed step.

An agent may also declare an optional asynchronous `control` function. It runs
at explicit model and tool boundaries and can redirect guidance and tool
visibility, request a final tool-disabled answer, suspend for application
input, or abort the current step. Controller observations, metrics, state, and
committed boundaries survive snapshots. Controller suspension uses the same
typed `Runtime::resume` entry point as an ordinary suspend node, with
`AgentResume` as its fixed resume value. Dynamic graph callers use
`Runtime::resume_value` with an existing Pravah `Value`.

Every serialized format has an explicit version. Synchronous execution with Fetch
uses snapshot format 10 and JSON wire format 8; older snapshots and wire
requests are rejected rather than interpreted with different semantics. During the `0.4.x`
line, incompatible versions are rejected and are not migrated automatically. Drain
in-flight workflows or keep the matching Pravah runtime when upgrading across
a format change.

## JSON Invocation

`JsonInvoker` binds one trusted graph and registry inside the host application.
External callers can submit only these versioned operations:

- `start` with an input value and execution UUID;
- `next` with the latest snapshot;
- `resume` with a suspended snapshot and resume value;
- `resume_fetch` with a snapshot, pending request UUID and response or failure.

`start` and `next` advance at most one instruction. Resume operations only accept
delivery; call `next` afterward. Every operation returns a fresh snapshot, and
`fetch` responses expose requests for external execution. Start and resume values receive schema validation. A
snapshot with a different graph fingerprint is rejected.

Pravah does not provide HTTP routes, authentication, snapshot storage, or
automatic retries. The host application owns those concerns.

## Diagrams

Graph diagrams show the authored workflow. Preparation may omit a small set of
dead shaping instructions from execution, but it never rewrites the serialized
graph or its diagrams.
See the [diagram example](../examples/graph_diagram.rs) for Mermaid and DOT output.

## Effects and Retries

Pravah does not claim exactly-once delivery for arbitrary external effects. A
caller that repeats a pending request can repeat the external effect or model
dispatch. Side-effecting handlers must therefore use application-level
idempotency keys or durable deduplication.

Agent configuration can supply `Message::user(text).with_key(application_key)`.
History stores and compactors receive the key through `entry.message.key`, and
snapshots preserve it. Keys are application metadata, not provider prompt content,
tool-call IDs, or automatic deduplication identifiers. This also works with
[Chat](chat.md#application-message-keys).

History entries have stable positions. `HistoryStore` implementations must
deduplicate repeated entry IDs so a partially persisted
batch can be retried safely.

Import `HistoryManager`, `HistoryStore`, `Compactor`, `CompactionRequest`, `HistoryEntry`, and
`CompactionResult` directly from `pravah`. The same types are available under
`pravah::history` for applications that prefer an explicit module path.

Use `HistoryManager::new().with_store(store).with_compactor(policy)` outside the
VM. Call `manager.maintain(&mut execution, ctx.clone()).await?` before stepping
and after completion. It persists original messages before any compaction.
Without a manager, history accumulates in runtime and snapshots without trimming.

The policy receives effective request options, framework guidance, and separate
committed/protected history views at an upcoming agent dispatch. Current input
and tool groups are protected. Invalid replacements leave history unchanged.
After a maintenance failure, retry maintenance without redispatching a completed
Fetch. See the [history guide](history.md) for the explicit execution loop.

## Legacy API

`pravah::legacy` remains available for compatibility. It receives fixes needed
to keep existing applications working, while new workflow capabilities target
the modern typed API and its underlying `pravah::graph` runtime.
