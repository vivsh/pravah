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

Compile a definition once when registering it separately from its agent worker:

```rust
let executor = flow.prepared().executor(ctx);
let flow_factory = move || flow.clone();
```

`CompiledFlow::clone()` shares the prepared graph and callback instances without
allocating or recompiling. Input and output types need not implement `Clone`.
Each start or restore owns independent execution state. The worker's providers
and services remain explicitly selected through `Context`.

Derive both registrations from this same definition rather than independently
building the factory. When updating an existing registration, preserve its
construction convention: `compile(factory)` and
`factory(Flow::root()).finish::<Input>()` currently produce different root names,
handler keys, and graph fingerprints. Sharing does not establish compatibility
between separately built deployments or different callback implementations.

## Drive One Step at a Time

Bind runtime-only dependencies to an external executor. The VM itself is
synchronous; keep its control loop in application code:

```rust
let executor = flow.prepared().executor(ctx);
let mut runtime = flow.start(input, execution_id)?;

loop {
    match runtime.next()? {
        Step::Continue => {}
        Step::Agent(request) => {
            save(runtime.snapshot()?);
            let response = executor.execute(&request).await;
            runtime.resume_agent(response)?;
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
restoration. Reinstall stores and compactors on the executor when the saved history policy requires them.
Live clients and service handles are absent from snapshots, but supplied request
data can contain credentials or other sensitive content; secure its storage.
Resolved agent configuration, memory, selected tools,
and resource text are checkpointed, so restoring does not rerun configuration
or reread MCP resources.

## Agent Activation

Agent structure and candidate tools are prepared with the graph. Fixed settings
can use `.model(...).instructions(...).build()` in the agent definition; input
is rendered as JSON text. Custom `.configure(...)` remains available for dynamic
behavior and message rendering. See [agent declarations](clients.md).

The configuration function runs once when that agent invocation begins and may choose
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

Every serialized format has an explicit version. Synchronous execution with agent workers
uses snapshot format 15 and JSON wire format 11; older snapshots and wire
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
- `resume_agent` with a snapshot and recorded `AgentResponse`.

`start` and `next` advance at most one instruction. Resume operations only accept
delivery; call `next` afterward. Every operation returns a fresh snapshot, and
`agent` responses expose requests for external execution. Start and resume values receive schema validation. A
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

Import `HistoryPolicy`, `HistoryStore`, `Compactor`, `CompactionRequest`, `HistoryEntry`, and
`CompactionResult` directly from `pravah`. The same types are available under
`pravah::history` for applications that prefer an explicit module path.

Enable history intent with `runtime.with_history(HistoryPolicy { persist: true,
load: true, compact: true })?` before stepping. Install matching dependencies with
`executor.with_store(store).with_compactor(policy)`. The emitted agent requests
carry accepted originals for persistence before any compaction. Without enabled
policy, history accumulates in runtime and snapshots without trimming.

The policy receives effective request options, framework guidance, and separate
committed/protected history views at an upcoming agent dispatch. Current input
and tool groups are protected. Invalid replacements leave history unchanged.
Worker responses retain successful partial acknowledgements even after failure.
Do not redispatch a completion already accepted by the VM. See the
[history guide](history.md) and [worker delivery guide](agent_execution.md).

## Legacy API

`pravah::legacy` remains available for compatibility. It receives fixes needed
to keep existing applications working, while new workflow capabilities target
the modern typed API and its underlying `pravah::graph` runtime.
