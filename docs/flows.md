# Flows

Read this when you are designing the graph itself: node types, execution rules,
suspension, nested flows, snapshots, and history. For provider configuration,
tools, and attachments, see [clients.md](clients.md).

## Execution Model

Pravah is intentionally:

- stepwise
- deterministic
- single-runtime
- explicit about state transitions

The runtime advances one bounded step at a time. Parallelism is expressed in the
graph through `split` and `merge`, not hidden inside a scheduler.

This makes the runtime predictable, replayable, and easy to suspend.

## Node Types

| Builder method      | What it does                                                              |
| ------------------- | ------------------------------------------------------------------------- |
| `agent::<A>()`      | LLM-backed node with structured output or tool loop                       |
| `work(f)`           | Effectful async transform: `async fn(I, Context) -> Result<O, FlowError>` |
| `map(f)`            | Pure synchronous transform: `fn(I) -> O`                                  |
| `either(f)`         | Route to one branch: `fn(I) -> Either<A, B>`                              |
| `split(f)`          | Fan out to multiple branches                                              |
| `merge(f)`          | Collect branch outputs once all are ready                                 |
| `suspend::<I, O>()` | Pause the flow and resume later with `O`                                  |
| `flow::<F>()`       | Embed another flow as a node                                              |

`fork` and `join` are binary aliases for `split` and `merge`.

See these runnable examples:

- [../examples/linear_flow.rs](../examples/linear_flow.rs)
- [../examples/split_merge.rs](../examples/split_merge.rs)
- [../examples/nested_flow.rs](../examples/nested_flow.rs)

## Pure Vs Effectful Nodes

Pravah keeps pure routing logic separate from effectful work.

Pure algebra nodes:

- `map`
- `either`
- `split`
- `merge`

These are infallible plain functions with no `Context`.

Effectful nodes:

- `work`
- `agent`

These may perform I/O and return `Result<_, FlowError>`.

This separation keeps routing logic simple and makes failures explicit only
where effects actually happen.

## Suspend And Resume

There are two suspension styles.

### Flow-Level Suspend

Use `suspend::<I, O>()` when the graph should pause at a dedicated node.

```rust
builder.suspend::<ApprovalRequest, ApprovalDecision>()
```

When a value of type `I` reaches that node, the runtime returns
`FlowStep::Suspend`. Resume by supplying a value of type `O`.

### Tool-Level Suspend

To suspend from inside an agent's tool loop, implement the tool as a sub-flow
that contains a `suspend::<I, O>()` node, then register it as both a tool and a
flow on the builder.

```rust
impl Flow for BlogRequest {
    type Output = FinalResult;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<BlogRequest>()
            .tool::<BlogRequest, HumanInput, HumanOutput>()
            .flow::<HumanInput>()   // HumanInput::build() contains a suspend node
            .build()
    }
}
```

When the agent calls the tool, the engine enters the `HumanInput` sub-flow.
If it reaches the `suspend` node, the runtime returns `FlowStep::Suspend` to
the caller, just like a top-level suspend.

Both end up with the same outer control loop:

```rust
loop {
    match runtime.next(ctx.clone()).await? {
        FlowStep::Continue => {}
        FlowStep::Done(v) => break,
        FlowStep::Suspend(sv) => {
            runtime.resume(ctx.clone(), decision).await?;
        }
    }
}
```

See [../examples/human_input.rs](../examples/human_input.rs) for the practical
shape.

## Multi-Turn Agent Conversations

By default, each time an agent node is entered it gets a fresh session id.
That isolates agents from each other, but means a looping agent (one connected
back to itself via `either`) sees no history from previous iterations.

Call `.keep_alive()` on `AgentConfig` to opt into continuous history across
loop iterations:

```rust
fn build() -> AgentConfig {
    AgentConfig::new("You are a helpful assistant.", "gemini://gemini-2.5-flash")
        .keep_alive()
}
```

With `keep_alive`, the engine assigns one stable session id to all invocations
of that agent within one parent frame. The LLM sees the full conversation
history on every re-entry. Multiple `keep_alive` agents in the same parent flow
each maintain their own independent session.

### Injecting Messages

Use `FlowRuntime::push_message` to append a user message to the active session
before the next LLM dispatch. Call it between `next()` calls when
`FlowInspector::is_agent_dispatch_ready` returns `true`:

```rust
if inspector.is_agent_dispatch_ready() {
    runtime.push_message("What else should I know?");
}
```

`is_agent_dispatch_ready` returns `true` while the top agent frame is in its
`Entry` phase — the next `next()` call will push the structured input to history
and dispatch to the LLM. Any message pushed here appears before that dispatch.

### Inspecting Session History

Use `FlowInspector::messages` to iterate over the live messages in the current
session, oldest first:

```rust
for msg in inspector.messages() {
    println!("{:?}: {}", msg.role, msg.content);
}
```

Evicted (compacted) messages are excluded. Only messages belonging to the
active frame's session id are returned.

## Nested Flows

A flow has the same outer shape as a node: typed input, typed output, stepwise
execution.

```rust
FlowGraph::builder()
    .flow::<PlannerFlow>()
    .flow::<ResearchFlow>()
    .flow::<ReviewFlow>()
    .build()
```

Nested flows keep the same guarantees as top-level flows: deterministic
execution, resumability, typed boundaries, and snapshot safety.

See [../examples/nested_flow.rs](../examples/nested_flow.rs).

## Persistence And Snapshots

Call `runtime.snapshot()` to capture the entire execution state, then restore it
later with `FlowRuntime::from_snapshot(snapshot)`.

Snapshots contain runtime state, pending branches, suspend points, nested flow
state, and execution progress. They do not capture closures, async tasks,
thread-local state, or executor-specific handles.

That keeps snapshots portable across processes and machines.

See [../examples/snapshot.rs](../examples/snapshot.rs).

## Run Limits

`FlowRuntime::run_until` accepts a `RunLimits` value to cap execution:

```rust
use pravah::flows::{RunLimits, RunOutcome};

match runtime.run_until(ctx, RunLimits::new().max_turns(10).max_steps(200)).await? {
    RunOutcome::Done(v) => { /* flow finished */ }
    RunOutcome::Suspend(sv) => { /* flow suspended */ }
    RunOutcome::LimitExceeded(kind) => { /* cap reached */ }
}
```

`run_until` returns `Result<RunOutcome<T>, FlowError>`. Most limits produce
`Ok(RunOutcome::LimitExceeded(LimitKind))`. The `max_turns` limit is a hard
error and returns `Err(FlowError::LimitExceeded(...))` instead.

Available limits:

| Limit          | Description                             |
| -------------- | --------------------------------------- |
| `max_steps`    | Total engine steps across the whole run |
| `max_turns`    | LLM dispatches (model round-trips)      |
| `max_depth`    | Maximum call-stack depth                |
| `max_duration` | Wall-clock time                         |

## History Management

LLM history is intentionally separate from runtime execution state. That lets
you vary storage, compaction, summarization, and retention policy without
changing the graph model itself.

Pravah includes:

- sliding-window compaction
- custom compactor hooks
- pluggable history stores

Use this separation when the runtime state must stay durable, but conversation
history needs a different lifecycle.

## Tracing And Operations

Pravah emits `tracing` events for runtime steps, client dispatches, tool calls,
retries, rate limiting, suspension, and run limits.

Use this when you need replayable execution plus operational visibility.

Client-layer retry, rate limiting, and tracing wrappers are documented in
[clients.md](clients.md#client-layers).

## Example Map

- [../examples/linear_flow.rs](../examples/linear_flow.rs): one agent, one work node
- [../examples/split_merge.rs](../examples/split_merge.rs): multi-branch flow
- [../examples/nested_flow.rs](../examples/nested_flow.rs): composition with subflows
- [../examples/human_input.rs](../examples/human_input.rs): suspend and resume
- [../examples/snapshot.rs](../examples/snapshot.rs): snapshot and restore
- [../examples/story.rs](../examples/story.rs): interactive looping flow
- [../examples/debate.rs](../examples/debate.rs): larger multi-agent orchestration
- [../examples/gen_diagrams.rs](../examples/gen_diagrams.rs): visualize the graph

If you are starting from the top, go back to [../README.md](../README.md).
