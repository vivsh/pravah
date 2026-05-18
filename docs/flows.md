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

| Builder method | What it does |
| -------------- | ------------ |
| `agent::<A>()` | LLM-backed node with structured output or tool loop |
| `work(f)` | Effectful async transform: `async fn(I, Context) -> Result<O, FlowError>` |
| `map(f)` | Pure synchronous transform: `fn(I) -> O` |
| `either(f)` | Route to one branch: `fn(I) -> Either<A, B>` |
| `split(f)` | Fan out to multiple branches |
| `merge(f)` | Collect branch outputs once all are ready |
| `suspend::<I, O>()` | Pause the flow and resume later with `O` |
| `flow::<F>()` | Embed another flow as a node |

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

Use `ToolBox::suspend::<T, Out>()` when a running agent should surface a typed
pause from inside its tool loop.

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

## Nested Flows

Flows compose because a flow has the same outer shape as a node: typed input,
typed output, stepwise execution.

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