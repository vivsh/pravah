# Flows

Read this when designing the graph itself: node types, execution rules,
suspension, nested flows, snapshots, and history. For providers, tools, and
attachments, see [clients.md](clients.md).

## Execution Model

Pravah is intentionally:

- stepwise
- deterministic
- single-runtime
- explicit about state transitions

The runtime advances one bounded step at a time. Parallelism is expressed in the
graph through `split` and `merge`, not hidden inside a scheduler.

Within one flow graph, one Rust type identifies one node. That rule gives you
deterministic routing, replayable progress, and unambiguous resume points.

## Building A Flow

Implement `Flow` for the input type, declare the output type, and assemble the
graph with `FlowBuilder`.

```rust
impl Flow for Proposal {
    type Output = Brief;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .split(split_proposal)
            .agent::<TechTrack>()
            .agent::<MktTrack>()
            .agent::<RiskTrack>()
            .merge(merge_brief)
    }
}
```

You describe nodes by input and output type. Pravah validates the graph and
computes how values move between nodes.

## Node Types

| Builder method      | What it does                                                              |
| ------------------- | ------------------------------------------------------------------------- |
| `agent::<A>()`      | LLM-backed node with structured output or a tool loop                     |
| `work(f)`           | Effectful async transform: `async fn(I, Context) -> Result<O, FlowError>` |
| `map(f)`            | Pure synchronous transform: `fn(I) -> O`                                  |
| `either(f)`         | Route to one branch: `fn(I) -> Either<A, B>`                              |
| `split(f)`          | Fan out to multiple branches                                              |
| `merge(f)`          | Collect branch outputs once all are ready                                 |
| `suspend::<I, O>()` | Pause the flow and resume later with `O`                                  |
| `flow::<F>()`       | Embed another flow as a node                                              |
| `each::<F>()`       | Run sub-flow `F` once per item in a `Vec<F>`, collect `Vec<F::Output>`    |

`fork` and `join` are binary aliases for `split` and `merge`.

Runnable examples:

- [../examples/linear_flow.rs](../examples/linear_flow.rs)
- [../examples/split_merge.rs](../examples/split_merge.rs)
- [../examples/nested_flow.rs](../examples/nested_flow.rs)

## Pure, Effectful, And Control Nodes

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

Control nodes:

- `flow`
- `suspend`

These shape execution without doing business logic themselves.

## Driving The Runtime

Create a `FlowRuntime`, then call `next()` until the flow finishes or suspends.

```rust
loop {
    match runtime.next(ctx.clone()).await? {
        FlowStep::Continue => {}
        FlowStep::Done(value) => break value,
        FlowStep::Suspend(_) => {
            runtime.resume(ctx.clone(), decision).await?;
        }
    }
}
```

`FlowStep::Continue` means one bounded step completed and more work remains.
`FlowStep::Done` returns the typed output. `FlowStep::Suspend` hands control
back until you resume.

If you want the runtime to manage the outer loop for you, use `run_until()`.

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

To suspend from inside an agent tool loop, implement the tool as a sub-flow
that contains a `suspend::<I, O>()` node, then register it as both a tool and a
flow.

```rust
impl Flow for BlogRequest {
    type Output = FinalResult;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .agent::<BlogRequest>()
            .tool_with::<BlogRequest, HumanInput, HumanOutput>()
            .flow::<HumanInput>()
    }
}
```

When the agent calls that tool, the engine enters the `HumanInput` sub-flow.
If the sub-flow suspends, the outer runtime also returns `FlowStep::Suspend`.

See [../examples/human_input.rs](../examples/human_input.rs).

## Multi-Turn Agent Conversations

By default, each agent entry gets a fresh session id. That isolates agents,
but a looping agent then sees no previous turns from earlier iterations.

Call `.keep_alive()` on `AgentConfig` to preserve one session across re-entries:

```rust
fn configure() -> AgentConfig {
    AgentConfig::new(
        "You are a helpful assistant.",
        "gemini:///gemini-2.5-flash",
    )
    .keep_alive()
}
```

With `keep_alive`, one agent keeps its own conversation history within the
current parent frame.

## Nested Flows

A nested flow has the same outer shape as a node: typed input, typed output,
and stepwise execution.

```rust
fn build(builder: FlowBuilder) -> FlowBuilder {
    builder
        .work(derive_query)
        .flow::<ResearchQuery>()
        .agent::<ResearchResult>()
}
```

Nested flows preserve the same guarantees as top-level flows: deterministic
execution, resumability, typed boundaries, and snapshot safety.

See [../examples/nested_flow.rs](../examples/nested_flow.rs).

## Each Node

`each::<F>()` fans out over a `Vec<F>` input, running the sub-flow `F` once for
each element sequentially, and collecting the results into `Vec<F::Output>`.

```rust
impl Flow for Vec<ReviewTask> {
    type Output = ReviewSummary;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .each::<ReviewTask>()           // Vec<ReviewTask> → Vec<ReviewOutput>
            .work(|results: Vec<ReviewOutput>, _| async move {
                Ok(ReviewSummary { items: results })
            })
    }
}
```

The flow entry is the `Vec<F>` itself. Items are processed one at a time; the
engine pushes a child frame for each item, lets it run to completion, then
moves on to the next. The parent frame is not resumed until all items are done.

An empty input vec immediately writes an empty `Vec<F::Output>` and continues
without ever pushing a child frame.

Snapshots work correctly mid-iteration: `each_queues` (remaining items and
partial results) is fully serialized.

### Limitations

- **No cross-item history.** Each item runs in a fresh child frame. Agents with
  `keep_alive` inside the sub-flow do not carry conversation history from one
  item to the next.

- **Completed sessions are not compacted.** History compaction only runs over
  live frames. For large fan-outs over agent-heavy sub-flows, `FlowHistory`
  grows unboundedly until the runtime is dropped.

- **Sequential only.** Items are processed in order, one at a time. There is no
  parallel dispatch.

## Snapshots And Persistence

Call `runtime.snapshot()` to capture execution state, then restore it later
with `FlowRuntime::from_snapshot(snapshot)`.

Snapshots contain runtime state, pending branches, suspend points, nested flow
frames, and execution progress. They do not capture closures, async tasks,
thread-local state, or executor handles.

If you also want prior conversation history after restore, reattach it with
`with_history()`.

See [../examples/snapshot.rs](../examples/snapshot.rs).

## Run Limits

`FlowRuntime::run_until` accepts `RunLimits` to cap execution.

```rust
use pravah::flows::{RunLimits, RunOutcome};
use std::time::Duration;

match runtime
    .run_until(
        ctx,
        RunLimits::new()
            .max_steps(200)
            .max_turns(10)
            .max_depth(8)
            .max_duration(Duration::from_secs(30)),
    )
    .await?
{
    RunOutcome::Done(value) => { /* flow finished */ }
    RunOutcome::Suspend(sv) => { /* flow suspended */ }
    RunOutcome::LimitExceeded(kind) => { /* soft limit reached */ }
}
```

`max_turns` is the one hard limit. It returns `Err(FlowError::LimitExceeded)`.
The other limits return `RunOutcome::LimitExceeded`.

Available limits:

| Limit          | Description                             |
| -------------- | --------------------------------------- |
| `max_steps`    | Total engine steps across the whole run |
| `max_turns`    | LLM dispatches                          |
| `max_depth`    | Maximum call-stack depth                |
| `max_duration` | Wall-clock time                         |

## History And Inspection

LLM history is separate from runtime execution state. That lets you vary
storage, compaction, and retention without changing the graph model.

Pravah includes:

- `NoopHistoryStore` and custom `HistoryStore` implementations
- `NoopCompactor`, `SlidingWindowCompactor`, and custom `HistoryCompactor`
- `FlowInspector` for live runtime inspection

Use `runtime.inspector()` to inspect the active frames and current session
messages.

```rust
for msg in runtime.inspector().messages() {
    println!("{:?}: {}", msg.role, msg.content);
}
```

To inject a user message before the next agent dispatch, wait until the agent is
at a dispatch boundary:

```rust
if runtime.inspector().is_agent_dispatch_ready() {
    runtime.inject_message("What else should I know?")?;
}
```

That message is appended to the active session before the next LLM call.

## Diagrams And Operations

Use `FlowGraphDiagram` when you want a structural view of the graph.

```rust
let diagram = FlowGraphDiagram::from_flow::<ArticleRequest>()?;

println!("{}", diagram.render_tree());
println!("{}", diagram.mermaid());
println!("{}", diagram.dot());
```

Pravah also emits `tracing` events for runtime steps, tool calls, retries, rate
limiting, suspension, and run limits.

See [../examples/gen_diagrams.rs](../examples/gen_diagrams.rs) for diagram
generation and [clients.md](clients.md#client-layers) for retry, tracing, and
rate-limit layers.

## Example Map

- [../examples/linear_flow.rs](../examples/linear_flow.rs): one agent, one work node
- [../examples/split_merge.rs](../examples/split_merge.rs): multi-branch composition
- [../examples/nested_flow.rs](../examples/nested_flow.rs): embedded subflows
- [../examples/human_input.rs](../examples/human_input.rs): suspend and resume through a tool
- [../examples/snapshot.rs](../examples/snapshot.rs): snapshot and restore
- [../examples/story.rs](../examples/story.rs): looping flow with repeated turns
- [../examples/each_node.rs](../examples/each_node.rs): fan-out over a list with the `each` node
- [../examples/debate.rs](../examples/debate.rs): larger multi-agent orchestration
- [../examples/gen_diagrams.rs](../examples/gen_diagrams.rs): tree, Mermaid, and DOT output

If you are starting from the top, go back to [../README.md](../README.md).
