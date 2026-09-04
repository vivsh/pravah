# Pravah

[![Crates.io](https://img.shields.io/crates/v/pravah)](https://crates.io/crates/pravah)
[![docs.rs](https://img.shields.io/docsrs/pravah)](https://docs.rs/pravah)
[![License](https://img.shields.io/crates/l/pravah)](LICENSE-MIT)

**Durable workflows and agentic applications in typed Rust.**

_Pravah_ (प्रवाह, _pruh-VAH_) means “flow” or “current”.

Pravah is a workflow engine for application work that cannot—or should
not—finish in one request. It brings ordinary Rust functions, asynchronous
operations, agents, tools, and human decisions into one composable programming
model.

Work advances explicitly. Progress can be saved, the process can stop, and the
workflow can resume later without reconstructing state from logs or
conversation history. This makes Pravah well suited to applications where
software and AI must cooperate without surrendering control to a black-box
agent loop.

## Installation

```toml
[dependencies]
pravah = "0.4.13"
```

## Flow, Agent, and Chat

Pravah's modern API has three closely related entry points:

- `Flow<Input>` expresses typed application workflows.
- `Agent<Input>` defines typed agents with application tools.
- `pravah::Chat<Input, Output>` provides durable conversations using those
  same agents.

Flows and agents are ordinary Rust functions. Their signatures describe how
they compose, so substantial workflows remain readable and reusable.

### Flow

```rust
use pravah::{Context, Flow, GraphError, Step, compile};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize, JsonSchema)]
struct AttemptCount {
    value: u32,
}

fn approval(root: Flow<Request>) -> Flow<Decision> {
    let attempts = root.local(AttemptCount::default());

    root
        .store(&attempts, |_, mut count| {
            count.value += 1;
            count
        })
        .map(prepare_request)
        .flow(collect_evidence)
        .agent(reviewer)
        .load(&attempts, |recommendation, count| ApprovalRequest {
            recommendation,
            attempt: count.value,
        })
        .suspend::<Decision>()
}

let request: Request = load_request().await?;
let workflow = compile(approval)?;
let mut execution = workflow.start(request, Context::default())?;

let checkpoint = loop {
    match execution.next().await? {
        Step::Continue => {}
        Step::Suspend(payload) => {
            present_for_approval(payload).await?;
            break execution.snapshot()?;
        }
        Step::Done(_) => return Err(GraphError::Invalid(
            "approval completed before suspension".into(),
        )),
    }
};
save_checkpoint(checkpoint).await?;

let checkpoint = load_checkpoint().await?;
let decision: Decision = load_decision().await?;
let mut execution = workflow.restore(checkpoint, Context::default())?;
let mut step = execution.resume(decision).await?;

loop {
    match step {
        Step::Continue => step = execution.next().await?,
        Step::Suspend(payload) => {
            present_for_review(payload).await?;
            break;
        }
        Step::Done(value) => {
            complete(workflow.decode_output(value)?).await?;
            break;
        }
    }
}
```

A `Flow` can combine pure transformations, asynchronous work, branches,
collections, local state, reusable child flows, agents, and suspension points.
Inputs and outputs remain application types throughout the workflow.

Here, a typed local records the attempt count without changing the value moving
through the flow. Both `store` and `load` borrow its `TypedVar` handle, and
`Request` does not need to implement `Clone`. Pravah prepares the request,
gathers evidence, asks an agent for a recommendation, and then suspends with an
`ApprovalRequest`.

`Request` is the application's input, while `execution` is one active run of
the reusable compiled workflow. Its explicit step loop lets the application
choose exactly when to persist, yield, cancel, or resume work.

The resulting `Snapshot` is the workflow's durable checkpoint. The application
can store it using any persistence system, restore it in a later request or
process, and resume with a typed `Decision`. The storage and driver functions
in the example are deliberately application-owned.

### Agent

```rust
use pravah::{Agent, Toolset};

fn reviewer(root: Agent<ReviewRequest>) -> Agent<Recommendation> {
    root
        .tools(review_tools)
        .configure(configure_reviewer)
}

fn review_tools(root: Toolset) -> Toolset {
    root
        .tool(search_policy)
        .flow(verify_evidence)
}
```

An `Agent` has typed invocation input and typed model output. Its model,
instructions, message, memory, available tools, and budgets can be selected for
each invocation using application data and `Context`.

Tools use the same composable style. A focused tool can be an asynchronous Rust
function; a larger tool can be a complete `Flow`. Agents can therefore operate
through the application's domain types and workflows instead of an unrelated
tool abstraction.

### `pravah::Chat`

```rust
use pravah::{Chat, Context};

let mut chat = Chat::new(support_agent, Context::default());

let first = chat.send(question).await?;
println!("{}", first.output.answer);

let snapshot = chat.snapshot()?;
let mut restored = Chat::from_snapshot(
    support_agent,
    snapshot,
    Context::default(),
)?;

let next = restored.send(follow_up).await?;
```

`Chat` turns a function-defined `Agent` into a typed multi-turn conversation.
It retains the conversation across turns, drives the workflow on behalf of the
caller, and supports snapshot restoration across requests or processes.

Because chat uses the same agent API, it also supports dynamic configuration,
typed tools, budgets, memory, and application services. A conversational
feature can grow into a wider business workflow without replacing its agent
model.

## Designed for Real Application Work

| Scenario | How Pravah helps |
| --- | --- |
| Approvals and human review | Suspend with a typed request and resume when a decision arrives |
| Long-running operations | Preserve progress across process restarts and external delays |
| Tool-using agents | Expose typed application capabilities with practical usage budgets |
| Multi-agent pipelines | Compose specialised agents with typed hand-offs and ordinary Rust work |
| Durable chat | Retain typed conversation state and restore it in a later request or process |
| Data enrichment | Apply reusable child workflows to collections with explicit progress |

Pravah works equally well for a short application task and a process that may
remain unfinished for days.

## Why Pravah

- **Durable progress.** Workflow and conversation state can be snapshotted and
  restored. Suspension is a normal outcome rather than an exceptional code
  path.
- **Typed composition.** Rust types describe workflow values, agent input and
  output, tool calls, chat turns, and resume values.
- **Controlled agentic programming.** Agents participate in a wider workflow;
  they do not have to become the application's orchestration layer.
- **Reusable functions.** The same function-defined flow can be embedded in a
  larger process or exposed to an agent as a tool.
- **Application ownership.** The host decides when work runs, where state is
  stored, how failures are retried, and which services and credentials are
  available.

## The Operational Boundary

Pravah provides resumable workflow state, not an operational platform. A
typical host application advances a workflow, stores its snapshot, schedules
the next step or waits for an event, and restores the workflow when work can
continue.

Pravah is not a queue, database, distributed scheduler, application server, or
promise of exactly-once side effects. External work should be idempotent or
deduplicated according to the application's requirements.

This boundary keeps Pravah usable inside web services, workers, desktop
applications, command-line programs, and queue consumers without dictating the
surrounding infrastructure.

## Examples

| Example | Demonstrates |
| --- | --- |
| [`chat`](examples/chat.rs) | Typed graph chat with snapshot restoration |
| [`graph_typed_composition`](examples/graph_typed_composition.rs) | Reusing a typed child flow at multiple call sites |
| [`graph_snapshot_resume`](examples/graph_snapshot_resume.rs) | Suspending, saving, restoring, and resuming a workflow |
| [`graph_typed`](examples/graph_typed.rs) | Maps, branches, local state, child flows, and collections |
| [`graph_agent_budgets`](examples/graph_agent_budgets.rs) | Model-turn and per-tool budgets without custom policy code |
| [`story`](examples/story.rs) | A substantial multi-agent creative workflow |

See the [complete example index](examples/README.md) for prerequisites and run
commands.

## Documentation

- [Chat](docs/chat.md): typed multi-turn conversations, tools, and restoration
- [Graph workflows](docs/graph.md): execution, snapshots, and operational
  responsibilities
- [Agents and clients](docs/clients.md): models, tools, memory, and attachments
- [API reference](https://docs.rs/pravah)

Pravah is currently on the `0.4.x` line. The public API may continue to evolve
while the workflow experience is refined toward a stable release.

## License

Licensed under either the MIT License or Apache License 2.0, at your option.
