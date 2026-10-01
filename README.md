# Pravah

[![Crates.io](https://img.shields.io/crates/v/pravah)](https://crates.io/crates/pravah)
[![docs.rs](https://img.shields.io/docsrs/pravah)](https://docs.rs/pravah)
[![License](https://img.shields.io/crates/l/pravah)](LICENSE-MIT)

**Build agents that converse, use tools, and carry their work forward.**

Pravah is a durable workflow engine with first-class support for agentic
programming in Rust. Start with a typed chat builder. Add application tools,
memory, and state. When the task needs more than a conversation, compose agents
with ordinary Rust functions and human approval steps.

Conversation history, application state, and workflow progress can be captured
in a checkpoint and restored later. Your application owns when work runs,
where checkpoints live, and what an agent is allowed to do.

_Pravah_ (प्रवाह, _pruh-VAH_) means “flow” or “current”.

## Start with a chat

Define the input and output your application needs. Configure the conversation
with `Chat::builder()`, then send ordinary typed values—no agent configuration
callback required.

```rust
use pravah::{Chat, ChatRequest, Context};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Question {
    topic: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct Report {
    summary: String,
    next_steps: Vec<String>,
}

let mut chat = Chat::builder::<Question, Report>()
    .model("openai:///gpt-5")
    .instructions("Help the user make well-reasoned product decisions.")
    .turn_budget(6)
    .max_output_tokens(2_000)
    .state(String::from("A scheduling app for small teams"))
    .build(Context::default())?;

let request = ChatRequest::from(Question {
    topic: "What should we include in our first release?".into(),
})
.memory(format!("Current project: {}", chat.get()?));

let reply = chat.send_with_key(request, "message-42").await?;
println!("{}", reply.output.summary);

let follow_up = chat.send(Question {
    topic: "Which of those steps should we tackle first?".into(),
}).await?;

let checkpoint = chat.snapshot()?;
```

The chat retains its conversation between sends. Its input, output, and
application state have fixed Rust types; model responses are decoded into
`Report`, ready for application code. Builder inputs are rendered as JSON text.
For a text-only conversation, use `Chat::builder::<String, String>()` and
`chat.send("Hello").await?`.

In composed workflows, give an agent an `AgentConfig::key(...)` to share one
conversation across steps and child flows. Chat supplies its own stable key by
default; conversation history remains part of its checkpoint.

These snippets run inside an async function returning a compatible `Result`.
Configure provider credentials before sending; see [client setup](docs/clients.md).
For a complete example that needs no provider credentials, see the
[ChatBuilder example](examples/graph_chat_builder.rs).

## Give the conversation application capabilities

ChatBuilder brings the same tools and controls available to workflow agents
into a concise, conversational interface:

- **Tools:** register asynchronous Rust functions or complete workflows with
  `.tools(toolset)`. Select a subset for a message with `ChatRequest::tools(...)`.
- **Practical limits:** set model-turn and per-tool budgets. Exhausted tools
  become unavailable; an exhausted turn budget requests a final answer.
- **Adaptive control:** use `.control(controller)` when decisions depend on
  progress or results. Redirect the agent, restrict tools, or request conclusion.
- **Context and resources:** supply per-message text memory and select MCP text
  resources without putting credentials in the conversation.
- **Working memory:** install `.compactor(policy)` to prepare history before
  model requests, retaining relevant context as conversations grow.
- **Persistence integration:** attach `.store(history_store)` for history
  delivery, and save snapshots through your application's storage system.

Configure state, tools, and services before the final `.build(ctx)?`.
See the [Chat guide](docs/chat.md) for builder options, dynamic tool selection,
budgets, and service setup.

## Keep state with the conversation

Application state should not require a separate persistence lifecycle just
because a conversation uses an agent. With `.state(initial_state)`, typed
`get()` and `set()` let the application manage its own data between turns:

```rust
let project = chat.get()?;
chat.set(format!("{project}; first release approved"))?;

let checkpoint = chat.snapshot()?;
save_checkpoint(checkpoint).await?;
```

One snapshot holds conversation history, application state, and execution
progress. State does not automatically enter the model's context: the first
example explicitly supplies it as memory. Per-message memory stays outside
conversation history, but may be retained in execution checkpoints.

Restore with the same builder definition and state type, attaching fresh
runtime dependencies. Instructions alone can change without discarding the conversation:

```rust
let checkpoint = load_checkpoint().await?;
let mut chat = Chat::builder::<Question, Report>()
    .model("openai:///gpt-5")
    .instructions("Help the user make well-reasoned product decisions.")
    .turn_budget(6)
    .max_output_tokens(2_000)
    .restore::<String>(checkpoint, Context::default())?;

let project = chat.get()?;
```

State comes from the snapshot; no initial `.state(...)` is needed on restore.
Updated instructions apply to new invocations; an already-configured invocation
retains its saved instructions until it finishes. Other settings must still match.
The save/load functions above belong to your application. Restoring a chat
between turns lets it continue in another request or process. For explicit
control over suspended or unfinished work, use a workflow execution loop.

## When the task is bigger than chat

Use `Flow<Input>` to coordinate application work and `Agent<Input>` to define
reusable agents. Both are ordinary Rust functions with typed outputs. Agents
can participate in a workflow, and workflows can become agent tools.

An approval process, for example, can collect evidence, ask an agent for a
recommendation, and pause for a human decision:

```rust
use pravah::Flow;

fn approval(root: Flow<Request>) -> Flow<Decision> {
    let attempts = root.local(0_u32);

    root
        .store(&attempts, |_, count| count.saturating_add(1))
        .map(prepare_request)
        .flow(collect_evidence)
        .agent(reviewer)
        .load(&attempts, |recommendation, attempt| ApprovalRequest {
            recommendation,
            attempt,
        })
        .suspend::<Decision>()
}
```

Here, the types and functions describe your application. `local`, `store`, and
`load` keep typed state alongside the value moving through the flow. The same
`reviewer` agent could also power a function-defined `Chat::new(reviewer, ctx)`.

Execution stays explicit. Advance one step at a time and decide when to
checkpoint, yield, or wait for input:

```rust
use pravah::{Context, Step, compile};

let workflow = compile(approval)?;
let executor = workflow.prepared().executor(Context::default());
let mut execution = workflow.start(request, uuid::Uuid::now_v7())?;

loop {
    match execution.next()? {
        Step::Continue => {}
        Step::Fetch(fetch) => {
            save_checkpoint(execution.snapshot()?).await?;
            let response = executor.execute(&fetch).await?;
            execution.resume_fetch(fetch.id(), Ok(response))?;
        }
        Step::Suspend(payload) => {
            save_checkpoint(execution.snapshot()?).await?;
            request_approval(payload).await?;
            break;
        }
        Step::Done(value) => {
            complete(workflow.decode_output(value)?).await?;
            break;
        }
    }
}
```

Later, use `workflow.restore(checkpoint)?` and `execution.resume(decision)?`,
then continue calling `next()` in the same explicit loop. The workflow advances
synchronously; the application executes external requests asynchronously.
Persist pending requests and their outcomes, and use idempotency or reconciliation
where effects may be repeated after a crash. Pravah does not provide exactly-once effects.

See the [suspend–restore–resume example](examples/graph_snapshot_resume.rs) and
[workflow guide](docs/graph.md) for the complete lifecycle.

## Where Pravah fits

| Application | What Pravah brings |
| --- | --- |
| Support assistants and product copilots | Typed chat, application state, memory, and domain tools |
| Research and investigation | Multi-step tool use, budgets, and result-aware intervention |
| Approvals and human review | Typed suspension requests and durable resumption |
| Multi-agent processes | Reusable agents with typed hand-offs and ordinary Rust work |
| Background enrichment and automation | Composable workflows with explicit progress and checkpoints |

Pravah provides resumable execution, not hosted infrastructure. Bring your own
storage, queue, scheduler, and retry policy. It does not promise automatic
retries, distributed execution, or exactly-once side effects; external work
should be idempotent or deduplicated as appropriate.

## Install and explore

```toml
[dependencies]
pravah = "0.4.20"
schemars = "1"
serde = { version = "1", features = ["derive"] }
```

- [Chat guide](docs/chat.md): builders, typed requests, state, and restoration
- [Agents and clients](docs/clients.md): models, tools, budgets, and configuration
- [History and compaction](docs/history.md): persistence and working-memory policies
- [MCP resources](docs/mcp.md): resource selection and runtime credentials
- [Workflow guide](docs/graph.md): composition, execution, and operational responsibilities
- [Runnable examples](examples/README.md): prerequisites and run commands
- [API reference](https://docs.rs/pravah)

Pravah is on the `0.4.x` release line. APIs may change as the programming model
is refined toward a stable release.

## License

Licensed under either the MIT License or Apache License 2.0, at your option.
