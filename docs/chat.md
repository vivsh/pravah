# Chat

`pravah::Chat` provides a typed multi-turn conversation backed by the
same agent and durable workflow facilities as other graph workflows.

Use it when a conversation needs dynamic agent configuration, typed tools,
budgets, application-controlled history persistence, or snapshot restoration.
The older direct-client chat helper is available only through
`pravah::legacy::Chat` for existing applications.

## Define A Chat Agent

A chat begins with an ordinary function-defined agent:

```rust
use pravah::clients::Message;
use pravah::{Agent, AgentConfig, Chat, Context, GraphError};

fn tutor(root: Agent<Question>) -> Agent<Answer> {
    root.configure(configure_tutor)
}

async fn configure_tutor(
    question: Question,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///gpt-5",
        "You are a concise Rust tutor.",
        Message::user(question.text),
    )
    .keep_alive())
}

let mut chat = Chat::new(tutor, Context::default());
```

`Question` and `Answer` are application types implementing `Serialize`,
`DeserializeOwned`, and `JsonSchema`. Enable `keep_alive` when later calls to
`send` should retain the same model-visible conversation.

The configuration function runs for each new chat input. It may select the
model, instructions, initial user message, memory, tools, resources, and
budgets from the input and `Context`.

## Send Typed Messages

```rust
let first = chat
    .send(Question::new("What is ownership?"))
    .await?;
println!("{}", first.output.text);

let second = chat
    .send(Question::new("Show a short example."))
    .await?;
println!("{}", second.output.text);
```

`send` drives the workflow until the agent produces one typed response. The
chat retains one runtime across turns; callers do not need to operate the
stepwise execution loop directly.

Provider clients come from the session-bound `Context`. `Context::default()`
uses Rath's default client factory, while `Context::with_client_factory`
installs an application-specific factory. A chat uses the same context for
every turn until it is snapshotted and restored with a new context.

## Tools And Budgets

Chat agents use the same toolset and configuration APIs as any graph agent:

```rust
fn support_agent(root: Agent<SupportQuestion>) -> Agent<SupportAnswer> {
    root
        .tools(support_tools)
        .configure(configure_support)
}

fn support_tools(root: Toolset) -> Toolset {
    root.tool(find_account).flow(verify_resolution)
}

async fn configure_support(
    question: SupportQuestion,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///gpt-5",
        "Resolve the request using only relevant account information.",
        Message::user(question.text),
    )
    .keep_alive()
    .turn_budget(6)
    .tool_budget::<FindAccount>(2))
}
```

See [clients.md](clients.md) for model URLs, tools, dynamic filters, memory,
attachments, and agent configuration.

## Snapshot And Restore

Take a snapshot after at least one completed turn and persist it with the
application's own storage:

```rust
let snapshot = chat.snapshot()?;

let mut restored = Chat::from_snapshot(tutor, snapshot, restored_ctx)?;
let next = restored
    .send(Question::new("Continue our discussion."))
    .await?;
```

Restoration requires the same agent definition. Live provider clients and
application services are not serialized; bind them through the restoration
`Context` or the service setters before continuing.

## History Persistence And Compaction

Attach an application history store or compactor when constructing or
restoring a chat:

```rust
let chat = Chat::new(tutor, ctx)
    .with_store(history_store)
    .with_compactor(history_compactor);

let restored = Chat::from_snapshot(tutor, snapshot, restored_ctx)?
    .with_store(restored_store)
    .with_compactor(restored_compactor);
```

Pravah records staged history before committing it to runtime history. A store
may observe a successfully written prefix if a later write fails, so stores
should deduplicate retries by stable history position.

Compaction changes the model-visible conversation while preserving independent
agent-loop metrics. Choose a policy that retains any context required by the
application.

## Operational Responsibilities

The application remains responsible for:

- storing snapshots and deciding when to restore them;
- scheduling chat work;
- supplying provider credentials and runtime services;
- making external tool effects idempotent or deduplicated;
- deciding how to retry failed sends.

For a complete runnable conversation, see
[`examples/chat.rs`](../examples/chat.rs).
