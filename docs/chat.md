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

let mut chat = Chat::new(tutor, Context::default()).await?;
```

`Question` and `Answer` are application types implementing `Serialize`,
`DeserializeOwned`, and `JsonSchema`. Enable `keep_alive` when later calls to
`send` should retain the same model-visible conversation.

The configuration function runs for each new chat input. It may select the
model, instructions, initial user message, memory, tools, resources, and
budgets from the input and `Context`.

Construction is asynchronous and fallible: it validates the chat and leaves it
waiting for input. It does not configure the agent, call a model, resolve MCP
resources, or record a message. Snapshots are available immediately.

## Persist Application State

Use `Chat<Input, Output, State>` when application data should travel with the
conversation checkpoint:

```rust
let mut chat = Chat::with_state(tutor, initial_state, ctx).await?;

let mut state = chat.get()?;
state.selected_project = Some(project_id);
chat.set(state)?;

let reply = chat.send(question).await?;
let snapshot = chat.snapshot()?;

let mut restored = Chat::<Question, Answer, SessionState>::from_snapshot(
    tutor, snapshot, restored_ctx,
)?;
let state = restored.get()?;
```

State implements `Serialize`, `DeserializeOwned`, `JsonSchema`, `Send`, and
`Sync`, with no borrowed non-static data. Neither `Default` nor `Clone` is
required. `get` decodes an owned value and can allocate; `set` converts and
validates the replacement before committing. A failed update leaves the
previous state unchanged. Only explicit state access performs typed conversion.

State is included in the same `Snapshot` as execution and conversation history;
no separate state record is needed. Persist the complete snapshot together.
It is independent of `keep_alive`, history summaries, and `AgentConfig::memory`.
Agents and tools cannot implicitly read or mutate it. Include relevant fields
explicitly in a message input when the model needs them; state is not
automatically placed in prompts or conversation history.

`set` is allowed before the first message and between completed turns. During
unfinished execution, `get` and `snapshot` remain available but `set` and `send`
return `GraphError::ChatNotReady`. There is no automatic retry or in-flight retry
method. If an agent controller or child tool suspends, `send` returns
`GraphError::ChatSuspended`, not an assistant response. Use the explicit workflow
runtime when the application needs arbitrary suspension/resumption or retry control.

See the [deterministic state example](../examples/graph_chat_state.rs) for a
complete conversation with JSON and CBOR restoration.

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

Take a snapshot before the first message or between completed turns and persist
it with the application's own storage:

```rust
let snapshot = chat.snapshot()?;

let mut restored = Chat::<_, _>::from_snapshot(tutor, snapshot, restored_ctx)?;
let next = restored
    .send(Question::new("Continue our discussion."))
    .await?;
```

Restoration requires the same agent definition. Live provider clients and
application services are not serialized; bind them through the restoration
`Context` or the service setters before continuing.

For stateful chats, specify the same state type during restoration. Two-argument
`Chat::<_, _>` selects unit state. Snapshot formats are unchanged, but snapshots
from the older lazy-construction chat graph have a different fingerprint and
cannot be restored into this API. Drain those chats or retain their matching runtime.

## History Persistence And Working Memory

Attach an application history store or preparation policy when constructing or
restoring a chat:

```rust
use pravah::{HistoryPreparation, HistoryPreparer, HistoryReplacement, HistoryStore};

let chat = Chat::new(tutor, ctx).await?
    .with_store(history_store)
    .with_history_preparer(working_memory);

let restored = Chat::<_, _>::from_snapshot(tutor, snapshot, restored_ctx)?
    .with_store(restored_store)
    .with_history_preparer(restored_working_memory);
```

The persistence and preparation contracts are part of Pravah's modern root API;
applications do not need to import `pravah::legacy` to implement them.

Pravah records staged history before committing it to runtime history. A store
may observe a successfully written prefix if a later write fails, so stores
should deduplicate retries by stable history position.

Implement `HistoryPreparer::prepare(&self, request, ctx)` to return
`Result<HistoryReplacement, YourError>`. The owned `Context` is a shared clone of
the execution-bound context; use `ctx.deps()` to access application services.
Restoration supplies the new execution context to each preparation call, so the
policy does not need to retain its own context. Existing implementations must
add `ctx: Context` (or `_ctx: Context` when unused) to their `prepare` signature.
Its borrowed `HistoryPreparation` exposes the resolved model, effective client
options (including the preamble, schemas, provider settings and currently active
tools), and the exact framework guidance accompanying the upcoming request.
Attachment materialization and provider wire transformations happen later;
this view does not provide exact token counts or strict token-budget enforcement.

`committed()` contains prior exchanges eligible for replacement. `protected()`
contains the newest user input and its ongoing tool exchange. The policy can
read both, but replacement indices refer only to `committed()`. Select a sorted,
contiguous prefix `0..n` ending at an exchange boundary. A non-empty `summary`
requires replaced entries; Pravah places it in a tagged system message before
retained exchanges. `HistoryReplacement::default()` leaves live history intact.

Pravah invokes the policy once before each model execution attempt, including
tool-loop redispatch and forced conclusion, and never after the final response.
It validates the whole decision and resulting tool-call/result groups before
changing history. Policy errors propagate as `GraphError::HistoryPreparation`
with the original error as their source; invalid decisions return
`GraphError::HistoryPreparationValidation`. Neither error executes the model or
changes the history/checkpoint present before preparation. The pending user
message has already been recorded by then. For application-controlled retries,
drive the workflow with `Runtime::next()` and retry that step, or restore its
snapshot with fresh dependencies. This change does not add an in-flight retry
operation to `Chat::send()`; calling `send()` again on an unfinished turn returns
`GraphError::ChatNotReady`. Restoring an unfinished Chat preserves that limitation;
it does not automatically resume the failed dispatch.

Successful replacement physically removes old rows from snapshots while
preserving retained row identities, positions, and accumulated usage. Audit
stores still receive original appended messages; preparation does not send
summary rows or delete commands to the store. Snapshots remain the source for
restoring prepared working memory. Repeated consolidation can bound retained
past exchanges, but the protected current exchange and summary text still take
space. Applications choose what knowledge to preserve and how large a summary
may become. A later attachment or provider failure does not undo a successful
preparation; policies should tolerate retries.

Policies and their dependencies are never serialized. Reattach a fresh policy
after restore, as above. Without a policy, ordinary chat retains its history.
If a policy saves extracted facts externally before evicting history, make
those writes idempotent using stable source-entry identities. External writes
and history replacement are not one transaction: cancellation or later failure
can leave facts persisted without eviction. Saved facts are not automatically
added to model context; retrieve them through configuration or a replacement
summary when needed.
`AgentConfig::memory` remains separate invocation memory; preparation does not
rewrite it. Legacy compaction remains under `pravah::legacy`.

See [the runnable working-memory example](../examples/graph_chat_working_memory.rs)
for a fallible policy and restoration with fresh dependencies. Applications
provide the summarizer, storage, retry handling, and any model-specific size
estimation. Configure a separate per-request output cap with
[`AgentConfig::max_output_tokens`](clients.md#limit-generated-output); history
preparation does not itself enforce that cap or an input-token budget.

## Operational Responsibilities

The application remains responsible for:

- storing snapshots and deciding when to restore them;
- scheduling chat work;
- supplying provider credentials and runtime services;
- making external tool effects idempotent or deduplicated;
- deciding how to retry failed sends.

For a complete runnable conversation, see
[`examples/chat.rs`](../examples/chat.rs).
