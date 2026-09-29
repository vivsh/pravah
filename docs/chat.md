# Chat

`pravah::Chat` provides a typed multi-turn conversation backed by the
same agent and durable workflow facilities as other graph workflows.

Use it when a conversation needs dynamic agent configuration, typed tools,
budgets, application-controlled history persistence, or snapshot restoration.
The older direct-client chat helper is available only through
`pravah::legacy::Chat` for existing applications.

## Build A Chat

Use the builder when the model and instructions are fixed and each message
supplies its own context:

```rust
use pravah::{Chat, ChatRequest, Context};

let mut chat = Chat::builder::<String, String>()
    .model("openai:///gpt-5")
    .instructions("Answer concisely using the available evidence.")
    .turn_budget(6)
    .max_output_tokens(2_000)
    .build(Context::default())?;

let reply = chat.send(
    ChatRequest::from("Compare these approaches.")
        .memory("The user prefers concise technical explanations."),
).await?;
```

Builder chats always retain their conversation between turns. Output can be
any supported structured type, not just `String`. Add `.tools(research_tools)`,
`.tool_budget::<SearchRequest>(2)`, or `.control(control_research)` using the
ordinary [agent APIs](clients.md). Controllers receive `AgentLoop<I>` with the
original typed input, not its JSON rendering or request envelope.
Provider-specific options use `.provider_config(...)`.

Both construction paths have fixed types `Chat<I, O, S>`. Pass `I` directly to
`send` or `send_with_key`, or wrap it in `ChatRequest<I>` for per-invocation options.
String chats also accept `&str`; no `.into()` is needed.

`ChatRequest::from(input)` only moves the input; it performs no serialization and
cannot fail. There is no `ChatRequest::new`. Inputs implement `Serialize`,
`DeserializeOwned`, `JsonSchema`, `Send`, and `Sync` and are owned (`'static`);
neither `Clone` nor `Default` is required.

Builder chats render inputs as JSON text at activation: structs become JSON
objects and strings include JSON quoting and escapes. The original typed input
remains available to controllers and durable execution. To customize the user
message, preserve a configured key, or add attachments, use a function-defined
agent. Rath `Message` is not a special input bypass for a typed chat.

`send` validates explicit tool selections and resource references and converts the
submission before accepting the turn. Such failures leave the chat ready for a
corrected submission. Message rendering, configuration, resource existence and
authorization are checked during activation; failures after acceptance leave an
unfinished turn, not a silently retried chat.

- `.memory(text)` replaces configured memory for this invocation and remains outside message
  history. Omitting it preserves `AgentConfig.memory`, not the previous request's override. Resolved
  memory can still be present in execution snapshots; do not treat it as secret storage.
- `.tools(["search_request"])` selects from declared candidate tools. Omission
  preserves the configured filter (all candidates for a builder); an empty list enables none. Names must be unique and
  known. Tools are exposed in declaration order, subject to controller decisions
  and budgets. Selection cannot add undeclared tools.
- `.resources(refs)` replaces configured resources, including with an empty list.
  MCP registrations and credentials belong in `Context`; see [MCP usage](mcp.md).

Scalar builder setters replace previous values. Invalid/repeated budgets,
duplicate tools, repeated controllers and missing models fail at build or restore.
Pravah's turn budget is not an exact context-token budget; provider output caps
are forwarded through Rath and remain provider-dependent.

Configure everything before the terminal `.build(ctx)?` call:

```rust
let mut chat = Chat::builder::<String, String>()
    .model("openai:///gpt-5")
    .instructions("Answer concisely using the available evidence.")
    .state(initial_state)
    .compactor(working_memory)
    .store(history_store)
    .build(ctx)?;
```

State needs neither `Clone` nor `Default`. Repeated state or service setters
replace previous values; their order does not matter. State moves into the graph's
existing application variable at build, while services remain runtime-only.
Omitting `.state(...)` selects unit state. There is no `build_with_state` method.

Restore synchronously with the same definition. Instructions alone may change:

```rust
let mut restored = Chat::builder::<String, String>()
    .model("openai:///gpt-5")
    .instructions("Answer concisely using the available evidence.")
    .turn_budget(6)
    .max_output_tokens(2_000)
    .restore::<()>(snapshot, restored_ctx)?;
let reply = restored.send("Continue.").await?;
```

For restoration, configure `.store(...)` and `.compactor(...)` on the builder
before `.restore::<State>(snapshot, ctx)`. State comes only from the snapshot;
do not supply `.state(...)` when restoring. Restore is unavailable on builders
carrying non-unit initial state, preventing silent state replacement. Construction and
restore do not call models or read resources. Changed models, budgets, provider options,
default resources, schemas, or tool definitions still reject incompatible snapshots.
New instructions apply to the next invocation whose configuration has not yet committed;
an already-configured invocation keeps its checkpointed instructions until it finishes.
Committed configuration and resolved resources are not evaluated again after restoration.
The new prompt can interpret existing history differently; compatibility does not guarantee
identical agent behavior. Use the manual Chat methods to drive unfinished turns;
`send` and `set` reject them.

Use the same input, output and state types when restoring. Chat snapshots created
before typed request envelopes or the separation of instructions from builder settings
are incompatible; ordinary workflow snapshots are
unaffected. No migration or automatic resubmission is performed.

See the runnable [builder example](../examples/graph_chat_builder.rs), including
keyed messages, state, and JSON restoration.

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

let mut chat = Chat::new(tutor, Context::default())?;
```

`Question` and `Answer` are application types implementing `Serialize`,
`DeserializeOwned`, and `JsonSchema`. Enable `keep_alive` when later calls to
`send` should retain the same model-visible conversation.

The configuration function runs for each new chat input. It may select the
model, instructions, initial user message, memory, tools, resources, and
budgets from the input and `Context`.

Construction is synchronous and fallible: it validates the chat and leaves it
waiting for input. It does not configure the agent, call a model, resolve MCP
resources, or record a message. Snapshots are available immediately.

## Application Message Keys

The `Message` supplied to `AgentConfig` can carry an opaque application key:

```rust
let message = Message::user(question.text).with_key(question.key);
```

This works for Chat and explicit graph workflows. The key is available as
`entry.message.key` to history stores and preparation policies and survives
snapshot restoration. Rath excludes it from provider requests and token counts.
Pravah-generated assistant messages, tool calls, tool results, and summaries
have no application key. A key does not enable deduplication or automatic retry.

Both builder-created and function-defined chats also support
`chat.send_with_key(input, "message-42").await?`. The explicit key overrides
the configured message key for that submission only. Ordinary `send` preserves
the key supplied by function-defined configuration. Builder messages are unkeyed
unless submitted with `send_with_key`.

## Persist Application State

Use `Chat<Input, Output, State>` when application data should travel with the
conversation checkpoint:

```rust
let mut chat = Chat::with_state(tutor, initial_state, ctx)?;

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
`GraphError::ChatSuspended`, not an assistant response. Use Chat's manual
`next`, `resume`, and `resume_fetch` methods when the application owns orchestration;
see [external request delivery](fetch.md).

Client creation and execution failures from either send method retain their Rath
diagnostics. Use `error.client_error()` or match
`GraphError::AgentClient { operation, source }`; see [client error inspection](clients.md#inspect-client-errors).
Async sends record a portable failure in the continuation before returning the
original local error. Restoration preserves the portable diagnostics, not the
original Rust source object, and does not redispatch that completed operation.
Inspecting an error does not make an unfinished Chat ready for another submission.

See the [deterministic Chat builder example](../examples/graph_chat_builder.rs) for a
complete conversation with JSON restoration. Snapshots also support CBOR through Serde.

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
uses Rath's built-in providers, while `Context::with_providers`
installs an application-owned `ProviderRegistry`. A chat uses the same context for
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

`MessageHistory` and the compaction request offer borrowed non-tool message
iteration, JSON byte size and completed-turn counts. See [history inspection and
compaction](history.md) for selection, indexing and external memory integration.

For builder chats, use `.store(history_store).compactor(working_memory)` before
the final `.build(ctx)?` or synchronous `.restore::<State>(snapshot, ctx)`.

Function-defined chats retain their consuming service methods:

```rust
use pravah::{CompactionRequest, Compactor, CompactionResult, HistoryStore};

let chat = Chat::new(tutor, ctx)?
    .with_store(history_store)
    .with_compactor(working_memory);

let restored = Chat::<_, _>::from_snapshot(tutor, snapshot, restored_ctx)?
    .with_store(restored_store)
    .with_compactor(restored_working_memory);
```

The persistence and preparation contracts are part of Pravah's modern root API;
applications do not need to import `pravah::legacy` to implement them.

Pravah records staged history before committing it to runtime history. A store
may observe a successfully written prefix if a later write fails, so stores
should deduplicate retries by stable history position.

Implement `Compactor::compact(&self, request, ctx)` to return
`Result<CompactionResult, YourError>`. The owned `Context` is a shared clone of
the execution-bound context; use `ctx.deps()` to access application services.
Restoration supplies the new execution context to each preparation call, so the
policy does not need to retain its own context. Existing implementations must
accept `ctx: Context` (or `_ctx: Context` when unused) in their `compact` signature.
Its borrowed `CompactionRequest` exposes the resolved model, effective client
options (including the preamble, schemas, provider settings and currently active
tools), and the exact framework guidance accompanying the upcoming request.
Attachment materialization and provider wire transformations happen later;
this view does not provide exact token counts or strict token-budget enforcement.

`committed()` contains prior exchanges eligible for replacement. `protected()`
contains the newest user input and its ongoing tool exchange. The policy can
read both, but replacement indices refer only to `committed()`. Select a sorted,
contiguous prefix `0..n` ending at an exchange boundary. A non-empty `summary`
requires replaced entries; Pravah places it in a tagged system message before
retained exchanges. `CompactionResult::default()` leaves live history intact.

Read existing memory with `request.summary()`, which borrows the original text
without its framework wrapper. Both history and request `enum_messages` helpers
exclude framework summaries and tool messages while preserving original indices.
Use the summary as prior context, not new conversation evidence. Replacement
indices must still cover the full prefix, including the old summary.

Pravah invokes the policy once before each model execution attempt, including
tool-loop redispatch and forced conclusion, and never after the final response.
It validates the whole decision and resulting tool-call/result groups before
changing history. Policy errors propagate as `GraphError::HistoryCompaction`
with the original error as their source; invalid decisions return
`GraphError::HistoryCompactionValidation`. Neither error executes the model or
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
for a small replacement policy. Applications
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
[`examples/graph_chat_builder.rs`](../examples/graph_chat_builder.rs).
