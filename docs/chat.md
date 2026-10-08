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
fn tutor(root: Agent<Question>) -> Agent<Answer> {
    root
        .model("openai:///gpt-5")
        .instructions("You are a concise Rust tutor.")
        .build()
}

let mut chat = Chat::new(tutor, Context::default())?;
```

This uses the same declaration settings as ChatBuilder, with JSON-rendered input.
For plain-text messages, attachments or invocation-dependent settings, use custom
configuration instead:

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
    ))
}

let mut chat = Chat::new(tutor, Context::default())?;
```

`Question` and `Answer` are application types implementing `Serialize`,
`DeserializeOwned`, and `JsonSchema`. Chat automatically retains the same
model-visible conversation across sends. By default its conversation key is
derived from the Chat's execution UUID and survives restoration. To choose an
application identity, use `.key("customer/42")` on a builder or `AgentConfig`.
These conversation keys are separate from per-message keys.

The configuration function runs for each new chat input. It may select the
model, instructions, initial user message, memory, tools, resources, and
budgets from the input and `Context`.

The shared declaration settings schema changed with declarative Agent support.
Builder-created Chat snapshots from before that change have incompatible graph
fingerprints and are rejected without migration. Snapshot formats are unchanged.

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
It is independent of conversation keys, history summaries, and `AgentConfig::memory`.
Agents and tools cannot implicitly read or mutate it. Include relevant fields
explicitly in a message input when the model needs them; state is not
automatically placed in prompts or conversation history.

`set` is allowed before the first message and between completed turns. During
unfinished execution, `get` and `snapshot` remain available but `set` and `send`
return `GraphError::ChatNotReady`. There is no automatic retry or in-flight retry
method. If an agent controller or child tool suspends, `send` returns
`GraphError::ChatSuspended`, not an assistant response. Use Chat's manual
`submit`, `next`, `resume`, and `resume_agent` methods when the application owns orchestration;
see [agent worker delivery](agent_execution.md).

Client creation and execution failures from either send method retain portable Rath
diagnostics. Use `error.agent_error()` or match `GraphError::AgentFailed { source }`;
see [client error inspection](clients.md#inspect-client-errors).
Async sends accept the same portable completion used by external workers.
Restoration preserves its diagnostics, not the original Rust source object,
and does not redispatch that completed operation.
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
    .turn_budget(6)
    .tool_budget::<FindAccount>(2))
}
```

See [clients.md](clients.md) for model URLs, tools, dynamic filters, memory,
attachments, and agent configuration.

### Catalogue-defined JSON tools

Use `Toolset::json` when a catalogue supplies tool definitions at runtime. Both
`Agent::tools` and `ChatBuilder::tools` execute their `FnOnce` builder immediately,
so it can capture fetched definitions. Each registered handler is a reusable,
capturing `Fn(serde_json::Value, Context)` returning an asynchronous
`Result<serde_json::Value, ToolError>`.

```rust
let mut chat = Chat::builder::<String, String>()
    .model("openai:///gpt-5")
    .instructions("Assist staff")
    .tools(move |tools| {
        tools.json(definition, Some(output_schema), move |args, ctx| {
            proxy(operation.clone(), args, ctx)
        })
    })
    .tool_budget_named("find_staff_member", 2)
    .build(ctx)?;
```

`definition` is Rath's `ToolDefinition`, with the model-facing alias, description
and canonical input schema. `operation` is an application-owned trusted binding;
it stays outside model-controlled arguments. The proxy uses `Context` to obtain
the application's service, current-user credentials and permissions. The same
proxy can serve many aliases without Rust request/response types per endpoint.
See the runnable [JSON tools example](../examples/graph_json_tools.rs).

Input requires an explicit root `type: "object"`. Schemas use a fixed Draft
2020-12 profile, with its standard dialect URI optionally declared at the root.
Supported assertions include required fields, nullable type unions, enums, numeric
and length bounds, additional/schema-valued properties, composition, conditionals
and unevaluated properties/items. Local `$defs` and acyclic same-document JSON
Pointer `$ref`s are supported, including `~0`/`~1` escapes and assertion siblings.
Unsupported dialects, unknown keywords/formats, remote/file references, recursion,
anchors, `$id` rebasing, custom vocabularies and content keywords fail construction.
No schema retrieval occurs. Descriptive annotations remain unchanged; defaults
are never inserted, and `readOnly`/`writeOnly` do not enforce permissions.

Formats are asserted: `date`, `date-time`, `time`, `duration`, `email`, `idn-email`,
`hostname`, `idn-hostname`, `ipv4`, `ipv6`, `uri`, `uri-reference`, `uri-template`,
`iri`, `iri-reference`, `uuid`, `regex`, `json-pointer` and `relative-json-pointer`.
Arguments and results are validated without coercion, filtering extra keys,
inventing null fields, or reparsing successful JSON strings. Values retain the
existing VM numeric limits: finite f64 and signed/unsigned 64-bit integers.

The output schema applies only to successful tool values. `None` explicitly
permits any JSON success; `Some(false)` rejects every success. `ToolError`
envelopes bypass this schema and retain their correctable/fatal classification.
Invalid arguments produce a correctable validation error before the handler runs
and consume the existing alias budget. Invalid successful outputs are fatal and
remain accepted failures across restore; the handler is not replayed automatically.
Named budgets work in `AgentConfig` too, sharing identity with typed budgets.
Duplicate names, reserved `__rath_final_output`, zero/unknown/duplicate budgets and
invalid schemas fail at the applicable build/activation boundary.

Pravah passes the canonical input schema to Rath and enforces it locally. Rath's
provider-specific presentation can impose stricter requirements or lose assertions;
this API does not guarantee equivalent provider support for every admitted schema.
Tool output schemas do not become the model's final-answer schema.

Persist the exact catalogue revision and compatible operation bindings alongside
the separately retained graph. Changing aliases, descriptions, order, or either
schema changes its fingerprint and rejects exact snapshot continuation. Restoring
uses a newly supplied `Context`, so handlers must check current permissions even
for checkpointed calls. Imported conversation history can start a new execution
with a new catalogue; it does not resume the old one. Detached worker hosts must
select the matching registry because a tool request alone carries no deployment
identity. HTTP routing, authentication, storage, retries and external-effect
idempotency remain application responsibilities.

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
`Chat::<_, _>` selects unit state. Older keep-alive snapshots are rejected: the
current API uses keyed conversations rather than per-node saved sessions. Retain
the matching older runtime to finish them; there is no automatic migration.

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

The synchronous runtime commits accepted messages as execution progresses.
Builder `.store(...)` enables persistence and loading; `.compactor(...)` enables
working-memory preparation. Their dependencies live only in the agent executor.
Original entries are persisted before replacement and final output is flushed
before a turn completes. Stores must deduplicate redelivery by stable entry ID.
Successful write acknowledgements survive later failures and restoration.

Manual `submit/next/resume_agent` and async `send` use the same worker operations;
neither requires a separate maintenance call. A store failure after generation
retains the recorded model result and never causes automatic regeneration.
Reinstall required services when restoring. Without a store or compactor,
messages remain in snapshots and history grows without trimming.

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

Async Chat invokes the policy once before each eligible model dispatch, including
tool-loop redispatch and forced conclusion, and never after the final response.
It validates the whole decision and resulting tool-call/result groups before
changing history. Worker failures become `GraphError::AgentFailed` with portable
diagnostics. A preparation failure prevents model execution and leaves working
history unchanged, although earlier successful store acknowledgements are retained.
The pending user message has already been recorded by then. Delivery and processing
are separate; once a failure is accepted, stepping does not redispatch its operation.
There is no in-flight retry operation in `Chat::send()`; another `send()` returns
`GraphError::ChatNotReady`. Restoring an unfinished Chat preserves that limitation;
it does not automatically resume the failed dispatch.

Successful replacement physically removes old rows from snapshots while
preserving retained row identities, positions, and accumulated usage. Audit
stores still receive original appended messages; preparation does not send
summary rows or delete commands to the store. Snapshots remain the source for
restoring prepared working memory. Repeated consolidation can bound retained
past exchanges, but the protected current exchange and summary text still take
space. Applications choose what knowledge to preserve and how large a summary
may become. A later provider failure does not undo successful preparation or
persistence. Policies and stores should tolerate host redelivery of unacknowledged work.

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

## Release Working History

Between completed turns, `chat.conversation_is_active(session_id)` reports false
and `chat.drop_conversation(session_id)?` releases that working conversation.
Use the exact `HistoryEntry::session_id` (for example `key:customer/42`). Release
does not delete the store's archive; the next keyed turn can reload it, or starts
fresh without a store. Active turns reject removal without mutation. See
[conversation lifetime](history.md#conversation-lifetime-and-release).

## Operational Responsibilities

The application remains responsible for:

- storing snapshots and deciding when to restore them;
- scheduling chat work;
- supplying provider credentials and runtime services;
- making external tool effects idempotent or deduplicated;
- deciding how to retry failed sends.

For a complete runnable conversation, see
[`examples/graph_chat_builder.rs`](../examples/graph_chat_builder.rs).
