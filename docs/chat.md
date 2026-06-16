# Chat

Read this when you want a simple multi-turn conversation without building a
flow graph. For graph construction, typed agents, tools, and runtime semantics,
start with [clients.md](clients.md) and [flows.md](flows.md). For full API
details, use [docs.rs](https://docs.rs/pravah).

## When To Use Chat

`Chat` is for single-session conversations with one model. It maintains
history automatically, optionally compacts it, and optionally persists it.

Use `Chat` when you want:

- a plain-text or JSON multi-turn conversation
- persistent history without managing it yourself
- snapshot and restore within a single session
- multimodal input (images, files) without a flow graph

Use [`FlowRuntime`](flows.md) instead when you need tool calls, branching,
multiple agents, or human approvals.

## Basic Usage

```rust
use pravah::{Chat, Context, FlowConf};

let ctx = Context::new(FlowConf::default());

let mut chat: Chat = Chat::builder("gemini:///gemini-2.5-flash-lite")
    .preamble("You are a concise Rust tutor.")
    .build()?;

let t1 = chat.send(ctx.clone(), "What is ownership?").await?;
println!("{}", t1.text());

let t2 = chat.send(ctx.clone(), "Give me a short example.").await?;
println!("{}", t2.text());
```

`Chat` defaults to `Chat<String, String>` — plain text in, plain text out.
Token usage is available on each turn when the provider reports it:

```rust
if let Some(usage) = t1.usage {
    println!("in={:?} out={:?}", usage.input, usage.output);
}
```

## Builder Options

| Method               | Effect                                                        |
| -------------------- | ------------------------------------------------------------- |
| `.preamble(text)`    | Static system prompt sent before the conversation history     |
| `.environment(text)` | Additional context appended to the preamble at build time     |
| `.temperature(f)`    | Sampling temperature passed to the provider                   |
| `.session_id(id)`    | Override the auto-generated UUID (useful for durable storage) |
| `.with_compactor(c)` | Attach a history compactor (e.g. `SlidingWindowCompactor`)    |
| `.with_store(s)`     | Attach a history store for persistence                        |
| `.with_memory(f)`    | Attach a memory factory for per-turn dynamic context          |

`.preamble` and `.environment` compose: the final system prompt is
`{preamble}\n\n{environment}` when both are set.

## Typed Input and Output

Substitute `String` with any `Serialize + DeserializeOwned + JsonSchema` type
to switch to JSON mode for that side:

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Question { topic: String }

#[derive(Serialize, Deserialize, JsonSchema)]
struct Answer { summary: String, confidence: f32 }

let mut chat: Chat<Question, Answer> =
    Chat::builder("gemini:///gemini-2.5-flash-lite")
        .preamble("You are a precise Q&A assistant.")
        .build()?;

let turn = chat.send(ctx, Question { topic: "ownership".into() }).await?;
println!("{} ({:.0}%)", turn.output.summary, turn.output.confidence * 100.0);
```

When either side is non-`String`, the builder automatically attaches the
JSON schema to the client options so the provider knows the expected format.

## Multimodal Input

On `Chat<String, Output>` use `send_message` to pass an explicit `Message` with
attachments:

```rust
use pravah::clients::{Message, Attachment};

let msg = Message::user("Describe this image.")
    .with_attachment(Attachment::image_file("diagram.png"));

let turn = chat.send_message(ctx, msg).await?;
println!("{}", turn.text());
```

## History Compaction

Attach a compactor to prevent unbounded history growth:

```rust
use pravah::flows::SlidingWindowCompactor;

let mut chat: Chat = Chat::builder("gemini:///gemini-2.5-flash-lite")
    .with_compactor(SlidingWindowCompactor::new(20))
    .build()?;
```

`SlidingWindowCompactor::new(n)` keeps the most recent `n` message pairs
and evicts older ones after each turn.

## Snapshot and Restore

```rust
// Take a snapshot (clones history and client options; no store needed).
let snap = chat.snapshot();

// Restore. The provider client is recreated; compactor and store revert to
// no-ops and must be re-attached if needed.
let mut restored = Chat::from_snapshot(snap)?;

let t3 = restored.send(ctx, "Summarise our conversation.").await?;
```

`ChatSnapshot` is `Clone` and contains the full `ClientOptions` and
`FlowHistory`, so all provider settings survive the round-trip. It does not
include the compactor or store.

## Durable Persistence

Attach a `HistoryStore` to flush history after each turn:

```rust
let mut chat: Chat = Chat::builder("gemini:///gemini-2.5-flash-lite")
    .session_id("user-123")
    .with_store(my_store)
    .build()?;
```

The store receives the full history after every `send` or `send_message` call.
If the flush fails, the error is returned as `ChatError::Store`; in-memory
history is already updated at that point.

## Memory Injection

Attach a [`MemoryFactory`](https://docs.rs/pravah) to inject dynamic context
into the system prompt before each turn. The retrieved text is prepended as a
transient system message; it is never stored in history and does not affect
compaction or snapshots.

```rust
use pravah::flows::memory::{MemoryFactory, MemoryQuery, MemoryResult};

struct UserProfile { store: ProfileStore }

impl MemoryFactory for UserProfile {
    async fn retrieve(&self, query: &MemoryQuery<'_>) -> MemoryResult {
        let user_id = query.input["user_id"].as_str().ok_or("missing user_id")?;
        let profile = self.store.get(user_id).await?;
        Ok(Some(format!("User profile: {profile}")))
    }
}

let mut chat: Chat = Chat::builder("gemini:///gemini-2.5-flash-lite")
    .preamble("You are a personalised assistant.")
    .with_memory(UserProfile { store })
    .build()?;
```

`MemoryQuery::agent_name` is always `"chat"` for `Chat` sessions.
`MemoryQuery::input` contains the serialized input value — a plain JSON string
for `Chat<String, _>` and a JSON object for typed inputs.

For per-agent routing across multiple agents, use [`MemoryRegistry`](https://docs.rs/pravah)
with [`FlowRuntime::with_memory`](flows.md).

## Model URLs

`Chat` accepts the same model URL format as flow agents:

```text
provider[+transport]://[authority][/prefix/]model[?param=value]
```

Examples: `openai:///gpt-4o`, `anthropic:///claude-sonnet-4-5`,
`gemini:///gemini-2.5-flash-lite`, `ollama://localhost:11434/qwen3:8b`.

See [clients.md — Model URLs](clients.md#model-urls) for the full reference.
