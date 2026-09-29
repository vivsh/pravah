# Message history and compaction

`MessageHistory` contains messages, stable history-entry identities and cumulative
usage for every agent session in an execution. Obtain it from `snapshot.history()`.
Choose a session explicitly when inspecting the full history:

```rust
let history = snapshot.history();
let bytes = history.byte_size(session_id)?;
let turns = history.turn_count(session_id);

for (index, message) in history.enum_messages(session_id, 6) {
    println!("{index}: {:?}: {}", message.key, message.content);
}
```

`enum_messages` borrows Rath messages oldest-first. It excludes evicted entries,
other sessions, framework summaries, tool-call proposals and tool results. User
messages, final assistant messages and other system messages remain visible.
`skip_recent` counts only exposed messages; zero skips none, and a count larger
than the history yields an empty iterator. No messages are cloned or collected
by the helper.

Indices refer to the original live session sequence **including summaries and
tool entries**, so they may have gaps. They are not stable identities: compaction
can change indices.
For durable correlation use `message.key` or the corresponding `HistoryEntry.id`.
Keys are application-owned and carry no uniqueness guarantee.

`byte_size` measures the exact compact JSON array encoding of **all** live messages
in the selected session, including summaries, tools, keys, usage and serialized attachments.
It excludes HistoryEntry metadata and does not open file attachments or fetch URLs.
An empty session measures two bytes (`[]`). It serializes into a byte counter, not
a buffer; computation is on demand and can fail. This is neither token count nor
provider request size.

`turn_count` counts user exchanges ending in a final assistant message. Intermediate
tool rounds, standalone assistant messages, system summaries and pending user input
do not add completed turns. It does not validate malformed histories and does not
alter cumulative usage or agent-budget metrics.

## Compactors

Configure builder chats with `.compactor(policy).store(history_store)` before
the final `.build(ctx)?`, or before synchronous snapshot restoration.
For function-defined chats and explicit workflows, install a fallible `Compactor` with `chat.with_compactor(policy)`,
`executor.with_compactor(policy)` or `RuntimeServices::with_compactor(policy)`.
The trait method is:

```rust
async fn compact(
    &self,
    request: CompactionRequest<'_>,
    ctx: Context,
) -> Result<CompactionResult, Self::Error>;
```

`CompactionRequest` exposes the session, effective model/options, framework guidance,
completed `committed()` history and the unmodifiable current `protected()` exchange.
Its helpers already select **committed history only**:

```rust
let bytes = request.byte_size()?;
let turns = request.turn_count();
let previous_summary: Option<&str> = request.summary();

for (index, message) in request.enum_messages(6) {
    let entry = &request.committed()[index];
    // Pass the borrowed message and stable entry.id/message.key to your fact store.
    // Access that application service through ctx.
}
```

The current user input and its entire ongoing tool exchange are always excluded
from these helpers. Raw `committed()` and `protected()` entries remain available
when a policy explicitly needs tool evidence or metadata.

`summary()` borrows the existing summary text without the framework wrapper,
preserving its whitespace. It returns `None` when the first committed entry is
not a recognized summary or its representation is malformed. Reading it changes
nothing and allocates nothing. Supply it as prior context to your summarizer or
fact extractor, separately from the new messages returned by `enum_messages`.
The summary still reaches the model and contributes to `byte_size()`.

Enumeration is for reading, not an eviction plan. **Do not collect its potentially
gapped indices into `evict_indices`.** Replacement must select a complete, contiguous
prefix of committed entries, including the existing summary and any tool messages
in that prefix:

```rust
// After successfully processing all committed exchanges:
Ok(CompactionResult {
    evict_indices: (0..request.committed().len()).collect(),
    summary: None, // Or non-empty replacement memory text.
})
```

A non-empty summary requires replaced entries. Invalid prefixes, incomplete tool
groups and attempts to alter protected input fail atomically. The compactor runs
once before each model execution attempt, never after final output. Application
errors propagate as `GraphError::HistoryCompaction`; unsafe decisions use
`GraphError::HistoryCompactionValidation`.

For Evimora or another external memory store, make ingestion idempotent using stable
source identities. External writes and Pravah history replacement are not one
transaction. Replacing history does not delete database records or automatically
retrieve saved facts for future prompts. Attach fresh compactors and Context
dependencies after snapshot restoration.

This naming change adds no snapshot format change or migration. The modern names
replace `FlowHistory`, `HistoryPreparer`, `HistoryPreparation`, `HistoryReplacement`
and `with_history_preparer`; the callback is now `compact`, not `prepare`. Legacy
compaction behavior remains separately available under `pravah::legacy`.

See [Chat usage](chat.md#history-persistence-and-working-memory) and the
[runnable compaction example](../examples/graph_chat_working_memory.rs).
