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

## Start a new workflow with completed history

Exact restoration uses the original workflow and its snapshot. To start new
work instead, supply completed working history to a freshly compiled workflow:

```rust
let history = MessageHistory::from_entries(stored_entries);
let mut execution = workflow.start_with_history(input, execution_id, history)?;
```

The runtime takes ownership; no replay or checkpoint migration occurs. Configure
an agent with `.key("customer/42")` to select entries whose `session_id` is
`key:customer/42`, regardless of their originating call site. Unkeyed agents
start fresh sessions. Use a new execution UUID for this new work, just as for any
independent start. Imported entries must have unique IDs, strictly increasing
positions, well-formed summaries, complete tool groups and no unfinished exchange.
Evicted rows are rejected; load only the working rows you intend to retain.
Different session histories may be interleaved by their global append positions.

`from_entries` reconstructs usage only from supplied messages. To retain cumulative
usage after compaction, supply a complete serialized `MessageHistory` instead.
This is not a database loader: the application retrieves rows, selects conversations
and manages external archival retention.

## Compactors

`MessageHistory` remains part of runtime and its snapshot. A runtime-only
`HistoryManager` owns store acknowledgements and policy dependencies—not another
history. It saves original user, assistant, tool-call and tool-result entries;
generated summaries belong to working history, not the append-only audit store.
After restoration, attach a fresh manager. Retained entries may be redelivered,
so the store must deduplicate by entry UUID. Rows already pruned cannot be
recovered from a checkpoint; persist them before pruning.

For manual graph execution:

```rust
let mut manager = HistoryManager::new()
    .with_store(store)
    .with_compactor(policy);
loop {
    manager.maintain(&mut execution, ctx.clone()).await?;
    if let Some(fetch) = execution.pending_fetch() {
        let id = fetch.id();
        let response = executor.execute(fetch).await?;
        execution.resume_fetch(id, Ok(response))?;
    }
    match execution.next()? {
        Step::Continue | Step::Fetch(_) => {}
        Step::Suspend(_) | Step::Done(_) => {
            manager.maintain(&mut execution, ctx.clone()).await?;
            break;
        }
    }
}
```

Retry a failed maintenance call without advancing execution. The manager retains
successful partial acknowledgements; a new manager may replay writes. Compaction
only runs before an unfrozen model request, not against a pending Fetch.

Policies may implement the synchronous trigger method:

```rust
fn needs_compaction(&self, history: &MessageHistory, session_id: &str) -> bool {
    history.turn_count(session_id) >= 8
}
```

The default triggers at each eligible dispatch. `manager.needs_compaction(&execution)`
checks this predicate without constructing a client. It does not check persistence.
Reported usage can inform a heuristic; this does not enforce exact token budgets.

Configure builder chats with `.compactor(policy).store(history_store)` before
the final `.build(ctx)?`, or before synchronous snapshot restoration.
For function-defined chats use `chat.with_compactor(policy)`. For explicit workflows use
`HistoryManager::new().with_compactor(policy)`.
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
once per eligible dispatch in async Chat, never after final output. Explicit
repeated maintenance at the same boundary can run it again; external writes must
be idempotent. Application
errors propagate as `GraphError::HistoryCompaction`; unsafe decisions use
`GraphError::HistoryCompactionValidation`.

For Evimora or another external memory store, make ingestion idempotent using stable
source identities. External writes and Pravah history replacement are not one
transaction. Replacing history does not delete database records or automatically
retrieve saved facts for future prompts. Attach fresh compactors and Context
dependencies after snapshot restoration.

The current history-boundary change rejects old agent continuations; recreate old
executions rather than rewriting fingerprints. The modern names
replace `FlowHistory`, `HistoryPreparer`, `HistoryPreparation`, `HistoryReplacement`
and `with_history_preparer`; the callback is now `compact`, not `prepare`. Legacy
compaction behavior remains separately available under `pravah::legacy`.

See [Chat usage](chat.md#history-persistence-and-working-memory) and the
[runnable compaction example](../examples/graph_chat_working_memory.rs).
