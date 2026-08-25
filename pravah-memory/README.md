# Pravah Memory

[![Crates.io](https://img.shields.io/crates/v/pravah-memory)](https://crates.io/crates/pravah-memory)
[![docs.rs](https://img.shields.io/docsrs/pravah-memory)](https://docs.rs/pravah-memory)
[![License](https://img.shields.io/crates/l/pravah-memory)](LICENSE-MIT)

Pravah Memory turns immutable, application-scoped evidence into concise,
searchable claims with provenance and temporal relationships. It can be used
with [`pravah`](https://crates.io/crates/pravah), another agent runtime, or
directly in an application.

Applications own authorization, evidence normalization, scheduling, and
retention. Pravah Memory owns extraction orchestration, immutable claims,
staleness propagation, reconciliation, and retrieval.

## Installation

Enable PostgreSQL memory support:

```toml
pravah-memory = { version = "0.1.1", features = ["postgres"] }
```

Add `recall-postgres` only when the application also wants durable,
observe-only recall outcomes. It includes `postgres`; ordinary retrieval
does not require the optional telemetry tables.

The default feature set contains backend-neutral memory values, provider
traits, deterministic context assembly, recall receipts, and evaluation
primitives. It does not pull in PostgreSQL, Mool, SQLx, or Tokio.

Pravah Memory accepts normalized text as immutable evidence, calls your
extractor once, embeds the resulting concise claims in a batch, and makes them
searchable before optional background reconciliation finishes.

Register the fixed table family in the application's normal Mool/Gaman schema:

```rust,ignore
use pravah_memory::postgres::{MemoryProfile, MemorySchemaExt};

let profile = MemoryProfile::new(
    "text-embedding-model",
    "2026-08",
    1536,
    "claim-text-v1",
    "memory-extractor-v1",
    "memory-reconciler-v1",
);

let schema = mool::schema()
    // .model::<ApplicationModel>()
    .with_memory(profile)
    .build()?;
# Ok::<(), mool::schema::SchemaLoadError>(())
```

Generate, review, and apply that schema through the application's migration
workflow. `MemoryManager::build()` validates the migrated singleton profile; it
never creates or alters tables at runtime.

Construct the manager with application-selected providers:

```rust,ignore
let manager = MemoryManager::builder(pool)
    .memory_extractor(memory_extractor)
    .embedding_provider(embedding_provider)
    .entity_extractor(entity_extractor) // optional
    .reconciler(reconciler)
    .reranker(reranker) // optional; used only when a search requests it
    .limits(MemoryLimits::default())
    .build()
    .await?;
```

Submit one already-normalized evidence item. The key is idempotent within its
`(user_key, agent_key)` scope:

```rust,ignore
let receipt = manager
    .ingestor(user_id.to_string(), agent_id.to_string())?
    .submit("profile:42:v3", "The user prefers aisle seats.")
    .await?;

assert!(matches!(receipt.processing, ProcessingState::Ready));
```

The same key and content resumes or returns the existing result. Reusing the key
with different content is rejected. Corrected content therefore uses a new,
versioned key.

Run reconciliation and deferred projection maintenance in an application-owned
worker:

```rust,ignore
let completed = manager
    .reconciler(user_id.to_string(), agent_id.to_string())?
    .reconcile_pending(32)
    .await?;

// Usually unnecessary; repairs a projection that exceeded the foreground cap.
let rebuilt_claims = manager
    .reconciler(user_id.to_string(), agent_id.to_string())?
    .refresh_projection(250_000)
    .await?;
```

Claims remain available if reconciliation is delayed or fails. To deliberately
run an ADD-only store, configure `ReconciliationMode::Disabled`; required mode is
the default.

Retrieve concise current claims with lexical, vector, entity, and optional
temporal channels:

```rust,ignore
let memories = manager
    .retriever(user_id.to_string(), agent_id.to_string())?
    .search("What seating does the user prefer?")
    .await?;
```

Use `search_with(SearchRequest)` for explicit entity keys, stale inclusion,
candidate limits, minimum fused relevance, channel weights, and the RRF
constant. Extracted entities may include bounded aliases, so explicit entity
keys can match either the canonical key or a stored alias. Valid time and
transaction time are independent:

```rust,ignore
let request = SearchRequest::new("Where did the user live?")?
    .bitemporal(valid_at, known_at);
let memories = manager
    .retriever(user_key, agent_key)?
    .search_with(request)
    .await?;
```

`as_of(valid_at)` answers with the current relation view at a historical valid
time. `known_at(recorded_at)` excludes claims and relations recorded later.
`history()` exposes all immutable versions. A deterministic query analyzer
recognizes explicit ISO dates and simple relative terms; set the timeline
explicitly when results must be reproducible.

Ordinary search embeds the query and may use the optional fast entity extractor,
but never calls a memory extractor, reconciler, or reranker. Reranking is
explicit and bounded:

```rust,ignore
let request = SearchRequest::new("What matters most to the user?")?
    .rerank(40);
let memories = manager
    .retriever(user_key, agent_key)?
    .search_with(request)
    .await?;
```

Each reranked result retains its database fusion score and adds
`rerank_score`. Provider calls occur after the database transaction has closed.
Unresolved conflict counterparts are retained even when that extends the base
result limit.

## Assemble bounded model context

Tracked search adds a UUIDv7 receipt without changing the returned result order
or performing a telemetry write. Feed its ordinary structured results into the
backend-neutral deterministic assembler:

```rust,ignore
use pravah_memory::context::{ContextAssembler, ContextOptions};

let tracked = manager
    .retriever(user_key, agent_key)?
    .search_tracked("What matters to this user?")
    .await?;

let context = ContextAssembler::compact()
    .assemble(&tracked.results, ContextOptions::default())?;

model_request.system_context = context.rendered.clone();
```

Defaults select at most eight claims within 8,000 Unicode scalar values and
include evidence keys, temporal annotations, and corroboration support counts.
Arbitrary extractor metadata is excluded. Unresolved conflict components are
atomic: all sides are included together or the entire component is omitted.
Claims are never shortened or silently truncated.

Use `ContextBudget::Tokens` only after attaching a tokenizer compatible with
the destination model through `with_token_counter`. Custom renderers receive
structured `ContextGroup` values. The final document—including renderer
prefixes, separators, and suffixes—is measured exactly. Renderer or tokenizer
failure returns an error without partial text.

`AssembledContext` retains selected and omitted memory IDs and structured
omission reasons. Assembly performs no database, embedding, entity, reranking,
or generative call.

## Record optional recall outcomes

Register the optional tables in the application's reviewed migration:

```rust,ignore
use pravah_memory::postgres::{
    MemoryRecallSchemaExt, MemorySchemaExt,
};

let schema = mool::schema()
    .with_memory(memory_profile)
    .with_memory_recall()
    .build()?;
```

Configure a separate store. It starts no scheduler and is never invoked by
retrieval:

```rust,ignore
use chrono::Duration;
use pravah_memory::postgres::RecallStore;

let recall_store = RecallStore::builder(pool.clone())
    .retention(Duration::days(90))
    .use_decay_half_life(Duration::days(30))
    .retrieved_sampling(0.0)
    .max_record_batch(1_024)
    .build()
    .await?;
```

Applications decide what really happened. Assembly itself does not report
acceptance or use:

```rust,ignore
use pravah_memory::RecallBatch;

let recorder = recall_store.recorder(user_key, agent_key)?;
let accepted = RecallBatch::accepted(
    &tracked.receipt,
    context.selected_memory_ids.iter().copied(),
)?;
let used = RecallBatch::used(
    &tracked.receipt,
    context.selected_memory_ids.iter().copied(),
)?;

// Queue this work outside request latency in the application.
recorder.record_many(&[accepted, used]).await?;
```

`accepted` means explicitly selected; `used` means actually consumed.
Non-reporting is not dismissal. `Corrected` requires the ID of separately
accepted, same-scope corrective evidence and never mutates an existing claim or
relation. Duplicate reports for the same receipt, memory, and event kind are
ignored idempotently.

Durable `Retrieved` events are disabled by default. To sample them, configure a
probability and construct the batch through `recorder.retrieved(&receipt)`.
Search tracing still records aggregate result counts when durable sampling is
zero. No query, prompt, claim, or evidence text is stored in recall telemetry.

Run bounded maintenance from an application-owned worker:

```rust,ignore
recall_store.aggregate_pending(5_000).await?;
recall_store.prune_expired(10_000).await?;

// Operational repair: rebuild one scope exactly from retained events.
recall_store.rebuild_scope_stats(user_key, agent_key).await?;
```

A common cadence is to flush every 30 seconds or 5,000 queued events and run
retention daily. The exact rebuild takes an exclusive recall-event table lock,
so reserve it for an operational maintenance window. Telemetry outages affect
only recorder and maintenance calls; search and context assembly continue
normally. Recall statistics are analytics only and do not influence ranking in
this release.

Evidence lifecycle operations remain scope-bound:

```rust,ignore
let ingestor = manager.ingestor(user_key, agent_key)?;
let evidence = ingestor.get("document:17:chunk:4:v2").await?;
ingestor.retry("document:17:chunk:4:v2").await?;
ingestor.mark_stale("document:17:chunk:4:v2").await?;
ingestor.delete("document:17:chunk:4:v2").await?;
```

Staleness is one-way and immediately propagates to direct claims. Deletion is
always explicit. Pravah does not accept preformed memories, parse chat messages,
or split oversized documents; construct separately keyed evidence in the
application.

Concurrent submissions of the same scoped evidence key have one processing
owner. Other callers receive a receipt whose `processing` value is
`Processing`; they do not duplicate provider work. Processing and reconciliation
leases, batch sizes, reconciliation candidates, and foreground projection work
are bounded through `MemoryLimits`.

Relation hydration is separately bounded by
`MemoryLimits::max_retrieval_relation_edges`. Exceeding it returns
`MemoryManagerError::RelationExpansionLimit` instead of a partial
corroboration, supersession, or conflict view.

See `examples/postgres.rs` for a complete provider and manager wiring
example:

```text
cargo run -p pravah-memory --example postgres --features postgres
```

Add `recall-postgres` to the feature list to run the detached recall-outcome
flush as well.
