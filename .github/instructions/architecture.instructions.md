---
applyTo: "src/flows/**/*.rs"
---

# Pravah Engine — Internal Architecture

This document describes the runtime internals. For the user-facing API (traits, builder, model URLs) see `ARCHITECTURE.md` at the repository root. This document is the authoritative reference for contributors working on the engine.

---

## Module Map

```
src/flows/
  mod.rs          — public re-exports only; no logic
  flows.rs        — FlowGraph, FlowBuilder, node handlers, step loop
  runtime.rs      — FlowRuntime<I>; thin orchestrator over FlowGraph
  nary.rs         — SplitOutputs, MergeInputs traits; macro impls for arities 2–16
  state.rs        — FlowState; frame stack, suspension, call_enter/call_exit
  phase.rs        — Phase enum (Entry | Continue(Option<Value>))
  errors.rs       — FlowError (public), BuildError (internal)
  history.rs      — FlowHistory + HistoryEntry; conversation store + validation
  compactor.rs    — HistoryCompactor trait, NoopCompactor, SlidingWindowCompactor
  store.rs        — HistoryStore trait, NoopHistoryStore
  interner.rs     — Interner; string ↔ NodeId bijection
  validation.rs   — validate_nodes (per-node), validate (reachability)
  diagram.rs      — Mermaid diagram generation; no runtime logic
```

`src/clients/mod.rs` contains `Message` — a **wire-only** type carrying `role`, `content`, and optional `usage`. It has no agent or session metadata. All pravah-internal metadata lives on `HistoryEntry` in `history.rs`.

---

## NodeId and the Interner

Every node and state slot is identified by a `NodeId(usize)` — a dense integer index into the graph's `Interner`. The interner holds a bidirectional map between `&str` (schema name) and `NodeId`.

`NodeId` values are **graph-local**: two different `FlowGraph` instances (e.g. root and a sub-flow) have independent interners. This matters for sub-flow nodes, which carry their own `Arc<FlowGraph>` with its own interner.

Tool node keys are interned as `"{agent_name}::{tool_name}"` to avoid collisions with other node keys in the same frame.

---

## FlowGraph: Nodes and Edges

`FlowGraph.nodes: HashMap<NodeId, FlowNode>` stores every node. `FlowNode` variants:

| Variant   | Dispatch model                               | State transition                                                         |
| --------- | -------------------------------------------- | ------------------------------------------------------------------------ |
| `Work`    | async closure                                | removes `name`, inserts `exit_name`                                      |
| `Agent`   | multi-step (see below)                       | see Agent Dispatch                                                       |
| `Tool`    | async, driven by parent agent                | removes `name`, or exits agent frame                                     |
| `Map`     | infallible sync `fn(&Value)->Value`          | removes `name`, inserts `exit_name`                                      |
| `Either`  | infallible sync `fn(&Value)->(NodeId,Value)` | removes `name`, inserts one of `left_name`/`right_name` (cycles allowed) |
| `Fork`    | infallible sync `fn(&Value)->Vec<StateNode>` | removes `name`, inserts all children                                     |
| `Join`    | infallible sync `fn(&[Value])->StateNode`    | fires only when all parents present; removes parents, inserts `target`   |
| `Suspend` | no closure — pauses the frame                | records `Suspension{src,dst}`; returns `FlowStep::Suspend(sv)`           |
| `Flow`    | pushes child frame                           | transfers entry value via `call_enter`                                   |

`Fork` and `Join` serve both the binary `fork`/`join` builder methods and the N-ary `split`/`merge` methods. `split` produces a `Fork` node whose `children` vec has N entries (arity 2–16); `merge` produces N `Join` nodes (one per parent key) that share the same `Arc`-ed shim and `target`. No new `FlowNode` variants are needed.

`Map`, `Either`, `Fork`, and `Join` are **pure algebra nodes**: their shims are `fn(&Value) -> …` with no `Context` argument and no `Result` wrapper beyond serialization errors. Any handler returning a user error must use `Work` or `Agent` instead.

Edges are **implicit**: each node writes the key its successor reads. There are no explicit edge structures. The graph supports cycles — an `Either` node can write a key that was already consumed earlier in the same flow, enabling retry loops and multi-turn conversations. Validation checks reachability and type contracts but does not forbid cycles.

---

## FlowState: The Frame Stack

`FlowState` is a `Vec<Frame>` where the top frame is the active one.

### Frame

```
Frame {
    states: IndexMap<NodeId, Value>,   // ordered; iteration order drives dispatch
    phase:  Option<Phase>,             // Some only while an agent node is active
    callable: Callable,                // wiring for call_enter/call_exit
}
```

`Callable` records five NodeId slots:

```
parent_entry  — key of the entry value in the parent frame (removed on call_enter)
parent_exit   — key where the result is written in the parent frame (written on call_exit)
entry         — key where the entry value lands in the child frame
exit          — key whose presence signals this frame is done
index         — index into FlowRuntime.callables for the FlowGraph serving this frame
```

### call_enter

Pushes a new child frame. The entry value is moved from the parent frame (`parent_entry` key) into the child frame (`entry` key). For the root frame there is no parent; the entry value is seeded directly by `FlowRuntime::new`.

### call_exit

Called at the end of every `FlowGraph::step` that returns `Continue`. Loops `handle_exit`:

- If the top frame's `exit` key is present in its state map, pops the frame.
- Transfers the exit value to the parent frame under `parent_exit`.
- If no parent exists (root frame popped), returns `Some(value)` → `FlowStep::Done`.
- If the frame cannot exit, loop breaks and `None` is returned.

**One `call_exit` call can cascade through multiple frames** (e.g. agent frame completes → sub-flow frame also ready to exit → root done in one shot).

`call_exit` is **never called** when `step` returns `Suspend` — the frame is frozen mid-step and must not be popped.

### Suspension

There are two suspension sources:

**Tool-level** (`ToolError::Suspend` from `handle_tool`)

`FlowState.suspension: Option<Suspension>` stores `{ src: NodeId, dst: NodeId }`. When a tool suspends:

- `src = tool node key` — holds the original `ToolCall` JSON for history reconstruction on resume
- `dst = agent node key` — where the resumed value lands, triggering `handle_child_agent`

**Flow-level** (`FlowNode::Suspend` from `handle_suspend`)

When `step_inner` hits a `Suspend` node:

1. Reads the input value from `info.entry`.
2. Deserializes it into a `SuspendedValue` via the baked-in `deserialize` closure.
3. Calls `states.suspend(info.entry, info.exit, info.output_type)` — records `Suspension { src: entry, dst: exit }`.
4. Returns `FlowStep::Suspend(sv)`.

In both cases `states.resume(value)` removes `src`, inserts `dst = value`, clears suspension, and execution continues from the exit node.

---

## Frame Lifecycle and Nesting

The `Vec<Frame>` is the execution call stack. Frame depth at any moment equals the number of active callables on the stack.

### When frames are pushed

| Trigger                                                  | Pusher                                          | `callable.index` set to                                                    |
| -------------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------- |
| `FlowRuntime::new` seeds the root frame                  | direct `state.call_enter` in `FlowRuntime::new` | `root_callable_index` (root `FlowGraph`)                                   |
| `step_inner` hits a `FlowNode::Flow` node                | `states.call_enter` in `step_inner`             | `inner.callable_index` (the inner `FlowGraph`)                             |
| `step_inner` hits a `FlowNode::Agent` node with no phase | `states.call_enter` in `handle_parent_agent`    | `flow.callable_index` — the **same** graph as the parent frame (see below) |

### When frames are popped

`call_exit` in `step` pops the top frame whenever `frame.callable.exit` is present in the state map. Popping is always triggered from `step` after `step_inner` returns `Continue`. It never fires on a `Suspend` return — the frame is frozen.

### Agent frames share the parent FlowGraph

An agent frame's `callable.index` is set to `flow.callable_index` — the index of the **currently active** `FlowGraph`, not a new one. This is intentional: `FlowNode::Tool` entries for the agent's tools are registered in the parent graph's `nodes` map. The engine uses the same graph to dispatch both the parent-frame work and the agent's tool calls.

The two frames are distinguished solely by `Phase`:

```
root frame   : phase = None       → step_inner dispatches normal nodes
agent frame  : phase = Some(...)  → step_inner enters handle_child_agent
```

When `callable_index()` returns the same index for both frames, `FlowRuntime` resolves to the same `Arc<FlowGraph>` and `step_inner` uses `states.phase()` to decide which path to take.

### Frame depth examples

```
Root flow only                   depth 1
Root → sub-flow                  depth 2
Root → agent                     depth 2
Root → sub-flow → agent          depth 3
Flow complete (stack empty)      depth 0  →  FlowError::Internal on next next()
```

### Fork and Join run in the same frame

`fork` writes multiple keys into the **same** top frame. There is no separate frame per branch and no parallel execution. Both branch keys must converge in the same frame's state map before `join` fires. `step_inner` dispatches one key per call to `next()`; the two branches advance in strict alternation (whichever key appears earlier in the `IndexMap`).

---

## Recursion

**Recursive flows are not supported.** Two independent mechanisms prevent them.

### 1 — Rust type system

`F::build()` must return a `FlowGraph` containing concrete `FlowNode` variants. To embed a sub-flow you write `.flow::<G>()`, which requires `G: Flow` as a concrete type. A type cannot name itself in `build()` without introducing an infinite-size type, which the compiler rejects. Box indirection through `dyn Flow` is not exposed by the builder API.

### 2 — `collect_callables` requires exclusive Arc ownership

At construction time, `collect_callables` calls `Arc::get_mut(inner)` on every `FlowNode::Flow` node to write a `callable_index`. `Arc::get_mut` returns `None` if any other strong reference exists. Consequences:

- The same sub-flow `Arc<FlowGraph>` cannot appear more than once in a graph (even in two branches of a `fork`) — the second occurrence causes `FlowError::Internal`.
- A graph cannot contain itself: cloning the arc and embedding it would leave two strong references, failing `get_mut` immediately.
- Mutually recursive graphs (A contains B contains A) are also blocked: `collect_callables` would diverge before reaching the `Arc::get_mut` check.

### What this means for callers

Each sub-flow type may appear **at most once** in any given graph. To re-use the same processing logic in multiple places, define a distinct wrapper type for each site, or lift the shared logic into a `work` node.

---

## step and step_inner

`FlowGraph::step` is the single entry point for both `next()` and `resume()`. It:

1. Guards against `ResumeRequired` / `UnexpectedResumption` mismatches.
2. On resume: reconstructs the `ToolCall` from the suspended `src` slot, pushes a `Role::Tool` history entry, then calls `states.resume()`.
3. Calls `step_inner`, which dispatches exactly one node per call.
4. After `step_inner` returns `Continue`, calls `states.call_exit()`. If `Some(v)` is returned, wraps it in `FlowStep::Done(v)`.

`FlowGraph::step_inner` iterates the top frame's state map in insertion order. For each entry:

- If the key has no corresponding node in `self.nodes`, it is a terminal state slot — `continue`.
- If a node is found, dispatch it and `return` immediately.

This means **only one node fires per call to `next()`**.

`step_inner` does not call `call_exit` — that is `step`'s responsibility. This separation ensures `call_exit` never runs on a suspended frame.

---

## Agent Dispatch: The Two-Frame Model

An agent node runs across **two frames**: one in the parent graph, one in the child graph (the agent's callable).

### Parent frame — `handle_parent_agent`

Fires when `step_inner` encounters an `Agent` node with **no active phase** in the top frame. It:

1. Builds a `Callable` pointing the child exit back to `node.exit` in the parent frame.
2. Calls `call_enter` to push the agent frame.
3. Sets `Phase::Entry` on the new top frame.

Returns `Continue`. `step_inner` then returns. On the **next** `next()` call, `callable_index()` now points to the agent graph, and `step_inner` re-enters with `Phase::Entry`.

### Child frame — `handle_child_agent`

Fires on every `next()` call while the agent frame is active. Dispatches based on `Phase`:

| Phase                   | Action                                                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `Entry`                 | Call `history.push(session_id, agent_name, user_message)`. Advance phase to `Continue(Dispatch)`. Call `dispatch_agent`.              |
| `Continue(Dispatch)`    | Call `dispatch_agent` directly.                                                                                                       |
| `Continue(PendingTool)` | Move agent key to tail of state map. If tool keys still present, return `Continue` (let tools dispatch first). Otherwise re-dispatch. |
| `Continue(None)`        | Agent has written its exit slot. Return `Continue`; `call_exit` in `step` will pop this frame.                                        |

### dispatch_agent

Calls the LLM once. Two response paths:

**Structured output (`ClientOutput::Output`)**:

- Calls `history.push(&session_id, &agent_name, Message::assistant(content))`.
- Writes `val` under `node.exit`, removes `node.id`.
- Sets `Phase::Continue(None)`.

**Tool calls (`ClientOutput::ToolCalls`)**:

- Validates tool names (unknown → `AgentError`) and uniqueness (duplicate → `AgentError`).
- Calls `history.push(&session_id, &agent_name, Message { role: AssistantToolCalls { calls }, ... })`.
- Writes each `ToolCall` JSON under its interned `NodeId`.
- Sets `Phase::Continue(PendingTool)`.

`flows.rs` never constructs `HistoryEntry` directly — all history writes go through `FlowHistory::push`.

---

## Tool Dispatch

`step_inner` reaches a `Tool` node when a `ToolCall` JSON value is present under the tool's key. `handle_tool`:

1. Deserializes the `ToolCall` from the state map.
2. Calls `tool_box.call_at_index`.
3. `Ok(output)` → push `Role::Tool` history, remove tool key, return `Continue`.
4. `Err(ToolError::Exit(value))` → `handle_tool_exit`.
5. `Err(ToolError::Suspend)` → suspend (`src=tool`, `dst=agent`), return `Suspend { value: call.args }`.
6. Any other error → `FlowError::AgentError`.

**`handle_tool_exit`** (the `submit` sentinel path):

- Pushes the final value as a `Role::Tool` history entry.
- Collects all sibling tool keys (same `agent_name`) still in the state map, pushes `"cancelled"` history entries for each, removes them.
- Writes `value` under `node.agent_exit`.
- Sets `Phase::Continue(None)`.

After `handle_tool_exit`, the agent key remains in the state map but `Phase::Continue(None)` means `handle_child_agent` will return `Continue` immediately. `call_exit` then sees `agent_exit == callable.exit` and pops the agent frame.

---

## State Map Ordering Invariant

`IndexMap` preserves insertion order, and `step_inner` iterates in insertion order. Tool keys must appear **before** the agent key so `step_inner` dispatches tools before re-entering the agent.

This is maintained by `handle_child_agent`'s `PendingTool` branch: it removes the agent key and re-inserts it at the tail. Tool keys inserted by `dispatch_agent` appear before the re-inserted agent key.

---

## FlowRuntime

`FlowRuntime<I>` is a thin shell over `FlowGraph`. Its responsibilities:

- Owns `FlowState`, `FlowHistory`, `factory`, `callables`, `compactor`, `store`.
- `callable_index()` determines which `FlowGraph` in `callables` is the active one for the top frame. This changes when frames are pushed/popped by `call_enter`/`call_exit`.
- `next()` and `resume()` resolve the active graph, delegate to `FlowGraph::step`, then run the compaction loop and store flush (see Conversation History and Compaction).
- Returns `FlowError::Internal` (not panic) if `callable_index()` is `None` (stack empty — flow already done) or if the index is out of range (invariant violation).
- `compactor` and `store` are both held as `Box<dyn Dyn*>` (internal erasure of the public RPITIT async traits). Default: `NoopCompactor` and `NoopHistoryStore`.

### callables

`FlowRuntime.callables: Vec<FlowCall(Arc<FlowGraph>)>` is built once at construction by `collect_callables`, which walks the root graph recursively and assigns `callable_index` to each `FlowGraph` (including sub-flows) before pushing them. The root graph is pushed last.

`Frame.callable.index` indexes into this flat list. The active graph is always `callables[callable_index()]`.

---

## Session Lifecycle

Every `Frame` owns a `session_id: String` generated in `Frame::new()` via `Uuid::now_v7()`. Session identity is therefore tied to frame lifetime: a session starts when `call_enter` pushes a frame and ends when `call_exit` pops it.

`FlowState::top_session_id() -> &str` returns the top frame's `session_id`. Agent nodes read this in `handle_child_agent` and `dispatch_agent` to tag history entries.

`FlowState::active_session_ids() -> Vec<&str>` returns all current frame session IDs bottom-to-top. `FlowRuntime` uses this after each step to drive the per-session compaction loop.

Flow frames also get a `session_id` but it is never read — only agent frames use it.

---

## Conversation History and Compaction

### HistoryEntry and FlowHistory

`FlowHistory` stores `Vec<HistoryEntry>`. **`HistoryEntry` is never constructed outside `FlowHistory`**; its `pub(crate) new()` is the only constructor.

```
HistoryEntry {
    pub id: Uuid,           // stable DB row key; generated by FlowHistory::push()
    pub position: u64,      // monotonic ordering; stamped by push()
    pub session_id: String,
    pub agent_id: String,
    pub evicted: bool,      // set by apply_compaction(); cleared by prune_evicted()
    pub message: Message,   // wire-format only; no agent/session metadata on Message
}
```

`FlowHistory::push(session_id, agent_id, message)` — the **only** way to add an entry. Constructs the `HistoryEntry` internally and stamps `position` from a monotonic counter.

All read paths (`for_session`, `validate_for_session`, `last_role`, `is_empty`) filter out `evicted` entries.

`FlowHistory::entries() -> &[HistoryEntry]` returns all entries including evicted; intended for `HistoryStore` only.

### Visibility contract

| Consumer                    | Sees                                                                                                                             |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `flows.rs`                  | `Message` only — calls `history.push(session_id, agent_id, message)`                                                             |
| `HistoryCompactor`          | reads `&[HistoryEntry]` (one session's non-evicted slice), writes `CompactionResult { evict_indices, summary: Option<Message> }` |
| `HistoryStore`              | reads `history.entries() -> &[HistoryEntry]` (all, including evicted)                                                            |
| `HistoryEntry` construction | exclusively inside `FlowHistory`                                                                                                 |

### Compaction

`HistoryCompactor` is a public async trait (RPITIT — no `async_trait` in the public API). It is erased internally; `FlowRuntime` holds `Box<dyn DynHistoryCompactor>`.

Compaction is always **session-scoped**. `FlowRuntime` drives the loop:

```
after each next() / resume():
  for session_id in state.active_session_ids():
      entries = history.session_entries(session_id)   // !evicted, this session only
      result  = compactor.compact(session_id, &entries).await
      history.apply_compaction(session_id, &entries, result)
  store.flush(&mut history).await
```

`compact()` receives one session's slice and returns indices **into that same slice** to evict, plus an optional summary `Message`. It never sees other sessions' data — cross-session contamination is structurally impossible.

`apply_compaction` maps relative indices → absolute positions in `self.entries`, marks them `evicted = true`, and if a summary `Message` was returned, wraps it into a `HistoryEntry` (new `id`, `position` = first evicted entry's position, `agent_id = "__summary__"`).

### Persistence

`HistoryStore` is a public async trait, erased internally the same way as `HistoryCompactor`. `flush` upserts new entries (by `position > watermark`), deletes DB rows for `evicted = true` entries using `entry.id`, then calls `history.prune_evicted()` to physically remove evicted entries from memory.

`NoopHistoryStore::flush` calls `prune_evicted()` immediately so evicted entries do not accumulate.

`load(session_ids)` restores entries on process restart for the given sessions.

---

## Validation Pipeline

Two passes, both run at build time:

1. **`validate_nodes`** — per-node structural rules. No entry point needed. Catches: agent `exit == id`, work `exit == name`, map `exit == name`, fork < 2 children, fork/join referencing unregistered nodes, join ≠ 2 parents, either `left == right`, sub-flow missing `parent_entry`, suspend `entry == exit`.

2. **`validate`** — called by `with_entry` after an entry is designated. Runs forward reachability (BFS from entry) and backward liveness (BFS from terminals toward entry). A node is invalid if it cannot be reached from entry, or if it has no path to any terminal. `Map` successor is `exit_name`; `Suspend` successor is `exit`.

**Implication for runtime**: if both passes succeed, every state key that ends up in a frame's state map is either a registered node key or the frame's exit key. A frame where `step_inner` falls through (no dispatch) and `call_exit` also returns `None` is a structural impossibility with a validated graph — not a detectable runtime condition.

---

## Phase and AgentContinuation

`Phase` is per-frame and drives `handle_child_agent`:

```
Phase::Entry              — first entry into agent frame; push user message
Phase::Continue(Option<Value>)
  None                    — agent done; exit slot written; waiting for call_exit
  Some(AgentContinuation::Dispatch)     — ready to call LLM
  Some(AgentContinuation::PendingTool)  — tool calls issued; waiting for tools
```

`AgentContinuation` is the typed payload stored inside `Phase::Continue(Some(...))`. It is serialized to `Value` for storage in `Phase` because `Phase` itself must be `Serialize/Deserialize` (it lives inside `Frame` inside `FlowSnapshot`).

`Phase` is `None` on all non-agent frames (Work, Fork, Join, Either, Flow).

---

## Error Handling Conventions

- `FlowError::Internal(msg)` — logic invariant violated; frame stack empty where it should not be, index out of range, etc. These indicate engine bugs.
- `FlowError::AgentError(msg)` — LLM/provider failures, unknown tool name, duplicate tool call.
- `FlowError::DeserializeError(msg)` — malformed JSON in state map or history (e.g. ToolCall deserialization).
- No `unwrap()` or `expect()` anywhere in the engine. All `Option` returns from state mutators (`set_phase`, `set_state`, `remove_state`) are checked; `false` → `FlowError::Internal`.
- `sibling.name` deserialization in `handle_tool_exit` propagates via `?` — silent swallowing is forbidden.

---

## Extending the Engine

### Adding a new node type

1. Add a new `struct FooInfo { ... }` with the fields your node needs.
2. Add `FlowNode::Foo(FooInfo)` variant.
3. Implement `handle_foo(...)` — async if needed. Follow the 50-line limit; split into helpers.
4. Wire into `step_inner`'s match arm. It must `return` after dispatching.
5. Add a builder method `FlowBuilder::foo::<From, Out, _>(...)` that registers the node.
6. Add per-node validation rules to `validate_nodes` if the node has structural invariants.
7. Add the new key to forward-reachability successors in `validate` if needed.
8. Update `diagram.rs` if the node type should appear in Mermaid output.

### Adding a new error variant

Add to `FlowError` in `errors.rs` with a `#[error("...")]` message. If it is a build-time error, add to `BuildError` and add a `From<BuildError>` arm. Do not use `Box<dyn Error>`.

### Adding a new LLM provider

Implement `ClientFactory` and `Client` in `src/clients/`. Register the URL scheme in `DefaultClientFactory::create`. No engine changes needed.
