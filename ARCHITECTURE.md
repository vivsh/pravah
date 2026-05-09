# Pravah Architecture Spec

Typed, single-threaded, stepwise flow runtime for LLM-agent pipelines.
Each `next()` call performs one bounded transition. State is JSON-backed and fully snapshottable between steps.

Out of scope: distributed orchestration, durable storage, parallel scheduling.

---

## Naming Convention (Critical)

Every node and state in the graph is keyed by a **schema name** — the string `schemars` derives from the Rust type name. By default this equals the unqualified struct name.

```
node_id() == schema_name() == "MyStruct"   // for struct MyStruct { ... }
```

All input and output types **must** derive:
```rust
#[derive(Serialize, Deserialize, JsonSchema)]
```
Omitting any of these causes a compile error or a silent runtime schema mismatch.

---

## Trait Signatures

### Flow

```rust
pub trait Flow: 'static + JsonSchema + Serialize + DeserializeOwned + Send + Sync {
    type Output: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static;
    fn build() -> Result<FlowGraph, FlowError>;
    fn node_id() -> String { Self::schema_name() }  // override only if needed
}
```

Minimal implementation:
```rust
impl Flow for MyInput {
    type Output = MyOutput;
    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            // ... nodes ...
            .build()
    }
}
```

### Agent

```rust
pub trait Agent: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static {
    type Output: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static;
    fn build() -> AgentConfig;
    fn node_id() -> String { Self::schema_name() }  // override only if needed
}
```

Minimal implementation:
```rust
impl Agent for MyInput {
    type Output = MyOutput;
    fn build() -> AgentConfig {
        AgentConfig::new("system prompt", "openai://gpt-4o")
    }
}
```

`AgentConfig` constructors:
```rust
AgentConfig::new(preamble, model_url)                   // no tools
AgentConfig::new(preamble, model_url).with_tools(box)   // with tools
```

The exit sentinel tool (`submit`) is **auto-injected** by the engine — do not add it manually to the `ToolBox`.

### Tool

```rust
pub trait Tool: DeserializeOwned + JsonSchema + Sized + Send {
    type Output: serde::Serialize + Send;
    fn name() -> &'static str;
    fn description() -> &'static str;
    fn call(self, ctx: Context) -> impl Future<Output = Result<Self::Output, ToolError>> + Send;
}
```

Minimal implementation:
```rust
impl Tool for MyTool {
    type Output = String;
    fn name() -> &'static str { "my_tool" }
    fn description() -> &'static str { "Does something" }
    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        Ok("result".into())
    }
}
```

To suspend (pause for external input):
```rust
return Err(ToolError::suspend(json!({"prompt": "needs approval"})));
```

---

## Builder API

All builder methods consume and return `self` (fluent chain). Call `.build()` at the end.

### `work::<From, Out, _, _>(async_fn)`

```rust
.work::<From, Out, _, _>(|from: From, ctx: Context| async move {
    Ok(Out { ... })
})
```
Async transform. `From` must already be a live state. Produces `Out` state; removes `From`.

### `agent::<A>()`

```rust
.agent::<A>()
```
Registers `A` as an LLM-driven agent node. `A: Agent`. Produces `A::Output` state.

**Mode switching**: if `A::build()` returns an empty `ToolBox`, the engine uses structured-output mode (schema passed to provider). If tools are present, it injects the `submit` sentinel and uses tool-calling mode.

### `either::<From, A, B, _>(sync_fn)`

```rust
.either::<From, A, B, _>(|from: From, ctx: Context| -> Result<Either<A, B>, FlowError> {
    if condition { Ok(Either::Left(A { ... })) }
    else         { Ok(Either::Right(B { ... })) }
})
```
Routes to exactly one of `A` or `B`. `A` and `B` must have distinct schema names.

### `fork::<From, A, B, _>(sync_fn)`

```rust
.fork::<From, A, B, _>(|from: From, ctx: Context| -> Result<(A, B), FlowError> {
    Ok((A { ... }, B { ... }))
})
```
Produces both `A` and `B` states simultaneously for parallel processing. Both branches must eventually converge via `join`.

### `join::<A, B, Out, _>(sync_fn)`

```rust
.join::<A, B, Out, _>(|a: A, b: B, ctx: Context| -> Result<Out, FlowError> {
    Ok(Out { ... })
})
```
Fires only when both `A` and `B` states are present. Consumes both and produces `Out`. `Out` must not equal `A` or `B` (validation enforced).

### `flow::<F>()`

```rust
.flow::<F>()
```
Inlines inner flow `F` into the outer graph. Entry (`F::schema_name()`) and exit (`F::Output::schema_name()`) keep their original names. All other inner node keys are prefixed `"{entry}::{inner_key}"` to prevent collisions.

---

## Model URL Formats

```
openai://gpt-4o
openai://gpt-4o-mini
anthropic://claude-opus-4-5
anthropic://claude-sonnet-4-5
gemini://gemini-2.5-flash
gemini://gemini-2.5-flash-lite
ollama://localhost:11434/qwen3:8b
ollama://localhost:11434/llama3.2
```

---

## Runtime API

```rust
FlowRuntime::new(input: I) -> Result<FlowRuntime<I>, FlowError>
FlowRuntime::from_snapshot(snapshot: FlowSnapshot) -> Result<FlowRuntime<I>, FlowError>

runtime.next(ctx: Context) -> Result<RunOut<I::Output>, FlowError>
runtime.resume(ctx: Context, (tool_id: String, payload: Value)) -> Result<RunOut<I::Output>, FlowError>
runtime.snapshot() -> FlowSnapshot
runtime.with_history(history: ClientHistory) -> Self
runtime.with_factory(factory: impl ClientFactory) -> Self
```

`RunOut<O>`:
```rust
RunOut::Continue                           // step done, more steps remain
RunOut::Done(O)                            // terminal state reached
RunOut::Suspend { value: Value, tool_id }  // tool requested suspension
```

---

## Suspend / Resume Protocol

1. A tool returns `Err(ToolError::suspend(value))`.
2. `next()` returns `RunOut::Suspend { value, tool_id }` where `tool_id = "{agent_name}::{tool_name}"`.
3. Caller persists `tool_id` and the snapshot.
4. When external input is available, call `runtime.resume(ctx, (tool_id, payload))`.
5. `resume()` replays pending tool calls with `payload` injected, then continues normally.

Invariants:
- `next()` while suspended → `FlowError::ResumeRequired(tool_id)`
- `resume()` while not suspended → `FlowError::UnexpectedResumption(tool_id)`
- `resume()` with wrong `tool_id` → `FlowError::ResumeMismatchError(expected_id)`

---

## Validation Invariants

Enforced at `FlowGraph::build()` time:
- No duplicate node keys
- Entry node must be registered
- No unreachable nodes from entry
- Every node must have a path to a terminal
- `either`: both branches must have distinct schema names
- `fork`: at least 2 children; all children must be registered nodes
- `join`: exactly 2 distinct parent ids; both parents must be registered; target ≠ either parent

Enforced at `FlowRuntime::new()` / `from_snapshot()` time:
- Exactly one distinct terminal state id must exist across all nodes
- That terminal id must equal `I::Output::schema_name()`

---

## Structured Output vs Tool Mode

When `AgentConfig` has an empty `ToolBox`:
- Provider receives `output_schema` (JSON Schema of `A::Output`)
- No tools are advertised; LLM returns a typed JSON object directly

When tools are present:
- Provider receives tool definitions including the auto-injected `submit` sentinel
- LLM calls `submit` with the final value to complete the agent step

Provider-specific structured output behavior:
- OpenAI: native JSON Schema response format
- Gemini: native JSON Schema response format (schema sanitized for provider compatibility)
- Ollama: `json_schema` response format when `output_schema` is set; falls back to `json_object` otherwise
- Anthropic: schema provided as strict prompt contract (best effort)
- GenAI: JSON schema response format via the `genai` adapter

---

## Context and Sandbox

```rust
Context::new(FlowConf {
    working_dir: Some(path),
    commands: vec!["git".into()],
    ..Default::default()
})
```

`ctx.resolve(raw_path)` resolves relative to `working_dir`:
- Absolute paths accepted only if within `working_dir`
- `..` traversal outside root → `ToolError::PathEscape`
- Symlinks allowed only if canonical target is within `working_dir`

`ctx.check_command(cmd)` → `ToolError::ForbiddenCommand` if `cmd` is not in the allowlist.

---

## Error Types

`FlowError` variants:
```
NotFound(String)             — node or state missing
BuildError(String)           — manual build failure
Invalid(Vec<String>)         — one or more validation problems
SerializeError(String)
DeserializeError(String)
SnapLoadError(String)
SnapStoreError(String)
AgentError(String)           — LLM/provider failure
Deadlock(String)             — states waiting but no join ready
ResumeRequired(String)       — next() called while suspended
UnexpectedResumption(String) — resume() called while not suspended
ResumeMismatchError(String)  — resume tool_id does not match suspended tool_id
```

`ToolError` variants:
```
Io(std::io::Error)
Serialize(serde_json::Error)
Deserialize(serde_json::Error)
UnknownTool(String)
PathEscape(String)
ForbiddenCommand(String)
Http(String)
Missing(DepsError)
Other(Box<dyn Error + Send + Sync>)
Exit(Value)    — internal; caught by engine; never propagates to user code
Suspend(Value) — internal; caught by engine; never propagates to user code
```

---

## Snapshot and History

`FlowSnapshot` serializes the flow state map only. The graph is rebuilt from `I::build()` on restore. LLM conversation history is **not** included in the snapshot; save and restore it separately:

```rust
let snapshot = runtime.snapshot();
let history = runtime.history().clone();
// restore:
let runtime = FlowRuntime::<MyFlow>::from_snapshot(snapshot)?.with_history(history);
```

---

## Nested Flow Inlining

`flow::<F>()` merges `F`'s nodes into the outer graph:
- Entry key (`F::schema_name()`) — unprefixed; shared with outer graph's existing state
- Exit key (`F::Output::schema_name()`) — unprefixed; emitted into outer graph
- All other inner keys — prefixed as `"{entry}::{inner_key}"`

`fork` and `either` closures bake state names at construction time. The inliner wraps them to rewrite emitted names to their prefixed equivalents. This is automatic and transparent.

---

## Determinism Boundary

Deterministic: graph validation, `work`/`either`/`fork`/`join` routing, snapshot/restore.
Non-deterministic: LLM responses, tool side effects, external I/O.
