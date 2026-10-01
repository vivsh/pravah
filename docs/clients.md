# Graph Agents and Clients

Read this when adding models, tools, memory, or MCP resources to a
Pravah workflow. For execution and persistence, start with
[graph.md](graph.md). The older trait-based agent API is available only through
[`pravah::legacy`](legacy.md).

## Define an Agent With a Function

Agent definitions mirror flow definitions:

```rust
use pravah::clients::Message;
use pravah::{Agent, AgentConfig, Context, Flow};

fn approval(root: Flow<Request>) -> Flow<Decision> {
    root.map(prepare).agent(reviewer).suspend::<Decision>()
}

fn reviewer(root: Agent<PreparedRequest>) -> Agent<Review> {
    root
        .tools(review_tools)
        .control(control_reviewer)
        .configure(configure_reviewer)
}

async fn configure_reviewer(
    request: PreparedRequest,
    ctx: Context,
) -> Result<AgentConfig, ConfigureError> {
    let memory = ctx.require::<ReviewMemory>()?.load(&request).await?;
    Ok(AgentConfig::new(
        "openai:///gpt-5",
        instructions(&request),
        Message::user(message(&request)),
    )
    .memory(memory))
}
```

The definition function declares structure. The asynchronous `configure`
function resolves one invocation's behavior from its owned input and `Context`.
It runs once; the resolved settings are checkpointed and reused after snapshot
restoration.

`AgentConfig` can set:

- model URL, instructions, and the initial user message;
- optional text memory;
- provider-specific JSON options;
- an optional per-request output-token cap;
- an optional stable conversation key;
- a runtime filter over prepared tools;
- selected MCP text resources.

Memory is system context, not conversation history. Configuration should do
only read-only or idempotent external work because a failed step may be retried.

## Share a conversation by key

Use `.key(...)` when later invocations should receive the same conversation:

```rust
AgentConfig::new(model, instructions, Message::user(question.text))
    .key(question.conversation_key)
```

Without a key, ordinary graph agents start a fresh session each time. With a
key, the runtime retains its conversation across call sites, subflows and `each`
children. The same key deliberately shares history even between different agent
definitions; use distinct keys for different users or conversations. Keys must
contain non-whitespace text and are otherwise preserved exactly.

History belongs to one runtime, not a global cache. Independent executions do
not share it unless supplied through `start_with_history`; a snapshot preserves
all sessions. The selected session ID is `key:` followed by the supplied key;
the entry's `agent_id` still records which graph agent produced it. A nested
invocation cannot select a conversation with an unfinished exchange and returns
`GraphError::AgentConversationBusy` before appending its input.

Chat supplies an execution-scoped default key automatically. `.key(...)` on its
builder or function-defined configuration overrides that default. Conversation
keys do not change per-message keys or carry tool budgets between invocations.

## Limit Generated Output

Use the provider-neutral setting instead of putting token-limit aliases in
`provider_config`:

```rust
AgentConfig::new(model, instructions, Message::user(question))
    .max_output_tokens(2048)
    .turn_budget(6)
```

`max_output_tokens` caps each model request, including tool-loop requests and
forced conclusion. It includes reasoning tokens where the provider counts them;
it is not a cap on the entire agent invocation or a guarantee that the prompt
fits the model's context window. `turn_budget` separately limits ordinary model
turns. With no output cap configured, Rath's existing provider defaults apply.

Zero and repeated declarations produce `GraphError::AgentConfigValidation`
before activation changes history. Rath validates provider-specific ranges and
rejects conflicting token-limit aliases in provider JSON. Custom client factories
must preserve the cap and use a client that honors it.

Provider-reported exhaustion returns `GraphError::AgentOutputLimit { agent,
provider }`, including during forced conclusion. Partial output is discarded,
not accepted as an answer or executed as tool calls. Pravah does not retry
automatically. The dispatch checkpoint remains retryable; any successful
pre-dispatch history preparation remains applied. Retrying an unchanged request
may hit the same cap again and incur another provider charge.

The resolved cap survives snapshot restoration. A history preparation policy
can inspect the effective cap through `request.options().max_output_tokens`.
This exposes request context, not exact token counting or strict input-budget
enforcement.

## Inspect Client Errors

Pravah uses Rath 0.3's structured `ClientError`. Inspect its classification
instead of matching the former error enum variants:

```rust
use pravah::clients::{ClientError, ErrorKind};

let error = ClientError::new(ErrorKind::Validation, "A model is required.");
assert_eq!(error.kind(), ErrorKind::Validation);
```

Direct client callers can inspect provider, operation, HTTP status, request ID,
retry-after metadata and the source chain. `ErrorBody` is also re-exported under
`pravah::clients`; accessing `response_body()` is explicit because it may contain
private content. Do not log raw response bodies by default.

Graph workflows and Chat retain the original Rath error in
`GraphError::AgentClient { operation, source }`. `AgentClientOperation::Create`
means the client factory failed; `Execute` means a model request failed. This
discriminator is separate from Rath's provider-specific `source.operation()`.
Rath's registry annotates construction failures with the selected provider and
`client construction` operation; distinct original context and HTTP diagnostics
remain in its typed cause chain. Pravah preserves the error returned by Rath.

```rust
use pravah::{AgentClientOperation, GraphError};
use pravah::clients::ErrorKind;

fn inspect_failure(error: &GraphError) {
    if let Some(client) = error.client_error() {
        let kind: ErrorKind = client.kind();
        let status: Option<u16> = client.http_status();
        let retry_after: Option<&str> = client.retry_after();
        // Apply application policy to this metadata; no text parsing is needed.
    }
    if let GraphError::AgentClient {
        operation: AgentClientOperation::Create, source,
    } = error {
        // Inspect source.kind() and source.provider() to diagnose client setup.
    }
}
```

The same error propagates from `Chat::send` and `Chat::send_with_key`.
`client_error()` borrows the original error without copying it, and standard
`std::error::Error::source()` also exposes it, preserving nested causes.
`Display` says only `agent client creation failed` or `agent client execution failed`;
it does not include provider messages, bodies, prompts, or generated output.
Treat explicit source inspection, error-chain reporting and Debug output as
diagnostic access, not automatically safe user-facing logging.

Migrate tuple matches `GraphError::AgentClient(_)` to the struct variant above.
Runtime rejection of an empty tool-call batch is instead
`GraphError::AgentResponseValidation`; it has no Rath source. Provider-reported
output exhaustion during execution still becomes `GraphError::AgentOutputLimit`,
discards partial output, and returns None from `client_error()`.

Pravah adds no automatic retry. The compatibility-only legacy retry layer retains
its broad provider/transport retry policy; it is not a provider-specific HTTP retry
policy. Chat's unfinished-turn restrictions are unchanged.

## Declare and Filter Tools

A toolset function declares the complete set of tool graphs that can be
prepared with the workflow:

```rust
use pravah::{ToolFilter, Toolset};

fn review_tools(tools: Toolset) -> Toolset {
    tools.tool(read_file).flow(verify_claim)
}

async fn configure_reviewer(
    request: PreparedRequest,
    _ctx: Context,
) -> Result<AgentConfig, ConfigureError> {
    let allow_files = request.may_read_files;
    Ok(AgentConfig::new(
        "openai:///gpt-5",
        "Review the request.",
        Message::user(request.text),
    )
    .tool_filter(ToolFilter::new(move |tool| {
        tool.name() != "read_file_input" || allow_files
    })))
}
```

`ToolFilter` may capture values resolved during configuration, but it can only
select from the declared toolset. Selected tools keep their prepared order.
Duplicate tool definitions fail graph compilation.

Use `Toolset::tool(tool_fn)` for a standalone asynchronous function and
`Toolset::flow(flow_fn)` for a reusable graph flow. Tool functions have this
shape:

```rust
async fn read_file(
    request: ReadFileRequest,
    ctx: Context,
) -> Result<ReadFileResult, ToolError> {
    // ...
}
```

Tool names and input schemas come from their Rust input types. Recoverable
`ToolError` values are returned to the model as tagged tool results;
`ToolError::Fatal` ends the workflow step with an error.

## Set Simple Agent Budgets

Most applications can bound an agent loop directly in its dynamic
configuration:

```rust
Ok(AgentConfig::new(model, instructions, message)
    .turn_budget(6)
    .tool_budget::<SearchRequest>(2)
    .tool_budget::<FetchRequest>(3))
```

`turn_budget` permits that many ordinary model requests. If the final request
proposes tools instead of returning the structured output, Pravah completes
the accepted tool batch and performs one additional tool-disabled conclusion
request. A tool budget counts accepted attempts, including recoverable
failures. Rejected proposals and calls to unavailable tools do not consume it.

When a tool reaches zero it is omitted from later model requests. If a batch
contains more calls than remain, Pravah admits them in proposal order and
returns the ordinary unavailable result for the excess. Structured output,
including Rath's provider-specific exit tool, remains available.

Budgets are invocation-local and survive snapshot restoration. They are hard
ceilings when combined with a custom controller. A controller can inspect
them without maintaining its own counters:

```rust
let turns = loop_.turns_remaining();
let searches = loop_.calls_remaining("search_request");
```

`Some(0)` means exhausted. `None` means unbudgeted; use
`configured_tools()` to distinguish an unknown tool name.

## Control Long Agent Loops

An optional asynchronous controller can inspect each meaningful loop boundary
and choose how execution proceeds:

```rust
use pravah::{
    AgentDecision, AgentInterventionPoint, AgentLoop, ToolFilter,
};

async fn control_reviewer(
    loop_: AgentLoop<PreparedRequest>,
    _ctx: Context,
) -> Result<AgentDecision, ControlError> {
    match loop_.point() {
        AgentInterventionPoint::AfterTools
            if loop_.metrics().repeated_results() >= 2 =>
        {
            Ok(AgentDecision::redirect()
                .guidance("Use the evidence already collected; do not repeat calls.")
                .tools(ToolFilter::new(|tool| tool.name() == "summarize_input")))
        }
        AgentInterventionPoint::AfterTools
            if loop_.metrics().consecutive_tool_rounds() >= 4 =>
        {
            Ok(AgentDecision::conclude(
                "Return the best supported structured answer now.",
            ))
        }
        _ => Ok(AgentDecision::continue_()),
    }
}
```

The controller sees typed invocation input, live history, configured and active
tools, pending calls, completed results, cumulative usage, failures, and
deterministic repetition metrics. It can continue, redirect with one-shot
guidance and another subset of the originally configured tools, force one
tool-disabled conclusion turn, suspend for application input, or abort the
step without mutation.

Changing tool visibility affects later model requests only. Accepted calls and
results already in history remain unchanged. Calls to a currently unavailable
tool are not executed; the model receives a generic recoverable result for that
turn while valid calls from the same batch continue.

Controller state and metrics are checkpointed independently of compacted
history. Restoring does not repeat a committed controller decision. See
[`graph_agent_control`](../examples/graph_agent_control.rs) for an end-to-end
example including suspension and typed resume.

## MCP Text Resources

Enable the `mcp` feature to use Streamable HTTP resource servers:

```toml
pravah = { version = "0.4.20", features = ["mcp"] }
```

Register credentials and headers on the runtime `Context`, not in the graph or
snapshot:

```rust
use pravah::{McpResourceRef, McpServer};

let ctx = Context::default().with_mcp_server(
    McpServer::new("handbook", "https://mcp.example.com")
        .bearer_token(token)
        .header("x-tenant", tenant_id),
);

let catalog = ctx.mcp_resources("handbook").await?;
```

Configuration selects concrete resource or template references with
`McpResourceRef`. Pravah preserves selection order, rejects duplicates and blob
content, and checkpoints the resolved text and provenance. Restoring the agent
therefore performs no MCP request.

See [MCP resources and agent tool filters](mcp.md) for catalog selection,
resource templates, dynamic filtering, and a complete runnable example.

## Model URLs and Credentials

Model URLs use this form:

```text
provider:///provider-native-model-id[?param=value]
```

Examples include `openai:///gpt-5`,
`anthropic:///claude-sonnet-4-5`,
`gemini:///gemini-2.5-flash-lite`, and
`ollama:///qwen3:8b?base_url=http://localhost:11434`.

Provider credentials use their usual environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `OLLAMA_API_KEY` when required by the server

The `api_key_env` query parameter can select another environment variable.
Use `base_url` for compatible proxies or self-hosted endpoints.

## Provider Registries

Graph agents use Rath's built-in providers unless the runtime `Context`
supplies an application-owned registry:

```rust
use pravah::{Context, clients::ProviderRegistry};

let providers = ProviderRegistry::with_builtins()
    .register("company", my_factory)?;
let ctx = Context::default().with_providers(providers);
```

Use `company:///model` to select that factory. Built-in scheme names cannot
be replaced. Use `ProviderRegistry::new()` to allow only explicit registrations.

The registry is runtime-only. Bind it through the context when starting or
restoring. Implement Rath's re-exported `ProviderFactory::llm` for asynchronous
construction and `LlmBackend` for execution. Return `Client::from_backend(backend)`;
`Client` is a concrete handle; do not wrap it in `Box<dyn Client>`.

## Client Layers

Compatibility-only client layers can decorate an application provider factory.
No retries or rate limits are installed by default:

```rust
use pravah::clients::{Provider, ProviderRegistry};
use pravah::legacy::{RateLimit, RateLimitLayer, RetryConfig, RetryLayer, TracingLayer};
use tokio::time::Duration;

let factory = TracingLayer.layer(my_factory);
let factory = RetryLayer::new(RetryConfig::new(2, Duration::from_millis(250)))
    .layer(factory);
let factory = RateLimitLayer::new()
    .with_limit(Provider::External("company".into()), RateLimit::new(60_000, 4))
    .layer(factory);

let providers = ProviderRegistry::with_builtins().register("company", factory)?;
let ctx = Context::default().with_providers(providers);
```

Compatibility-only `FlowRuntime` also accepts a registry through `with_providers`.
Legacy Chat client construction is asynchronous: use `.build().await?` and
`Chat::from_snapshot(snapshot).await?`. Modern Chat construction remains
synchronous, as does its snapshot restoration. Modern `send` remains asynchronous.

## Attachments

Build the initial user `Message` during configuration when it needs files,
URLs, or inline data:

```rust
use pravah::clients::Message;

let message = Message::user("Describe this image.")
    .with_file("image/png", "diagram.png");
```

File attachments are resolved against `Context::working_dir` before provider
dispatch. Graph tool outputs are rendered as typed JSON results.

## Direct Client Usage

For one provider call without a workflow, create a client directly:

```rust
use pravah::clients::{ClientOptions, ClientOutput, Message};

let client = ClientOptions::default()
    .with_preamble("Answer concisely.")
    .create("ollama:///qwen3:8b?base_url=http://localhost:11434").await?;

match client.execute(&[Message::user("What is Rust?")]).await?.output {
    ClientOutput::Output(value) => println!("{value}"),
    ClientOutput::ToolCalls { .. } => {
        return Err(AppError::UnexpectedToolCalls);
    }
}
```

For runnable graph and legacy examples, see the [example index](../examples/README.md).
