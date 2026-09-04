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
- keep-alive session behavior;
- a runtime filter over prepared tools;
- selected MCP text resources.

Memory is system context, not conversation history. Configuration should do
only read-only or idempotent external work because a failed step may be retried.

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
pravah = { version = "0.4.13", features = ["mcp"] }
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

## Client Factories

Graph agents use Rath's default client factory unless the runtime `Context`
supplies another one:

```rust
let ctx = Context::default().with_client_factory(my_factory);
```

The factory is runtime-only. Bind a context containing it when a workflow
starts or restores. This is the single graph-path override for testing,
tracing, retry, rate limiting, or custom provider clients.

## Client Layers

Client layers compose around Rath's default factory and can be installed on a
graph `Context`:

```rust
use pravah::clients::{ClientFactory, DefaultClientFactory, Provider};
use pravah::legacy::{RateLimit, RateLimitLayer, RetryConfig, RetryLayer, TracingLayer};
use tokio::time::Duration;

let factory = DefaultClientFactory
    .layer(TracingLayer)
    .layer(RetryLayer::new(RetryConfig::new(2, Duration::from_millis(250))))
    .layer(RateLimitLayer::new().with_limit(
        Provider::OpenAi,
        RateLimit::new(60_000, 4),
    ));

let ctx = Context::default().with_client_factory(factory);
```

The same factory can be supplied to compatibility-only `FlowRuntime` with
`with_factory`.

## Attachments

Build the initial user `Message` during configuration when it needs files,
URLs, or inline data:

```rust
use pravah::clients::{Attachment, Message};

let message = Message::user("Describe this image.")
    .with_attachment(Attachment::image_file("diagram.png"));
```

File attachments are resolved against `Context::working_dir` before provider
dispatch. Graph tool outputs are rendered as typed JSON results.

## Direct Client Usage

For one provider call without a workflow, create a client directly:

```rust
use pravah::clients::{ClientOptions, ClientOutput, Message};

let client = ClientOptions::default()
    .with_preamble("Answer concisely.")
    .create("ollama:///qwen3:8b?base_url=http://localhost:11434")?;

match client.execute(&[Message::user("What is Rust?")]).await?.output {
    ClientOutput::Output(value) => println!("{value}"),
    ClientOutput::ToolCalls { .. } => {
        return Err(AppError::UnexpectedToolCalls);
    }
}
```

For runnable graph and legacy examples, see the [example index](../examples/README.md).
