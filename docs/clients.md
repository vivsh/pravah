# Clients

Read this when wiring models, providers, tools, or attachments into a flow.
For graph semantics, suspension, and runtime behavior, start with
[flows.md](flows.md). For full API details, use [docs.rs](https://docs.rs/pravah).

## What Lives Here

Pravah's client layer covers:

- `Agent` definitions
- model URLs and provider selection
- structured output
- tool calls and tool results
- multimodal attachments
- direct client usage
- layered client factories for retry, rate limiting, and tracing

## Agents

Agents are typed LLM nodes defined by implementing `Agent`.

```rust
use pravah::flows::{Agent, AgentConfig};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct PlannerInput {
    goal: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct Plan {
    steps: Vec<String>,
}

impl Agent for PlannerInput {
    type Output = Plan;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a careful planning agent.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}
```

The input type is the boundary contract. The output type is the structured
result the model must produce.

Common `AgentConfig` options:

| Method                            | Effect                                                                       |
| --------------------------------- | ---------------------------------------------------------------------------- |
| `AgentConfig::new(preamble, url)` | Set the system prompt and model URL                                          |
| `.keep_alive()`                   | Reuse one session id across loop re-entries for that agent                   |
| `.with_turn_budget(n)`            | Cap how many LLM dispatch turns the agent may take in one execution          |
| `.with_turn_budget_message(msg)`  | Override the final reminder message injected when the turn budget is reached |

Override `Agent::to_message(self, ctx)` when the first user turn must be built
manually. Use it for initial attachments or custom user text. See
[../examples/image_prompt.rs](../examples/image_prompt.rs).

## Model URLs

Model URLs encode provider, optional transport, optional base URL, model name,
and optional query parameters in one string:

```text
provider[+transport]://[authority][/prefix/]model[?param=value]
```

Examples:

- `openai:///gpt-4o`
- `anthropic:///claude-sonnet-4-5`
- `claude:///claude-sonnet-4-5`
- `gemini:///gemini-2.5-flash-lite`
- `ollama://localhost:11434/qwen3:8b`

`claude://...` is an alias for `anthropic://...`.

Supported query parameters are:

- `temperature`
- `thinking`
- `api_key_env`

Thinking levels are `off`, `low`, `medium`, `high`, and `xhigh`.

### Feature Flags And Credentials

Provider support is controlled by Cargo features:

- `provider-openai` uses `OPENAI_API_KEY`
- `provider-anthropic` uses `ANTHROPIC_API_KEY`
- `provider-gemini` uses `GEMINI_API_KEY`
- `provider-ollama` uses `OLLAMA_API_KEY` when present, but can also run without auth

`api_key_env` overrides the default environment variable. It is resolved when
the model URL is parsed. If the named variable is missing, Pravah fails early.

### Compatible Endpoints

Use `+transport` when routing a provider through a compatible host or proxy:

- `openai+https://openrouter.ai/api/v1/gpt-4o?api_key_env=OPENROUTER_API_KEY`
- `anthropic+https://anthropic-proxy.example/v1/claude-sonnet-4-5?api_key_env=ANTHROPIC_PROXY_KEY`
- `ollama+https://ollama.example/qwen3:8b?api_key_env=OLLAMA_API_KEY`

Provider behavior comes from the scheme, not the hostname. For example,
`openai+https://openrouter.ai/api/v1/gpt-4o` still behaves as `Provider::OpenAi`
for retry, rate limiting, and tracing layers.

### Provider Boundaries

- `openai[+transport]://...` uses the OpenAI Responses API
- `anthropic[+transport]://...` and `claude[+transport]://...` use the Anthropic Messages API
- `ollama[+transport]://...` uses OpenAI-compatible chat for generation and `/api/embed` for embeddings
- `gemini://...` uses the native Gemini adapter and does not support custom base URLs

## Structured Output

When no tools are attached, Pravah runs agents in structured-output mode. The
wire protocol is provider-specific, but the flow boundary stays typed: input
comes from the agent input type, and output must deserialize into
`Agent::Output`.

Current behavior:

- OpenAI uses native JSON Schema mode
- Gemini uses native JSON Schema mode with schema sanitization
- Ollama uses native JSON Schema mode when `output_schema` is present, otherwise JSON object mode
- Anthropic expresses the schema contract in the prompt

When tools are attached, the agent runs in tool-loop mode and returns either a
typed final output or a batch of tool calls.

## Tools

There are two ways to attach a tool to an agent.

**Primary API — implement `Tool`:**

```rust
use pravah::tools::{Tool, ToolError};

struct ReadFile;

impl Tool for ReadFile {
    type Input = ReadFileInput;
    type Output = ReadFileOutput;

    async fn call(input: Self::Input, ctx: Context) -> Result<Self::Output, ToolError> {
        // …
    }
}

// In the flow:
builder.agent::<MyAgent>().tool::<MyAgent, ReadFile>()
```

`.tool::<A, T>()` registers the `Tool` impl and wires the work node automatically.

**Shorthand API — `tool_flow` for sub-flow tools:**

Use `.tool_flow::<A, F>()` when the tool implementation is itself a full flow.
The model sees the tool name derived from `F`, and calling it runs the entire
sub-flow inline. `F::Output` must implement `ToolOutput`.

```rust
impl Flow for ArticleRequest {
    type Output = ArticleSummary;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .agent::<ArticleRequest>()
            .tool_flow::<ArticleRequest, VerifyClaim>()
    }
}
```

This is equivalent to `.tool_with::<ArticleRequest, VerifyClaim, VerificationResult>().flow::<VerifyClaim>()`.

See [../examples/tool_flow.rs](../examples/tool_flow.rs).

**Explicit API — `tool_with` for flow-backed tools:**

Use `.tool_with::<A, I, O>()` when you want to back a tool with an embedded flow
or supply the work node manually. `O` must implement `ToolOutput`.

```rust
impl Flow for BlogRequest {
    type Output = FinalResult;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .agent::<BlogRequest>()
            .tool_with::<BlogRequest, HumanInput, HumanOutput>()
            .flow::<HumanInput>()
    }
}
```

Tool names are derived from the input type schema name. The model sees that
schema as the tool contract.

### Tool Results

For `.tool::<A, T>()`, override `Tool::to_message` on your `Tool` impl when
you need custom text or attachments instead of plain JSON.

For `.tool_with::<A, I, O>()`, `O` must implement `ToolOutput`. Override
`ToolOutput::to_message` for the same purpose:

```rust
use pravah::clients::Message;
use pravah::tools::{ToolError, ToolOutput};

impl ToolOutput for ReadNoteOutput {
    fn to_message(self) -> Result<Message, ToolError> {
        let content = serde_json::to_string(&self).map_err(ToolError::Serialize)?;
        Ok(Message::tool_output(String::new(), content)
            .with_inline("text/plain", self.content.as_bytes()))
    }
}
```

### Tool Errors

All `ToolError` variants except `Fatal` are serialized as a structured JSON
message and sent back to the model as a tool result. The model can inspect
`error_kind` and `message` to decide how to proceed:

```json
{
  "tool": "ReadFile",
  "ok": false,
  "error_kind": "NotFound",
  "message": "…",
  "recoverable": true
}
```

| Variant                      | When to use                                                           |
| ---------------------------- | --------------------------------------------------------------------- |
| `ToolError::NotFound(msg)`   | Tool or resource not found                                            |
| `ToolError::TypeError(e)`    | Model passed a value with the wrong JSON shape                        |
| `ToolError::Validation(msg)` | Constraint or argument violation the model can correct                |
| `ToolError::Security(msg)`   | Path escape or forbidden command attempt                              |
| `ToolError::Io(e)`           | Filesystem error                                                      |
| `ToolError::Http(msg)`       | Network error                                                         |
| `ToolError::Serialize(e)`    | Output serialization failure                                          |
| `ToolError::Other(msg)`      | Any other soft error                                                  |
| `ToolError::Fatal(msg)`      | Abort the flow immediately — the only variant that terminates the run |

`Context::check_command` and `Context::resolve` both return `ToolError::Security`
when the check fails. The model receives the structured error and may retry or
give up, but the flow is never aborted on their behalf.

## Attachments

Attachments let agents or tools send binary data alongside text or JSON.

| Variant                                  | Use                                                                |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `Attachment::Inline { mime_type, data }` | Base64-encoded inline bytes                                        |
| `Attachment::File { mime_type, path }`   | File under the current `working_dir`, materialized before dispatch |
| `Attachment::Url { mime_type, url }`     | Public URL reference                                               |

`Attachment::File` is resolved through `Context::resolve`, so it stays inside
the configured working directory.

### Initial User Attachments

Override `Agent::to_message(self, ctx)` when the first user turn should include
attachments. See [../examples/image_prompt.rs](../examples/image_prompt.rs).

### Tool Attachments

Tool outputs can also attach files or inline bytes through
`ToolOutput::to_message`.

### Provider Attachment Support

Attachments are materialized before provider dispatch, then translated into the
provider's wire format.

| Provider  | Inline or File                              | URL                                         |
| --------- | ------------------------------------------- | ------------------------------------------- |
| Anthropic | Image-only blocks on this path              | Image-only blocks on this path              |
| OpenAI    | Image-only `input_image` items on this path | Image-only `input_image` items on this path |
| Gemini    | Arbitrary mime via `InlineData`             | Arbitrary mime via `FileData`               |
| Ollama    | Image-only `image_url` parts on this path   | Image-only `image_url` parts on this path   |

## Direct Clients

Use `ClientOptions` when you want one provider call without a flow, or a small
focused test around one provider adapter.

```rust
use pravah::clients::{ClientOptions, ClientOutput, Message};
use serde_json::json;

let client = ClientOptions::default()
    .with_preamble("You are a helpful assistant.")
    .with_input_schema(json!({
        "type": "object",
        "properties": { "question": { "type": "string" } },
        "required": ["question"]
    }))
    .with_output_schema(json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"]
    }))
    .create("ollama://localhost:11434/qwen3:8b")?;

let response = client
    .execute(&[Message::user(r#"{"question":"What is Rust?"}"#)])
    .await?;

match response.output {
    ClientOutput::Output(value) => println!("{value}"),
    ClientOutput::ToolCalls { .. } => unreachable!(),
}
```

If you need embeddings, call `Client::embed` with `EmbedRequest`. Providers
that do not support embeddings return `UnsupportedCapability`.

## Client Layers

`FlowRuntime` accepts a custom client factory via `with_factory()`. Layers wrap
`DefaultClientFactory` without changing the flow definition itself.

```rust
use pravah::clients::{DefaultClientFactory, Provider};
use pravah::flows::{FlowRuntime, RateLimit, RateLimitLayer, RetryConfig, RetryLayer, TracingLayer};
use tokio::time::Duration;

let factory = DefaultClientFactory
    .layer(TracingLayer)
    .layer(RetryLayer::new(RetryConfig::new(2, Duration::from_millis(250))))
    .layer(RateLimitLayer::new().with_limit(Provider::OpenAi, RateLimit::new(60_000, 4)));

let runtime = FlowRuntime::new(input)?.with_factory(factory);
```

The most recently added layer becomes the outermost wrapper.

```text
RateLimit -> Retry -> Tracing -> DefaultClientFactory -> provider client
```

## Example Map

- [../examples/linear_flow.rs](../examples/linear_flow.rs): minimal typed agent
- [../examples/human_input.rs](../examples/human_input.rs): tool call that suspends through a subflow
- [../examples/image_prompt.rs](../examples/image_prompt.rs): initial user attachments
- [../examples/ollama_client.rs](../examples/ollama_client.rs): direct provider client usage
- [../examples/debate.rs](../examples/debate.rs): multi-agent model usage across branches

If you are done with provider setup, continue with [flows.md](flows.md) for
graph semantics, suspension, snapshots, and history.
