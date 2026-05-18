# Clients

Read this when you are wiring models, providers, tools, or attachments into a
flow. If you want the runtime and graph semantics instead, start with
[flows.md](flows.md).

For API details, use [docs.rs](https://docs.rs/pravah). This document focuses on
how the pieces fit together in practice.

## What Lives Here

Pravah's client layer covers:

- `Agent` definitions
- model URL parsing and provider selection
- client factory layers such as retry, rate limiting, and tracing
- structured output
- tools and tool loops
- multimodal attachments

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

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a careful planning agent.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}
```

The input type is the user-side contract. The output type is the structured
result the model must produce.

For the simplest end-to-end example, see
[../examples/linear_flow.rs](../examples/linear_flow.rs).

## Model URLs

Model URLs follow `provider://model-id`.

- `openai://gpt-4o`
- `anthropic://claude-opus-4-5`
- `claude://claude-sonnet-4-5`
- `gemini://gemini-2.5-flash-lite`
- `ollama://localhost:11434/llama3`

`claude://...` is an alias for `anthropic://...`.

Provider identity comes from the URL scheme, not the upstream hostname. That
means a layered client factory still treats
`openai://gpt-4o?base_url=https://openrouter.ai/api/v1&api_key_env=OPENROUTER_API_KEY`
as `Provider::OpenAi` for rate limiting, retries, and tracing.

## Compatible Endpoints

OpenAI-, Anthropic-, and Ollama-compatible hosts can override transport details
with query params.

- `openai://gpt-4o?base_url=https://openrouter.ai/api/v1&api_key_env=OPENROUTER_API_KEY`
- `anthropic://claude-sonnet-4-5?base_url=https://anthropic-proxy.example/v1&api_key_env=ANTHROPIC_PROXY_KEY`
- `ollama://qwen3:8b?base_url=https://ollama.example&api_key_env=OLLAMA_API_KEY`

Inline keys still work as `provider://key@model-id`.

Legacy Ollama URLs are still supported:

- `ollama://localhost:11434/qwen3:8b`

Auth is resolved in this order:

1. inline key in the model URL
2. `api_key_env` from the query string
3. the provider default environment variable

Provider defaults are:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OLLAMA_API_KEY`

`api_key_env` is resolved when the model URL is parsed. If the named variable is
missing, client creation fails early instead of silently falling back.

### Protocol Boundaries

- `openai://...` uses the OpenAI Responses API. `base_url` should point to the `/v1` root, and Pravah will call `{base_url}/responses` and `{base_url}/embeddings`.
- `anthropic://...` and `claude://...` use the Anthropic Messages API. `base_url` should point to the `/v1` root, and Pravah will call `{base_url}/messages`.
- `ollama://...` uses `{base_url}/v1/chat/completions` for generation and `{base_url}/api/embed` for embeddings. Auth is optional and is only sent when a key is configured.
- `gemini://...` uses the native Gemini adapter and does not use the same compatibility URL shape.

## Structured Output

With no tools, Pravah runs agents in structured-output mode. The exact wire
shape is provider-specific, but the input and output contracts stay typed at the
flow boundary.

Current behavior:

- OpenAI: native JSON Schema mode
- Gemini: native JSON Schema mode with provider-compatible schema sanitization
- Ollama: native JSON Schema mode when `output_schema` is present, otherwise JSON object mode
- Anthropic: schema guidance is expressed in the prompt as a strict contract

## Tools

Attach tools with `.with_tools()` on `AgentConfig`.

```rust
fn build() -> AgentConfig {
    AgentConfig::new(
        "You are a careful planning agent.",
        "gemini://gemini-2.5-flash-lite",
    )
    .with_tools(ToolBox::new().tool::<ReadNote>())
}
```

`Tool::call` returns `ToolOutput<T>`, not just `T`. That allows a tool to return
typed JSON plus attachments in the same result.

See [../examples/image_prompt.rs](../examples/image_prompt.rs) for initial user
attachments and the tool/attachment sections below for the provider behavior.

## Client Layers

`FlowRuntime` accepts a custom client factory via `with_factory()`. Layers wrap
the built-in `DefaultClientFactory`.

```rust
use pravah::clients::{DefaultClientFactory, Provider};
use pravah::flows::{ClientFactory, RateLimit, RateLimitLayer, RetryConfig, RetryLayer, TracingLayer};
use tokio::time::Duration;

let factory = DefaultClientFactory
    .layer(TracingLayer)
    .layer(RetryLayer::new(RetryConfig::new(2, Duration::from_millis(250))))
    .layer(RateLimitLayer::new().with_limit(Provider::OpenAi, RateLimit::new(60_000, 4)));
```

The last layer you add is the outermost wrapper.

```text
RateLimit -> Retry -> Tracing -> DefaultClientFactory -> provider client
```

Use this when you need request tracing, rate limiting, or transport retries
without changing the flow definition itself.

## Attachments

Attachments let tools or agents send binary data alongside text or JSON.

Three forms are supported:

| Variant | Use |
| ------- | --- |
| `Attachment::Inline { mime_type, data }` | Base64-encoded inline data |
| `Attachment::File { mime_type, path }` | File under the current `working_dir`, materialized before dispatch |
| `Attachment::Url { mime_type, url }` | Public URL |

`Attachment::File` is resolved through `Context::resolve`, so it stays confined
to the configured working directory.

### Initial User Attachments

Override `Agent::to_message(self, ctx)` when the first user turn should include
attachments.

See [../examples/image_prompt.rs](../examples/image_prompt.rs) for a complete
vision example built around `Attachment::File`.

### Tool Attachments

Tools can return JSON and attachments together:

```rust
use pravah::clients::Attachment;
use pravah::tools::{Tool, ToolError, ToolOutput};

async fn call(self, ctx: Context) -> Result<ToolOutput<Self::Output>, ToolError> {
    let content = tokio::fs::read_to_string(ctx.resolve("note.txt")?).await?;
    Ok(ToolOutput::with_attachment(
        ReadNoteOutput { content: content.clone() },
        Attachment::Inline {
            mime_type: "text/plain".into(),
            data: base64::engine::general_purpose::STANDARD.encode(content.as_bytes()),
        },
    ))
}
```

### Provider Attachment Support

Attachments are materialized before provider dispatch, then translated into each
provider's wire format.

| Provider | Inline / File | URL |
| -------- | ------------- | --- |
| Anthropic | Image-only blocks on this path | Image-only blocks on this path |
| OpenAI | Image-only `input_image` items on this path | Image-only `input_image` items on this path |
| Gemini | `Part::InlineData` with arbitrary mime types | `Part::FileData` with arbitrary mime types |
| Ollama | Image-only `image_url` parts on this path | Image-only `image_url` parts on this path |

## Example Map

- [../examples/linear_flow.rs](../examples/linear_flow.rs): minimal typed agent
- [../examples/image_prompt.rs](../examples/image_prompt.rs): initial image upload
- [../examples/debate.rs](../examples/debate.rs): multi-agent client usage across branches

If you are done with provider and tool setup, continue with
[flows.md](flows.md) for graph semantics, suspension, snapshots, and history.