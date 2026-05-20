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
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}
```

The input type is the user-side contract. The output type is the structured
result the model must produce.

Common `AgentConfig` options:

| Method                        | Effect                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| `.keep_alive()`               | Reuse the same session id across loop re-entries so the LLM sees full conversation history |
| `.with_max_tool_calls(n)`     | Cap the number of calls allowed per tool name per agent execution                          |
| `.with_loop_break_message(m)` | Message returned to the LLM when a tool's call budget is exhausted                         |

To set temperature or thinking level, add query parameters to the model URL:

    gemini:///gemini-2.5-pro?temperature=0.7&thinking=medium

Supported thinking levels are `off`, `low`, `medium`, `high`, and `xhigh`.

For the simplest end-to-end example, see
[../examples/linear_flow.rs](../examples/linear_flow.rs).

## Model URLs

Model URLs encode the provider, optional transport, host (if needed), base path,
model name, and optional parameters in a single string:

    provider[+transport]://[authority][/prefix/]model[?param=value]

- `openai:///gpt-4o`
- `anthropic:///claude-opus-4-5`
- `claude:///claude-sonnet-4-5`
- `gemini:///gemini-2.5-flash-lite`
- `ollama://localhost:11434/llama3`

`claude://...` is an alias for `anthropic://...`.

The model name is always the final path segment. Authority and prefix are
optional; omitting both gives the triple-slash form (`provider:///model`).

Provider identity comes from the scheme, not the upstream hostname. A layered
client factory still treats
`openai+https://openrouter.ai/api/v1/gpt-4o` as `Provider::OpenAi` for rate
limiting, retries, and tracing.

## Compatible Endpoints

To route through a third-party or private host, embed the base URL in the
authority and path, and use `+transport` to make the scheme explicit.

- `openai+https://openrouter.ai/api/v1/gpt-4o?api_key_env=OPENROUTER_API_KEY`
- `anthropic+https://anthropic-proxy.example/v1/claude-sonnet-4-5?api_key_env=ANTHROPIC_PROXY_KEY`
- `ollama+https://ollama.example/qwen3:8b?api_key_env=OLLAMA_API_KEY`
- `ollama://localhost:11434/qwen3:8b`

Auth is resolved in this order:

1. `api_key_env` from the query string
2. the provider default environment variable

Provider defaults are:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OLLAMA_API_KEY`

`api_key_env` is resolved when the model URL is parsed. If the named variable is
missing, client creation fails early instead of silently falling back.

### Protocol Boundaries

- `openai[+transport]://...` uses the OpenAI Responses API. Pravah calls `/responses` and `/embeddings` relative to the base prefix.
- `anthropic[+transport]://...` and `claude[+transport]://...` use the Anthropic Messages API. Pravah calls `/messages` relative to the base prefix.
- `ollama[+transport]://...` uses `{prefix}/v1/chat/completions` for generation and `{prefix}/api/embed` for embeddings. Auth is optional and only sent when a key is present.
- `gemini://...` uses the native Gemini adapter. Custom base URLs are not supported for Gemini.

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

Attach tools to an agent using `.tool::<A, I, O>()` on the `FlowGraph` builder.
`I` is the tool's input type (its `schema_name()` becomes the tool name) and `O`
is the output type. Register a matching `work` handler for `I` → `O` separately.

```rust
impl Flow for PlannerInput {
    type Output = Plan;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .agent::<PlannerInput>()
            .tool::<PlannerInput, ReadNoteInput, ReadNoteOutput>()
            .work(read_note_handler)
    }
}
```

Work handlers return `Result<O, FlowError>` where `O` implements the `ToolOutput`
trait. Override `ToolOutput::to_message` on your output type when you need to
return attachments alongside the JSON payload.

### Tool Errors

Return `Err(ToolError::Other(msg))` from a work handler to report a failure.
Pravah always propagates work-node errors to the caller as `FlowError::Tool`.

Two error variants abort the flow without going back to the model:

| Variant                          | Behaviour                                       |
| -------------------------------- | ----------------------------------------------- |
| `ToolError::PathEscape(p)`       | Path escaped the working directory — hard abort |
| `ToolError::ForbiddenCommand(c)` | Command not on the allow-list — hard abort      |

All other variants (including `ToolError::Other`) surface as `FlowError::Tool`
and terminate the run.

To cap how many times the LLM may call a given tool in one agent execution,
use `AgentConfig::with_max_tool_calls(n)` together with
`with_loop_break_message(msg)`. When the limit is reached Pravah returns
`msg` to the LLM as the tool result instead of calling the handler, letting the
model conclude or take a different path.

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

| Variant                                  | Use                                                                |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `Attachment::Inline { mime_type, data }` | Base64-encoded inline data                                         |
| `Attachment::File { mime_type, path }`   | File under the current `working_dir`, materialized before dispatch |
| `Attachment::Url { mime_type, url }`     | Public URL                                                         |

`Attachment::File` is resolved through `Context::resolve`, so it stays confined
to the configured working directory.

### Initial User Attachments

Override `Agent::to_message(self, ctx)` when the first user turn should include
attachments.

See [../examples/image_prompt.rs](../examples/image_prompt.rs) for a complete
vision example built around `Attachment::File`.

### Tool Attachments

Override `ToolOutput::to_message` on the output type to include attachments
alongside the JSON payload:

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

### Provider Attachment Support

Attachments are materialized before provider dispatch, then translated into each
provider's wire format.

| Provider  | Inline / File                                | URL                                         |
| --------- | -------------------------------------------- | ------------------------------------------- |
| Anthropic | Image-only blocks on this path               | Image-only blocks on this path              |
| OpenAI    | Image-only `input_image` items on this path  | Image-only `input_image` items on this path |
| Gemini    | `Part::InlineData` with arbitrary mime types | `Part::FileData` with arbitrary mime types  |
| Ollama    | Image-only `image_url` parts on this path    | Image-only `image_url` parts on this path   |

## Example Map

- [../examples/linear_flow.rs](../examples/linear_flow.rs): minimal typed agent
- [../examples/image_prompt.rs](../examples/image_prompt.rs): initial image upload
- [../examples/debate.rs](../examples/debate.rs): multi-agent client usage across branches

If you are done with provider and tool setup, continue with
[flows.md](flows.md) for graph semantics, suspension, snapshots, and history.
