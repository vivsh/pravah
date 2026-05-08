# Pravah

`Pravah` is a Rust library for building typed, stepwise agentic
information flows.

It is intentionally not a general workflow engine. A Pravah flow is a
single-threaded graph that moves information from one typed node to the next.
Each call to the runtime advances the graph by one transaction-sized step:
one LLM turn, one tool batch, one deterministic transform, one branch, one fork,
or one join. The caller decides when to persist state, where to store it, and
how to resume it.

The result is a small but capable foundation for agent systems that need:

- typed LLM inputs and outputs
- typed tools with JSON Schema definitions
- deterministic information transitions between agent turns
- human-in-the-loop suspend/resume
- non-linear flows through fork and join nodes
- nested flows for modular agent design
- validation before a flow runs
- provider-agnostic model clients
- caller-owned persistence and deployment policy

## The Core Idea

A flow graph is made of nodes. Each node consumes one input type and produces
another type. The input type is the node's identity.

This rule is critical:

**Within a single flow graph, a type can be the input for only one node.**

That means `PlanInput` can identify one agent node, one work node, one branch
node, one fork node, or one join participant, but not multiple nodes at once.
The builder rejects duplicate node identities. This removes routing ambiguity:
when a value of type `PlanInput` is present in flow state, there is exactly one
node that can consume it.

In exchange for this constraint, the runtime stays simple and predictable:
the active state is just a typed value, the next transition is unambiguous, and
flow progress can be checkpointed between steps by the caller.

## Why Types Matter

Pravah uses Rust types as the contract at every boundary:

- agent input structs define the shape of the first user message
- agent output types define the final result schema
- tool structs define LLM-callable arguments
- work, either, fork, and join handlers receive typed Rust values
- terminal outputs are deserialized into the flow's declared output type

The runtime stores values internally as JSON so state can be serialized, but
user code works with typed data.

## Transactional Stepping

The flow runtime is deliberately stepwise. A call to `next()` or `resume()`
does bounded work and then returns:

- an agent node performs one model call and handles the returned tool calls
- a work node runs one deterministic async transform
- an either node chooses one of two next values
- a fork node splits one value into two active values
- a join node waits until both parent values are available
- a suspend tool pauses the flow and returns a resume token to the caller

This makes it natural to persist state after each step, retry failed steps, or
hand control back to a UI, job runner, queue, or service boundary.

Persistence is caller-owned by design. `Pravah` defines serializable flow
state; it does not prescribe whether snapshots live in a file, database, queue,
object store, or memory.

## Non-Linear Information Flow

Not all useful agent flows are linear.

Pravah supports:

- `agent<A>()` for an LLM-backed node
- `work<From, Out>()` for deterministic async computation
- `either<From, A, B>()` for typed branching
- `fork<From, A, B>()` for splitting information into two branches
- `join<A, B, Out>()` for combining two branches once both are ready

Fork and join are not about parallel execution. They model information shape.
A single-threaded runner can still represent a non-linear graph where one
piece of information splits into independent branches and later recombines.

## Nested Flows

Flows are meant to compose. A flow has the same basic shape as a node: it
accepts a typed input, advances step by step, and eventually produces a typed
output. That makes nested flows the natural way to keep large agent systems
modular.

Use nested flows when a portion of an agent process has its own internal
structure but should look like one typed information transform to the parent
flow. For example:

- a planning flow can contain a research sub-flow
- a coding flow can contain a review-and-fix sub-flow
- an approval flow can contain a clarification sub-flow
- a larger product agent can reuse the same issue-triage sub-flow in multiple
  places

The same node-identity rule applies at each graph boundary: inside a flow
graph, each input type identifies exactly one node. Nested flows let you keep
that constraint local. Instead of flattening every detail into one large graph,
you can give a sub-flow its own typed internal graph and expose only its input
and output types to the surrounding system.

## Agents

An agent input type implements `Agent`. The input type identifies the node; the
associated `Output` type identifies the value produced when the agent exits.

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use pravah::flows::Agent;
use pravah::tools::ToolBox;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct PlannerInput {
    goal: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Plan {
    steps: Vec<String>,
}

impl Agent for PlannerInput {
    type Output = Plan;

    fn preamble() -> String {
        "You are a careful planning agent.".to_string()
    }

    fn model_url() -> String {
        "gemini://gemini-2.5-flash-lite".to_string()
    }

    fn tool_box() -> ToolBox {
        ToolBox::builder().build()
    }
}
```

If an agent has no tools, Pravah uses structured-output mode and asks the
provider for the declared output schema directly. If an agent has tools,
Pravah injects a typed exit sentinel so the model can submit the final value.

## Tools

A tool is a typed struct that implements `Tool`. Its fields become the JSON
Schema sent to the model.

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use pravah::context::Context;
use pravah::tools::{Tool, ToolError};

#[derive(Debug, Deserialize, JsonSchema)]
struct ReadNote {
    path: String,
}

#[derive(Debug, Serialize)]
struct ReadNoteOutput {
    content: String,
}

impl Tool for ReadNote {
    type Output = ReadNoteOutput;

    fn name() -> &'static str {
        "read_note"
    }

    fn description() -> &'static str {
        "Read a note from the working directory."
    }

    async fn call(self, ctx: Context) -> Result<Self::Output, ToolError> {
        let path = ctx.resolve(&self.path)?;
        let content = tokio::fs::read_to_string(path).await?;
        Ok(ReadNoteOutput { content })
    }
}
```

Tools receive a `Context`, which carries the working directory, command
allowlist, dependency container, and shared HTTP client.

## Suspend And Resume

Tools can request external input by returning `ToolError::suspend(value)`.
The runtime returns a suspension value and a tool id to the caller. The caller
can persist state, show the request to a user, wait for a webhook, or route it
through any other external system.

Later, the caller resumes the flow with the matching tool id and a JSON payload.
Resume continues the pending tool batch without making a new LLM call first.

This is useful for approval gates, missing credentials, clarification prompts,
payments, ticket handoffs, or any agent action that needs outside confirmation.

## Building A Flow

Implement `Flow` for the initial input type and return a validated builder.

```rust
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use pravah::context::Context;
use pravah::flows::{Flow, FlowError, FlowGraph};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalAnswer {
    text: String,
}

impl Flow for PlannerInput {
    type Output = FinalAnswer;

    fn build() -> pravah::flows::FlowBuilder {
        FlowGraph::builder()
            .agent::<PlannerInput>()
            .work::<Plan, FinalAnswer, _>(finish_plan)
            .entry(<Self as Flow>::node_id())
    }
}

fn finish_plan(plan: Plan, _ctx: Context) -> BoxFuture<'static, Result<FinalAnswer, FlowError>> {
    Box::pin(async move {
        Ok(FinalAnswer {
            text: plan.steps.join("\n"),
        })
    })
}
```

The builder validates:

- duplicate node identities
- missing entry node
- unreachable nodes
- nodes with no path to a terminal value
- invalid fork and join definitions
- branch definitions that would route both sides to the same type

## Running A Flow

`FlowRuntime` owns the graph, current state, conversation history, and model
factory. It advances one step at a time.

```rust
use pravah::context::Context;
use pravah::flows::{FlowRuntime, RunOut};

let ctx = Context::new(std::env::current_dir()?);
let mut runtime = FlowRuntime::new(PlannerInput {
    goal: "Write a migration plan".to_string(),
})?;

loop {
    match runtime.next(ctx.clone()).await? {
        RunOut::Continue => {
            // Persist state here if desired.
        }
        RunOut::Suspend { value, tool_id } => {
            // Persist state and collect external input.
            let input = serde_json::json!({ "approved": true });
            runtime.resume(ctx.clone(), (tool_id, input)).await?;
        }
        RunOut::Done(output) => {
            println!("{}", output.text);
            break;
        }
    }
}
```

## Clients

The client layer is provider-agnostic. Default features include native support
for OpenAI, Anthropic, Gemini, and Ollama. Model URLs select the backend, and
the scheme is authoritative even for custom model names:

- `gemini://gemini-2.5-flash-lite`
- `openai://gpt-4o`
- `anthropic://claude-sonnet-4-5`
- `claude://claude-opus-4-5`
- `ollama://localhost:11434/qwen3:8b`

`provider-genai` can be enabled as an optional experimental adapter for extra
providers, but the major providers do not depend on it.

Custom `ClientFactory` implementations can be injected for tests, local
providers, recording/replay, or hosted model gateways.

## When To Use Pravah

Use `Pravah` when you want agentic flows that are:

- type-directed
- inspectable
- resumable
- easy to test with fake clients
- explicit about information movement
- free from hidden background scheduling

Do not use it when you need a distributed workflow engine, parallel job
scheduler, queue processor, or durable storage system. Pravah can sit inside
those systems, but it does not try to replace them.
