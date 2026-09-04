# Pravah

[![Crates.io](https://img.shields.io/crates/v/pravah)](https://crates.io/crates/pravah)
[![docs.rs](https://img.shields.io/docsrs/pravah)](https://docs.rs/pravah)
[![License](https://img.shields.io/crates/l/pravah)](LICENSE-MIT)

**Durable workflows for Rust—agentic when you need them, ordinary when you
don't.**

_Pravah_ (प्रवाह, _pruh-VAH_) means “flow” or “current”.

Important work rarely fits neatly inside one request. It waits for an approval,
calls an unreliable service, fans out across many records, asks an agent to use
tools, or pauses until a person supplies the missing decision. Pravah lets that
work advance in clear steps, preserve its progress, and continue later.

Pravah is built for applications that want durable execution without giving up
control. Your application decides when a workflow runs, where its state is
stored, and how failures are retried. Pravah supplies the typed workflow and
the state needed to resume it.

## Where Pravah Fits

Use Pravah for workflows such as:

- **Approvals and human review.** Prepare a case, gather evidence, pause for a
  decision, and resume from exactly that point.
- **Long-running application work.** Coordinate enrichment, validation,
  imports, notifications, and external service calls across process restarts.
- **Tool-using agents.** Give an agent a typed set of application tools, choose
  those tools per invocation, and place practical limits on model turns and
  tool use.
- **Multi-stage AI pipelines.** Compose researchers, reviewers, classifiers,
  and writers as reusable parts of a larger workflow instead of hiding the
  process inside one model call.
- **Mixed human, software, and AI work.** Use ordinary Rust functions, async
  operations, agents, and suspension points in the same flow.
- **Repeatable collection processing.** Apply a reusable child workflow to
  every item while retaining explicit progress.

The same model works for a small three-step task and a workflow that may remain
unfinished for days.

## Why Pravah

- **Durable progress.** Snapshot a running workflow, store it with your
  application data, and restore it when work can continue.
- **Typed composition.** Inputs and outputs remain Rust types across maps,
  async work, branches, reusable flows, collections, agents, and resumptions.
- **Application-controlled execution.** Advance work deliberately rather than
  handing ownership to a hidden scheduler or background loop.
- **First-class agentic programming.** Agents participate as typed workflow
  steps. Their configuration can depend on invocation data and application
  context, and their tools can themselves be functions or complete flows.
- **Human intervention without a separate system.** Suspension and resumption
  are ordinary parts of a workflow, making review and approval natural.
- **Explicit operational boundaries.** Pravah does not pretend to provide
  storage, scheduling, or exactly-once external effects. Those decisions stay
  with the application that understands them.

## Installation

```toml
[dependencies]
pravah = "0.4.11"
```

## A Typed Workflow Feels Like Rust

Flows are ordinary functions. They are easy to reuse, nest, test, and read:

```rust
use pravah::graph::Flow;

fn verify(root: Flow<Claim>) -> Flow<VerifiedClaim> {
    root
        .map(normalize_claim)
        .work(fetch_evidence)
        .map(score_evidence)
}

fn review(root: Flow<Submission>) -> Flow<Decision> {
    root
        .map(extract_claim)
        .flow(verify)
        .suspend::<Decision>()
}
```

The flow can pause after verification, be saved by the application, and resume
when a reviewer returns a `Decision`.

Agents use the same function-based style:

```rust
use pravah::graph::{Agent, Flow, Toolset};

fn research(root: Flow<Question>) -> Flow<Answer> {
    root.agent(researcher)
}

fn researcher(root: Agent<Question>) -> Agent<Answer> {
    root
        .tools(research_tools)
        .configure(configure_researcher)
}

fn research_tools(root: Toolset) -> Toolset {
    root
        .tool(search)
        .flow(verify_source)
}
```

The agent's model, instructions, memory, available tools, and practical budgets
can be selected for each invocation. Tool calls remain visible workflow work,
and a tool can be a small async Rust function or a reusable Pravah flow.

## Durability Is a Partnership

Pravah makes workflow progress serializable; it does not choose a database or
run a scheduler for you. A typical application:

1. advances the workflow by one step;
2. saves its snapshot alongside application state;
3. schedules the next step or waits for an external event;
4. restores the workflow and continues.

This separation works equally well in a web service, worker process, desktop
application, command-line tool, or queue consumer.

Retries are also application policy. If external work may be repeated after a
failure, make it idempotent or deduplicate it using application-owned keys.

## Examples

| Example | What it demonstrates |
| --- | --- |
| [`graph_typed_composition`](examples/graph_typed_composition.rs) | Reusing a typed child flow at multiple call sites |
| [`graph_snapshot_resume`](examples/graph_snapshot_resume.rs) | Saving a suspended workflow, restoring it, and resuming |
| [`graph_typed`](examples/graph_typed.rs) | Maps, branches, local state, child flows, and collections |
| [`graph_agent_budgets`](examples/graph_agent_budgets.rs) | Agent and per-tool budgets without custom policy code |
| [`story`](examples/story.rs) | A substantial multi-agent creative workflow |

See the [complete example index](examples/README.md) for prerequisites and run
commands.

## Pravah Deliberately Does Not Replace Your Application

Pravah is not a queue, database, distributed scheduler, application server, or
promise of exactly-once side effects. It is the durable workflow engine inside
your application. You retain ownership of deployment, persistence, retries,
authentication, and operational policy.

That boundary is intentional: Pravah makes the workflow explicit without
taking the application away from you.

## Learn More

- [`docs/graph.md`](docs/graph.md): typed execution, snapshots, restoration,
  and operational responsibilities
- [`docs/clients.md`](docs/clients.md): model providers, agents, tools, and
  attachments
- [`docs/chat.md`](docs/chat.md): a simple multi-turn conversation
- [docs.rs](https://docs.rs/pravah): API reference

Pravah is currently on the `0.4.x` line. The API may continue to evolve while
the workflow experience is refined toward a stable release.

## License

Licensed under either the MIT License or Apache License 2.0, at your option.
