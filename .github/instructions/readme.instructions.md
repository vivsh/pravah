---
applyTo: "README.md"
---

# README Authoring Instructions

This document defines the intent, structure, style, and maintenance rules for
`README.md`.

It does **not** contain README content. It exists solely to guide future edits
and ensure the README remains focused, compelling, accurate, and consistent with
the rest of the documentation.

The README is the public front door of Pravah. Most readers will decide within
30 seconds whether to continue reading, try the library, or leave.

The goal is not to explain everything.

The goal is to make the reader understand:

1. What Pravah is.
2. Why it exists.
3. How it differs from other workflow or agent frameworks.
4. How to get started.
5. Where to learn more.

Anything beyond that belongs in the dedicated documentation pages.

---

# Documentation Structure

Pravah documentation intentionally consists of a small number of focused pages.

## README.md

The README is a concise introduction.

It should be readable in a few minutes and should fit within roughly two to
three printed pages.

It introduces the concepts and provides a minimal example.

It should not attempt to document every feature.

## docs/flows.md

The canonical reference for:

- execution model
- node types
- graph construction
- split and merge
- nested flows
- suspend and resume
- snapshots
- runtime behavior

Anything related to flow semantics belongs here rather than in the README.

## docs/clients.md

The canonical reference for:

- agents
- providers
- tools
- attachments
- model configuration
- structured outputs
- client APIs

Anything related to model integration belongs here rather than in the README.

---

# Reader Journey

The README should move the reader through four stages.

## 1. Recognition

The reader immediately understands:

- this is a Rust library
- it executes typed flow graphs
- it is stepwise and transactional
- it is not only for AI agents

The reader should understand the core idea before reaching the first code block.

## 2. Differentiation

The reader should quickly understand what makes Pravah different.

Prefer architectural distinctions over feature lists.

Avoid competing on:

- provider count
- tool count
- model support
- buzzwords

Emphasize:

- explicit execution
- deterministic progression
- bounded execution steps
- suspend and resume
- snapshot and replay
- strongly typed transitions

## 3. Credibility

Demonstrate that the design is coherent.

Use a small runnable example.

Show real API usage.

Avoid marketing language.

Avoid unsupported claims.

## 4. Action

Every reader should know the next step.

Direct them toward:

- installation
- examples
- flows documentation
- client documentation
- docs.rs

---

# Core Positioning

Pravah is fundamentally a stepwise transactional flow engine.

Agentic workflows are one application of the model.

They are not the model itself.

Every major section should reinforce one or more of the following ideas:

- one bounded step at a time
- explicit execution
- deterministic state transitions
- suspend and resume
- snapshot and replay
- type-safe graph composition

Provider integrations, tools, and multi-agent workflows are supporting features.

They should never become the primary message of the README.

---

# README Structure

The README should contain the following sections and only these sections unless
explicitly requested otherwise.

## Badge Row

One line of badges:

- crates.io version
- docs.rs
- license

Keep links accurate.

## Title and Tagline

The title is:

```text
# Pravah
```

Follow with transliteration and pronunciation.

The tagline should describe Pravah as a Rust library and should contain the
concept of flow.

Keep it concise.

## Opening Paragraph

The most important paragraph in the README.

It must answer:

- What is Pravah?
- Why does it exist?
- What problem does it solve?

The reader should understand the core idea before reading any other section.

Use concrete language.

Prefer technical clarity over excitement.

Good:

> Pravah is a stepwise transactional flow engine for Rust.

Bad:

> Pravah is a revolutionary next-generation AI orchestration framework.

## Why Pravah

Explain the execution model.

Contrast explicit orchestration with implicit orchestration.

Explain what a single call to `next()` represents.

Examples:

- one LLM interaction
- one tool batch
- one transform
- one branch transition
- one merge transition

Describe when Pravah is useful.

Examples:

- resumable workflows
- long-running processes
- human-in-the-loop systems
- stateful AI applications
- transactional execution

## Mental Model

Show the simplest possible pipeline diagram.

The diagram should communicate:

- typed transitions
- branching
- merging
- suspension
- completion

Introduce the one-type-per-node invariant.

Keep the explanation short.

## Installation

Show:

- normal dependency configuration
- feature-minimal configuration

Use the currently published version.

Keep feature examples synchronized with Cargo.toml.

## Getting Started

Provide a complete but minimal example.

The example should demonstrate:

- one `Agent`
- one `work` node
- one `Flow`

The example should remain short enough to understand in a single screen.

Prefer clarity over completeness.

## Read Next

Always present documentation links in this order:

1. `docs/clients.md`
2. `docs/flows.md`
3. `examples/`
4. docs.rs

## Examples

Provide a table containing every example.

The table should be ordered by learning progression rather than filename.

Descriptions should be short phrases.

Avoid explanations longer than one sentence.

## When To Use Pravah

Provide two short sections.

### Use Pravah When

Describe appropriate use cases.

### Do Not Use Pravah When

Explicitly state what Pravah is not.

Mention:

- queue systems
- distributed schedulers
- background task runners
- durable storage layers

Avoid creating unrealistic expectations.

## License

Use the standard dual-license statement.

Do not add commentary.

---

# Writing Style

The README should feel like a technical introduction written by an engineer.

Not marketing copy.

Not a reference manual.

Not a research paper.

## Tone

Use:

- direct language
- precise language
- concrete examples

Avoid:

- hype
- exaggeration
- slogans
- sales language

## Sentence Length

Prefer short sentences.

Target fewer than twenty words per sentence.

Avoid unnecessary clauses.

## Vocabulary

Favor simple technical terms.

Avoid:

- revolutionary
- next-generation
- cutting-edge
- game-changing
- enterprise-grade
- industry-leading
- powerful

If an advantage is claimed, explain why.

Good:

> Each call to `next()` executes exactly one bounded step.

Bad:

> Pravah provides predictable execution.

## Honesty

Never claim features that do not exist.

Never imply distributed execution if execution is local.

Never imply durability if persistence is provided by the caller.

Never describe design goals as implemented features.

---

# Consistency Checks

Before completing any README edit, verify the following.

## Version

The version string in the `[dependencies]` snippet must match the `version`
field in `Cargo.toml`.

Do not leave a stale version after a release.

## Examples Table

Every `.rs` file under `examples/` must have a corresponding row in the
examples table.

If a new example is added to `examples/`, add a row.

If an example is removed, remove its row.

## Node Types

Node type names and builder method names in the Mental Model section must
match the table in `docs/flows.md`.

Do not introduce node names that are not present in that reference.

## Documentation Links

All relative links to `docs/clients.md`, `docs/flows.md`, and files under
`examples/` must resolve from the repository root.

---

# Narrative Guidelines

The README should read as a coherent story.

Preferred flow:

1. Problem
2. Existing approaches
3. What Pravah changes
4. Mental model
5. Minimal example
6. Further reading

Sections should feel connected.

Avoid presenting the README as a disconnected collection of features.

---

# Cross-Reference Rules

Whenever README.md is updated, verify consistency with:

## Cargo.toml

Ensure all version strings match published versions.

## docs/flows.md

Treat this file as the canonical source for:

- node types
- execution semantics
- flow behavior

README summaries must not contradict it.

## docs/clients.md

Treat this file as the canonical source for:

- providers
- tools
- attachments
- agent configuration

README summaries must not contradict it.

## examples/

Every example file must appear in the examples table.

If a new example is added, update the table.

---

# Success Criteria

A reader who spends less than five minutes with the README should understand:

- what Pravah is
- how execution works at a high level
- why execution is stepwise
- why type-safe flows matter
- how to run the first example
- where to continue learning

The reader should leave with curiosity, not confusion.

The README should encourage further exploration without attempting to teach the entire library.
