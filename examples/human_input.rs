//! # Example — Human-in-the-Loop via Tool Call
//!
//! Demonstrates Gemini using `HumanInput` as a tool call to pause and collect
//! a decision from the human before producing its final answer.
//!
//! ```text
//! BlogRequest ──agent(HumanInput tool)──► HumanInput sub-flow ──► FinalResult
//! ```
//!
//! The agent drafts a short blog post, then calls the `HumanInput` tool with
//! the draft and three choices. Because [`CliMode`] is in context the sub-flow
//! reads the answer from stdin. The `HumanOutput` is returned to the agent as
//! a tool result and the agent submits the [`FinalResult`].
//!
//! ## Running
//!
//! ```shell
//! GEMINI_API_KEY=<key> cargo run --example human_input
//! ```

use pravah::deps::Deps;
use pravah::flows::{Agent, AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
use pravah::tools::ToolBox;
use pravah::{CliMode, Context, FlowConf, HumanInput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BlogRequest {
    /// The topic to write about.
    topic: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalResult {
    /// A short message describing the outcome based on the human's decision.
    outcome: String,
}

// ── Agent ─────────────────────────────────────────────────────────────────────

impl Agent for BlogRequest {
    type Output = FinalResult;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a blog writer. When the user gives you a topic:\n\
             1. Draft a short (3–4 sentence) blog post on that topic.\n\
             2. Call the `HumanInput` tool. Pass the draft in `prompt` and offer \
                three choices: 'Approve and publish', 'Request a revision', 'Discard'. \
                Set `allow_other` to false.\n\
             3. Once you have the HumanOutput, submit a FinalResult:\n\
                - choice 0 → outcome: 'Published: <draft>'\n\
                - choice 1 → outcome: 'Revision requested.'\n\
                - choice 2 → outcome: 'Draft discarded.'",
            "gemini://gemini-2.5-flash-lite",
        )
        .with_tools(ToolBox::new().flow::<HumanInput>())
    }
}

// ── Flow ──────────────────────────────────────────────────────────────────────

impl Flow for BlogRequest {
    type Output = FinalResult;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder().agent::<BlogRequest>().build()
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let mut deps = Deps::default();
    deps.insert(Arc::new(CliMode));
    let ctx = Context::new(FlowConf::default()).with_deps(deps);

    let input = BlogRequest {
        topic: "why Rust is great for building AI pipelines".into(),
    };
    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(result) => {
                println!("\n{}", result.outcome);
                break;
            }
            FlowStep::Suspend(_) => {
                eprintln!("Unexpected suspension — is CliMode in context?");
                break;
            }
        }
    }

    Ok(())
}

