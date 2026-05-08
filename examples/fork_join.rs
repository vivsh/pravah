//! # Example 2 — Fork / Join Flow
//!
//! Demonstrates information splitting and recombining:
//!
//! ```text
//!                  ┌─ ResearchRequest ──agent──► ResearchFindings ─┐
//! ProjectRequest ──┤fork                                            ├join──► FinalReport (terminal)
//!                  └─ DesignRequest ────agent──► DesignProposal ───┘
//! ```
//!
//! The fork decomposes a project request into two independent tracks — a
//! research track and a design track.  Each track runs its own agent.  Once
//! both tracks have produced a result the join merges them into a single
//! `FinalReport`.
//!
//! The flow is single-threaded: the runner processes one agent call per
//! `next()` invocation.  Fork and join model information shape, not
//! parallelism.
//!
//! ## Running
//!
//! ```shell
//! GEMINI_API_KEY=<key> cargo run --example fork_join
//! ```

use pravah::flows::{Agent, Flow, FlowError, FlowGraph, FlowRuntime, RunOut};
use pravah::tools::ToolBox;
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ProjectRequest {
    title: String,
    description: String,
}

// Fork children ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchRequest {
    topic: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DesignRequest {
    feature: String,
}

// Agent outputs ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchFindings {
    summary: String,
    references: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DesignProposal {
    approach: String,
    trade_offs: Vec<String>,
}

// Join output (terminal) ──────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalReport {
    research: String,
    design: String,
    recommendation: String,
}

// ── Agents ────────────────────────────────────────────────────────────────────

impl Agent for ResearchRequest {
    type Output = ResearchFindings;

    fn preamble() -> String {
        "You are a research analyst. Given a topic, produce a concise summary \
         and a short list of relevant references."
            .to_string()
    }

    fn model_url() -> String {
        "gemini://gemini-2.5-flash-lite".to_string()
    }

    fn tool_box() -> ToolBox {
        ToolBox::builder().build()
    }
}

impl Agent for DesignRequest {
    type Output = DesignProposal;

    fn preamble() -> String {
        "You are a software architect. Given a feature description, propose an \
         implementation approach and list the main trade-offs."
            .to_string()
    }

    fn model_url() -> String {
        "gemini://gemini-2.5-flash-lite".to_string()
    }

    fn tool_box() -> ToolBox {
        ToolBox::builder().build()
    }
}

// ── Flow ──────────────────────────────────────────────────────────────────────

impl Flow for ProjectRequest {
    type Output = FinalReport;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            // Split the project request into two independent tracks.
            .fork::<ProjectRequest, ResearchRequest, DesignRequest, _>(
                |req, _ctx| {
                    Ok((
                        ResearchRequest { topic: req.title.clone() },
                        DesignRequest { feature: req.description },
                    ))
                },
            )
            // Each track runs its own agent.
            .agent::<ResearchRequest>()
            .agent::<DesignRequest>()
            // Once both tracks are done, merge into the final report.
            .join::<ResearchFindings, DesignProposal, FinalReport, _>(
                |research, design, _ctx| {
                    Ok(FinalReport {
                        research: research.summary,
                        design: design.approach,
                        recommendation: format!(
                            "Proceed with design. Key trade-offs: {}",
                            design.trade_offs.join("; ")
                        ),
                    })
                },
            )
            .build()
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new(FlowConf::default());

    let input = ProjectRequest {
        title: "Distributed rate limiter".to_string(),
        description: "Token-bucket algorithm backed by Redis, with per-user \
                      and per-route limits, suitable for a multi-region API."
            .to_string(),
    };

    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            RunOut::Continue => {}
            RunOut::Done(report) => {
                println!("## Research\n{}\n", report.research);
                println!("## Design\n{}\n", report.design);
                println!("## Recommendation\n{}", report.recommendation);
                break;
            }
            RunOut::Suspend { value, tool_id } => {
                eprintln!("Unexpected suspension at '{tool_id}': {value}");
                break;
            }
        }
    }

    Ok(())
}
