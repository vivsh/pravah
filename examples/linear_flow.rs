//! # Example 1 — Linear Flow
//!
//! Demonstrates a straight pipeline:
//!
//! ```text
//! SummariseRequest ──agent──► BulletPoints ──work──► Report (terminal)
//! ```
//!
//! The agent receives the raw text, distils it into bullet points, and a
//! deterministic `work` node formats those into a Markdown report — no second
//! LLM call needed.
//!
//! ## Running
//!
//! ```shell
//! GEMINI_API_KEY=<key> cargo run --example linear_flow
//! ```

use pravah::flows::{Agent, Flow, FlowError, FlowGraph, FlowRuntime, RunOut};
use pravah::tools::ToolBox;
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SummariseRequest {
    text: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BulletPoints {
    points: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Report {
    markdown: String,
}

// ── Agent ─────────────────────────────────────────────────────────────────────

impl Agent for SummariseRequest {
    type Output = BulletPoints;

    fn preamble() -> String {
        "You are a concise summariser. Extract the key points from the text \
         the user sends and return them as a JSON array of short strings."
            .to_string()
    }

    fn model_url() -> String {
        "gemini://gemini-2.5-flash-lite".to_string()
    }

    fn tool_box() -> ToolBox {
        ToolBox::builder().build()
    }
}

// ── Work node ─────────────────────────────────────────────────────────────────

async fn format_bullets(bullets: BulletPoints, _ctx: Context) -> Result<Report, FlowError> {
    let markdown = bullets.points.iter().map(|p| format!("- {p}")).collect::<Vec<_>>().join("\n");
    Ok(Report { markdown })
}

// ── Flow ──────────────────────────────────────────────────────────────────────

impl Flow for SummariseRequest {
    type Output = Report;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<SummariseRequest>()
            .work(format_bullets)
            .build()
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let ctx = Context::new(FlowConf::default());

    let input = SummariseRequest {
        text: "Rust is a systems programming language that runs blazingly fast, \
               prevents segfaults, and guarantees thread safety. It achieves memory \
               safety without a garbage collector through its ownership system, and \
               is used for everything from embedded systems to web assembly."
            .to_string(),
    };

    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            RunOut::Continue => {}
            RunOut::Done(report) => {
                println!("## Summary\n\n{}", report.markdown);
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
