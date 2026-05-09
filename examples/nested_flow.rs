//! # Example 3 — Nested Flow
//!
//! Demonstrates composing flows: an outer flow delegates part of its work to
//! a fully independent inner flow.
//!
//! ```text
//! Outer flow
//! ──────────────────────────────────────────────────────────────
//! BlogRequest ──work──► ResearchQuery ──work (inner flow)──► ResearchResult ──agent──► FinalArticle (terminal)
//!
//! Inner flow  (ResearchFlow)
//! ──────────────────────────────────────────────────────────────
//! ResearchQuery ──agent──► ResearchResult (terminal)
//! ```
//!
//! The `work` node that bridges the two flows runs `ResearchFlow` to
//! completion synchronously inside its async closure.  The inner flow is
//! completely unaware it is being hosted by the outer flow.
//!
//! ## Running
//!
//! ```shell
//! GEMINI_API_KEY=<key> cargo run --example nested_flow
//! ```

use pravah::flows::{Agent, AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, RunOut};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Shared type ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchQuery {
    query: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchResult {
    findings: String,
}

// ── Inner flow ────────────────────────────────────────────────────────────────

impl Agent for ResearchQuery {
    type Output = ResearchResult;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a research assistant. Answer the user's query with a concise \
             paragraph of factual findings.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

impl Flow for ResearchQuery {
    type Output = ResearchResult;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<ResearchQuery>()
            .build()
    }
}

// ── Outer flow types ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BlogRequest {
    topic: String,
    audience: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalArticle {
    title: String,
    body: String,
}

// ── Writer agent (used in outer flow) ─────────────────────────────────────────

impl Agent for ResearchResult {
    type Output = FinalArticle;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a blog writer. Using the research findings provided, write a \
             short, engaging blog post with a title and two paragraphs.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

// ── Node handlers ─────────────────────────────────────────────────────────────

async fn derive_query(req: BlogRequest, _ctx: Context) -> Result<ResearchQuery, FlowError> {
    Ok(ResearchQuery {
        query: format!("{} (for {})", req.topic, req.audience),
    })
}

async fn run_research(query: ResearchQuery, ctx: Context) -> Result<ResearchResult, FlowError> {
    let mut rt = FlowRuntime::new(query)?;
    loop {
        match rt.next(ctx.clone()).await? {
            RunOut::Continue => {}
            RunOut::Done(result) => return Ok(result),
            RunOut::Suspend { value, tool_id } => {
                return Err(FlowError::AgentError(format!(
                    "inner flow suspended unexpectedly at '{tool_id}': {value}"
                )));
            }
        }
    }
}

// ── Outer flow ────────────────────────────────────────────────────────────────

impl Flow for BlogRequest {
    type Output = FinalArticle;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(derive_query)
            .work(run_research)
            .agent::<ResearchResult>()
            .build()
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let ctx = Context::new(FlowConf::default());

    let input = BlogRequest {
        topic: "Why Rust's ownership model matters".to_string(),
        audience: "developers new to systems programming".to_string(),
    };

    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            RunOut::Continue => {}
            RunOut::Done(article) => {
                println!("# {}\n\n{}", article.title, article.body);
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
