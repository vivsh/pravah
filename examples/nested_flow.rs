//! # Example 3 — Nested Flow
//!
//! Demonstrates composing flows: an outer flow delegates part of its work to
//! a fully independent inner flow using the `.flow::<F>()` builder node.
//!
//! ```text
//! Outer flow
//! ──────────────────────────────────────────────────────────────
//! BlogRequest ──work──► ResearchQuery ──flow──► ResearchResult ──agent──► FinalArticle (terminal)
//!
//! Inner flow  (ResearchQuery flow)
//! ──────────────────────────────────────────────────────────────
//! ResearchQuery ──agent──► ResearchResult (terminal)
//! ```
//!
//! The `.flow::<ResearchQuery>()` node runs the inner flow to completion
//! step-by-step and forwards its typed output into the outer flow. The inner
//! flow is completely unaware it is being hosted by the outer flow.
//!
//! ## Running
//!
//! ```shell
//! GEMINI_API_KEY=<key> cargo run --example nested_flow
//! ```

use pravah::flows::{Agent, AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
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

// Node handlers

async fn derive_query(req: BlogRequest, _ctx: Context) -> Result<ResearchQuery, FlowError> {
    Ok(ResearchQuery {
        query: format!("{} (for {})", req.topic, req.audience),
    })
}

// Outer flow

impl Flow for BlogRequest {
    type Output = FinalArticle;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .work(derive_query)
            .flow::<ResearchQuery>()
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
            FlowStep::Continue => {}
            FlowStep::Done(article) => {
                println!("# {}\n\n{}", article.title, article.body);
                break;
            }
            FlowStep::Suspend(_) => {
                eprintln!("Unexpected suspension");
                break;
            }
        }
    }

    Ok(())
}
