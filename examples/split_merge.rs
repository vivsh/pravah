//! # Example — Split / Merge Flow
//!
//! Demonstrates fanning a single input out to three independent agent tracks
//! and collecting all results into one merged output:
//!
//! ```text
//!             ┌── TechTrack  ──agent──► TechAnalysis  ──┐
//! Proposal ─split── MktTrack  ──agent──► MktAnalysis   ──┼──merge──► Brief (terminal)
//!             └── RiskTrack  ──agent──► RiskAnalysis  ──┘
//! ```
//!
//! `split` fans the input out to an arbitrary number of typed branches in one
//! step — no chaining needed. `merge` fires once every branch has a value and
//! produces a single output. Both support arities 2–16.
//!
//! The flow is single-threaded: the runner processes one agent call per
//! `next()` invocation. Split and merge model information shape, not
//! parallelism.
//!
//! ## Running
//!
//! ```shell
//! GEMINI_API_KEY=<key> cargo run --example split_merge
//! ```

use pravah::flows::{Agent, AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Proposal {
    title: String,
    description: String,
}

// Split branches ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TechTrack {
    description: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MktTrack {
    description: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RiskTrack {
    description: String,
}

// Agent outputs ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TechAnalysis {
    feasibility: String,
    effort: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MktAnalysis {
    opportunity: String,
    competitors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RiskAnalysis {
    risks: Vec<String>,
    mitigation: String,
}

// Merge output (terminal) ─────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Brief {
    tech: String,
    market: String,
    risk: String,
    recommendation: String,
}

// ── Agents ────────────────────────────────────────────────────────────────────

impl Agent for TechTrack {
    type Output = TechAnalysis;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a software engineer. Assess the feasibility and implementation \
             effort for the described feature. Be concise.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

impl Agent for MktTrack {
    type Output = MktAnalysis;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a market analyst. Identify the market opportunity and name \
             three direct competitors.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

impl Agent for RiskTrack {
    type Output = RiskAnalysis;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a risk analyst. List the top three risks and propose one \
             concrete mitigation strategy.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

// ── Split and merge handlers ──────────────────────────────────────────────────

fn split_proposal(p: Proposal) -> (TechTrack, MktTrack, RiskTrack) {
    (
        TechTrack { description: p.description.clone() },
        MktTrack { description: p.description.clone() },
        RiskTrack { description: p.description },
    )
}

fn merge_brief((tech, mkt, risk): (TechAnalysis, MktAnalysis, RiskAnalysis)) -> Brief {
    Brief {
        tech: format!("{} (effort: {})", tech.feasibility, tech.effort),
        market: format!(
            "{} Competitors: {}",
            mkt.opportunity,
            mkt.competitors.join(", ")
        ),
        risk: format!("{} Mitigation: {}", risk.risks.join("; "), risk.mitigation),
        recommendation: "Review all three tracks before committing to the roadmap.".to_owned(),
    }
}

// ── Flow ──────────────────────────────────────────────────────────────────────

impl Flow for Proposal {
    type Output = Brief;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .split(split_proposal)
            .agent::<TechTrack>()
            .agent::<MktTrack>()
            .agent::<RiskTrack>()
            .merge(merge_brief)
            .build()
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let ctx = Context::new(FlowConf::default());

    let input = Proposal {
        title: "AI-powered code review extension".to_owned(),
        description: "A VS Code extension that uses an LLM to review pull requests, \
                      detect security vulnerabilities, and suggest refactors in real time."
            .to_owned(),
    };

    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(brief) => {
                println!("## Technical\n{}\n", brief.tech);
                println!("## Market\n{}\n", brief.market);
                println!("## Risk\n{}\n", brief.risk);
                println!("## Recommendation\n{}", brief.recommendation);
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
