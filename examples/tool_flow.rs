//! Compatibility-only legacy tool flow backed by a nested flow.
//!
//! Requires `GEMINI_API_KEY`.
//!
//! The outer agent summarises an article. When it needs to verify a claim it
//! calls the `verify_claim` tool, which is itself a full sub-flow backed by
//! its own agent. The outer agent collects those results and includes them in
//! the final summary.
//!
//! Key fluent call:
//! ```text
//! root
//!     .agent_with(|toolbox| toolbox.tool_flow::<VerifyClaim>())
//! ```
//! For a one-shot handler, use the low-level `tool_handler` on `Toolbox`.

mod support;

use pravah::legacy::{Agent, AgentConfig, Flow, FlowRuntime, FlowStep, Node};
use pravah::tools::ToolOutput;
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use support::ExampleError;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct VerifyClaim {
    /// The specific claim to verify.
    claim: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct VerificationResult {
    /// "supported", "refuted", or "unverifiable"
    verdict: String,
    /// One-sentence explanation.
    explanation: String,
}

impl ToolOutput for VerificationResult {}

impl Agent for VerifyClaim {
    type Output = VerificationResult;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a fact-checker. Given a claim, return a verdict of \
             \"supported\", \"refuted\", or \"unverifiable\", and a one-sentence \
             explanation. Be concise and objective.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}

impl Flow for VerifyClaim {
    type Output = VerificationResult;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent()
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ArticleRequest {
    article: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ArticleSummary {
    summary: String,
    fact_checks: Vec<String>,
}

impl Agent for ArticleRequest {
    type Output = ArticleSummary;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a summarising fact-checker. Read the article provided by the user. \
             For each significant factual claim, call the `verify_claim` tool to check it. \
             Once you have gathered the verification results, produce a concise summary of \
             the article and list the fact-check outcomes in `fact_checks`.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}

impl Flow for ArticleRequest {
    type Output = ArticleSummary;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        root.agent_with(|toolbox| toolbox.tool_flow::<VerifyClaim>())
    }
}

#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    dotenvy::dotenv().ok();
    let ctx = Context::new(FlowConf::default());

    let input = ArticleRequest {
        article: "The Eiffel Tower was completed in 1889 and stands 330 metres tall. \
                  It was originally intended as a temporary structure and was almost \
                  demolished in 1909. Today it attracts over 7 million visitors per \
                  year, making it the most visited paid monument in the world."
            .to_string(),
    };

    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(result) => {
                println!("## Summary\n{}\n", result.summary);
                if !result.fact_checks.is_empty() {
                    println!("## Fact Checks");
                    for fc in &result.fact_checks {
                        println!("- {fc}");
                    }
                }
                break;
            }
            FlowStep::Suspend(_) => {
                return Err(ExampleError::unexpected("tool flow unexpectedly suspended"));
            }
        }
    }

    Ok(())
}
