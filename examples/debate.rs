//! # Example 5 — Debate Analyser
//!
//! Takes a claim from stdin and runs it through a multi-stage pipeline:
//!
//! ```text
//! Outer flow
//! ─────────────────────────────────────────────────────────────────────────────
//!                  ┌─ ProRequest  ──agent──► ProArguments  ─┐
//! DebateInput ─fork┤                                         ├join──► DebateDraft ──flow──► DebateReport
//!                  └─ ConRequest  ──agent──► ConArguments  ─┘
//!
//! Inner flow  (DebateDraft → DebateReport)
//! ─────────────────────────────────────────────────────────────────────────────
//! DebateDraft ──agent──► DebateVerdict ──work──► DebateReport
//! ```
//!
//! The outer flow forks into two independent agent branches, rejoins them into
//! a draft, then delegates verdict writing to a nested sub-flow. The inner flow
//! keeps the verdict agent isolated so it can be tested or reused on its own.
//!
//! ## Running
//!
//! ```shell
//! GEMINI_API_KEY=<key> cargo run --example debate
//! # or supply the claim directly:
//! GEMINI_API_KEY=<key> cargo run --example debate -- "Remote work improves productivity"
//! ```

use std::io::{self, Write};

use pravah::flows::{Agent, AgentConfig, Flow, FlowError, FlowGraph, FlowRuntime, FlowStep};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Input ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DebateInput {
    claim: String,
}

// ── Fork children ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ProRequest {
    claim: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ConRequest {
    claim: String,
}

// ── Agent outputs ─────────────────────────────────────────────────────────────

/// Pro arguments; echoes the original claim so the join can forward it.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ProArguments {
    claim: String,
    points: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ConArguments {
    points: Vec<String>,
}

// ── Inner-flow types ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DebateDraft {
    claim: String,
    pro_points: Vec<String>,
    con_points: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DebateVerdict {
    winner: String,
    reasoning: String,
    caveats: Vec<String>,
}

// ── Final output ──────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DebateReport {
    markdown: String,
}

// ── Agents ────────────────────────────────────────────────────────────────────

impl Agent for ProRequest {
    type Output = ProArguments;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a skilled debater building the strongest possible case FOR a claim. \
             Return the original claim verbatim in the `claim` field and provide \
             3–5 concise supporting arguments in `points`.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

impl Agent for ConRequest {
    type Output = ConArguments;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are a skilled debater building the strongest possible case AGAINST a claim. \
             Provide 3–5 concise counter-arguments in `points`.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

impl Agent for DebateDraft {
    type Output = DebateVerdict;

    fn build() -> AgentConfig {
        AgentConfig::new(
            "You are an impartial debate judge. Weigh the pro and con arguments provided \
             for the claim and deliver a balanced verdict. \
             Set `winner` to 'pro', 'con', or 'draw'. \
             Provide clear `reasoning` and note any important `caveats`.",
            "gemini://gemini-2.5-flash-lite",
        )
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────────

fn split_claim(input: DebateInput, _ctx: Context) -> Result<(ProRequest, ConRequest), FlowError> {
    Ok((
        ProRequest { claim: input.claim.clone() },
        ConRequest { claim: input.claim },
    ))
}

fn merge_arguments(
    pro: ProArguments,
    con: ConArguments,
    _ctx: Context,
) -> Result<DebateDraft, FlowError> {
    Ok(DebateDraft {
        claim: pro.claim,
        pro_points: pro.points,
        con_points: con.points,
    })
}

async fn format_verdict(verdict: DebateVerdict, _ctx: Context) -> Result<DebateReport, FlowError> {
    let caveats_md = if verdict.caveats.is_empty() {
        String::new()
    } else {
        format!(
            "\n\n### Caveats\n{}",
            verdict.caveats.iter().map(|c| format!("- {c}")).collect::<Vec<_>>().join("\n")
        )
    };
    let markdown = format!(
        "## Verdict: **{}**\n\n{}{}",
        verdict.winner.to_uppercase(),
        verdict.reasoning,
        caveats_md,
    );
    Ok(DebateReport { markdown })
}

// ── Inner flow ────────────────────────────────────────────────────────────────

impl Flow for DebateDraft {
    type Output = DebateReport;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .agent::<DebateDraft>()
            .work(format_verdict)
            .build()
    }
}

// ── Outer flow ────────────────────────────────────────────────────────────────

impl Flow for DebateInput {
    type Output = DebateReport;

    fn build() -> Result<FlowGraph, FlowError> {
        FlowGraph::builder()
            .fork(split_claim)
            .agent::<ProRequest>()
            .agent::<ConRequest>()
            .join(merge_arguments)
            .flow::<DebateDraft>()
            .build()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn read_claim() -> String {
    let mut args = std::env::args().skip(1);
    if let Some(arg) = args.next() {
        return arg;
    }
    print!("Enter a claim to debate (or press Enter for a default): ");
    io::stdout().flush().ok();
    let mut line = String::new();
    io::stdin().read_line(&mut line).ok();
    let trimmed = line.trim().to_string();
    if trimmed.is_empty() {
        "Artificial intelligence will eventually replace most creative professions".to_string()
    } else {
        trimmed
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let claim = read_claim();
    println!("\nDebating: \"{claim}\"\n");

    let ctx = Context::new(FlowConf::default());
    let input = DebateInput { claim };
    let mut runtime = FlowRuntime::new(input)?;

    loop {
        match runtime.next(ctx.clone()).await? {
            FlowStep::Continue => {}
            FlowStep::Done(report) => {
                println!("{}", report.markdown);
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
