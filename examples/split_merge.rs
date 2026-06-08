//! Split/merge example with three agent branches converging into one brief.

use pravah::flows::{Agent, AgentConfig, Flow, FlowRuntime, FlowStep, Node};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Proposal {
    title: String,
    description: String,
}


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


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Brief {
    tech: String,
    market: String,
    risk: String,
    recommendation: String,
}


impl Agent for TechTrack {
    type Output = TechAnalysis;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a software engineer. Assess the feasibility and implementation \
             effort for the described feature. Be concise.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}

impl Agent for MktTrack {
    type Output = MktAnalysis;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a market analyst. Identify the market opportunity and name \
             three direct competitors.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}

impl Agent for RiskTrack {
    type Output = RiskAnalysis;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a risk analyst. List the top three risks and propose one \
             concrete mitigation strategy.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}


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


impl Flow for Proposal {
    type Output = Brief;

    fn build(root: Node<Self>) -> Node<Self::Output> {
        let (tech, mkt, risk) = root.split(split_proposal);

        tech.agent()
            .merge((mkt.agent(), risk.agent()), merge_brief)
    }
}


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
