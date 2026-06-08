//! Linear flow example with one agent followed by one deterministic work node.

use pravah::flows::{Agent, AgentConfig, Flow, FlowBuilder, FlowError, FlowRuntime, FlowStep, Node};
use pravah::{Context, FlowConf};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};


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


impl Agent for SummariseRequest {
    type Output = BulletPoints;

    fn configure() -> AgentConfig {
        AgentConfig::new(
            "You are a concise summariser. Extract the key points from the text \
             the user sends and return them as a JSON array of short strings.",
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}


async fn format_bullets(bullets: BulletPoints, _ctx: Context) -> Result<Report, FlowError> {
    let markdown = bullets.points.iter().map(|p| format!("- {p}")).collect::<Vec<_>>().join("\n");
    Ok(Report { markdown })
}


impl Flow for SummariseRequest {
    type Output = Report;

    fn define(root: Node<Self>) -> FlowBuilder {
        root
            .agent()
            .work(format_bullets)
            .finalize()
    }
}


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
            FlowStep::Continue => {}
            FlowStep::Done(report) => {
                println!("## Summary\n\n{}", report.markdown);
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
