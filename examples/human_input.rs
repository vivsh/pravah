//! Human-in-the-loop example using `HumanInput` as a tool call.

use pravah::deps::Deps;
use pravah::flows::{Agent, AgentConfig, Flow, FlowBuilder, FlowRuntime, FlowStep};
use pravah::{CliMode, Context, FlowConf, HumanInput, HumanOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;


#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct BlogRequest {
    topic: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FinalResult {
    outcome: String,
}


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
            "gemini:///gemini-2.5-flash-lite",
        )
    }
}


impl Flow for BlogRequest {
    type Output = FinalResult;

    fn build(builder: FlowBuilder) -> FlowBuilder {
        builder
            .agent::<BlogRequest>()
            .tool::<BlogRequest, HumanInput, HumanOutput>()
            .flow::<HumanInput>()
    }
}


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

