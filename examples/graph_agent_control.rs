//! A controller asks for approval after a tool result; the host resumes with a conclusion.
//!
//! Run with --features testing. The model and tool are deterministic and offline.

use pravah::clients::Message;
use pravah::graph::Value;
use pravah::testing::{ScriptedFactory, mock_tool_call};
use pravah::tools::ToolError;
use pravah::{
    Agent, AgentConfig, AgentDecision, AgentInterventionPoint, AgentLoop, AgentResume, Context,
    Flow, GraphError, Step, Toolset, compile,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct ReadPolicy {
    name: String,
}

async fn read_policy(request: ReadPolicy, _ctx: Context) -> Result<String, ToolError> {
    Ok(format!("{} requires a recorded approval.", request.name))
}

fn tools(tools: Toolset) -> Toolset {
    tools.tool(read_policy)
}

fn reviewer(root: Agent<String>) -> Agent<String> {
    root.tools(tools).control(control).configure(configure)
}

async fn configure(question: String, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///scripted",
        "Read the policy before answering.",
        Message::user(question),
    ))
}

async fn control(loop_: AgentLoop<String>, _ctx: Context) -> Result<AgentDecision, GraphError> {
    if loop_.point() == AgentInterventionPoint::AfterTools {
        return Ok(AgentDecision::suspend(Value::from(
            "Policy retrieved. Approve using it in the answer?",
        )));
    }
    Ok(AgentDecision::continue_())
}

fn review(root: Flow<String>) -> Flow<String> {
    root.agent(reviewer)
}

/// Drives external effects explicitly and supplies a simulated human approval.
#[tokio::main]
async fn main() -> Result<(), GraphError> {
    let client = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "policy-1",
            "read_policy",
            serde_json::json!({"name": "Refunds"}),
        )])
        .then_output(serde_json::json!("Refunds require a recorded approval."));
    let ctx = Context::default().with_providers(pravah::testing::providers(client)?);
    let workflow = compile(review)?;
    let executor = workflow.prepared().executor(ctx);
    let mut execution =
        workflow.start("What is the refund policy?".into(), uuid::Uuid::now_v7())?;

    loop {
        match execution.next()? {
            Step::Continue => {}
            Step::Agent(fetch) => {
                let response = executor.execute(&fetch).await;
                execution.resume_agent(response)?;
            }
            Step::Suspend(payload) => {
                println!("Approval request: {payload}");
                execution.resume(AgentResume::Conclude {
                    guidance: "Approved. Answer using the retrieved policy.".into(),
                })?;
            }
            Step::Done(value) => {
                println!("{}", workflow.decode_output(value)?);
                return Ok(());
            }
        }
    }
}
