//! Adaptive graph-agent intervention with tool visibility, conclusion, and suspension.
//!
//! Requires credentials for the model selected by `PRAVAH_MODEL_URL`. The tools
//! are local and deterministic; model calls may incur provider charges.

mod support;

use std::env;

use pravah::Context;
use pravah::clients::Message;
use pravah::graph::{
    self, Agent, AgentConfig, AgentDecision, AgentInterventionPoint, AgentLoop, AgentResume,
    AgentToolResult, Flow, GraphError, Step, ToolFilter, Toolset, Value, to_value,
};
use pravah::tools::ToolError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use support::ExampleError;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ResearchRequest {
    question: String,
    model: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ResearchAnswer {
    answer: String,
    sources: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SearchRequest {
    query: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    pages: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FetchRequest {
    page: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FetchResult {
    page: String,
    text: String,
    approval_required: bool,
}

/// Declares the complete candidate tool surface prepared with the agent.
fn research_tools(tools: Toolset) -> Toolset {
    tools.tool(search).tool(fetch)
}

async fn search(input: SearchRequest, _ctx: Context) -> Result<SearchResult, ToolError> {
    Ok(SearchResult {
        pages: vec![format!("policy://{}", input.query.replace(' ', "-"))],
    })
}

async fn fetch(input: FetchRequest, _ctx: Context) -> Result<FetchResult, ToolError> {
    Ok(FetchResult {
        approval_required: input.page.contains("restricted"),
        text: "The local policy requires evidence and a recorded decision.".into(),
        page: input.page,
    })
}

fn researcher(root: Agent<ResearchRequest>) -> Agent<ResearchAnswer> {
    root.tools(research_tools)
        .control(control_researcher)
        .configure(configure_researcher)
}

/// Resolves invocation-specific model settings without changing loop policy.
async fn configure_researcher(
    request: ResearchRequest,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        request.model,
        "Research with the available tools, then return a cited structured answer.",
        Message::user(request.question),
    ))
}

/// Selects policy behavior from the current explicit intervention boundary.
async fn control_researcher(
    loop_: AgentLoop<ResearchRequest>,
    _ctx: Context,
) -> Result<AgentDecision, GraphError> {
    match loop_.point() {
        AgentInterventionPoint::BeforeModel | AgentInterventionPoint::BeforeTools => {
            control_before_dispatch(&loop_)
        }
        AgentInterventionPoint::AfterTools => control_after_tools(&loop_),
    }
}

/// Redirects semantic proposal repetition before it becomes another tool call.
fn control_before_dispatch(
    loop_: &AgentLoop<ResearchRequest>,
) -> Result<AgentDecision, GraphError> {
    if loop_.metrics().repeated_proposals() >= 3 {
        return Ok(AgentDecision::redirect()
            .guidance("Do not repeat the same call. Use the evidence already collected.")
            .tools(ToolFilter::new(|tool| tool.name() == "fetch_request")));
    }
    Ok(AgentDecision::continue_())
}

/// Chooses recovery, restriction, conclusion, or approval from tool results.
fn control_after_tools(loop_: &AgentLoop<ResearchRequest>) -> Result<AgentDecision, GraphError> {
    if loop_.results().iter().any(AgentToolResult::is_error) {
        return Ok(AgentDecision::redirect()
            .guidance("A tool was unavailable for that turn. Choose from the exposed tools.")
            .tools(ToolFilter::all()));
    }
    if loop_.metrics().consecutive_tool_rounds() >= 4 {
        return Ok(AgentDecision::conclude(
            "Stop gathering evidence and return the best supported structured answer now.",
        ));
    }
    if loop_.results().iter().any(requires_approval) {
        return Ok(AgentDecision::suspend(runtime_value(
            "Approval is required before using the restricted evidence",
        )?));
    }
    if loop_
        .results()
        .iter()
        .any(|result| result.tool_name() == "search_request")
    {
        return Ok(AgentDecision::redirect()
            .guidance("Inspect one relevant page before answering.")
            .tools(ToolFilter::new(|tool| tool.name() == "fetch_request")));
    }
    Ok(AgentDecision::continue_())
}

fn requires_approval(result: &AgentToolResult) -> bool {
    result.tool_name() == "fetch_request"
        && result
            .value()
            .get("approval_required")
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

fn runtime_value(value: &str) -> Result<Value, GraphError> {
    to_value(value).map_err(|error| GraphError::ValueConversion {
        target: "agent suspension payload".into(),
        reason: error.to_string(),
    })
}

fn research(root: Flow<ResearchRequest>) -> Flow<ResearchAnswer> {
    root.agent(researcher)
}

/// Builds one invocation from command-line and environment settings.
async fn run() -> Result<(), ExampleError> {
    dotenvy::dotenv().ok();
    let request = ResearchRequest {
        question: env::args()
            .nth(1)
            .unwrap_or_else(|| "What does the approval policy require?".into()),
        model: env::var("PRAVAH_MODEL_URL").unwrap_or_else(|_| "openai:///gpt-5-mini".into()),
    };
    let flow = graph::compile(research)?;
    let mut runtime = flow.runtime(request)?;
    drive(&flow, &mut runtime, Context::default()).await
}

/// Drives the workflow and routes controller suspensions to the application.
async fn drive(
    flow: &graph::CompiledFlow<ResearchRequest, ResearchAnswer>,
    runtime: &mut graph::Runtime,
    ctx: Context,
) -> Result<(), ExampleError> {
    loop {
        match runtime.next(ctx.clone()).await? {
            Step::Continue => {}
            Step::Done(value) => {
                println!("{:#?}", flow.decode_output(value)?);
                return Ok(());
            }
            Step::Suspend(payload) => resume_approved(runtime, ctx.clone(), payload).await?,
        }
    }
}

/// Demonstrates an application-selected typed resume decision.
async fn resume_approved(
    runtime: &mut graph::Runtime,
    ctx: Context,
    payload: Value,
) -> Result<(), ExampleError> {
    println!("agent requested intervention: {payload}");
    let resume = AgentResume::Conclude {
        guidance: "Approval granted. Return the answer using the collected evidence.".into(),
    };
    let value = to_value(resume).map_err(|error| {
        ExampleError::unexpected(format!("failed to encode agent resume: {error}"))
    })?;
    match runtime.resume(value, ctx).await? {
        Step::Continue => Ok(()),
        other => Err(ExampleError::unexpected(format!(
            "agent resume returned {other:?}"
        ))),
    }
}

#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    run().await
}
