//! Supply one MCP text resource and choose whether a local search tool is available.
//!
//! Requires --features mcp, PRAVAH_MCP_URL, PRAVAH_MCP_RESOURCE_URI, and model
//! credentials. PRAVAH_MODEL_URL selects the model. Model calls may incur charges.

mod support;

use std::env;

use pravah::clients::Message;
use pravah::tools::ToolError;
use pravah::{
    Agent, AgentConfig, Chat, Context, GraphError, McpResourceRef, McpServer, ToolFilter, Toolset,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use support::ExampleError;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Question {
    text: String,
    resource_uri: String,
    allow_search: bool,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct SearchKnowledge {
    query: String,
}

/// Searches a tiny local collection; replace it with your application's lookup.
async fn search(request: SearchKnowledge, _ctx: Context) -> Result<Vec<String>, ToolError> {
    let query = request.query.to_lowercase();
    Ok([
        "Refunds require approval.",
        "Security incidents require escalation.",
    ]
    .into_iter()
    .filter(|text| text.to_lowercase().contains(&query))
    .map(str::to_owned)
    .collect())
}

fn tools(tools: Toolset) -> Toolset {
    tools.tool(search)
}

fn assistant(root: Agent<Question>) -> Agent<String> {
    root.tools(tools).configure(configure)
}

/// References select context; the filter controls actions independently.
async fn configure(question: Question, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        env::var("PRAVAH_MODEL_URL").unwrap_or_else(|_| "openai:///gpt-5-mini".into()),
        "Answer from the selected resource. Cite its URI.",
        Message::user(question.text),
    )
    .keep_alive()
    .resources([McpResourceRef::new("handbook", question.resource_uri)])
    .tool_filter(ToolFilter::new(move |_| question.allow_search)))
}

fn required_env(name: &str) -> Result<String, ExampleError> {
    env::var(name).map_err(|_| ExampleError::from(format!("set {name} before running")))
}

/// Keeps server credentials in Context and selects the resource for this invocation.
#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    dotenvy::dotenv().ok();
    let mut server = McpServer::new("handbook", required_env("PRAVAH_MCP_URL")?);
    if let Ok(token) = env::var("PRAVAH_MCP_BEARER_TOKEN") {
        server = server.bearer_token(token);
    }
    let ctx = Context::default().with_mcp_server(server);
    let mut chat = Chat::new(assistant, ctx)?;

    let reply = chat
        .send(Question {
            text: env::args()
                .nth(1)
                .unwrap_or_else(|| "What does this policy require?".into()),
            resource_uri: required_env("PRAVAH_MCP_RESOURCE_URI")?,
            allow_search: env::var("PRAVAH_ALLOW_SEARCH").is_ok_and(|value| value == "1"),
        })
        .await?;
    println!("{}", reply.output);
    Ok(())
}
