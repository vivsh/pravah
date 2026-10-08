//! One ordinary model turn and one search call, followed by forced conclusion.
//!
//! Run with --features testing. Both model replies and tool results are local.

use pravah::testing::{ScriptedFactory, mock_tool_call};
use pravah::tools::ToolError;
use pravah::{Agent, Chat, Context, GraphError, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Search {
    query: String,
}

fn tools(tools: Toolset) -> Toolset {
    tools.tool(search)
}

async fn search(request: Search, _ctx: Context) -> Result<String, ToolError> {
    println!("Running search: {}", request.query);
    Ok("Pravah supports durable, stepwise workflows.".into())
}

fn assistant(root: Agent<String>) -> Agent<String> {
    root.model("test:///scripted")
        .instructions("Research the question, then give a brief answer.")
        .tools(tools)
        .turn_budget(1)
        .tool_budget::<Search>(1)
        .build()
}

/// The second proposed search is unavailable; the next model request must conclude.
#[tokio::main]
async fn main() -> Result<(), GraphError> {
    let client = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("first", "search", json!({"query": "Pravah"})),
            mock_tool_call("second", "search", json!({"query": "more evidence"})),
        ])
        .then_output(json!("One search ran before the agent concluded."));
    let ctx = Context::default().with_providers(pravah::testing::providers(client)?);

    let mut chat = Chat::new(assistant, ctx)?;

    println!("{}", chat.send("What is Pravah?").await?.output);
    Ok(())
}
