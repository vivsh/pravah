use super::*;
use pravah::tools::ToolError;
use pravah::{Agent, AgentConfig, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Search {
    query: String,
}
async fn search(_: Search, _: Context) -> Result<String, ToolError> {
    Ok("found".into())
}
fn tools(root: Toolset) -> Toolset {
    root.tool(search)
}
fn agent(root: Agent<String>) -> Agent<String> {
    root.tools(tools).configure(configure)
}
async fn configure(memory: String, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///recorded",
        "Instructions",
        Message::user("question"),
    )
    .memory(memory)
    .provider_config(serde_json::json!({"opaque": vec!["large".repeat(1000); 32]}))
    .max_output_tokens(123))
}
fn flow(root: Flow<String>) -> Flow<String> {
    root.agent(agent)
}

/// Direct generation requests retain resolved settings without another envelope or preparation task.
#[tokio::test]
async fn generation_construction_preserves_options() -> Result<(), GraphError> {
    let workflow = compile(flow)?;
    let executor = workflow.prepared().executor(Context::default());
    for memory in ["", "  Unicode 🌊\n "] {
        let mut runtime = workflow.start(memory.into(), Uuid::nil())?;
        let request = next_agent(&mut runtime)?;
        runtime.resume_agent(executor.execute(&request).await)?;
        let request = next_agent(&mut runtime)?;
        assert_eq!(request.kind(), "generate");
        let wire = serde_json::to_value(&request).map_err(codec)?;
        assert_eq!(wire["options"]["max_output_tokens"], 123);
        assert_eq!(
            wire["options"]["provider_config"]["opaque"]
                .as_array()
                .map(Vec::len),
            Some(32)
        );
        assert_eq!(wire["options"]["tools"].as_array().map(Vec::len), Some(1));
        assert_eq!(
            wire["options"]["preamble"],
            format!("Instructions\n\n<memory>\n{memory}\n</memory>")
        );
        assert_eq!(wire["entries"][0]["message"]["content"], "question");
        assert_eq!(request.model(), Some("openai:///recorded"));
    }
    Ok(())
}
