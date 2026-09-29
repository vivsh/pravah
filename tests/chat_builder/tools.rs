use super::*;
use pravah::testing::mock_tool_call;
use pravah::tools::ToolError;
use pravah::{AgentDecision, AgentLoop, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
pub(super) struct Search {
    query: String,
}

pub(super) fn toolset(tools: Toolset) -> Toolset {
    tools.tool(search)
}

async fn search(input: Search, _ctx: Context) -> Result<String, ToolError> {
    Ok(format!("found {}", input.query))
}

async fn control(loop_: AgentLoop<String>, _ctx: Context) -> Result<AgentDecision, GraphError> {
    assert_eq!(loop_.input(), "question");
    Ok(AgentDecision::continue_())
}

/// Builder tools use normal deterministic admission, hard budgets and typed controllers.
#[tokio::test]
async fn selected_tools_and_budgets_use_agent_execution() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("one", "search", serde_json::json!({"query":"one"})),
            mock_tool_call("two", "search", serde_json::json!({"query":"two"})),
        ])
        .then_output(serde_json::json!("answer"));
    let mut chat = builder()
        .tools(toolset)
        .control(control)
        .turn_budget(1)
        .tool_budget::<Search>(1)
        .build(context(&factory)?)?;
    assert_eq!(
        chat.send(ChatRequest::from("question").tools(["search"]))
            .await?
            .output,
        "answer"
    );
    let snapshot = chat.snapshot()?;
    let results: Vec<_> = snapshot
        .history()
        .entries()
        .iter()
        .filter(|entry| matches!(entry.message.role, Role::Tool { .. }))
        .map(|entry| (&entry.message.role, entry.message.content.as_str()))
        .collect();
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(
        |(role, content)| matches!(role, Role::Tool { call_id } if call_id == "one")
            && content.contains("found one")
    ));
    assert!(results.iter().any(
        |(role, content)| matches!(role, Role::Tool { call_id } if call_id == "two")
            && content.contains("unavailable")
    ));
    assert_eq!(factory.calls().len(), 2);
    Ok(())
}

/// An empty invocation subset hides declared tools without invalidating their budgets.
#[tokio::test]
async fn empty_selection_rejects_calls_without_execution() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "one",
            "search",
            serde_json::json!({"query":"one"}),
        )])
        .then_output(serde_json::json!("answer"));
    let mut chat = builder()
        .tools(toolset)
        .tool_budget::<Search>(1)
        .build(context(&factory)?)?;
    chat.send(ChatRequest::from("question").tools(Vec::<String>::new()))
        .await?;
    assert!(
        chat.snapshot()?
            .history()
            .entries()
            .iter()
            .any(|entry| matches!(entry.message.role, Role::Tool { .. })
                && entry.message.content.contains("unavailable"))
    );
    Ok(())
}

/// Builder control declarations preserve the graph API's single-controller invariant.
#[tokio::test]
async fn repeated_controller_is_rejected() {
    assert!(
        builder()
            .control(control)
            .control(control)
            .build(Context::default())
            .is_err()
    );
}
