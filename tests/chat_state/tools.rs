use super::*;
use pravah::testing::mock_tool_call;
use pravah::tools::ToolError;
use pravah::{
    AgentDecision, AgentLoop, CompactionRequest, CompactionResult, Compactor, Flow, Toolset,
};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Lookup {
    query: String,
}
async fn lookup(input: Lookup, _ctx: Context) -> Result<String, ToolError> {
    Ok(input.query)
}
fn tools(root: Toolset) -> Toolset {
    root.tool(lookup)
}
fn researcher(root: Agent<String>) -> Agent<String> {
    root.tools(tools).configure(configure)
}

struct Summarize;
impl Compactor for Summarize {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        input: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        Ok(CompactionResult {
            evict_indices: (0..input.committed().len()).collect(),
            summary: (!input.committed().is_empty()).then(|| "prior conversation".into()),
        })
    }
}

/// Tool execution and repeated history consolidation preserve large graph-backed application state.
#[tokio::test]
async fn tools_and_history_preparation_do_not_touch_state() -> Result<(), TestError> {
    let script = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "lookup-1",
            "lookup",
            json!({"query":"evidence"}),
        )])
        .then_output(json!("first"))
        .then_output(json!("second"));
    let state = vec![vec!["private-state".repeat(100); 16]; 16];
    let mut chat =
        Chat::with_state(researcher, state, context(script.clone())?)?.with_compactor(Summarize);
    let before = serde_json::to_value(chat.snapshot()?)?;
    chat.send("first").await?;
    for snapshot in roundtrips(&chat.snapshot()?)? {
        let restored = Chat::<String, String, Vec<Vec<String>>>::from_snapshot(
            researcher,
            snapshot,
            Context::default(),
        )?;
        assert_eq!(restored.get()?, chat.get()?);
    }
    chat.send("second").await?;
    let after = serde_json::to_value(chat.snapshot()?)?;
    assert_eq!(
        before.pointer("/state/frames/0/variables"),
        after.pointer("/state/frames/0/variables")
    );
    assert_eq!(script.calls().len(), 3);
    assert!(!serde_json::to_string(chat.snapshot()?.history())?.contains("private-state"));
    Ok(())
}

fn controlled(root: Agent<String>) -> Agent<String> {
    root.control(control).configure(configure)
}
async fn control(_input: AgentLoop<String>, _ctx: Context) -> Result<AgentDecision, GraphError> {
    Ok(AgentDecision::suspend(pravah::graph::Value::from(
        "policy pause",
    )))
}
fn waiting_tool(root: Flow<Lookup>) -> Flow<String> {
    root.map(|input| input.query).suspend::<String>()
}
fn waiting_tools(root: Toolset) -> Toolset {
    root.flow(waiting_tool)
}
fn tool_suspender(root: Agent<String>) -> Agent<String> {
    root.tools(waiting_tools).configure(configure)
}

/// Controller and child-tool suspensions never masquerade as assistant responses or editable boundaries.
#[tokio::test]
async fn nested_suspensions_are_distinct() -> Result<(), TestError> {
    for agent in [controlled, tool_suspender] {
        let script = ScriptedFactory::new().then_tool_calls(vec![mock_tool_call(
            "lookup-1",
            "lookup",
            json!({"query":"pause"}),
        )]);
        let mut chat = Chat::with_state(agent, initial_state(), context(script)?)?;
        assert!(matches!(
            chat.send("question").await,
            Err(GraphError::ChatSuspended)
        ));
        let before = serde_json::to_value(chat.snapshot()?)?;
        assert!(matches!(
            chat.set(initial_state()),
            Err(GraphError::ChatNotReady { .. })
        ));
        assert!(matches!(
            chat.send("another").await,
            Err(GraphError::ChatNotReady { .. })
        ));
        assert_eq!(chat.get()?, initial_state());
        assert_eq!(before, serde_json::to_value(chat.snapshot()?)?);
    }
    Ok(())
}
