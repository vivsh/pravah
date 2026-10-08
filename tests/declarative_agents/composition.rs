use super::*;
use pravah::tools::ToolError;
use pravah::{AgentDecision, AgentInterventionPoint, AgentLoop, Toolset};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Search {
    query: String,
}

async fn search(request: Search, _: Context) -> Result<String, ToolError> {
    Ok(format!("source: {}", request.query))
}

fn tools(tools: Toolset) -> Toolset {
    tools.tool(search)
}

async fn control(loop_: AgentLoop<Question>, _: Context) -> Result<AgentDecision, GraphError> {
    assert_eq!(loop_.input().topic, "durability");
    assert!(loop_.turns_remaining().is_some());
    if loop_.point() == AgentInterventionPoint::AfterTools {
        assert_eq!(loop_.calls_remaining("search"), Some(0));
    }
    Ok(AgentDecision::continue_())
}

fn controlled(root: Agent<Question>) -> Agent<Notes> {
    root.model("test:///model")
        .instructions("Research.")
        .tools(tools)
        .control(control)
        .turn_budget(3)
        .tool_budget::<Search>(1)
        .build()
}

fn controlled_flow(root: Flow<Question>) -> Flow<Notes> {
    root.agent(controlled)
}

/// Declarative tools and budgets use ordinary control boundaries with the original typed input.
#[tokio::test]
async fn controls_and_tools_share_existing_execution() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_tool_calls(vec![pravah::testing::mock_tool_call(
            "call-1",
            "search",
            serde_json::json!({"query":"durability"}),
        )])
        .then_output(serde_json::json!({"finding":"verified"}));
    let flow = compile(controlled_flow)?;
    let executor = flow
        .prepared()
        .executor(Context::default().with_providers(providers(script.clone())?));
    let mut runtime = flow.start(
        Question {
            topic: "durability".into(),
        },
        Uuid::nil(),
    )?;
    assert!(matches!(
        host::finish(&mut runtime, &executor).await?,
        Step::Done(_)
    ));
    assert_eq!(script.calls().len(), 2);
    Ok(())
}

fn repeated(root: Flow<Vec<Question>>) -> Flow<Vec<Notes>> {
    root.each(research)
        .map(|notes: Vec<Notes>| {
            notes
                .into_iter()
                .map(|note| Question {
                    topic: note.finding,
                })
                .collect()
        })
        .each(research)
}

/// Repeated embedded flows and agents retain independent call sites with shared keyed history.
#[tokio::test]
async fn reused_each_flows_prepare_and_execute() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(serde_json::json!({"finding":"one"}))
        .then_output(serde_json::json!({"finding":"two"}))
        .then_output(serde_json::json!({"finding":"three"}))
        .then_output(serde_json::json!({"finding":"four"}));
    let flow = compile(repeated)?;
    let executor = flow
        .prepared()
        .executor(Context::default().with_providers(providers(script.clone())?));
    let input = ["first", "second"]
        .into_iter()
        .map(|topic| Question {
            topic: topic.into(),
        })
        .collect();
    let mut runtime = flow.start(input, Uuid::nil())?;
    let Step::Done(value) = host::finish(&mut runtime, &executor).await? else {
        return Err(GraphError::Invalid("expected completion".into()));
    };
    assert_eq!(
        flow.decode_output(value)?,
        vec![
            Notes {
                finding: "three".into()
            },
            Notes {
                finding: "four".into()
            }
        ]
    );
    assert_eq!(script.calls().len(), 4);
    assert_eq!(runtime.history().entries().len(), 8);
    Ok(())
}

fn unkeyed(root: Agent<Question>) -> Agent<Notes> {
    root.model("test:///model").build()
}

fn unkeyed_flow(root: Flow<Question>) -> Flow<Notes> {
    root.agent(unkeyed)
}

/// Declarative graph agents without keys retain existing frame-local history cleanup.
#[tokio::test]
async fn unkeyed_history_is_frame_local() -> Result<(), GraphError> {
    let script = ScriptedFactory::new().then_output(serde_json::json!({"finding":"verified"}));
    let flow = compile(unkeyed_flow)?;
    let executor = flow
        .prepared()
        .executor(Context::default().with_providers(providers(script)?));
    let mut runtime = flow.start(
        Question {
            topic: "durability".into(),
        },
        Uuid::nil(),
    )?;
    assert!(matches!(
        host::finish(&mut runtime, &executor).await?,
        Step::Done(_)
    ));
    assert!(runtime.history().entries().is_empty());
    Ok(())
}
