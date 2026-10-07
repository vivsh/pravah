use super::*;
use pravah::clients::Role;

use pravah::testing::{ScriptedFactory, mock_tool_call};
use pravah::tools::ToolError;
use pravah::{
    Agent, AgentConfig, AgentDecision, AgentLoop, CompactionRequest, CompactionResult, Compactor,
    Toolset,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

fn agent(root: Agent<Vec<String>>) -> Agent<String> {
    root.configure(configure)
}
async fn configure(input: Vec<String>, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///test",
        "Test",
        Message::user(input.join(" ")),
    ))
}
fn flow(root: Flow<Vec<String>>) -> Flow<String> {
    root.agent(agent)
}

/// Growing opaque input does not recursively allocate while the VM creates a configure request.
#[test]
fn configure_boundary_allocations_do_not_scale_with_input() -> Result<(), GraphError> {
    let workflow = compile(flow)?;
    let mut counts = Vec::new();
    for size in [1, 1000] {
        let value = Value::array((0..size).map(|_| Value::from("data".repeat(100))));
        let mut runtime = workflow.prepared().start(value, Uuid::from_u128(7))?;
        let mut outcome = Ok(Step::Continue);
        let allocations = allocation_counter::measure(|| {
            outcome = runtime.next();
        });
        assert!(matches!(outcome?, Step::Agent(_)));
        counts.push((allocations.count_total, allocations.bytes_total));
    }
    assert_eq!(counts[0], counts[1]);
    Ok(())
}

/// Configuration persists the complete authored definition and invocation input.
#[test]
fn configure_request_preserves_values() -> Result<(), GraphError> {
    let workflow = compile(flow)?;
    let input = Value::array((0..10).map(|i| Value::from(format!("{i}:data"))));
    let mut runtime = workflow
        .prepared()
        .start(input.clone(), Uuid::from_u128(7))?;
    let request = next_agent(&mut runtime)?;
    let encoded = serde_json::to_value(&request).map_err(codec)?;
    assert_eq!(
        encoded["input"],
        serde_json::to_value(input).map_err(codec)?
    );
    assert!(encoded["definition"]["agent_id"].is_string());
    Ok(())
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct Search {
    query: String,
}
fn tools(tools: Toolset) -> Toolset {
    tools.tool(search)
}
async fn search(input: Search, _: Context) -> Result<String, ToolError> {
    Ok(format!("found {}", input.query))
}
async fn control(input: AgentLoop<String>, _: Context) -> Result<AgentDecision, GraphError> {
    assert_eq!(input.input(), "question");
    Ok(AgentDecision::continue_())
}

fn controlled_builder() -> pravah::ChatBuilder<String, String> {
    builder()
        .tools(tools)
        .control(control)
        .turn_budget(1)
        .tool_budget::<Search>(1)
}

/// Reused envelope codecs retain controller decisions, budget admission and mixed result batches.
#[tokio::test]
async fn controlled_budget_tool_round_survives_restore() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![
            mock_tool_call("one", "search", serde_json::json!({"query":"one"})),
            mock_tool_call("two", "search", serde_json::json!({"query":"two"})),
        ])
        .then_output(serde_json::json!("answer"));
    let context = || {
        Context::default().with_providers(ProviderRegistry::with_builtin_factory(factory.clone()))
    };
    let mut chat = controlled_builder().build(context())?;
    chat.submit("question")?;
    loop {
        match chat.next()? {
            ChatStep::Continue => {}
            ChatStep::Agent(fetch) => {
                chat = controlled_builder().restore(
                    cbor_roundtrip(json_roundtrip(chat.snapshot()?)?)?,
                    context(),
                )?;
                let response = chat.executor().execute(&fetch).await;
                chat.resume_agent(response)?;
            }
            ChatStep::Done(turn) => {
                assert_eq!(turn.output, "answer");
                break;
            }
            ChatStep::Suspend(_) => return Err(GraphError::ChatSuspended),
        }
    }
    assert_eq!(factory.calls().len(), 2);
    assert_tool_results(chat.snapshot()?)
}

/// Checks both admitted and unavailable results from the same accepted proposal.
fn assert_tool_results(snapshot: Snapshot) -> Result<(), GraphError> {
    let results = snapshot
        .history()
        .entries()
        .iter()
        .filter(|entry| matches!(entry.message.role, Role::Tool { .. }))
        .collect::<Vec<_>>();
    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .any(|entry| entry.message.content.contains("found one"))
    );
    assert!(
        results
            .iter()
            .any(|entry| entry.message.content.contains("unavailable"))
    );
    Ok(())
}

struct Summarize;
impl Compactor for Summarize {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        if request.committed().is_empty() {
            return Ok(CompactionResult::default());
        }
        if request.committed().len() == 3 {
            assert_eq!(request.summary(), Some("retained memory"));
        }
        Ok(CompactionResult {
            evict_indices: (0..request.committed().len()).collect(),
            summary: Some("retained memory".into()),
        })
    }
}

/// Repeated consolidation bounds history and survives both codecs without extra model dispatches.
#[tokio::test]
async fn prepared_history_is_bounded_and_reaches_client() -> Result<(), GraphError> {
    let mut factory = ScriptedFactory::new();
    for _ in 0..10 {
        factory = factory.then_output(serde_json::json!("answer"));
    }
    let context = || {
        Context::default().with_providers(ProviderRegistry::with_builtin_factory(factory.clone()))
    };
    let definition = || builder().compactor(Summarize);
    let mut chat = definition().build(context())?;
    for i in 0..10 {
        chat.send_with_key("question", format!("key-{i}")).await?;
        let snapshot = cbor_roundtrip(json_roundtrip(chat.snapshot()?)?)?;
        assert_eq!(
            snapshot.history().entries().len(),
            if i == 0 { 2 } else { 3 }
        );
        chat = definition().restore(snapshot, context())?;
    }
    let calls = factory.calls();
    assert_eq!(calls.len(), 10);
    for (_, messages) in calls.iter().skip(1) {
        let summary = messages
            .first()
            .ok_or_else(|| GraphError::Invalid("missing summary".into()))?;
        assert!(matches!(summary.role, Role::System));
        assert!(summary.content.contains("retained memory"));
        assert_eq!(messages.len(), 2);
    }
    Ok(())
}
