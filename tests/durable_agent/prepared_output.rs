use super::*;
use pravah::graph::{NodeKind, PreparedGraph, to_value};
use pravah::{Agent, AgentConfig};

async fn configure(input: String, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///test",
        "test",
        Message::user(input),
    ))
}

fn flow(root: Flow<String>) -> Flow<String> {
    root.agent(|root: Agent<String>| root.configure(configure))
}

/// Uses the same registry with an authored output schema, independent of the handler's Rust type.
fn prepare(schema: serde_json::Value) -> Result<PreparedGraph, GraphError> {
    let flow = compile(flow)?;
    let mut graph = flow.graph().clone();
    for node in &mut graph.nodes {
        if let NodeKind::Continuation { payload, .. } = &mut node.kind {
            let mut data = serde_json::to_value(&*payload).map_err(codec)?;
            data["output_schema"] = schema.clone();
            *payload = to_value(data).map_err(codec)?;
        }
    }
    PreparedGraph::new(graph, flow.registry().clone())
}

/// Invalid authored schemas fail at preparation, before any execution or model call.
#[test]
fn invalid_output_schema_fails_preparation() -> Result<(), GraphError> {
    assert!(matches!(
        prepare(serde_json::json!({"type": 42})),
        Err(GraphError::GraphValidation(_))
    ));
    Ok(())
}

/// Every response still obeys the authored schema, across independent starts and restores.
#[tokio::test]
async fn prepared_validator_enforces_authored_constraints() -> Result<(), GraphError> {
    let prepared = prepare(serde_json::json!({"type": "string", "minLength": 10}))?;
    let calls = Arc::new(AtomicUsize::new(0));
    let executor = prepared.executor(context(&calls));
    for _ in 0..2 {
        let mut runtime = prepared.start("question".into(), Uuid::nil())?;
        loop {
            let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
            match runtime.next() {
                Ok(Step::Continue) => {}
                Ok(Step::Agent(fetch)) => {
                    runtime =
                        prepared.restore(cbor_roundtrip(json_roundtrip(runtime.snapshot()?)?)?)?;
                    let response = executor.execute(&fetch).await;
                    runtime.resume_agent(response)?;
                }
                Err(GraphError::Schema { .. }) => {
                    assert_eq!(
                        before,
                        serde_json::to_value(runtime.snapshot()?).map_err(codec)?
                    );
                    assert!(runtime.pending_agent().is_none());
                    assert_eq!(runtime.history().entries().len(), 1);
                    break;
                }
                other => return Err(GraphError::Invalid(format!("unexpected step: {other:?}"))),
            }
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}
