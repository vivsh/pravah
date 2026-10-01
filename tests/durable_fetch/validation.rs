use super::*;
use pravah::graph::{FetchBody, NodeKind, PreparedGraph, to_value};
use pravah::{Agent, AgentConfig, AgentDecision, AgentLoop};

/// Locates the first callback request without executing external work.
fn hook(chat: &mut Chat<String, String>) -> Result<Fetch, GraphError> {
    loop {
        match chat.next()? {
            ChatStep::Continue => {}
            ChatStep::Fetch(fetch) => return Ok(fetch),
            _ => return Err(GraphError::Invalid("expected hook".into())),
        }
    }
}

/// Replaces one serialized definition field to exercise the executor's actual external boundary.
fn changed_payload(
    fetch: &Fetch,
    key: &str,
    value: serde_json::Value,
) -> Result<Fetch, GraphError> {
    let mut encoded = serde_json::to_value(fetch).map_err(codec)?;
    let payload = encoded
        .pointer_mut("/request/body/data/payload")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| GraphError::Invalid("missing hook payload".into()))?;
    payload.insert(key.to_owned(), value);
    serde_json::from_value(encoded).map_err(codec)
}

/// Invalid hook definitions still fail before external callbacks or any runtime/history mutation.
#[tokio::test]
async fn compiled_contract_rejects_malformed_external_settings() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().build(context(&calls))?;
    chat.submit("question")?;
    let fetch = hook(&mut chat)?;
    let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    let Some(FetchBody::Value(body)) = fetch.request().body_ref() else {
        return Err(GraphError::Invalid("missing structured hook".into()));
    };
    let data = body
        .get("payload")
        .and_then(|payload| payload.get("configuration"))
        .ok_or_else(|| GraphError::Invalid("missing configuration".into()))?;
    let mut wrong_schema = serde_json::to_value(data).map_err(codec)?;
    wrong_schema["schema"] = serde_json::json!({"type": "boolean"});
    let mut wrong_value = serde_json::to_value(data).map_err(codec)?;
    wrong_value["value"]["model"] = serde_json::json!(false);
    for (key, value) in [
        ("configuration", serde_json::Value::Null),
        ("configuration", wrong_schema),
        ("configuration", wrong_value),
        ("version", serde_json::json!(0)),
        ("tools", serde_json::json!([{}])),
        ("output_type_name", serde_json::json!(false)),
        ("control_handler_key", serde_json::json!("mismatched")),
    ] {
        let invalid = changed_payload(&fetch, key, value)?;
        assert!(chat.executor().execute(&invalid).await.is_err(), "{key}");
        assert_eq!(
            before,
            serde_json::to_value(chat.snapshot()?).map_err(codec)?
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let response = chat.executor().execute(&fetch).await?;
    chat.resume_fetch(fetch.id(), Ok(response))?;
    Ok(())
}

async fn control(_: AgentLoop<String>, _: Context) -> Result<AgentDecision, GraphError> {
    Ok(AgentDecision::continue_())
}

/// Controller hooks enforce the same settings contract instead of bypassing definition checks.
#[tokio::test]
async fn controlled_hook_rejects_missing_configuration() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().control(control).build(context(&calls))?;
    chat.submit("question")?;
    let fetch = loop {
        let fetch = hook(&mut chat)?;
        let control = match fetch.request().body_ref() {
            Some(FetchBody::Value(body)) => body
                .get("operation")
                .and_then(|value| value.get("Control"))
                .is_some(),
            _ => false,
        };
        if control {
            break fetch;
        }
        let response = chat.executor().execute(&fetch).await?;
        chat.resume_fetch(fetch.id(), Ok(response))?;
    };
    let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    let invalid = changed_payload(&fetch, "configuration", serde_json::Value::Null)?;
    assert!(chat.executor().execute(&invalid).await.is_err());
    assert_eq!(
        before,
        serde_json::to_value(chat.snapshot()?).map_err(codec)?
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    chat.executor().execute(&fetch).await?;
    Ok(())
}

async fn configure(input: String, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///test",
        "test",
        Message::user(input),
    ))
}

fn agent(root: Agent<String>) -> Agent<String> {
    root.configure(configure)
}

fn flow(root: Flow<String>) -> Flow<String> {
    root.agent(agent)
}

/// Ordinary configure functions reject injected settings at both preparation and Fetch boundaries.
#[tokio::test]
async fn ordinary_agent_rejects_unregistered_settings() -> Result<(), GraphError> {
    let workflow = compile(flow)?;
    let data = serde_json::json!({"schema": {}, "value": {}});
    let mut graph = workflow.graph().clone();
    for node in &mut graph.nodes {
        if let NodeKind::Continuation { payload, .. } = &mut node.kind {
            let mut json = serde_json::to_value(&*payload).map_err(codec)?;
            json["configuration"] = data.clone();
            *payload = to_value(json).map_err(codec)?;
        }
    }
    assert!(matches!(
        PreparedGraph::new(graph, workflow.registry().clone()),
        Err(GraphError::GraphValidation(_))
    ));
    let mut runtime = workflow.start("question".into(), Uuid::nil())?;
    let fetch = next_fetch(&mut runtime)?;
    let invalid = changed_payload(&fetch, "configuration", data)?;
    let executor = workflow.prepared().executor(Context::default());
    assert!(matches!(
        executor.execute(&invalid).await,
        Err(GraphError::GraphValidation(_))
    ));
    executor.execute(&fetch).await?;
    Ok(())
}

/// External agent hooks retain their handlers after the prepared workflow is dropped.
#[tokio::test]
async fn executor_outlives_workflow_but_missing_registry_fails() -> Result<(), GraphError> {
    let workflow = compile(flow)?;
    let mut runtime = workflow.start("question".into(), Uuid::from_u128(9))?;
    let fetch = next_fetch(&mut runtime)?;
    let executor = workflow.prepared().executor(Context::default());
    assert!(std::ptr::eq(workflow.registry(), executor.registry()));
    let registry = Arc::new(workflow.registry().clone());
    let manually_configured =
        pravah::FetchExecutor::new(Context::default()).with_registry(Arc::clone(&registry));
    assert!(std::ptr::eq(
        registry.as_ref(),
        manually_configured.registry()
    ));
    drop(runtime);
    drop(workflow);

    let missing = pravah::FetchExecutor::new(Context::default());
    let missing_error = missing
        .execute(&fetch)
        .await
        .err()
        .ok_or_else(|| GraphError::Invalid("expected missing handler".into()))?;
    assert!(matches!(
        missing_error,
        GraphError::FetchValidation(ref reason) if reason == "missing external hook handler"
    ));
    assert_eq!(executor.execute(&fetch).await?.status(), 200);
    assert_eq!(manually_configured.execute(&fetch).await?.status(), 200);
    Ok(())
}

/// Rebuilding the immutable contract preserves instruction-only restoration and recorded outcomes.
#[tokio::test]
async fn rebuilt_contract_preserves_pending_and_accepted_configuration() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().instructions("original").build(context(&calls))?;
    chat.submit_with_key("question", "request-1")?;
    let fetch = hook(&mut chat)?;
    let snapshot = cbor_roundtrip(json_roundtrip(chat.snapshot()?)?)?;
    let mut restored = builder()
        .instructions("updated")
        .restore::<()>(snapshot, context(&calls))?;
    assert_eq!(restored.pending_fetch().map(Fetch::id), Some(fetch.id()));
    let response = restored.executor().execute(&fetch).await?;
    let Some(FetchBody::Value(body)) = response.body_ref() else {
        return Err(GraphError::Invalid("missing configuration response".into()));
    };
    assert_eq!(
        body.get("resolved")
            .and_then(|v| v.get("instructions"))
            .and_then(Value::as_str),
        Some("updated")
    );
    restored.resume_fetch(fetch.id(), Ok(response))?;
    let accepted = restored.snapshot()?;
    let mut restored = builder().instructions("later").restore::<()>(
        cbor_roundtrip(json_roundtrip(accepted.clone())?)?,
        context(&calls),
    )?;
    assert!(restored.pending_fetch().is_none());
    assert_eq!(
        serde_json::to_value(accepted).map_err(codec)?,
        serde_json::to_value(restored.snapshot()?).map_err(codec)?
    );
    assert_eq!(hook(&mut restored)?.request().url(), "rath://generate");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

/// Boundary validation borrows unused input schema data; schema size cannot grow callback allocations.
#[test]
fn hook_validation_does_not_decode_unused_schemas() -> Result<(), GraphError> {
    let workflow = compile(flow)?;
    let mut measured = Vec::new();
    for count in [1, 1000] {
        let mut graph = workflow.graph().clone();
        for node in &mut graph.nodes {
            if let NodeKind::Continuation { payload, .. } = &mut node.kind {
                let mut json = serde_json::to_value(&*payload).map_err(codec)?;
                json["input_schema"] = serde_json::json!({
                    "type": "string",
                    "$defs": {"Unused": {"enum": vec!["value".repeat(100); count]}},
                });
                *payload = to_value(json).map_err(codec)?;
            }
        }
        let prepared = PreparedGraph::new(graph, workflow.registry().clone())?;
        let mut runtime = prepared.start(Value::from("question"), Uuid::nil())?;
        let fetch = next_fetch(&mut runtime)?;
        let executor = prepared.executor(Context::default());
        futures::executor::block_on(executor.execute(&fetch))?;
        let mut result = None;
        let allocations = allocation_counter::measure(|| {
            result = Some(futures::executor::block_on(executor.execute(&fetch)));
        });
        result.ok_or_else(|| GraphError::Invalid("missing outcome".into()))??;
        measured.push((allocations.count_total, allocations.bytes_total));
    }
    assert_eq!(measured[0], measured[1]);
    Ok(())
}
