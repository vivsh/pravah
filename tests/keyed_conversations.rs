use pravah::clients::Message;
use pravah::graph::FetchExecutor;
use pravah::history::MessageHistory;
use pravah::testing::ScriptedFactory;
use pravah::{Agent, AgentConfig, Chat, Context, Flow, GraphError, Runtime, Step, compile};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

#[path = "support/host.rs"]
mod host;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Request {
    text: String,
    key: Option<String>,
}

fn request(key: Option<&str>) -> Request {
    Request {
        text: "question".into(),
        key: key.map(str::to_owned),
    }
}

async fn configure(input: Request, _ctx: Context) -> Result<AgentConfig, GraphError> {
    let config = AgentConfig::new("test:///test", "Answer.", Message::user(input.text));
    Ok(match input.key {
        Some(key) => config.key(key),
        None => config,
    })
}

fn assistant(root: Agent<Request>) -> Agent<String> {
    root.configure(configure)
}

fn ask(root: Flow<Request>) -> Flow<String> {
    root.agent(assistant)
}

fn parent(root: Flow<Request>) -> Flow<String> {
    root.agent(assistant)
        .map(|_| request(Some("review")))
        .flow(ask)
        .map(|_| request(Some("review")))
        .flow(ask)
}

fn each(root: Flow<Vec<Request>>) -> Flow<Vec<String>> {
    root.each(ask)
}

fn nested_agent(root: Agent<Request>) -> Agent<String> {
    root.tools(|tools| tools.flow(ask)).configure(configure)
}

fn nested(root: Flow<Request>) -> Flow<String> {
    root.agent(nested_agent)
}

/// A tool cannot re-enter its caller's open conversation; failure retains a retryable checkpoint.
#[tokio::test]
async fn nested_key_reuse_cannot_mix_unfinished_exchanges() -> Result<(), GraphError> {
    let script = ScriptedFactory::new().then_tool_calls(vec![pravah::testing::mock_tool_call(
        "child",
        "request",
        json!({"text": "nested", "key": "review"}),
    )]);
    let flow = compile(nested)?;
    let mut runtime = flow.start(request(Some("review")), Uuid::nil())?;
    let executor = flow.prepared().executor(context(&script)?);
    assert!(matches!(
        host::finish(&mut runtime, &executor).await,
        Err(GraphError::AgentConversationBusy)
    ));
    let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    runtime = flow.restore(serde_json::from_value(before.clone()).map_err(codec)?)?;
    assert!(matches!(
        runtime.next(),
        Err(GraphError::AgentConversationBusy)
    ));
    assert_eq!(
        before,
        serde_json::to_value(runtime.snapshot()?).map_err(codec)?
    );
    assert_eq!(script.calls().len(), 1);
    assert_eq!(runtime.history().entries().len(), 2);
    Ok(())
}

/// Explicit Chat keys override the execution-scoped default without adding state to Chat.
#[tokio::test]
async fn explicit_chat_key_survives_restore() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"));
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .key("customer/42")
        .build(context(&script)?)?;
    chat.send("one").await?;
    let mut restored = Chat::builder::<String, String>()
        .model("test:///test")
        .key("customer/42")
        .restore::<()>(chat.snapshot()?, context(&script)?)?;
    restored.send("two").await?;
    assert!(
        restored
            .snapshot()?
            .history()
            .entries()
            .iter()
            .all(|entry| entry.session_id == "key:customer/42")
    );
    assert_eq!(script.calls()[1].1.len(), 3);
    Ok(())
}

fn context(script: &ScriptedFactory) -> Result<Context, GraphError> {
    Ok(Context::default().with_providers(pravah::testing::providers(script.clone())?))
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::SnapshotValidation(error.to_string())
}

/// Restores every boundary through alternating JSON/CBOR codecs without replaying accepted effects.
async fn finish_restoring(
    flow: &pravah::CompiledFlow<Request, String>,
    runtime: &mut Runtime,
    executor: &FetchExecutor,
) -> Result<(), GraphError> {
    for index in 0..200 {
        if matches!(host::step(runtime, executor).await?, Step::Done(_)) {
            return Ok(());
        }
        let snapshot = runtime.snapshot()?;
        let snapshot = if index % 2 == 0 {
            serde_json::from_slice(&serde_json::to_vec(&snapshot).map_err(codec)?).map_err(codec)?
        } else {
            let mut bytes = Vec::new();
            ciborium::into_writer(&snapshot, &mut bytes).map_err(codec)?;
            ciborium::from_reader(bytes.as_slice()).map_err(codec)?
        };
        *runtime = flow.restore(snapshot)?;
    }
    Err(GraphError::Invalid("test execution did not finish".into()))
}

/// Parent and distinct child call sites share a keyed conversation across both snapshot codecs.
#[tokio::test]
async fn parent_and_children_share_keyed_history_after_restore() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"))
        .then_output(json!("three"));
    let flow = compile(parent)?;
    let executor = flow.prepared().executor(context(&script)?);
    let mut runtime = flow.start(request(Some("review")), Uuid::nil())?;
    finish_restoring(&flow, &mut runtime, &executor).await?;
    assert_eq!(
        script
            .calls()
            .iter()
            .map(|(_, messages)| messages.len())
            .collect::<Vec<_>>(),
        [1, 3, 5]
    );
    let entries = runtime.history().entries();
    assert_eq!(entries.len(), 6);
    assert!(entries.iter().all(|entry| entry.session_id == "key:review"));
    assert_ne!(entries[0].agent_id, entries[2].agent_id);
    let json = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    assert!(!json.to_string().contains("keep_alive"));
    Ok(())
}

/// Each frames share explicit keys while omitted keys and separate runtimes stay isolated.
#[tokio::test]
async fn each_keeps_keyed_sessions_and_isolates_fresh_invocations() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(json!("a"))
        .then_output(json!("b"))
        .then_output(json!("c"))
        .then_output(json!("d"))
        .then_output(json!("e"));
    let flow = compile(each)?;
    let executor = flow.prepared().executor(context(&script)?);
    let mut runtime = flow.start(
        vec![
            request(Some("a")),
            request(Some("b")),
            request(Some("a")),
            request(None),
            request(None),
        ],
        Uuid::nil(),
    )?;
    host::finish(&mut runtime, &executor).await?;
    assert_eq!(
        script
            .calls()
            .iter()
            .map(|(_, messages)| messages.len())
            .collect::<Vec<_>>(),
        [1, 1, 3, 1, 1]
    );
    let entries = runtime.history().entries();
    assert_ne!(entries[6].session_id, entries[8].session_id);
    let independent = flow.start(vec![request(Some("a"))], Uuid::from_u128(1))?;
    assert!(independent.history().is_empty());
    Ok(())
}

/// A changed graph imports completed history by key without changing archived provenance or IDs.
#[tokio::test]
async fn new_graph_selects_existing_history_by_key() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(json!("old"))
        .then_output(json!("one"))
        .then_output(json!("two"))
        .then_output(json!("three"));
    let (first, history) = completed_history(&script).await?;
    let old_ids = history
        .entries()
        .iter()
        .map(|entry| entry.id)
        .collect::<Vec<_>>();
    let next = compile(parent)?;
    assert_ne!(
        first.prepared().fingerprint(),
        next.prepared().fingerprint()
    );
    let mut runtime =
        next.start_with_history(request(Some("review")), Uuid::from_u128(2), history)?;
    host::finish(&mut runtime, &next.prepared().executor(context(&script)?)).await?;
    assert_eq!(
        script
            .calls()
            .iter()
            .map(|(_, messages)| messages.len())
            .collect::<Vec<_>>(),
        [1, 3, 5, 7]
    );
    assert_eq!(
        runtime
            .history()
            .entries()
            .iter()
            .take(2)
            .map(|entry| entry.id)
            .collect::<Vec<_>>(),
        old_ids
    );
    assert!(
        runtime
            .history()
            .entries()
            .windows(2)
            .all(|pair| pair[0].position < pair[1].position)
    );
    Ok(())
}

/// Creates a completed conversation for import without changing stored identities or positions.
async fn completed_history(
    script: &ScriptedFactory,
) -> Result<(pravah::CompiledFlow<Request, String>, MessageHistory), GraphError> {
    let flow = compile(ask)?;
    let mut runtime = flow.start(request(Some("review")), Uuid::nil())?;
    host::finish(&mut runtime, &flow.prepared().executor(context(script)?)).await?;
    Ok((flow, runtime.history().clone()))
}

/// Invalid imports and empty keys fail before model execution and without changing runtime history.
#[tokio::test]
async fn invalid_history_and_keys_are_rejected() -> Result<(), GraphError> {
    let flow = compile(ask)?;
    let mut history = MessageHistory::new();
    history.push("key:review", "old", Message::user("unfinished"));
    assert!(matches!(
        flow.start_with_history(request(Some("review")), Uuid::nil(), history),
        Err(GraphError::HistoryValidation(_))
    ));
    let script = ScriptedFactory::new();
    let mut runtime = flow.start(request(Some("  ")), Uuid::nil())?;
    assert!(
        host::finish(&mut runtime, &flow.prepared().executor(context(&script)?))
            .await
            .is_err()
    );
    assert!(runtime.history().is_empty());
    assert!(script.calls().is_empty());
    assert!(
        Chat::builder::<String, String>()
            .model("test:///test")
            .key("")
            .build(Context::default())
            .is_err()
    );
    Ok(())
}

/// Default Chat keys persist through restore and differ between independent Chat executions.
#[tokio::test]
async fn chat_default_keys_are_stable_and_isolated() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"))
        .then_output(json!("three"));
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .build(context(&script)?)?;
    chat.send("one").await?;
    let snapshot = chat.snapshot()?;
    let session = snapshot.history().entries()[0].session_id.clone();
    let mut restored = Chat::builder::<String, String>()
        .model("test:///test")
        .restore::<()>(snapshot, context(&script)?)?;
    restored.send("two").await?;
    assert!(
        restored
            .snapshot()?
            .history()
            .entries()
            .iter()
            .all(|entry| entry.session_id == session)
    );
    let mut independent = Chat::builder::<String, String>()
        .model("test:///test")
        .build(context(&script)?)?;
    independent.send("three").await?;
    assert_ne!(
        independent.snapshot()?.history().entries()[0].session_id,
        session
    );
    assert_eq!(
        script
            .calls()
            .iter()
            .map(|(_, messages)| messages.len())
            .collect::<Vec<_>>(),
        [1, 3, 1]
    );
    Ok(())
}
