//! Frame lifetime, explicit keyed release, and durable conversation reactivation.

use pravah::clients::Message;
use pravah::testing::{CapturingHistoryStore, ScriptedFactory, providers};
use pravah::{
    Agent, AgentConfig, Chat, Context, Flow, GraphError, Runtime, Snapshot, Step, compile,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

#[path = "support/host.rs"]
mod host;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Request {
    key: Option<String>,
}

fn request(key: Option<&str>) -> Request {
    Request {
        key: key.map(str::to_owned),
    }
}

async fn configure(input: Request, _: Context) -> Result<AgentConfig, GraphError> {
    let config = AgentConfig::new("test:///test", "Answer.", Message::user("question"));
    Ok(match input.key {
        Some(key) => config.key(key),
        None => config,
    })
}

fn assistant(root: Agent<Request>) -> Agent<String> {
    root.configure(configure)
}

fn paused(root: Flow<Request>) -> Flow<String> {
    root.agent(assistant).suspend::<String>()
}

fn nested(root: Flow<Request>) -> Flow<bool> {
    root.agent(assistant)
        .map(|_| request(None))
        .flow(paused)
        .suspend::<bool>()
}

fn each(root: Flow<Vec<Request>>) -> Flow<Vec<String>> {
    root.each(paused)
}

fn twice(root: Flow<Request>) -> Flow<String> {
    root.agent(assistant)
        .map(|_| request(None))
        .agent(assistant)
        .suspend::<String>()
}

fn tool_agent(root: Agent<Request>) -> Agent<String> {
    root.tools(|tools| tools.flow(paused)).configure(configure)
}

fn tool_parent(root: Flow<Request>) -> Flow<String> {
    root.agent(tool_agent).suspend::<String>()
}

async fn control(
    _: pravah::AgentLoop<Request>,
    _: Context,
) -> Result<pravah::AgentDecision, GraphError> {
    Ok(pravah::AgentDecision::suspend(pravah::graph::Value::from(
        "approval",
    )))
}

fn controlled(root: Flow<Request>) -> Flow<String> {
    root.agent::<String>(|agent| agent.control(control).configure(configure))
        .suspend::<String>()
}

fn context(script: &ScriptedFactory) -> Result<Context, GraphError> {
    Ok(Context::default().with_providers(providers(script.clone())?))
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::SnapshotValidation(error.to_string())
}

fn wire(runtime: &Runtime) -> Result<serde_json::Value, GraphError> {
    serde_json::to_value(runtime.snapshot()?).map_err(codec)
}

/// Encodes the same continuation using either supported codec, without retaining another runtime owner.
fn roundtrip(snapshot: Snapshot, cbor: bool) -> Result<Snapshot, GraphError> {
    if cbor {
        let mut bytes = Vec::new();
        ciborium::into_writer(&snapshot, &mut bytes).map_err(codec)?;
        ciborium::from_reader(bytes.as_slice()).map_err(codec)
    } else {
        serde_json::from_slice(&serde_json::to_vec(&snapshot).map_err(codec)?).map_err(codec)
    }
}

/// Unkeyed history survives agent completion and restoration, then disappears exactly at root-frame exit.
#[tokio::test]
async fn root_lifetime_survives_both_codecs_and_cleans_without_a_store() -> Result<(), GraphError> {
    for cbor in [false, true] {
        let script = ScriptedFactory::new().then_output(json!("answer"));
        let flow = compile(paused)?;
        let executor = flow.prepared().executor(context(&script)?);
        let mut runtime = flow.start(request(None), Uuid::nil())?;
        assert!(matches!(
            host::finish(&mut runtime, &executor).await?,
            Step::Suspend(_)
        ));
        let session = runtime.history().entries()[0].session_id.clone();
        assert_eq!(runtime.history().entries().len(), 2);
        assert!(runtime.conversation_is_active(&session));
        let before = wire(&runtime)?;
        assert!(matches!(
            runtime.drop_conversation(&session),
            Err(GraphError::AgentConversationBusy)
        ));
        assert_eq!(wire(&runtime)?, before);
        runtime = flow.restore(roundtrip(runtime.snapshot()?, cbor)?)?;
        assert!(runtime.conversation_is_active(&session));
        runtime.resume("finish")?;
        assert!(matches!(
            host::finish(&mut runtime, &executor).await?,
            Step::Done(_)
        ));
        assert!(runtime.history().is_empty());
        assert!(!runtime.conversation_is_active(&session));
        assert_eq!(wire(&runtime)?["history"]["next_position"], 2);
        assert!(
            flow.restore(roundtrip(runtime.snapshot()?, cbor)?)?
                .history()
                .is_empty()
        );
        assert_eq!(script.calls().len(), 1);
    }
    Ok(())
}

/// Child cleanup releases only child sessions; a parent session remains frame-owned or keyed as configured.
#[tokio::test]
async fn child_exit_preserves_parent_and_keyed_history() -> Result<(), GraphError> {
    for key in [None, Some("parent")] {
        let script = ScriptedFactory::new()
            .then_output(json!("parent"))
            .then_output(json!("child"));
        let flow = compile(nested)?;
        let executor = flow.prepared().executor(context(&script)?);
        let mut runtime = flow.start(request(key), Uuid::nil())?;
        assert!(matches!(
            host::finish(&mut runtime, &executor).await?,
            Step::Suspend(_)
        ));
        assert_eq!(runtime.state().frame_depth(), 2);
        let parent = runtime.history().entries()[0].session_id.clone();
        let child = runtime.history().entries()[2].session_id.clone();
        runtime = flow.restore(roundtrip(runtime.snapshot()?, true)?)?;
        runtime.resume("child complete")?;
        assert!(matches!(
            host::finish(&mut runtime, &executor).await?,
            Step::Suspend(_)
        ));
        assert_eq!(runtime.state().frame_depth(), 1);
        assert_eq!(runtime.history().entries().len(), 2);
        assert!(!runtime.conversation_is_active(&child));
        assert_eq!(runtime.conversation_is_active(&parent), key.is_none());
        runtime.resume(true)?;
        assert!(matches!(
            host::finish(&mut runtime, &executor).await?,
            Step::Done(_)
        ));
        assert_eq!(
            runtime.history().entries().len(),
            if key.is_some() { 2 } else { 0 }
        );
        assert!(!runtime.conversation_is_active(&parent));
    }
    Ok(())
}

/// Repeated instances of one each child get distinct ownership, not a shared authored-agent lifetime.
#[tokio::test]
async fn each_releases_each_completed_child_independently() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"));
    let flow = compile(each)?;
    let executor = flow.prepared().executor(context(&script)?);
    let mut runtime = flow.start(vec![request(None), request(None)], Uuid::nil())?;
    let mut previous = None;
    for index in 0..2 {
        assert!(matches!(
            host::finish(&mut runtime, &executor).await?,
            Step::Suspend(_)
        ));
        assert_eq!(runtime.history().entries().len(), 2);
        let session = runtime.history().entries()[0].session_id.clone();
        assert_ne!(previous.as_ref(), Some(&session));
        if let Some(previous) = &previous {
            assert!(!runtime.conversation_is_active(previous));
        }
        runtime = flow.restore(roundtrip(runtime.snapshot()?, index == 1)?)?;
        runtime.resume("complete")?;
        previous = Some(session);
    }
    assert!(matches!(
        host::finish(&mut runtime, &executor).await?,
        Step::Done(_)
    ));
    assert!(runtime.history().is_empty());
    Ok(())
}

/// Dropping a completed keyed chat removes only working rows; its next turn reloads archived IDs exactly once.
#[tokio::test]
async fn keyed_drop_restores_and_reloads_without_deleting_the_archive() -> Result<(), GraphError> {
    let store = CapturingHistoryStore::new();
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"));
    let builder = || {
        Chat::builder::<String, String>()
            .model("test:///test")
            .key("thread")
            .store(store.clone())
    };
    let mut chat = builder().build(context(&script)?)?;
    chat.send_with_key("first", "message-1").await?;
    let original = store.all_entries();
    assert!(!chat.conversation_is_active("key:thread"));
    chat.drop_conversation("key:thread")?;
    assert!(chat.snapshot()?.history().is_empty());
    assert_eq!(store.record_count(), 2);
    chat = builder().restore::<()>(roundtrip(chat.snapshot()?, true)?, context(&script)?)?;
    chat.send("second").await?;
    assert_eq!(script.calls()[1].1.len(), 3);
    let snapshot = chat.snapshot()?;
    assert_eq!(snapshot.history().entries().len(), 4);
    assert_eq!(snapshot.history().entries()[0].id, original[0].id);
    assert_eq!(
        snapshot.history().entries()[0].message.key.as_deref(),
        Some("message-1")
    );
    assert_eq!(store.record_count(), 4);
    assert!(
        snapshot
            .history()
            .entries()
            .windows(2)
            .all(|pair| pair[0].position < pair[1].position)
    );
    Ok(())
}

/// Ordered ownership is checked at restore, including missing lists and sessions moved to another frame.
#[tokio::test]
async fn corrupt_ownership_cannot_restore() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"));
    let flow = compile(twice)?;
    let executor = flow.prepared().executor(context(&script)?);
    let mut runtime = flow.start(request(None), Uuid::nil())?;
    host::finish(&mut runtime, &executor).await?;
    let snapshot = wire(&runtime)?;
    let owners = &snapshot["state"]["frames"][0]["unkeyed_conversations"];
    let invalid = [
        json!([]),
        json!([""]),
        json!(["key:other"]),
        json!([owners[0], owners[0]]),
        json!([owners[1], owners[0]]),
    ];
    for owners in invalid {
        let mut corrupted = snapshot.clone();
        corrupted["state"]["frames"][0]["unkeyed_conversations"] = owners;
        assert!(matches!(
            flow.restore(serde_json::from_value(corrupted).map_err(codec)?),
            Err(GraphError::SnapshotValidation(_))
        ));
    }
    assert_swapped_owners_rejected().await
}

/// Even ordered ownership lists cannot be moved between distinct authored frames.
async fn assert_swapped_owners_rejected() -> Result<(), GraphError> {
    let flow = compile(nested)?;
    let script = ScriptedFactory::new()
        .then_output(json!("parent"))
        .then_output(json!("child"));
    let mut runtime = flow.start(request(None), Uuid::nil())?;
    host::finish(&mut runtime, &flow.prepared().executor(context(&script)?)).await?;
    let snapshot = wire(&runtime)?;
    let mut corrupted = snapshot.clone();
    corrupted["state"]["frames"][0]["unkeyed_conversations"] =
        snapshot["state"]["frames"][1]["unkeyed_conversations"].clone();
    corrupted["state"]["frames"][1]["unkeyed_conversations"] =
        snapshot["state"]["frames"][0]["unkeyed_conversations"].clone();
    assert!(matches!(
        flow.restore(serde_json::from_value(corrupted).map_err(codec)?),
        Err(GraphError::SnapshotValidation(_))
    ));
    Ok(())
}

/// Pending and accepted worker results retain their conversation until the owning transition commits.
#[tokio::test]
async fn pending_and_accepted_work_prevent_keyed_drop_atomically() -> Result<(), GraphError> {
    let store = CapturingHistoryStore::new();
    let script = ScriptedFactory::new().then_output(json!("answer"));
    let flow = compile(paused)?;
    let executor = flow
        .prepared()
        .executor(context(&script)?)
        .with_store(store);
    let mut runtime = flow
        .start(request(Some("thread")), Uuid::nil())?
        .with_history(pravah::HistoryPolicy {
            persist: true,
            load: true,
            compact: false,
        })?;
    let mut boundaries = Vec::new();
    for _ in 0..40 {
        match runtime.next()? {
            Step::Continue => {}
            Step::Agent(request) => {
                if request.kind() != "configure" {
                    assert_drop_rejected(&mut runtime, "key:thread")?;
                }
                boundaries.push(request.kind());
                runtime = flow.restore(roundtrip(runtime.snapshot()?, true)?)?;
                runtime.resume_agent(executor.execute(&request).await)?;
                assert_drop_rejected(&mut runtime, "key:thread")?;
            }
            Step::Suspend(_) => break,
            Step::Done(_) => return Err(codec("unexpected completion")),
        }
    }
    assert_eq!(boundaries, ["configure", "generate", "persist_history"]);
    assert!(!runtime.conversation_is_active("key:thread"));
    runtime.drop_conversation("key:thread")?;
    assert!(runtime.history().is_empty());
    Ok(())
}

/// A rejected release must preserve the entire snapshot, not merely its message count.
fn assert_drop_rejected(runtime: &mut Runtime, session: &str) -> Result<(), GraphError> {
    let before = wire(runtime)?;
    assert!(runtime.conversation_is_active(session));
    assert!(matches!(
        runtime.drop_conversation(session),
        Err(GraphError::AgentConversationBusy)
    ));
    assert_eq!(wire(runtime)?, before);
    Ok(())
}

/// A completed key may be explicitly discarded without a store; the following turn starts fresh.
#[tokio::test]
async fn keyed_drop_without_store_discards_context_and_keeps_other_sessions()
-> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"));
    let builder = || {
        Chat::builder::<String, String>()
            .model("test:///test")
            .key("thread")
    };
    let mut chat = builder().build(context(&script)?)?;
    chat.send("first").await?;
    chat.drop_conversation("key:missing")?;
    assert_eq!(chat.snapshot()?.history().entries().len(), 2);
    chat.drop_conversation("key:thread")?;
    chat.drop_conversation("key:thread")?;
    chat = builder().restore::<()>(roundtrip(chat.snapshot()?, false)?, context(&script)?)?;
    chat.send("second").await?;
    assert_eq!(script.calls()[1].1.len(), 1);
    assert_eq!(chat.snapshot()?.history().entries()[0].position, 2);
    Ok(())
}

/// A suspended agent-tool child owns only its own conversation; parent tool groups survive child cleanup.
#[tokio::test]
async fn tool_child_cleanup_preserves_the_parent_exchange() -> Result<(), GraphError> {
    for key in [None, Some("parent")] {
        assert_tool_child_cleanup(key).await?;
    }
    Ok(())
}

/// Drives a suspended tool flow and checks its parent tool-call/result group after child exit.
async fn assert_tool_child_cleanup(key: Option<&str>) -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_tool_calls(vec![pravah::testing::mock_tool_call(
            "call",
            "request",
            json!({"key": null}),
        )])
        .then_output(json!("child"))
        .then_output(json!("final"));
    let flow = compile(tool_parent)?;
    let executor = flow.prepared().executor(context(&script)?);
    let mut runtime = flow.start(request(key), Uuid::nil())?;
    assert!(matches!(
        host::finish(&mut runtime, &executor).await?,
        Step::Suspend(_)
    ));
    let parent = runtime.history().entries()[0].session_id.clone();
    let child = runtime.history().entries()[2].session_id.clone();
    assert_drop_rejected(&mut runtime, &parent)?;
    assert_drop_rejected(&mut runtime, &child)?;
    runtime = flow.restore(roundtrip(runtime.snapshot()?, true)?)?;
    runtime.resume("tool result")?;
    assert!(matches!(
        host::finish(&mut runtime, &executor).await?,
        Step::Suspend(_)
    ));
    assert!(!runtime.conversation_is_active(&child));
    let rows = runtime.history().entries();
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|entry| entry.session_id == parent));
    assert_tool_group(rows);
    assert_eq!(script.calls()[2].1.len(), 3);
    runtime.resume("done")?;
    host::finish(&mut runtime, &executor).await?;
    assert_eq!(
        runtime.history().entries().len(),
        if key.is_some() { 4 } else { 0 }
    );
    Ok(())
}

fn assert_tool_group(rows: &[pravah::HistoryEntry]) {
    assert!(matches!(
        rows[1].message.role,
        pravah::clients::Role::AssistantToolCalls { .. }
    ));
    assert!(matches!(
        rows[2].message.role,
        pravah::clients::Role::Tool { .. }
    ));
}

/// A suspended controller retains keyed and frame-local sessions through restoration and explicit resume.
#[tokio::test]
async fn controller_suspension_cannot_release_its_conversation() -> Result<(), GraphError> {
    for key in [None, Some("thread")] {
        let script = ScriptedFactory::new().then_output(json!("answer"));
        let flow = compile(controlled)?;
        let executor = flow.prepared().executor(context(&script)?);
        let mut runtime = flow.start(request(key), Uuid::nil())?;
        assert!(matches!(
            host::finish(&mut runtime, &executor).await?,
            Step::Suspend(_)
        ));
        assert!(script.calls().is_empty());
        let session = runtime.history().entries()[0].session_id.clone();
        assert_drop_rejected(&mut runtime, &session)?;
        runtime = flow.restore(roundtrip(runtime.snapshot()?, key.is_some())?)?;
        assert_drop_rejected(&mut runtime, &session)?;
        runtime.resume(pravah::AgentResume::Continue)?;
        assert!(matches!(
            host::finish(&mut runtime, &executor).await?,
            Step::Suspend(_)
        ));
        assert_eq!(runtime.conversation_is_active(&session), key.is_none());
        runtime.resume("done")?;
        host::finish(&mut runtime, &executor).await?;
        assert!(!runtime.conversation_is_active(&session));
        assert_eq!(
            runtime.history().entries().len(),
            if key.is_some() { 2 } else { 0 }
        );
        assert_eq!(script.calls().len(), 1);
    }
    Ok(())
}

/// Automatic frame cleanup neither deletes archived rows nor waits for a store after acknowledgements.
#[tokio::test]
async fn automatic_cleanup_leaves_the_archive_intact() -> Result<(), GraphError> {
    let store = CapturingHistoryStore::new();
    let script = ScriptedFactory::new().then_output(json!("answer"));
    let flow = compile(paused)?;
    let executor = flow
        .prepared()
        .executor(context(&script)?)
        .with_store(store.clone());
    let mut runtime =
        flow.start(request(None), Uuid::nil())?
            .with_history(pravah::HistoryPolicy {
                persist: true,
                load: false,
                compact: false,
            })?;
    host::finish(&mut runtime, &executor).await?;
    let archived = store.all_entries();
    assert_eq!(archived.len(), 2);
    runtime.resume("done")?;
    host::finish(&mut runtime, &executor).await?;
    assert!(runtime.history().is_empty());
    assert_eq!(
        serde_json::to_value(store.all_entries()).map_err(codec)?,
        serde_json::to_value(archived).map_err(codec)?
    );
    assert_eq!(store.record_count(), 2);
    Ok(())
}

/// A failed accepted generation remains owned across restoration and cannot be dropped or dispatched again.
#[tokio::test]
async fn failed_generation_remains_active_and_retryable() -> Result<(), GraphError> {
    let script = ScriptedFactory::new().then_err(pravah::clients::ClientError::new(
        pravah::clients::ErrorKind::Provider,
        "offline failure",
    ));
    let flow = compile(paused)?;
    let executor = flow.prepared().executor(context(&script)?);
    let mut runtime = flow.start(request(Some("thread")), Uuid::nil())?;
    assert!(matches!(
        host::finish(&mut runtime, &executor).await,
        Err(GraphError::AgentFailed { .. })
    ));
    assert_drop_rejected(&mut runtime, "key:thread")?;
    runtime = flow.restore(roundtrip(runtime.snapshot()?, true)?)?;
    let before = wire(&runtime)?;
    assert!(matches!(
        runtime.next(),
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(wire(&runtime)?, before);
    assert_drop_rejected(&mut runtime, "key:thread")?;
    assert_eq!(script.calls().len(), 1);
    Ok(())
}

/// Automatic cleanup changes neither cumulative usage nor next positions after original messages are removed.
#[tokio::test]
async fn automatic_cleanup_preserves_usage() -> Result<(), GraphError> {
    let usage = pravah::clients::TokenUsage::new()
        .with_input(10)
        .with_output(2);
    let response = pravah::testing::output_response(json!("answer")).with_usage(Some(usage));
    let script = ScriptedFactory::new().then(response);
    let flow = compile(paused)?;
    let executor = flow.prepared().executor(context(&script)?);
    let mut runtime = flow.start(request(None), Uuid::nil())?;
    host::finish(&mut runtime, &executor).await?;
    assert_eq!(runtime.history().total_usage(), Some(12));
    runtime.resume("done")?;
    host::finish(&mut runtime, &executor).await?;
    assert!(runtime.history().is_empty());
    assert_eq!(runtime.history().total_usage(), Some(12));
    assert_eq!(wire(&runtime)?["history"]["next_position"], 2);
    Ok(())
}

/// Archived usage remains inspectable but repeated drop/reload adds only newly generated usage to counters.
#[tokio::test]
async fn archive_reloads_never_double_count_usage() -> Result<(), GraphError> {
    let store = CapturingHistoryStore::new();
    seed_archive(&store).await?;
    let first = pravah::clients::TokenUsage::new()
        .with_input(10)
        .with_output(2);
    let second = pravah::clients::TokenUsage::new()
        .with_input(20)
        .with_output(4);
    let script = ScriptedFactory::new()
        .then(pravah::testing::output_response(json!("one")).with_usage(Some(first)))
        .then(pravah::testing::output_response(json!("two")).with_usage(Some(second)));
    let builder = || {
        Chat::builder::<String, String>()
            .model("test:///test")
            .key("thread")
            .store(store.clone())
    };
    let mut chat = builder().build(context(&script)?)?;
    chat.send("first").await?;
    assert_eq!(chat.snapshot()?.history().total_usage(), Some(12));
    chat.drop_conversation("key:thread")?;
    assert_eq!(chat.snapshot()?.history().total_usage(), Some(12));
    chat = builder().restore::<()>(roundtrip(chat.snapshot()?, true)?, context(&script)?)?;
    chat.send("second").await?;
    let snapshot = chat.snapshot()?;
    assert_eq!(snapshot.history().total_usage(), Some(36));
    assert_eq!(
        snapshot.history().entries()[1]
            .message
            .usage
            .and_then(|usage| usage.total()),
        Some(18)
    );
    assert_eq!(store.record_count(), 6);
    assert_eq!(
        script
            .calls()
            .iter()
            .map(|(_, messages)| messages.len())
            .collect::<Vec<_>>(),
        [3, 5]
    );
    Ok(())
}

/// Seeds a completed conversation using the existing store, without a runtime or a new history owner.
async fn seed_archive(store: &CapturingHistoryStore) -> Result<(), GraphError> {
    use pravah::HistoryStore;
    let mut history = pravah::MessageHistory::new();
    history.push("key:thread", "old", Message::user("prior"));
    history.push(
        "key:thread",
        "old",
        Message::assistant("prior").with_usage(
            pravah::clients::TokenUsage::new()
                .with_input(15)
                .with_output(3),
        ),
    );
    for entry in history.entries() {
        store.record(entry).await.map_err(|never| match never {})?;
    }
    Ok(())
}

/// Dropping one completed key cannot remove other sessions or reset the runtime append cursor.
#[tokio::test]
async fn explicit_drop_is_session_local() -> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"));
    let flow = compile(each)?;
    let executor = flow.prepared().executor(context(&script)?);
    let mut runtime = flow.start(
        vec![request(Some("one")), request(Some("two"))],
        Uuid::nil(),
    )?;
    for _ in 0..2 {
        host::finish(&mut runtime, &executor).await?;
        runtime.resume("done")?;
    }
    host::finish(&mut runtime, &executor).await?;
    runtime.drop_conversation("key:one")?;
    assert_eq!(runtime.history().entries().len(), 2);
    assert!(
        runtime
            .history()
            .entries()
            .iter()
            .all(|entry| entry.session_id == "key:two")
    );
    assert_eq!(wire(&runtime)?["history"]["next_position"], 4);
    Ok(())
}
