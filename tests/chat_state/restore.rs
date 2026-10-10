use super::*;
use pravah::Flow;
use pravah::clients::ClientError;
use pravah::clients::ErrorKind;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Restoring unfinished turns preserves state and configuration without dispatching or reconfiguring.
#[tokio::test]
async fn failed_turn_roundtrips_keep_fresh_services_idle() -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let script =
        ScriptedFactory::new().then_err(ClientError::new(ErrorKind::Validation, "offline"));
    let mut chat = Chat::with_state(
        assistant,
        initial_state(),
        lifecycle::counted_context(script, calls.clone())?,
    )?;
    assert!(matches!(
        chat.send("question").await,
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    for snapshot in roundtrips(&chat.snapshot()?)? {
        let fresh = Arc::new(AtomicUsize::new(0));
        let script = ScriptedFactory::new();
        let mut restored = Chat::<String, String, Session>::from_snapshot(
            assistant,
            snapshot,
            lifecycle::counted_context(script.clone(), fresh.clone())?,
        )?;
        assert_eq!(restored.get()?, initial_state());
        assert!(matches!(
            restored.set(initial_state()),
            Err(GraphError::ChatNotReady { .. })
        ));
        assert!(matches!(
            restored.send("another").await,
            Err(GraphError::ChatNotReady { .. })
        ));
        assert_eq!(fresh.load(Ordering::SeqCst), 0);
        assert!(script.calls().is_empty());
    }
    Ok(())
}

/// Rejects missing state, bad values, malformed metadata and incompatible state types.
#[tokio::test]
async fn malformed_state_is_rejected() -> Result<(), TestError> {
    let chat = Chat::with_state(assistant, initial_state(), Context::default())?;
    let original = serde_json::to_value(chat.snapshot()?)?;
    let corruptions = [
        ("/state/application_state", json!(null)),
        (
            "/state/application_state/1",
            json!({"project":5,"visits":0}),
        ),
        ("/state/application_state/0/name", json!("")),
        ("/state/application_state/0/schema/type", json!(500)),
    ];
    for (path, value) in corruptions {
        let mut bad = original.clone();
        *bad.pointer_mut(path)
            .ok_or(TestError::Missing("snapshot field"))? = value;
        assert!(
            Chat::<String, String, Session>::from_snapshot(
                assistant,
                serde_json::from_value(bad)?,
                Context::default()
            )
            .is_err()
        );
    }
    assert!(matches!(
        Chat::<String, String, u64>::from_snapshot(assistant, chat.snapshot()?, Context::default()),
        Err(GraphError::SnapshotValidation(_))
    ));
    assert_eq!(original, serde_json::to_value(chat.snapshot()?)?);
    Ok(())
}

/// Graphs without the current Chat bootstrap are rejected by fingerprint.
#[tokio::test]
async fn old_chat_graph_is_rejected() -> Result<(), TestError> {
    let root = Flow::<String>::root();
    let start = root.mark();
    let _loop_edge = root
        .clone()
        .agent(assistant)
        .suspend::<String>()
        .goto(start);
    let flow = root.map(|value| value).finish::<String>()?;
    let old = flow
        .start("question".into(), uuid::Uuid::nil())?
        .snapshot()?;
    assert!(matches!(
        Chat::<String, String>::from_snapshot(assistant, old, Context::default()),
        Err(GraphError::GraphMismatch { .. })
    ));
    Ok(())
}
