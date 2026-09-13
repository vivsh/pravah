use super::*;
use pravah::Flow;
use pravah::clients::ClientError;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Restoring unfinished turns preserves state and configuration without dispatching or reconfiguring.
#[tokio::test]
async fn failed_turn_roundtrips_keep_fresh_services_idle() -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let script = ScriptedFactory::new().then_err(ClientError::Validation("offline".into()));
    let mut chat = Chat::with_state(
        assistant,
        initial_state(),
        lifecycle::counted_context(script, calls.clone()),
    )
    .await?;
    assert!(matches!(
        chat.send("question".into()).await,
        Err(GraphError::AgentClient(_))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    for snapshot in roundtrips(&chat.snapshot()?)? {
        let fresh = Arc::new(AtomicUsize::new(0));
        let script = ScriptedFactory::new();
        let mut restored = Chat::<String, String, Session>::from_snapshot(
            assistant,
            snapshot,
            lifecycle::counted_context(script.clone(), fresh.clone()),
        )?;
        assert_eq!(restored.get()?, initial_state());
        assert!(matches!(
            restored.set(initial_state()),
            Err(GraphError::ChatNotReady { .. })
        ));
        assert!(matches!(
            restored.send("another".into()).await,
            Err(GraphError::ChatNotReady { .. })
        ));
        assert_eq!(fresh.load(Ordering::SeqCst), 0);
        assert!(script.calls().is_empty());
    }
    Ok(())
}

/// Rejects missing state, bad values and corrupted stable variable identities or epochs.
#[tokio::test]
async fn malformed_state_is_rejected() -> Result<(), TestError> {
    let chat = Chat::with_state(assistant, initial_state(), Context::default()).await?;
    let original = serde_json::to_value(chat.snapshot()?)?;
    let corruptions = [
        ("/state/frames/0/variables", json!([])),
        (
            "/state/frames/0/variables/0/value",
            json!({"project":5,"visits":0}),
        ),
        ("/state/frames/0/variables/0/variable", json!(500)),
        ("/state/frames/0/variables/0/epoch", json!(0)),
        ("/state/frames/0/variables/0/epoch", json!(u64::MAX)),
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
        Err(GraphError::GraphMismatch { .. })
    ));
    assert_eq!(original, serde_json::to_value(chat.snapshot()?)?);
    Ok(())
}

/// Old lazy-chat graphs are rejected by fingerprint without changing the global snapshot format.
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
        .start("question".into(), Context::default())?
        .snapshot()?;
    assert!(matches!(
        Chat::<String, String>::from_snapshot(assistant, old, Context::default()),
        Err(GraphError::GraphMismatch { .. })
    ));
    Ok(())
}
