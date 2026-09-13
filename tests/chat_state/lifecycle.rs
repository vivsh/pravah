use super::*;
use pravah::deps::Deps;
use pravah::testing::CapturingHistoryStore;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Provides a runtime-only configuration counter, never part of application state.
pub(super) fn counted_context(script: ScriptedFactory, calls: Arc<AtomicUsize>) -> Context {
    let mut deps = Deps::default();
    deps.insert(calls);
    context(script).with_deps(deps)
}

/// Construction and pristine restoration neither configure agents nor call clients or history stores.
#[tokio::test]
async fn initialized_chat_is_snapshot_ready_without_external_work() -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let script = ScriptedFactory::new().then_output(json!("answer"));
    let store = CapturingHistoryStore::new();
    let chat = Chat::with_state(
        assistant,
        initial_state(),
        counted_context(script.clone(), calls.clone()),
    )
    .await?
    .with_store(store.clone());
    for snapshot in roundtrips(&chat.snapshot()?)? {
        let mut restored = Chat::<String, String, Session>::from_snapshot(
            assistant,
            snapshot,
            counted_context(script.clone(), calls.clone()),
        )?;
        assert_eq!(restored.get()?, initial_state());
        restored.set(Session {
            visits: 3,
            ..initial_state()
        })?;
        assert!(restored.snapshot()?.history().entries().is_empty());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(store.record_count(), 0);
    assert!(script.calls().is_empty());
    Ok(())
}

/// Two stateful chats share a graph identity but never share mutable state.
#[tokio::test]
async fn values_do_not_change_fingerprints_or_leak_between_chats() -> Result<(), TestError> {
    let mut first = Chat::with_state(assistant, initial_state(), Context::default()).await?;
    let second = Chat::with_state(
        assistant,
        Session {
            visits: 5,
            ..initial_state()
        },
        Context::default(),
    )
    .await?;
    let first_json = serde_json::to_value(first.snapshot()?)?;
    let second_json = serde_json::to_value(second.snapshot()?)?;
    assert_eq!(
        first_json.get("graph_fingerprint"),
        second_json.get("graph_fingerprint")
    );
    first.set(Session {
        visits: 9,
        ..initial_state()
    })?;
    assert_eq!(second.get()?.visits, 5);
    let mut unit = Chat::new(assistant, Context::default()).await?;
    unit.get()?;
    unit.set(())?;
    assert!(unit.snapshot()?.history().entries().is_empty());
    Ok(())
}

/// Cancelled configuration retains runtime state and blocks input/state replacement.
#[tokio::test]
async fn cancelled_turn_is_not_an_input_boundary() -> Result<(), TestError> {
    use futures::FutureExt;
    let mut chat = Chat::with_state(waiting_agent, initial_state(), Context::default()).await?;
    assert!(chat.send("question".into()).now_or_never().is_none());
    let before = serde_json::to_value(chat.snapshot()?)?;
    assert!(matches!(
        chat.set(initial_state()),
        Err(GraphError::ChatNotReady { operation: "set" })
    ));
    assert!(matches!(
        chat.send("another".into()).await,
        Err(GraphError::ChatNotReady { operation: "send" })
    ));
    assert_eq!(chat.get()?, initial_state());
    assert_eq!(before, serde_json::to_value(chat.snapshot()?)?);
    Ok(())
}

fn waiting_agent(root: Agent<String>) -> Agent<String> {
    root.configure(wait_forever)
}
async fn wait_forever(_input: String, _ctx: Context) -> Result<AgentConfig, GraphError> {
    std::future::pending().await
}
