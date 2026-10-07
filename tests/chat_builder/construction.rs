use super::*;
use pravah::testing::CapturingHistoryStore;
use pravah::{CompactionRequest, CompactionResult, Compactor};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Serialize, Deserialize, JsonSchema)]
struct State {
    project: String,
}

struct CountCalls(Arc<AtomicUsize>);

impl Compactor for CountCalls {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        _: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(CompactionResult::default())
    }
}

/// State and services are configured before build, retained through state-type changes and rebound on restore.
#[tokio::test]
async fn final_build_moves_state_and_services() -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let store = CapturingHistoryStore::new();
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let mut chat = builder()
        .compactor(CountCalls(calls.clone()))
        .state(123u32)
        .store(store.clone())
        .state(State {
            project: "Pravah".into(),
        })
        .build(context(&factory)?)?;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(store.record_count(), 0);
    assert!(factory.calls().is_empty());
    assert_eq!(chat.get()?.project, "Pravah");
    chat.send("question").await?;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(store.record_count(), 2);
    for snapshot in copies(&chat.snapshot()?)? {
        check_restored(snapshot).await?;
    }
    assert_eq!(store.record_count(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Restore takes only persisted state and freshly installed services, without external work.
async fn check_restored(snapshot: Snapshot) -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let store = CapturingHistoryStore::new();
    let factory = ScriptedFactory::new().then_output(serde_json::json!("next"));
    let mut chat = builder()
        .store(store.clone())
        .compactor(CountCalls(calls.clone()))
        .restore::<State>(snapshot, context(&factory)?)?;
    assert_eq!(chat.get()?.project, "Pravah");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(store.record_count(), 0);
    assert!(factory.calls().is_empty());
    chat.send("next").await?;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(store.record_count(), 2);
    Ok(())
}

/// Repeated service setters replace prior services without running either during construction.
#[tokio::test]
async fn service_setters_replace_previous_values() -> Result<(), TestError> {
    let previous = Arc::new(AtomicUsize::new(0));
    let current = Arc::new(AtomicUsize::new(0));
    let old_store = CapturingHistoryStore::new();
    let new_store = CapturingHistoryStore::new();
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let mut chat = builder()
        .compactor(CountCalls(previous.clone()))
        .store(old_store.clone())
        .compactor(CountCalls(current.clone()))
        .store(new_store.clone())
        .build(context(&factory)?)?;
    chat.send("question").await?;
    assert_eq!(previous.load(Ordering::SeqCst), 0);
    assert_eq!(current.load(Ordering::SeqCst), 1);
    assert_eq!(old_store.record_count(), 0);
    assert_eq!(new_store.record_count(), 2);
    Ok(())
}

/// Services remain external while their execution intent is explicitly checkpointed.
#[tokio::test]
async fn services_do_not_change_initial_snapshot() -> Result<(), TestError> {
    let factory = ScriptedFactory::new();
    let plain = builder().state(7u32).build(context(&factory)?)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let configured = builder()
        .state(7u32)
        .store(CapturingHistoryStore::new())
        .compactor(CountCalls(calls.clone()))
        .build(context(&factory)?)?;
    let mut plain = serde_json::to_value(plain.snapshot()?)?;
    let mut configured = serde_json::to_value(configured.snapshot()?)?;
    assert_ne!(
        plain["state"]["execution_id"],
        configured["state"]["execution_id"]
    );
    plain["state"]
        .as_object_mut()
        .ok_or(TestError::Missing("state"))?
        .remove("execution_id");
    configured["state"]
        .as_object_mut()
        .ok_or(TestError::Missing("state"))?
        .remove("execution_id");
    assert_eq!(plain["graph_fingerprint"], configured["graph_fingerprint"]);
    assert_eq!(
        configured["state"]["history_policy"],
        serde_json::json!({"persist":true,"load":true,"compact":true})
    );
    configured["state"]
        .as_object_mut()
        .expect("state")
        .remove("history_policy");
    assert_eq!(plain, configured);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(factory.calls().is_empty());
    Ok(())
}
