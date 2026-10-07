//! Worker-owned persistence and preparation of a synchronous graph's retained history.

use pravah::clients::Message;
use pravah::testing::{CapturingHistoryStore, ScriptedFactory};
use pravah::{
    Agent, AgentConfig, Chat, CompactionRequest, CompactionResult, Compactor, Context, Flow,
    GraphError, HistoryEntry, HistoryPolicy, HistoryStore, Runtime, Step, compile,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use uuid::Uuid;

fn agent(root: Agent<String>) -> Agent<String> {
    root.configure(configure)
}
async fn configure(input: String, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new("test:///test", "Answer", Message::user(input)).key("shared"))
}
fn flow(root: Flow<String>) -> Flow<String> {
    root.agent(agent)
}
fn context(factory: ScriptedFactory) -> Result<Context, GraphError> {
    Ok(Context::default().with_providers(pravah::testing::providers(factory)?))
}

/// Drives the same VM, accepting each portable worker completion exactly once.
async fn finish(runtime: &mut Runtime, executor: &pravah::AgentExecutor) -> Result<(), GraphError> {
    loop {
        match runtime.next()? {
            Step::Continue => {}
            Step::Agent(request) => runtime.resume_agent(executor.execute(&request).await)?,
            Step::Done(_) => return Ok(()),
            Step::Suspend(_) => return Err(GraphError::ChatSuspended),
        }
    }
}

/// A persisted graph acknowledges every original message once, including its final response.
#[tokio::test]
async fn graph_persists_once_through_worker() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let workflow = compile(flow)?;
    let store = CapturingHistoryStore::new();
    let executor = workflow
        .prepared()
        .executor(context(factory.clone())?)
        .with_store(store.clone());
    let mut runtime = workflow
        .start("question".into(), Uuid::from_u128(1))?
        .with_history(HistoryPolicy {
            persist: true,
            ..Default::default()
        })?;
    finish(&mut runtime, &executor).await?;
    assert_eq!(runtime.history().entries().len(), 2);
    assert_eq!(
        serde_json::to_value(runtime.snapshot()?).map_err(codec)?["state"]["persisted_history_position"],
        2
    );
    assert_eq!(store.record_count(), 2);
    assert_eq!(factory.calls().len(), 1);
    Ok(())
}

struct Summarize(Arc<AtomicUsize>);
impl Compactor for Summarize {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(if request.committed().is_empty() {
            CompactionResult::default()
        } else {
            CompactionResult {
                evict_indices: (0..request.committed().len()).collect(),
                summary: Some("summary".into()),
            }
        })
    }
}

/// Chat prepares once per generation, persists every original row and physically bounds snapshots.
#[tokio::test]
async fn chat_bounds_history_and_preserves_all_original_messages() -> Result<(), GraphError> {
    let mut factory = ScriptedFactory::new();
    for _ in 0..8 {
        factory = factory.then_output(serde_json::json!("answer"));
    }
    let store = CapturingHistoryStore::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = Chat::new(agent, context(factory.clone())?)?
        .with_store(store.clone())
        .with_compactor(Summarize(calls.clone()));
    for _ in 0..8 {
        chat.send("question").await?;
    }
    assert_eq!(calls.load(Ordering::SeqCst), 8);
    assert_eq!(store.record_count(), 16);
    assert_eq!(chat.snapshot()?.history().entries().len(), 3);
    assert_eq!(
        factory.calls().last().map(|(_, messages)| messages.len()),
        Some(2)
    );
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error("store unavailable")]
struct StoreFailure;
#[derive(Clone)]
struct FailAssistant {
    fail: Arc<AtomicBool>,
    store: CapturingHistoryStore,
}
impl HistoryStore for FailAssistant {
    type Error = StoreFailure;
    async fn load(&self, key: &str) -> Result<Vec<HistoryEntry>, Self::Error> {
        self.store.load(key).await.map_err(|never| match never {})
    }
    async fn record(&self, entry: &HistoryEntry) -> Result<(), Self::Error> {
        if matches!(entry.message.role, pravah::clients::Role::Assistant)
            && self.fail.load(Ordering::SeqCst)
        {
            return Err(StoreFailure);
        }
        self.store
            .record(entry)
            .await
            .map_err(|never| match never {})
    }
}

/// An accepted persistence failure retains the completed generation without implicit retry.
#[tokio::test]
async fn accepted_persistence_failure_never_regenerates() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let fail = Arc::new(AtomicBool::new(true));
    let store = CapturingHistoryStore::new();
    let mut chat = Chat::new(agent, context(factory.clone())?)?.with_store(FailAssistant {
        fail: fail.clone(),
        store: store.clone(),
    });
    assert!(matches!(
        chat.send("question").await,
        Err(GraphError::AgentFailed { .. })
    ));
    let before =
        serde_json::to_value(chat.snapshot()?).map_err(|e| GraphError::Invalid(e.to_string()))?;
    assert_eq!(chat.snapshot()?.history().entries().len(), 2);
    assert_eq!(store.record_count(), 1);
    assert!(chat.send("another").await.is_err());
    fail.store(false, Ordering::SeqCst);
    assert!(matches!(chat.next(), Err(GraphError::AgentFailed { .. })));
    assert_eq!(store.record_count(), 1);
    assert_eq!(factory.calls().len(), 1);
    assert_eq!(
        before,
        serde_json::to_value(chat.snapshot()?).map_err(|e| GraphError::Invalid(e.to_string()))?
    );
    Ok(())
}

/// No policy preserves the complete conversation without preparation or persistence.
#[tokio::test]
async fn absent_policy_retains_all_history() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!("a"))
        .then_output(serde_json::json!("b"));
    let mut chat = Chat::new(agent, context(factory)?)?;
    chat.send("one").await?;
    chat.send("two").await?;
    assert_eq!(chat.snapshot()?.history().entries().len(), 4);
    Ok(())
}

/// One manager isolates acknowledgement positions for independent executions sharing a graph.
#[tokio::test]
async fn acknowledgement_positions_are_execution_scoped() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!("a"))
        .then_output(serde_json::json!("b"));
    let workflow = compile(flow)?;
    let store = CapturingHistoryStore::new();
    let executor = workflow
        .prepared()
        .executor(context(factory)?)
        .with_store(store.clone());
    for id in [Uuid::from_u128(1), Uuid::from_u128(2)] {
        let mut runtime = workflow
            .start("question".into(), id)?
            .with_history(HistoryPolicy {
                persist: true,
                ..Default::default()
            })?;
        finish(&mut runtime, &executor).await?;
    }
    let rows = store.all_entries();
    assert_eq!(rows.len(), 4);
    assert_ne!(rows.first().map(|e| e.id), rows.get(2).map(|e| e.id));
    assert_eq!(
        rows.first().map(|e| e.position),
        rows.get(2).map(|e| e.position)
    );
    Ok(())
}

/// Fresh workers restore saved acknowledgement positions rather than redelivering completed rows.
#[tokio::test]
async fn fresh_worker_preserves_acknowledgements_without_model_execution() -> Result<(), GraphError>
{
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let workflow = compile(flow)?;
    let store = CapturingHistoryStore::new();
    let executor = workflow
        .prepared()
        .executor(context(factory.clone())?)
        .with_store(store.clone());
    let mut runtime = workflow
        .start("question".into(), Uuid::from_u128(3))?
        .with_history(HistoryPolicy {
            persist: true,
            ..Default::default()
        })?;
    finish(&mut runtime, &executor).await?;
    let snapshot = runtime.snapshot()?;
    let json = serde_json::to_vec(&snapshot).map_err(codec)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor).map_err(codec)?;
    for copy in [
        serde_json::from_slice(&json).map_err(codec)?,
        ciborium::from_reader(cbor.as_slice()).map_err(codec)?,
    ] {
        let mut restored = workflow.restore(copy)?;
        let before = serde_json::to_value(restored.snapshot()?).map_err(codec)?;
        assert!(restored.next().is_err());
        assert_eq!(store.record_count(), 2);
        assert_eq!(
            before,
            serde_json::to_value(restored.snapshot()?).map_err(codec)?
        );
    }
    assert_eq!(factory.calls().len(), 1);
    Ok(())
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::Invalid(error.to_string())
}

struct NeverCompact(Arc<AtomicUsize>);
impl Compactor for NeverCompact {
    type Error = std::convert::Infallible;
    fn needs_compaction(&self, _: &CompactionRequest<'_>) -> bool {
        false
    }
    async fn compact(
        &self,
        _: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(CompactionResult::default())
    }
}

/// Synchronous policy triggers may skip all preparation without suppressing persistence.
#[tokio::test]
async fn disabled_trigger_keeps_history_and_still_persists() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let store = CapturingHistoryStore::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .store(store.clone())
        .compactor(NeverCompact(calls.clone()))
        .build(context(factory)?)?;
    chat.send("question").await?;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(store.record_count(), 2);
    assert_eq!(chat.snapshot()?.history().entries().len(), 2);
    Ok(())
}
