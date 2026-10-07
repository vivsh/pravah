//! Keyed loading and partial acknowledgements at the isolated worker boundary.

use pravah::clients::Message;
use pravah::testing::{CapturingHistoryStore, ScriptedFactory, providers};
use pravah::{AgentResponse, Chat, Context, GraphError, HistoryEntry, HistoryStore};
use serde_json::json;
use std::convert::Infallible;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone)]
struct Store {
    rows: CapturingHistoryStore,
    loads: Arc<AtomicUsize>,
}

impl HistoryStore for Store {
    type Error = Infallible;
    async fn load(&self, key: &str) -> Result<Vec<HistoryEntry>, Infallible> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.rows.load(key).await
    }
    async fn record(&self, entry: &HistoryEntry) -> Result<(), Infallible> {
        self.rows.record(entry).await
    }
}

fn builder() -> pravah::ChatBuilder<String, String> {
    Chat::builder().model("test:///model").key("thread")
}
fn context(script: &ScriptedFactory) -> Result<Context, GraphError> {
    Ok(Context::default().with_providers(providers(script.clone())?))
}
fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::SnapshotValidation(error.to_string())
}

struct Summarize;
/// Archived positions from different executions may overlap or decrease; store order and UUIDs are authoritative.
#[tokio::test]
async fn loads_preserve_store_order_across_execution_local_positions() -> Result<(), GraphError> {
    let store = CapturingHistoryStore::new();
    let mut prior = pravah::MessageHistory::new();
    for _ in 0..6 {
        prior.push("other", "agent", Message::user("unrelated"));
    }
    prior.push("key:thread", "agent", Message::user("original question"));
    prior.push("key:thread", "agent", Message::assistant("original answer"));
    for entry in prior.entries().iter().skip(6) {
        store.record(entry).await.map_err(|never| match never {})?;
    }
    let original_id = prior.entries()[6].id;
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"));
    let mut first = builder().store(store.clone()).build(context(&script)?)?;
    first.send("new question").await?;
    assert_eq!(
        store
            .all_entries()
            .iter()
            .map(|row| row.position)
            .collect::<Vec<_>>(),
        [6, 7, 2, 3]
    );
    let mut second = builder().store(store.clone()).build(context(&script)?)?;
    second.send("follow up").await?;
    assert_eq!(script.calls()[1].1.len(), 5);
    assert_eq!(second.snapshot()?.history().entries()[0].id, original_id);
    assert_eq!(store.record_count(), 6);
    Ok(())
}
impl pravah::Compactor for Summarize {
    type Error = Infallible;
    async fn compact(
        &self,
        request: pravah::CompactionRequest<'_>,
        _: Context,
    ) -> Result<pravah::CompactionResult, Infallible> {
        Ok(pravah::CompactionResult {
            evict_indices: (0..request.committed().len()).collect(),
            summary: (!request.committed().is_empty()).then(|| "prior exchange".into()),
        })
    }
}

/// Persistence and summary acknowledgements survive a later provider failure, without regenerating on restore.
#[tokio::test]
async fn successful_stages_survive_failed_generation() -> Result<(), GraphError> {
    let store = CapturingHistoryStore::new();
    let script = ScriptedFactory::new().then_output(json!("one")).then_err(
        pravah::clients::ClientError::new(pravah::clients::ErrorKind::Provider, "offline failure"),
    );
    let mut chat = builder()
        .store(store.clone())
        .compactor(Summarize)
        .build(context(&script)?)?;
    chat.send("first").await?;
    assert!(matches!(
        chat.send("second").await,
        Err(GraphError::AgentFailed { .. })
    ));
    let snapshot = chat.snapshot()?;
    assert_eq!(store.record_count(), 3);
    assert_eq!(snapshot.history().entries().len(), 2);
    assert!(
        snapshot.history().entries()[0]
            .message
            .content
            .contains("prior exchange")
    );
    let encoded = serde_json::to_value(&snapshot).map_err(codec)?;
    assert_eq!(encoded["state"]["persisted_history_position"], 3);
    let mut restored = builder().store(store).compactor(Summarize).restore::<()>(
        serde_json::from_value(encoded).map_err(codec)?,
        context(&script)?,
    )?;
    assert!(matches!(
        restored.next(),
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(script.calls().len(), 2);
    assert_eq!(
        serde_json::to_value(restored.snapshot()?).map_err(codec)?,
        serde_json::to_value(snapshot).map_err(codec)?
    );
    Ok(())
}

/// A new execution reloads accepted original messages once; restored loaded context is not fetched again.
#[tokio::test]
async fn keyed_load_survives_restore_without_reloading_or_duplicate_persistence()
-> Result<(), GraphError> {
    let store = Store {
        rows: CapturingHistoryStore::new(),
        loads: Arc::new(AtomicUsize::new(0)),
    };
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"));
    let mut first = builder().store(store.clone()).build(context(&script)?)?;
    first.send_with_key("first", "message-1").await?;
    let original = store.rows.all_entries();
    assert_eq!(original.len(), 2);
    let mut second = builder().store(store.clone()).build(context(&script)?)?;
    second.submit("second")?;
    let request = match second.next()? {
        pravah::ChatStep::Agent(request) => request,
        other => return Err(codec(format!("expected configure: {other:?}"))),
    };
    second.resume_agent(second.executor().execute(&request).await)?;
    let snapshot = second.snapshot()?;
    assert_eq!(snapshot.history().entries().len(), 2);
    assert_eq!(snapshot.history().entries()[0].id, original[0].id);
    assert_eq!(
        snapshot.history().entries()[0].message.key.as_deref(),
        Some("message-1")
    );
    let mut bytes = Vec::new();
    ciborium::into_writer(&snapshot, &mut bytes).map_err(codec)?;
    let snapshot = ciborium::from_reader(bytes.as_slice()).map_err(codec)?;
    let mut restored = builder()
        .store(store.clone())
        .restore::<()>(snapshot, context(&script)?)?;
    loop {
        match restored.next()? {
            pravah::ChatStep::Continue => {}
            pravah::ChatStep::Agent(request) => {
                restored.resume_agent(restored.executor().execute(&request).await)?
            }
            pravah::ChatStep::Done(turn) => {
                assert_eq!(turn.output, "two");
                break;
            }
            other => return Err(codec(format!("unexpected boundary: {other:?}"))),
        }
    }
    assert_eq!(store.loads.load(Ordering::SeqCst), 2);
    assert_eq!(store.rows.record_count(), 4);
    assert_eq!(script.calls()[1].1.len(), 3);
    Ok(())
}

/// Missing load acknowledgements and malformed loaded configurations are rejected atomically.
#[tokio::test]
async fn load_delivery_validates_configuration_and_completion_before_mutation()
-> Result<(), GraphError> {
    let store = Store {
        rows: CapturingHistoryStore::new(),
        loads: Arc::new(AtomicUsize::new(0)),
    };
    let mut history = pravah::MessageHistory::new();
    history.push("key:thread", "agent", Message::user("old"));
    history.push("key:thread", "agent", Message::assistant("answer"));
    for entry in history.entries() {
        store.record(entry).await.map_err(|never| match never {})?;
    }
    let script = ScriptedFactory::new();
    let mut chat = builder().store(store).build(context(&script)?)?;
    chat.submit("new")?;
    let request = match chat.next()? {
        pravah::ChatStep::Agent(request) => request,
        other => return Err(codec(format!("{other:?}"))),
    };
    let response = chat.executor().execute(&request).await;
    let valid = serde_json::to_value(&response).map_err(codec)?;
    let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    for mutation in 0..3 {
        let mut invalid = valid.clone();
        match mutation {
            0 => invalid["loaded"] = serde_json::Value::Null,
            1 => invalid["outcome"]["Ok"]["resolved"]["model"] = json!(""),
            _ => invalid["loaded"][1][0]["session_id"] = json!("key:other"),
        }
        let response: AgentResponse = serde_json::from_value(invalid).map_err(codec)?;
        assert!(chat.resume_agent(response).is_err());
        assert_eq!(
            serde_json::to_value(chat.snapshot()?).map_err(codec)?,
            before
        );
    }
    chat.resume_agent(response)?;
    assert_eq!(chat.snapshot()?.history().entries().len(), 2);
    assert!(script.calls().is_empty());
    Ok(())
}
