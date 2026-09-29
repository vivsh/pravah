use super::*;
use pravah::clients::Role;
use pravah::testing::CapturingHistoryStore;
use pravah::{ChatBuilder, CompactionRequest, CompactionResult, Compactor, Snapshot};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const MODEL: &str = "openai:///recorded-model";
const SUMMARY: &str = "The first question was answered.\n京都";

fn definition() -> ChatBuilder<String, String> {
    Chat::builder::<String, String>()
        .model(MODEL)
        .instructions("Answer briefly.")
}

struct Summarize(Arc<AtomicUsize>);

impl Compactor for Summarize {
    type Error = std::convert::Infallible;

    /// Replaces only completed exchanges, preserving the current keyed submission.
    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        assert_eq!(request.model(), MODEL);
        assert_eq!(request.protected().len(), 1);
        assert!(
            request
                .protected()
                .first()
                .is_some_and(|e| e.message.key.is_some())
        );
        if request.committed().is_empty() {
            return Ok(CompactionResult::default());
        }
        if request.summary().is_some() {
            assert_eq!(request.summary(), Some(SUMMARY));
        }
        Ok(CompactionResult {
            evict_indices: (0..request.committed().len()).collect(),
            summary: Some(SUMMARY.into()),
        })
    }
}

/// Ordinary production-model Chat submissions can be rebuilt offline and restored unchanged.
#[tokio::test]
async fn recorded_builtin_chat_restores_with_production_context() -> Result<(), TestError> {
    let script = ScriptedFactory::new()
        .then_output(json!("first"))
        .then_output(json!("second"));
    let preparations = Arc::new(AtomicUsize::new(0));
    let store = CapturingHistoryStore::new();
    let ctx =
        Context::default().with_providers(ProviderRegistry::with_builtin_factory(script.clone()));
    let mut chat = definition()
        .state(7u32)
        .store(store.clone())
        .compactor(Summarize(preparations.clone()))
        .build(ctx)?;
    assert!(script.calls().is_empty());
    restore_production(chat.snapshot()?)?;
    for (input, key, output) in [
        ("question one", "message-1", "first"),
        ("question two", "message-2", "second"),
    ] {
        assert_eq!(chat.send_with_key(input, key).await?.output, output);
    }
    assert_eq!(preparations.load(Ordering::SeqCst), 2);
    assert_eq!(script.remaining(), 0);
    assert_rebuilt_history(&chat.snapshot()?, &script, &store)?;
    restore_production(chat.snapshot()?)?;
    assert_eq!(preparations.load(Ordering::SeqCst), 2);
    assert_eq!(script.calls().len(), 2);
    Ok(())
}

/// Restore rebinds dependencies without configuration, preparation, or production dispatch.
fn restore_production(snapshot: Snapshot) -> Result<(), TestError> {
    let before = serde_json::to_value(&snapshot)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor)?;
    let copies = [
        serde_json::from_value(before.clone())?,
        ciborium::from_reader(cbor.as_slice())?,
    ];
    for snapshot in copies {
        let preparations = Arc::new(AtomicUsize::new(0));
        let chat = definition()
            .compactor(Summarize(preparations.clone()))
            .restore::<u32>(snapshot, Context::default())?;
        assert_eq!(chat.get()?, 7);
        assert_eq!(before, serde_json::to_value(chat.snapshot()?)?);
        assert_eq!(preparations.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

/// Summary replacement retains stable live/store identities and the exact production model.
fn assert_rebuilt_history(
    snapshot: &Snapshot,
    script: &ScriptedFactory,
    store: &CapturingHistoryStore,
) -> Result<(), TestError> {
    let entries = snapshot.history().entries();
    let [summary, user, assistant] = entries else {
        return Err(TestError::Missing("compacted exchange"));
    };
    assert!(matches!(summary.message.role, Role::System));
    assert_eq!(
        summary.message.content,
        format!("<pravah_working_memory>\n{SUMMARY}\n</pravah_working_memory>")
    );
    assert_eq!(summary.message.key, None);
    assert_eq!(user.message.key.as_deref(), Some("message-2"));
    assert!(matches!(assistant.message.role, Role::Assistant));
    assert_stored_entries(store, [user, assistant])?;
    let calls = script.calls();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|(model, _)| model == MODEL));
    let (_, messages) = calls
        .last()
        .ok_or(TestError::Missing("recorded dispatch"))?;
    assert_eq!(messages.len(), 2);
    assert_eq!(
        serde_json::to_value(messages.first())?,
        serde_json::to_value(Some(&summary.message))?
    );
    assert_eq!(
        serde_json::to_value(messages.last())?,
        serde_json::to_value(Some(&user.message))?
    );
    Ok(())
}

/// Persisted rows retain identity and position even when older runtime rows are compacted away.
fn assert_stored_entries(
    store: &CapturingHistoryStore,
    live: [&pravah::history::HistoryEntry; 2],
) -> Result<(), TestError> {
    let stored = store.all_entries();
    for entry in live {
        let delivered = stored
            .iter()
            .find(|record| record.id == entry.id)
            .ok_or(TestError::Missing("stored entry"))?;
        assert_eq!(
            serde_json::to_value(delivered)?,
            serde_json::to_value(entry)?
        );
    }
    assert!(
        stored
            .iter()
            .any(|entry| entry.message.key.as_deref() == Some("message-1"))
    );
    Ok(())
}
