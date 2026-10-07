//! Store retrieval is independent of graph construction and VM ownership.

use pravah::clients::Message;
use pravah::testing::CapturingHistoryStore;
use pravah::{GraphError, HistoryStore, MessageHistory};

/// A worker can retrieve one conversation by its application key without a parent runtime.
#[tokio::test]
async fn store_load_is_keyed_ordered_and_idempotent() -> Result<(), GraphError> {
    let store = CapturingHistoryStore::new();
    let mut history = MessageHistory::new();
    history.push("key:first", "assistant", Message::user("one"));
    history.push("key:other", "assistant", Message::user("unrelated"));
    history.push("key:first", "assistant", Message::assistant("two"));
    for entry in history.entries() {
        store.record(entry).await.map_err(|never| match never {})?;
        store.record(entry).await.map_err(|never| match never {})?;
    }
    let entries = store.load("first").await.map_err(|never| match never {})?;
    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries.first().map(|entry| entry.id),
        history.entries().first().map(|entry| entry.id)
    );
    assert_eq!(
        entries.last().map(|entry| entry.id),
        history.entries().last().map(|entry| entry.id)
    );
    assert!(entries.iter().all(|entry| entry.session_id == "key:first"));
    assert!(
        store
            .load("missing")
            .await
            .map_err(|never| match never {})?
            .is_empty()
    );
    Ok(())
}
