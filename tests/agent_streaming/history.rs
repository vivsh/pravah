use super::chat::builder;
use super::*;
use pravah::{CompactionRequest, CompactionResult, Compactor, HistoryEntry, HistoryStore};
use std::sync::atomic::Ordering;

struct Store {
    stats: Arc<Stats>,
    fail: bool,
}

impl HistoryStore for Store {
    type Error = GraphError;
    async fn load(&self, _: &str) -> Result<Vec<HistoryEntry>, Self::Error> {
        Ok(Vec::new())
    }
    async fn record(&self, _: &HistoryEntry) -> Result<(), Self::Error> {
        if self.fail {
            return Err(GraphError::HistoryPersistence("fixture rejection".into()));
        }
        self.stats.records.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct Consolidate(Arc<Stats>);

struct RejectPreparation;

impl Compactor for RejectPreparation {
    type Error = GraphError;
    async fn compact(
        &self,
        _: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        Err(GraphError::Invalid("fixture preparation failure".into()))
    }
}

impl Compactor for Consolidate {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        self.0.compactions.fetch_add(1, Ordering::SeqCst);
        assert!(!request.protected().is_empty());
        if request.committed().is_empty() {
            return Ok(CompactionResult::default());
        }
        Ok(CompactionResult {
            evict_indices: (0..request.committed().len()).collect(),
            summary: Some("consolidated memory".into()),
        })
    }
}

/// Streaming prepares exactly once per generation and persists authoritative user/assistant rows.
#[tokio::test]
async fn compaction_and_persistence_keep_existing_order() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let mut chat = builder()
        .store(Store {
            stats: stats.clone(),
            fail: false,
        })
        .compactor(Consolidate(stats.clone()))
        .build(context(Mode::Complete, stats.clone())?)?;
    for key in ["first", "second"] {
        chat.send_stream_with_key("question", key, |_, _| std::future::ready(()))
            .await?;
    }
    assert_eq!(stats.starts.load(Ordering::SeqCst), 2);
    assert_eq!(stats.compactions.load(Ordering::SeqCst), 2);
    assert_eq!(stats.summaries_seen.load(Ordering::SeqCst), 1);
    assert_eq!(stats.records.load(Ordering::SeqCst), 4);
    let snapshot = chat.snapshot()?;
    assert_eq!(snapshot.history().entries().len(), 3);
    assert_eq!(
        snapshot.history().entries()[1].message.key.as_deref(),
        Some("second")
    );
    assert_eq!(snapshot.history().total_input(), Some(20));
    Ok(())
}

/// A failed stream retains earlier store acknowledgements and is never regenerated on restore.
#[tokio::test]
async fn failed_stream_keeps_persisted_cursor_across_restore() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let definition = || {
        builder().store(Store {
            stats: stats.clone(),
            fail: false,
        })
    };
    let mut chat = definition().build(context(Mode::FailureAfterProgress, stats.clone())?)?;
    assert!(matches!(
        chat.send_stream("question", |_, _| std::future::ready(()))
            .await,
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(stats.records.load(Ordering::SeqCst), 1);
    let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    let mut restored =
        definition().restore::<()>(chat.snapshot()?, context(Mode::Complete, stats.clone())?)?;
    assert!(matches!(
        restored.next(),
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(
        before,
        serde_json::to_value(restored.snapshot()?).map_err(codec)?
    );
    assert_eq!(stats.starts.load(Ordering::SeqCst), 1);
    assert_eq!(stats.records.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Store failure prevents both streaming startup and compaction; no previews can escape.
#[tokio::test]
async fn persistence_failure_prevents_generation() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let mut chat = builder()
        .store(Store {
            stats: stats.clone(),
            fail: true,
        })
        .compactor(Consolidate(stats.clone()))
        .build(context(Mode::Complete, stats.clone())?)?;
    let mut previews = 0;
    assert!(
        chat.send_stream("question", |_, _| {
            previews += 1;
            std::future::ready(())
        })
        .await
        .is_err()
    );
    assert_eq!(previews, 0);
    assert_eq!(stats.starts.load(Ordering::SeqCst), 0);
    assert_eq!(stats.compactions.load(Ordering::SeqCst), 0);
    Ok(())
}

/// A fallible preparation failure cannot start a stream or replace the accepted input history.
#[tokio::test]
async fn compaction_failure_prevents_generation() -> Result<(), GraphError> {
    let stats = Arc::new(Stats::default());
    let mut chat = builder()
        .compactor(RejectPreparation)
        .build(context(Mode::Complete, stats.clone())?)?;
    let mut previews = 0;
    assert!(matches!(
        chat.send_stream("question", |_, _| {
            previews += 1;
            std::future::ready(())
        })
        .await,
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(stats.starts.load(Ordering::SeqCst), 0);
    assert_eq!(previews, 0);
    let snapshot = chat.snapshot()?;
    assert_eq!(snapshot.history().entries().len(), 1);
    assert_eq!(
        snapshot.history().entries()[0].message.content,
        "\"question\""
    );
    Ok(())
}
