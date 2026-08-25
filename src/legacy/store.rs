use async_trait::async_trait;

use crate::legacy::history::HistoryEntry;

/// Records appended flow history entries.
///
/// Snapshots own runtime history for restore. Stores are append sinks for
/// audit, export, or external persistence. Implementations must treat an
/// existing entry position as an idempotent replay so a failed batch can be
/// retried without duplicating durable rows.
pub trait HistoryStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Records one newly appended history entry.
    fn record(
        &self,
        entry: &HistoryEntry,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send;
}

#[async_trait]
pub(crate) trait DynHistoryStore: Send + Sync {
    async fn record_dyn(
        &self,
        entry: &HistoryEntry,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
impl<T: HistoryStore> DynHistoryStore for T {
    async fn record_dyn(
        &self,
        entry: &HistoryEntry,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.record(entry).await.map_err(|e| Box::new(e) as _)
    }
}

/// No-op store that accepts history entries without persisting them.
pub struct NoopHistoryStore;

impl HistoryStore for NoopHistoryStore {
    type Error = std::convert::Infallible;

    async fn record(&self, _entry: &HistoryEntry) -> Result<(), Self::Error> {
        Ok(())
    }
}
