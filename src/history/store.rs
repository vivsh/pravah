use async_trait::async_trait;

use super::entries::HistoryEntry;

/// Records accepted conversation entries and retrieves conversations by application key.
///
/// Snapshots own runtime history for restore. Stores are append sinks for
/// audit, export, external persistence and retrieval. Implementations must treat an
/// existing entry ID as an idempotent replay so a failed batch or restoration
/// can redeliver without duplicating durable rows. Positions are scoped to history,
/// not globally unique. Original rows are saved before working-history pruning;
/// generated compaction summaries are not appended to this audit sink.
pub trait HistoryStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Records one newly appended history entry.
    fn record(
        &self,
        entry: &HistoryEntry,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send;

    /// Loads accepted entries for an application conversation key in conversation order.
    /// Preserve entry identities and complete tool groups; an empty result means no history.
    /// Store ordering must span executions: VM positions alone are not a global sort key.
    /// Automatic keyed loading assigns fresh execution-local positions without rewriting stored rows.
    /// This retrieves context for a fresh invocation, not a resumable VM checkpoint.
    fn load(
        &self,
        key: &str,
    ) -> impl std::future::Future<Output = Result<Vec<HistoryEntry>, Self::Error>> + Send;
}

#[async_trait]
pub(crate) trait DynHistoryStore: Send + Sync {
    async fn load_dyn(
        &self,
        key: &str,
    ) -> Result<Vec<HistoryEntry>, Box<dyn std::error::Error + Send + Sync>>;
    async fn record_dyn(
        &self,
        entry: &HistoryEntry,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
impl<T: HistoryStore> DynHistoryStore for T {
    async fn load_dyn(
        &self,
        key: &str,
    ) -> Result<Vec<HistoryEntry>, Box<dyn std::error::Error + Send + Sync>> {
        self.load(key).await.map_err(|error| Box::new(error) as _)
    }
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

    async fn load(&self, _key: &str) -> Result<Vec<HistoryEntry>, Self::Error> {
        Ok(Vec::new())
    }

    async fn record(&self, _entry: &HistoryEntry) -> Result<(), Self::Error> {
        Ok(())
    }
}
