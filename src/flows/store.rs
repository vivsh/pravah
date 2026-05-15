use async_trait::async_trait;

use crate::flows::history::{FlowHistory, HistoryEntry};

/// Persists and loads flow conversation history.
///
/// Implementors own the storage backend (SQLite, Postgres, in-memory, etc.).
/// After a successful flush, evicted entries should be pruned via
/// [`FlowHistory::prune_evicted`] so they don't accumulate in memory.
pub trait HistoryStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn flush(
        &self,
        history: &mut FlowHistory,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send;

    fn load(
        &self,
        session_ids: &[&str],
    ) -> impl std::future::Future<Output = Result<Vec<HistoryEntry>, Self::Error>> + Send;
}

// ── dyn-safe internal erasure ──────────────────────────────────────────────

#[async_trait]
pub(crate) trait DynHistoryStore: Send + Sync {
    async fn flush_dyn(
        &self,
        history: &mut FlowHistory,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    async fn load_dyn(
        &self,
        session_ids: &[&str],
    ) -> Result<Vec<HistoryEntry>, Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait]
impl<T: HistoryStore> DynHistoryStore for T {
    async fn flush_dyn(
        &self,
        history: &mut FlowHistory,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.flush(history).await.map_err(|e| Box::new(e) as _)
    }

    async fn load_dyn(
        &self,
        session_ids: &[&str],
    ) -> Result<Vec<HistoryEntry>, Box<dyn std::error::Error + Send + Sync>> {
        self.load(session_ids).await.map_err(|e| Box::new(e) as _)
    }
}

// ── built-in implementations ───────────────────────────────────────────────

/// No-op store that immediately prunes evicted entries from memory.
///
/// Suitable for ephemeral flows that don't require persistence.
pub struct NoopHistoryStore;

impl HistoryStore for NoopHistoryStore {
    type Error = std::convert::Infallible;

    async fn flush(&self, history: &mut FlowHistory) -> Result<(), Self::Error> {
        history.prune_evicted();
        Ok(())
    }

    async fn load(&self, _session_ids: &[&str]) -> Result<Vec<HistoryEntry>, Self::Error> {
        Ok(vec![])
    }
}
