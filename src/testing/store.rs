use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use crate::flows::{FlowHistory, HistoryEntry, HistoryStore};

/// A [`HistoryStore`] that captures a full snapshot of history entries on every flush.
///
/// Useful for asserting that the correct messages were recorded and that compaction
/// evicted the expected entries.
///
/// Clone before injecting to retain a spy handle:
///
/// ```rust,ignore
/// let store = CapturingHistoryStore::new();
/// let spy = store.clone();
/// let mut runtime = FlowRuntime::new(input)?.with_store(store);
/// // drive the flow …
/// assert_eq!(spy.flush_count(), 1);
/// let snapshot = spy.last_snapshot().unwrap();
/// assert_eq!(snapshot.iter().filter(|e| !e.evicted).count(), 4);
/// ```
#[derive(Clone)]
pub struct CapturingHistoryStore {
    snapshots: Arc<Mutex<Vec<Vec<HistoryEntry>>>>,
}

impl CapturingHistoryStore {
    /// Creates an empty store with no recorded snapshots.
    pub fn new() -> Self {
        Self {
            snapshots: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Number of times [`flush`](HistoryStore::flush) has been called.
    pub fn flush_count(&self) -> usize {
        self.snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Snapshot taken during the most recent flush, or `None` if flush has not run yet.
    ///
    /// Each entry carries its `evicted` flag so tests can verify compaction results.
    pub fn last_snapshot(&self) -> Option<Vec<HistoryEntry>> {
        self.snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
    }

    /// All snapshots in flush order, one per flush call.
    pub fn all_snapshots(&self) -> Vec<Vec<HistoryEntry>> {
        self.snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl Default for CapturingHistoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoryStore for CapturingHistoryStore {
    type Error = Infallible;

    async fn flush(&self, history: &mut FlowHistory) -> Result<(), Infallible> {
        let snapshot = history.entries().to_vec();
        self.snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(snapshot);
        history.prune_evicted();
        Ok(())
    }

    async fn load(&self, _session_ids: &[&str]) -> Result<Vec<HistoryEntry>, Infallible> {
        Ok(self
            .snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
            .unwrap_or_default())
    }
}
