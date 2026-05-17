use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use crate::flows::{FlowHistory, HistoryEntry, HistoryStore};

/// [`HistoryStore`] test double that records a full snapshot on every flush.
#[derive(Clone)]
pub struct CapturingHistoryStore {
    snapshots: Arc<Mutex<Vec<Vec<HistoryEntry>>>>,
}

impl CapturingHistoryStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            snapshots: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns how many times `flush` has run.
    pub fn flush_count(&self) -> usize {
        self.snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Returns the most recent flush snapshot.
    pub fn last_snapshot(&self) -> Option<Vec<HistoryEntry>> {
        self.snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
    }

    /// Returns all snapshots in flush order.
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
