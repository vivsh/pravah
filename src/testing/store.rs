use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use crate::legacy::{HistoryEntry, HistoryStore};

/// [`HistoryStore`] test double that records every appended history entry.
#[derive(Clone)]
pub struct CapturingHistoryStore {
    entries: Arc<Mutex<Vec<HistoryEntry>>>,
}

impl CapturingHistoryStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns how many entries have been recorded.
    pub fn record_count(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Returns the most recent recorded entry.
    pub fn last_entry(&self) -> Option<HistoryEntry> {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
    }

    /// Returns all recorded entries in append order.
    pub fn all_entries(&self) -> Vec<HistoryEntry> {
        self.entries
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

    async fn record(&self, entry: &HistoryEntry) -> Result<(), Infallible> {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(entry.clone());
        Ok(())
    }
}
