use async_trait::async_trait;

use super::entries::HistoryEntry;
use crate::clients::{Message, Role};

/// Compaction decision for one session slice.
/// `evict_indices` are relative to the slice passed into `compact`.
pub struct CompactionResult {
    /// Positions (relative to the input slice) of entries to evict.
    pub evict_indices: Vec<usize>,
    /// Optional replacement summary message injected after eviction.
    pub summary: Option<Message>,
}

/// Chooses which history entries to evict for one session.
/// The runtime always passes entries from a single session.
pub trait HistoryCompactor: Send + Sync {
    /// Decides which entries to evict. `entries` are always from one session.
    fn compact(
        &self,
        session_id: &str,
        entries: &[&HistoryEntry],
    ) -> impl std::future::Future<Output = CompactionResult> + Send;
}

#[async_trait]
pub(crate) trait DynHistoryCompactor: Send + Sync {
    async fn compact_dyn(&self, session_id: &str, entries: &[&HistoryEntry]) -> CompactionResult;
}

#[async_trait]
impl<T: HistoryCompactor> DynHistoryCompactor for T {
    async fn compact_dyn(&self, session_id: &str, entries: &[&HistoryEntry]) -> CompactionResult {
        self.compact(session_id, entries).await
    }
}

/// Compactor that never evicts — suitable for short sessions or testing.
pub struct NoopCompactor;

impl HistoryCompactor for NoopCompactor {
    async fn compact(&self, _session_id: &str, _entries: &[&HistoryEntry]) -> CompactionResult {
        CompactionResult {
            evict_indices: vec![],
            summary: None,
        }
    }
}

/// Drops the oldest complete turns until the session fits the configured window.
/// Incomplete tool turns are never evicted.
pub struct SlidingWindowCompactor {
    /// Maximum number of complete assistant turns to retain per session.
    pub max_turns_per_session: usize,
}

impl HistoryCompactor for SlidingWindowCompactor {
    async fn compact(&self, _session_id: &str, entries: &[&HistoryEntry]) -> CompactionResult {
        let mut turn_count = count_complete_turns(entries);
        if turn_count <= self.max_turns_per_session {
            return CompactionResult {
                evict_indices: vec![],
                summary: None,
            };
        }

        let mut evict_indices: Vec<usize> = Vec::new();
        while turn_count > self.max_turns_per_session {
            match first_complete_turn_indices(entries, &evict_indices) {
                Some(indices) => {
                    evict_indices.extend_from_slice(&indices);
                    turn_count -= 1;
                }
                None => break,
            }
        }

        CompactionResult {
            evict_indices,
            summary: None,
        }
    }
}

/// Counts complete assistant turns in one session slice.
pub(crate) fn count_complete_turns(entries: &[&HistoryEntry]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < entries.len() {
        match &entries[i].message.role {
            Role::Assistant => {
                count += 1;
                i += 1;
            }
            Role::AssistantToolCalls { calls } => {
                let call_ids: std::collections::HashSet<&str> =
                    calls.iter().map(|c| c.id.as_str()).collect();
                let mut found_ids = std::collections::HashSet::new();
                for entry in entries.iter().skip(i + 1) {
                    if let Role::Tool { call_id } = &entry.message.role
                        && call_ids.contains(call_id.as_str())
                    {
                        found_ids.insert(call_id.as_str());
                    }
                }
                if found_ids.len() == call_ids.len() {
                    count += 1;
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    count
}

/// Returns the first complete turn not already marked for eviction.
fn first_complete_turn_indices(
    entries: &[&HistoryEntry],
    already_evicted: &[usize],
) -> Option<Vec<usize>> {
    for (i, entry) in entries.iter().enumerate() {
        if already_evicted.contains(&i) {
            continue;
        }
        match &entry.message.role {
            Role::Assistant => return Some(vec![i]),
            Role::AssistantToolCalls { calls } => {
                let call_ids: std::collections::HashSet<&str> =
                    calls.iter().map(|c| c.id.as_str()).collect();
                let mut found: Vec<usize> = Vec::new();
                let mut found_ids = std::collections::HashSet::new();
                for (j, e2) in entries.iter().enumerate().skip(i + 1) {
                    if already_evicted.contains(&j) {
                        continue;
                    }
                    if let Role::Tool { call_id } = &e2.message.role
                        && call_ids.contains(call_id.as_str())
                    {
                        found_ids.insert(call_id.as_str());
                        found.push(j);
                    }
                }
                if found_ids.len() == call_ids.len() {
                    let mut indices = vec![i];
                    indices.extend_from_slice(&found);
                    return Some(indices);
                }
            }
            _ => {}
        }
    }
    None
}
