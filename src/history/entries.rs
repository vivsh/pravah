use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::compactor::CompactionResult;
use crate::clients::{ClientError, Message, Role, TokenUsage};

/// One history row with Pravah metadata around a wire-format [`Message`].
/// External code should create entries through [`FlowHistory::push`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Stable row id for persistence.
    pub id: Uuid,
    /// Monotonic position assigned by [`FlowHistory::push`].
    pub position: u64,
    /// Session this entry belongs to.
    pub session_id: String,
    /// Agent node that produced this entry.
    pub agent_id: String,
    /// Marks entries scheduled for pruning after a successful flush.
    pub evicted: bool,
    /// Provider-facing message payload.
    pub message: Message,
}

impl HistoryEntry {
    pub(crate) fn new(session_id: &str, agent_id: &str, message: Message) -> Self {
        Self {
            id: Uuid::now_v7(),
            position: 0,
            session_id: session_id.to_owned(),
            agent_id: agent_id.to_owned(),
            evicted: false,
            message,
        }
    }
}

/// Append-only history for all active agent sessions.
/// Token counters are updated on every [`push`](FlowHistory::push).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowHistory {
    entries: Vec<HistoryEntry>,
    next_position: u64,
    last_usage: Option<TokenUsage>,
    total_input: Option<u32>,
    total_output: Option<u32>,
}

impl FlowHistory {
    /// Creates an empty history with zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reconstructs a [`FlowHistory`] from a flat list of stored entries.
    ///
    /// Use this when loading row-per-entry data from a relational database.
    /// Token totals are recomputed only from the provided entries; if evicted
    /// rows were hard-deleted before loading, the totals will reflect only the
    /// surviving rows.
    pub fn from_entries(entries: Vec<HistoryEntry>) -> Self {
        let mut next_position: u64 = 0;
        let mut last_usage_pos: Option<u64> = None;
        let mut last_usage: Option<TokenUsage> = None;
        let mut total_input: Option<u32> = None;
        let mut total_output: Option<u32> = None;

        for entry in &entries {
            if entry.position >= next_position {
                next_position = entry.position.saturating_add(1);
            }
            if let Some(u) = entry.message.usage {
                if last_usage_pos.is_none_or(|p| entry.position > p) {
                    last_usage_pos = Some(entry.position);
                    last_usage = Some(u);
                }
                total_input = add_opt(total_input, u.input);
                total_output = add_opt(total_output, u.output);
            }
        }

        Self {
            entries,
            next_position,
            last_usage,
            total_input,
            total_output,
        }
    }

    /// Appends a new history entry.
    pub fn push(&mut self, session_id: &str, agent_id: &str, message: Message) {
        let entry = self.prepare_entry(session_id, agent_id, message);
        self.commit_entry(entry);
    }

    pub(crate) fn prepare_entry(
        &self,
        session_id: &str,
        agent_id: &str,
        message: Message,
    ) -> HistoryEntry {
        let mut entry = HistoryEntry::new(session_id, agent_id, message);
        entry.position = self.next_position;
        entry
    }

    pub(crate) fn commit_entry(&mut self, entry: HistoryEntry) {
        if let Some(u) = entry.message.usage {
            self.last_usage = Some(u);
            self.total_input = add_opt(self.total_input, u.input);
            self.total_output = add_opt(self.total_output, u.output);
        }
        self.next_position = self.next_position.max(entry.position.saturating_add(1));
        self.entries.push(entry);
    }

    /// Returns the live entries for one session.
    /// Pass this exact slice back to [`apply_compaction`](FlowHistory::apply_compaction).
    pub fn session_entries(&self, session_id: &str) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|e| !e.evicted && e.session_id == session_id)
            .collect()
    }

    /// Returns the live messages for one session.
    pub fn for_session(&self, session_id: &str) -> Vec<Message> {
        self.entries
            .iter()
            .filter(|e| !e.evicted && e.session_id == session_id)
            .map(|e| e.message.clone())
            .collect()
    }

    /// Rejects sessions that still end with unresolved tool calls.
    pub fn validate_for_session(&self, session_id: &str) -> Result<(), ClientError> {
        let last = self
            .entries
            .iter()
            .rev()
            .find(|e| !e.evicted && e.session_id == session_id);
        if matches!(
            last.map(|e| &e.message.role),
            Some(Role::AssistantToolCalls { .. })
        ) {
            return Err(ClientError::Validation(
                "history ends with assistant tool calls without tool results".into(),
            ));
        }
        Ok(())
    }

    /// Returns all entries, including evicted ones.
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Applies one compaction decision.
    /// `session_slice` must match the slice returned by [`session_entries`](FlowHistory::session_entries).
    /// Returns an error when a compactor reports an out-of-bounds index.
    pub fn apply_compaction(
        &mut self,
        session_id: &str,
        session_slice: &[&HistoryEntry],
        result: CompactionResult,
    ) -> Result<(), ClientError> {
        if result.evict_indices.is_empty() && result.summary.is_none() {
            return Ok(());
        }

        let mut evict_ids: Vec<Uuid> = Vec::with_capacity(result.evict_indices.len());
        let mut first_position: Option<u64> = None;
        for &rel_idx in &result.evict_indices {
            let entry = session_slice.get(rel_idx).ok_or_else(|| {
                ClientError::Validation(format!(
                    "compaction index {rel_idx} out of bounds for session '{session_id}'"
                ))
            })?;
            if first_position.is_none_or(|p| entry.position < p) {
                first_position = Some(entry.position);
            }
            evict_ids.push(entry.id);
        }

        for entry in &mut self.entries {
            if evict_ids.contains(&entry.id) {
                entry.evicted = true;
            }
        }

        if let Some(summary_message) = result.summary {
            let position = first_position.unwrap_or(self.next_position);
            let insert_at = self
                .entries
                .iter()
                .position(|e| evict_ids.contains(&e.id))
                .unwrap_or(self.entries.len());
            let summary_entry = HistoryEntry {
                id: Uuid::now_v7(),
                position,
                session_id: session_id.to_owned(),
                agent_id: "__summary__".to_owned(),
                evicted: false,
                message: summary_message,
            };
            self.entries.insert(insert_at, summary_entry);
        }

        Ok(())
    }

    /// Removes entries already marked as evicted.
    pub fn prune_evicted(&mut self) {
        self.entries.retain(|e| !e.evicted);
    }

    /// Returns the role of the newest live entry across all sessions.
    pub fn last_role(&self) -> Option<&Role> {
        self.entries
            .iter()
            .rev()
            .find(|e| !e.evicted)
            .map(|e| &e.message.role)
    }

    /// Returns whether the history contains no live entries.
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|e| e.evicted)
    }

    /// Returns the token usage recorded on the newest entry that supplied usage.
    pub fn last_usage(&self) -> Option<TokenUsage> {
        self.last_usage
    }

    /// Returns cumulative input tokens when any entry supplied that count.
    pub fn total_input(&self) -> Option<u32> {
        self.total_input
    }

    /// Returns cumulative output tokens when any entry supplied that count.
    pub fn total_output(&self) -> Option<u32> {
        self.total_output
    }

    /// Returns cumulative input and output tokens when both are known.
    pub fn total_usage(&self) -> Option<u32> {
        match (self.total_input, self.total_output) {
            (Some(i), Some(o)) => Some(i.saturating_add(o)),
            _ => None,
        }
    }
}

fn add_opt(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.saturating_add(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}
