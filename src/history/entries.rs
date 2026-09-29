use crate::clients::ErrorKind;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::legacy_compactor;
use crate::clients::{ClientError, Message, Role, TokenUsage};

mod replacement;

pub(crate) use replacement::{
    ValidatedCompactionResult, prepare_compaction, protected_start, summary_uuid,
    validate_message_groups,
};

/// One history row with Pravah metadata around a wire-format [`Message`].
/// External code should create entries through [`MessageHistory::push`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Stable row id for persistence.
    pub id: Uuid,
    /// Append position; a working-memory summary inherits the first replaced position.
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

/// Runtime conversation history and cumulative usage for all agent sessions.
/// Preparation may replace completed exchanges; counters retain their original usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageHistory {
    entries: Vec<HistoryEntry>,
    next_position: u64,
    last_usage: Option<TokenUsage>,
    total_input: Option<u32>,
    total_output: Option<u32>,
}

impl MessageHistory {
    pub(crate) fn next_position(&self) -> u64 {
        self.next_position
    }

    /// Stages deterministic modern entries without mutating history or generating random IDs.
    pub(crate) fn stage_entries(
        &self,
        execution: Uuid,
        session: &str,
        agent: &str,
        messages: Vec<Message>,
    ) -> Result<Vec<HistoryEntry>, crate::graph::GraphError> {
        let count = u64::try_from(messages.len()).map_err(|_| {
            crate::graph::GraphError::HistoryPersistence("history batch too large".into())
        })?;
        self.next_position.checked_add(count).ok_or_else(|| {
            crate::graph::GraphError::HistoryPersistence("history positions exhausted".into())
        })?;
        Ok(messages
            .into_iter()
            .zip(self.next_position..)
            .map(|(message, position)| HistoryEntry {
                id: Uuid::new_v5(
                    &execution,
                    format!("pravah.history.v1:{position}").as_bytes(),
                ),
                position,
                session_id: session.into(),
                agent_id: agent.into(),
                evicted: false,
                message,
            })
            .collect())
    }
    /// Creates an empty history with zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrows live messages excluding framework summaries and tool calls/results, oldest-first.
    /// Skips the newest `skip_recent` exposed messages; indices refer to all live session entries.
    /// This selection does not authorize eviction of partial exchanges or tool groups.
    pub fn enum_messages<'a>(
        &'a self,
        session_id: &'a str,
        skip_recent: usize,
    ) -> impl Iterator<Item = (usize, &'a Message)> {
        super::inspection::enum_messages(self.live_entries(session_id), skip_recent)
    }

    /// Returns the compact JSON array size of all live session messages, including tool data.
    /// Includes keys, usage and attachment metadata/data, but not HistoryEntry metadata or file contents.
    /// This is not provider request size or token count; serialization/size overflow can fail.
    pub fn byte_size(&self, session_id: &str) -> Result<usize, serde_json::Error> {
        super::inspection::byte_size(self.live_messages(session_id))
    }

    /// Counts live user exchanges with a final assistant reply; tool rounds and summaries add no turns.
    /// Does not validate message groups or change cumulative provider usage counters.
    pub fn turn_count(&self, session_id: &str) -> usize {
        super::inspection::turn_count(self.live_messages(session_id))
    }

    fn live_messages<'a>(
        &'a self,
        session_id: &'a str,
    ) -> impl Iterator<Item = &'a Message> + Clone {
        self.live_entries(session_id).map(|entry| &entry.message)
    }

    fn live_entries<'a>(
        &'a self,
        session_id: &'a str,
    ) -> impl Iterator<Item = &'a HistoryEntry> + Clone {
        self.entries
            .iter()
            .filter(move |entry| !entry.evicted && entry.session_id == session_id)
    }

    /// Reconstructs a [`MessageHistory`] from a flat list of stored entries.
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
    /// Pass this exact slice back to [`apply_compaction`](MessageHistory::apply_compaction).
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
            return Err(ClientError::new(
                ErrorKind::Validation,
                "history ends with assistant tool calls without tool results",
            ));
        }
        Ok(())
    }

    /// Returns all entries, including evicted ones.
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Applies one compaction decision.
    /// `session_slice` must match the slice returned by [`session_entries`](MessageHistory::session_entries).
    /// Returns an error when a compactor reports an out-of-bounds index.
    pub fn apply_compaction(
        &mut self,
        session_id: &str,
        session_slice: &[&HistoryEntry],
        result: legacy_compactor::CompactionResult,
    ) -> Result<(), ClientError> {
        if result.evict_indices.is_empty() && result.summary.is_none() {
            return Ok(());
        }

        let mut evict_ids: Vec<Uuid> = Vec::with_capacity(result.evict_indices.len());
        let mut first_position: Option<u64> = None;
        for &rel_idx in &result.evict_indices {
            let entry = session_slice.get(rel_idx).ok_or_else(|| {
                ClientError::new(
                    ErrorKind::Validation,
                    format!("compaction index {rel_idx} out of bounds for session '{session_id}'"),
                )
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
                agent_id: super::inspection::SUMMARY_AGENT_ID.to_owned(),
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
