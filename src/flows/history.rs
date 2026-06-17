use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::clients::{ClientError, Message, Role, TokenUsage};
use crate::flows::compactor::CompactionResult;

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
                next_position = entry.position + 1;
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
        if let Some(u) = message.usage {
            self.last_usage = Some(u);
            self.total_input = add_opt(self.total_input, u.input);
            self.total_output = add_opt(self.total_output, u.output);
        }
        let mut entry = HistoryEntry::new(session_id, agent_id, message);
        entry.position = self.next_position;
        self.next_position += 1;
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

    pub fn last_role(&self) -> Option<&Role> {
        self.entries
            .iter()
            .rev()
            .find(|e| !e.evicted)
            .map(|e| &e.message.role)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|e| e.evicted)
    }

    pub fn last_usage(&self) -> Option<TokenUsage> {
        self.last_usage
    }

    pub fn total_input(&self) -> Option<u32> {
        self.total_input
    }

    pub fn total_output(&self) -> Option<u32> {
        self.total_output
    }

    /// Returns cumulative input and output tokens when both are known.
    pub fn total_usage(&self) -> Option<u32> {
        match (self.total_input, self.total_output) {
            (Some(i), Some(o)) => Some(i + o),
            _ => None,
        }
    }
}

fn add_opt(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::ToolCall;

    fn usage(input: u32, output: u32) -> TokenUsage {
        TokenUsage {
            input: Some(input),
            output: Some(output),
        }
    }

    fn push_assistant(h: &mut FlowHistory, session: &str, u: Option<TokenUsage>) {
        let mut msg = Message::assistant("hi");
        if let Some(u) = u {
            msg = msg.with_usage(u);
        }
        h.push(session, "agent", msg);
    }

    fn push_atc(h: &mut FlowHistory, session: &str, calls: Vec<ToolCall>) {
        h.push(
            session,
            "agent",
            Message {
                role: Role::AssistantToolCalls { calls },
                content: String::new(),
                attachments: Vec::new(),
                usage: None,
            },
        );
    }

    fn push_tool(h: &mut FlowHistory, session: &str, call_id: &str) {
        h.push(session, "agent", Message::tool_output(call_id.into(), "ok"));
    }

    fn push_user(h: &mut FlowHistory, session: &str, content: &str) {
        h.push(session, "agent", Message::user(content));
    }

    fn dummy_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "f".into(),
            args: serde_json::json!({}),
            thought_signatures: None,
        }
    }

    /// `push` records the most recently seen token usage.
    #[test]
    fn push_records_last_usage() {
        let mut h = FlowHistory::new();
        push_assistant(&mut h, "s1", Some(usage(10, 5)));
        let u = h.last_usage().unwrap();
        assert_eq!(u.input, Some(10));
        assert_eq!(u.output, Some(5));
    }

    /// `push` accumulates total input and output tokens across all messages.
    #[test]
    fn push_accumulates_totals() {
        let mut h = FlowHistory::new();
        push_assistant(&mut h, "s1", Some(usage(10, 5)));
        push_assistant(&mut h, "s1", Some(usage(20, 8)));
        assert_eq!(h.total_input(), Some(30));
        assert_eq!(h.total_output(), Some(13));
        assert_eq!(h.total_usage(), Some(43));
    }

    /// `for_session` returns only non-evicted messages for the exact session.
    #[test]
    fn for_session_excludes_other_sessions() {
        let mut h = FlowHistory::new();
        push_user(&mut h, "s1", "task");
        push_atc(&mut h, "s1", vec![dummy_call("1")]);
        push_tool(&mut h, "s1", "1");
        push_user(&mut h, "s2", "task");
        push_atc(&mut h, "s2", vec![dummy_call("2")]);
        let s1 = h.for_session("s1");
        assert_eq!(s1.len(), 3);
    }

    /// `validate_for_session` rejects a session whose last message is unresolved tool calls.
    #[test]
    fn validate_for_session_rejects_dangling() {
        let mut h = FlowHistory::new();
        push_atc(&mut h, "s1", vec![dummy_call("1")]);
        assert!(matches!(
            h.validate_for_session("s1"),
            Err(ClientError::Validation(_))
        ));
        assert!(h.validate_for_session("s2").is_ok());
    }

    /// `last_role` returns the role of the most recent non-evicted entry.
    #[test]
    fn last_role_returns_correct_role() {
        let mut h = FlowHistory::new();
        assert!(h.last_role().is_none());
        push_user(&mut h, "s1", "hi");
        assert!(matches!(h.last_role(), Some(Role::User)));
    }

    /// `total_usage` returns `None` when either input or output is missing.
    #[test]
    fn total_usage_requires_both_values() {
        let mut h = FlowHistory::new();
        h.push(
            "s1",
            "agent",
            Message {
                role: Role::Assistant,
                content: "x".into(),
                attachments: Vec::new(),
                usage: Some(TokenUsage {
                    input: Some(5),
                    output: None,
                }),
            },
        );
        assert_eq!(h.total_input(), Some(5));
        assert_eq!(h.total_output(), None);
        assert_eq!(h.total_usage(), None);
    }

    /// `apply_compaction` marks targeted entries evicted and inserts a summary.
    #[test]
    fn apply_compaction_marks_evicted_and_inserts_summary() {
        let mut h = FlowHistory::new();
        push_atc(&mut h, "s1", vec![dummy_call("1")]);
        push_tool(&mut h, "s1", "1");
        push_atc(&mut h, "s1", vec![dummy_call("2")]);
        push_tool(&mut h, "s1", "2");

        let owned: Vec<_> = h.session_entries("s1").into_iter().cloned().collect();
        let refs: Vec<_> = owned.iter().collect();
        let result = CompactionResult {
            evict_indices: vec![0, 1], // first turn
            summary: Some(Message::assistant("summary")),
        };
        h.apply_compaction("s1", &refs, result).unwrap();

        let active = h.for_session("s1");
        // summary + second turn (2 messages)
        assert_eq!(active.len(), 3);
        assert!(matches!(active[0].role, Role::Assistant));
    }

    /// `apply_compaction` returns an error for an out-of-bounds index.
    #[test]
    fn apply_compaction_rejects_out_of_bounds_index() {
        let mut h = FlowHistory::new();
        push_atc(&mut h, "s1", vec![dummy_call("1")]);
        let owned: Vec<_> = h.session_entries("s1").into_iter().cloned().collect();
        let refs: Vec<_> = owned.iter().collect();
        let result = CompactionResult {
            evict_indices: vec![99],
            summary: None,
        };
        assert!(h.apply_compaction("s1", &refs, result).is_err());
    }

    /// `prune_evicted` physically removes evicted entries.
    #[test]
    fn prune_evicted_removes_entries() {
        let mut h = FlowHistory::new();
        push_atc(&mut h, "s1", vec![dummy_call("1")]);
        push_tool(&mut h, "s1", "1");
        push_atc(&mut h, "s1", vec![dummy_call("2")]);
        push_tool(&mut h, "s1", "2");

        let owned: Vec<_> = h.session_entries("s1").into_iter().cloned().collect();
        let refs: Vec<_> = owned.iter().collect();
        let result = CompactionResult {
            evict_indices: vec![0, 1],
            summary: None,
        };
        h.apply_compaction("s1", &refs, result).unwrap();
        assert_eq!(h.entries().len(), 4); // still present, just evicted
        h.prune_evicted();
        assert_eq!(h.entries().len(), 2);
    }
}
