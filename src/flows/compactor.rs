use async_trait::async_trait;

use crate::clients::{Message, Role};
use crate::flows::history::HistoryEntry;

/// Result returned by a [`HistoryCompactor`].
///
/// `evict_indices` are 0-based indices into the `session_entries` slice that was passed
/// to [`HistoryCompactor::compact`]. `summary` is an optional replacement message that
/// [`FlowHistory::apply_compaction`] inserts at the eviction site.
pub struct CompactionResult {
    pub evict_indices: Vec<usize>,
    pub summary: Option<Message>,
}

/// Decides which history entries for a single session should be evicted.
///
/// # Session isolation
///
/// The `entries` slice contains only one session's non-evicted entries.
/// [`FlowRuntime`](crate::flows::FlowRuntime) drives a per-session loop, so structural
/// contamination across sessions is impossible.
pub trait HistoryCompactor: Send + Sync {
    fn compact(
        &self,
        session_id: &str,
        entries: &[&HistoryEntry],
    ) -> impl std::future::Future<Output = CompactionResult> + Send;
}

// ── dyn-safe internal erasure ──────────────────────────────────────────────

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

// ── built-in implementations ───────────────────────────────────────────────

/// Compactor that never evicts anything.
pub struct NoopCompactor;

impl HistoryCompactor for NoopCompactor {
    async fn compact(&self, _session_id: &str, _entries: &[&HistoryEntry]) -> CompactionResult {
        CompactionResult {
            evict_indices: vec![],
            summary: None,
        }
    }
}

/// Evicts the oldest complete assistant turns until the session is within `max_turns`.
///
/// A turn is either:
/// - A single `Role::Assistant` message, or
/// - A `Role::AssistantToolCalls` message **plus** all its matching `Role::Tool` results.
///
/// Incomplete turns (missing some tool results) are never evicted.
pub struct SlidingWindowCompactor {
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

/// Counts complete assistant turns in a session slice.
///
/// An `AssistantToolCalls` turn is complete only when every expected tool result is present.
fn count_complete_turns(entries: &[&HistoryEntry]) -> usize {
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
                for j in (i + 1)..entries.len() {
                    if let Role::Tool { call_id } = &entries[j].message.role {
                        if call_ids.contains(call_id.as_str()) {
                            found_ids.insert(call_id.as_str());
                        }
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

/// Returns the relative indices (into `entries`) of the first complete turn that has not
/// already been included in `already_evicted`.
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
                    if let Role::Tool { call_id } = &e2.message.role {
                        if call_ids.contains(call_id.as_str()) {
                            found_ids.insert(call_id.as_str());
                            found.push(j);
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::{ToolCall};
    use crate::flows::history::FlowHistory;

    fn dummy_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "f".into(),
            args: serde_json::json!({}),
            thought_signatures: None,
        }
    }

    fn push_atc(h: &mut FlowHistory, session: &str, calls: Vec<ToolCall>) {
        h.push(
            session,
            "agent",
            Message {
                role: Role::AssistantToolCalls { calls },
                content: String::new(),
                usage: None,
            },
        );
    }

    fn push_tool(h: &mut FlowHistory, session: &str, call_id: &str) {
        h.push(session, "agent", Message::tool_output(call_id.into(), "ok"));
    }

    fn push_assistant(h: &mut FlowHistory, session: &str) {
        h.push(session, "agent", Message::assistant("answer"));
    }

    /// Noop compactor never evicts.
    #[test]
    fn noop_never_evicts() {
        let mut h = FlowHistory::new();
        push_atc(&mut h, "s1", vec![dummy_call("1")]);
        push_tool(&mut h, "s1", "1");
        let owned: Vec<_> = h.session_entries("s1").into_iter().cloned().collect();
        let refs: Vec<_> = owned.iter().collect();
        let result = futures::executor::block_on(NoopCompactor.compact("s1", &refs));
        assert!(result.evict_indices.is_empty());
    }

    /// SlidingWindowCompactor evicts the oldest complete turn.
    #[test]
    fn sliding_evicts_oldest_turn() {
        let mut h = FlowHistory::new();
        push_atc(&mut h, "s1", vec![dummy_call("1")]);
        push_tool(&mut h, "s1", "1");
        push_atc(&mut h, "s1", vec![dummy_call("2")]);
        push_tool(&mut h, "s1", "2");
        let owned: Vec<_> = h.session_entries("s1").into_iter().cloned().collect();
        let refs: Vec<_> = owned.iter().collect();
        let compactor = SlidingWindowCompactor { max_turns_per_session: 1 };
        let result = futures::executor::block_on(compactor.compact("s1", &refs));
        assert_eq!(result.evict_indices, vec![0, 1]);
    }

    /// Incomplete turn (missing tool result) is never evicted.
    #[test]
    fn incomplete_tool_turn_not_evicted() {
        let mut h = FlowHistory::new();
        push_atc(&mut h, "s1", vec![dummy_call("a"), dummy_call("b")]);
        push_tool(&mut h, "s1", "a"); // only one of two results
        let owned: Vec<_> = h.session_entries("s1").into_iter().cloned().collect();
        let refs: Vec<_> = owned.iter().collect();
        let compactor = SlidingWindowCompactor { max_turns_per_session: 0 };
        let result = futures::executor::block_on(compactor.compact("s1", &refs));
        assert!(result.evict_indices.is_empty());
    }

    /// Plain assistant turn (no tool calls) counts as one turn and can be evicted.
    #[test]
    fn plain_assistant_turn_evicted() {
        let mut h = FlowHistory::new();
        push_assistant(&mut h, "s1");
        push_assistant(&mut h, "s1");
        let owned: Vec<_> = h.session_entries("s1").into_iter().cloned().collect();
        let refs: Vec<_> = owned.iter().collect();
        let compactor = SlidingWindowCompactor { max_turns_per_session: 1 };
        let result = futures::executor::block_on(compactor.compact("s1", &refs));
        assert_eq!(result.evict_indices, vec![0]);
    }

    /// Session isolation: compactor only sees one session's slice.
    #[test]
    fn compactor_ignores_other_sessions() {
        let mut h = FlowHistory::new();
        push_atc(&mut h, "s1", vec![dummy_call("1")]);
        push_tool(&mut h, "s1", "1");
        push_atc(&mut h, "s2", vec![dummy_call("2")]);
        push_tool(&mut h, "s2", "2");
        push_atc(&mut h, "s1", vec![dummy_call("3")]);
        push_tool(&mut h, "s1", "3");
        let owned: Vec<_> = h.session_entries("s1").into_iter().cloned().collect();
        let refs: Vec<_> = owned.iter().collect();
        let compactor = SlidingWindowCompactor { max_turns_per_session: 1 };
        let result = futures::executor::block_on(compactor.compact("s1", &refs));
        // Only first s1 turn evicted
        assert_eq!(result.evict_indices, vec![0, 1]);
    }
}
