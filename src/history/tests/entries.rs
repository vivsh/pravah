use crate::clients::{ClientError, Message, Role, TokenUsage};
use crate::history::MessageHistory;
use crate::legacy::CompactionResult;

use super::support::{push_assistant, push_tool, push_tool_calls, push_user, tool_call, usage};

/// Verifies that appending an entry records its token usage as the latest usage.
#[test]
fn push_records_last_usage() {
    let mut history = MessageHistory::new();
    push_assistant(&mut history, "s1", Some(usage(10, 5)));
    assert_eq!(history.last_usage().and_then(|usage| usage.input), Some(10));
    assert_eq!(history.last_usage().and_then(|usage| usage.output), Some(5));
}

/// Verifies that token totals accumulate across appended history entries.
#[test]
fn push_accumulates_totals() {
    let mut history = MessageHistory::new();
    push_assistant(&mut history, "s1", Some(usage(10, 5)));
    push_assistant(&mut history, "s1", Some(usage(20, 8)));
    assert_eq!(history.total_input(), Some(30));
    assert_eq!(history.total_output(), Some(13));
    assert_eq!(history.total_usage(), Some(43));
}

/// Verifies that session queries return only live messages for the requested session.
#[test]
fn for_session_excludes_other_sessions() {
    let mut history = MessageHistory::new();
    push_user(&mut history, "s1", "task");
    push_tool_calls(&mut history, "s1", vec![tool_call("1")]);
    push_tool(&mut history, "s1", "1");
    push_user(&mut history, "s2", "task");
    push_tool_calls(&mut history, "s2", vec![tool_call("2")]);
    assert_eq!(history.for_session("s1").len(), 3);
}

/// Verifies that unresolved assistant tool calls make only their session invalid.
#[test]
fn validate_for_session_rejects_dangling_calls() {
    let mut history = MessageHistory::new();
    push_tool_calls(&mut history, "s1", vec![tool_call("1")]);
    assert!(matches!(
        history.validate_for_session("s1"),
        Err(ClientError::Validation(_))
    ));
    assert!(history.validate_for_session("s2").is_ok());
}

/// Verifies that the last role is taken from the newest live history entry.
#[test]
fn last_role_returns_latest_live_role() {
    let mut history = MessageHistory::new();
    assert!(history.last_role().is_none());
    push_user(&mut history, "s1", "hi");
    assert!(matches!(history.last_role(), Some(Role::User)));
}

/// Verifies that cumulative usage requires both input and output token totals.
#[test]
fn total_usage_requires_both_values() {
    let mut history = MessageHistory::new();
    history.push(
        "s1",
        "agent",
        Message {
            key: None,
            role: Role::Assistant,
            content: "x".into(),
            attachments: Vec::new(),
            usage: Some(TokenUsage {
                input: Some(5),
                output: None,
            }),
        },
    );
    assert_eq!(history.total_input(), Some(5));
    assert_eq!(history.total_output(), None);
    assert_eq!(history.total_usage(), None);
}

/// Verifies that compaction evicts selected entries and inserts a live summary.
#[test]
fn apply_compaction_marks_entries_and_inserts_summary() {
    let mut history = two_turn_history();
    let owned: Vec<_> = history.session_entries("s1").into_iter().cloned().collect();
    let refs: Vec<_> = owned.iter().collect();
    let result = CompactionResult {
        evict_indices: vec![0, 1],
        summary: Some(Message::assistant("summary")),
    };

    assert!(history.apply_compaction("s1", &refs, result).is_ok());
    let active = history.for_session("s1");
    assert_eq!(active.len(), 3);
    assert!(matches!(active[0].role, Role::Assistant));
}

/// Verifies that compaction rejects an index outside the supplied session slice.
#[test]
fn apply_compaction_rejects_out_of_bounds_index() {
    let mut history = MessageHistory::new();
    push_tool_calls(&mut history, "s1", vec![tool_call("1")]);
    let owned: Vec<_> = history.session_entries("s1").into_iter().cloned().collect();
    let refs: Vec<_> = owned.iter().collect();
    let result = CompactionResult {
        evict_indices: vec![99],
        summary: None,
    };
    assert!(history.apply_compaction("s1", &refs, result).is_err());
}

/// Verifies that pruning physically removes entries previously marked as evicted.
#[test]
fn prune_evicted_removes_marked_entries() {
    let mut history = two_turn_history();
    let owned: Vec<_> = history.session_entries("s1").into_iter().cloned().collect();
    let refs: Vec<_> = owned.iter().collect();
    let result = CompactionResult {
        evict_indices: vec![0, 1],
        summary: None,
    };

    assert!(history.apply_compaction("s1", &refs, result).is_ok());
    assert_eq!(history.entries().len(), 4);
    history.prune_evicted();
    assert_eq!(history.entries().len(), 2);
}

/// Builds two complete tool-call turns for compaction behavior tests.
fn two_turn_history() -> MessageHistory {
    let mut history = MessageHistory::new();
    push_tool_calls(&mut history, "s1", vec![tool_call("1")]);
    push_tool(&mut history, "s1", "1");
    push_tool_calls(&mut history, "s1", vec![tool_call("2")]);
    push_tool(&mut history, "s1", "2");
    history
}
