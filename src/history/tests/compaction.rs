use crate::history::MessageHistory;
use crate::legacy::{HistoryCompactor, NoopCompactor, SlidingWindowCompactor};

use super::support::{push_assistant, push_tool, push_tool_calls, tool_call};

/// Verifies that the no-op compactor never selects history for eviction.
#[test]
fn noop_never_evicts() {
    let mut history = MessageHistory::new();
    push_tool_calls(&mut history, "s1", vec![tool_call("1")]);
    push_tool(&mut history, "s1", "1");
    let owned: Vec<_> = history.session_entries("s1").into_iter().cloned().collect();
    let refs: Vec<_> = owned.iter().collect();
    let result = futures::executor::block_on(NoopCompactor.compact("s1", &refs));
    assert!(result.evict_indices.is_empty());
}

/// Verifies that sliding-window compaction selects the oldest complete turn.
#[test]
fn sliding_window_evicts_oldest_turn() {
    let history = two_tool_turn_history();
    let owned: Vec<_> = history.session_entries("s1").into_iter().cloned().collect();
    let refs: Vec<_> = owned.iter().collect();
    let compactor = SlidingWindowCompactor {
        max_turns_per_session: 1,
    };
    let result = futures::executor::block_on(compactor.compact("s1", &refs));
    assert_eq!(result.evict_indices, vec![0, 1]);
}

/// Verifies that an incomplete tool turn is retained even with a zero-turn window.
#[test]
fn incomplete_tool_turn_is_not_evicted() {
    let mut history = MessageHistory::new();
    push_tool_calls(&mut history, "s1", vec![tool_call("a"), tool_call("b")]);
    push_tool(&mut history, "s1", "a");
    let owned: Vec<_> = history.session_entries("s1").into_iter().cloned().collect();
    let refs: Vec<_> = owned.iter().collect();
    let compactor = SlidingWindowCompactor {
        max_turns_per_session: 0,
    };
    let result = futures::executor::block_on(compactor.compact("s1", &refs));
    assert!(result.evict_indices.is_empty());
}

/// Verifies that plain assistant responses count as complete compactable turns.
#[test]
fn plain_assistant_turn_is_evicted() {
    let mut history = MessageHistory::new();
    push_assistant(&mut history, "s1", None);
    push_assistant(&mut history, "s1", None);
    let owned: Vec<_> = history.session_entries("s1").into_iter().cloned().collect();
    let refs: Vec<_> = owned.iter().collect();
    let compactor = SlidingWindowCompactor {
        max_turns_per_session: 1,
    };
    let result = futures::executor::block_on(compactor.compact("s1", &refs));
    assert_eq!(result.evict_indices, vec![0]);
}

/// Verifies that compaction decisions remain isolated to the supplied session.
#[test]
fn compactor_ignores_other_sessions() {
    let mut history = MessageHistory::new();
    push_tool_calls(&mut history, "s1", vec![tool_call("1")]);
    push_tool(&mut history, "s1", "1");
    push_tool_calls(&mut history, "s2", vec![tool_call("2")]);
    push_tool(&mut history, "s2", "2");
    push_tool_calls(&mut history, "s1", vec![tool_call("3")]);
    push_tool(&mut history, "s1", "3");
    let owned: Vec<_> = history.session_entries("s1").into_iter().cloned().collect();
    let refs: Vec<_> = owned.iter().collect();
    let compactor = SlidingWindowCompactor {
        max_turns_per_session: 1,
    };
    let result = futures::executor::block_on(compactor.compact("s1", &refs));
    assert_eq!(result.evict_indices, vec![0, 1]);
}

/// Builds two complete tool-call turns for compactor selection tests.
fn two_tool_turn_history() -> MessageHistory {
    let mut history = MessageHistory::new();
    push_tool_calls(&mut history, "s1", vec![tool_call("1")]);
    push_tool(&mut history, "s1", "1");
    push_tool_calls(&mut history, "s1", vec![tool_call("2")]);
    push_tool(&mut history, "s1", "2");
    history
}
