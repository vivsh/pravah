use crate::clients::{Message, Role};
use crate::history::{FlowHistory, HistoryReplacement, protected_start, validate_message_groups};

use super::support::*;

/// Builds two complete exchanges followed by one protected user input.
fn conversation() -> FlowHistory {
    let mut history = FlowHistory::new();
    push_user(&mut history, "s", "old");
    push_tool_calls(&mut history, "s", vec![tool_call("a"), tool_call("b")]);
    push_tool(&mut history, "s", "b");
    push_tool(&mut history, "s", "a");
    push_assistant(&mut history, "s", Some(usage(10, 5)));
    push_user(&mut history, "s", "retained");
    push_assistant(&mut history, "s", Some(usage(20, 8)));
    push_user(&mut history, "s", "pending");
    history
}

fn replace(
    history: &mut FlowHistory,
    decision: HistoryReplacement,
) -> Result<Vec<Message>, String> {
    let owned = history
        .session_entries("s")
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let refs = owned.iter().collect::<Vec<_>>();
    history.replace_prepared_history("s", &refs, decision)
}

/// Rejects malformed decisions and partial exchanges without changing any serialized field.
#[test]
fn invalid_replacements_are_atomic() {
    let cases = [
        (vec![99], None),
        (vec![7], None),
        (vec![0, 0], None),
        (vec![1, 0], None),
        (vec![0, 2], None),
        (vec![0], None),
        (vec![0, 1, 2], None),
        (vec![], Some("summary")),
        ((0..5).collect(), Some("  ")),
    ];
    for (evict_indices, summary) in cases {
        let mut history = conversation();
        let before = serde_json::to_value(&history).expect("encode history");
        let result = replace(
            &mut history,
            HistoryReplacement {
                evict_indices,
                summary: summary.map(str::to_owned),
            },
        );
        assert!(result.is_err());
        assert_eq!(
            serde_json::to_value(&history).expect("encode history"),
            before
        );
    }
}

/// A summary precedes retained exchanges while IDs, positions, usage and other sessions survive.
#[test]
fn replacement_preserves_accounting_identity_and_session_isolation() {
    let mut history = conversation();
    push_user(&mut history, "other", "untouched");
    let retained = history
        .entries()
        .iter()
        .skip(5)
        .cloned()
        .collect::<Vec<_>>();
    let next_position = history.entries().last().expect("last entry").position + 1;
    let messages = replace(
        &mut history,
        HistoryReplacement {
            evict_indices: (0..5).collect(),
            summary: Some("memory".into()),
        },
    )
    .expect("valid replacement");
    assert!(matches!(messages[0].role, Role::System));
    assert_eq!(messages[1].content, "retained");
    assert_eq!(history.entries().len(), 5);
    for (actual, expected) in history.entries().iter().skip(1).zip(retained) {
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.position, expected.position);
    }
    assert_eq!(history.total_input(), Some(30));
    assert_eq!(history.total_output(), Some(13));
    assert_eq!(history.last_usage().and_then(|u| u.input), Some(20));
    push_assistant(&mut history, "s", None);
    assert_eq!(
        history.entries().last().expect("last entry").position,
        next_position
    );
}

/// Stale policy observations cannot replace history appended while the policy was awaiting.
#[test]
fn stale_observations_are_rejected_without_mutation() {
    let mut history = conversation();
    let old = history
        .session_entries("s")
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    push_assistant(&mut history, "s", None);
    let before = serde_json::to_value(&history).expect("encode history");
    assert!(
        history
            .replace_prepared_history(
                "s",
                &old.iter().collect::<Vec<_>>(),
                HistoryReplacement::default()
            )
            .is_err()
    );
    assert_eq!(
        serde_json::to_value(&history).expect("encode history"),
        before
    );
}

/// Only the target session's old tombstones are removed after successful preparation.
#[test]
fn preparation_prunes_old_tombstones_only_in_its_session() {
    let mut history = conversation();
    push_user(&mut history, "other", "keep tombstone");
    let mut entries = history.entries().to_vec();
    entries[0].evicted = true;
    entries.last_mut().expect("last entry").evicted = true;
    let mut history = FlowHistory::from_entries(entries);
    let before = history.entries().len();
    replace(&mut history, HistoryReplacement::default()).expect("valid preparation");
    assert_eq!(history.entries().len(), before - 1);
    assert!(history.entries().last().expect("other session").evicted);
}

/// The ongoing user input and all of its tool messages are outside the replaceable prefix.
#[test]
fn current_tool_exchange_is_protected() {
    let mut history = conversation();
    push_tool_calls(&mut history, "s", vec![tool_call("current")]);
    push_tool(&mut history, "s", "current");
    let entries = history.session_entries("s");
    assert_eq!(protected_start(&entries), 7);
    assert_eq!(entries.len() - protected_start(&entries), 3);
    let mut no_user = FlowHistory::new();
    push_assistant(&mut no_user, "s", None);
    assert_eq!(protected_start(&no_user.session_entries("s")), 0);
}

/// Rejects missing, duplicate, orphan and interleaved results; accepts result reordering.
#[test]
fn tool_group_validation_is_complete() {
    for invalid_case in 0..5 {
        let mut history = FlowHistory::new();
        match invalid_case {
            0 => push_tool(&mut history, "s", "orphan"),
            1 => push_tool_calls(&mut history, "s", vec![tool_call("a"), tool_call("a")]),
            2 => {
                push_tool_calls(&mut history, "s", vec![tool_call("a")]);
                push_user(&mut history, "s", "interleaved");
            }
            3 => {
                push_tool_calls(&mut history, "s", vec![tool_call("a"), tool_call("b")]);
                push_tool(&mut history, "s", "a");
            }
            _ => {
                push_tool_calls(&mut history, "s", vec![tool_call("a")]);
                push_tool(&mut history, "s", "a");
                push_tool(&mut history, "s", "a");
            }
        }
        assert!(validate_message_groups(history.entries().iter().map(|e| &e.message)).is_err());
    }
    let history = conversation();
    assert!(validate_message_groups(history.entries().iter().map(|e| &e.message)).is_ok());
}
