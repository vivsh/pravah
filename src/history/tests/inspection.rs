use super::support::*;
use crate::clients::{ClientOptions, Message, Role};
use crate::history::{CompactionRequest, MessageHistory};

/// Creates interleaved sessions, a completed tool exchange and a pending user input.
fn conversation() -> MessageHistory {
    let mut history = MessageHistory::new();
    history.push("s", "__summary__", {
        let mut message =
            Message::user("<pravah_working_memory>\nsummary\n</pravah_working_memory>");
        message.role = Role::System;
        message
    });
    history.push(
        "s",
        "agent",
        Message::user("old é\n\"question\"").with_key("db:1"),
    );
    push_tool_calls(&mut history, "s", vec![tool_call("one")]);
    push_user(&mut history, "other", "private other session");
    push_tool(&mut history, "s", "one");
    push_assistant(&mut history, "s", Some(usage(3, 4)));
    push_user(&mut history, "s", "recent");
    push_assistant(&mut history, "s", None);
    push_user(&mut history, "s", "pending");
    history
}

/// Tool messages disappear without renumbering conversation indices or mixing sessions.
#[test]
fn enumeration_preserves_live_indices_and_borrows() {
    let history = conversation();
    let indices: Vec<_> = history
        .enum_messages("s", 0)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(indices, [1, 4, 5, 6, 7]);
    let selected: Vec<_> = history.enum_messages("s", 2).collect();
    assert_eq!(
        selected.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
        [1, 4, 5]
    );
    assert_eq!(selected[0].1.key.as_deref(), Some("db:1"));
    assert!(std::ptr::eq(selected[0].1, &history.entries()[1].message));
    assert_eq!(history.enum_messages("other", 0).count(), 1);
    assert_eq!(history.enum_messages("absent", 0).count(), 0);
    assert_eq!(history.enum_messages("s", usize::MAX).count(), 0);
}

/// Evicted rows are excluded before assigning session-relative indices and measuring history.
#[test]
fn inspection_ignores_evicted_entries() -> Result<(), serde_json::Error> {
    let mut entries = conversation().entries().to_vec();
    entries[0].evicted = true;
    let history = MessageHistory::from_entries(entries);
    let selected: Vec<_> = history.enum_messages("s", 0).collect();
    assert_eq!(selected[0].0, 0);
    assert_eq!(selected[0].1.key.as_deref(), Some("db:1"));
    let live = history.for_session("s");
    assert_eq!(history.byte_size("s")?, serde_json::to_vec(&live)?.len());
    Ok(())
}

/// Exact JSON size includes tool data, Unicode escaping, message keys, usage and attachments.
#[test]
fn byte_size_matches_json_without_buffer_allocation() -> Result<(), serde_json::Error> {
    let mut history = conversation();
    let attachment = serde_json::from_value(serde_json::json!({
        "role":{"role":"user"}, "content":"image", "attachments":[
            {"type":"inline","mime_type":"image/png","data":"AA=="},
            {"type":"file","mime_type":"text/plain","path":"not-opened.txt"}
        ]
    }))?;
    history.push("s", "agent", attachment);
    let expected = serde_json::to_vec(&history.for_session("s"))?.len();
    let mut actual = Ok(0);
    let allocations = allocation_counter::measure(|| {
        actual = history.byte_size("s");
    });
    assert_eq!(actual?, expected);
    assert_eq!(allocations.count_total, 0);
    assert_eq!(history.byte_size("absent")?, 2);
    Ok(())
}

/// User exchanges count once despite tool rounds, extra assistant messages and pending input.
#[test]
fn turns_are_completed_user_exchanges() {
    let mut history = conversation();
    assert_eq!(history.turn_count("s"), 2);
    assert_eq!(history.turn_count("other"), 0);
    assert_eq!(history.turn_count("missing"), 0);
    push_tool_calls(&mut history, "s", vec![tool_call("last")]);
    push_tool(&mut history, "s", "last");
    assert_eq!(history.turn_count("s"), 2);
    push_assistant(&mut history, "s", None);
    push_assistant(&mut history, "s", None);
    assert_eq!(history.turn_count("s"), 3);
    assert_eq!(history.total_usage(), Some(7));
}

/// Request helpers cover committed entries only, excluding protected input and framework guidance.
#[test]
fn request_helpers_are_scoped_to_committed_history() -> Result<(), serde_json::Error> {
    let history = conversation();
    let entries = history.session_entries("s");
    let (committed, protected) = entries.split_at(7);
    let options = ClientOptions::default();
    let request = CompactionRequest {
        session_id: "s",
        model: "test",
        options: &options,
        framework_messages: &[{
            let mut message = Message::user("conclude");
            message.role = Role::System;
            message
        }],
        committed,
        protected,
    };
    assert_eq!(
        request.enum_messages(0).map(|(i, _)| i).collect::<Vec<_>>(),
        [1, 4, 5, 6]
    );
    assert_eq!(
        request.enum_messages(2).map(|(i, _)| i).collect::<Vec<_>>(),
        [1, 4]
    );
    assert_eq!(request.turn_count(), 2);
    assert_eq!(request.summary(), Some("summary"));
    let messages: Vec<_> = committed.iter().map(|entry| &entry.message).collect();
    assert_eq!(request.byte_size()?, serde_json::to_vec(&messages)?.len());
    let allocations = allocation_counter::measure(|| {
        assert_eq!(request.enum_messages(1).count(), 3);
        assert_eq!(history.enum_messages("s", 1).count(), 4);
        assert_eq!(request.summary(), Some("summary"));
    });
    assert_eq!(allocations.count_total, 0);
    Ok(())
}

/// The Rust type rename preserves the serialized history layout and its original fields.
#[test]
fn history_layout_is_unchanged() -> Result<(), serde_json::Error> {
    let old = serde_json::json!({"entries":[], "next_position":0, "last_usage":null,
        "total_input":null, "total_output":null});
    let history: MessageHistory = serde_json::from_value(old.clone())?;
    assert_eq!(serde_json::to_value(history)?, old);
    Ok(())
}
