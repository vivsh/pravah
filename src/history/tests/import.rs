use super::*;
use crate::clients::Message;

/// Imports completed exchanges without reassigning their identities, positions or usage.
#[test]
fn completed_history_and_summary_only_sessions_are_valid() -> Result<(), GraphError> {
    let mut history = MessageHistory::new();
    history.push("key:first", "old", Message::user("question"));
    history.push("key:first", "old", Message::assistant("answer"));
    history.push(
        "key:second",
        SUMMARY_AGENT_ID,
        Message::new(
            Role::System,
            "<pravah_working_memory>\nremember\n</pravah_working_memory>".into(),
        ),
    );
    history.validate_import()?;
    assert!(!history.session_is_busy("key:first"));
    assert!(!history.session_is_busy("key:second"));
    Ok(())
}

/// Import rejects malformed identities, ordering, incomplete exchanges and summary metadata.
#[test]
fn malformed_imports_are_rejected() {
    let mut history = MessageHistory::new();
    history.push("key:first", "old", Message::user("question"));
    history.push("key:first", "old", Message::assistant("answer"));
    for case in 0..8 {
        let mut rows = history.entries().to_vec();
        match case {
            0 => rows[0].evicted = true,
            1 => rows[0].session_id.clear(),
            2 => rows[1].id = rows[0].id,
            3 => rows.reverse(),
            4 => rows[1].position = u64::MAX,
            5 => rows[1].message = Message::user("unfinished"),
            6 => rows[0].agent_id = SUMMARY_AGENT_ID.into(),
            _ => rows[1].agent_id = SUMMARY_AGENT_ID.into(),
        }
        assert!(matches!(
            MessageHistory::from_entries(rows).validate_import(),
            Err(GraphError::HistoryValidation(_))
        ));
    }
}
