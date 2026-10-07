use super::*;

/// Keys keep exact text, occupy a separate namespace, and cannot enter an open exchange.
#[test]
fn session_selection_preserves_keys_and_rejects_overlap() -> Result<(), GraphError> {
    let mut history = MessageHistory::new();
    let key = " reviewer / 用户 ";
    let session = select_session(Some(key.into()), Uuid::nil(), &history)?;
    assert_eq!(session, format!("key:{key}"));
    assert_ne!(select_session(None, Uuid::nil(), &history)?, session);
    history.push(&session, "parent", Message::user("question"));
    assert!(matches!(
        select_session(Some(key.into()), Uuid::nil(), &history),
        Err(GraphError::AgentConversationBusy)
    ));
    history.push(&session, "parent", Message::assistant("answer"));
    assert_eq!(
        select_session(Some(key.into()), Uuid::nil(), &history)?,
        session
    );
    Ok(())
}

/// Whitespace-only keys and obsolete node-local state are rejected without modifying history.
#[test]
fn empty_keys_and_saved_state_are_invalid() {
    for key in ["", " ", "\n\t"] {
        assert!(matches!(
            select_session(Some(key.into()), Uuid::nil(), &MessageHistory::new()),
            Err(GraphError::AgentConfigValidation(_))
        ));
    }
    assert!(validate_empty_state(Some(&Value::NULL)).is_err());
    assert!(validate_empty_state(None).is_ok());
}

/// Prior checkpoint versions cannot silently restore old effect layouts.
#[test]
fn obsolete_agent_checkpoints_are_rejected() {
    let value = super::super::effects::encode(
        serde_json::json!({"effect":"configure","version":CHECKPOINT_VERSION - 1,"input":null}),
    )
    .unwrap();
    assert!(super::super::effects::validate_effect_checkpoint(&[], &value).is_err());
}
