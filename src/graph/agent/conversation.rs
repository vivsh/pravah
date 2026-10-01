//! Conversation selection from activation keys; Runtime owns all working history.

use super::*;
use crate::history::MessageHistory;

/// Namespaces application keys and fresh invocation IDs without another identity table.
pub(super) fn select_session(
    key: Option<String>,
    invocation: Uuid,
    history: &MessageHistory,
) -> Result<String, GraphError> {
    let Some(key) = key else {
        return Ok(invocation.to_string());
    };
    if key.trim().is_empty() {
        return Err(GraphError::AgentConfigValidation(
            "conversation key must not be empty".into(),
        ));
    }
    let capacity = key
        .len()
        .checked_add(4)
        .ok_or_else(|| GraphError::AgentConfigValidation("conversation key is too large".into()))?;
    let mut session = String::with_capacity(capacity);
    session.push_str("key:");
    session.push_str(&key);
    if history.session_is_busy(&session) {
        return Err(GraphError::AgentConversationBusy);
    }
    Ok(session)
}

/// Runtime-owned conversations need no additional agent node-local saved state.
pub(super) fn validate_empty_state(state: Option<&Value>) -> Result<(), GraphError> {
    if state.is_some() {
        return Err(GraphError::SnapshotValidation(
            "agent node-local saved state is no longer supported".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/conversation.rs"]
mod tests;
