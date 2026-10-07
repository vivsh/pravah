//! Frame-owned ephemeral conversations and caller-controlled keyed working history.

use super::*;
use crate::graph::agent_request::AgentOperation;
use crate::history::HistoryEntry;

impl Runtime {
    /// Reports whether execution still retains or references this exact history session ID.
    /// Unkeyed sessions remain active until their owning frame exits. Keyed sessions
    /// remain active while a checkpoint or outstanding operation references them.
    /// This borrowed inspection allocates nothing; absent sessions are inactive.
    pub fn conversation_is_active(&self, conversation_id: &str) -> bool {
        self.history.session_is_busy(conversation_id)
            || self.state.frames.iter().any(|frame| {
                frame
                    .unkeyed_conversations
                    .binary_search_by(|session| session.as_str().cmp(conversation_id))
                    .is_ok()
                    || self.frame_references_conversation(frame, conversation_id)
            })
            || self
                .pending_agent()
                .is_some_and(|request| request_references(request, conversation_id))
    }

    /// Drops inactive working history by its `HistoryEntry::session_id`, not its application key.
    /// Active execution rejects removal without mutation. Missing IDs are a no-op.
    /// No store is called and unpersisted rows may be lost. Keyed removal also clears
    /// the load acknowledgement so a later invocation may reload through its store.
    /// Append positions, usage totals, snapshots' execution identity and archives are unchanged.
    pub fn drop_conversation(&mut self, conversation_id: &str) -> Result<(), GraphError> {
        if self.conversation_is_active(conversation_id) {
            return Err(GraphError::AgentConversationBusy);
        }
        self.history.remove_conversation(conversation_id);
        if let Some(key) = conversation_id.strip_prefix("key:") {
            self.state.loaded_conversation_keys.remove(key);
        }
        Ok(())
    }

    /// Borrows only agent checkpoints and accepted work; no decoded lifecycle cache is retained.
    fn frame_references_conversation(&self, frame: &Frame, session: &str) -> bool {
        let checkpoints = self.callables.get(frame.graph_index).is_some_and(|graph| {
            graph
                .nodes
                .iter()
                .zip(&frame.checkpoints)
                .any(|(node, checkpoint)| {
                    is_agent_node(node)
                        && checkpoint.as_ref().and_then(agent_session) == Some(session)
                })
        });
        checkpoints
            || frame.continuation_inboxes.iter().flatten().any(|input| {
                let ContinuationInput::Agent { request, response } = input else {
                    return false;
                };
                request_references(request, session)
                    || (matches!(request.operation.as_ref(), AgentOperation::Configure { .. })
                        && session.starts_with("key:")
                        && response.outcome().ok().is_some_and(|value| {
                            value.get("key").and_then(Value::as_str) == session.strip_prefix("key:")
                        }))
            })
    }

    /// Rejects ambiguous ownership before the enclosing continuation can mutate its frame or history.
    pub(super) fn validate_conversation_append(
        &self,
        frame_index: usize,
        entries: &[HistoryEntry],
    ) -> Result<(), GraphError> {
        self.frame(frame_index)?;
        for entry in entries {
            if entry.session_id.is_empty() {
                return Err(GraphError::HistoryValidation(
                    "empty conversation ID".into(),
                ));
            }
            if !entry.session_id.starts_with("key:")
                && self.state.frames.iter().enumerate().any(|(index, frame)| {
                    index != frame_index
                        && frame
                            .unkeyed_conversations
                            .binary_search(&entry.session_id)
                            .is_ok()
                })
            {
                return Err(GraphError::HistoryValidation(
                    "unkeyed conversation belongs to another frame".into(),
                ));
            }
        }
        Ok(())
    }

    /// Registers each unkeyed session once at the successful append boundary, in stable order.
    pub(super) fn register_frame_conversations(
        &mut self,
        frame_index: usize,
        entries: &[HistoryEntry],
    ) -> Result<(), GraphError> {
        let frame = self.frame_mut(frame_index)?;
        register_sessions(frame, entries);
        Ok(())
    }
}

/// Records unique unkeyed sessions at initial history import or a successful frame-local append.
pub(super) fn register_sessions(frame: &mut Frame, entries: &[HistoryEntry]) {
    for entry in entries {
        if !entry.session_id.starts_with("key:")
            && let Err(index) = frame.unkeyed_conversations.binary_search(&entry.session_id)
        {
            frame
                .unkeyed_conversations
                .insert(index, entry.session_id.clone());
        }
    }
}

fn is_agent_node(node: &CompiledNode) -> bool {
    matches!(&node.kind, CompiledNodeKind::Continuation { payload, .. }
        if payload.get("agent_id").is_some() && payload.get("output_schema").is_some())
}

/// Borrows the session from a trusted checkpoint; configuration and final flush have no inner loop.
fn agent_session(checkpoint: &Value) -> Option<&str> {
    checkpoint
        .get("session_id")
        .and_then(Value::as_str)
        .or_else(|| checkpoint.get("checkpoint")?.get("session_id")?.as_str())
}

/// Frozen persistence rows, selections and request inputs cannot be invalidated by manual removal.
fn request_references(request: &AgentRequest, session: &str) -> bool {
    if request
        .persist
        .as_deref()
        .is_some_and(|entries| entries.iter().any(|entry| entry.session_id == session))
    {
        return true;
    }
    match request.operation.as_ref() {
        AgentOperation::Configure { loaded_keys, .. } => session
            .strip_prefix("key:")
            .is_some_and(|key| loaded_keys.contains(key)),
        AgentOperation::Control { observation, .. } => {
            observation.get("session_id").and_then(Value::as_str) == Some(session)
        }
        AgentOperation::Generate { session_id, .. } => session_id == session,
        AgentOperation::Tool { .. } | AgentOperation::PersistHistory => false,
    }
}

/// Validates ordered, exclusive ownership and live agent/frame relationships before restoration.
pub(super) fn validate_conversation_owners(
    callables: &[CompiledGraph],
    state: &State,
    history: &MessageHistory,
) -> Result<(), GraphError> {
    let mut owners = std::collections::BTreeMap::new();
    for (index, frame) in state.frames.iter().enumerate() {
        let mut previous = None;
        for session in &frame.unkeyed_conversations {
            if session.is_empty()
                || session.starts_with("key:")
                || previous.is_some_and(|previous| previous >= session)
                || owners.insert(session.as_str(), index).is_some()
            {
                return Err(invalid_ownership());
            }
            previous = Some(session);
        }
        validate_frame_sessions(callables, frame)?;
    }
    for entry in history.entries() {
        if !entry.session_id.starts_with("key:") {
            let index = owners
                .get(entry.session_id.as_str())
                .ok_or_else(invalid_ownership)?;
            let frame = state.frames.get(*index).ok_or_else(invalid_ownership)?;
            if entry.id == history::history_uuid(state.execution_id, entry.position) {
                validate_entry_owner(callables, frame, entry)?;
            }
        }
    }
    Ok(())
}

/// Every active unkeyed agent must belong to its checkpoint's exact frame, including suspensions.
fn validate_frame_sessions(callables: &[CompiledGraph], frame: &Frame) -> Result<(), GraphError> {
    let graph = callables
        .get(frame.graph_index)
        .ok_or_else(invalid_ownership)?;
    for (node, checkpoint) in graph.nodes.iter().zip(&frame.checkpoints) {
        if is_agent_node(node)
            && let Some(session) = checkpoint.as_ref().and_then(agent_session)
            && !session.starts_with("key:")
            && frame
                .unkeyed_conversations
                .binary_search_by(|id| id.as_str().cmp(session))
                .is_err()
        {
            return Err(invalid_ownership());
        }
    }
    Ok(())
}

/// Local agent rows cannot move to a different authored frame; imported rows keep their provenance.
fn validate_entry_owner(
    callables: &[CompiledGraph],
    frame: &Frame,
    entry: &HistoryEntry,
) -> Result<(), GraphError> {
    let authors = |graph: &CompiledGraph| {
        graph.nodes.iter().any(|node| {
            matches!(&node.kind, CompiledNodeKind::Continuation { payload, .. }
            if payload.get("agent_id").and_then(Value::as_str) == Some(entry.agent_id.as_str()))
        })
    };
    if callables.iter().any(authors) && !callables.get(frame.graph_index).is_some_and(authors) {
        return Err(invalid_ownership());
    }
    Ok(())
}

fn invalid_ownership() -> GraphError {
    GraphError::SnapshotValidation("invalid frame conversation ownership".into())
}

#[cfg(test)]
#[path = "tests/conversations.rs"]
mod tests;
