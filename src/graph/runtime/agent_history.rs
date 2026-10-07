//! Validates worker history acknowledgements before committing any completion state.

use super::*;
use crate::graph::agent_request::AgentOperation;
use crate::history::{HistoryEntry, ValidatedCompactionResult, prepare_compaction};

/// Operation-local validated changes; no duplicate history owner or retained decisions.
type PreparedAgentHistory = (
    Vec<HistoryEntry>,
    Option<(String, ValidatedCompactionResult)>,
);

impl Runtime {
    /// Rejects forged acknowledgement ranges and validates completed imported conversations.
    pub(super) fn prepare_agent_history(
        &self,
        request: &AgentRequest,
        response: &AgentResponse,
    ) -> Result<PreparedAgentHistory, GraphError> {
        self.validate_persistence_ack(request, response)?;
        let loaded = self.prepare_loaded_history(request, response)?;
        let compacted = self.prepare_compacted_history(request, response)?;
        Ok((loaded, compacted))
    }

    /// A successful completion must acknowledge its entire requested batch; failures may acknowledge a prefix.
    fn validate_persistence_ack(
        &self,
        request: &AgentRequest,
        response: &AgentResponse,
    ) -> Result<(), GraphError> {
        let entries = request.persist.as_deref().unwrap_or_default();
        if response.outcome().is_ok()
            && entries
                .last()
                .is_some_and(|entry| entry.position.checked_add(1) != response.persisted_through)
        {
            return Err(GraphError::AgentRequestValidation(
                "successful operation did not acknowledge persistence".into(),
            ));
        }
        if let Some(position) = response.persisted_through
            && (position < self.state.persisted_history_position
                || !entries
                    .iter()
                    .any(|entry| entry.position.checked_add(1) == Some(position)))
        {
            return Err(GraphError::AgentRequestValidation(
                "invalid persistence acknowledgement".into(),
            ));
        }
        Ok(())
    }

    /// Replacement decisions must refer to the exact session prefix sent to the worker.
    fn prepare_compacted_history(
        &self,
        request: &AgentRequest,
        response: &AgentResponse,
    ) -> Result<Option<(String, ValidatedCompactionResult)>, GraphError> {
        Ok(match &response.compaction {
            Some((session, decision)) => {
                let AgentOperation::Generate {
                    session_id,
                    entries,
                    compact: true,
                    ..
                } = request.operation.as_ref()
                else {
                    return Err(GraphError::AgentRequestValidation(
                        "unexpected compaction acknowledgement".into(),
                    ));
                };
                let current = self.history.session_entries(session);
                if session != session_id
                    || !current
                        .iter()
                        .map(|entry| (entry.id, entry.position))
                        .eq(entries.iter().map(|entry| (entry.id, entry.position)))
                {
                    return Err(GraphError::AgentRequestValidation(
                        "compaction source differs from runtime history".into(),
                    ));
                }
                let replacement =
                    prepare_compaction(session, &current, decision.clone(), Some(response.id()))
                        .map_err(|reason| GraphError::HistoryCompactionValidation {
                            session_id: session.clone(),
                            reason,
                        })?;
                Some((session.clone(), replacement))
            }
            None => None,
        })
    }

    /// Imports completed rows under their stable original identities and new execution-local positions.
    fn prepare_loaded_history(
        &self,
        request: &AgentRequest,
        response: &AgentResponse,
    ) -> Result<Vec<HistoryEntry>, GraphError> {
        validate_required_load(request, response)?;
        let Some((key, entries)) = &response.loaded else {
            return Ok(Vec::new());
        };
        validate_loaded_selection(request, response, key)?;
        if key.trim().is_empty() || self.state.loaded_conversation_keys.contains(key) {
            return Err(GraphError::AgentRequestValidation(
                "unexpected loaded conversation".into(),
            ));
        }
        let session = format!("key:{key}");
        if !self.history.session_entries(&session).is_empty()
            || entries.iter().any(|entry| {
                entry.session_id != session
                    || self.history.entries().iter().any(|old| old.id == entry.id)
            })
        {
            return Err(GraphError::AgentRequestValidation(
                "loaded history overlaps existing context or a different key".into(),
            ));
        }
        let mut loaded = entries.clone();
        let mut position = self.history.next_position();
        for entry in &mut loaded {
            entry.position = position;
            position = position.checked_add(1).ok_or_else(|| {
                GraphError::HistoryPersistence("history positions exhausted".into())
            })?;
        }
        MessageHistory::from_entries(loaded.clone()).validate_import()?;
        Ok(loaded)
    }

    /// Commits already validated history and progress before recording the owner's accepted input.
    pub(super) fn commit_agent_history(
        &mut self,
        prepared: PreparedAgentHistory,
        response: &AgentResponse,
    ) {
        if let Some(position) = response.persisted_through {
            self.state.persisted_history_position = position;
        }
        let (loaded, compacted) = prepared;
        for entry in loaded {
            self.history.commit_loaded_entry(entry);
        }
        if let Some((key, _)) = &response.loaded {
            self.state.loaded_conversation_keys.insert(key.clone());
            self.state.persisted_history_position = self.history.next_position();
        }
        if let Some((session, replacement)) = compacted {
            self.history.commit_replacement(&session, replacement);
        }
    }
}

/// Successful activation must acknowledge any required first load, including an empty conversation.
fn validate_required_load(
    request: &AgentRequest,
    response: &AgentResponse,
) -> Result<(), GraphError> {
    if let AgentOperation::Configure {
        load_history: true,
        loaded_keys,
        ..
    } = request.operation.as_ref()
        && let Ok(value) = response.outcome()
        && let Some(key) = value.get("key").and_then(Value::as_str)
        && !loaded_keys.contains(key)
        && response
            .loaded
            .as_ref()
            .is_none_or(|(loaded, _)| loaded != key)
    {
        return Err(GraphError::AgentRequestValidation(
            "missing loaded conversation acknowledgement".into(),
        ));
    }
    Ok(())
}

/// Only a successful valid configuration can import its newly selected conversation.
fn validate_loaded_selection(
    request: &AgentRequest,
    response: &AgentResponse,
    key: &str,
) -> Result<(), GraphError> {
    let AgentOperation::Configure {
        definition,
        load_history: true,
        loaded_keys,
        ..
    } = request.operation.as_ref()
    else {
        return Err(GraphError::AgentRequestValidation(
            "unexpected loaded conversation".into(),
        ));
    };
    let value = response.outcome().map_err(|_| {
        GraphError::AgentRequestValidation("failed operation cannot load history".into())
    })?;
    if loaded_keys.contains(key) || value.get("key").and_then(Value::as_str) != Some(key) {
        return Err(GraphError::AgentRequestValidation(
            "unexpected loaded conversation".into(),
        ));
    }
    crate::graph::agent::validate_loaded_configuration(definition, value)
}
