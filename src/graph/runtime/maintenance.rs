//! Read-only dispatch inspection and atomic caller-requested history replacement.

use super::*;
use crate::history::{CompactionResult, HistoryPolicy, prepare_compaction, summary_uuid};

impl Runtime {
    /// Sets execution intent before any instruction runs. No worker service is retained.
    /// Changing policy after stepping fails without modifying runtime state.
    pub fn with_history(mut self, policy: HistoryPolicy) -> Result<Self, GraphError> {
        if self.state.frames.len() != 1
            || self.state.waiting.is_some()
            || self.state.frames.iter().any(|frame| {
                frame.node_epochs.iter().any(|epoch| *epoch != 0)
                    || frame.checkpoints.iter().any(Option::is_some)
            })
        {
            return Err(GraphError::HistoryValidation(
                "history policy must be installed before stepping".into(),
            ));
        }
        self.state.history_policy = policy;
        Ok(self)
    }

    /// Borrows the execution's checkpointed maintenance intent, not worker availability.
    pub fn history_policy(&self) -> &HistoryPolicy {
        &self.state.history_policy
    }
    /// Replaces a complete prefix while protecting current input and required tool groups.
    /// Validation errors leave history and execution unchanged. The caller must persist
    /// original rows first; replacement never changes an already frozen pending request.
    pub fn compact_history(
        &mut self,
        session: &str,
        decision: CompactionResult,
    ) -> Result<(), GraphError> {
        if self.pending_agent().is_some() {
            return Err(GraphError::HistoryCompactionValidation {
                session_id: session.into(),
                reason: "cannot compact while agent work is pending".into(),
            });
        }
        let entries = self.history.session_entries(session);
        let id = summary_uuid(
            self.state.execution_id,
            session,
            self.history.next_position(),
        );
        let replacement =
            prepare_compaction(session, &entries, decision, Some(id)).map_err(|reason| {
                GraphError::HistoryCompactionValidation {
                    session_id: session.into(),
                    reason,
                }
            })?;
        self.history.commit_replacement(session, replacement);
        Ok(())
    }
}

/// Rejects impossible durable acknowledgements before returning a restored runtime.
pub(super) fn validate_history_progress(
    state: &State,
    history: &crate::history::MessageHistory,
) -> Result<(), GraphError> {
    if state.persisted_history_position > history.next_position()
        || state
            .loaded_conversation_keys
            .iter()
            .any(|key| key.trim().is_empty())
    {
        return Err(GraphError::SnapshotValidation(
            "invalid history acknowledgement or loaded conversation key".into(),
        ));
    }
    Ok(())
}
