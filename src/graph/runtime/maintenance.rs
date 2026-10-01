//! Read-only dispatch inspection and atomic caller-requested history replacement.

use super::*;
use crate::history::{CompactionResult, prepare_compaction, summary_uuid};

impl Runtime {
    /// Replaces a complete prefix while protecting current input and required tool groups.
    /// Validation errors leave history and execution unchanged. The caller must persist
    /// original rows first; replacement never changes an already frozen pending request.
    pub fn compact_history(
        &mut self,
        session: &str,
        decision: CompactionResult,
    ) -> Result<(), GraphError> {
        if self.pending_fetch().is_some() {
            return Err(GraphError::HistoryCompactionValidation {
                session_id: session.into(),
                reason: "cannot compact while a Fetch is pending".into(),
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

    pub(crate) fn history_dispatch_session(&self) -> Option<&str> {
        let (_, checkpoint) = self.history_dispatch()?;
        checkpoint.get("session_id")?.as_str()
    }

    #[expect(
        clippy::type_complexity,
        reason = "operation-local request context avoids a second configuration owner"
    )]
    pub(crate) fn history_dispatch_request(
        &self,
    ) -> Result<
        Option<(
            String,
            super::super::fetch::rath::RathRequest,
            Vec<crate::clients::Message>,
            bool,
        )>,
        GraphError,
    > {
        self.history_dispatch()
            .map(|(payload, checkpoint)| crate::graph::agent::dispatch_request(payload, checkpoint))
            .transpose()
    }

    /// Mirrors ascending instruction readiness using borrowed frames; only the next instruction matters.
    fn history_dispatch(&self) -> Option<(&Value, &Value)> {
        if self.state.waiting.is_some() {
            return None;
        }
        let frame = self.state.frames.last()?;
        let compiled = self.callables.get(frame.graph_index)?;
        for node_id in compiled.instructions.iter() {
            let node = compiled.nodes.get(node_id.0)?;
            let checkpoint = frame.checkpoints.get(node.id.0)?.as_ref();
            if checkpoint.is_none() && !inputs_ready_with_new_epoch(frame, node).ok()? {
                continue;
            }
            let CompiledNodeKind::Continuation { payload, .. } = &node.kind else {
                return None;
            };
            let checkpoint = checkpoint?;
            if payload.get("agent_id").is_none()
                || !frame.continuation_inboxes.get(node.id.0)?.is_empty()
                || !frame.continuation_child_queues.get(node.id.0)?.is_empty()
                || checkpoint.get("phase")?.get("kind")?.as_str()? != "dispatch"
            {
                return None;
            }
            return Some((payload, checkpoint));
        }
        None
    }
}
