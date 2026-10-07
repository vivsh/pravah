//! Operation-local validated deltas; Runtime retains the sole committed history owner.

use super::*;
use crate::graph::registry::HistoryChange;
use crate::history::{HistoryEntry, ValidatedCompactionResult, prepare_compaction, summary_uuid};

/// A complete validated change, discarded after the enclosing VM transition commits.
#[expect(
    clippy::large_enum_variant,
    reason = "operation-local batches keep summary entries inline instead of allocating another box"
)]
pub(super) enum ValidatedHistoryChange {
    Append(Vec<HistoryEntry>),
    Compact {
        session_id: String,
        replacement: ValidatedCompactionResult,
    },
}

impl Runtime {
    /// Validates sequential changes against borrowed previews before any VM mutation.
    pub(super) fn prepare_history_changes(
        &self,
        frame_index: usize,
        changes: Vec<HistoryChange>,
    ) -> Result<Vec<ValidatedHistoryChange>, GraphError> {
        let mut prepared = Vec::with_capacity(changes.len());
        let mut position = self.history.next_position();
        for change in changes {
            let change = match change {
                HistoryChange::Append(entries) => {
                    self.validate_conversation_append(frame_index, &entries)?;
                    position = validate_append(&entries, position, self.state.execution_id)?;
                    ValidatedHistoryChange::Append(entries)
                }
                HistoryChange::Compact {
                    session_id,
                    decision,
                } => {
                    let current = preview(&self.history, &prepared);
                    let session = current
                        .into_iter()
                        .filter(|entry| entry.session_id == session_id && !entry.evicted)
                        .collect::<Vec<_>>();
                    let id = summary_uuid(self.state.execution_id, &session_id, position);
                    let replacement = prepare_compaction(&session_id, &session, decision, Some(id))
                        .map_err(|reason| GraphError::HistoryCompactionValidation {
                            session_id: session_id.clone(),
                            reason,
                        })?;
                    ValidatedHistoryChange::Compact {
                        session_id,
                        replacement,
                    }
                }
            };
            prepared.push(change);
        }
        Ok(prepared)
    }

    /// Applies already validated deltas after the full continuation transition is accepted.
    pub(super) fn commit_history_changes(
        &mut self,
        frame_index: usize,
        changes: Vec<ValidatedHistoryChange>,
    ) -> Result<(), GraphError> {
        for change in changes {
            match change {
                ValidatedHistoryChange::Append(entries) => {
                    self.register_frame_conversations(frame_index, &entries)?;
                    for entry in entries {
                        self.history.commit_entry(entry);
                    }
                }
                ValidatedHistoryChange::Compact {
                    session_id,
                    replacement,
                } => {
                    self.history.commit_replacement(&session_id, replacement);
                }
            }
        }
        Ok(())
    }
}

/// Replays staged metadata on borrowed rows, never cloning committed messages or accounting.
fn preview<'a>(
    history: &'a MessageHistory,
    changes: &'a [ValidatedHistoryChange],
) -> Vec<&'a HistoryEntry> {
    let mut entries = history.entries().iter().collect::<Vec<_>>();
    for change in changes {
        match change {
            ValidatedHistoryChange::Append(batch) => entries.extend(batch),
            ValidatedHistoryChange::Compact {
                session_id,
                replacement,
            } => replacement.preview(session_id, &mut entries),
        }
    }
    entries
}

/// Checks stable positions and identities; usage is accounted exactly once during commit.
fn validate_append(
    entries: &[HistoryEntry],
    mut position: u64,
    execution: Uuid,
) -> Result<u64, GraphError> {
    for entry in entries {
        if entry.position != position || entry.evicted {
            return Err(GraphError::HistoryPersistence(
                "invalid append position or evicted entry".into(),
            ));
        }
        position = position
            .checked_add(1)
            .ok_or_else(|| GraphError::HistoryPersistence("history positions exhausted".into()))?;
        if entry.id != history_uuid(execution, entry.position) {
            return Err(GraphError::HistoryPersistence(
                "invalid stable history identity".into(),
            ));
        }
    }
    Ok(position)
}

pub(crate) fn history_uuid(execution_id: Uuid, position: u64) -> Uuid {
    Uuid::new_v5(
        &execution_id,
        format!("pravah.history.v1:{position}").as_bytes(),
    )
}
