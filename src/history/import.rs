//! Validation of working history supplied to a new execution.

use std::collections::BTreeSet;

use super::inspection::{SUMMARY_AGENT_ID, summary_text};
use super::{MessageHistory, validate_message_groups};
use crate::clients::Role;
use crate::graph::GraphError;

impl MessageHistory {
    /// Checks imported row identities, ordering and completed exchanges before runtime ownership.
    pub(crate) fn validate_import(&self) -> Result<(), GraphError> {
        let mut ids = BTreeSet::new();
        let mut sessions = BTreeSet::new();
        let mut previous = None;
        for entry in self.entries() {
            if entry.evicted
                || entry.session_id.is_empty()
                || !ids.insert(entry.id)
                || previous.is_some_and(|position| position >= entry.position)
                || entry.position >= self.next_position()
            {
                return Err(invalid("invalid identity, position or evicted row"));
            }
            previous = Some(entry.position);
            sessions.insert(entry.session_id.as_str());
        }
        for session in sessions {
            let entries = self.session_entries(session);
            for (index, entry) in entries.iter().enumerate() {
                if entry.agent_id == SUMMARY_AGENT_ID
                    && (index != 0 || summary_text(entry).is_none())
                {
                    return Err(invalid("invalid or misplaced working-memory summary"));
                }
            }
            validate_message_groups(entries.iter().map(|entry| &entry.message))
                .map_err(|reason| invalid(&reason))?;
            if self.session_is_busy(session) {
                return Err(invalid("imported conversation has an unfinished exchange"));
            }
        }
        Ok(())
    }

    /// Detects an unfinished exchange without allocating or treating a summary as an open turn.
    pub(crate) fn session_is_busy(&self, session: &str) -> bool {
        self.entries()
            .iter()
            .rev()
            .find(|entry| {
                !entry.evicted
                    && entry.session_id == session
                    && !matches!(entry.message.role, Role::System)
            })
            .is_some_and(|entry| !matches!(entry.message.role, Role::Assistant))
    }
}

fn invalid(reason: &str) -> GraphError {
    GraphError::HistoryValidation(reason.into())
}

#[cfg(test)]
#[path = "tests/import.rs"]
mod tests;
