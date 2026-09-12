use std::collections::BTreeSet;

use super::*;
use crate::history::HistoryReplacement;

/// Fully validated operation-local changes; never stored in snapshots or runtime services.
struct ValidatedHistoryReplacement {
    remove_ids: Vec<Uuid>,
    summary: Option<HistoryEntry>,
}

/// Protects the newest user input and all its tool rounds until final model output.
/// With no identifiable user input, conservatively protect the entire session.
pub(crate) fn protected_start(entries: &[&HistoryEntry]) -> usize {
    entries
        .iter()
        .rposition(|entry| matches!(entry.message.role, Role::User))
        .unwrap_or(0)
}

/// Checks complete contiguous tool batches without depending on provider-specific adapters.
pub(crate) fn validate_message_groups<'a>(
    messages: impl IntoIterator<Item = &'a Message>,
) -> Result<(), String> {
    let mut pending = BTreeSet::new();
    for message in messages {
        match &message.role {
            Role::Tool { call_id } => {
                if !pending.remove(call_id.as_str()) {
                    return Err(format!(
                        "orphan, duplicate, or unknown tool result '{call_id}'"
                    ));
                }
            }
            role => {
                if !pending.is_empty() {
                    return Err("tool-call batch is missing results before the next message".into());
                }
                if let Role::AssistantToolCalls { calls } = role {
                    if calls.is_empty() {
                        return Err("assistant tool-call batch is empty".into());
                    }
                    for call in calls {
                        if call.id.is_empty() || !pending.insert(call.id.as_str()) {
                            return Err(
                                "assistant tool-call IDs must be non-empty and unique".into()
                            );
                        }
                    }
                }
            }
        }
    }
    if pending.is_empty() {
        Ok(())
    } else {
        Err("history ends with unresolved tool calls".into())
    }
}

impl FlowHistory {
    /// Validates a replacement and returns the exact prepared request history after commit.
    /// Policy and validation failures occur before any mutation, including tombstone pruning.
    pub(crate) fn replace_prepared_history(
        &mut self,
        session_id: &str,
        observed: &[&HistoryEntry],
        decision: HistoryReplacement,
    ) -> Result<Vec<Message>, String> {
        let current = self.session_entries(session_id);
        if !current
            .iter()
            .map(|entry| (entry.id, entry.position))
            .eq(observed.iter().map(|entry| (entry.id, entry.position)))
        {
            return Err("session history changed during preparation".into());
        }
        validate_message_groups(current.iter().map(|entry| &entry.message))?;
        let replacement = validate_replacement(session_id, &current, decision)?;
        let mut messages = Vec::with_capacity(current.len());
        if let Some(summary) = &replacement.summary {
            messages.push(summary.message.clone());
        }
        messages.extend(
            current
                .iter()
                .skip(replacement.remove_ids.len())
                .map(|e| e.message.clone()),
        );
        validate_message_groups(&messages)?;
        self.commit_replacement(session_id, replacement);
        Ok(messages)
    }

    /// Removes only this session's replaced entries and tombstones without touching accounting.
    fn commit_replacement(&mut self, session_id: &str, replacement: ValidatedHistoryReplacement) {
        let insert_at = self
            .entries
            .iter()
            .position(|entry| entry.session_id == session_id);
        self.entries.retain(|entry| {
            entry.session_id != session_id
                || (!entry.evicted && !replacement.remove_ids.contains(&entry.id))
        });
        if let Some(summary) = replacement.summary {
            self.entries.insert(
                insert_at
                    .unwrap_or(self.entries.len())
                    .min(self.entries.len()),
                summary,
            );
        }
    }
}

/// Rejects invalid prefix decisions and stages a summary with a fresh ID and stable position.
fn validate_replacement(
    session_id: &str,
    entries: &[&HistoryEntry],
    decision: HistoryReplacement,
) -> Result<ValidatedHistoryReplacement, String> {
    validate_indices(entries, &decision.evict_indices)?;
    let count = decision.evict_indices.len();
    let removed = entries.get(..count).ok_or("invalid replacement prefix")?;
    validate_message_groups(removed.iter().map(|entry| &entry.message))?;
    if count > 0
        && entries
            .get(count)
            .is_some_and(|entry| !matches!(entry.message.role, Role::User))
    {
        return Err("replacement must end at a complete exchange boundary".into());
    }
    let summary = match decision.summary {
        Some(text) => {
            let first = removed
                .first()
                .ok_or("a summary requires replaced history")?;
            if text.trim().is_empty() {
                return Err("summary text must not be empty".into());
            }
            Some(summary_entry(session_id, first.position, text))
        }
        None => None,
    };
    Ok(ValidatedHistoryReplacement {
        remove_ids: removed.iter().map(|entry| entry.id).collect(),
        summary,
    })
}

/// Checks the policy's authored indices before constructing any replacement entries.
fn validate_indices(entries: &[&HistoryEntry], indices: &[usize]) -> Result<(), String> {
    let protected = protected_start(entries);
    for (expected, &index) in indices.iter().enumerate() {
        if index >= entries.len() {
            return Err(format!("history index {index} is out of bounds"));
        }
        if index >= protected {
            return Err(format!(
                "history index {index} belongs to the protected current exchange"
            ));
        }
        if index != expected {
            return Err("eviction indices must be a sorted, unique contiguous prefix 0..n".into());
        }
    }
    Ok(())
}

/// Creates a text-only system summary without changing append positions or usage accounting.
fn summary_entry(session_id: &str, position: u64, text: String) -> HistoryEntry {
    let mut entry = HistoryEntry::new(
        session_id,
        "__summary__",
        Message {
            role: Role::System,
            content: format!("<pravah_working_memory>\n{text}\n</pravah_working_memory>"),
            attachments: Vec::new(),
            usage: None,
        },
    );
    entry.position = position;
    entry
}
