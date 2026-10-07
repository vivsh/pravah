use std::io::{self, Write};

use super::HistoryEntry;
use crate::clients::{Message, Role};

pub(crate) const SUMMARY_AGENT_ID: &str = "__summary__";
pub(super) const SUMMARY_PREFIX: &str = "<pravah_working_memory>\n";
pub(super) const SUMMARY_SUFFIX: &str = "\n</pravah_working_memory>";

fn is_summary(entry: &HistoryEntry) -> bool {
    entry.agent_id == SUMMARY_AGENT_ID
}

/// Borrows the original non-empty summary text only from a recognized system summary.
pub(super) fn summary_text(entry: &HistoryEntry) -> Option<&str> {
    if !is_summary(entry) || !matches!(entry.message.role, Role::System) {
        return None;
    }
    entry
        .message
        .content
        .strip_prefix(SUMMARY_PREFIX)?
        .strip_suffix(SUMMARY_SUFFIX)
        .filter(|text| !text.trim().is_empty())
}

/// Enumerates conversation messages without changing indices in the original live history.
pub(super) fn enum_messages<'a>(
    entries: impl Iterator<Item = &'a HistoryEntry> + Clone,
    skip_recent: usize,
) -> impl Iterator<Item = (usize, &'a Message)> {
    let retained = if skip_recent == 0 {
        usize::MAX
    } else {
        entries
            .clone()
            .filter(|entry| is_conversation(entry))
            .count()
            .saturating_sub(skip_recent)
    };
    entries
        .enumerate()
        .filter(|(_, entry)| is_conversation(entry))
        .take(retained)
        .map(|(index, entry)| (index, &entry.message))
}

fn is_conversation(entry: &HistoryEntry) -> bool {
    !is_summary(entry)
        && !matches!(
            entry.message.role,
            Role::AssistantToolCalls { .. } | Role::Tool { .. }
        )
}

/// Counts completed user exchanges, not intermediate tool rounds or standalone assistant messages.
pub(super) fn turn_count<'a>(messages: impl Iterator<Item = &'a Message>) -> usize {
    let mut pending_user = false;
    let mut complete = 0;
    for message in messages {
        match message.role {
            Role::User => pending_user = true,
            Role::Assistant if pending_user => {
                complete += 1;
                pending_user = false;
            }
            _ => {}
        }
    }
    complete
}

/// Counts the exact compact JSON array encoding without allocating a serialized message buffer.
pub(super) fn byte_size<'a>(
    messages: impl Iterator<Item = &'a Message>,
) -> Result<usize, serde_json::Error> {
    let mut size = JsonByteCount(0);
    size.write_all(b"[").map_err(serde_json::Error::io)?;
    for (index, message) in messages.enumerate() {
        if index > 0 {
            size.write_all(b",").map_err(serde_json::Error::io)?;
        }
        serde_json::to_writer(&mut size, message)?;
    }
    size.write_all(b"]").map_err(serde_json::Error::io)?;
    Ok(size.0)
}

/// Operation-local byte counter; no history, encoded bytes or cached size is retained.
struct JsonByteCount(usize);

impl Write for JsonByteCount {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::other("serialized history size exceeds usize"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
