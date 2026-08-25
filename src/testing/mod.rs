//! Test helpers and fakes for integration testing flows.
//! Enable them with the `testing` feature.

pub mod client;
pub mod store;

pub use client::{
    ScriptedFactory, mock_tool_call, output_response, tool_call_response,
    tool_call_response_with_thought,
};
pub use store::CapturingHistoryStore;

use crate::legacy::{FlowHistory, HistoryEntry};

/// Returns the number of live history entries for `session_id`.
pub fn session_message_count(history: &FlowHistory, session_id: &str) -> usize {
    history.session_entries(session_id).len()
}

/// Returns the live [`HistoryEntry`] values for `session_id`.
pub fn session_entries<'a>(history: &'a FlowHistory, session_id: &str) -> Vec<&'a HistoryEntry> {
    history.session_entries(session_id)
}
