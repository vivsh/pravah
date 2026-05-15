//! Test helpers, mocks, and fakes for integration testing flows.
//!
//! Enable with the `testing` crate feature:
//!
//! ```toml
//! [dev-dependencies]
//! pravah = { path = "…", features = ["testing"] }
//! ```
//!
//! # Quick start
//!
//! ```rust,ignore
//! use pravah::testing::{ScriptedFactory, CapturingHistoryStore, output_response, mock_tool_call};
//! use serde_json::json;
//!
//! let factory = ScriptedFactory::new()
//!     .then_tool_calls(vec![mock_tool_call("c1", "my_tool", json!({"x": 1}))])
//!     .then_output(json!({ "result": "done" }));
//! let spy = factory.clone();
//!
//! let store = CapturingHistoryStore::new();
//! let store_spy = store.clone();
//!
//! let mut runtime = FlowRuntime::new(input)?
//!     .with_factory(factory)
//!     .with_store(store);
//!
//! // drive the flow to completion …
//!
//! assert_eq!(spy.calls().len(), 2);       // two LLM dispatches
//! assert_eq!(store_spy.flush_count(), 1); // one compaction flush
//! ```

pub mod client;
pub mod store;

pub use client::{
    mock_tool_call, output_response, tool_call_response, tool_call_response_with_thought,
    ScriptedFactory,
};
pub use store::CapturingHistoryStore;

use crate::flows::{FlowHistory, HistoryEntry};

/// Returns the number of non-evicted history entries for `session_id`.
pub fn session_message_count(history: &FlowHistory, session_id: &str) -> usize {
    history.session_entries(session_id).len()
}

/// Returns a snapshot of non-evicted [`HistoryEntry`] values for `session_id`.
pub fn session_entries<'a>(history: &'a FlowHistory, session_id: &str) -> Vec<&'a HistoryEntry> {
    history.session_entries(session_id)
}
