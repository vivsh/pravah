use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Execution phase for a frame.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) enum Phase {
    /// Frame has not started yet.
    Entry,
    /// Frame is in progress.
    /// The optional value stores node-specific continuation state.
    Continue(Option<Value>),
}
