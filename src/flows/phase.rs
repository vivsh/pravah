use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Generic per-frame execution phase.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) enum Phase {
    /// Frame has not started yet (initial state of every frame).
    Entry,
    /// Frame is in progress; optional `Value` carries node-defined continuation state.
    Continue(Option<Value>),
}
