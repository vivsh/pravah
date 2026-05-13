use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::clients::ToolCall;

/// Generic per-frame execution phase. The payload inside `Continue` is an opaque JSON blob
/// whose structure is defined by the node type (e.g. `AgentContinuation`).
#[derive(Serialize, Deserialize, Clone)]
pub(crate) enum Phase {
    /// Node has not started yet (initial state of every frame).
    Entry,
    /// Node is in progress; the `Value` carries node-defined continuation state.
    Continue(Value),
    /// Node has a final value; the `Agent` arm flushes this inline and resets to `Entry`.
    Exit(Value),
}

/// Agent-specific continuation state, serialized as the `Value` inside `Phase::Continue`.
///
/// `ToolsDispatched` is ephemeral — `step_inner` converts it to `Waiting` immediately and
/// never persists it across a step boundary.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind")]
pub(crate) enum AgentContinuation {
    /// LLM just issued tool calls; `step_inner` will write `PendingCall` states then convert.
    ToolsDispatched { calls: Vec<ToolCall> },
    /// Tool calls have been written to the state map; waiting for them to complete.
    Waiting {
        count: usize,
        submitted: Option<Value>,
    },
}
