use thiserror::Error;

use super::ids::{EdgeId, HandlerKey, NodeId, VarId};

#[derive(Debug, Error)]
/// Error type for graph construction, validation, and VM execution failures.
pub enum GraphError {
    /// A serialized or constructed graph failed structural validation.
    #[error("graph validation failed: {0}")]
    GraphValidation(String),

    /// A restored snapshot failed VM-state validation.
    #[error("snapshot validation failed: {0}")]
    SnapshotValidation(String),

    /// A continuation belongs to a different prepared graph.
    #[error("snapshot graph fingerprint {got} does not match prepared graph {expected}")]
    GraphMismatch { expected: String, got: String },

    /// JSON input could not be decoded for the named boundary.
    #[error("failed to decode {target} JSON: {reason}")]
    JsonDecode { target: String, reason: String },

    /// A public JSON value could not be encoded for the named boundary.
    #[error("failed to encode {target} JSON: {reason}")]
    JsonEncode { target: String, reason: String },

    /// A Rust or boundary value could not enter or leave the VM value domain.
    #[error("failed to convert {target}: {reason}")]
    ValueConversion { target: String, reason: String },

    /// Runtime history could not be persisted safely.
    #[error("history persistence failed: {0}")]
    HistoryPersistence(String),

    /// An agent's activation-time configuration function failed.
    #[error("agent configuration failed for '{agent}': {reason}")]
    AgentConfiguration { agent: String, reason: String },

    /// An agent returned an invalid resolved configuration.
    #[error("agent configuration is invalid: {0}")]
    AgentConfigValidation(String),

    /// An LLM client could not be created or executed.
    #[error("agent client operation failed: {0}")]
    AgentClient(String),

    /// An MCP resource could not be listed, resolved, or read.
    #[error("MCP resource operation failed: {0}")]
    McpResource(String),

    /// A versioned serialized payload is incompatible with this runtime.
    #[error("unsupported {format} version {got}; expected {expected}")]
    UnsupportedVersion {
        format: &'static str,
        got: u32,
        expected: u32,
    },

    /// Graph, snapshot, or transition invariant failed.
    #[error("invalid graph: {0}")]
    Invalid(String),

    /// An edge id referenced a missing dense slot.
    #[error("missing edge: {0:?}")]
    MissingEdge(EdgeId),

    /// A node id referenced a missing dense slot.
    #[error("missing node: {0:?}")]
    MissingNode(NodeId),

    /// A variable id referenced a missing dense slot.
    #[error("missing variable: {0:?}")]
    MissingVariable(VarId),

    /// A graph referenced a handler key absent from the registry.
    #[error("missing handler: {0}")]
    MissingHandler(String),

    /// A registered handler returned a domain failure.
    #[error("handler '{key}' failed: {reason}")]
    Handler { key: HandlerKey, reason: String },

    /// A node returned the wrong number of outputs.
    #[error("node '{node}' expected {expected} output(s), got {got}")]
    OutputArity {
        node: String,
        expected: usize,
        got: usize,
    },

    /// `next()` was called while a suspend node is waiting for resume.
    #[error("runtime is suspended; resume before calling next")]
    ResumeRequired,

    /// `resume()` was called without an active suspend node.
    #[error("runtime is not suspended")]
    UnexpectedResume,

    /// No node can run and the active frame cannot exit.
    #[error("graph deadlock: {0}")]
    Deadlock(String),

    /// A continuation handler returned an invalid transition.
    #[error("continuation transition for node '{node}' is invalid: {reason}")]
    InvalidContinuationTransition { node: String, reason: String },

    /// A runtime value failed the backend's shape check.
    #[error("{label} does not match expected schema '{expected}': {value}")]
    Schema {
        label: String,
        expected: String,
        value: String,
    },

    /// Snapshot data was produced by an incompatible runtime version.
    #[error("snapshot version {got} is unsupported; expected {expected}")]
    SnapshotVersion { got: u32, expected: u32 },
}
