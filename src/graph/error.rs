use thiserror::Error;

use super::ids::{EdgeId, HandlerKey, NodeId, VarId};

#[derive(Debug, Error)]
/// Error type for graph construction, validation, and VM execution failures.
pub enum GraphError {
    /// A Chat request failed validation before the suspended execution accepted it.
    #[error("invalid chat request: {reason}")]
    ChatRequestValidation {
        /// Reason the request must be corrected before submission.
        reason: String,
    },
    /// A chat operation requires an idle application-input boundary.
    #[error(
        "chat is not ready for '{operation}'; an unfinished turn cannot accept input or state changes"
    )]
    ChatNotReady {
        /// Application operation that requires a between-turn input boundary.
        operation: &'static str,
    },

    /// An agent or tool suspended somewhere other than a chat response boundary.
    #[error("chat execution suspended inside an agent or tool; its snapshot remains available")]
    ChatSuspended,
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

    /// An application could not prepare working memory for the upcoming request.
    #[error("history compaction failed for '{session_id}': {source}")]
    HistoryCompaction {
        /// Session whose history remains unchanged.
        session_id: String,
        /// Original application error, available for inspection and downcasting.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A history replacement or resulting message sequence is unsafe.
    #[error("history compaction is invalid for '{session_id}': {reason}")]
    HistoryCompactionValidation {
        /// Session whose history remains unchanged.
        session_id: String,
        /// Invalid index, message group, or stale observation.
        reason: String,
    },

    /// An agent's activation-time configuration function failed.
    #[error("agent configuration failed for '{agent}': {reason}")]
    AgentConfiguration { agent: String, reason: String },

    /// An agent returned an invalid resolved configuration.
    #[error("agent configuration is invalid: {0}")]
    AgentConfigValidation(String),

    /// An agent's intervention controller could not evaluate a boundary.
    #[error("agent control failed for '{agent}': {reason}")]
    AgentControl { agent: String, reason: String },

    /// An agent controller returned an invalid intervention decision.
    #[error("agent control decision is invalid: {0}")]
    AgentControlValidation(String),

    /// An application policy deliberately stopped an agent step.
    #[error("agent intervention aborted for '{agent}': {reason}")]
    AgentPolicyAbort { agent: String, reason: String },

    /// External input could not resume a suspended agent intervention.
    #[error("agent resume decision is invalid: {0}")]
    AgentResumeValidation(String),

    /// A forced final model turn did not produce structured output.
    #[error("agent conclusion failed for '{agent}': {reason}")]
    AgentConclusion { agent: String, reason: String },

    /// An LLM client could not be created or executed.
    #[error("agent client operation failed: {0}")]
    AgentClient(String),

    /// The provider exhausted its generation-token cap; partial output is discarded.
    #[error("agent '{agent}' reached the output token limit for provider '{provider:?}'")]
    AgentOutputLimit {
        /// Stable identity of the agent whose request was interrupted.
        agent: String,
        /// Provider that reported output exhaustion.
        provider: crate::clients::Provider,
    },

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
