use thiserror::Error;

#[derive(Debug, Error)]
pub enum FlowError {
    #[error("Node not found: {0}")]
    NotFound(String),

    #[error("failed to serialize: {0}")]
    Serialize(#[source] serde_json::Error),

    #[error("failed to deserialize: {0}")]
    Deserialize(#[source] serde_json::Error),

    #[error("Flow is suspended — call resume() with a resumption payload, not next()")]
    ResumeRequired,

    #[error("Flow is not suspended — unexpected resumption payload supplied")]
    UnexpectedResumption,

    #[error("resume type mismatch: expected '{expected}', got '{got}'")]
    ResumptionTypeMismatch { expected: String, got: String },

    #[error("Flow deadlock: states [{0}] are waiting but no join is ready")]
    Deadlock(String),

    #[error("Internal error in {handler}: {detail}")]
    Internal {
        handler: &'static str,
        detail: String,
    },

    #[error(transparent)]
    Build(#[from] BuildError),

    #[error(transparent)]
    Agent(#[from] AgentError),
}

/// Errors raised while building or validating a flow graph.
#[derive(Debug, Error)]
pub enum BuildError {
    /// Collected validation failures.
    #[error("Flow graph is invalid:\n{}", .0.join("\n"))]
    Invalid(Vec<String>),

    /// Duplicate node registration.
    #[error("Duplicate node: {0}")]
    DuplicateNode(String),

    /// Fork output count did not match the declared children.
    #[error("Fork child count mismatch: {0}")]
    ChildCountMismatch(String),

    /// Snapshot deserialization failed.
    #[error("Snapshot load error: {0}")]
    SnapshotLoad(String),

    /// Snapshot serialization failed.
    #[error("Snapshot store error: {0}")]
    SnapshotStore(String),
}

/// Errors raised while an agent or tool is running.
#[derive(Debug, Error)]
pub enum AgentError {
    /// Model client creation, validation, or execution failed.
    #[error("Agent '{agent}' LLM call failed: {reason}")]
    LlmFailed { agent: String, reason: String },

    /// Model requested an unregistered tool.
    #[error("Agent '{agent}' called unknown tool '{tool}'")]
    UnknownTool { agent: String, tool: String },

    /// Model repeated a call id within one turn.
    #[error("Agent '{agent}' issued duplicate call_id for tool '{tool}'")]
    DuplicateToolCall { agent: String, tool: String },

    /// A registered tool failed without suspending or exiting.
    #[error("Tool '{tool}' failed: {reason}")]
    ToolFailed { tool: String, reason: String },

    /// Agent continuation serialization failed.
    #[error("failed to serialize agent state: {0}")]
    Serialize(#[source] serde_json::Error),

    /// Agent continuation deserialization failed.
    #[error("failed to deserialize agent state: {0}")]
    Deserialize(#[source] serde_json::Error),
}
