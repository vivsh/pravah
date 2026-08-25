use thiserror::Error;

/// Errors that can occur during flow execution or construction.
#[derive(Debug, Error)]
pub enum FlowError {
    /// A node referenced by id does not exist in the graph.
    #[error("Node not found: {0}")]
    NotFound(String),

    /// State serialization failed.
    #[error("failed to serialize: {0}")]
    Serialize(#[source] serde_json::Error),

    /// State deserialization failed.
    #[error("failed to deserialize: {0}")]
    Deserialize(#[source] serde_json::Error),

    /// `FlowRuntime::next` was called on a suspended flow — use `FlowRuntime::resume` instead.
    #[error("Flow is suspended — call resume() with a resumption payload, not next()")]
    ResumeRequired,

    /// `FlowRuntime::resume` was called on a flow that is not suspended.
    #[error("Flow is not suspended — unexpected resumption payload supplied")]
    UnexpectedResumption,

    /// The resumption type does not match the type the flow is waiting for.
    #[error("resume type mismatch: expected '{expected}', got '{got}'")]
    ResumptionTypeMismatch { expected: String, got: String },

    /// Multiple branches are waiting for a join that can never fire.
    #[error("Flow deadlock: states [{0}] are waiting but no join is ready")]
    Deadlock(String),

    /// An unexpected internal condition. Always indicates a bug.
    #[error("Internal error in {handler}: {detail}")]
    Internal {
        handler: &'static str,
        detail: String,
    },

    /// A configured [`RunLimits`](crate::legacy::RunLimits) was exceeded.
    #[error("limit exceeded: {0}")]
    LimitExceeded(String),

    /// History compaction or persistence failed.
    #[error("history {operation} failed: {reason}")]
    History {
        operation: &'static str,
        reason: String,
    },

    /// A flow used as a tool produced output that did not match its declared type.
    #[error("tool '{tool}' produced invalid output for {expected}: {reason}; raw={raw}")]
    ToolOutput {
        tool: String,
        expected: String,
        reason: String,
        raw: String,
    },

    /// Flow graph construction failed.
    #[error(transparent)]
    Build(#[from] BuildError),

    /// An agent or tool produced an unrecoverable error.
    #[error(transparent)]
    Agent(#[from] AgentError),

    /// A [`MemoryFactory`](crate::legacy::MemoryFactory) returned an error during retrieval.
    #[error("memory retrieval failed for agent '{agent}': {reason}")]
    MemoryError { agent: String, reason: String },
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

    /// Agent continuation serialization failed.
    #[error("failed to serialize agent state: {0}")]
    Serialize(#[source] serde_json::Error),

    /// Agent continuation deserialization failed.
    #[error("failed to deserialize agent state: {0}")]
    Deserialize(#[source] serde_json::Error),
}
