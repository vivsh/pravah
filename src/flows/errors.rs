use thiserror::Error;

// ── Flow runtime error ───────────────────────────────────────────────────────

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

// ── Build-time errors ─────────────────────────────────────────────────────────

/// Errors that occur while constructing or validating a [`super::FlowGraph`].
///
/// Converts into [`super::FlowError::Build`] automatically via `#[from]`.
#[derive(Debug, Error)]
pub enum BuildError {
    /// One or more structural/semantic validation failures collected by the builder.
    #[error("Flow graph is invalid:\n{}", .0.join("\n"))]
    Invalid(Vec<String>),

    /// A node with the same key was registered more than once.
    #[error("Duplicate node: {0}")]
    DuplicateNode(String),

    /// A fork node produced a different number of child states than it declared children.
    #[error("Fork child count mismatch: {0}")]
    ChildCountMismatch(String),

    /// Failed to deserialize a snapshot back into [`super::FlowState`].
    #[error("Snapshot load error: {0}")]
    SnapshotLoad(String),

    /// Failed to serialize [`super::FlowState`] into a snapshot.
    #[error("Snapshot store error: {0}")]
    SnapshotStore(String),
}

// ── Agent / tool runtime errors ───────────────────────────────────────────────

/// Errors that occur while an agent node is executing (LLM call or tool dispatch).
///
/// Converts into [`super::FlowError::Agent`] automatically via `#[from]`.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The LLM client returned an error (factory creation, history validation, or execute call).
    #[error("Agent '{agent}' LLM call failed: {reason}")]
    LlmFailed { agent: String, reason: String },

    /// The LLM requested a tool that is not registered on the agent.
    #[error("Agent '{agent}' called unknown tool '{tool}'")]
    UnknownTool { agent: String, tool: String },

    /// The LLM issued two calls to the same tool in a single turn.
    #[error("Agent '{agent}' issued duplicate call to tool '{tool}'")]
    DuplicateToolCall { agent: String, tool: String },

    /// A registered tool returned a non-suspend, non-exit error.
    #[error("Tool '{tool}' failed: {reason}")]
    ToolFailed { tool: String, reason: String },

    /// Serialization failed while encoding an agent continuation or pending call.
    #[error("failed to serialize agent state: {0}")]
    Serialize(#[source] serde_json::Error),

    /// Deserialization failed while decoding an agent continuation or tool call.
    #[error("failed to deserialize agent state: {0}")]
    Deserialize(#[source] serde_json::Error),
}
