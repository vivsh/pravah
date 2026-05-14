use thiserror::Error;

// ── Flow runtime error ───────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum FlowError {
    #[error("Node not found: {0}")]
    NotFound(String),

    #[error("Snapshot load error: {0}")]
    SnapLoadError(String),

    #[error("Snapshot store error: {0}")]
    SnapStoreError(String),

    #[error("Build error: {0}")]
    BuildError(String),

    #[error("Serialize error: {0}")]
    SerializeError(String),

    #[error("Deserialize error: {0}")]
    DeserializeError(String),

    #[error("Flow is suspended — call resume() with a resumption payload, not next()")]
    ResumeRequired,

    #[error("Flow is not suspended — unexpected resumption payload supplied")]
    UnexpectedResumption,

    #[error("Agent error: {0}")]
    AgentError(String),

    #[error("Flow deadlock: states [{0}] are waiting but no join is ready")]
    Deadlock(String),

    #[error("Flow graph is invalid:\n{}", .0.join("\n"))]
    Invalid(Vec<String>),

    #[error("Internal error: {0}")]
    Internal(String),
}

// ── Build-time errors ─────────────────────────────────────────────────────────

/// Errors that occur while constructing or validating a [`super::FlowGraph`].
///
/// All variants convert into [`super::FlowError`] via [`From`].
#[derive(Debug, Error)]
pub(crate) enum BuildError {
    /// One or more structural/semantic validation failures collected by the builder.
    #[error("Flow graph is invalid:\n{}", .0.join("\n"))]
    Invalid(Vec<String>),

    /// A node key conflict or contract mismatch detected at build time.
    #[error("Build error: {0}")]
    NodeConflict(String),

    /// Failed to deserialize a snapshot back into [`super::FlowState`].
    #[error("Snapshot load error: {0}")]
    SnapLoad(String),

    /// Failed to serialize [`super::FlowState`] into a snapshot.
    #[error("Snapshot store error: {0}")]
    SnapStore(String),
}

impl From<BuildError> for FlowError {
    fn from(e: BuildError) -> Self {
        match e {
            BuildError::Invalid(v)       => FlowError::Invalid(v),
            BuildError::NodeConflict(s)  => FlowError::BuildError(s),
            BuildError::SnapLoad(s)      => FlowError::SnapLoadError(s),
            BuildError::SnapStore(s)     => FlowError::SnapStoreError(s),
        }
    }
}

// ── Agent / tool runtime errors ───────────────────────────────────────────────

/// Errors that occur while an agent node is executing (LLM call or tool dispatch).
///
/// All variants convert into [`super::FlowError`] via [`From`].
#[derive(Debug, Error)]
pub(crate) enum AgentError {
    /// The LLM client returned an error.
    #[error("Agent error: {0}")]
    Llm(String),

    /// The LLM requested a tool that is not registered on the agent.
    #[error("Agent error: unknown tool '{0}'")]
    ToolUnknown(String),

    /// A registered tool returned a non-suspend, non-exit error.
    #[error("Agent error: {0}")]
    ToolFailed(String),

    /// Serialization failed while encoding an agent continuation or pending call.
    #[error("Serialize error: {0}")]
    Serialize(String),

    /// Deserialization failed while decoding an agent continuation.
    #[error("Deserialize error: {0}")]
    Deserialize(String),
}

impl From<AgentError> for FlowError {
    fn from(e: AgentError) -> Self {
        match e {
            AgentError::Llm(s)         => FlowError::AgentError(s),
            AgentError::ToolUnknown(s) => FlowError::AgentError(s),
            AgentError::ToolFailed(s)  => FlowError::AgentError(s),
            AgentError::Serialize(s)   => FlowError::SerializeError(s),
            AgentError::Deserialize(s) => FlowError::DeserializeError(s),
        }
    }
}
