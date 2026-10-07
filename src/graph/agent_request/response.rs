//! Portable completions and acknowledgements, delivered atomically to the owning VM.

use crate::graph::Value;
use crate::history::{CompactionResult, HistoryEntry};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Portable worker failure. Its diagnostics may contain private provider content.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentError {
    code: String,
    message: String,
    details: Option<Value>,
}

impl AgentError {
    /// Constructs a failure without automatically displaying its diagnostic message.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }
    /// Attaches explicitly inspectable portable diagnostics.
    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }
    /// Borrows the failure classification.
    pub fn code(&self) -> &str {
        &self.code
    }
    /// Borrows potentially sensitive diagnostic text.
    pub fn message(&self) -> &str {
        &self.message
    }
    /// Borrows potentially sensitive provider diagnostics.
    pub fn details(&self) -> Option<&Value> {
        self.details.as_ref()
    }
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("agent operation failed")
    }
}
impl fmt::Debug for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentError").finish_non_exhaustive()
    }
}
impl std::error::Error for AgentError {}

/// One completed operation, including successful stages preceding a later failure.
/// Workers own this completion, never the runtime's history or lifecycle.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentResponse {
    pub(crate) version: u32,
    pub(crate) id: Uuid,
    pub(crate) outcome: Result<Value, AgentError>,
    pub(crate) persisted_through: Option<u64>,
    pub(crate) loaded: Option<(String, Vec<HistoryEntry>)>,
    pub(crate) compaction: Option<(String, CompactionResult)>,
}

impl AgentResponse {
    /// Creates a worker completion without history acknowledgements.
    pub fn new(id: Uuid, outcome: Result<Value, AgentError>) -> Self {
        Self {
            version: super::AGENT_REQUEST_VERSION,
            id,
            outcome,
            persisted_through: None,
            loaded: None,
            compaction: None,
        }
    }
    /// Returns the exact request identity this completion belongs to.
    pub fn id(&self) -> Uuid {
        self.id
    }
    /// Borrows the portable outcome without decoding or displaying its contents.
    pub fn outcome(&self) -> Result<&Value, &AgentError> {
        self.outcome.as_ref()
    }
}

impl fmt::Debug for AgentResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentResponse")
            .field("id", &self.id)
            .field("success", &self.outcome.is_ok())
            .finish_non_exhaustive()
    }
}
