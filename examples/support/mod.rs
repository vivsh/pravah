//! Shared error handling for runnable examples.

use pravah::GraphError;
use pravah::clients::ClientError;
use pravah::legacy::FlowError;
use thiserror::Error;

/// Structured failures that can be reported by the example programs.
#[derive(Debug, Error)]
pub(crate) enum ExampleError {
    /// A provider client operation failed.
    #[error(transparent)]
    Client(#[from] ClientError),
    /// A graph workflow operation failed.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// A compatibility-only legacy flow operation failed.
    #[error(transparent)]
    Legacy(#[from] FlowError),
    /// Local file access failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON encoding or decoding failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// An example observed an outcome outside its documented path.
    #[error("unexpected example outcome: {0}")]
    Unexpected(String),
}

impl ExampleError {
    /// Creates an error for an outcome that the example does not support.
    pub(crate) fn unexpected(message: impl Into<String>) -> Self {
        Self::Unexpected(message.into())
    }
}
