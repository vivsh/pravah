//! Shared error handling for runnable examples.

use pravah::GraphError;
use pravah::clients::ClientError;
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

impl From<String> for ExampleError {
    fn from(message: String) -> Self {
        Self::Unexpected(message)
    }
}

impl From<&str> for ExampleError {
    fn from(message: &str) -> Self {
        Self::Unexpected(message.to_owned())
    }
}
