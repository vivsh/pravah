use std::path::PathBuf;

use thiserror::Error;

/// Failures raised by deterministic adapters, Pravah runs, and ANN comparison.
#[derive(Debug, Error)]
pub enum EvaluationError {
    /// A public dataset does not satisfy its documented schema.
    #[error("invalid {dataset} dataset at {location}: {message}")]
    InvalidDataset {
        /// Dataset name used in the diagnostic.
        dataset: &'static str,
        /// Stable item or field location.
        location: String,
        /// Sanitized validation failure.
        message: String,
    },
    /// Evaluation configuration is incomplete or outside its safe bounds.
    #[error("invalid evaluation configuration: {0}")]
    InvalidConfiguration(String),
    /// A filesystem operation failed.
    #[error("evaluation file operation failed for {path}: {source}")]
    Io {
        /// File involved in the failed operation.
        path: PathBuf,
        /// Underlying I/O failure.
        source: std::io::Error,
    },
    /// JSON input or output failed.
    #[error("evaluation JSON operation failed: {0}")]
    Json(#[from] serde_json::Error),
    /// Pravah rejected ingestion, reconciliation, or retrieval.
    #[error("Pravah memory operation failed: {0}")]
    Memory(#[from] pravah_memory::MemoryManagerError),
    /// A core memory value or request was invalid.
    #[error("invalid memory evaluation value: {0}")]
    MemoryValue(#[from] pravah_memory::MemoryError),
    /// PostgreSQL or pgvector execution failed.
    #[error("PostgreSQL evaluation failed: {0}")]
    Database(#[from] sqlx::Error),
    /// PostgreSQL used a plan that cannot prove the requested comparison path.
    #[error("unexpected PostgreSQL plan for {mode}: {message}")]
    UnexpectedPlan {
        /// Requested execution path.
        mode: &'static str,
        /// Plan validation detail.
        message: String,
    },
}

impl EvaluationError {
    /// Creates a path-preserving filesystem diagnostic.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Creates a dataset diagnostic without retaining source content.
    pub fn dataset(
        dataset: &'static str,
        location: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::InvalidDataset {
            dataset,
            location: location.into(),
            message: message.into(),
        }
    }
}
