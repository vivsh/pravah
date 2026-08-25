//! Isolated, reproducible memory evaluations for Pravah.
//!
//! Dataset normalization is deterministic, evaluation runs require an explicitly
//! configured [`pravah_memory::MemoryManager`], and PostgreSQL ANN comparison
//! keeps exact and HNSW execution paths separate.

mod error;
mod hnsw;
mod model;
mod runner;
mod score;

/// Adapters for pinned public memory-evaluation datasets.
pub mod datasets;

pub use error::EvaluationError;
pub use hnsw::{
    HnswCase, HnswComparator, HnswComparison, HnswComparisonOptions, HnswQuery, LatencyDistribution,
};
pub use model::{
    DatasetKind, EvaluationDataset, EvaluationEvidence, EvaluationGroup, EvaluationQuestion,
    EvaluationRun, EvaluationRunManifest, EvidenceGranularity, QuestionObservation, RetrievedClaim,
};
pub use runner::{EvaluationRunner, EvaluationRunnerBuilder};
pub use score::{CategoryScore, RetrievalScore, score_retrieval};
