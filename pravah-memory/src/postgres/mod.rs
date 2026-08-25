//! PostgreSQL persistence, schema registration, and hybrid retrieval.

pub(crate) mod models;
mod projection;
pub(crate) mod repository;
mod schema;
mod search;
mod sql;

#[cfg(feature = "recall-postgres")]
mod recall;

#[cfg(feature = "recall-postgres")]
pub use recall::{
    MemoryRecallSchemaExt, RecallRecorder, RecallStore, RecallStoreBuilder, RecallStoreError,
};
pub(crate) use repository::MemoryRepository;
pub use schema::{MemoryProfile, MemorySchemaExt};
pub(crate) use search::hybrid_search;
