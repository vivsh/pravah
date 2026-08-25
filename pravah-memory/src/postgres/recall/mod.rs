//! Optional PostgreSQL recall-event recording and rebuildable analytics.

mod models;
mod schema;
mod sql;
mod store;

pub use schema::MemoryRecallSchemaExt;
pub use store::{RecallRecorder, RecallStore, RecallStoreBuilder, RecallStoreError};
