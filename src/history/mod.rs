//! Agent-history persistence and fallible preparation of request working memory.

mod compactor;
mod entries;
mod inspection;
pub(crate) mod legacy_compactor;
mod store;
#[cfg(test)]
mod tests;

pub(crate) use compactor::DynCompactor;
pub use compactor::{CompactionRequest, CompactionResult, Compactor};
pub use entries::{HistoryEntry, MessageHistory};
pub(crate) use entries::{protected_start, validate_message_groups};
pub(crate) use legacy_compactor::{DynHistoryCompactor, count_complete_turns};
pub(crate) use legacy_compactor::{HistoryCompactor, NoopCompactor};
pub(crate) use store::DynHistoryStore;
pub use store::{HistoryStore, NoopHistoryStore};
