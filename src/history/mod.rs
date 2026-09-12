//! Agent-history persistence and fallible preparation of request working memory.

pub(crate) mod compactor;
mod entries;
mod preparer;
mod store;
#[cfg(test)]
mod tests;

pub(crate) use compactor::{DynHistoryCompactor, count_complete_turns};
pub(crate) use compactor::{HistoryCompactor, NoopCompactor};
pub use entries::{FlowHistory, HistoryEntry};
pub(crate) use entries::{protected_start, validate_message_groups};
pub(crate) use preparer::DynHistoryPreparer;
pub use preparer::{HistoryPreparation, HistoryPreparer, HistoryReplacement};
pub(crate) use store::DynHistoryStore;
pub use store::{HistoryStore, NoopHistoryStore};
