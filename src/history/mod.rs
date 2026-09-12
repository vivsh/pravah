//! Shared agent-history persistence and compaction contracts.

mod compactor;
mod entries;
mod store;
#[cfg(test)]
mod tests;

pub use compactor::{CompactionResult, HistoryCompactor, NoopCompactor, SlidingWindowCompactor};
pub(crate) use compactor::{DynHistoryCompactor, count_complete_turns};
pub use entries::{FlowHistory, HistoryEntry};
pub(crate) use store::DynHistoryStore;
pub use store::{HistoryStore, NoopHistoryStore};
