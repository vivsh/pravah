//! Compatibility re-exports for agent history compaction.

pub use crate::history::compactor::{
    CompactionResult, HistoryCompactor, NoopCompactor, SlidingWindowCompactor,
};
