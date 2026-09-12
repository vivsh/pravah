//! Compatibility re-exports for agent history compaction.

pub use crate::history::{
    CompactionResult, HistoryCompactor, NoopCompactor, SlidingWindowCompactor,
};
