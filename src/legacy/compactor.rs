//! Compatibility re-exports for agent history compaction.

pub use crate::history::legacy_compactor::{
    CompactionResult, HistoryCompactor, NoopCompactor, SlidingWindowCompactor,
};
