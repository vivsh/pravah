#![doc = include_str!("../README.md")]

mod model;
mod providers;
mod recall;

/// Deterministic, backend-neutral selection and rendering of retrieved claims.
pub mod context;

/// Dataset-neutral quality, cost, latency, and ANN-recall evaluation primitives.
pub mod evaluation;

pub use model::{
    ClaimView, Embedding, Entity, EntityId, Evidence, EvidenceId, Memory, MemoryError, MemoryId,
    MemoryKind, MemoryLimits, ProcessingState, ReconciliationState, RelationKind, SearchRequest,
    SearchResult, SearchTimeline, SearchWeights, StalePolicy, TemporalMetadata, TemporalPrecision,
    TemporalState, ValidTime,
};
pub use providers::{
    EmbeddingProfile, EmbeddingProvider, EntityExtractionInput, EntityExtractor, ExtractedEntity,
    ExtractedMemory, ExtractionRequest, MemoryExtractor, MemoryReconciler, MemoryReranker,
    ProviderError, ReconciliationClaim, ReconciliationDecision, ReconciliationGroup,
    ReconciliationOutcome, RerankCandidate, RerankRequest, RerankResult,
};
pub use recall::{
    RecallBatch, RecallCandidate, RecallError, RecallEventKind, RecallId, RecallReceipt,
    TrackedSearch,
};

/// PostgreSQL manager, schema, and typed persistence implementation.
#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "postgres")]
mod manager;

#[cfg(feature = "postgres")]
pub use manager::{
    EvidenceIngestor, EvidenceReceipt, EvidenceSubmission, MemoryManager, MemoryManagerBuilder,
    MemoryManagerError, MemoryReconciliation, MemoryRetriever, ReconciliationMode,
};
