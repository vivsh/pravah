use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use thiserror::Error;

use super::{Embedding, MemoryId, MemoryKind, TemporalMetadata};

/// Safe provider failure that excludes prompts, credentials, and raw responses.
#[derive(Debug, Error)]
#[error("provider operation failed ({code}): {message}")]
pub struct ProviderError {
    /// Stable machine-readable error category.
    pub code: String,
    /// Sanitized operator-facing description.
    pub message: String,
    /// Whether retrying may succeed without changing the request.
    pub retryable: bool,
}

impl ProviderError {
    /// Creates a sanitized provider failure.
    pub fn new(code: impl Into<String>, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            retryable,
        }
    }
}

/// Immutable context supplied to the one-pass memory extractor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractionRequest {
    /// Normalized textual evidence.
    pub evidence: String,
    /// Time anchoring relative language in the evidence.
    pub observed_at: DateTime<Utc>,
    /// Maximum claims accepted by Pravah.
    pub max_memories: usize,
    /// Target size of one concise claim.
    pub memory_token_target: u32,
}

/// One canonical entity emitted by extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Stable canonical key used inside one user/agent scope.
    pub entity_key: String,
    /// Entity class such as person or organization.
    pub kind: String,
    /// Canonical display name.
    pub canonical_name: String,
    /// Optional alternate stable keys or names accepted during query matching.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Optional extractor metadata.
    #[serde(default = "super::model::empty_metadata")]
    pub metadata: JsonValue,
}

/// One evidence-supported memory claim emitted in the single extraction call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedMemory {
    /// Concise, independently understandable claim.
    pub text: String,
    /// Authoritative entities, confirmed empty entities, or omitted analysis.
    pub entities: Option<Vec<ExtractedEntity>>,
    /// Semantic memory class.
    pub kind: MemoryKind,
    /// Evidence-supported temporal interpretation.
    #[serde(default)]
    pub temporal: TemporalMetadata,
    /// Optional extractor metadata.
    #[serde(default = "super::model::empty_metadata")]
    pub metadata: JsonValue,
}

/// One fallback or query entity-extraction input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityExtractionInput {
    /// Stable position used to correlate batched output.
    pub position: usize,
    /// Text to analyze.
    pub text: String,
}

/// Active embedding-space identity persisted in the singleton profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingProfile {
    /// Provider/model name.
    pub model: String,
    /// Provider revision or deployment identity.
    pub revision: String,
    /// Exact active vector dimension.
    pub dimensions: usize,
    /// Revision of the text formatting passed to the provider.
    pub document_format_revision: String,
}

/// Extracts all durable claims and optionally their entities in one LLM call.
#[async_trait]
pub trait MemoryExtractor: Send + Sync {
    /// Stable extractor revision recorded with accepted evidence.
    fn revision(&self) -> &str;

    /// Produces zero or more evidence-supported claims in source order.
    async fn extract(
        &self,
        request: ExtractionRequest,
    ) -> Result<Vec<ExtractedMemory>, ProviderError>;
}

/// Generates embeddings for a batch of claim or query texts.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Returns the immutable active embedding profile.
    fn profile(&self) -> EmbeddingProfile;

    /// Returns exactly one vector per input in input order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, ProviderError>;
}

/// Optional low-latency fallback and query entity extractor.
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Returns exactly one entity vector per input in input order.
    async fn extract(
        &self,
        inputs: &[EntityExtractionInput],
    ) -> Result<Vec<Vec<ExtractedEntity>>, ProviderError>;
}

/// Minimal immutable claim projection supplied to the reconciler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconciliationClaim {
    /// Durable claim identity.
    pub id: MemoryId,
    /// Immutable claim text.
    pub text: String,
    /// Semantic memory class.
    pub kind: MemoryKind,
    /// Temporal interpretation.
    pub temporal: TemporalMetadata,
}

/// Bounded claim group requiring semantic relation classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconciliationGroup {
    /// Newly inserted claims.
    pub new_claims: Vec<ReconciliationClaim>,
    /// Existing same-scope candidates.
    pub candidates: Vec<ReconciliationClaim>,
}

/// Structured relationship outcome returned by a reconciler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReconciliationOutcome {
    /// Claims are unrelated or insufficiently connected.
    Independent,
    /// Claims support the same knowledge.
    Corroborates,
    /// The first claim replaces the second at an optional effective time.
    Supersedes {
        /// Time after which the older claim is no longer current.
        effective_at: Option<DateTime<Utc>>,
    },
    /// Evidence-supported claims cannot be jointly resolved.
    Conflicts,
}

/// One validated classification between claim IDs in the supplied group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconciliationDecision {
    /// First supplied claim identity.
    pub from_memory_id: MemoryId,
    /// Second supplied claim identity.
    pub to_memory_id: MemoryId,
    /// Allowed relation classification.
    pub outcome: ReconciliationOutcome,
}

/// Classifies relations without rewriting or deleting immutable claims.
#[async_trait]
pub trait MemoryReconciler: Send + Sync {
    /// Stable model or policy revision recorded on derived relations.
    fn revision(&self) -> &str;

    /// Classifies a bounded connected claim group.
    async fn reconcile(
        &self,
        group: ReconciliationGroup,
    ) -> Result<Vec<ReconciliationDecision>, ProviderError>;
}

/// Candidate supplied to an optional post-fusion relevance reranker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankCandidate {
    /// Durable claim identity.
    pub id: MemoryId,
    /// Immutable claim text.
    pub text: String,
    /// Database hybrid-fusion score.
    pub fused_score: f64,
}

/// Bounded query and candidate set supplied outside the retrieval transaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankRequest {
    /// Original caller query.
    pub query: String,
    /// Candidates in database-fusion order.
    pub candidates: Vec<RerankCandidate>,
    /// Maximum independently ranked results requested by the caller.
    pub limit: u32,
}

/// One provider-scored candidate in final descending relevance order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    /// Identity from the supplied candidate set.
    pub memory_id: MemoryId,
    /// Finite provider-specific relevance score.
    pub score: f64,
}

/// Optional post-fusion reranker invoked only by explicitly reranked searches.
#[async_trait]
pub trait MemoryReranker: Send + Sync {
    /// Reranks a bounded candidate set without changing claim content.
    async fn rerank(&self, request: RerankRequest) -> Result<Vec<RerankResult>, ProviderError>;
}
