use std::{fmt, str::FromStr};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use thiserror::Error;
use uuid::Uuid;

macro_rules! uuid_id {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(Uuid);

        impl $name {
            /// Creates a time-ordered UUIDv7 identifier.
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }

            /// Returns the underlying UUID.
            pub const fn as_uuid(self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl FromStr for $name {
            type Err = uuid::Error;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Uuid::parse_str(value).map(Self)
            }
        }

        impl From<Uuid> for $name {
            fn from(value: Uuid) -> Self {
                Self(value)
            }
        }
    };
}

uuid_id!(
    EvidenceId,
    "Internal identity of one immutable evidence item."
);
uuid_id!(MemoryId, "Internal identity of one immutable memory claim.");
uuid_id!(
    EntityId,
    "Internal identity of one canonical scoped entity."
);

/// One immutable, normalized evidence item owned by Pravah.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// Internal UUIDv7 identity.
    pub id: EvidenceId,
    /// Application scope owning the evidence.
    pub user_key: String,
    /// Agent scope owning the evidence.
    pub agent_key: String,
    /// Application-supplied idempotency and provenance key.
    pub evidence_key: String,
    /// Immutable normalized textual content.
    pub content: String,
    /// Application metadata that Pravah does not interpret.
    pub metadata: JsonValue,
    /// Time used to resolve relative temporal language.
    pub observed_at: DateTime<Utc>,
    /// Evidence acceptance time.
    pub created_at: DateTime<Utc>,
    /// Whether the source is excluded by default from retrieval.
    pub stale: bool,
    /// Current asynchronous processing state.
    pub processing: ProcessingState,
    /// Current asynchronous reconciliation state.
    pub reconciliation: ReconciliationState,
}

/// One immutable, evidence-derived memory claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    /// Internal UUIDv7 identity.
    pub id: MemoryId,
    /// Single evidence item from which the claim was derived.
    pub evidence_id: EvidenceId,
    /// Application scope owning the claim.
    pub user_key: String,
    /// Agent scope owning the claim.
    pub agent_key: String,
    /// Stable position in the extractor output.
    pub position: u32,
    /// Concise, independently understandable claim text.
    pub text: String,
    /// Semantic class selected by the extractor.
    pub kind: MemoryKind,
    /// Evidence-supported temporal interpretation.
    pub temporal: TemporalMetadata,
    /// Extractor metadata that Pravah does not interpret.
    pub metadata: JsonValue,
    /// Materialized source staleness.
    pub stale: bool,
    /// Rebuildable hot-retrieval projection.
    pub current_for_retrieval: bool,
}

/// One canonical entity mentioned by a memory claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Internal UUIDv7 identity.
    pub id: EntityId,
    /// Stable extractor or application-compatible entity identity.
    pub entity_key: String,
    /// Entity class such as `person`, `organization`, or `product`.
    pub kind: String,
    /// Canonical display name.
    pub canonical_name: String,
    /// Alternate scoped keys or names recognized during entity retrieval.
    pub aliases: Vec<String>,
    /// Extractor metadata that Pravah does not interpret.
    pub metadata: JsonValue,
}

/// Validated dense vector in the active embedding space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Creates a non-empty vector containing only finite values.
    pub fn new(values: Vec<f32>) -> Result<Self, MemoryError> {
        if values.is_empty() {
            return Err(MemoryError::EmptyEmbedding);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(MemoryError::NonFiniteEmbedding);
        }
        Ok(Self(values))
    }

    /// Borrows values in model order.
    pub fn values(&self) -> &[f32] {
        &self.0
    }

    /// Returns the embedding dimension count.
    pub fn dimensions(&self) -> usize {
        self.0.len()
    }
}

/// Semantic memory-claim classes supported by the extraction contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Stable factual knowledge.
    Fact,
    /// A bounded occurrence.
    Event,
    /// A condition that may change over time.
    State,
    /// An intended future action.
    Plan,
    /// A preference attributable to an entity.
    Preference,
    /// A relationship between entities.
    Relationship,
}

impl MemoryKind {
    #[cfg(feature = "postgres")]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Event => "event",
            Self::State => "state",
            Self::Plan => "plan",
            Self::Preference => "preference",
            Self::Relationship => "relationship",
        }
    }
}

/// Precision attached to extractor-provided temporal values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TemporalPrecision {
    /// No reliable precision was present.
    #[default]
    Unknown,
    /// Calendar year precision.
    Year,
    /// Calendar month precision.
    Month,
    /// Calendar day precision.
    Day,
    /// Exact timestamp precision.
    Instant,
}

impl TemporalPrecision {
    #[cfg(feature = "postgres")]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Year => "year",
            Self::Month => "month",
            Self::Day => "day",
            Self::Instant => "instant",
        }
    }
}

/// Temporal state of a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TemporalState {
    /// No ongoing/completed interpretation is supported.
    #[default]
    Unspecified,
    /// The state or plan remains active.
    Ongoing,
    /// The event, state, or plan completed.
    Completed,
}

impl TemporalState {
    #[cfg(feature = "postgres")]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::Ongoing => "ongoing",
            Self::Completed => "completed",
        }
    }
}

/// Evidence-supported temporal interpretation of a claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TemporalMetadata {
    /// Inclusive start of validity.
    pub valid_from: Option<DateTime<Utc>>,
    /// Exclusive end of validity.
    pub valid_until: Option<DateTime<Utc>>,
    /// Time of a point event.
    pub event_at: Option<DateTime<Utc>>,
    /// Precision of populated temporal values.
    pub precision: TemporalPrecision,
    /// Ongoing or completed interpretation.
    pub state: TemporalState,
}

impl TemporalMetadata {
    /// Rejects reversed ranges and incompatible point-event ranges.
    pub fn validate(&self) -> Result<(), MemoryError> {
        if self
            .valid_from
            .zip(self.valid_until)
            .is_some_and(|(from, until)| from >= until)
        {
            return Err(MemoryError::InvalidTemporalRange);
        }
        if self.event_at.is_some_and(|event| {
            self.valid_from.is_some_and(|from| event < from)
                || self.valid_until.is_some_and(|until| event >= until)
        }) {
            return Err(MemoryError::EventOutsideTemporalRange);
        }
        Ok(())
    }
}

/// Durable evidence processing status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingState {
    /// Accepted and awaiting extraction.
    Pending,
    /// A worker currently owns extraction.
    Processing,
    /// Claims are persisted and searchable.
    Ready,
    /// Processing failed safely and may be retried.
    Failed,
}

/// Durable claim-reconciliation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconciliationState {
    /// No claims require reconciliation.
    NotRequired,
    /// Claims await relation classification.
    Pending,
    /// A worker currently owns reconciliation.
    Processing,
    /// All accepted claims are reconciled for the recorded scope revision.
    Ready,
    /// Reconciliation failed safely and may be retried.
    Failed,
}

/// Claim relationship created by asynchronous reconciliation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    /// Claims independently support the same knowledge.
    Corroborates,
    /// The `from` claim replaces the `to` claim after an effective time.
    Supersedes,
    /// Both evidence-supported claims cannot be jointly resolved.
    Conflicts,
}

/// Whether retrieval resolves the current relation view or exposes every claim version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ClaimView {
    /// Resolve corroboration and supersession for one coherent knowledge view.
    #[default]
    Current,
    /// Expose all immutable claim versions without supersession suppression.
    AllVersions,
}

/// Valid-time constraint applied independently from relation-version selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValidTime {
    /// Use the execution clock and the optimized current projection.
    #[default]
    Current,
    /// Apply no valid-time constraint.
    Any,
    /// Select state valid at an instant and events that had occurred by it.
    At(DateTime<Utc>),
    /// Select events occurring or intervals overlapping a half-open range.
    Between {
        /// Inclusive range start.
        start: DateTime<Utc>,
        /// Exclusive range end.
        end: DateTime<Utc>,
    },
    /// Select temporally anchored claims before an instant.
    Before(DateTime<Utc>),
    /// Select temporally anchored claims after an instant.
    After(DateTime<Utc>),
}

/// Orthogonal valid-time, transaction-time, and relation-view search controls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SearchTimeline {
    /// Relation-version view exposed to the caller.
    pub view: ClaimView,
    /// Evidence-supported valid-time constraint.
    pub valid_time: ValidTime,
    /// Optional transaction-time cutoff for claims and derived relations.
    pub known_at: Option<DateTime<Utc>>,
    /// Optional clock anchor used only for temporal-proximity ranking.
    pub reference_time: Option<DateTime<Utc>>,
}

/// Controls stale evidence participation in retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StalePolicy {
    /// Exclude stale evidence and claims.
    #[default]
    Exclude,
    /// Include current and stale claims.
    Include,
    /// Return only stale claims.
    Only,
}

/// Relative contribution of each bounded retrieval channel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SearchWeights {
    /// PostgreSQL full-text rank contribution.
    pub lexical: f64,
    /// Cosine-vector rank contribution.
    pub vector: f64,
    /// Canonical entity-link contribution.
    pub entity: f64,
    /// Explicit temporal-proximity contribution.
    pub temporal: f64,
}

impl Default for SearchWeights {
    fn default() -> Self {
        Self {
            lexical: 1.0,
            vector: 1.0,
            entity: 0.8,
            temporal: 0.7,
        }
    }
}

/// Configurable bounded hybrid-retrieval request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Natural-language search text.
    pub text: String,
    /// Maximum independently ranked claims; conflict counterparts may extend it.
    pub limit: u32,
    /// Maximum candidates requested from each channel.
    pub candidate_limit: u32,
    /// Optional canonical entity identities that bypass query extraction.
    pub entity_keys: Vec<String>,
    /// Minimum fused relevance score.
    pub minimum_fused_score: f64,
    /// Valid-time, transaction-time, and relation-view controls.
    pub timeline: SearchTimeline,
    /// Stale inclusion policy.
    pub stale: StalePolicy,
    /// Reciprocal-rank-fusion constant.
    pub reciprocal_rank_k: u32,
    /// Per-channel reciprocal-rank-fusion weights.
    pub weights: SearchWeights,
    /// Optional bounded post-fusion candidate count for provider reranking.
    pub rerank_candidate_limit: Option<u32>,
}

impl SearchRequest {
    /// Creates a current, non-stale hybrid query with conservative bounds.
    pub fn new(text: impl Into<String>) -> Result<Self, MemoryError> {
        let text = required("search text", text.into())?;
        Ok(Self {
            text,
            limit: 8,
            candidate_limit: 50,
            entity_keys: Vec::new(),
            minimum_fused_score: 0.0,
            timeline: SearchTimeline::default(),
            stale: StalePolicy::Exclude,
            reciprocal_rank_k: 60,
            weights: SearchWeights::default(),
            rerank_candidate_limit: None,
        })
    }

    /// Selects the optimized current knowledge view at the execution clock.
    pub fn current(mut self) -> Self {
        self.timeline = SearchTimeline::default();
        self
    }

    /// Resolves the current knowledge view at a supplied valid-time instant.
    pub fn as_of(mut self, at: DateTime<Utc>) -> Self {
        self.timeline.valid_time = ValidTime::At(at);
        self.timeline.reference_time = Some(at);
        self
    }

    /// Restricts knowledge to claims and relations recorded by an instant.
    pub fn known_at(mut self, at: DateTime<Utc>) -> Self {
        self.timeline.known_at = Some(at);
        self
    }

    /// Resolves valid time and transaction time independently.
    pub fn bitemporal(mut self, valid_at: DateTime<Utc>, known_at: DateTime<Utc>) -> Self {
        self.timeline.valid_time = ValidTime::At(valid_at);
        self.timeline.known_at = Some(known_at);
        self.timeline.reference_time = Some(valid_at);
        self
    }

    /// Selects claims anchored in or overlapping a half-open valid-time range.
    pub fn between(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.timeline.valid_time = ValidTime::Between { start, end };
        self.timeline.reference_time = Some(start);
        self
    }

    /// Exposes all immutable claim versions across valid time.
    pub fn history(mut self) -> Self {
        self.timeline.view = ClaimView::AllVersions;
        self.timeline.valid_time = ValidTime::Any;
        self
    }

    /// Explicitly requests optional provider reranking over this many fused candidates.
    pub fn rerank(mut self, candidate_limit: u32) -> Self {
        self.rerank_candidate_limit = Some(candidate_limit);
        self
    }
}

/// One fused result with provenance and relation annotations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Selected immutable memory claim.
    pub memory: Memory,
    /// Application evidence key without eagerly loaded evidence content.
    pub evidence_key: String,
    /// Reciprocal-rank-fusion score.
    pub score: f64,
    /// Optional provider score when explicit reranking was used.
    pub rerank_score: Option<f64>,
    /// Number of non-stale corroborating claims collapsed into this result.
    pub support_count: u32,
    /// Conflicting claim identities that should be shown together.
    pub conflicts: Vec<MemoryId>,
}

/// Limits applied before and after provider calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum accepted evidence bytes.
    pub max_evidence_bytes: usize,
    /// Maximum claims accepted from one extraction.
    pub max_memories_per_evidence: usize,
    /// Hard UTF-8 byte limit for one claim.
    pub max_memory_bytes: usize,
    /// Prompt-level extractor token target.
    pub memory_token_target: u32,
    /// Maximum entities attached to one claim.
    pub max_entities_per_memory: usize,
    /// Maximum aliases accepted for one extracted entity.
    pub max_aliases_per_entity: usize,
    /// Maximum claims in one reconciliation call.
    pub max_reconciliation_group: usize,
    /// Maximum candidates contributed by each channel for one incoming claim.
    pub reconciliation_candidates_per_claim: u32,
    /// Maximum cosine distance admitted by reconciliation vector retrieval.
    pub reconciliation_max_cosine_distance: f32,
    /// Maximum relation-component claims rebuilt inside a foreground transaction.
    pub max_foreground_projection_nodes: u32,
    /// Maximum active relation edges expanded while hydrating one retrieval.
    pub max_retrieval_relation_edges: u32,
    /// Expiring ownership duration for one evidence-processing attempt.
    pub processing_lease_seconds: u32,
    /// Expiring ownership duration for one reconciliation worker.
    pub reconciliation_lease_seconds: u32,
    /// Maximum pending evidence items accepted by one reconciliation call.
    pub max_reconciliation_batch: u32,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_evidence_bytes: 128 * 1024,
            max_memories_per_evidence: 64,
            max_memory_bytes: 2 * 1024,
            memory_token_target: 160,
            max_entities_per_memory: 32,
            max_aliases_per_entity: 16,
            max_reconciliation_group: 96,
            reconciliation_candidates_per_claim: 16,
            reconciliation_max_cosine_distance: 0.45,
            max_foreground_projection_nodes: 10_000,
            max_retrieval_relation_edges: 2_000,
            processing_lease_seconds: 900,
            reconciliation_lease_seconds: 900,
            max_reconciliation_batch: 100,
        }
    }
}

/// Errors raised by memory value validation.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// A required textual field was empty.
    #[error("{0} must not be empty")]
    Empty(&'static str),
    /// An embedding contained no values.
    #[error("embeddings must contain at least one dimension")]
    EmptyEmbedding,
    /// An embedding contained NaN or infinity.
    #[error("embeddings must contain only finite values")]
    NonFiniteEmbedding,
    /// A temporal range was empty or reversed.
    #[error("valid_from must be earlier than valid_until")]
    InvalidTemporalRange,
    /// An event time fell outside its declared validity interval.
    #[error("event_at must fall inside the declared validity interval")]
    EventOutsideTemporalRange,
}

pub(crate) fn required(name: &'static str, value: String) -> Result<String, MemoryError> {
    if value.trim().is_empty() {
        Err(MemoryError::Empty(name))
    } else {
        Ok(value)
    }
}

pub(crate) fn empty_metadata() -> JsonValue {
    JsonValue::Object(Default::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies generated durable identifiers are UUIDv7 values.
    #[test]
    fn durable_identifiers_use_uuid_v7() {
        assert_eq!(EvidenceId::new().as_uuid().get_version_num(), 7);
        assert_eq!(MemoryId::new().as_uuid().get_version_num(), 7);
    }

    /// Verifies invalid temporal ranges cannot cross the provider boundary.
    #[test]
    fn temporal_ranges_must_advance() {
        let now = Utc::now();
        let temporal = TemporalMetadata {
            valid_from: Some(now),
            valid_until: Some(now),
            ..Default::default()
        };
        assert!(matches!(
            temporal.validate(),
            Err(MemoryError::InvalidTemporalRange)
        ));
    }

    /// Verifies non-finite embedding values are rejected before persistence.
    #[test]
    fn embeddings_require_finite_values() {
        assert!(matches!(
            Embedding::new(vec![f32::NAN]),
            Err(MemoryError::NonFiniteEmbedding)
        ));
    }
}
