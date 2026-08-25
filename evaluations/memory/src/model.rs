use chrono::{DateTime, Utc};
use pravah_memory::MemoryId;
use serde::{Deserialize, Serialize};

/// Supported public benchmark family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetKind {
    /// SNAP Research's ten long-conversation benchmark.
    #[serde(rename = "locomo")]
    LoCoMo,
    /// Cleaned LongMemEval v1 benchmark.
    #[serde(rename = "longmemeval")]
    LongMemEval,
}

/// Application-owned normalization boundary applied before Pravah ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceGranularity {
    /// Concatenate every turn in one source session into one evidence item.
    Session,
    /// Submit every source turn as independently keyed evidence.
    Turn,
}

/// Deterministic, backend-neutral benchmark representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationDataset {
    /// Public benchmark family.
    pub kind: DatasetKind,
    /// Pinned upstream revision represented by the source file.
    pub revision: String,
    /// SHA-256 checksum expected for the unmodified source file.
    pub source_sha256: String,
    /// Evidence normalization policy.
    pub granularity: EvidenceGranularity,
    /// Non-fatal upstream annotation defects handled deterministically.
    pub normalization_warnings: Vec<String>,
    /// Isolated histories and their questions.
    pub groups: Vec<EvaluationGroup>,
}

/// One history that must remain isolated from every other benchmark history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationGroup {
    /// Stable upstream conversation or question identity.
    pub id: String,
    /// Ordered evidence submitted before this group's questions.
    pub evidence: Vec<EvaluationEvidence>,
    /// Questions evaluated only against this evidence scope.
    pub questions: Vec<EvaluationQuestion>,
}

/// One normalized application evidence submission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationEvidence {
    /// Stable, dataset-derived application evidence key.
    pub evidence_key: String,
    /// Normalized speaker-labelled textual content.
    pub content: String,
    /// Source timestamp interpreted as UTC because datasets provide no timezone.
    pub observed_at: Option<DateTime<Utc>>,
    /// Upstream session identity.
    pub source_session_id: String,
    /// Upstream turn identity when turn granularity is selected.
    pub source_turn_id: Option<String>,
}

/// One benchmark question and retrieval ground truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationQuestion {
    /// Stable upstream question identity.
    pub id: String,
    /// Query supplied to Pravah retrieval.
    pub question: String,
    /// Upstream reference answer retained for an external answer judge.
    pub expected_answer: String,
    /// Upstream category or question type.
    pub category: String,
    /// Question time interpreted as UTC when available.
    pub asked_at: Option<DateTime<Utc>>,
    /// Normalized evidence keys containing the documented answer evidence.
    pub relevant_evidence_keys: Vec<String>,
    /// Whether the benchmark expects abstention rather than retrieval evidence.
    pub abstention: bool,
}

/// Reproducibility metadata for one completed Pravah run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationRunManifest {
    /// Unique identity of this run.
    pub run_id: uuid::Uuid,
    /// Benchmark family.
    pub dataset: DatasetKind,
    /// Pinned upstream revision.
    pub dataset_revision: String,
    /// Source checksum.
    pub source_sha256: String,
    /// Evidence normalization policy.
    pub granularity: EvidenceGranularity,
    /// Opaque caller-supplied provider/configuration label.
    pub system_label: String,
    /// Search result limit.
    pub search_limit: u32,
    /// Per-channel retrieval candidate limit.
    pub candidate_limit: u32,
    /// Whether pending relations were reconciled before questions ran.
    pub reconciliation_enabled: bool,
    /// Completion time.
    pub completed_at: DateTime<Utc>,
}

/// One compact retrieved claim retained in the reproducible run artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedClaim {
    /// Immutable memory identity.
    pub memory_id: MemoryId,
    /// Source application evidence key.
    pub evidence_key: String,
    /// Immutable claim text.
    pub text: String,
    /// One-based retrieval rank.
    pub rank: u32,
    /// Hybrid fusion score.
    pub score: f64,
    /// Active corroboration support count.
    pub support_count: u32,
}

/// Retrieval observation for one benchmark question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuestionObservation {
    /// Isolated group identity.
    pub group_id: String,
    /// Stable question identity.
    pub question_id: String,
    /// Upstream category.
    pub category: String,
    /// Ground-truth evidence keys.
    pub relevant_evidence_keys: Vec<String>,
    /// Whether this item expects abstention.
    pub abstention: bool,
    /// Ranked retrieved claims.
    pub retrieved: Vec<RetrievedClaim>,
    /// End-to-end retrieval latency in microseconds.
    pub retrieval_latency_us: u64,
}

/// Complete machine-readable result of one dataset run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationRun {
    /// Reproducibility metadata.
    pub manifest: EvaluationRunManifest,
    /// Question observations in deterministic dataset order.
    pub observations: Vec<QuestionObservation>,
}
