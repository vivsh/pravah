//! Backend-neutral recall receipts and validated telemetry reports.

use std::{collections::HashSet, fmt, str::FromStr};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use super::{EvidenceId, MemoryId, SearchResult};

/// Identity shared by one tracked retrieval and every outcome reported for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecallId(Uuid);

impl RecallId {
    /// Creates a time-ordered UUIDv7 recall identity.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Returns the underlying UUID.
    pub const fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for RecallId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RecallId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for RecallId {
    type Err = uuid::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(value).map(Self)
    }
}

impl From<Uuid> for RecallId {
    fn from(value: Uuid) -> Self {
        Self(value)
    }
}

/// One result included in a tracked retrieval receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecallCandidate {
    /// Retrieved immutable claim identity.
    pub memory_id: MemoryId,
    /// One-based rank in the final returned result order.
    pub rank: u32,
}

/// Immutable vocabulary for reporting outcomes against one retrieval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecallReceipt {
    /// Unique identity of this tracked retrieval.
    pub id: RecallId,
    /// Application user scope.
    pub user_key: String,
    /// Application agent scope.
    pub agent_key: String,
    /// Time at which the retrieval completed.
    pub retrieved_at: DateTime<Utc>,
    /// Ordered memory identities returned to the application.
    pub candidates: Vec<RecallCandidate>,
}

impl RecallReceipt {
    /// Creates a validated receipt at the current time.
    pub fn new(
        user_key: impl Into<String>,
        agent_key: impl Into<String>,
        candidates: Vec<RecallCandidate>,
    ) -> Result<Self, RecallError> {
        Self::at(user_key, agent_key, candidates, Utc::now())
    }

    /// Creates a receipt with an explicit completion time.
    pub fn at(
        user_key: impl Into<String>,
        agent_key: impl Into<String>,
        candidates: Vec<RecallCandidate>,
        retrieved_at: DateTime<Utc>,
    ) -> Result<Self, RecallError> {
        let receipt = Self {
            id: RecallId::new(),
            user_key: required_scope("user_key", user_key.into())?,
            agent_key: required_scope("agent_key", agent_key.into())?,
            retrieved_at,
            candidates,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    /// Validates scope, candidate identity, and rank invariants.
    pub fn validate(&self) -> Result<(), RecallError> {
        required_scope_ref("user_key", &self.user_key)?;
        required_scope_ref("agent_key", &self.agent_key)?;
        validate_candidates(&self.candidates)
    }
}

/// A tracked retrieval whose results are otherwise identical to untracked search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackedSearch {
    /// Receipt used by optional outcome reports.
    pub receipt: RecallReceipt,
    /// Ordered structured search results.
    pub results: Vec<SearchResult>,
}

/// Observe-only recall event classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallEventKind {
    /// A sampled result was returned by retrieval.
    Retrieved,
    /// The application explicitly selected a result.
    Accepted,
    /// The application actually consumed a result.
    Used,
    /// The application explicitly rejected a result.
    Dismissed,
    /// Separately accepted evidence corrects a result.
    Corrected,
}

impl RecallEventKind {
    #[cfg(feature = "recall-postgres")]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Retrieved => "retrieved",
            Self::Accepted => "accepted",
            Self::Used => "used",
            Self::Dismissed => "dismissed",
            Self::Corrected => "corrected",
        }
    }
}

/// A validated, retry-safe group of recall events for one receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct RecallBatch {
    user_key: String,
    agent_key: String,
    recall_id: RecallId,
    events: Vec<RecallEvent>,
}

impl RecallBatch {
    /// Samples retrieved candidates deterministically at the supplied probability.
    ///
    /// Probability zero creates an empty batch, allowing durable retrieval telemetry
    /// to be disabled without a special caller branch.
    pub fn retrieved(
        receipt: &RecallReceipt,
        sample_probability: f64,
    ) -> Result<Self, RecallError> {
        receipt.validate()?;
        validate_probability(sample_probability)?;
        let events = sampled_retrieval_events(receipt, sample_probability);
        Ok(Self::from_events(receipt, events))
    }

    /// Reports claims explicitly selected by the application.
    pub fn accepted(
        receipt: &RecallReceipt,
        memory_ids: impl IntoIterator<Item = MemoryId>,
    ) -> Result<Self, RecallError> {
        Self::outcomes(receipt, memory_ids, RecallEventKind::Accepted)
    }

    /// Reports claims actually consumed by the application.
    pub fn used(
        receipt: &RecallReceipt,
        memory_ids: impl IntoIterator<Item = MemoryId>,
    ) -> Result<Self, RecallError> {
        Self::outcomes(receipt, memory_ids, RecallEventKind::Used)
    }

    /// Reports claims explicitly rejected by the application.
    pub fn dismissed(
        receipt: &RecallReceipt,
        memory_ids: impl IntoIterator<Item = MemoryId>,
    ) -> Result<Self, RecallError> {
        Self::outcomes(receipt, memory_ids, RecallEventKind::Dismissed)
    }

    /// Reports claims corrected by separately accepted evidence.
    pub fn corrected(
        receipt: &RecallReceipt,
        corrections: impl IntoIterator<Item = (MemoryId, EvidenceId)>,
    ) -> Result<Self, RecallError> {
        receipt.validate()?;
        let mut seen = HashSet::new();
        let membership = receipt_membership(receipt);
        let mut events = Vec::new();
        for (memory_id, evidence_id) in corrections {
            validate_reported_id(memory_id, &membership, &mut seen)?;
            events.push(RecallEvent::explicit(
                memory_id,
                RecallEventKind::Corrected,
                Some(evidence_id),
            ));
        }
        Ok(Self::from_events(receipt, events))
    }

    /// Returns the number of durable events in this batch.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns whether sampling or an empty outcome selection produced no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    fn outcomes(
        receipt: &RecallReceipt,
        memory_ids: impl IntoIterator<Item = MemoryId>,
        kind: RecallEventKind,
    ) -> Result<Self, RecallError> {
        receipt.validate()?;
        let mut seen = HashSet::new();
        let membership = receipt_membership(receipt);
        let mut events = Vec::new();
        for memory_id in memory_ids {
            validate_reported_id(memory_id, &membership, &mut seen)?;
            events.push(RecallEvent::explicit(memory_id, kind, None));
        }
        Ok(Self::from_events(receipt, events))
    }

    fn from_events(receipt: &RecallReceipt, events: Vec<RecallEvent>) -> Self {
        Self {
            user_key: receipt.user_key.clone(),
            agent_key: receipt.agent_key.clone(),
            recall_id: receipt.id,
            events,
        }
    }

    #[cfg(feature = "recall-postgres")]
    pub(crate) fn user_key(&self) -> &str {
        &self.user_key
    }

    #[cfg(feature = "recall-postgres")]
    pub(crate) fn agent_key(&self) -> &str {
        &self.agent_key
    }

    #[cfg(feature = "recall-postgres")]
    pub(crate) const fn recall_id(&self) -> RecallId {
        self.recall_id
    }

    #[cfg(any(feature = "recall-postgres", test))]
    pub(crate) fn events(&self) -> &[RecallEvent] {
        &self.events
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecallEvent {
    pub(crate) id: Uuid,
    pub(crate) memory_id: MemoryId,
    pub(crate) kind: RecallEventKind,
    pub(crate) rank: Option<u32>,
    pub(crate) occurred_at: DateTime<Utc>,
    pub(crate) correction_evidence_id: Option<EvidenceId>,
    pub(crate) sample_probability: f64,
}

impl RecallEvent {
    fn retrieved(candidate: RecallCandidate, occurred_at: DateTime<Utc>, probability: f64) -> Self {
        Self {
            id: Uuid::now_v7(),
            memory_id: candidate.memory_id,
            kind: RecallEventKind::Retrieved,
            rank: Some(candidate.rank),
            occurred_at,
            correction_evidence_id: None,
            sample_probability: probability,
        }
    }

    fn explicit(
        memory_id: MemoryId,
        kind: RecallEventKind,
        correction_evidence_id: Option<EvidenceId>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            memory_id,
            kind,
            rank: None,
            occurred_at: Utc::now(),
            correction_evidence_id,
            sample_probability: 1.0,
        }
    }
}

/// Validation failures raised before telemetry reaches a durable backend.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum RecallError {
    /// A required scope key was empty.
    #[error("{0} must not be empty")]
    EmptyScope(&'static str),
    /// Receipt candidates repeated an identity.
    #[error("recall receipt contains duplicate {0}")]
    DuplicateReceiptField(&'static str),
    /// Receipt ranks must be one-based.
    #[error("recall candidate rank must be greater than zero")]
    InvalidRank,
    /// Receipt ranks must exactly match their one-based result positions.
    #[error("recall candidate ranks must be contiguous and match result order")]
    InvalidRankOrder,
    /// A reported memory was not returned by this receipt.
    #[error("reported memory {0} does not belong to the recall receipt")]
    ForeignMemory(MemoryId),
    /// A report repeated one memory identity.
    #[error("reported memory {0} occurs more than once")]
    DuplicateMemory(MemoryId),
    /// Retrieved sampling must be finite and within the closed unit interval.
    #[error("retrieved sample probability must be finite and between zero and one")]
    InvalidSampleProbability,
}

fn validate_candidates(candidates: &[RecallCandidate]) -> Result<(), RecallError> {
    let mut ids = HashSet::with_capacity(candidates.len());
    for (position, candidate) in candidates.iter().enumerate() {
        if candidate.rank == 0 {
            return Err(RecallError::InvalidRank);
        }
        if usize::try_from(candidate.rank).ok() != position.checked_add(1) {
            return Err(RecallError::InvalidRankOrder);
        }
        if !ids.insert(candidate.memory_id) {
            return Err(RecallError::DuplicateReceiptField("memory identity"));
        }
    }
    Ok(())
}

fn required_scope(name: &'static str, value: String) -> Result<String, RecallError> {
    required_scope_ref(name, &value)?;
    Ok(value)
}

fn required_scope_ref(name: &'static str, value: &str) -> Result<(), RecallError> {
    if value.trim().is_empty() {
        Err(RecallError::EmptyScope(name))
    } else {
        Ok(())
    }
}

fn validate_probability(probability: f64) -> Result<(), RecallError> {
    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(RecallError::InvalidSampleProbability)
    }
}

fn receipt_membership(receipt: &RecallReceipt) -> HashSet<MemoryId> {
    receipt
        .candidates
        .iter()
        .map(|candidate| candidate.memory_id)
        .collect()
}

fn validate_reported_id(
    memory_id: MemoryId,
    membership: &HashSet<MemoryId>,
    seen: &mut HashSet<MemoryId>,
) -> Result<(), RecallError> {
    if !membership.contains(&memory_id) {
        return Err(RecallError::ForeignMemory(memory_id));
    }
    if !seen.insert(memory_id) {
        return Err(RecallError::DuplicateMemory(memory_id));
    }
    Ok(())
}

fn sampled_retrieval_events(receipt: &RecallReceipt, probability: f64) -> Vec<RecallEvent> {
    if probability == 0.0 {
        return Vec::new();
    }
    receipt
        .candidates
        .iter()
        .copied()
        .filter(|candidate| sampled(receipt.id, candidate.memory_id, probability))
        .map(|candidate| RecallEvent::retrieved(candidate, receipt.retrieved_at, probability))
        .collect()
}

fn sampled(recall_id: RecallId, memory_id: MemoryId, probability: f64) -> bool {
    if probability == 1.0 {
        return true;
    }
    let recall = recall_id.as_uuid().as_u128();
    let memory = memory_id.as_uuid().as_u128();
    let mixed = recall ^ memory.rotate_left(61) ^ 0x9e37_79b9_7f4a_7c15_6a09_e667_f3bc_c909;
    let bucket = (mixed ^ (mixed >> 64)) as u64;
    (bucket as f64) / (u64::MAX as f64) < probability
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt() -> RecallReceipt {
        RecallReceipt::new(
            "user",
            "agent",
            vec![
                RecallCandidate {
                    memory_id: MemoryId::new(),
                    rank: 1,
                },
                RecallCandidate {
                    memory_id: MemoryId::new(),
                    rank: 2,
                },
            ],
        )
        .expect("valid receipt")
    }

    /// Verifies zero sampling disables durable retrieved events without rejecting the report.
    #[test]
    fn zero_retrieved_sampling_creates_empty_batch() {
        let batch = RecallBatch::retrieved(&receipt(), 0.0).expect("valid disabled sampling");

        assert!(batch.is_empty());
    }

    /// Verifies full sampling retains every candidate and its one-based rank.
    #[test]
    fn full_retrieved_sampling_keeps_every_candidate() {
        let receipt = receipt();
        let batch = RecallBatch::retrieved(&receipt, 1.0).expect("valid full sampling");

        assert_eq!(batch.len(), receipt.candidates.len());
        assert_eq!(batch.events()[0].rank, Some(1));
        assert_eq!(batch.events()[1].rank, Some(2));
    }

    /// Verifies deterministic sampling makes retrying one receipt produce the same identities.
    #[test]
    fn retrieved_sampling_is_deterministic_for_receipt() {
        let receipt = receipt();
        let first = RecallBatch::retrieved(&receipt, 0.5).expect("valid sampling");
        let second = RecallBatch::retrieved(&receipt, 0.5).expect("valid sampling");
        let first_ids = first
            .events()
            .iter()
            .map(|event| event.memory_id)
            .collect::<Vec<_>>();
        let second_ids = second
            .events()
            .iter()
            .map(|event| event.memory_id)
            .collect::<Vec<_>>();

        assert_eq!(first_ids, second_ids);
    }

    /// Verifies explicit outcomes reject identities outside the receipt.
    #[test]
    fn outcomes_reject_foreign_memories() {
        let receipt = receipt();
        let foreign = MemoryId::new();
        let result = RecallBatch::used(&receipt, [foreign]);

        assert_eq!(result, Err(RecallError::ForeignMemory(foreign)));
    }

    /// Verifies explicit outcomes reject duplicate claim identities.
    #[test]
    fn outcomes_reject_duplicate_memories() {
        let receipt = receipt();
        let memory_id = receipt.candidates[0].memory_id;
        let result = RecallBatch::accepted(&receipt, [memory_id, memory_id]);

        assert_eq!(result, Err(RecallError::DuplicateMemory(memory_id)));
    }

    /// Verifies correction events retain the separately accepted evidence identity.
    #[test]
    fn corrected_reports_require_typed_evidence_identity() {
        let receipt = receipt();
        let memory_id = receipt.candidates[0].memory_id;
        let evidence_id = EvidenceId::new();
        let batch =
            RecallBatch::corrected(&receipt, [(memory_id, evidence_id)]).expect("valid correction");

        assert_eq!(batch.events()[0].correction_evidence_id, Some(evidence_id));
    }

    /// Verifies malformed receipt rank order is rejected before reports are constructed.
    #[test]
    fn receipt_rejects_duplicate_ranks() {
        let memory_ids = [MemoryId::new(), MemoryId::new()];
        let result = RecallReceipt::new(
            "user",
            "agent",
            vec![
                RecallCandidate {
                    memory_id: memory_ids[0],
                    rank: 1,
                },
                RecallCandidate {
                    memory_id: memory_ids[1],
                    rank: 1,
                },
            ],
        );

        assert_eq!(result, Err(RecallError::InvalidRankOrder));
    }
}
