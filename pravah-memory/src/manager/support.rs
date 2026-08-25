use super::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::{Embedding, ExtractedEntity, ExtractedMemory, MemoryId, ReconciliationOutcome};

pub(super) fn analyze_temporal_query(request: &mut SearchRequest, now: DateTime<Utc>) {
    if request.timeline != crate::SearchTimeline::default() {
        return;
    }
    let text = request.text.to_lowercase();
    if text.contains("all time") || text.contains(" ever ") {
        request.timeline.view = crate::ClaimView::AllVersions;
        request.timeline.valid_time = crate::ValidTime::Any;
    } else if text.contains("historical") || text.contains("in the past") {
        request.timeline.valid_time = crate::ValidTime::Before(now);
    } else if text.contains("yesterday") {
        request.timeline.valid_time = crate::ValidTime::At(now - chrono::Duration::days(1));
    } else if text.contains("tomorrow") {
        request.timeline.valid_time = crate::ValidTime::At(now + chrono::Duration::days(1));
    } else if text.contains("as of")
        && let Some(at) = date_in_query(&text)
    {
        request.timeline.valid_time = crate::ValidTime::At(at);
    }
    request.timeline.reference_time = Some(match request.timeline.valid_time {
        crate::ValidTime::At(at) | crate::ValidTime::Before(at) | crate::ValidTime::After(at) => at,
        crate::ValidTime::Between { start, .. } => start,
        crate::ValidTime::Current | crate::ValidTime::Any => now,
    });
}

fn date_in_query(text: &str) -> Option<DateTime<Utc>> {
    text.split_whitespace().find_map(|token| {
        let token =
            token.trim_matches(|character: char| !character.is_ascii_digit() && character != '-');
        let date = chrono::NaiveDate::parse_from_str(token, "%Y-%m-%d").ok()?;
        let naive = date.and_hms_opt(0, 0, 0)?;
        Some(DateTime::from_naive_utc_and_offset(naive, Utc))
    })
}

impl MemoryManagerError {
    pub(super) fn safe_code(&self) -> String {
        match self {
            Self::Provider(error) => error.code.clone(),
            Self::InvalidInput(_) | Self::Memory(_) => "invalid_provider_output".to_owned(),
            Self::Recall(_) => "invalid_recall_receipt".to_owned(),
            Self::ProfileMismatch(_) => "profile_mismatch".to_owned(),
            Self::Persistence(_) => "persistence_failed".to_owned(),
            Self::ReconciliationRace => "reconciliation_race".to_owned(),
            Self::RelationExpansionLimit => "relation_expansion_limit".to_owned(),
            Self::ProcessingSuperseded => "processing_superseded".to_owned(),
            Self::EvidenceKeyConflict => "evidence_key_conflict".to_owned(),
            Self::InvalidConfiguration(_)
            | Self::EvidenceNotFound
            | Self::ReconciliationDisabled
            | Self::RerankerDisabled => "memory_failed".to_owned(),
        }
    }
}

impl From<RepositoryError> for MemoryManagerError {
    fn from(error: RepositoryError) -> Self {
        match error {
            RepositoryError::EvidenceKeyConflict => Self::EvidenceKeyConflict,
            RepositoryError::EvidenceNotFound => Self::EvidenceNotFound,
            RepositoryError::ScopeRevisionChanged => Self::ReconciliationRace,
            RepositoryError::ProcessingSuperseded | RepositoryError::EvidenceStale => {
                Self::ProcessingSuperseded
            }
            RepositoryError::ReconciliationBusy => Self::ReconciliationRace,
            RepositoryError::ReconciliationSuperseded => Self::ReconciliationRace,
            RepositoryError::RelationExpansionLimit => Self::RelationExpansionLimit,
            other => Self::Persistence(other.to_string()),
        }
    }
}

pub(super) fn receipt(
    evidence: &EvidenceRow,
    memory_count: usize,
    accepted: bool,
    reconciliation_pending: bool,
) -> EvidenceReceipt {
    EvidenceReceipt {
        evidence_id: EvidenceId::from(evidence.id),
        evidence_key: evidence.evidence_key.clone(),
        memory_count,
        accepted,
        processing: match evidence.processing_state.as_str() {
            "pending" => crate::ProcessingState::Pending,
            "processing" => crate::ProcessingState::Processing,
            "ready" => crate::ProcessingState::Ready,
            _ => crate::ProcessingState::Failed,
        },
        stale: evidence.stale,
        reconciliation_pending,
    }
}

pub(super) fn validate_profile(profile: &crate::EmbeddingProfile, errors: &mut Vec<String>) {
    if profile.model.trim().is_empty() {
        errors.push("embedding model must not be empty".to_owned());
    }
    if profile.revision.trim().is_empty() {
        errors.push("embedding revision must not be empty".to_owned());
    }
    if profile.document_format_revision.trim().is_empty() {
        errors.push("document format revision must not be empty".to_owned());
    }
    if profile.dimensions == 0 || profile.dimensions > 2_000 {
        errors.push("embedding dimensions must be between 1 and 2000 for vector HNSW".to_owned());
    }
}

pub(super) fn validate_database_profile(
    stored: &crate::postgres::models::MemoryProfileRow,
    active: &crate::EmbeddingProfile,
    extractor_revision: &str,
    reconciler: Option<&dyn MemoryReconciler>,
) -> Result<(), MemoryManagerError> {
    let active_reconciler = reconciler.map_or(stored.reconciler_revision.as_str(), |value| {
        value.revision()
    });
    let matches = stored.embedding_model == active.model
        && stored.schema_version == 2
        && stored.derived_index_revision == 2
        && stored.embedding_revision == active.revision
        && stored.embedding_dimensions == active.dimensions as i32
        && stored.document_format_revision == active.document_format_revision
        && stored.extractor_revision == extractor_revision
        && stored.reconciler_revision == active_reconciler
        && !stored.text_search_configuration.trim().is_empty()
        && stored.distance_metric == "cosine";
    if !matches {
        return Err(MemoryManagerError::ProfileMismatch(
            "provider identities or dimensions differ from the migrated profile".to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn validate_search(request: &SearchRequest) -> Result<(), MemoryManagerError> {
    validate_key("search text", request.text.clone())?;
    validate_search_bounds(request)?;
    validate_search_semantics(request)?;
    validate_search_weights(request)
}

fn validate_search_bounds(request: &SearchRequest) -> Result<(), MemoryManagerError> {
    if request.limit == 0 || request.candidate_limit == 0 || request.reciprocal_rank_k == 0 {
        return Err(MemoryManagerError::InvalidInput(
            "search limits and RRF constant must be positive".to_owned(),
        ));
    }
    if request.limit > 100 || request.candidate_limit > 500 {
        return Err(MemoryManagerError::InvalidInput(
            "search limit must not exceed 100 and candidate limit must not exceed 500".to_owned(),
        ));
    }
    if request.entity_keys.len() > 64 {
        return Err(MemoryManagerError::InvalidInput(
            "search accepts at most 64 explicit entity keys".to_owned(),
        ));
    }
    if request
        .rerank_candidate_limit
        .is_some_and(|limit| limit < request.limit || limit > request.candidate_limit)
    {
        return Err(MemoryManagerError::InvalidInput(
            "rerank candidate limit must be between result and candidate limits".to_owned(),
        ));
    }
    Ok(())
}

fn validate_search_semantics(request: &SearchRequest) -> Result<(), MemoryManagerError> {
    if !request.minimum_fused_score.is_finite() || request.minimum_fused_score < 0.0 {
        return Err(MemoryManagerError::InvalidInput(
            "minimum fused score must be finite and non-negative".to_owned(),
        ));
    }
    if matches!(
        request.timeline.valid_time,
        crate::ValidTime::Between { start, end } if start >= end
    ) {
        return Err(MemoryManagerError::InvalidInput(
            "temporal range start must be earlier than end".to_owned(),
        ));
    }
    Ok(())
}

fn validate_search_weights(request: &SearchRequest) -> Result<(), MemoryManagerError> {
    let weights = [
        request.weights.lexical,
        request.weights.vector,
        request.weights.entity,
        request.weights.temporal,
    ];
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || weights.iter().all(|weight| *weight == 0.0)
    {
        return Err(MemoryManagerError::InvalidInput(
            "search weights must be finite, non-negative, and not all zero".to_owned(),
        ));
    }
    Ok(())
}

/// Validates provider order, identities, scores, and the caller's final bound.
pub(super) fn apply_reranking(
    results: Vec<SearchResult>,
    ranked: Vec<crate::RerankResult>,
    limit: u32,
) -> Result<Vec<SearchResult>, MemoryManagerError> {
    if ranked.len() > limit as usize {
        return Err(MemoryManagerError::InvalidInput(
            "reranker returned more results than requested".to_owned(),
        ));
    }
    let conflicts = results
        .iter()
        .map(|result| (result.memory.id, result.conflicts.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut by_id = results
        .into_iter()
        .enumerate()
        .map(|(position, result)| (result.memory.id, (position, result)))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut selected = ranked
        .into_iter()
        .map(|ranked| {
            if !ranked.score.is_finite() || !seen.insert(ranked.memory_id) {
                return Err(MemoryManagerError::InvalidInput(
                    "reranker returned a non-finite score or duplicate id".to_owned(),
                ));
            }
            let (_, mut result) = by_id.remove(&ranked.memory_id).ok_or_else(|| {
                MemoryManagerError::InvalidInput(
                    "reranker referenced an unknown candidate id".to_owned(),
                )
            })?;
            result.rerank_score = Some(ranked.score);
            Ok(result)
        })
        .collect::<Result<Vec<_>, _>>()?;
    append_conflict_components(&mut selected, &mut by_id, &conflicts)?;
    Ok(selected)
}

/// Restores every connected conflict side in original retrieval order.
fn append_conflict_components(
    selected: &mut Vec<SearchResult>,
    remaining: &mut BTreeMap<MemoryId, (usize, SearchResult)>,
    conflicts: &BTreeMap<MemoryId, Vec<MemoryId>>,
) -> Result<(), MemoryManagerError> {
    let mut reached = selected
        .iter()
        .map(|result| result.memory.id)
        .collect::<BTreeSet<_>>();
    let mut pending = reached.iter().copied().collect::<VecDeque<_>>();
    while let Some(memory_id) = pending.pop_front() {
        for conflict_id in conflicts.get(&memory_id).into_iter().flatten() {
            if !conflicts.contains_key(conflict_id) {
                return Err(MemoryManagerError::InvalidInput(
                    "retrieval returned an incomplete conflict component".to_owned(),
                ));
            }
            if reached.insert(*conflict_id) {
                pending.push_back(*conflict_id);
            }
        }
    }
    let mut additions = reached
        .into_iter()
        .filter_map(|id| remaining.remove(&id))
        .collect::<Vec<_>>();
    additions.sort_by_key(|(position, _)| *position);
    selected.extend(additions.into_iter().map(|(_, result)| result));
    Ok(())
}

pub(super) fn validate_limits(limits: &MemoryLimits, errors: &mut Vec<String>) {
    if limits.max_evidence_bytes == 0 {
        errors.push("max evidence bytes must be positive".to_owned());
    }
    if limits.max_memories_per_evidence == 0 {
        errors.push("max memories per evidence must be positive".to_owned());
    }
    if limits.max_memory_bytes == 0 {
        errors.push("max memory bytes must be positive".to_owned());
    }
    if limits.memory_token_target == 0 {
        errors.push("memory token target must be positive".to_owned());
    }
    if limits.max_reconciliation_group <= limits.max_memories_per_evidence
        || limits.max_reconciliation_group > 1_000
    {
        errors.push(
            "max reconciliation group must exceed memories per evidence and not exceed 1000"
                .to_owned(),
        );
    }
    if limits.reconciliation_candidates_per_claim == 0
        || limits.reconciliation_candidates_per_claim > 100
    {
        errors.push("reconciliation candidates per claim must be between 1 and 100".to_owned());
    }
    if !limits.reconciliation_max_cosine_distance.is_finite()
        || !(0.0..=2.0).contains(&limits.reconciliation_max_cosine_distance)
    {
        errors.push("reconciliation cosine distance must be finite and between 0 and 2".to_owned());
    }
    if limits.max_foreground_projection_nodes == 0 {
        errors.push("foreground projection node limit must be positive".to_owned());
    }
    if limits.max_retrieval_relation_edges == 0 {
        errors.push("retrieval relation edge limit must be positive".to_owned());
    }
    if limits.max_aliases_per_entity > 64 {
        errors.push("entity alias limit must not exceed 64".to_owned());
    }
    if !(30..=86_400).contains(&limits.processing_lease_seconds) {
        errors.push("processing lease must be between 30 seconds and one day".to_owned());
    }
    if !(30..=86_400).contains(&limits.reconciliation_lease_seconds) {
        errors.push("reconciliation lease must be between 30 seconds and one day".to_owned());
    }
    if limits.max_reconciliation_batch == 0 || limits.max_reconciliation_batch > 10_000 {
        errors.push("reconciliation batch limit must be between 1 and 10000".to_owned());
    }
}

pub(super) fn validate_decisions(
    decisions: &[ReconciliationDecision],
    allowed: &BTreeSet<MemoryId>,
    new_ids: &BTreeSet<MemoryId>,
) -> Result<(), MemoryManagerError> {
    let mut seen = BTreeSet::new();
    for decision in decisions {
        let from_new = new_ids.contains(&decision.from_memory_id);
        let to_new = new_ids.contains(&decision.to_memory_id);
        let pair = if decision.from_memory_id < decision.to_memory_id {
            (decision.from_memory_id, decision.to_memory_id)
        } else {
            (decision.to_memory_id, decision.from_memory_id)
        };
        if decision.from_memory_id == decision.to_memory_id
            || !allowed.contains(&decision.from_memory_id)
            || !allowed.contains(&decision.to_memory_id)
            || from_new == to_new
            || matches!(
                decision.outcome,
                crate::ReconciliationOutcome::Supersedes { .. }
            ) && !from_new
            || !seen.insert(pair)
        {
            return Err(MemoryManagerError::InvalidInput(
                "reconciler referenced an invalid claim id".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Runs one bounded provider call for the ambiguous connected candidate group.
pub(super) async fn semantic_relations(
    reconciler: &dyn MemoryReconciler,
    ambiguous: (Vec<&MemoryRow>, Vec<&MemoryRow>),
) -> Result<Vec<ReconciliationDecision>, MemoryManagerError> {
    if ambiguous.0.is_empty() || ambiguous.1.is_empty() {
        return Ok(Vec::new());
    }
    reconcile_group(reconciler, ambiguous.0, ambiguous.1).await
}

/// Validates one semantic relation group and its provider classifications.
async fn reconcile_group(
    reconciler: &dyn MemoryReconciler,
    claims: Vec<&MemoryRow>,
    candidates: Vec<&MemoryRow>,
) -> Result<Vec<ReconciliationDecision>, MemoryManagerError> {
    let group = ReconciliationGroup {
        new_claims: claims
            .iter()
            .map(reconciliation_claim)
            .collect::<Result<_, _>>()?,
        candidates: candidates
            .iter()
            .map(reconciliation_claim)
            .collect::<Result<_, _>>()?,
    };
    let allowed = group
        .new_claims
        .iter()
        .chain(&group.candidates)
        .map(|claim| claim.id)
        .collect::<BTreeSet<_>>();
    let new_ids = group
        .new_claims
        .iter()
        .map(|claim| claim.id)
        .collect::<BTreeSet<_>>();
    let decisions = reconciler.reconcile(group).await?;
    validate_decisions(&decisions, &allowed, &new_ids)?;
    Ok(decisions)
}

fn reconciliation_claim(row: &&MemoryRow) -> Result<ReconciliationClaim, MemoryManagerError> {
    let memory = crate::postgres::repository::memory_from_row((*row).clone())?;
    Ok(ReconciliationClaim {
        id: memory.id,
        text: memory.text,
        kind: memory.kind,
        temporal: memory.temporal,
    })
}

pub(super) fn validate_key(
    name: &'static str,
    value: String,
) -> Result<String, MemoryManagerError> {
    if value.trim().is_empty() {
        return Err(MemoryManagerError::InvalidInput(format!(
            "{name} must not be empty"
        )));
    }
    if value.len() > 512 {
        return Err(MemoryManagerError::InvalidInput(format!(
            "{name} exceeds 512 bytes"
        )));
    }
    Ok(value)
}

pub(super) fn validate_content(value: String) -> Result<String, MemoryManagerError> {
    if value.trim().is_empty() {
        return Err(MemoryManagerError::InvalidInput(
            "evidence content must not be empty".to_owned(),
        ));
    }
    Ok(value)
}

pub(super) fn validate_submission(
    submission: &EvidenceSubmission,
    limits: &MemoryLimits,
) -> Result<(), MemoryManagerError> {
    if submission.content.len() > limits.max_evidence_bytes {
        return Err(MemoryManagerError::InvalidInput(format!(
            "evidence exceeds {} bytes; chunk it in the application",
            limits.max_evidence_bytes
        )));
    }
    if !submission.metadata.is_object() {
        return Err(MemoryManagerError::InvalidInput(
            "evidence metadata must be a JSON object".to_owned(),
        ));
    }
    Ok(())
}

/// Enforces bounded, independently understandable extractor output.
pub(super) fn validate_extracted(
    extracted: &[ExtractedMemory],
    limits: &MemoryLimits,
) -> Result<(), MemoryManagerError> {
    if extracted.len() > limits.max_memories_per_evidence {
        return Err(MemoryManagerError::InvalidInput(
            "extractor returned too many memories".to_owned(),
        ));
    }
    for memory in extracted {
        validate_extracted_memory(memory, limits)?;
    }
    Ok(())
}

fn validate_extracted_memory(
    memory: &ExtractedMemory,
    limits: &MemoryLimits,
) -> Result<(), MemoryManagerError> {
    if memory.text.trim().is_empty() || memory.text.len() > limits.max_memory_bytes {
        return Err(MemoryManagerError::InvalidInput(
            "extractor returned an empty or oversized memory".to_owned(),
        ));
    }
    memory.temporal.validate()?;
    if !memory.metadata.is_object() {
        return Err(MemoryManagerError::InvalidInput(
            "memory metadata must be a JSON object".to_owned(),
        ));
    }
    if let Some(entities) = &memory.entities {
        if entities.len() > limits.max_entities_per_memory {
            return Err(MemoryManagerError::InvalidInput(
                "extractor returned too many entities".to_owned(),
            ));
        }
        validate_entities(entities, limits.max_aliases_per_entity)?;
    }
    Ok(())
}

fn validate_entities(
    entities: &[ExtractedEntity],
    max_aliases: usize,
) -> Result<(), MemoryManagerError> {
    for entity in entities {
        validate_key("entity key", entity.entity_key.clone())?;
        validate_key("entity kind", entity.kind.clone())?;
        validate_key("canonical entity name", entity.canonical_name.clone())?;
        if entity.aliases.len() > max_aliases {
            return Err(MemoryManagerError::InvalidInput(
                "entity extractor returned too many aliases".to_owned(),
            ));
        }
        for alias in &entity.aliases {
            validate_key("entity alias", alias.clone())?;
        }
        if !entity.metadata.is_object() {
            return Err(MemoryManagerError::InvalidInput(
                "entity metadata must be a JSON object".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn missing_entity_inputs(extracted: &[ExtractedMemory]) -> Vec<EntityExtractionInput> {
    extracted
        .iter()
        .enumerate()
        .filter(|(_, memory)| memory.entities.is_none())
        .map(|(position, memory)| EntityExtractionInput {
            position,
            text: memory.text.clone(),
        })
        .collect()
}

pub(super) async fn extract_fallback(
    provider: Option<&dyn EntityExtractor>,
    inputs: &[EntityExtractionInput],
) -> Result<Vec<Vec<ExtractedEntity>>, ProviderError> {
    match provider {
        Some(provider) if !inputs.is_empty() => provider.extract(inputs).await,
        _ => Ok(vec![Vec::new(); inputs.len()]),
    }
}

pub(super) fn validate_embeddings(
    embeddings: &[Embedding],
    expected: usize,
    dimensions: i32,
) -> Result<(), MemoryManagerError> {
    if embeddings.len() != expected {
        return Err(MemoryManagerError::InvalidInput(
            "embedding provider returned a different batch length".to_owned(),
        ));
    }
    if embeddings
        .iter()
        .any(|embedding| embedding.dimensions() != dimensions as usize)
    {
        return Err(MemoryManagerError::InvalidInput(
            "embedding provider returned the wrong dimensions".to_owned(),
        ));
    }
    Ok(())
}

/// Correlates provider batches by position into immutable persistence inputs.
pub(super) fn prepare_memories(
    extracted: Vec<ExtractedMemory>,
    embeddings: Vec<Embedding>,
    missing: Vec<EntityExtractionInput>,
    fallback: Vec<Vec<ExtractedEntity>>,
    limits: &MemoryLimits,
) -> Result<Vec<PreparedMemory>, MemoryManagerError> {
    if fallback.len() != missing.len() {
        return Err(MemoryManagerError::InvalidInput(
            "entity extractor returned a different batch length".to_owned(),
        ));
    }
    let fallback = missing
        .into_iter()
        .zip(fallback)
        .map(|(input, entities)| (input.position, entities))
        .collect::<std::collections::BTreeMap<_, _>>();
    extracted
        .into_iter()
        .zip(embeddings)
        .enumerate()
        .map(|(position, values)| prepare_memory(position, values, &fallback, limits))
        .collect()
}

fn prepare_memory(
    position: usize,
    values: (ExtractedMemory, Embedding),
    fallback: &std::collections::BTreeMap<usize, Vec<ExtractedEntity>>,
    limits: &MemoryLimits,
) -> Result<PreparedMemory, MemoryManagerError> {
    let (memory, embedding) = values;
    let entities = memory
        .entities
        .unwrap_or_else(|| fallback.get(&position).cloned().unwrap_or_default());
    if entities.len() > limits.max_entities_per_memory {
        return Err(MemoryManagerError::InvalidInput(
            "entity extractor returned too many entities".to_owned(),
        ));
    }
    validate_entities(&entities, limits.max_aliases_per_entity)?;
    Ok(PreparedMemory {
        id: MemoryId::new(),
        position: position as u32,
        text: memory.text,
        kind: memory.kind,
        temporal: memory.temporal,
        embedding,
        entities,
        metadata: memory.metadata,
    })
}

/// Resolves exact hashes and isolates every remaining semantic candidate.
pub(super) fn deterministic_relations<'a>(
    claims: &'a [MemoryRow],
    candidates: &'a [MemoryRow],
) -> (
    Vec<ReconciliationDecision>,
    (Vec<&'a MemoryRow>, Vec<&'a MemoryRow>),
) {
    let mut decisions = Vec::new();
    let mut ambiguous_new = Vec::new();
    let mut ambiguous_existing = BTreeSet::new();
    for claim in claims {
        let mut ambiguous = false;
        for candidate in candidates {
            if claim.content_hash == candidate.content_hash {
                decisions.push(ReconciliationDecision {
                    from_memory_id: MemoryId::from(claim.id),
                    to_memory_id: MemoryId::from(candidate.id),
                    outcome: ReconciliationOutcome::Corroborates,
                });
            } else {
                ambiguous_existing.insert(candidate.id);
                ambiguous = true;
            }
        }
        if ambiguous {
            ambiguous_new.push(claim);
        }
    }
    let existing = candidates
        .iter()
        .filter(|candidate| ambiguous_existing.contains(&candidate.id))
        .collect();
    (decisions, (ambiguous_new, existing))
}

pub(super) struct MissingExtractor;

#[async_trait::async_trait]
impl MemoryExtractor for MissingExtractor {
    fn revision(&self) -> &str {
        "missing"
    }

    async fn extract(&self, _: ExtractionRequest) -> Result<Vec<ExtractedMemory>, ProviderError> {
        Err(ProviderError::new(
            "missing_extractor",
            "memory extractor is missing",
            false,
        ))
    }
}

pub(super) struct MissingEmbedder;

#[async_trait::async_trait]
impl EmbeddingProvider for MissingEmbedder {
    fn profile(&self) -> crate::EmbeddingProfile {
        crate::EmbeddingProfile {
            model: String::new(),
            revision: String::new(),
            dimensions: 0,
            document_format_revision: String::new(),
        }
    }

    async fn embed(&self, _: &[String]) -> Result<Vec<crate::Embedding>, ProviderError> {
        Err(ProviderError::new(
            "missing_embedder",
            "embedding provider is missing",
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EvidenceId, Memory, MemoryKind, ReconciliationOutcome, RerankResult, TemporalMetadata,
    };

    /// Creates one minimal structured result for conflict-safe reranking tests.
    fn search_result(id: MemoryId, conflicts: Vec<MemoryId>) -> SearchResult {
        SearchResult {
            memory: Memory {
                id,
                evidence_id: EvidenceId::new(),
                user_key: "user".to_owned(),
                agent_key: "agent".to_owned(),
                position: 0,
                text: format!("claim {id}"),
                kind: MemoryKind::Fact,
                temporal: TemporalMetadata::default(),
                metadata: serde_json::json!({}),
                stale: false,
                current_for_retrieval: true,
            },
            evidence_key: format!("evidence:{id}"),
            score: 1.0,
            rerank_score: None,
            support_count: 1,
            conflicts,
        }
    }

    /// Verifies builder limit errors accumulate instead of failing at setter calls.
    #[test]
    fn builder_limits_accumulate_errors() {
        let mut errors = Vec::new();
        let limits = MemoryLimits {
            max_evidence_bytes: 0,
            max_memory_bytes: 0,
            ..MemoryLimits::default()
        };
        validate_limits(&limits, &mut errors);
        assert_eq!(errors.len(), 2);
    }

    /// Verifies retrieval relation expansion cannot be configured without a hard bound.
    #[test]
    fn retrieval_relation_limit_must_be_positive() {
        let mut errors = Vec::new();
        let limits = MemoryLimits {
            max_retrieval_relation_edges: 0,
            ..MemoryLimits::default()
        };
        validate_limits(&limits, &mut errors);
        assert_eq!(
            errors,
            vec!["retrieval relation edge limit must be positive"]
        );
    }

    /// Verifies relation expansion remains a distinct public retrieval failure.
    #[test]
    fn relation_expansion_error_is_not_collapsed_into_persistence() {
        let error = MemoryManagerError::from(RepositoryError::RelationExpansionLimit);
        assert!(matches!(error, MemoryManagerError::RelationExpansionLimit));
        assert_eq!(error.safe_code(), "relation_expansion_limit");
    }

    /// Verifies evidence content uses the configured evidence bound rather than the key bound.
    #[test]
    fn evidence_content_may_exceed_key_length() {
        let submission = EvidenceSubmission::new("document:v1", "x".repeat(1_024)).unwrap();
        assert!(validate_submission(&submission, &MemoryLimits::default()).is_ok());
    }

    /// Verifies entity fallback is requested only for explicit extractor omissions.
    #[test]
    fn entity_fallback_respects_some_empty() {
        let memories = vec![
            ExtractedMemory {
                text: "one".to_owned(),
                entities: Some(Vec::new()),
                kind: MemoryKind::Fact,
                temporal: Default::default(),
                metadata: JsonValue::Object(Default::default()),
            },
            ExtractedMemory {
                text: "two".to_owned(),
                entities: None,
                kind: MemoryKind::Fact,
                temporal: Default::default(),
                metadata: JsonValue::Object(Default::default()),
            },
        ];
        assert_eq!(
            missing_entity_inputs(&memories),
            vec![EntityExtractionInput {
                position: 1,
                text: "two".to_owned()
            }]
        );
    }

    /// Verifies extractor entity aliases obey the configured per-entity bound.
    #[test]
    fn extracted_entity_aliases_are_bounded() {
        let extracted = vec![ExtractedMemory {
            text: "The user likes tea.".to_owned(),
            entities: Some(vec![ExtractedEntity {
                entity_key: "drink:tea".to_owned(),
                kind: "drink".to_owned(),
                canonical_name: "Tea".to_owned(),
                aliases: vec!["alias".to_owned(); 17],
                metadata: JsonValue::Object(Default::default()),
            }]),
            kind: MemoryKind::Preference,
            temporal: Default::default(),
            metadata: JsonValue::Object(Default::default()),
        }];
        assert!(validate_extracted(&extracted, &MemoryLimits::default()).is_err());
    }

    /// Verifies reranking cannot request fewer candidates than the final result limit.
    #[test]
    fn rerank_candidate_limit_must_cover_results() {
        let request = SearchRequest::new("preferences").unwrap().rerank(4);
        assert!(validate_search(&request).is_err());
    }

    /// Verifies reranking restores a complete transitive conflict component in retrieval order.
    #[test]
    fn reranking_preserves_transitive_conflict_components() {
        let first = MemoryId::from(uuid::Uuid::from_u128(1));
        let second = MemoryId::from(uuid::Uuid::from_u128(2));
        let third = MemoryId::from(uuid::Uuid::from_u128(3));
        let unrelated = MemoryId::from(uuid::Uuid::from_u128(4));
        let results = vec![
            search_result(third, vec![second]),
            search_result(unrelated, Vec::new()),
            search_result(second, vec![first, third]),
            search_result(first, vec![second]),
        ];
        let reranked = vec![RerankResult {
            memory_id: first,
            score: 0.9,
        }];
        let selected = apply_reranking(results, reranked, 1).expect("valid reranking");
        let ids = selected
            .iter()
            .map(|result| result.memory.id)
            .collect::<Vec<_>>();

        assert_eq!(ids, vec![first, third, second]);
    }

    /// Verifies invalid relation IDs never reach durable relation persistence.
    #[test]
    fn reconciliation_decisions_require_group_ids() {
        let first = MemoryId::new();
        let second = MemoryId::new();
        let decision = ReconciliationDecision {
            from_memory_id: first,
            to_memory_id: second,
            outcome: ReconciliationOutcome::Conflicts,
        };
        assert!(
            validate_decisions(
                &[decision],
                &BTreeSet::from([first]),
                &BTreeSet::from([first])
            )
            .is_err()
        );
    }

    /// Verifies deterministic temporal analysis recognizes explicit as-of dates.
    #[test]
    fn temporal_query_analysis_sets_as_of_mode() {
        let now = Utc::now();
        let mut request = SearchRequest::new("status as of 2025-03-04").unwrap();
        analyze_temporal_query(&mut request, now);
        let expected = DateTime::parse_from_rfc3339("2025-03-04T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(request.timeline.valid_time, crate::ValidTime::At(expected));
        assert_eq!(request.timeline.reference_time, Some(expected));
    }
}
