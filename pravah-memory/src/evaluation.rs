use serde::{Deserialize, Serialize};

use super::{MemoryId, RelationKind};

/// One expected or observed claim relation used by memory evaluations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluatedRelation {
    /// First fixture-local claim label.
    pub from: String,
    /// Second fixture-local claim label.
    pub to: String,
    /// Expected or observed relation class.
    pub kind: RelationKind,
}

/// Measured output for one LoCoMo, LongMemEval, or local fixture case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationObservation {
    /// Evidence-supported claims expected by the fixture.
    pub expected_claims: Vec<String>,
    /// Claims produced by the extractor.
    pub extracted_claims: Vec<String>,
    /// Produced claims judged unsupported by the source evidence.
    pub unsupported_claims: usize,
    /// Produced claims judged to duplicate existing knowledge.
    pub duplicate_claims: usize,
    /// Expected corroboration, supersession, and conflict relations.
    pub expected_relations: Vec<EvaluatedRelation>,
    /// Relations produced by reconciliation.
    pub actual_relations: Vec<EvaluatedRelation>,
    /// Relevant claim IDs for the evaluated query.
    pub relevant_memory_ids: Vec<MemoryId>,
    /// Ranked claim IDs returned by retrieval.
    pub retrieved_memory_ids: Vec<MemoryId>,
    /// Whether a temporal current/as-of/upcoming answer was correct.
    pub temporal_correct: Option<bool>,
    /// Whether the system correctly abstained or preserved ambiguity.
    pub abstention_correct: Option<bool>,
    /// Context tokens assembled from retrieved claims.
    pub context_tokens: u64,
    /// LLM calls used for ingestion and reconciliation.
    pub llm_calls: u64,
    /// LLM input and output tokens used for the evidence item.
    pub llm_tokens: u64,
    /// Whether the evidence required a reconciliation attempt.
    pub reconciliation_attempted: bool,
    /// Queue delay before reconciliation started.
    pub reconciliation_queue_lag_ms: Option<f64>,
    /// Evidence-to-searchable latency in milliseconds.
    pub searchable_latency_ms: f64,
    /// Evidence-to-reconciled latency in milliseconds when reconciliation ran.
    pub reconciled_latency_ms: Option<f64>,
    /// End-to-end search latency in milliseconds.
    pub search_latency_ms: f64,
    /// Exact scoped vector top-K used as an HNSW recall oracle.
    pub exact_vector_ids: Vec<MemoryId>,
    /// HNSW scoped vector top-K for the same query.
    pub hnsw_vector_ids: Vec<MemoryId>,
    /// Total table and index bytes attributed to this measured fixture batch.
    pub storage_bytes: Option<u64>,
    /// Claims represented by `storage_bytes`.
    pub stored_claims: Option<u64>,
}

/// Aggregate quality, cost, and latency metrics over evaluation observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEvaluationReport {
    /// Number of evaluated cases.
    pub cases: usize,
    /// Exact normalized claim precision against fixture expectations.
    pub extraction_precision: f64,
    /// Fraction of produced claims unsupported by evidence.
    pub unsupported_claim_rate: f64,
    /// Fraction of produced claims classified as duplicate knowledge.
    pub duplicate_rate: f64,
    /// Fraction of actual derived relations that preserve a conflict.
    pub conflict_rate: f64,
    /// Exact relation-classification accuracy.
    pub relation_accuracy: f64,
    /// Mean retrieval recall at each fixture's supplied K.
    pub retrieval_recall_at_k: f64,
    /// Mean reciprocal rank of the first relevant retrieved claim.
    pub retrieval_mrr: f64,
    /// Accuracy over cases with explicit temporal expectations.
    pub temporal_accuracy: f64,
    /// Accuracy over cases requiring abstention or conflict preservation.
    pub abstention_accuracy: f64,
    /// Mean context-token budget.
    pub mean_context_tokens: f64,
    /// Mean LLM calls per evidence item.
    pub mean_llm_calls: f64,
    /// Mean LLM tokens per evidence item.
    pub mean_llm_tokens: f64,
    /// Fraction of evidence items requiring semantic reconciliation.
    pub reconciliation_rate: f64,
    /// Mean reconciliation queue delay in milliseconds.
    pub mean_reconciliation_queue_lag_ms: f64,
    /// Mean evidence-to-searchable latency in milliseconds.
    pub mean_searchable_latency_ms: f64,
    /// Mean evidence-to-reconciled latency in milliseconds.
    pub mean_reconciled_latency_ms: f64,
    /// Search latency p50 in milliseconds.
    pub search_p50_ms: f64,
    /// Search latency p95 in milliseconds.
    pub search_p95_ms: f64,
    /// Search latency p99 in milliseconds.
    pub search_p99_ms: f64,
    /// Mean scoped HNSW recall against exact vector search.
    pub hnsw_recall_at_k: f64,
    /// Projected table-plus-index bytes per million stored claims.
    pub storage_bytes_per_million_claims: f64,
}

/// Measured deterministic context-selection and optional recall-telemetry output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextEvaluationObservation {
    /// Relevant claims for the evaluated query, including claims outside the retrieval limit.
    pub relevant_memory_ids: Vec<MemoryId>,
    /// Relevant claims made available to the context assembler by retrieval.
    pub retrieved_relevant_memory_ids: Vec<MemoryId>,
    /// Claims selected by the context assembler under the evaluated budget.
    pub selected_memory_ids: Vec<MemoryId>,
    /// Claims explicitly accepted by the application.
    pub accepted_memory_ids: Vec<MemoryId>,
    /// Claims actually consumed by the application or model context.
    pub used_memory_ids: Vec<MemoryId>,
    /// Complete unresolved conflict components expected in retrieval output.
    pub conflict_groups: Vec<Vec<MemoryId>>,
    /// Selected claims that retained their evidence provenance.
    pub selected_with_provenance: usize,
    /// Unicode scalar values in the rendered context.
    pub rendered_characters: u64,
    /// Model tokens in the rendered context when a token counter was measured.
    pub rendered_tokens: Option<u64>,
    /// Context assembly CPU time in microseconds.
    pub assembly_cpu_micros: f64,
    /// Context assembly allocations measured by the benchmark harness.
    pub assembly_allocations: Option<u64>,
    /// Durable recall-event rows emitted by this case.
    pub telemetry_event_rows: u64,
    /// Event rows and indexes attributed to this case.
    pub telemetry_storage_bytes: Option<u64>,
}

/// Aggregate context-selection, assembly-cost, and telemetry-volume measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextEvaluationReport {
    /// Number of evaluated context cases.
    pub cases: usize,
    /// Relevant claims selected as a fraction of all fixture-relevant claims.
    pub context_selection_recall: f64,
    /// Relevant selected claims as a fraction of relevant retrieved claims.
    pub retrieved_relevance_retention: f64,
    /// Application-accepted claims that were fixture-relevant.
    pub accepted_precision: f64,
    /// Application-used claims that were fixture-relevant.
    pub used_precision: f64,
    /// Conflict components that were either wholly selected or wholly omitted.
    pub conflict_group_preservation: f64,
    /// Selected claims that retained evidence provenance.
    pub provenance_preservation: f64,
    /// Mean rendered Unicode scalar values per case.
    pub mean_rendered_characters: f64,
    /// Mean rendered tokens over cases that supplied a token measurement.
    pub mean_rendered_tokens: f64,
    /// Context assembly p95 CPU time in microseconds.
    pub assembly_p95_micros: f64,
    /// Mean allocations over cases that supplied an allocation measurement.
    pub mean_assembly_allocations: f64,
    /// Total durable recall-event rows across evaluated cases.
    pub telemetry_event_rows: u64,
    /// Total measured event-row and index bytes.
    pub telemetry_storage_bytes: u64,
}

impl ContextEvaluationReport {
    /// Computes deterministic context and telemetry aggregates from fixture observations.
    pub fn from_observations(observations: &[ContextEvaluationObservation]) -> Self {
        let selected = observations
            .iter()
            .flat_map(|item| &item.selected_memory_ids)
            .count();
        Self {
            cases: observations.len(),
            context_selection_recall: context_selection_recall(observations),
            retrieved_relevance_retention: retrieved_relevance_retention(observations),
            accepted_precision: outcome_precision(observations, |item| &item.accepted_memory_ids),
            used_precision: outcome_precision(observations, |item| &item.used_memory_ids),
            conflict_group_preservation: conflict_group_preservation(observations),
            provenance_preservation: ratio(
                observations
                    .iter()
                    .map(|item| item.selected_with_provenance)
                    .sum(),
                selected,
            ),
            mean_rendered_characters: mean(
                observations
                    .iter()
                    .map(|item| item.rendered_characters as f64),
            ),
            mean_rendered_tokens: mean(
                observations
                    .iter()
                    .filter_map(|item| item.rendered_tokens.map(|value| value as f64)),
            ),
            assembly_p95_micros: percentile(
                &observations
                    .iter()
                    .map(|item| item.assembly_cpu_micros)
                    .collect::<Vec<_>>(),
                0.95,
            ),
            mean_assembly_allocations: mean(
                observations
                    .iter()
                    .filter_map(|item| item.assembly_allocations.map(|value| value as f64)),
            ),
            telemetry_event_rows: observations
                .iter()
                .map(|item| item.telemetry_event_rows)
                .sum(),
            telemetry_storage_bytes: observations
                .iter()
                .filter_map(|item| item.telemetry_storage_bytes)
                .sum(),
        }
    }
}

impl MemoryEvaluationReport {
    /// Computes deterministic aggregate metrics without hiding empty denominators.
    pub fn from_observations(observations: &[EvaluationObservation]) -> Self {
        let (extraction_precision, unsupported_claim_rate, duplicate_rate) =
            extraction_metrics(observations);
        let (relation_accuracy, conflict_rate) = relation_metrics(observations);
        let (retrieval_recall_at_k, retrieval_mrr, hnsw_recall_at_k) =
            retrieval_metrics(observations);
        let (temporal_accuracy, abstention_accuracy) = correctness_metrics(observations);
        let (mean_context_tokens, mean_llm_calls, mean_llm_tokens, reconciliation_rate) =
            cost_metrics(observations);
        let latency = latency_metrics(observations);
        Self {
            cases: observations.len(),
            extraction_precision,
            unsupported_claim_rate,
            duplicate_rate,
            conflict_rate,
            relation_accuracy,
            retrieval_recall_at_k,
            retrieval_mrr,
            temporal_accuracy,
            abstention_accuracy,
            mean_context_tokens,
            mean_llm_calls,
            mean_llm_tokens,
            reconciliation_rate,
            mean_reconciliation_queue_lag_ms: latency.0,
            mean_searchable_latency_ms: latency.1,
            mean_reconciled_latency_ms: latency.2,
            search_p50_ms: latency.3,
            search_p95_ms: latency.4,
            search_p99_ms: latency.5,
            hnsw_recall_at_k,
            storage_bytes_per_million_claims: storage_growth(observations),
        }
    }
}

fn extraction_metrics(observations: &[EvaluationObservation]) -> (f64, f64, f64) {
    let counts = extraction_counts(observations);
    (
        ratio(counts.0, counts.1),
        ratio(counts.2, counts.1),
        ratio(counts.3, counts.1),
    )
}

fn relation_metrics(observations: &[EvaluationObservation]) -> (f64, f64) {
    let counts = relation_counts(observations);
    (ratio(counts.0, counts.1), conflict_rate(observations))
}

fn retrieval_metrics(observations: &[EvaluationObservation]) -> (f64, f64, f64) {
    (
        mean(observations.iter().map(retrieval_recall)),
        mean(observations.iter().map(reciprocal_rank)),
        mean(observations.iter().map(hnsw_recall)),
    )
}

fn correctness_metrics(observations: &[EvaluationObservation]) -> (f64, f64) {
    (
        optional_accuracy(observations.iter().map(|item| item.temporal_correct)),
        optional_accuracy(observations.iter().map(|item| item.abstention_correct)),
    )
}

/// Aggregates context and provider budgets without treating absent cases as successes.
fn cost_metrics(observations: &[EvaluationObservation]) -> (f64, f64, f64, f64) {
    (
        mean(observations.iter().map(|item| item.context_tokens as f64)),
        mean(observations.iter().map(|item| item.llm_calls as f64)),
        mean(observations.iter().map(|item| item.llm_tokens as f64)),
        ratio(
            observations
                .iter()
                .filter(|item| item.reconciliation_attempted)
                .count(),
            observations.len(),
        ),
    )
}

/// Aggregates pipeline and search latency distributions independently.
fn latency_metrics(observations: &[EvaluationObservation]) -> (f64, f64, f64, f64, f64, f64) {
    let search = observations
        .iter()
        .map(|item| item.search_latency_ms)
        .collect::<Vec<_>>();
    (
        mean(
            observations
                .iter()
                .filter_map(|item| item.reconciliation_queue_lag_ms),
        ),
        mean(observations.iter().map(|item| item.searchable_latency_ms)),
        mean(
            observations
                .iter()
                .filter_map(|item| item.reconciled_latency_ms),
        ),
        percentile(&search, 0.50),
        percentile(&search, 0.95),
        percentile(&search, 0.99),
    )
}

/// Counts matched, produced, unsupported, and duplicate extracted claims.
fn extraction_counts(observations: &[EvaluationObservation]) -> (usize, usize, usize, usize) {
    let mut matched = 0;
    let mut produced = 0;
    let mut unsupported = 0;
    let mut duplicate = 0;
    for observation in observations {
        produced += observation.extracted_claims.len();
        unsupported += observation.unsupported_claims;
        duplicate += observation.duplicate_claims;
        matched += observation
            .extracted_claims
            .iter()
            .filter(|claim| observation.expected_claims.contains(claim))
            .count();
    }
    (matched, produced, unsupported, duplicate)
}

/// Measures how frequently derived relations preserve unresolved conflicts.
fn conflict_rate(observations: &[EvaluationObservation]) -> f64 {
    let relations = observations
        .iter()
        .flat_map(|observation| &observation.actual_relations)
        .collect::<Vec<_>>();
    ratio(
        relations
            .iter()
            .filter(|relation| relation.kind == RelationKind::Conflicts)
            .count(),
        relations.len(),
    )
}

/// Counts exact relation matches against fixture expectations.
fn relation_counts(observations: &[EvaluationObservation]) -> (usize, usize) {
    observations
        .iter()
        .fold((0, 0), |(correct, total), observation| {
            let matches = observation
                .actual_relations
                .iter()
                .filter(|relation| observation.expected_relations.contains(relation))
                .count();
            (
                correct + matches,
                total + observation.expected_relations.len(),
            )
        })
}

/// Computes recall at the K supplied by one fixture observation.
fn retrieval_recall(observation: &EvaluationObservation) -> f64 {
    if observation.relevant_memory_ids.is_empty() {
        return 1.0;
    }
    let found = observation
        .relevant_memory_ids
        .iter()
        .filter(|id| observation.retrieved_memory_ids.contains(id))
        .count();
    ratio(found, observation.relevant_memory_ids.len())
}

fn reciprocal_rank(observation: &EvaluationObservation) -> f64 {
    observation
        .retrieved_memory_ids
        .iter()
        .position(|id| observation.relevant_memory_ids.contains(id))
        .map_or(0.0, |position| 1.0 / (position + 1) as f64)
}

/// Measures relevant context selection against every fixture-relevant claim.
fn context_selection_recall(observations: &[ContextEvaluationObservation]) -> f64 {
    let relevant = observations
        .iter()
        .map(|item| item.relevant_memory_ids.len())
        .sum();
    let selected = relevant_outcome_count(observations, |item| &item.selected_memory_ids);
    ratio(selected, relevant)
}

/// Measures how much retrieved relevance survives deterministic context budgeting.
fn retrieved_relevance_retention(observations: &[ContextEvaluationObservation]) -> f64 {
    let retrieved = observations
        .iter()
        .map(|item| item.retrieved_relevant_memory_ids.len())
        .sum();
    let selected = observations
        .iter()
        .map(|item| {
            item.selected_memory_ids
                .iter()
                .filter(|id| item.retrieved_relevant_memory_ids.contains(id))
                .count()
        })
        .sum();
    ratio(selected, retrieved)
}

/// Measures an explicit application outcome against fixture relevance.
fn outcome_precision(
    observations: &[ContextEvaluationObservation],
    outcome: impl Fn(&ContextEvaluationObservation) -> &[MemoryId],
) -> f64 {
    let total = observations.iter().map(|item| outcome(item).len()).sum();
    ratio(relevant_outcome_count(observations, outcome), total)
}

fn relevant_outcome_count(
    observations: &[ContextEvaluationObservation],
    outcome: impl Fn(&ContextEvaluationObservation) -> &[MemoryId],
) -> usize {
    observations
        .iter()
        .map(|item| {
            outcome(item)
                .iter()
                .filter(|id| item.relevant_memory_ids.contains(id))
                .count()
        })
        .sum()
}

/// Measures whether every unresolved conflict group is selected atomically.
fn conflict_group_preservation(observations: &[ContextEvaluationObservation]) -> f64 {
    let groups = observations
        .iter()
        .flat_map(|item| item.conflict_groups.iter().map(move |group| (item, group)))
        .collect::<Vec<_>>();
    let preserved = groups
        .iter()
        .filter(|(item, group)| conflict_group_is_atomic(&item.selected_memory_ids, group))
        .count();
    ratio(preserved, groups.len())
}

fn conflict_group_is_atomic(selected: &[MemoryId], group: &[MemoryId]) -> bool {
    let selected_count = group.iter().filter(|id| selected.contains(id)).count();
    selected_count == 0 || selected_count == group.len()
}

/// Compares approximate scoped neighbors with the exact vector oracle.
fn hnsw_recall(observation: &EvaluationObservation) -> f64 {
    if observation.exact_vector_ids.is_empty() {
        return 1.0;
    }
    let found = observation
        .exact_vector_ids
        .iter()
        .filter(|id| observation.hnsw_vector_ids.contains(id))
        .count();
    ratio(found, observation.exact_vector_ids.len())
}

fn optional_accuracy(values: impl Iterator<Item = Option<bool>>) -> f64 {
    let values = values.flatten().collect::<Vec<_>>();
    ratio(values.iter().filter(|value| **value).count(), values.len())
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Computes a deterministic nearest-rank latency percentile.
fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}

/// Projects measured table and index bytes to one million claims.
fn storage_growth(observations: &[EvaluationObservation]) -> f64 {
    let bytes = observations
        .iter()
        .filter_map(|item| item.storage_bytes)
        .sum::<u64>();
    let claims = observations
        .iter()
        .filter_map(|item| item.stored_claims)
        .sum::<u64>();
    if claims == 0 {
        0.0
    } else {
        bytes as f64 / claims as f64 * 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies the report computes retrieval, unsupported-claim, and latency metrics.
    #[test]
    fn evaluation_report_aggregates_quality_and_latency() {
        let id = MemoryId::new();
        let observation = EvaluationObservation {
            expected_claims: vec!["claim".to_owned()],
            extracted_claims: vec!["claim".to_owned()],
            unsupported_claims: 0,
            duplicate_claims: 0,
            expected_relations: Vec::new(),
            actual_relations: Vec::new(),
            relevant_memory_ids: vec![id],
            retrieved_memory_ids: vec![id],
            temporal_correct: Some(true),
            abstention_correct: None,
            context_tokens: 20,
            llm_calls: 1,
            llm_tokens: 100,
            reconciliation_attempted: false,
            reconciliation_queue_lag_ms: None,
            searchable_latency_ms: 50.0,
            reconciled_latency_ms: None,
            search_latency_ms: 10.0,
            exact_vector_ids: vec![id],
            hnsw_vector_ids: vec![id],
            storage_bytes: Some(1_000),
            stored_claims: Some(10),
        };
        let report = MemoryEvaluationReport::from_observations(&[observation]);

        assert_eq!(report.extraction_precision, 1.0);
        assert_eq!(report.retrieval_recall_at_k, 1.0);
        assert_eq!(report.search_p99_ms, 10.0);
    }

    /// Verifies context selection, explicit outcomes, conflicts, and telemetry stay separate.
    #[test]
    fn context_report_aggregates_budget_and_feedback_metrics() {
        let first = MemoryId::new();
        let second = MemoryId::new();
        let irrelevant = MemoryId::new();
        let observation = ContextEvaluationObservation {
            relevant_memory_ids: vec![first, second],
            retrieved_relevant_memory_ids: vec![first, second],
            selected_memory_ids: vec![first, second],
            accepted_memory_ids: vec![first, irrelevant],
            used_memory_ids: vec![second],
            conflict_groups: vec![vec![first, second]],
            selected_with_provenance: 2,
            rendered_characters: 120,
            rendered_tokens: Some(30),
            assembly_cpu_micros: 80.0,
            assembly_allocations: Some(4),
            telemetry_event_rows: 3,
            telemetry_storage_bytes: Some(900),
        };
        let report = ContextEvaluationReport::from_observations(&[observation]);

        assert_eq!(report.context_selection_recall, 1.0);
        assert_eq!(report.accepted_precision, 0.5);
        assert_eq!(report.used_precision, 1.0);
        assert_eq!(report.conflict_group_preservation, 1.0);
        assert_eq!(report.provenance_preservation, 1.0);
        assert_eq!(report.telemetry_event_rows, 3);
    }
}
