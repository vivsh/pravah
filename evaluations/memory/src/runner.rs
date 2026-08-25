use std::{sync::Arc, time::Instant};

use futures::{StreamExt, TryStreamExt, stream};
use pravah_memory::{EvidenceSubmission, MemoryManager, SearchRequest, SearchResult, ValidTime};
use serde_json::json;
use uuid::Uuid;

use crate::{
    EvaluationDataset, EvaluationError, EvaluationEvidence, EvaluationGroup, EvaluationRun,
    EvaluationRunManifest, QuestionObservation, RetrievedClaim,
};

/// Executes an already normalized benchmark through an application-configured manager.
#[derive(Clone)]
pub struct EvaluationRunner {
    manager: MemoryManager,
    options: Arc<RunnerOptions>,
}

struct RunnerOptions {
    system_label: String,
    user_key: String,
    agent_prefix: String,
    search_limit: u32,
    candidate_limit: u32,
    group_concurrency: usize,
    reconciliation: Option<ReconciliationOptions>,
}

#[derive(Clone, Copy)]
struct ReconciliationOptions {
    batch_limit: u32,
    max_batches: u32,
}

/// Deferred-validation builder for a reproducible evaluation runner.
pub struct EvaluationRunnerBuilder {
    manager: MemoryManager,
    system_label: Option<String>,
    user_key: String,
    agent_prefix: String,
    search_limit: u32,
    candidate_limit: u32,
    group_concurrency: usize,
    reconciliation: Option<ReconciliationOptions>,
    errors: Vec<String>,
}

impl EvaluationRunner {
    /// Starts a builder around providers and persistence chosen by the application.
    pub fn builder(manager: MemoryManager) -> EvaluationRunnerBuilder {
        EvaluationRunnerBuilder {
            manager,
            system_label: None,
            user_key: "pravah-eval".to_owned(),
            agent_prefix: "memory".to_owned(),
            search_limit: 10,
            candidate_limit: 50,
            group_concurrency: 1,
            reconciliation: None,
            errors: Vec::new(),
        }
    }

    /// Ingests isolated histories and records ranked retrieval provenance.
    pub async fn run(&self, dataset: &EvaluationDataset) -> Result<EvaluationRun, EvaluationError> {
        let run_id = Uuid::now_v7();
        let mut groups = stream::iter(dataset.groups.iter().cloned().enumerate())
            .map(|(index, group)| self.run_group(run_id, index, group))
            .buffer_unordered(self.options.group_concurrency)
            .try_collect::<Vec<_>>()
            .await?;
        groups.sort_by_key(|(index, _)| *index);
        let observations = groups.into_iter().flat_map(|(_, rows)| rows).collect();
        Ok(EvaluationRun {
            manifest: self.manifest(dataset, run_id),
            observations,
        })
    }

    async fn run_group(
        &self,
        run_id: Uuid,
        index: usize,
        group: EvaluationGroup,
    ) -> Result<(usize, Vec<QuestionObservation>), EvaluationError> {
        let agent_key = format!("{}:{run_id}:{}", self.options.agent_prefix, group.id);
        let ingestor = self.manager.ingestor(&self.options.user_key, &agent_key)?;
        for evidence in &group.evidence {
            ingestor
                .submit_with(submission(evidence, &group.id)?)
                .await?;
        }
        self.reconcile(&agent_key).await?;
        let retriever = self.manager.retriever(&self.options.user_key, &agent_key)?;
        let mut observations = Vec::with_capacity(group.questions.len());
        for question in &group.questions {
            let request = self.search_request(question)?;
            let started = Instant::now();
            let results = retriever.search_with(request).await?;
            observations.push(observation(&group.id, question, results, started));
        }
        Ok((index, observations))
    }

    async fn reconcile(&self, agent_key: &str) -> Result<(), EvaluationError> {
        let Some(options) = self.options.reconciliation else {
            return Ok(());
        };
        let reconciler = self.manager.reconciler(&self.options.user_key, agent_key)?;
        for _ in 0..options.max_batches {
            if reconciler.reconcile_pending(options.batch_limit).await? == 0 {
                return Ok(());
            }
        }
        Err(EvaluationError::InvalidConfiguration(
            "reconciliation did not drain within max_batches".to_owned(),
        ))
    }

    fn search_request(
        &self,
        question: &crate::EvaluationQuestion,
    ) -> Result<SearchRequest, EvaluationError> {
        let mut request = SearchRequest::new(&question.question)?.history();
        request.limit = self.options.search_limit;
        request.candidate_limit = self.options.candidate_limit;
        request.timeline.valid_time = ValidTime::Any;
        request.timeline.reference_time = question.asked_at;
        Ok(request)
    }

    fn manifest(&self, dataset: &EvaluationDataset, run_id: Uuid) -> EvaluationRunManifest {
        EvaluationRunManifest {
            run_id,
            dataset: dataset.kind,
            dataset_revision: dataset.revision.clone(),
            source_sha256: dataset.source_sha256.clone(),
            granularity: dataset.granularity,
            system_label: self.options.system_label.clone(),
            search_limit: self.options.search_limit,
            candidate_limit: self.options.candidate_limit,
            reconciliation_enabled: self.options.reconciliation.is_some(),
            completed_at: chrono::Utc::now(),
        }
    }
}

impl EvaluationRunnerBuilder {
    /// Records the exact provider, prompt, model, and configuration identity.
    pub fn system_label(mut self, label: impl Into<String>) -> Self {
        let label = label.into();
        if label.trim().is_empty() {
            self.errors
                .push("system label must not be empty".to_owned());
        }
        self.system_label = Some(label);
        self
    }

    /// Replaces the evaluation-only database scope prefix.
    pub fn scope(mut self, user_key: impl Into<String>, agent_prefix: impl Into<String>) -> Self {
        self.user_key = user_key.into();
        self.agent_prefix = agent_prefix.into();
        validate_scope_key("user key", &self.user_key, &mut self.errors);
        validate_scope_key("agent prefix", &self.agent_prefix, &mut self.errors);
        self
    }

    /// Sets final and per-channel candidate bounds.
    pub fn search_limits(mut self, result_limit: u32, candidate_limit: u32) -> Self {
        if result_limit == 0 || candidate_limit < result_limit {
            self.errors.push(
                "search limits require result_limit > 0 and candidate_limit >= result_limit"
                    .to_owned(),
            );
        }
        self.search_limit = result_limit;
        self.candidate_limit = candidate_limit;
        self
    }

    /// Bounds concurrently processed isolated histories.
    pub fn group_concurrency(mut self, concurrency: usize) -> Self {
        if concurrency == 0 || concurrency > 128 {
            self.errors
                .push("group concurrency must be between 1 and 128".to_owned());
        }
        self.group_concurrency = concurrency;
        self
    }

    /// Drains asynchronous reconciliation before querying each history.
    pub fn reconcile(mut self, batch_limit: u32, max_batches: u32) -> Self {
        if batch_limit == 0 || max_batches == 0 {
            self.errors
                .push("reconciliation bounds must be greater than zero".to_owned());
        }
        self.reconciliation = Some(ReconciliationOptions {
            batch_limit,
            max_batches,
        });
        self
    }

    /// Returns all accumulated configuration failures at the terminal step.
    pub fn build(self) -> Result<EvaluationRunner, EvaluationError> {
        let mut errors = self.errors;
        if self.system_label.is_none() {
            errors.push("system label is required".to_owned());
        }
        if !errors.is_empty() {
            return Err(EvaluationError::InvalidConfiguration(errors.join("; ")));
        }
        Ok(EvaluationRunner {
            manager: self.manager,
            options: Arc::new(RunnerOptions {
                system_label: self.system_label.unwrap_or_default(),
                user_key: self.user_key,
                agent_prefix: self.agent_prefix,
                search_limit: self.search_limit,
                candidate_limit: self.candidate_limit,
                group_concurrency: self.group_concurrency,
                reconciliation: self.reconciliation,
            }),
        })
    }
}

fn validate_scope_key(name: &str, value: &str, errors: &mut Vec<String>) {
    if value.trim().is_empty() || value.len() > 128 {
        errors.push(format!("{name} must contain 1 to 128 bytes"));
    }
}

fn submission(
    evidence: &EvaluationEvidence,
    group_id: &str,
) -> Result<EvidenceSubmission, EvaluationError> {
    let metadata = json!({
        "evaluation_group": group_id,
        "source_session_id": evidence.source_session_id,
        "source_turn_id": evidence.source_turn_id,
    });
    let submission =
        EvidenceSubmission::new(&evidence.evidence_key, &evidence.content)?.with_metadata(metadata);
    Ok(match evidence.observed_at {
        Some(observed_at) => submission.with_observed_at(observed_at),
        None => submission,
    })
}

fn observation(
    group_id: &str,
    question: &crate::EvaluationQuestion,
    results: Vec<SearchResult>,
    started: Instant,
) -> QuestionObservation {
    let retrieved = results
        .into_iter()
        .enumerate()
        .map(|(index, result)| RetrievedClaim {
            memory_id: result.memory.id,
            evidence_key: result.evidence_key,
            text: result.memory.text,
            rank: u32::try_from(index + 1).unwrap_or(u32::MAX),
            score: result.score,
            support_count: result.support_count,
        })
        .collect();
    QuestionObservation {
        group_id: group_id.to_owned(),
        question_id: question.id.clone(),
        category: question.category.clone(),
        relevant_evidence_keys: question.relevant_evidence_keys.clone(),
        abstention: question.abstention,
        retrieved,
        retrieval_latency_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
    }
}
