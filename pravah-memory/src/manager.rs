use std::{sync::Arc, time::Instant};

mod recall;
mod support;

use support::{
    MissingEmbedder, MissingExtractor, analyze_temporal_query, apply_reranking,
    deterministic_relations, extract_fallback, missing_entity_inputs, prepare_memories, receipt,
    semantic_relations, validate_content, validate_database_profile, validate_embeddings,
    validate_extracted, validate_key, validate_limits, validate_profile, validate_search,
    validate_submission,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use thiserror::Error;

use super::postgres::models::{EvidenceRow, MemoryRow};
use super::postgres::repository::{PreparedMemory, ProcessingStart, RepositoryError, sha256};
use super::postgres::{MemoryRepository, hybrid_search};
use super::{
    EmbeddingProvider, EntityExtractionInput, EntityExtractor, Evidence, EvidenceId,
    ExtractedEntity, ExtractionRequest, MemoryExtractor, MemoryLimits, MemoryReconciler,
    MemoryReranker, ProviderError, ReconciliationClaim, ReconciliationDecision,
    ReconciliationGroup, RerankCandidate, RerankRequest, SearchRequest, SearchResult,
};

/// Whether relation reconciliation is mandatory or deliberately disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReconciliationMode {
    /// Require a reconciler and mark new evidence pending by default.
    #[default]
    Required,
    /// Keep immutable ADD-only claims without relation classification.
    Disabled,
}

/// Advanced immutable evidence submission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSubmission {
    evidence_key: String,
    content: String,
    observed_at: Option<DateTime<Utc>>,
    metadata: JsonValue,
}

impl EvidenceSubmission {
    /// Creates normalized evidence with a required application key.
    pub fn new(
        evidence_key: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<Self, MemoryManagerError> {
        Ok(Self {
            evidence_key: validate_key("evidence key", evidence_key.into())?,
            content: validate_content(content.into())?,
            observed_at: None,
            metadata: JsonValue::Object(Default::default()),
        })
    }

    /// Overrides the default evidence acceptance-time temporal anchor.
    pub fn with_observed_at(mut self, observed_at: DateTime<Utc>) -> Self {
        self.observed_at = Some(observed_at);
        self
    }

    /// Attaches opaque application provenance metadata.
    pub fn with_metadata(mut self, metadata: JsonValue) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Durable acknowledgement returned after claims become searchable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReceipt {
    /// Internal UUIDv7 evidence identity.
    pub evidence_id: EvidenceId,
    /// Application evidence key.
    pub evidence_key: String,
    /// Number of immutable derived claims.
    pub memory_count: usize,
    /// Whether this call accepted the evidence row for the first time.
    pub accepted: bool,
    /// Current durable extraction state, including another active owner.
    pub processing: super::ProcessingState,
    /// Whether the evidence has been invalidated and cannot publish claims.
    pub stale: bool,
    /// Whether background reconciliation remains pending.
    pub reconciliation_pending: bool,
}

/// Top-level evidence, reconciliation, and retrieval service.
#[derive(Clone)]
pub struct MemoryManager {
    inner: Arc<ManagerInner>,
}

struct ManagerInner {
    repository: MemoryRepository,
    memory_extractor: Arc<dyn MemoryExtractor>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    entity_extractor: Option<Arc<dyn EntityExtractor>>,
    reconciler: Option<Arc<dyn MemoryReconciler>>,
    reranker: Option<Arc<dyn MemoryReranker>>,
    limits: MemoryLimits,
    reconciliation_mode: ReconciliationMode,
    embedding_dimensions: i32,
    text_search_configuration: String,
}

impl MemoryManager {
    /// Starts a builder whose errors are deferred until [`MemoryManagerBuilder::build`].
    pub fn builder(pool: mool::DbPool) -> MemoryManagerBuilder {
        MemoryManagerBuilder::new(pool)
    }

    /// Creates an ergonomic ingestion and evidence-lifecycle handle for one scope.
    pub fn ingestor(
        &self,
        user_key: impl Into<String>,
        agent_key: impl Into<String>,
    ) -> Result<EvidenceIngestor, MemoryManagerError> {
        Ok(EvidenceIngestor {
            manager: self.clone(),
            user_key: validate_key("user key", user_key.into())?,
            agent_key: validate_key("agent key", agent_key.into())?,
        })
    }

    /// Creates a hybrid retrieval handle for one authorized scope.
    pub fn retriever(
        &self,
        user_key: impl Into<String>,
        agent_key: impl Into<String>,
    ) -> Result<MemoryRetriever, MemoryManagerError> {
        Ok(MemoryRetriever {
            manager: self.clone(),
            user_key: validate_key("user key", user_key.into())?,
            agent_key: validate_key("agent key", agent_key.into())?,
        })
    }

    /// Creates a background reconciliation handle for one scope.
    pub fn reconciler(
        &self,
        user_key: impl Into<String>,
        agent_key: impl Into<String>,
    ) -> Result<MemoryReconciliation, MemoryManagerError> {
        if self.inner.reconciliation_mode == ReconciliationMode::Disabled {
            return Err(MemoryManagerError::ReconciliationDisabled);
        }
        Ok(MemoryReconciliation {
            manager: self.clone(),
            user_key: validate_key("user key", user_key.into())?,
            agent_key: validate_key("agent key", agent_key.into())?,
        })
    }
}

/// Deferred-validation builder for a memory manager.
pub struct MemoryManagerBuilder {
    pool: mool::DbPool,
    memory_extractor: Option<Arc<dyn MemoryExtractor>>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    entity_extractor: Option<Arc<dyn EntityExtractor>>,
    reconciler: Option<Arc<dyn MemoryReconciler>>,
    reranker: Option<Arc<dyn MemoryReranker>>,
    limits: MemoryLimits,
    reconciliation_mode: ReconciliationMode,
    errors: Vec<String>,
}

impl MemoryManagerBuilder {
    fn new(pool: mool::DbPool) -> Self {
        Self {
            pool,
            memory_extractor: None,
            embedding_provider: None,
            entity_extractor: None,
            reconciler: None,
            reranker: None,
            limits: MemoryLimits::default(),
            reconciliation_mode: ReconciliationMode::Required,
            errors: Vec::new(),
        }
    }

    /// Sets the required one-pass memory extractor.
    pub fn memory_extractor(mut self, provider: impl MemoryExtractor + 'static) -> Self {
        self.memory_extractor = Some(Arc::new(provider));
        self
    }

    /// Sets the required batch embedding provider.
    pub fn embedding_provider(mut self, provider: impl EmbeddingProvider + 'static) -> Self {
        self.embedding_provider = Some(Arc::new(provider));
        self
    }

    /// Sets optional fallback and query entity extraction.
    pub fn entity_extractor(mut self, provider: impl EntityExtractor + 'static) -> Self {
        self.entity_extractor = Some(Arc::new(provider));
        self
    }

    /// Sets the relation-classification provider required by default.
    pub fn reconciler(mut self, provider: impl MemoryReconciler + 'static) -> Self {
        self.reconciler = Some(Arc::new(provider));
        self
    }

    /// Sets an optional provider used only by explicitly reranked searches.
    pub fn reranker(mut self, provider: impl MemoryReranker + 'static) -> Self {
        self.reranker = Some(Arc::new(provider));
        self
    }

    /// Replaces provider-output and candidate bounds.
    pub fn limits(mut self, limits: MemoryLimits) -> Self {
        validate_limits(&limits, &mut self.errors);
        self.limits = limits;
        self
    }

    /// Explicitly enables or disables background relation reconciliation.
    pub fn reconciliation_mode(mut self, mode: ReconciliationMode) -> Self {
        self.reconciliation_mode = mode;
        self
    }

    /// Validates providers and the migrated database profile without running DDL.
    pub async fn build(mut self) -> Result<MemoryManager, MemoryManagerError> {
        let extractor = self.memory_extractor.take().unwrap_or_else(|| {
            self.errors.push("memory extractor is required".to_owned());
            Arc::new(MissingExtractor)
        });
        let embedder = self.embedding_provider.take().unwrap_or_else(|| {
            self.errors
                .push("embedding provider is required".to_owned());
            Arc::new(MissingEmbedder)
        });
        if self.reconciliation_mode == ReconciliationMode::Required && self.reconciler.is_none() {
            self.errors.push(
                "reconciler is required unless reconciliation is explicitly disabled".to_owned(),
            );
        }
        let embedding_profile = embedder.profile();
        validate_profile(&embedding_profile, &mut self.errors);
        if !self.errors.is_empty() {
            return Err(MemoryManagerError::InvalidConfiguration(self.errors));
        }
        let repository = MemoryRepository::new(self.pool);
        let database_profile = repository.profile().await?;
        validate_database_profile(
            &database_profile,
            &embedding_profile,
            extractor.revision(),
            self.reconciler.as_deref(),
        )?;
        Ok(MemoryManager {
            inner: Arc::new(ManagerInner {
                repository,
                memory_extractor: extractor,
                embedding_provider: embedder,
                entity_extractor: self.entity_extractor,
                reconciler: self.reconciler,
                reranker: self.reranker,
                limits: self.limits,
                reconciliation_mode: self.reconciliation_mode,
                embedding_dimensions: embedding_profile.dimensions as i32,
                text_search_configuration: database_profile.text_search_configuration,
            }),
        })
    }
}

/// Scope-bound evidence ingestion and lifecycle API.
#[derive(Clone)]
pub struct EvidenceIngestor {
    manager: MemoryManager,
    user_key: String,
    agent_key: String,
}

impl EvidenceIngestor {
    /// Accepts evidence and makes its extracted claims searchable.
    pub async fn submit(
        &self,
        evidence_key: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<EvidenceReceipt, MemoryManagerError> {
        self.submit_with(EvidenceSubmission::new(evidence_key, content)?)
            .await
    }

    /// Accepts evidence with an explicit temporal anchor and metadata.
    pub async fn submit_with(
        &self,
        submission: EvidenceSubmission,
    ) -> Result<EvidenceReceipt, MemoryManagerError> {
        validate_submission(&submission, &self.manager.inner.limits)?;
        let (evidence, accepted) = self.accept(submission).await?;
        if evidence.processing_state == "ready" || evidence.stale {
            return self.receipt(&evidence, accepted).await;
        }
        self.process(evidence, accepted).await
    }

    /// Loads one scoped evidence item without its claim payloads.
    pub async fn get(&self, evidence_key: &str) -> Result<Option<Evidence>, MemoryManagerError> {
        self.manager
            .inner
            .repository
            .evidence(&self.user_key, &self.agent_key, evidence_key)
            .await
            .map_err(Into::into)
    }

    /// Retries pending or failed processing idempotently.
    pub async fn retry(&self, evidence_key: &str) -> Result<EvidenceReceipt, MemoryManagerError> {
        let evidence = self.evidence_row(evidence_key).await?;
        if evidence.processing_state == "ready" || evidence.stale {
            return self.receipt(&evidence, false).await;
        }
        self.process(evidence, false).await
    }

    /// One-way stales evidence and all directly derived claims atomically.
    pub async fn mark_stale(&self, evidence_key: &str) -> Result<(), MemoryManagerError> {
        self.manager
            .inner
            .repository
            .mark_stale(
                &self.user_key,
                &self.agent_key,
                evidence_key,
                self.manager.inner.limits.max_foreground_projection_nodes,
            )
            .await
            .map_err(Into::into)
    }

    /// Explicitly deletes evidence and cascades its derived claims and links.
    pub async fn delete(&self, evidence_key: &str) -> Result<(), MemoryManagerError> {
        self.manager
            .inner
            .repository
            .delete_evidence(
                &self.user_key,
                &self.agent_key,
                evidence_key,
                self.manager.inner.limits.max_foreground_projection_nodes,
            )
            .await
            .map_err(Into::into)
    }

    /// Constructs and durably accepts immutable evidence before any provider work.
    async fn accept(
        &self,
        submission: EvidenceSubmission,
    ) -> Result<(EvidenceRow, bool), MemoryManagerError> {
        let now = Utc::now();
        let row = EvidenceRow {
            id: EvidenceId::new().as_uuid(),
            user_key: self.user_key.clone(),
            agent_key: self.agent_key.clone(),
            evidence_key: submission.evidence_key,
            content_hash: sha256(&submission.content),
            content: submission.content,
            metadata: submission.metadata,
            observed_at: submission.observed_at.unwrap_or(now),
            created_at: now,
            processed_at: None,
            processing_state: "pending".to_owned(),
            processing_token: None,
            processing_lease_until: None,
            processing_attempts: 0,
            published_revision: None,
            reconciliation_state: "not_required".to_owned(),
            extractor_revision: self.manager.inner.memory_extractor.revision().to_owned(),
            reconciler_revision: self
                .manager
                .inner
                .reconciler
                .as_ref()
                .map(|value| value.revision().to_owned()),
            error_code: None,
            stale: false,
        };
        self.manager
            .inner
            .repository
            .accept_evidence(row)
            .await
            .map_err(Into::into)
    }

    /// Owns one retryable extraction attempt after evidence acceptance.
    async fn process(
        &self,
        evidence: EvidenceRow,
        accepted: bool,
    ) -> Result<EvidenceReceipt, MemoryManagerError> {
        let started = Instant::now();
        let now = Utc::now();
        let token = uuid::Uuid::now_v7();
        let start = self
            .manager
            .inner
            .repository
            .try_begin_processing(
                &evidence,
                token,
                now,
                now + chrono::Duration::seconds(i64::from(
                    self.manager.inner.limits.processing_lease_seconds,
                )),
            )
            .await?;
        let (evidence, token) = match start {
            ProcessingStart::Acquired { evidence, token } => (evidence, token),
            ProcessingStart::Ready(evidence)
            | ProcessingStart::Stale(evidence)
            | ProcessingStart::InProgress(evidence) => {
                return self.receipt(&evidence, accepted).await;
            }
        };
        match self.extract_and_prepare(&evidence).await {
            Ok(prepared) => {
                self.persist_processed(evidence, token, prepared, accepted, started)
                    .await
            }
            Err(error) => {
                self.fail_processing(evidence.id, token, error, started)
                    .await
            }
        }
    }

    /// Atomically publishes prepared claims and emits bounded processing telemetry.
    async fn persist_processed(
        &self,
        evidence: EvidenceRow,
        processing_token: uuid::Uuid,
        prepared: Vec<PreparedMemory>,
        accepted: bool,
        started: Instant,
    ) -> Result<EvidenceReceipt, MemoryManagerError> {
        let required = self.manager.inner.reconciliation_mode == ReconciliationMode::Required
            && !prepared.is_empty();
        self.manager
            .inner
            .repository
            .persist_memories(&evidence, processing_token, &prepared, required)
            .await?;
        tracing::info!(target: "pravah_memory", operation = "evidence_process",
            evidence_id = %evidence.id, claim_count = prepared.len(),
            latency_ms = started.elapsed().as_millis(), "evidence claims became searchable");
        let mut ready = evidence;
        ready.processing_state = "ready".to_owned();
        ready.processing_token = None;
        ready.processing_lease_until = None;
        ready.reconciliation_state = if required { "pending" } else { "not_required" }.to_owned();
        Ok(receipt(&ready, prepared.len(), accepted, required))
    }

    /// Retains failed evidence for retry while persisting only a safe error code.
    async fn fail_processing(
        &self,
        evidence_id: uuid::Uuid,
        processing_token: uuid::Uuid,
        error: MemoryManagerError,
        started: Instant,
    ) -> Result<EvidenceReceipt, MemoryManagerError> {
        let _ = self
            .manager
            .inner
            .repository
            .mark_failed(evidence_id, processing_token, error.safe_code())
            .await;
        tracing::warn!(target: "pravah_memory", operation = "evidence_process",
            evidence_id = %evidence_id, error_code = error.safe_code(),
            latency_ms = started.elapsed().as_millis(), "evidence processing failed");
        Err(error)
    }

    /// Validates one-pass extraction and prepares embeddings and entity links concurrently.
    async fn extract_and_prepare(
        &self,
        evidence: &EvidenceRow,
    ) -> Result<Vec<PreparedMemory>, MemoryManagerError> {
        let extracted = self
            .manager
            .inner
            .memory_extractor
            .extract(ExtractionRequest {
                evidence: evidence.content.clone(),
                observed_at: evidence.observed_at,
                max_memories: self.manager.inner.limits.max_memories_per_evidence,
                memory_token_target: self.manager.inner.limits.memory_token_target,
            })
            .await?;
        validate_extracted(&extracted, &self.manager.inner.limits)?;
        if extracted.is_empty() {
            return Ok(Vec::new());
        }
        let texts = extracted
            .iter()
            .map(|memory| memory.text.clone())
            .collect::<Vec<_>>();
        let missing = missing_entity_inputs(&extracted);
        let embeddings = self.manager.inner.embedding_provider.embed(&texts);
        let fallback = extract_fallback(self.manager.inner.entity_extractor.as_deref(), &missing);
        let (embeddings, fallback) = tokio::join!(embeddings, fallback);
        let embeddings = embeddings?;
        let fallback = fallback?;
        validate_embeddings(
            &embeddings,
            extracted.len(),
            self.manager.inner.embedding_dimensions,
        )?;
        prepare_memories(
            extracted,
            embeddings,
            missing,
            fallback,
            &self.manager.inner.limits,
        )
    }

    async fn evidence_row(&self, evidence_key: &str) -> Result<EvidenceRow, MemoryManagerError> {
        self.manager
            .inner
            .repository
            .evidence_row(&self.user_key, &self.agent_key, evidence_key)
            .await?
            .ok_or(MemoryManagerError::EvidenceNotFound)
    }

    async fn receipt(
        &self,
        evidence: &EvidenceRow,
        accepted: bool,
    ) -> Result<EvidenceReceipt, MemoryManagerError> {
        let count = self
            .manager
            .inner
            .repository
            .memory_count(evidence.id)
            .await?;
        Ok(receipt(
            evidence,
            count,
            accepted,
            matches!(
                evidence.reconciliation_state.as_str(),
                "pending" | "failed" | "processing"
            ),
        ))
    }
}

/// Scope-bound hybrid and temporal retrieval API.
#[derive(Clone)]
pub struct MemoryRetriever {
    manager: MemoryManager,
    user_key: String,
    agent_key: String,
}

impl MemoryRetriever {
    /// Runs a default current hybrid search without a generative LLM call.
    pub async fn search(
        &self,
        text: impl Into<String>,
    ) -> Result<Vec<SearchResult>, MemoryManagerError> {
        self.search_with(SearchRequest::new(text)?).await
    }

    /// Runs configurable hybrid, entity, stale, and temporal retrieval.
    pub async fn search_with(
        &self,
        mut request: SearchRequest,
    ) -> Result<Vec<SearchResult>, MemoryManagerError> {
        analyze_temporal_query(&mut request, Utc::now());
        let started = Instant::now();
        validate_search(&request)?;
        let embedding = self.prepare_query(&mut request).await?;
        let final_limit = request.limit;
        let mut database_request = request.clone();
        if let Some(candidate_limit) = request.rerank_candidate_limit {
            database_request.limit = candidate_limit;
        }
        let results = hybrid_search(
            &self.manager.inner.repository,
            (&self.user_key, &self.agent_key),
            &database_request,
            embedding.as_ref(),
            self.manager.inner.embedding_dimensions,
            &self.manager.inner.text_search_configuration,
            self.manager.inner.limits.max_retrieval_relation_edges,
        )
        .await
        .map_err(MemoryManagerError::from)?;
        let results = self
            .rerank_if_requested(&request, results, final_limit)
            .await?;
        tracing::info!(target: "pravah_memory", operation = "search",
            result_count = results.len(), candidate_limit = request.candidate_limit,
            latency_ms = started.elapsed().as_millis(), "memory search completed");
        Ok(results)
    }

    /// Applies explicit post-fusion reranking outside the database transaction.
    async fn rerank_if_requested(
        &self,
        request: &SearchRequest,
        results: Vec<SearchResult>,
        limit: u32,
    ) -> Result<Vec<SearchResult>, MemoryManagerError> {
        if request.rerank_candidate_limit.is_none() {
            return Ok(results);
        }
        let reranker = self
            .manager
            .inner
            .reranker
            .as_deref()
            .ok_or(MemoryManagerError::RerankerDisabled)?;
        let candidates = results
            .iter()
            .map(|result| RerankCandidate {
                id: result.memory.id,
                text: result.memory.text.clone(),
                fused_score: result.score,
            })
            .collect();
        let ranked = reranker
            .rerank(RerankRequest {
                query: request.text.clone(),
                candidates,
                limit,
            })
            .await?;
        apply_reranking(results, ranked, limit)
    }

    /// Concurrently prepares only the enabled query embedding and entity channels.
    async fn prepare_query(
        &self,
        request: &mut SearchRequest,
    ) -> Result<Option<super::Embedding>, MemoryManagerError> {
        let (embeddings, entities) =
            tokio::join!(self.query_embedding(request), self.query_entities(request),);
        let embedding = match embeddings? {
            Some(mut embeddings) => {
                validate_embeddings(&embeddings, 1, self.manager.inner.embedding_dimensions)?;
                Some(embeddings.remove(0))
            }
            None => None,
        };
        if request.entity_keys.is_empty() {
            request.entity_keys = entities
                .into_iter()
                .flatten()
                .map(|entity| entity.entity_key)
                .collect();
        }
        Ok(embedding)
    }

    /// Embeds the query only when the vector channel is enabled.
    async fn query_embedding(
        &self,
        request: &SearchRequest,
    ) -> Result<Option<Vec<super::Embedding>>, ProviderError> {
        if request.weights.vector == 0.0 {
            return Ok(None);
        }
        self.manager
            .inner
            .embedding_provider
            .embed(std::slice::from_ref(&request.text))
            .await
            .map(Some)
    }

    /// Treats optional query entity extraction failure as a skipped channel.
    async fn query_entities(&self, request: &SearchRequest) -> Vec<Vec<ExtractedEntity>> {
        if request.weights.entity == 0.0 || !request.entity_keys.is_empty() {
            return Vec::new();
        }
        let inputs = [EntityExtractionInput {
            position: 0,
            text: request.text.clone(),
        }];
        match extract_fallback(self.manager.inner.entity_extractor.as_deref(), &inputs).await {
            Ok(entities) => entities,
            Err(error) => {
                tracing::warn!(target: "pravah_memory", operation = "query_entity_extraction",
                    error_code = error.code, "optional query entity channel was skipped");
                Vec::new()
            }
        }
    }
}

/// Scope-bound asynchronous relation reconciliation API.
#[derive(Clone)]
pub struct MemoryReconciliation {
    manager: MemoryManager,
    user_key: String,
    agent_key: String,
}

impl MemoryReconciliation {
    /// Rebuilds a deferred current-retrieval projection under an explicit scope bound.
    pub async fn refresh_projection(&self, claim_limit: u32) -> Result<usize, MemoryManagerError> {
        if claim_limit == 0 {
            return Err(MemoryManagerError::InvalidInput(
                "projection claim limit must be greater than zero".to_owned(),
            ));
        }
        let token = self.acquire_lease().await?;
        let result = self
            .manager
            .inner
            .repository
            .rebuild_projection(&self.user_key, &self.agent_key, token, claim_limit)
            .await;
        let release = self.release_lease(token).await;
        match result {
            Ok(count) => {
                release?;
                Ok(count)
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Reconciles up to `batch_limit` pending evidence items for this scope.
    pub async fn reconcile_pending(&self, batch_limit: u32) -> Result<usize, MemoryManagerError> {
        let started = Instant::now();
        self.validate_batch_limit(batch_limit)?;
        let Some(token) = self.try_acquire_lease().await? else {
            return Ok(0);
        };
        let completed = self.run_leased_batch(token, batch_limit).await?;
        tracing::info!(target: "pravah_memory", operation = "reconcile_pending",
            evidence_count = completed, latency_ms = started.elapsed().as_millis(),
            "pending memory reconciliation completed");
        Ok(completed)
    }

    fn validate_batch_limit(&self, batch_limit: u32) -> Result<(), MemoryManagerError> {
        let maximum = self.manager.inner.limits.max_reconciliation_batch;
        if batch_limit == 0 || batch_limit > maximum {
            return Err(MemoryManagerError::InvalidInput(format!(
                "batch limit must be between 1 and {maximum}"
            )));
        }
        Ok(())
    }

    /// Completes owned work and always attempts a token-fenced lease release.
    async fn run_leased_batch(
        &self,
        token: uuid::Uuid,
        batch_limit: u32,
    ) -> Result<usize, MemoryManagerError> {
        let result = self.reconcile_owned(token, batch_limit).await;
        let release = self.release_lease(token).await;
        match result {
            Ok(completed) => {
                release?;
                Ok(completed)
            }
            Err(error) => Err(error),
        }
    }

    /// Attempts one expiring multi-instance reconciliation lease.
    async fn try_acquire_lease(&self) -> Result<Option<uuid::Uuid>, MemoryManagerError> {
        let now = Utc::now();
        let token = uuid::Uuid::now_v7();
        match self
            .manager
            .inner
            .repository
            .try_acquire_reconciliation(
                &self.user_key,
                &self.agent_key,
                token,
                now,
                now + chrono::Duration::seconds(i64::from(
                    self.manager.inner.limits.reconciliation_lease_seconds,
                )),
            )
            .await
        {
            Ok(_) => Ok(Some(token)),
            Err(RepositoryError::ReconciliationBusy) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Acquires one expiring multi-instance reconciliation lease.
    async fn acquire_lease(&self) -> Result<uuid::Uuid, MemoryManagerError> {
        self.try_acquire_lease()
            .await?
            .ok_or(MemoryManagerError::ReconciliationRace)
    }

    /// Releases a lease only when this worker still owns it.
    async fn release_lease(&self, token: uuid::Uuid) -> Result<(), MemoryManagerError> {
        self.manager
            .inner
            .repository
            .release_reconciliation(&self.user_key, &self.agent_key, token)
            .await
            .map_err(Into::into)
    }

    /// Reconciles one leased batch while every commit remains token-fenced.
    async fn reconcile_owned(
        &self,
        token: uuid::Uuid,
        batch_limit: u32,
    ) -> Result<usize, MemoryManagerError> {
        let pending = self
            .manager
            .inner
            .repository
            .pending_claims(&self.user_key, &self.agent_key, batch_limit)
            .await?;
        let mut completed = 0;
        for (evidence, claims) in pending {
            if let Err(error) = self.reconcile_evidence(token, &evidence, &claims).await {
                let _ = self
                    .manager
                    .inner
                    .repository
                    .mark_reconciliation_failed(
                        &self.user_key,
                        &self.agent_key,
                        evidence.id,
                        token,
                        error.safe_code(),
                    )
                    .await;
                return Err(error);
            }
            completed += 1;
        }
        Ok(completed)
    }

    /// Retries optimistic scope-revision races without persisting partial relations.
    async fn reconcile_evidence(
        &self,
        reconciliation_token: uuid::Uuid,
        evidence: &EvidenceRow,
        claims: &[MemoryRow],
    ) -> Result<(), MemoryManagerError> {
        for attempt in 0..3 {
            match self
                .reconcile_once(reconciliation_token, evidence, claims)
                .await
            {
                Err(MemoryManagerError::ReconciliationRace) if attempt < 2 => continue,
                result => return result,
            }
        }
        Err(MemoryManagerError::ReconciliationRace)
    }

    /// Computes and commits one optimistic reconciliation attempt.
    async fn reconcile_once(
        &self,
        reconciliation_token: uuid::Uuid,
        evidence: &EvidenceRow,
        claims: &[MemoryRow],
    ) -> Result<(), MemoryManagerError> {
        let reconciler = self
            .manager
            .inner
            .reconciler
            .as_ref()
            .ok_or(MemoryManagerError::ReconciliationDisabled)?;
        let expected_revision = self
            .manager
            .inner
            .repository
            .scope_revision(&self.user_key, &self.agent_key)
            .await?;
        let candidates = self.reconciliation_candidates(evidence, claims).await?;
        let (mut decisions, ambiguous) = deterministic_relations(claims, &candidates);
        decisions.extend(semantic_relations(reconciler.as_ref(), ambiguous).await?);
        self.manager
            .inner
            .repository
            .commit_reconciliation(
                evidence,
                reconciliation_token,
                expected_revision,
                &decisions,
                reconciler.revision(),
                self.manager.inner.limits.max_foreground_projection_nodes,
            )
            .await
            .map_err(Into::into)
    }

    /// Loads a candidate set whose total provider group stays within its hard bound.
    async fn reconciliation_candidates(
        &self,
        evidence: &EvidenceRow,
        claims: &[MemoryRow],
    ) -> Result<Vec<MemoryRow>, MemoryManagerError> {
        let limits = &self.manager.inner.limits;
        let capacity = limits.max_reconciliation_group.saturating_sub(claims.len()) as u32;
        self.manager
            .inner
            .repository
            .reconciliation_candidates(
                evidence,
                claims,
                capacity,
                limits.reconciliation_candidates_per_claim,
                limits.reconciliation_max_cosine_distance,
                &self.manager.inner.text_search_configuration,
            )
            .await
            .map_err(Into::into)
    }
}

/// Errors exposed by manager construction and runtime operations.
#[derive(Debug, Error)]
pub enum MemoryManagerError {
    /// Builder validation found every listed configuration problem.
    #[error("invalid memory manager configuration: {0:?}")]
    InvalidConfiguration(Vec<String>),
    /// A request violated a public input bound.
    #[error("invalid memory input: {0}")]
    InvalidInput(String),
    /// The migrated singleton profile does not match configured providers.
    #[error("memory database profile mismatch: {0}")]
    ProfileMismatch(String),
    /// A scoped evidence key was not found.
    #[error("evidence was not found")]
    EvidenceNotFound,
    /// A scoped evidence key is already bound to different immutable content.
    #[error("evidence key is already bound to different content")]
    EvidenceKeyConflict,
    /// Reconciliation was explicitly disabled.
    #[error("memory reconciliation is disabled")]
    ReconciliationDisabled,
    /// A request enabled reranking without configuring a provider.
    #[error("memory reranking was requested but no reranker is configured")]
    RerankerDisabled,
    /// Relation hydration exceeded its configured edge bound before a complete view was available.
    #[error("memory relation expansion exceeded its configured retrieval limit")]
    RelationExpansionLimit,
    /// Another writer changed the scope while reconciliation providers were running.
    #[error("memory scope changed during reconciliation")]
    ReconciliationRace,
    /// An evidence attempt completed after its lease was replaced or invalidated.
    #[error("evidence processing ownership changed before publication")]
    ProcessingSuperseded,
    /// A required external provider failed safely.
    #[error(transparent)]
    Provider(#[from] ProviderError),
    /// A core value failed validation.
    #[error(transparent)]
    Memory(#[from] super::MemoryError),
    /// A tracked retrieval receipt violated recall-reporting invariants.
    #[error(transparent)]
    Recall(#[from] super::RecallError),
    /// PostgreSQL persistence or stored-state validation failed.
    #[error("memory persistence failed: {0}")]
    Persistence(String),
}
