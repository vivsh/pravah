#![cfg(feature = "postgres")]

use std::str::FromStr;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::Notify;

use async_trait::async_trait;
use pravah_memory::postgres::{MemoryProfile, MemorySchemaExt};
use pravah_memory::{
    Embedding, EmbeddingProfile, EmbeddingProvider, ExtractedEntity, ExtractedMemory,
    ExtractionRequest, MemoryExtractor, MemoryKind, MemoryLimits, MemoryManager, MemoryReconciler,
    ProcessingState, ProviderError, ReconciliationDecision, ReconciliationGroup,
    ReconciliationOutcome, SearchRequest, SearchWeights, TemporalMetadata, TemporalPrecision,
    TemporalState,
};

struct FlakyExtractor {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl MemoryExtractor for FlakyExtractor {
    fn revision(&self) -> &str {
        "live-extractor-v1"
    }

    async fn extract(
        &self,
        request: ExtractionRequest,
    ) -> Result<Vec<ExtractedMemory>, ProviderError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(ProviderError::new(
                "transient_extraction",
                "intentional live-test failure",
                true,
            ));
        }
        Ok(vec![ExtractedMemory {
            text: request.evidence,
            entities: Some(vec![ExtractedEntity {
                entity_key: "seat:aisle".to_owned(),
                kind: "preference".to_owned(),
                canonical_name: "Aisle seat".to_owned(),
                aliases: vec!["aisle".to_owned()],
                metadata: serde_json::json!({}),
            }]),
            kind: MemoryKind::Fact,
            temporal: TemporalMetadata::default(),
            metadata: serde_json::json!({}),
        }])
    }
}

struct Embedder;

#[async_trait]
impl EmbeddingProvider for Embedder {
    fn profile(&self) -> EmbeddingProfile {
        EmbeddingProfile {
            model: "live-embedding".to_owned(),
            revision: "v1".to_owned(),
            dimensions: 3,
            document_format_revision: "claim-text-v1".to_owned(),
        }
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, ProviderError> {
        texts
            .iter()
            .map(|_| {
                Embedding::new(vec![1.0, 0.0, 0.0]).map_err(|error| {
                    ProviderError::new("invalid_test_embedding", error.to_string(), false)
                })
            })
            .collect()
    }
}

struct Reconciler;

#[async_trait]
impl MemoryReconciler for Reconciler {
    fn revision(&self) -> &str {
        "live-reconciler-v1"
    }

    async fn reconcile(
        &self,
        _group: ReconciliationGroup,
    ) -> Result<Vec<ReconciliationDecision>, ProviderError> {
        Ok(Vec::new())
    }
}

struct BlockingExtractor {
    calls: Arc<AtomicUsize>,
    started: Arc<Notify>,
    release: Arc<Notify>,
}

struct StaticExtractor;

#[async_trait]
impl MemoryExtractor for StaticExtractor {
    fn revision(&self) -> &str {
        "live-extractor-v1"
    }

    async fn extract(
        &self,
        request: ExtractionRequest,
    ) -> Result<Vec<ExtractedMemory>, ProviderError> {
        Ok(vec![ExtractedMemory {
            text: request.evidence,
            entities: Some(Vec::new()),
            kind: MemoryKind::Fact,
            temporal: TemporalMetadata::default(),
            metadata: serde_json::json!({}),
        }])
    }
}

struct BlockingReconciler {
    calls: Arc<AtomicUsize>,
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl MemoryReconciler for BlockingReconciler {
    fn revision(&self) -> &str {
        "live-reconciler-v1"
    }

    async fn reconcile(
        &self,
        _group: ReconciliationGroup,
    ) -> Result<Vec<ReconciliationDecision>, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        self.release.notified().await;
        Ok(Vec::new())
    }
}

struct TemporalExtractor;

#[async_trait]
impl MemoryExtractor for TemporalExtractor {
    fn revision(&self) -> &str {
        "live-extractor-v1"
    }

    async fn extract(
        &self,
        request: ExtractionRequest,
    ) -> Result<Vec<ExtractedMemory>, ProviderError> {
        let valid_from = if request.evidence.contains("Paris") {
            Some(timestamp("2020-01-01T00:00:00Z"))
        } else if request.evidence.contains("Berlin") {
            Some(timestamp("2024-01-01T00:00:00Z"))
        } else {
            None
        };
        Ok(vec![ExtractedMemory {
            text: request.evidence,
            entities: Some(Vec::new()),
            kind: MemoryKind::State,
            temporal: TemporalMetadata {
                valid_from,
                valid_until: None,
                event_at: None,
                precision: TemporalPrecision::Day,
                state: TemporalState::Ongoing,
            },
            metadata: serde_json::json!({}),
        }])
    }
}

struct TemporalReconciler;

#[async_trait]
impl MemoryReconciler for TemporalReconciler {
    fn revision(&self) -> &str {
        "live-reconciler-v1"
    }

    async fn reconcile(
        &self,
        group: ReconciliationGroup,
    ) -> Result<Vec<ReconciliationDecision>, ProviderError> {
        let mut decisions = Vec::new();
        for new_claim in &group.new_claims {
            for candidate in &group.candidates {
                if new_claim.text.contains("Berlin") && candidate.text.contains("Paris") {
                    decisions.push(ReconciliationDecision {
                        from_memory_id: new_claim.id,
                        to_memory_id: candidate.id,
                        outcome: ReconciliationOutcome::Supersedes {
                            effective_at: Some(timestamp("2024-01-01T00:00:00Z")),
                        },
                    });
                }
            }
        }
        Ok(decisions)
    }
}

struct ConflictReconciler;

#[async_trait]
impl MemoryReconciler for ConflictReconciler {
    fn revision(&self) -> &str {
        "live-reconciler-v1"
    }

    async fn reconcile(
        &self,
        group: ReconciliationGroup,
    ) -> Result<Vec<ReconciliationDecision>, ProviderError> {
        let mut decisions = Vec::new();
        for new_claim in &group.new_claims {
            for candidate in &group.candidates {
                if new_claim.text.contains("tea") && candidate.text.contains("tea") {
                    decisions.push(ReconciliationDecision {
                        from_memory_id: new_claim.id,
                        to_memory_id: candidate.id,
                        outcome: ReconciliationOutcome::Conflicts,
                    });
                }
            }
        }
        Ok(decisions)
    }
}

#[async_trait]
impl MemoryExtractor for BlockingExtractor {
    fn revision(&self) -> &str {
        "live-extractor-v1"
    }

    async fn extract(
        &self,
        request: ExtractionRequest,
    ) -> Result<Vec<ExtractedMemory>, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        self.release.notified().await;
        Ok(vec![ExtractedMemory {
            text: request.evidence,
            entities: Some(Vec::new()),
            kind: MemoryKind::Fact,
            temporal: TemporalMetadata::default(),
            metadata: serde_json::json!({}),
        }])
    }
}

/// Verifies evidence durability before provider failure, retry idempotency, search, and staleness on PostgreSQL/pgvector.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn evidence_pipeline_executes_on_live_postgres() {
    let (sqlx_pool, schema_name) = test_database().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let manager = test_manager(&sqlx_pool, &calls).await;
    exercise_pipeline(&manager, &calls).await;
    drop(manager);
    execute_sql(&sqlx_pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Verifies concurrent identical submissions execute providers once and expose processing state.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn concurrent_submission_has_one_processing_owner() {
    let (pool, schema_name) = test_database().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let manager = blocking_manager(&pool, &calls, &started, &release).await;
    let first_manager = manager.clone();
    let first = tokio::spawn(async move {
        first_manager
            .ingestor("user-race", "agent")
            .unwrap()
            .submit("same:v1", "The user likes tea.")
            .await
    });
    started.notified().await;
    let concurrent = manager
        .ingestor("user-race", "agent")
        .unwrap()
        .submit("same:v1", "The user likes tea.")
        .await
        .unwrap();
    assert_eq!(concurrent.processing, ProcessingState::Processing);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    release.notify_one();
    assert_eq!(
        first.await.unwrap().unwrap().processing,
        ProcessingState::Ready
    );
    drop(manager);
    execute_sql(&pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Verifies staleness fences provider output that completes after source invalidation.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn stale_evidence_cannot_publish_inflight_claims() {
    let (pool, schema_name) = test_database().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let manager = blocking_manager(&pool, &calls, &started, &release).await;
    let first_manager = manager.clone();
    let first = tokio::spawn(async move {
        first_manager
            .ingestor("user-stale", "agent")
            .unwrap()
            .submit("source:v1", "The user likes coffee.")
            .await
    });
    started.notified().await;
    manager
        .ingestor("user-stale", "agent")
        .unwrap()
        .mark_stale("source:v1")
        .await
        .unwrap();
    release.notify_one();
    assert!(first.await.unwrap().is_err());
    let evidence = manager
        .ingestor("user-stale", "agent")
        .unwrap()
        .get("source:v1")
        .await
        .unwrap()
        .unwrap();
    assert!(evidence.stale);
    assert!(
        manager
            .retriever("user-stale", "agent")
            .unwrap()
            .search("coffee")
            .await
            .unwrap()
            .is_empty()
    );
    drop(manager);
    execute_sql(&pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Verifies the scope lease prevents duplicate multi-instance reconciliation work.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn reconciliation_scope_has_one_lease_owner() {
    let (pool, schema_name) = test_database().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let manager = reconciliation_blocking_manager(&pool, &calls, &started, &release).await;
    let ingestor = manager.ingestor("user-reconcile", "agent").unwrap();
    ingestor.submit("one", "The user likes tea.").await.unwrap();
    ingestor
        .submit("two", "The user likes coffee.")
        .await
        .unwrap();
    let first_manager = manager.clone();
    let first = tokio::spawn(async move {
        first_manager
            .reconciler("user-reconcile", "agent")
            .unwrap()
            .reconcile_pending(1)
            .await
    });
    started.notified().await;
    let competing = manager
        .reconciler("user-reconcile", "agent")
        .unwrap()
        .reconcile_pending(1)
        .await
        .unwrap();
    assert_eq!(competing, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    release.notify_one();
    assert_eq!(first.await.unwrap().unwrap(), 1);
    drop(manager);
    execute_sql(&pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Verifies effective supersession, historical resolution, and stale-superseder reactivation.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn temporal_supersession_resolves_current_and_as_of_views() {
    let (pool, schema_name) = test_database().await;
    let manager = temporal_manager(&pool).await;
    let ingestor = manager.ingestor("user-time", "agent").unwrap();
    let before_ingestion = chrono::Utc::now();
    ingestor
        .submit("home:v1", "The user lives in Paris.")
        .await
        .unwrap();
    ingestor
        .submit("home:v2", "The user lives in Berlin.")
        .await
        .unwrap();
    assert_eq!(
        manager
            .reconciler("user-time", "agent")
            .unwrap()
            .reconcile_pending(10)
            .await
            .unwrap(),
        2
    );
    assert_temporal_retrieval(&manager, &ingestor, before_ingestion).await;
    drop(manager);
    execute_sql(&pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Checks transaction time, valid time, and stale-superseder projection behavior.
async fn assert_temporal_retrieval(
    manager: &MemoryManager,
    ingestor: &pravah_memory::EvidenceIngestor,
    before_ingestion: chrono::DateTime<chrono::Utc>,
) {
    let retriever = manager.retriever("user-time", "agent").unwrap();
    assert!(
        retriever
            .search_with(
                SearchRequest::new("Where did the user live?")
                    .unwrap()
                    .history()
                    .known_at(before_ingestion),
            )
            .await
            .unwrap()
            .is_empty()
    );
    let current = retriever.search("Where does the user live?").await.unwrap();
    assert_eq!(current.len(), 1);
    assert!(current[0].memory.text.contains("Berlin"));
    let historical = retriever
        .search_with(
            SearchRequest::new("Where did the user live?")
                .unwrap()
                .as_of(timestamp("2022-01-01T00:00:00Z")),
        )
        .await
        .unwrap();
    assert_eq!(historical.len(), 1);
    assert!(historical[0].memory.text.contains("Paris"));
    ingestor.mark_stale("home:v2").await.unwrap();
    let reactivated = retriever.search("Where does the user live?").await.unwrap();
    assert_eq!(reactivated.len(), 1);
    assert!(reactivated[0].memory.text.contains("Paris"));
}

/// Verifies oversized foreground components use archive fallback until a bounded rebuild.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn deferred_projection_remains_searchable_and_rebuildable() {
    let (pool, schema_name) = test_database().await;
    let limits = MemoryLimits {
        max_foreground_projection_nodes: 1,
        ..MemoryLimits::default()
    };
    let manager = static_manager(&pool, limits).await;
    let ingestor = manager.ingestor("user-projection", "agent").unwrap();
    ingestor
        .submit("same:v1", "The user prefers quiet rooms.")
        .await
        .unwrap();
    ingestor
        .submit("same:v2", "The user prefers quiet rooms.")
        .await
        .unwrap();
    let reconciler = manager.reconciler("user-projection", "agent").unwrap();
    assert_eq!(reconciler.reconcile_pending(10).await.unwrap(), 2);
    let results = manager
        .retriever("user-projection", "agent")
        .unwrap()
        .search("quiet rooms")
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].support_count, 2);
    assert_eq!(reconciler.refresh_projection(10).await.unwrap(), 2);
    drop(manager);
    execute_sql(&pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Verifies recursive fallback collapses a three-claim chain whose middle claim was not ranked.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn recursive_relation_fallback_resolves_non_ranked_intermediate_claim() {
    let (pool, schema_name) = test_database().await;
    let manager = static_manager(&pool, MemoryLimits::default()).await;
    let ingestor = manager.ingestor("user-chain", "agent").unwrap();
    ingestor.submit("chain:v1", "anchor alpha").await.unwrap();
    ingestor
        .submit("chain:v2", "unranked bridge statement")
        .await
        .unwrap();
    ingestor.submit("chain:v3", "anchor gamma").await.unwrap();
    seed_corroboration_chain(&pool).await;
    let mut request = SearchRequest::new("anchor").unwrap();
    request.weights = SearchWeights {
        lexical: 1.0,
        vector: 0.0,
        entity: 0.0,
        temporal: 0.0,
    };
    let results = manager
        .retriever("user-chain", "agent")
        .unwrap()
        .search_with(request)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].support_count, 3);
    drop(manager);
    execute_sql(&pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Verifies retrieval fails instead of exposing a relation component beyond its edge bound.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn recursive_relation_fallback_rejects_partial_component() {
    let (pool, schema_name) = test_database().await;
    let manager = static_manager(
        &pool,
        MemoryLimits {
            max_retrieval_relation_edges: 1,
            ..MemoryLimits::default()
        },
    )
    .await;
    let ingestor = manager.ingestor("user-chain", "agent").unwrap();
    ingestor.submit("chain:v1", "anchor alpha").await.unwrap();
    ingestor.submit("chain:v2", "bridge").await.unwrap();
    ingestor.submit("chain:v3", "anchor gamma").await.unwrap();
    seed_corroboration_chain(&pool).await;
    let error = manager
        .retriever("user-chain", "agent")
        .unwrap()
        .search("anchor")
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        pravah_memory::MemoryManagerError::RelationExpansionLimit
    ));
    drop(manager);
    execute_sql(&pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Verifies unresolved conflicts retain and annotate both evidence-supported claims.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn unresolved_conflicts_return_both_claims() {
    let (pool, schema_name) = test_database().await;
    let manager = conflict_manager(&pool).await;
    let ingestor = manager.ingestor("user-conflict", "agent").unwrap();
    ingestor
        .submit("tea:v1", "The user likes tea.")
        .await
        .unwrap();
    ingestor
        .submit("tea:v2", "The user dislikes tea.")
        .await
        .unwrap();
    manager
        .reconciler("user-conflict", "agent")
        .unwrap()
        .reconcile_pending(10)
        .await
        .unwrap();
    let results = manager
        .retriever("user-conflict", "agent")
        .unwrap()
        .search("tea")
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| result.conflicts.len() == 1));
    drop(manager);
    execute_sql(&pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Verifies PostgreSQL can plan the partial GIN and HNSW indexes used by current retrieval.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn current_retrieval_specialized_indexes_are_plannable() {
    let (pool, schema_name) = test_database().await;
    let manager = static_manager(&pool, MemoryLimits::default()).await;
    manager
        .ingestor("user-index", "agent")
        .unwrap()
        .submit("claim:v1", "The user likes tea.")
        .await
        .unwrap();
    let vector_plan = explain(
        &pool,
        "SELECT id FROM pravah_memories WHERE stale = FALSE \
         AND current_for_retrieval = TRUE \
         ORDER BY embedding::vector(3) <=> '[1,0,0]'::vector(3) LIMIT 5",
    )
    .await;
    assert!(vector_plan.contains("pravah_memories_current_embedding_idx"));
    let text_plan = explain(
        &pool,
        "SELECT id FROM pravah_memories WHERE stale = FALSE \
         AND current_for_retrieval = TRUE \
         AND search_vector @@ plainto_tsquery('simple', 'tea')",
    )
    .await;
    assert!(text_plan.contains("pravah_memories_current_text_idx"));
    drop(manager);
    execute_sql(&pool, &format!("DROP SCHEMA {schema_name} CASCADE")).await;
}

/// Creates an isolated migrated schema within the configured live test database.
async fn test_database() -> (sqlx::PgPool, String) {
    let database_url = std::env::var("MEMORY_POSTGRES_DATABASE_URL")
        .expect("MEMORY_POSTGRES_DATABASE_URL must select an isolated test database");
    let admin = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(&database_url)
        .await
        .unwrap();
    let schema = format!("pravah_memory_test_{}", uuid::Uuid::now_v7().simple());
    execute_sql(&admin, &format!("CREATE SCHEMA {schema}")).await;
    execute_sql(
        &admin,
        "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public",
    )
    .await;
    drop(admin);
    let options = sqlx::postgres::PgConnectOptions::from_str(&database_url)
        .unwrap()
        .options([("search_path", format!("{schema},public"))]);
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(8)
        .connect_with(options)
        .await
        .unwrap();
    for statement in migration_sql() {
        execute_sql(&pool, &statement).await;
    }
    (pool, schema)
}

/// Seeds a canonical two-edge chain and forces archive-index relation fallback.
async fn seed_corroboration_chain(pool: &sqlx::PgPool) {
    let rows = sqlx::query_as::<_, (uuid::Uuid, uuid::Uuid)>(
        "SELECT memory.id, memory.evidence_id FROM pravah_memories memory \
         JOIN pravah_evidence evidence ON evidence.id = memory.evidence_id \
         WHERE memory.user_key = 'user-chain' AND memory.agent_key = 'agent' \
         ORDER BY evidence.evidence_key",
    )
    .fetch_all(pool)
    .await
    .unwrap();
    assert_eq!(rows.len(), 3);
    insert_corroboration(pool, rows[0], rows[1]).await;
    insert_corroboration(pool, rows[1], rows[2]).await;
    execute_sql(
        pool,
        "UPDATE pravah_memory_scopes SET projection_pending = TRUE \
         WHERE user_key = 'user-chain' AND agent_key = 'agent'",
    )
    .await;
}

/// Inserts one symmetric relation using canonical UUID ordering.
async fn insert_corroboration(
    pool: &sqlx::PgPool,
    left: (uuid::Uuid, uuid::Uuid),
    right: (uuid::Uuid, uuid::Uuid),
) {
    let (from, to) = if left.0 < right.0 {
        (left.0, right.0)
    } else {
        (right.0, left.0)
    };
    sqlx::query(
        "INSERT INTO pravah_memory_relations \
         (from_memory_id, to_memory_id, user_key, agent_key, origin_evidence_id, \
          kind, effective_at, reconciler_revision, created_at) \
         VALUES ($1, $2, 'user-chain', 'agent', $3, 'corroborates', NULL, 'fixture', now())",
    )
    .bind(from)
    .bind(to)
    .bind(right.1)
    .execute(pool)
    .await
    .unwrap();
}

/// Builds a manager whose extraction can be held across competing lifecycle calls.
async fn blocking_manager(
    pool: &sqlx::PgPool,
    calls: &Arc<AtomicUsize>,
    started: &Arc<Notify>,
    release: &Arc<Notify>,
) -> MemoryManager {
    MemoryManager::builder(mool::DbPool::from_pool(pool.clone()))
        .memory_extractor(BlockingExtractor {
            calls: Arc::clone(calls),
            started: Arc::clone(started),
            release: Arc::clone(release),
        })
        .embedding_provider(Embedder)
        .reconciler(Reconciler)
        .build()
        .await
        .unwrap()
}

/// Builds a manager whose reconciliation call can be held while another worker competes.
async fn reconciliation_blocking_manager(
    pool: &sqlx::PgPool,
    calls: &Arc<AtomicUsize>,
    started: &Arc<Notify>,
    release: &Arc<Notify>,
) -> MemoryManager {
    MemoryManager::builder(mool::DbPool::from_pool(pool.clone()))
        .memory_extractor(StaticExtractor)
        .embedding_provider(Embedder)
        .reconciler(BlockingReconciler {
            calls: Arc::clone(calls),
            started: Arc::clone(started),
            release: Arc::clone(release),
        })
        .build()
        .await
        .unwrap()
}

/// Builds deterministic temporal providers for relation-time acceptance tests.
async fn temporal_manager(pool: &sqlx::PgPool) -> MemoryManager {
    MemoryManager::builder(mool::DbPool::from_pool(pool.clone()))
        .memory_extractor(TemporalExtractor)
        .embedding_provider(Embedder)
        .reconciler(TemporalReconciler)
        .build()
        .await
        .unwrap()
}

/// Builds deterministic conflict classification providers.
async fn conflict_manager(pool: &sqlx::PgPool) -> MemoryManager {
    MemoryManager::builder(mool::DbPool::from_pool(pool.clone()))
        .memory_extractor(StaticExtractor)
        .embedding_provider(Embedder)
        .reconciler(ConflictReconciler)
        .build()
        .await
        .unwrap()
}

/// Builds deterministic providers with caller-selected operational limits.
async fn static_manager(pool: &sqlx::PgPool, limits: MemoryLimits) -> MemoryManager {
    MemoryManager::builder(mool::DbPool::from_pool(pool.clone()))
        .memory_extractor(StaticExtractor)
        .embedding_provider(Embedder)
        .reconciler(Reconciler)
        .limits(limits)
        .build()
        .await
        .unwrap()
}

fn timestamp(value: &str) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339(value)
        .unwrap()
        .with_timezone(&chrono::Utc)
}

/// Returns a deterministic PostgreSQL plan with sequential scans disabled for index assertions.
async fn explain(pool: &sqlx::PgPool, sql: &str) -> String {
    let mut connection = pool.acquire().await.unwrap();
    sqlx::query("SET enable_seqscan = off")
        .execute(&mut *connection)
        .await
        .unwrap();
    sqlx::query_scalar::<_, String>(&format!("EXPLAIN (COSTS OFF) {sql}"))
        .fetch_all(&mut *connection)
        .await
        .unwrap()
        .join("\n")
}

/// Builds deterministic providers against the migrated live schema.
async fn test_manager(pool: &sqlx::PgPool, calls: &Arc<AtomicUsize>) -> MemoryManager {
    MemoryManager::builder(mool::DbPool::from_pool(pool.clone()))
        .memory_extractor(FlakyExtractor {
            calls: Arc::clone(calls),
        })
        .embedding_provider(Embedder)
        .reconciler(Reconciler)
        .build()
        .await
        .unwrap()
}

/// Exercises failure recovery, reconciliation, isolation, representative promotion, and cascades.
async fn exercise_pipeline(manager: &MemoryManager, calls: &Arc<AtomicUsize>) {
    let ingestor = manager.ingestor("user-1", "agent-1").unwrap();
    assert!(
        ingestor
            .submit("profile:v1", "The user prefers aisle seats.")
            .await
            .is_err()
    );
    let failed = ingestor.get("profile:v1").await.unwrap().unwrap();
    assert_eq!(failed.processing, ProcessingState::Failed);

    let receipt = ingestor.retry("profile:v1").await.unwrap();
    assert_eq!(receipt.memory_count, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    ingestor
        .submit("profile:v2", "The user prefers aisle seats.")
        .await
        .unwrap();
    assert_eq!(
        manager
            .reconciler("user-1", "agent-1")
            .unwrap()
            .reconcile_pending(10)
            .await
            .unwrap(),
        2
    );
    assert_retrieval_and_lifecycle(manager, &ingestor).await;
}

/// Verifies search isolation, representative promotion, staleness, and deletion cascades.
async fn assert_retrieval_and_lifecycle(
    manager: &MemoryManager,
    ingestor: &pravah_memory::EvidenceIngestor,
) {
    assert_initial_retrieval(manager).await;
    assert_stale_lifecycle(manager, ingestor).await;
}

/// Checks corroboration collapse, aliases, and exact scope isolation.
async fn assert_initial_retrieval(manager: &MemoryManager) {
    let results = manager
        .retriever("user-1", "agent-1")
        .unwrap()
        .search("aisle seats")
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].support_count, 2);
    let mut alias_request = SearchRequest::new("seat preference").unwrap();
    alias_request.entity_keys = vec!["aisle".to_owned()];
    alias_request.weights = SearchWeights {
        lexical: 0.0,
        vector: 0.0,
        entity: 1.0,
        temporal: 0.0,
    };
    assert_eq!(
        manager
            .retriever("user-1", "agent-1")
            .unwrap()
            .search_with(alias_request)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        manager
            .retriever("user-2", "agent-1")
            .unwrap()
            .search("aisle seats")
            .await
            .unwrap()
            .is_empty()
    );
}

/// Checks representative promotion, complete staleness exclusion, and cascades.
async fn assert_stale_lifecycle(
    manager: &MemoryManager,
    ingestor: &pravah_memory::EvidenceIngestor,
) {
    ingestor.mark_stale("profile:v1").await.unwrap();
    let promoted = manager
        .retriever("user-1", "agent-1")
        .unwrap()
        .search("aisle seats")
        .await
        .unwrap();
    assert_eq!(promoted.len(), 1);
    assert_eq!(promoted[0].evidence_key, "profile:v2");
    ingestor.mark_stale("profile:v2").await.unwrap();
    assert!(
        manager
            .retriever("user-1", "agent-1")
            .unwrap()
            .search("aisle seats")
            .await
            .unwrap()
            .is_empty()
    );
    ingestor.delete("profile:v1").await.unwrap();
    ingestor.delete("profile:v2").await.unwrap();
    assert!(ingestor.get("profile:v1").await.unwrap().is_none());
}

/// Lowers the fixed Mool schema into reviewed PostgreSQL migration statements.
fn migration_sql() -> Vec<String> {
    let profile = MemoryProfile::new(
        "live-embedding",
        "v1",
        3,
        "claim-text-v1",
        "live-extractor-v1",
        "live-reconciler-v1",
    );
    let schema = mool::schema().with_memory(profile).build().unwrap();
    let planner = mool::gaman::core::OfflinePlanner::new(mool::gaman::core::Dialect::Postgres);
    let pending = planner.make_migration(schema.clone(), &[]).unwrap_err();
    let mool::gaman::core::OfflineError::NeedsInput(clarifications) = pending else {
        panic!("opaque memory indexes must require review")
    };
    let decisions = clarifications
        .into_iter()
        .map(|clarification| mool::gaman::core::Decision {
            clarification_id: clarification.id,
            answer: mool::gaman::core::Answer::AcceptRisk,
        })
        .collect::<Vec<_>>();
    let migration = planner.make_migration(schema, &decisions).unwrap().unwrap();
    planner.sql_migrate(&[migration]).unwrap()
}

async fn execute_sql(pool: &sqlx::PgPool, sql: &str) {
    sqlx::query(sql).execute(pool).await.unwrap();
}
