#![cfg(feature = "recall-postgres")]

use std::str::FromStr;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use pravah_memory::postgres::{
    MemoryProfile, MemoryRecallSchemaExt, MemorySchemaExt, RecallStore, RecallStoreError,
};
use pravah_memory::{
    Embedding, EmbeddingProfile, EmbeddingProvider, ExtractedMemory, ExtractionRequest,
    MemoryExtractor, MemoryKind, MemoryManager, MemoryReconciler, ProviderError, RecallBatch,
    RecallId, ReconciliationDecision, ReconciliationGroup, TemporalMetadata,
};
use uuid::Uuid;

const USER: &str = "recall-user";
const AGENT: &str = "recall-agent";

struct StaticExtractor;

#[async_trait]
impl MemoryExtractor for StaticExtractor {
    fn revision(&self) -> &str {
        "recall-live-extractor-v1"
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

struct Embedder;

#[async_trait]
impl EmbeddingProvider for Embedder {
    fn profile(&self) -> EmbeddingProfile {
        EmbeddingProfile {
            model: "recall-live-embedding".to_owned(),
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
        "recall-live-reconciler-v1"
    }

    async fn reconcile(
        &self,
        _group: ReconciliationGroup,
    ) -> Result<Vec<ReconciliationDecision>, ProviderError> {
        Ok(Vec::new())
    }
}

#[derive(Debug, sqlx::FromRow)]
struct StatsSnapshot {
    sampled_retrieved_count: i64,
    estimated_retrieved_count: f64,
    accepted_count: i64,
    used_count: i64,
    dismissed_count: i64,
    corrected_count: i64,
    last_used_at: Option<DateTime<Utc>>,
    decayed_use_mass: f64,
}

struct TestDatabase {
    pool: sqlx::PgPool,
    schema: String,
}

struct MigrationStages {
    base: Vec<String>,
    recall: Vec<String>,
}

impl TestDatabase {
    /// Removes the isolated schema after all handles using it have been dropped.
    async fn cleanup(self) {
        execute_sql(&self.pool, &format!("DROP SCHEMA {} CASCADE", self.schema)).await;
        self.pool.close().await;
    }
}

/// Verifies tracked retrieval parity, retry-safe event writes, and exact aggregation counts.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn tracked_search_events_are_idempotent_and_aggregate_exactly() {
    let database = test_database().await;
    let manager = test_manager(&database.pool).await;
    manager
        .ingestor(USER, AGENT)
        .unwrap()
        .submit("preference:v1", "The user prefers aisle seats.")
        .await
        .unwrap();
    let retriever = manager.retriever(USER, AGENT).unwrap();
    let raw = retriever.search("aisle seats").await.unwrap();
    let tracked = retriever.search_tracked("aisle seats").await.unwrap();
    assert_eq!(tracked.results, raw);
    let correction = manager
        .ingestor(USER, AGENT)
        .unwrap()
        .submit("preference:v2", "The user now prefers window seats.")
        .await
        .unwrap();
    assert_exact_telemetry(&database.pool, &tracked, correction.evidence_id).await;
    drop(manager);
    database.cleanup().await;
}

/// Verifies bound recorders reject foreign receipts before performing database work.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn recorder_rejects_cross_scope_batches() {
    let database = test_database().await;
    let manager = test_manager(&database.pool).await;
    ingest_claim(&manager, USER, AGENT, "scope:v1", "The user likes tea.").await;
    let tracked = manager
        .retriever(USER, AGENT)
        .unwrap()
        .search_tracked("tea")
        .await
        .unwrap();
    let batch = RecallBatch::accepted(&tracked.receipt, [tracked.results[0].memory.id]).unwrap();
    let store = test_recall_store(&database.pool).await;
    let foreign = store.recorder("other-user", AGENT).unwrap();
    assert!(matches!(
        foreign.record_many(&[batch]).await,
        Err(RecallStoreError::ScopeMismatch)
    ));
    assert_eq!(event_count(&database.pool).await, 0);
    drop((foreign, store, manager));
    database.cleanup().await;
}

/// Verifies correction evidence must share scope and its deletion cascades correction events.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn correction_scope_and_deletion_are_database_enforced() {
    let database = test_database().await;
    let manager = test_manager(&database.pool).await;
    ingest_claim(&manager, USER, AGENT, "source:v1", "The user likes tea.").await;
    let tracked = manager
        .retriever(USER, AGENT)
        .unwrap()
        .search_tracked("tea")
        .await
        .unwrap();
    let foreign = manager
        .ingestor("other-user", AGENT)
        .unwrap()
        .submit("correction:foreign", "The user dislikes tea.")
        .await
        .unwrap();
    assert_invalid_correction_is_atomic(&database.pool, &tracked, foreign.evidence_id).await;
    let local = manager
        .ingestor(USER, AGENT)
        .unwrap()
        .submit("correction:local", "The user now dislikes tea.")
        .await
        .unwrap();
    assert_correction_cascade(&database.pool, &manager, &tracked, local.evidence_id).await;
    drop(manager);
    database.cleanup().await;
}

/// Verifies concurrent aggregators, exact rebuilds, and bounded retention repair derived stats.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn aggregation_rebuild_and_retention_are_retry_safe() {
    let database = test_database().await;
    let manager = test_manager(&database.pool).await;
    ingest_claim(&manager, USER, AGENT, "retention:v1", "The user likes tea.").await;
    let retriever = manager.retriever(USER, AGENT).unwrap();
    let first = retriever.search_tracked("tea").await.unwrap();
    let second = retriever.search_tracked("tea").await.unwrap();
    let memory_id = first.results[0].memory.id;
    let batches = [
        RecallBatch::used(&first.receipt, [memory_id]).unwrap(),
        RecallBatch::used(&second.receipt, [memory_id]).unwrap(),
    ];
    let store = test_recall_store(&database.pool).await;
    assert_eq!(
        store
            .recorder(USER, AGENT)
            .unwrap()
            .record_many(&batches)
            .await
            .unwrap(),
        2
    );
    assert_eq!(run_concurrent_aggregation(&store).await, 2);
    assert_rebuild_and_retention(&database.pool, &store, memory_id.as_uuid()).await;
    drop((store, manager));
    database.cleanup().await;
}

/// Verifies source deletion cascades both retained events and rebuildable statistics.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn source_deletion_cascades_recall_state() {
    let database = test_database().await;
    let manager = test_manager(&database.pool).await;
    ingest_claim(&manager, USER, AGENT, "delete:v1", "The user likes tea.").await;
    let tracked = manager
        .retriever(USER, AGENT)
        .unwrap()
        .search_tracked("tea")
        .await
        .unwrap();
    let memory_id = tracked.results[0].memory.id;
    let batch = RecallBatch::accepted(&tracked.receipt, [memory_id]).unwrap();
    let store = test_recall_store(&database.pool).await;
    store
        .recorder(USER, AGENT)
        .unwrap()
        .record_many(&[batch])
        .await
        .unwrap();
    store.aggregate_pending(10).await.unwrap();
    assert!(
        read_stats(&database.pool, memory_id.as_uuid())
            .await
            .is_some()
    );
    manager
        .ingestor(USER, AGENT)
        .unwrap()
        .delete("delete:v1")
        .await
        .unwrap();
    assert_eq!(event_count(&database.pool).await, 0);
    assert!(
        read_stats(&database.pool, memory_id.as_uuid())
            .await
            .is_none()
    );
    drop((store, manager));
    database.cleanup().await;
}

/// Verifies the optional upgrade preserves v2 data and telemetry failures stay isolated.
#[tokio::test]
#[ignore = "requires MEMORY_POSTGRES_DATABASE_URL with pgvector installed"]
async fn schema_v2_to_recall_replay_preserves_memory_and_search() {
    let database = empty_test_database().await;
    let stages = migration_stages();
    apply_statements(&database.pool, &stages.base).await;
    let manager = test_manager(&database.pool).await;
    ingest_claim(
        &manager,
        USER,
        AGENT,
        "upgrade:v1",
        "The user likes quiet rooms.",
    )
    .await;
    let tracked = assert_search_parity(&manager, "quiet rooms").await;
    let memory_id = tracked.results[0].memory.id;
    let batch = RecallBatch::accepted(&tracked.receipt, [memory_id]).unwrap();
    let store = test_recall_store(&database.pool).await;
    let recorder = store.recorder(USER, AGENT).unwrap();
    assert!(
        recorder
            .record_many(std::slice::from_ref(&batch))
            .await
            .is_err()
    );
    assert_search_parity(&manager, "quiet rooms").await;
    apply_statements(&database.pool, &stages.recall).await;
    assert_eq!(recorder.record_many(&[batch]).await.unwrap(), 1);
    assert_eq!(store.aggregate_pending(10).await.unwrap(), 1);
    assert!(
        read_stats(&database.pool, memory_id.as_uuid())
            .await
            .is_some()
    );
    assert_search_parity(&manager, "quiet rooms").await;
    assert!(
        manager
            .ingestor(USER, AGENT)
            .unwrap()
            .get("upgrade:v1")
            .await
            .unwrap()
            .is_some()
    );
    drop((recorder, store, manager));
    database.cleanup().await;
}

/// Records every supported positive observation twice and checks conflict-ignore idempotency.
async fn assert_exact_telemetry(
    pool: &sqlx::PgPool,
    tracked: &pravah_memory::TrackedSearch,
    correction_evidence_id: pravah_memory::EvidenceId,
) {
    let memory_id = tracked.results[0].memory.id;
    let store = test_recall_store(pool).await;
    let recorder = store.recorder(USER, AGENT).unwrap();
    let batches = vec![
        recorder.retrieved(&tracked.receipt).unwrap(),
        RecallBatch::accepted(&tracked.receipt, [memory_id]).unwrap(),
        RecallBatch::used(&tracked.receipt, [memory_id]).unwrap(),
        RecallBatch::dismissed(&tracked.receipt, [memory_id]).unwrap(),
        RecallBatch::corrected(&tracked.receipt, [(memory_id, correction_evidence_id)]).unwrap(),
    ];
    assert_eq!(recorder.record_many(&batches).await.unwrap(), 5);
    assert_eq!(recorder.record_many(&batches).await.unwrap(), 0);
    assert_eq!(store.aggregate_pending(10).await.unwrap(), 5);
    let stats = read_stats(pool, memory_id.as_uuid()).await.unwrap();
    assert_eq!(stats.sampled_retrieved_count, 1);
    assert_eq!(stats.estimated_retrieved_count, 1.0);
    assert_eq!((stats.accepted_count, stats.used_count), (1, 1));
    assert_eq!((stats.dismissed_count, stats.corrected_count), (1, 1));
    assert!(stats.last_used_at.is_some());
    assert!((0.99..=1.0).contains(&stats.decayed_use_mass));
}

/// Proves an invalid correction rolls back a valid sibling event in the same bulk statement.
async fn assert_invalid_correction_is_atomic(
    pool: &sqlx::PgPool,
    tracked: &pravah_memory::TrackedSearch,
    foreign_evidence_id: pravah_memory::EvidenceId,
) {
    let memory_id = tracked.results[0].memory.id;
    let accepted = RecallBatch::accepted(&tracked.receipt, [memory_id]).unwrap();
    let corrected =
        RecallBatch::corrected(&tracked.receipt, [(memory_id, foreign_evidence_id)]).unwrap();
    let store = test_recall_store(pool).await;
    let recorder = store.recorder(USER, AGENT).unwrap();
    assert!(recorder.record_many(&[accepted, corrected]).await.is_err());
    assert_eq!(events_for_recall(pool, tracked.receipt.id).await, 0);
}

/// Records a valid correction, then proves correction-evidence deletion cascades it.
async fn assert_correction_cascade(
    pool: &sqlx::PgPool,
    manager: &MemoryManager,
    tracked: &pravah_memory::TrackedSearch,
    evidence_id: pravah_memory::EvidenceId,
) {
    let memory_id = tracked.results[0].memory.id;
    let corrected = RecallBatch::corrected(&tracked.receipt, [(memory_id, evidence_id)]).unwrap();
    let store = test_recall_store(pool).await;
    let recorder = store.recorder(USER, AGENT).unwrap();
    assert_eq!(recorder.record_many(&[corrected]).await.unwrap(), 1);
    assert_eq!(store.aggregate_pending(10).await.unwrap(), 1);
    assert_eq!(events_for_recall(pool, tracked.receipt.id).await, 1);
    manager
        .ingestor(USER, AGENT)
        .unwrap()
        .delete("correction:local")
        .await
        .unwrap();
    assert_eq!(events_for_recall(pool, tracked.receipt.id).await, 0);
    assert_eq!(store.rebuild_scope_stats(USER, AGENT).await.unwrap(), 0);
    assert!(read_stats(pool, memory_id.as_uuid()).await.is_none());
}

/// Runs two maintenance workers against one pending set and totals their disjoint claims.
async fn run_concurrent_aggregation(store: &RecallStore) -> u64 {
    let left_store = store.clone();
    let right_store = store.clone();
    let left = tokio::spawn(async move { left_store.aggregate_pending(1).await.unwrap() });
    let right = tokio::spawn(async move { right_store.aggregate_pending(1).await.unwrap() });
    left.await.unwrap() + right.await.unwrap()
}

/// Checks rebuild equality, then confirms retention removes both events and derived stats.
async fn assert_rebuild_and_retention(pool: &sqlx::PgPool, store: &RecallStore, memory_id: Uuid) {
    let before = read_stats(pool, memory_id).await.unwrap();
    assert_eq!(before.used_count, 2);
    assert_eq!(store.rebuild_scope_stats(USER, AGENT).await.unwrap(), 1);
    let rebuilt = read_stats(pool, memory_id).await.unwrap();
    assert_eq!(rebuilt.used_count, before.used_count);
    assert_eq!(rebuilt.last_used_at, before.last_used_at);
    assert!(rebuilt.decayed_use_mass > 1.9);
    let cutoff = Utc::now() + Duration::seconds(1);
    assert_eq!(store.prune_before(cutoff, 10).await.unwrap(), 2);
    assert_eq!(event_count(pool).await, 0);
    assert!(read_stats(pool, memory_id).await.is_none());
}

/// Confirms tracked search is an additive receipt over unchanged ordered results.
async fn assert_search_parity(
    manager: &MemoryManager,
    query: &str,
) -> pravah_memory::TrackedSearch {
    let retriever = manager.retriever(USER, AGENT).unwrap();
    let raw = retriever.search(query).await.unwrap();
    let tracked = retriever.search_tracked(query).await.unwrap();
    assert_eq!(tracked.results, raw);
    tracked
}

/// Ingests one independently searchable claim into the requested scope.
async fn ingest_claim(
    manager: &MemoryManager,
    user_key: &str,
    agent_key: &str,
    evidence_key: &str,
    text: &str,
) {
    manager
        .ingestor(user_key, agent_key)
        .unwrap()
        .submit(evidence_key, text)
        .await
        .unwrap();
}

/// Builds deterministic providers against the migrated live schema.
async fn test_manager(pool: &sqlx::PgPool) -> MemoryManager {
    MemoryManager::builder(mool::DbPool::from_pool(pool.clone()))
        .memory_extractor(StaticExtractor)
        .embedding_provider(Embedder)
        .reconciler(Reconciler)
        .build()
        .await
        .unwrap()
}

/// Builds recall maintenance with full retrieval sampling for exact test counts.
async fn test_recall_store(pool: &sqlx::PgPool) -> RecallStore {
    RecallStore::builder(mool::DbPool::from_pool(pool.clone()))
        .retrieved_sampling(1.0)
        .build()
        .await
        .unwrap()
}

/// Creates an isolated schema and applies the reviewed combined memory migration.
async fn test_database() -> TestDatabase {
    let database = empty_test_database().await;
    apply_migration(&database.pool).await;
    database
}

/// Creates an isolated schema without applying either memory migration stage.
async fn empty_test_database() -> TestDatabase {
    let database_url = std::env::var("MEMORY_POSTGRES_DATABASE_URL")
        .expect("MEMORY_POSTGRES_DATABASE_URL must select an isolated test database");
    let admin = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(&database_url)
        .await
        .unwrap();
    let schema = format!("pravah_recall_test_{}", Uuid::now_v7().simple());
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
    TestDatabase { pool, schema }
}

/// Executes every statement in the generated, explicitly reviewed migration.
async fn apply_migration(pool: &sqlx::PgPool) {
    apply_statements(pool, &migration_sql()).await;
}

/// Applies an already reviewed migration stage in declaration order.
async fn apply_statements(pool: &sqlx::PgPool, statements: &[String]) {
    for statement in statements {
        execute_sql(pool, statement).await;
    }
}

/// Lowers the fixed memory plus optional recall schema into reviewed PostgreSQL SQL.
fn migration_sql() -> Vec<String> {
    let schema = mool::schema()
        .with_memory(memory_profile())
        .with_memory_recall()
        .build()
        .unwrap();
    let planner = mool::gaman::core::OfflinePlanner::new(mool::gaman::core::Dialect::Postgres);
    reviewed_migration_sql(&planner, schema)
}

/// Produces the exact reviewed base migration and its independently planned recall delta.
fn migration_stages() -> MigrationStages {
    let empty = mool::gaman::core::OfflinePlanner::new(mool::gaman::core::Dialect::Postgres);
    let base_schema = mool::schema()
        .with_memory(memory_profile())
        .build()
        .unwrap();
    let base = reviewed_migration(&empty, base_schema);
    let base_sql = empty.sql_migrate(std::slice::from_ref(&base)).unwrap();
    let planner = mool::gaman::core::OfflinePlanner::new(mool::gaman::core::Dialect::Postgres)
        .from_migrations(vec![base]);
    let desired = mool::schema()
        .with_memory(memory_profile())
        .with_memory_recall()
        .build()
        .unwrap();
    let recall = reviewed_migration(&planner, desired);
    let recall_sql = planner.sql_migrate(&[recall]).unwrap();
    MigrationStages {
        base: base_sql,
        recall: recall_sql,
    }
}

fn memory_profile() -> MemoryProfile {
    MemoryProfile::new(
        "recall-live-embedding",
        "v1",
        3,
        "claim-text-v1",
        "recall-live-extractor-v1",
        "recall-live-reconciler-v1",
    )
}

/// Accepts every opaque index declaration for this isolated test migration only.
fn reviewed_migration_sql(
    planner: &mool::gaman::core::OfflinePlanner,
    schema: mool::gaman::schema::Schema,
) -> Vec<String> {
    let migration = reviewed_migration(planner, schema);
    planner.sql_migrate(&[migration]).unwrap()
}

/// Resolves only explicit opaque-DDL review prompts and returns the planned migration.
fn reviewed_migration(
    planner: &mool::gaman::core::OfflinePlanner,
    schema: mool::gaman::schema::Schema,
) -> mool::gaman::Migration {
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
    planner.make_migration(schema, &decisions).unwrap().unwrap()
}

/// Loads the complete derived statistics row used by behavioral assertions.
async fn read_stats(pool: &sqlx::PgPool, memory_id: Uuid) -> Option<StatsSnapshot> {
    sqlx::query_as::<_, StatsSnapshot>(
        "SELECT sampled_retrieved_count, estimated_retrieved_count, accepted_count, \
         used_count, dismissed_count, corrected_count, last_used_at, decayed_use_mass \
         FROM pravah_memory_recall_stats WHERE memory_id = $1",
    )
    .bind(memory_id)
    .fetch_optional(pool)
    .await
    .unwrap()
}

async fn event_count(pool: &sqlx::PgPool) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM pravah_memory_recall_events")
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn events_for_recall(pool: &sqlx::PgPool, recall_id: RecallId) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM pravah_memory_recall_events WHERE recall_id = $1")
        .bind(recall_id.as_uuid())
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn execute_sql(pool: &sqlx::PgPool, sql: &str) {
    sqlx::query(sql).execute(pool).await.unwrap();
}
