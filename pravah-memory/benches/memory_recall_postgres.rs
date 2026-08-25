use std::{env, str::FromStr, time::Instant};

use chrono::Utc;
use pravah_memory::postgres::{
    MemoryProfile, MemoryRecallSchemaExt, MemorySchemaExt, RecallRecorder, RecallStore,
    RecallStoreError,
};
use pravah_memory::{MemoryId, RecallBatch, RecallCandidate, RecallError, RecallReceipt};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use thiserror::Error;
use uuid::Uuid;

const USER_KEY: &str = "recall-benchmark-user";
const AGENT_KEY: &str = "recall-benchmark-agent";
const RECORD_BATCH_SIZES: [usize; 4] = [1, 64, 512, 1_024];
const RECORD_SAMPLES: usize = 20;
const AGGREGATION_BATCH: usize = 5_000;
const AGGREGATION_SAMPLES: usize = 20;

/// Runs only when an explicit live PostgreSQL target is supplied.
#[tokio::main]
async fn main() {
    let Some(database_url) = benchmark_database_url() else {
        println!("memory/recall_postgres: skipped; set MEMORY_POSTGRES_DATABASE_URL to run");
        return;
    };
    if let Err(error) = run(&database_url).await {
        eprintln!("memory recall PostgreSQL benchmark failed: {error}");
        std::process::exit(1);
    }
}

/// Reads the opt-in database target without inventing a local default.
fn benchmark_database_url() -> Option<String> {
    match env::var("MEMORY_POSTGRES_DATABASE_URL") {
        Ok(value) if !value.trim().is_empty() => Some(value),
        Ok(_) | Err(env::VarError::NotPresent) => None,
        Err(error) => {
            eprintln!("cannot read MEMORY_POSTGRES_DATABASE_URL: {error}");
            None
        }
    }
}

/// Guarantees an attempted schema cleanup after every benchmark outcome.
async fn run(database_url: &str) -> Result<(), BenchmarkError> {
    let database = BenchmarkDatabase::create(database_url).await?;
    let result = run_in_schema(&database.pool).await;
    let cleanup = database.cleanup().await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => {
            eprintln!("benchmark schema cleanup also failed: {cleanup_error}");
            Err(error)
        }
    }
}

/// Runs recording, storage, and aggregation measurements inside one isolated schema.
async fn run_in_schema(pool: &sqlx::PgPool) -> Result<(), BenchmarkError> {
    apply_memory_schema(pool).await?;
    let memory_id = seed_memory(pool).await?;
    let store = RecallStore::builder(mool::DbPool::from_pool(pool.clone()))
        .retrieved_sampling(0.0)
        .max_record_batch(1_024)
        .build()
        .await?;
    let recorder = store.recorder(USER_KEY, AGENT_KEY)?;
    let reports = benchmark_recording(&recorder, memory_id).await?;
    for report in reports {
        report.print();
    }
    measure_event_storage(pool).await?.print();
    reset_recall_tables(pool).await?;
    benchmark_aggregation(pool, &store, &recorder, memory_id)
        .await?
        .print();
    Ok(())
}

/// Measures every required recorder batch size with independent durable events.
async fn benchmark_recording(
    recorder: &RecallRecorder,
    memory_id: MemoryId,
) -> Result<Vec<LatencyReport>, BenchmarkError> {
    let mut reports = Vec::with_capacity(RECORD_BATCH_SIZES.len());
    for batch_size in RECORD_BATCH_SIZES {
        let samples = record_samples(recorder, memory_id, batch_size).await?;
        reports.push(LatencyReport::new(
            format!("memory/recall_record_{batch_size}"),
            batch_size,
            samples,
        ));
    }
    Ok(reports)
}

/// Times only the transactional recorder call, excluding event construction.
async fn record_samples(
    recorder: &RecallRecorder,
    memory_id: MemoryId,
    batch_size: usize,
) -> Result<Vec<f64>, BenchmarkError> {
    let mut samples = Vec::with_capacity(RECORD_SAMPLES);
    for _ in 0..RECORD_SAMPLES {
        let batches = recall_batches(memory_id, batch_size)?;
        let started = Instant::now();
        let inserted = recorder.record_many(&batches).await?;
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        ensure_count("recorded events", batch_size, inserted)?;
    }
    Ok(samples)
}

/// Measures one exact 5,000-event aggregation against a clean table per sample.
async fn benchmark_aggregation(
    pool: &sqlx::PgPool,
    store: &RecallStore,
    recorder: &RecallRecorder,
    memory_id: MemoryId,
) -> Result<LatencyReport, BenchmarkError> {
    let mut samples = Vec::with_capacity(AGGREGATION_SAMPLES);
    for _ in 0..AGGREGATION_SAMPLES {
        seed_pending_events(recorder, memory_id).await?;
        let started = Instant::now();
        let aggregated = store.aggregate_pending(AGGREGATION_BATCH).await?;
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        ensure_count("aggregated events", AGGREGATION_BATCH, aggregated)?;
        reset_recall_tables(pool).await?;
    }
    Ok(LatencyReport::new(
        "memory/recall_aggregate_5000".to_owned(),
        AGGREGATION_BATCH,
        samples,
    ))
}

/// Populates exactly one aggregation batch outside the measured interval.
async fn seed_pending_events(
    recorder: &RecallRecorder,
    memory_id: MemoryId,
) -> Result<(), BenchmarkError> {
    let batches = recall_batches(memory_id, AGGREGATION_BATCH)?;
    for chunk in batches.chunks(1_024) {
        let inserted = recorder.record_many(chunk).await?;
        ensure_count("aggregation seed events", chunk.len(), inserted)?;
    }
    Ok(())
}

/// Creates unique one-event outcome batches for the same valid memory claim.
fn recall_batches(memory_id: MemoryId, count: usize) -> Result<Vec<RecallBatch>, BenchmarkError> {
    (0..count)
        .map(|_| {
            let receipt = RecallReceipt::new(
                USER_KEY,
                AGENT_KEY,
                vec![RecallCandidate { memory_id, rank: 1 }],
            )?;
            RecallBatch::used(&receipt, [memory_id]).map_err(Into::into)
        })
        .collect()
}

async fn reset_recall_tables(pool: &sqlx::PgPool) -> Result<(), BenchmarkError> {
    sqlx::query("TRUNCATE TABLE pravah_memory_recall_events, pravah_memory_recall_stats")
        .execute(pool)
        .await?;
    Ok(())
}

/// Rejects partial inserts or maintenance work instead of reporting misleading timings.
fn ensure_count(
    operation: &'static str,
    expected: usize,
    actual: u64,
) -> Result<(), BenchmarkError> {
    if u64::try_from(expected).ok() == Some(actual) {
        Ok(())
    } else {
        Err(BenchmarkError::UnexpectedCount {
            operation,
            expected,
            actual,
        })
    }
}

#[derive(Debug)]
struct LatencyReport {
    name: String,
    items: usize,
    samples_ms: Vec<f64>,
}

impl LatencyReport {
    fn new(name: String, items: usize, mut samples_ms: Vec<f64>) -> Self {
        samples_ms.sort_by(f64::total_cmp);
        Self {
            name,
            items,
            samples_ms,
        }
    }

    /// Prints operation percentiles and median event throughput without query payloads.
    fn print(&self) {
        let p50 = percentile(&self.samples_ms, 50);
        let p95 = percentile(&self.samples_ms, 95);
        let p99 = percentile(&self.samples_ms, 99);
        let events_per_second = self.items as f64 / (p50 / 1_000.0);
        println!(
            "{}: p50 {p50:.3} ms/op; p95 {p95:.3} ms/op; p99 {p99:.3} ms/op; \
             {:.0} events/s at p50; {} sample(s)",
            self.name,
            events_per_second,
            self.samples_ms.len(),
        );
    }
}

fn percentile(samples: &[f64], percentile: usize) -> f64 {
    let rank = samples
        .len()
        .saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1);
    samples[rank.min(samples.len().saturating_sub(1))]
}

#[derive(Debug, sqlx::FromRow)]
struct EventStorage {
    event_count: i64,
    table_bytes: i64,
    index_bytes: i64,
    total_bytes: i64,
}

impl EventStorage {
    /// Prints measured table and index bytes plus their observed per-event average.
    fn print(&self) {
        let bytes_per_event = if self.event_count == 0 {
            0.0
        } else {
            self.total_bytes as f64 / self.event_count as f64
        };
        println!(
            "memory/recall_event_storage: {} event(s); {} table byte(s); {} index byte(s); \
             {} total byte(s); {bytes_per_event:.1} total byte(s)/event",
            self.event_count, self.table_bytes, self.index_bytes, self.total_bytes,
        );
    }
}

/// Vacuums the retained fixture before reading PostgreSQL relation sizes.
async fn measure_event_storage(pool: &sqlx::PgPool) -> Result<EventStorage, BenchmarkError> {
    sqlx::query("VACUUM (ANALYZE) pravah_memory_recall_events")
        .execute(pool)
        .await?;
    let storage = sqlx::query_as::<_, EventStorage>(
        "SELECT COUNT(*)::bigint AS event_count, \
         pg_table_size('pravah_memory_recall_events'::regclass)::bigint AS table_bytes, \
         pg_indexes_size('pravah_memory_recall_events'::regclass)::bigint AS index_bytes, \
         pg_total_relation_size('pravah_memory_recall_events'::regclass)::bigint AS total_bytes \
         FROM pravah_memory_recall_events",
    )
    .fetch_one(pool)
    .await?;
    Ok(storage)
}

struct BenchmarkDatabase {
    admin: sqlx::PgPool,
    pool: sqlx::PgPool,
    schema: String,
}

impl BenchmarkDatabase {
    /// Creates an isolated schema and a UTC pool whose search path selects it.
    async fn create(database_url: &str) -> Result<Self, BenchmarkError> {
        let admin = PgPoolOptions::new()
            .max_connections(1)
            .connect(database_url)
            .await?;
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            .execute(&admin)
            .await?;
        let schema = format!("pravah_recall_bench_{}", Uuid::now_v7().simple());
        sqlx::query(&format!("CREATE SCHEMA {schema}"))
            .execute(&admin)
            .await?;
        match benchmark_pool(database_url, &schema).await {
            Ok(pool) => Ok(Self {
                admin,
                pool,
                schema,
            }),
            Err(error) => {
                let _ = sqlx::query(&format!("DROP SCHEMA {schema} CASCADE"))
                    .execute(&admin)
                    .await;
                admin.close().await;
                Err(error)
            }
        }
    }

    /// Closes benchmark connections before dropping only the generated schema.
    async fn cleanup(self) -> Result<(), BenchmarkError> {
        self.pool.close().await;
        let result = sqlx::query(&format!("DROP SCHEMA {} CASCADE", self.schema))
            .execute(&self.admin)
            .await;
        self.admin.close().await;
        result?;
        Ok(())
    }
}

/// Builds the scoped benchmark pool with Mool's required UTC session contract.
async fn benchmark_pool(database_url: &str, schema: &str) -> Result<sqlx::PgPool, BenchmarkError> {
    let options = PgConnectOptions::from_str(database_url)?
        .options([("search_path", format!("{schema},public"))]);
    let pool = PgPoolOptions::new()
        .min_connections(1)
        .max_connections(8)
        .after_connect(|connection, _| {
            Box::pin(async move {
                sqlx::query("SET TIME ZONE 'UTC'")
                    .execute(connection)
                    .await?;
                Ok(())
            })
        })
        .connect_with(options)
        .await?;
    Ok(pool)
}

async fn apply_memory_schema(pool: &sqlx::PgPool) -> Result<(), BenchmarkError> {
    for statement in migration_sql()? {
        sqlx::query(&statement).execute(pool).await?;
    }
    Ok(())
}

fn migration_sql() -> Result<Vec<String>, BenchmarkError> {
    let schema = mool::schema()
        .with_memory(memory_profile())
        .with_memory_recall()
        .build()?;
    let planner = mool::gaman::core::OfflinePlanner::new(mool::gaman::core::Dialect::Postgres);
    let migration = reviewed_migration(&planner, schema)?;
    planner.sql_migrate(&[migration]).map_err(Into::into)
}

/// Accepts only the reviewed opaque-index prompts used by the fixed memory schema.
fn reviewed_migration(
    planner: &mool::gaman::core::OfflinePlanner,
    schema: mool::gaman::schema::Schema,
) -> Result<mool::gaman::Migration, BenchmarkError> {
    let clarifications = match planner.make_migration(schema.clone(), &[]) {
        Err(mool::gaman::core::OfflineError::NeedsInput(items)) => items,
        Ok(Some(migration)) => return Ok(migration),
        Ok(None) => return Err(BenchmarkError::EmptyMigration),
        Err(error) => return Err(error.into()),
    };
    if clarifications.iter().any(|item| {
        !matches!(
            item.kind,
            mool::gaman::core::ClarificationKind::OpaqueEntity { .. }
        )
    }) {
        return Err(BenchmarkError::UnexpectedMigrationClarification);
    }
    let decisions = clarifications
        .into_iter()
        .map(|item| mool::gaman::core::Decision {
            clarification_id: item.id,
            answer: mool::gaman::core::Answer::AcceptRisk,
        })
        .collect::<Vec<_>>();
    planner
        .make_migration(schema, &decisions)?
        .ok_or(BenchmarkError::EmptyMigration)
}

fn memory_profile() -> MemoryProfile {
    MemoryProfile::new(
        "recall-benchmark-embedding",
        "v1",
        3,
        "claim-text-v1",
        "recall-benchmark-extractor-v1",
        "recall-benchmark-reconciler-v1",
    )
}

/// Inserts one valid evidence-backed memory directly, avoiding provider latency.
async fn seed_memory(pool: &sqlx::PgPool) -> Result<MemoryId, BenchmarkError> {
    let evidence_id = Uuid::now_v7();
    let memory_id = Uuid::now_v7();
    let now = Utc::now();
    seed_evidence(pool, evidence_id, now).await?;
    sqlx::query(
        "INSERT INTO pravah_memories (id, user_key, agent_key, evidence_id, position, text, \
         content_hash, kind, valid_from, valid_until, event_at, temporal_precision, \
         temporal_state, embedding, metadata, created_at, stale, current_for_retrieval) \
         VALUES ($1, $2, $3, $4, 0, $5, $6, 'fact', NULL, NULL, NULL, 'unknown', \
         'unspecified', CAST($7 AS vector(3)), $8, $9, FALSE, TRUE)",
    )
    .bind(memory_id)
    .bind(USER_KEY)
    .bind(AGENT_KEY)
    .bind(evidence_id)
    .bind("The benchmark user prefers deterministic memory telemetry.")
    .bind(vec![2_u8; 32])
    .bind("[1,0,0]")
    .bind(serde_json::json!({}))
    .bind(now)
    .execute(pool)
    .await?;
    Ok(MemoryId::from(memory_id))
}

/// Inserts the immutable evidence row required by the benchmark memory foreign key.
async fn seed_evidence(
    pool: &sqlx::PgPool,
    evidence_id: Uuid,
    now: chrono::DateTime<Utc>,
) -> Result<(), BenchmarkError> {
    sqlx::query(
        "INSERT INTO pravah_evidence (id, user_key, agent_key, evidence_key, content, \
         content_hash, metadata, observed_at, created_at, processed_at, processing_state, \
         processing_token, processing_lease_until, processing_attempts, published_revision, \
         reconciliation_state, extractor_revision, reconciler_revision, error_code, stale) \
         VALUES ($1, $2, $3, 'benchmark:evidence:v1', $4, $5, $6, $7, $7, $7, 'ready', \
         NULL, NULL, 1, 1, 'ready', 'recall-benchmark-extractor-v1', \
         'recall-benchmark-reconciler-v1', NULL, FALSE)",
    )
    .bind(evidence_id)
    .bind(USER_KEY)
    .bind(AGENT_KEY)
    .bind("The benchmark user prefers deterministic memory telemetry.")
    .bind(vec![1_u8; 32])
    .bind(serde_json::json!({}))
    .bind(now)
    .execute(pool)
    .await?;
    Ok(())
}

#[derive(Debug, Error)]
enum BenchmarkError {
    #[error(transparent)]
    Database(#[from] sqlx::Error),
    #[error(transparent)]
    Recall(#[from] RecallError),
    #[error(transparent)]
    RecallStore(#[from] RecallStoreError),
    #[error(transparent)]
    Schema(#[from] mool::gaman::schema::SchemaLoadError),
    #[error(transparent)]
    Migration(#[from] mool::gaman::core::OfflineError),
    #[error("memory migration unexpectedly contained no operations")]
    EmptyMigration,
    #[error("memory migration requested a non-opaque review decision")]
    UnexpectedMigrationClarification,
    #[error("{operation} returned {actual} rows; expected {expected}")]
    UnexpectedCount {
        operation: &'static str,
        expected: usize,
        actual: u64,
    },
}
