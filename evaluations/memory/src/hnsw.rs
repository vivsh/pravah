use std::{collections::BTreeSet, time::Instant};

use mool::types::Vector;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::{PgPool, Postgres, Transaction};
use uuid::Uuid;

use crate::EvaluationError;

const CURRENT_INDEX: &str = "pravah_memories_current_embedding_idx";

/// One scoped precomputed query embedding for exact-versus-ANN comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswQuery {
    /// Stable caller identity for the query.
    pub query_id: String,
    /// Memory user scope.
    pub user_key: String,
    /// Memory agent scope.
    pub agent_key: String,
    /// Query vector in the active profile's embedding space.
    pub embedding: Vec<f32>,
}

/// Bounded settings applied identically across a comparison run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswComparisonOptions {
    /// Top-K exact and approximate result count.
    pub k: u32,
    /// pgvector HNSW candidate-list size.
    pub ef_search: u32,
    /// HNSW tuples visited before strict iterative scan stops.
    pub max_scan_tuples: u32,
    /// Unmeasured executions per path and query.
    pub warmups: u32,
    /// Measured executions per path and query.
    pub repetitions: u32,
}

impl Default for HnswComparisonOptions {
    fn default() -> Self {
        Self {
            k: 10,
            ef_search: 40,
            max_scan_tuples: 20_000,
            warmups: 1,
            repetitions: 3,
        }
    }
}

impl HnswComparisonOptions {
    /// Validates bounds before any database work begins.
    pub fn validate(self) -> Result<Self, EvaluationError> {
        if self.k == 0 || self.k > 1_000 {
            return Err(configuration("k must be between 1 and 1000"));
        }
        if self.ef_search == 0 || self.max_scan_tuples == 0 {
            return Err(configuration("HNSW scan bounds must be greater than zero"));
        }
        if self.repetitions == 0 || self.repetitions > 100 {
            return Err(configuration("repetitions must be between 1 and 100"));
        }
        Ok(self)
    }
}

/// Microsecond latency quantiles over measured database statements only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyDistribution {
    /// Number of measured statements.
    pub samples: usize,
    /// Median latency.
    pub p50_us: u64,
    /// 95th-percentile latency.
    pub p95_us: u64,
    /// 99th-percentile latency.
    pub p99_us: u64,
}

/// Exact and HNSW result identities for one query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswCase {
    /// Caller query identity.
    pub query_id: String,
    /// Exact sequential-scan top-K oracle.
    pub exact_ids: Vec<Uuid>,
    /// Forced HNSW top-K result.
    pub hnsw_ids: Vec<Uuid>,
    /// Fraction of exact top-K identities returned by HNSW.
    pub recall_at_k: f64,
}

/// Reproducible scoped ANN comparison and plan evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswComparison {
    /// Effective comparison settings.
    pub options: HnswComparisonOptions,
    /// Active memory embedding dimensions.
    pub embedding_dimensions: usize,
    /// Macro-average recall against exact top-K.
    pub mean_recall_at_k: f64,
    /// Exact statement latency distribution.
    pub exact_latency: LatencyDistribution,
    /// Approximate statement latency distribution.
    pub hnsw_latency: LatencyDistribution,
    /// Per-query identities and recall.
    pub cases: Vec<HnswCase>,
    /// Representative forced exact plan in PostgreSQL JSON format.
    pub exact_plan: JsonValue,
    /// Representative forced HNSW plan in PostgreSQL JSON format.
    pub hnsw_plan: JsonValue,
}

/// PostgreSQL comparator that proves both execution paths before measuring recall.
#[derive(Clone)]
pub struct HnswComparator {
    pool: PgPool,
}

impl HnswComparator {
    /// Creates a comparator over an already migrated Pravah database.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Compares forced exact and HNSW paths for all supplied scoped vectors.
    pub async fn compare(
        &self,
        queries: &[HnswQuery],
        options: HnswComparisonOptions,
    ) -> Result<HnswComparison, EvaluationError> {
        let options = options.validate()?;
        let dimensions = self.embedding_dimensions().await?;
        validate_queries(queries, dimensions)?;
        let plans = self
            .validate_plans(&queries[0], dimensions, options)
            .await?;
        let mut exact_latencies = Vec::new();
        let mut hnsw_latencies = Vec::new();
        let mut cases = Vec::with_capacity(queries.len());
        for query in queries {
            let measured = self.measure_query(query, dimensions, options).await?;
            exact_latencies.extend(measured.exact_latencies);
            hnsw_latencies.extend(measured.hnsw_latencies);
            cases.push(measured.case);
        }
        Ok(finish_comparison(
            options,
            dimensions,
            plans,
            cases,
            exact_latencies,
            hnsw_latencies,
        ))
    }

    async fn embedding_dimensions(&self) -> Result<usize, EvaluationError> {
        let value: i32 = sqlx::query_scalar(
            "SELECT embedding_dimensions FROM pravah_memory_profile WHERE id = 1",
        )
        .fetch_one(&self.pool)
        .await?;
        usize::try_from(value).map_err(|_| configuration("embedding dimensions are invalid"))
    }

    async fn validate_plans(
        &self,
        query: &HnswQuery,
        dimensions: usize,
        options: HnswComparisonOptions,
    ) -> Result<(JsonValue, JsonValue), EvaluationError> {
        let exact = self
            .explain(query, dimensions, options, SearchPath::Exact)
            .await?;
        let hnsw = self
            .explain(query, dimensions, options, SearchPath::Hnsw)
            .await?;
        let exact_text = exact.to_string();
        if exact_text.contains(CURRENT_INDEX) {
            return Err(plan_error("exact", "HNSW index appeared in exact plan"));
        }
        if !exact_text.contains("Seq Scan") {
            return Err(plan_error("exact", "sequential scan was not selected"));
        }
        if !hnsw.to_string().contains(CURRENT_INDEX) {
            return Err(plan_error("hnsw", "current HNSW index was not selected"));
        }
        Ok((exact, hnsw))
    }

    async fn explain(
        &self,
        query: &HnswQuery,
        dimensions: usize,
        options: HnswComparisonOptions,
        path: SearchPath,
    ) -> Result<JsonValue, EvaluationError> {
        let mut transaction = self.pool.begin().await?;
        configure_path(&mut transaction, path, options).await?;
        let sql = format!("EXPLAIN (FORMAT JSON) {}", search_sql(dimensions));
        let vector = query_vector(query)?;
        let plan = sqlx::query_scalar::<_, JsonValue>(&sql)
            .bind(&query.user_key)
            .bind(&query.agent_key)
            .bind(&vector)
            .bind(i64::from(options.k))
            .fetch_one(&mut *transaction)
            .await?;
        transaction.rollback().await?;
        Ok(plan)
    }

    async fn measure_query(
        &self,
        query: &HnswQuery,
        dimensions: usize,
        options: HnswComparisonOptions,
    ) -> Result<MeasuredQuery, EvaluationError> {
        for _ in 0..options.warmups {
            self.execute(query, dimensions, options, SearchPath::Exact)
                .await?;
            self.execute(query, dimensions, options, SearchPath::Hnsw)
                .await?;
        }
        let exact = self
            .measure_path(query, dimensions, options, SearchPath::Exact)
            .await?;
        let hnsw = self
            .measure_path(query, dimensions, options, SearchPath::Hnsw)
            .await?;
        let recall = recall_at_k(&exact.ids, &hnsw.ids);
        Ok(MeasuredQuery {
            case: HnswCase {
                query_id: query.query_id.clone(),
                exact_ids: exact.ids,
                hnsw_ids: hnsw.ids,
                recall_at_k: recall,
            },
            exact_latencies: exact.latencies,
            hnsw_latencies: hnsw.latencies,
        })
    }

    async fn measure_path(
        &self,
        query: &HnswQuery,
        dimensions: usize,
        options: HnswComparisonOptions,
        path: SearchPath,
    ) -> Result<MeasuredPath, EvaluationError> {
        let mut ids = Vec::new();
        let mut latencies = Vec::with_capacity(options.repetitions as usize);
        for _ in 0..options.repetitions {
            let measured = self.execute(query, dimensions, options, path).await?;
            if !ids.is_empty() && ids != measured.ids {
                return Err(plan_error(
                    path.name(),
                    "result identities changed during repetitions",
                ));
            }
            ids = measured.ids;
            latencies.push(measured.latency_us);
        }
        Ok(MeasuredPath { ids, latencies })
    }

    async fn execute(
        &self,
        query: &HnswQuery,
        dimensions: usize,
        options: HnswComparisonOptions,
        path: SearchPath,
    ) -> Result<Execution, EvaluationError> {
        let mut transaction = self.pool.begin().await?;
        configure_path(&mut transaction, path, options).await?;
        let sql = search_sql(dimensions);
        let vector = query_vector(query)?;
        let started = Instant::now();
        let ids = bind_query(&sql, query, &vector, options.k)
            .fetch_all(&mut *transaction)
            .await?;
        let latency_us = elapsed_us(started);
        transaction.rollback().await?;
        Ok(Execution { ids, latency_us })
    }
}

#[derive(Clone, Copy)]
enum SearchPath {
    Exact,
    Hnsw,
}

impl SearchPath {
    const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Hnsw => "hnsw",
        }
    }
}

struct Execution {
    ids: Vec<Uuid>,
    latency_us: u64,
}

struct MeasuredPath {
    ids: Vec<Uuid>,
    latencies: Vec<u64>,
}

struct MeasuredQuery {
    case: HnswCase,
    exact_latencies: Vec<u64>,
    hnsw_latencies: Vec<u64>,
}

async fn configure_path(
    transaction: &mut Transaction<'_, Postgres>,
    path: SearchPath,
    options: HnswComparisonOptions,
) -> Result<(), sqlx::Error> {
    match path {
        SearchPath::Exact => {
            sqlx::query("SET LOCAL enable_indexscan = off")
                .execute(&mut **transaction)
                .await?;
            sqlx::query("SET LOCAL enable_indexonlyscan = off")
                .execute(&mut **transaction)
                .await?;
            sqlx::query("SET LOCAL enable_bitmapscan = off")
                .execute(&mut **transaction)
                .await?;
        }
        SearchPath::Hnsw => configure_hnsw(transaction, options).await?,
    }
    Ok(())
}

async fn configure_hnsw(
    transaction: &mut Transaction<'_, Postgres>,
    options: HnswComparisonOptions,
) -> Result<(), sqlx::Error> {
    sqlx::query("SET LOCAL enable_seqscan = off")
        .execute(&mut **transaction)
        .await?;
    set_local(transaction, "hnsw.ef_search", options.ef_search).await?;
    set_local(transaction, "hnsw.max_scan_tuples", options.max_scan_tuples).await?;
    sqlx::query("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        .execute(&mut **transaction)
        .await?;
    Ok(())
}

async fn set_local(
    transaction: &mut Transaction<'_, Postgres>,
    name: &str,
    value: u32,
) -> Result<(), sqlx::Error> {
    sqlx::query("SELECT set_config($1, $2, true)")
        .bind(name)
        .bind(value.to_string())
        .execute(&mut **transaction)
        .await?;
    Ok(())
}

fn search_sql(dimensions: usize) -> String {
    format!(
        "SELECT id FROM pravah_memories \
         WHERE user_key = $1 AND agent_key = $2 AND stale = FALSE \
         AND current_for_retrieval = TRUE \
         ORDER BY embedding::vector({dimensions}) <=> $3::vector({dimensions}), id LIMIT $4"
    )
}

fn bind_query<'q>(
    sql: &'q str,
    query: &'q HnswQuery,
    vector: &'q Vector,
    k: u32,
) -> sqlx::query::QueryScalar<'q, Postgres, Uuid, sqlx::postgres::PgArguments> {
    sqlx::query_scalar(sql)
        .bind(&query.user_key)
        .bind(&query.agent_key)
        .bind(vector)
        .bind(i64::from(k))
}

fn query_vector(query: &HnswQuery) -> Result<Vector, EvaluationError> {
    Vector::try_from_vec(query.embedding.clone())
        .map_err(|error| configuration(format!("invalid query vector: {error}")))
}

fn validate_queries(queries: &[HnswQuery], dimensions: usize) -> Result<(), EvaluationError> {
    if queries.is_empty() {
        return Err(configuration("at least one HNSW query is required"));
    }
    let mut ids = BTreeSet::new();
    for query in queries {
        if query.query_id.trim().is_empty() || !ids.insert(&query.query_id) {
            return Err(configuration("query IDs must be non-empty and unique"));
        }
        if query.user_key.trim().is_empty() || query.agent_key.trim().is_empty() {
            return Err(configuration("query scopes must not be empty"));
        }
        if query.embedding.len() != dimensions || query.embedding.iter().any(|v| !v.is_finite()) {
            return Err(configuration(
                "query embeddings must match the finite active profile",
            ));
        }
    }
    Ok(())
}

fn finish_comparison(
    options: HnswComparisonOptions,
    dimensions: usize,
    plans: (JsonValue, JsonValue),
    cases: Vec<HnswCase>,
    exact_latencies: Vec<u64>,
    hnsw_latencies: Vec<u64>,
) -> HnswComparison {
    let mean_recall_at_k =
        cases.iter().map(|case| case.recall_at_k).sum::<f64>() / cases.len() as f64;
    HnswComparison {
        options,
        embedding_dimensions: dimensions,
        mean_recall_at_k,
        exact_latency: latency_distribution(exact_latencies),
        hnsw_latency: latency_distribution(hnsw_latencies),
        cases,
        exact_plan: plans.0,
        hnsw_plan: plans.1,
    }
}

fn recall_at_k(exact: &[Uuid], approximate: &[Uuid]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let approximate = approximate.iter().collect::<BTreeSet<_>>();
    exact.iter().filter(|id| approximate.contains(id)).count() as f64 / exact.len() as f64
}

fn latency_distribution(mut samples: Vec<u64>) -> LatencyDistribution {
    samples.sort_unstable();
    LatencyDistribution {
        samples: samples.len(),
        p50_us: percentile(&samples, 50),
        p95_us: percentile(&samples, 95),
        p99_us: percentile(&samples, 99),
    }
}

fn percentile(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let rank = (percentile * samples.len()).div_ceil(100);
    samples[rank.saturating_sub(1)]
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn configuration(message: impl Into<String>) -> EvaluationError {
    EvaluationError::InvalidConfiguration(message.into())
}

fn plan_error(mode: &'static str, message: impl Into<String>) -> EvaluationError {
    EvaluationError::UnexpectedPlan {
        mode,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies recall is measured against the exact result count, including short scopes.
    #[test]
    fn computes_recall_against_exact_ids() {
        let first = Uuid::now_v7();
        let second = Uuid::now_v7();
        assert_eq!(recall_at_k(&[first, second], &[second]), 0.5);
    }

    /// Verifies latency percentiles use deterministic nearest-rank selection.
    #[test]
    fn computes_nearest_rank_percentiles() {
        let distribution = latency_distribution(vec![1, 2, 3, 4, 100]);
        assert_eq!(distribution.p50_us, 3);
        assert_eq!(distribution.p95_us, 100);
    }

    /// Verifies unsafe or meaningless comparison bounds fail before SQL execution.
    #[test]
    fn rejects_invalid_options() {
        let options = HnswComparisonOptions {
            repetitions: 0,
            ..HnswComparisonOptions::default()
        };
        assert!(options.validate().is_err());
    }
}
