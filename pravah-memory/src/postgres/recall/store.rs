use std::{collections::HashSet, sync::Arc, time::Instant};

use chrono::{DateTime, Duration, Utc};
use mool as db;
use mool::backend::IgnoreConflictsExt;
use mool::prelude::*;
use thiserror::Error;
use uuid::Uuid;

use crate::{EvidenceId, RecallBatch, RecallError, RecallReceipt};

use super::models::{RecallEventRow, RecallStatsRow};
use super::sql;

/// Optional PostgreSQL recall-event store and explicit maintenance surface.
#[derive(Clone)]
pub struct RecallStore {
    inner: Arc<RecallStoreInner>,
}

struct RecallStoreInner {
    pool: db::DbPool,
    config: RecallStoreConfig,
}

#[derive(Debug, Clone)]
struct RecallStoreConfig {
    retention: Duration,
    use_decay_half_life: Duration,
    retrieved_sampling: f64,
    max_record_batch: usize,
}

impl RecallStore {
    /// Starts a builder whose configuration errors are deferred until `build`.
    pub fn builder(pool: db::DbPool) -> RecallStoreBuilder {
        RecallStoreBuilder::new(pool)
    }

    /// Creates a recorder permanently bound to one user and agent scope.
    pub fn recorder(
        &self,
        user_key: impl Into<String>,
        agent_key: impl Into<String>,
    ) -> Result<RecallRecorder, RecallStoreError> {
        let user_key = validate_scope_key("user_key", user_key.into())?;
        let agent_key = validate_scope_key("agent_key", agent_key.into())?;
        Ok(RecallRecorder {
            store: self.clone(),
            user_key,
            agent_key,
        })
    }

    /// Aggregates one lock-skipping batch of pending events and marks it atomically.
    pub async fn aggregate_pending(&self, batch_limit: usize) -> Result<u64, RecallStoreError> {
        let started = Instant::now();
        let result = self.aggregate_pending_inner(batch_limit).await;
        trace_operation("aggregate_pending", started, batch_limit, &result);
        result
    }

    /// Preserves the pending-event claim and stats update in one transaction.
    async fn aggregate_pending_inner(&self, batch_limit: usize) -> Result<u64, RecallStoreError> {
        let batch_limit = validate_maintenance_limit(batch_limit)?;
        let mut transaction = self.inner.pool.begin().await?;
        let claimed = claim_pending(&mut transaction, batch_limit).await?;
        if claimed.is_empty() {
            transaction.commit().await?;
            return Ok(0);
        }
        aggregate_claimed(&mut transaction, &self.inner.config, &claimed).await?;
        transaction.commit().await?;
        Ok(claimed.len() as u64)
    }

    /// Deletes one bounded event batch older than `cutoff` and repairs affected stats.
    pub async fn prune_before(
        &self,
        cutoff: DateTime<Utc>,
        batch_limit: usize,
    ) -> Result<u64, RecallStoreError> {
        let started = Instant::now();
        let result = self.prune_before_inner(cutoff, batch_limit).await;
        trace_operation("prune_before", started, batch_limit, &result);
        result
    }

    /// Preserves bounded deletion and affected-stat repair in one transaction.
    async fn prune_before_inner(
        &self,
        cutoff: DateTime<Utc>,
        batch_limit: usize,
    ) -> Result<u64, RecallStoreError> {
        let batch_limit = validate_maintenance_limit(batch_limit)?;
        let mut transaction = self.inner.pool.begin().await?;
        let claimed = claim_prunable(&mut transaction, cutoff, batch_limit).await?;
        if claimed.is_empty() {
            transaction.commit().await?;
            return Ok(0);
        }
        let affected = unique_memory_ids(claimed.iter().map(|event| event.memory_id));
        lock_memory_stats(&mut transaction, &affected).await?;
        let deleted = delete_prunable(&mut transaction, &claimed).await?;
        recompute_affected(&mut transaction, &self.inner.config, &affected).await?;
        transaction.commit().await?;
        Ok(deleted)
    }

    /// Applies the configured retention window to one bounded deletion batch.
    pub async fn prune_expired(&self, batch_limit: usize) -> Result<u64, RecallStoreError> {
        let cutoff = Utc::now()
            .checked_sub_signed(self.inner.config.retention)
            .ok_or(RecallStoreError::RetentionCutoffOverflow)?;
        self.prune_before(cutoff, batch_limit).await
    }

    /// Rebuilds one scope's statistics exactly from its currently retained events.
    pub async fn rebuild_scope_stats(
        &self,
        user_key: impl Into<String>,
        agent_key: impl Into<String>,
    ) -> Result<u64, RecallStoreError> {
        let started = Instant::now();
        let result = self
            .rebuild_scope_stats_inner(user_key.into(), agent_key.into())
            .await;
        trace_operation("rebuild_scope_stats", started, 1, &result);
        result
    }

    /// Rebuilds one scope under a table lock so concurrent inserts cannot be skipped.
    async fn rebuild_scope_stats_inner(
        &self,
        user_key: String,
        agent_key: String,
    ) -> Result<u64, RecallStoreError> {
        let user_key = validate_scope_key("user_key", user_key)?;
        let agent_key = validate_scope_key("agent_key", agent_key)?;
        let mut transaction = self.inner.pool.begin().await?;
        lock_events_for_rebuild(&mut transaction).await?;
        delete_scope_stats(&mut transaction, &user_key, &agent_key).await?;
        let rows =
            aggregate_scope(&mut transaction, &self.inner.config, &user_key, &agent_key).await?;
        upsert_stats(&mut transaction, &rows).await?;
        mark_scope_aggregated(&mut transaction, &user_key, &agent_key).await?;
        transaction.commit().await?;
        Ok(rows.len() as u64)
    }
}

/// Builder for an optional recall store.
pub struct RecallStoreBuilder {
    pool: db::DbPool,
    config: RecallStoreConfig,
    errors: Vec<String>,
}

impl RecallStoreBuilder {
    fn new(pool: db::DbPool) -> Self {
        Self {
            pool,
            config: RecallStoreConfig {
                retention: Duration::days(90),
                use_decay_half_life: Duration::days(30),
                retrieved_sampling: 0.0,
                max_record_batch: 1_024,
            },
            errors: Vec::new(),
        }
    }

    /// Sets the retained event window used by [`RecallStore::prune_expired`].
    pub fn retention(mut self, retention: Duration) -> Self {
        if retention <= Duration::zero() {
            self.errors.push("retention must be positive".to_owned());
        } else {
            self.config.retention = retention;
        }
        self
    }

    /// Sets the half-life used only for analytical `used` event decay.
    pub fn use_decay_half_life(mut self, half_life: Duration) -> Self {
        if half_life.num_milliseconds() <= 0 {
            self.errors
                .push("use decay half-life must be positive".to_owned());
        } else {
            self.config.use_decay_half_life = half_life;
        }
        self
    }

    /// Sets deterministic durable sampling for recorder-created retrieval batches.
    pub fn retrieved_sampling(mut self, probability: f64) -> Self {
        if probability.is_finite() && (0.0..=1.0).contains(&probability) {
            self.config.retrieved_sampling = probability;
        } else {
            self.errors
                .push("retrieved sampling must be finite and between zero and one".to_owned());
        }
        self
    }

    /// Sets the largest event count accepted by one atomic recorder call.
    pub fn max_record_batch(mut self, max_record_batch: usize) -> Self {
        if max_record_batch == 0 {
            self.errors
                .push("maximum record batch must be greater than zero".to_owned());
        } else {
            self.config.max_record_batch = max_record_batch;
        }
        self
    }

    /// Validates every accumulated option without creating or altering database objects.
    pub async fn build(self) -> Result<RecallStore, RecallStoreError> {
        if !self.errors.is_empty() {
            return Err(RecallStoreError::InvalidConfiguration(self.errors));
        }
        Ok(RecallStore {
            inner: Arc::new(RecallStoreInner {
                pool: self.pool,
                config: self.config,
            }),
        })
    }
}

/// Scope-bound, retry-safe recall event recorder.
#[derive(Clone)]
pub struct RecallRecorder {
    store: RecallStore,
    user_key: String,
    agent_key: String,
}

impl RecallRecorder {
    /// Builds a retrieved batch using this store's configured sampling probability.
    pub fn retrieved(&self, receipt: &RecallReceipt) -> Result<RecallBatch, RecallStoreError> {
        self.validate_receipt_scope(receipt)?;
        RecallBatch::retrieved(receipt, self.store.inner.config.retrieved_sampling)
            .map_err(Into::into)
    }

    /// Atomically inserts one bounded set of validated batches with duplicate ignore.
    pub async fn record_many(&self, batches: &[RecallBatch]) -> Result<u64, RecallStoreError> {
        let started = Instant::now();
        let result = self.record_many_inner(batches).await;
        trace_operation("record_many", started, batches.len(), &result);
        result
    }

    /// Validates and inserts one retry-safe event batch in a single transaction.
    async fn record_many_inner(&self, batches: &[RecallBatch]) -> Result<u64, RecallStoreError> {
        let rows = self.prepare_rows(batches)?;
        if rows.is_empty() {
            return Ok(0);
        }
        let table = RecallEventRow::table();
        let mut transaction = self.store.inner.pool.begin().await?;
        let affected = db::from(&table)
            .insert_many(&rows)
            .single_statement()
            .ignore_conflicts_on((
                &table.user_key,
                &table.agent_key,
                &table.recall_id,
                &table.memory_id,
                &table.kind,
            ))
            .exec(&mut transaction)
            .await?;
        transaction.commit().await?;
        Ok(affected)
    }

    fn prepare_rows(
        &self,
        batches: &[RecallBatch],
    ) -> Result<Vec<RecallEventRow>, RecallStoreError> {
        let count = batches.iter().try_fold(0usize, |count, batch| {
            count
                .checked_add(batch.len())
                .ok_or(RecallStoreError::BatchSizeOverflow)
        })?;
        if count > self.store.inner.config.max_record_batch {
            return Err(RecallStoreError::RecordBatchTooLarge {
                count,
                maximum: self.store.inner.config.max_record_batch,
            });
        }
        let mut rows = Vec::with_capacity(count);
        for batch in batches {
            self.append_batch_rows(batch, &mut rows)?;
        }
        Ok(rows)
    }

    fn append_batch_rows(
        &self,
        batch: &RecallBatch,
        rows: &mut Vec<RecallEventRow>,
    ) -> Result<(), RecallStoreError> {
        self.validate_batch_scope(batch)?;
        for event in batch.events() {
            let rank = event
                .rank
                .map(i32::try_from)
                .transpose()
                .map_err(|_| RecallStoreError::RankOutOfRange)?;
            rows.push(RecallEventRow {
                id: event.id,
                user_key: self.user_key.clone(),
                agent_key: self.agent_key.clone(),
                recall_id: batch.recall_id().as_uuid(),
                memory_id: event.memory_id.as_uuid(),
                kind: event.kind.as_str().to_owned(),
                rank,
                occurred_at: event.occurred_at,
                correction_evidence_id: event.correction_evidence_id.map(EvidenceId::as_uuid),
                sample_probability: event.sample_probability,
                aggregated_at: None,
            });
        }
        Ok(())
    }

    fn validate_batch_scope(&self, batch: &RecallBatch) -> Result<(), RecallStoreError> {
        if batch.user_key() != self.user_key || batch.agent_key() != self.agent_key {
            return Err(RecallStoreError::ScopeMismatch);
        }
        Ok(())
    }

    fn validate_receipt_scope(&self, receipt: &RecallReceipt) -> Result<(), RecallStoreError> {
        if receipt.user_key != self.user_key || receipt.agent_key != self.agent_key {
            return Err(RecallStoreError::ScopeMismatch);
        }
        Ok(())
    }
}

/// Failures isolated to optional recall recording and maintenance.
#[derive(Debug, Error)]
pub enum RecallStoreError {
    /// Builder validation found every listed configuration problem.
    #[error("invalid recall store configuration: {0:?}")]
    InvalidConfiguration(Vec<String>),
    /// A scope key was empty.
    #[error("{0} must not be empty")]
    EmptyScope(&'static str),
    /// A receipt or batch does not belong to the bound recorder.
    #[error("recall batch does not belong to the recorder scope")]
    ScopeMismatch,
    /// Summing event counts exceeded the platform's addressable size.
    #[error("recall batch size overflowed")]
    BatchSizeOverflow,
    /// One atomic recorder call exceeded its configured bound.
    #[error("recall batch contains {count} events; configured maximum is {maximum}")]
    RecordBatchTooLarge {
        /// Actual durable event count.
        count: usize,
        /// Configured maximum count.
        maximum: usize,
    },
    /// A candidate rank cannot be represented by the PostgreSQL schema.
    #[error("recall candidate rank exceeds PostgreSQL INTEGER")]
    RankOutOfRange,
    /// A maintenance batch limit was zero or exceeded PostgreSQL BIGINT.
    #[error("maintenance batch limit must be between one and PostgreSQL BIGINT maximum")]
    InvalidMaintenanceLimit,
    /// The configured retention interval cannot be subtracted from the current time.
    #[error("configured retention interval produced an invalid cutoff")]
    RetentionCutoffOverflow,
    /// Backend-neutral report validation failed.
    #[error(transparent)]
    Recall(#[from] RecallError),
    /// PostgreSQL recording or maintenance failed without affecting retrieval.
    #[error(transparent)]
    Database(#[from] db::DbError),
}

#[derive(Debug, sqlx::FromRow)]
struct ClaimedEvent {
    id: Uuid,
    memory_id: Uuid,
}

#[derive(Debug, sqlx::FromRow)]
struct AggregateRow {
    memory_id: Uuid,
    user_key: String,
    agent_key: String,
    window_start: DateTime<Utc>,
    sampled_retrieved_count: i64,
    estimated_retrieved_count: f64,
    accepted_count: i64,
    used_count: i64,
    dismissed_count: i64,
    corrected_count: i64,
    last_retrieved_at: Option<DateTime<Utc>>,
    last_accepted_at: Option<DateTime<Utc>>,
    last_used_at: Option<DateTime<Utc>>,
    last_dismissed_at: Option<DateTime<Utc>>,
    last_corrected_at: Option<DateTime<Utc>>,
    decayed_use_mass: f64,
}

impl AggregateRow {
    fn into_stats(self, anchor: DateTime<Utc>) -> RecallStatsRow {
        RecallStatsRow {
            memory_id: self.memory_id,
            user_key: self.user_key,
            agent_key: self.agent_key,
            window_start: self.window_start,
            sampled_retrieved_count: self.sampled_retrieved_count,
            estimated_retrieved_count: self.estimated_retrieved_count,
            accepted_count: self.accepted_count,
            used_count: self.used_count,
            dismissed_count: self.dismissed_count,
            corrected_count: self.corrected_count,
            last_retrieved_at: self.last_retrieved_at,
            last_accepted_at: self.last_accepted_at,
            last_used_at: self.last_used_at,
            last_dismissed_at: self.last_dismissed_at,
            last_corrected_at: self.last_corrected_at,
            decayed_use_mass: self.decayed_use_mass,
            decay_anchor: anchor,
            updated_at: anchor,
        }
    }
}

async fn claim_pending(
    transaction: &mut db::DbTransaction<'_>,
    batch_limit: i64,
) -> Result<Vec<ClaimedEvent>, RecallStoreError> {
    db::query(sql::CLAIM_PENDING_EVENTS)
        .bind("batch_limit", batch_limit)
        .all(transaction)
        .await
        .map_err(Into::into)
}

async fn aggregate_claimed(
    transaction: &mut db::DbTransaction<'_>,
    config: &RecallStoreConfig,
    claimed: &[ClaimedEvent],
) -> Result<(), RecallStoreError> {
    let memory_ids = unique_memory_ids(claimed.iter().map(|event| event.memory_id));
    let event_ids = claimed.iter().map(|event| event.id).collect::<Vec<_>>();
    let anchor = Utc::now();
    lock_memory_stats(transaction, &memory_ids).await?;
    let rows = aggregate_memory_ids(transaction, config, &memory_ids, anchor).await?;
    upsert_stats(transaction, &rows).await?;
    mark_events_aggregated(transaction, &event_ids, anchor).await
}

async fn recompute_affected(
    transaction: &mut db::DbTransaction<'_>,
    config: &RecallStoreConfig,
    affected: &[Uuid],
) -> Result<(), RecallStoreError> {
    let anchor = Utc::now();
    let rows = aggregate_memory_ids(transaction, config, affected, anchor).await?;
    let retained = rows.iter().map(|row| row.memory_id).collect::<Vec<_>>();
    upsert_stats(transaction, &rows).await?;
    delete_empty_stats(transaction, affected, &retained).await
}

async fn aggregate_memory_ids(
    transaction: &mut db::DbTransaction<'_>,
    config: &RecallStoreConfig,
    memory_ids: &[Uuid],
    anchor: DateTime<Utc>,
) -> Result<Vec<RecallStatsRow>, RecallStoreError> {
    let rows = db::query(sql::AGGREGATE_MEMORY_IDS)
        .bind("decay_anchor", anchor)
        .bind("half_life_seconds", half_life_seconds(config))
        .bind("memory_ids", memory_ids.to_vec())
        .all::<AggregateRow>(transaction)
        .await?;
    Ok(rows.into_iter().map(|row| row.into_stats(anchor)).collect())
}

async fn aggregate_scope(
    transaction: &mut db::DbTransaction<'_>,
    config: &RecallStoreConfig,
    user_key: &str,
    agent_key: &str,
) -> Result<Vec<RecallStatsRow>, RecallStoreError> {
    let anchor = Utc::now();
    let rows = db::query(sql::AGGREGATE_SCOPE)
        .bind("decay_anchor", anchor)
        .bind("half_life_seconds", half_life_seconds(config))
        .bind("user_key", user_key.to_owned())
        .bind("agent_key", agent_key.to_owned())
        .all::<AggregateRow>(transaction)
        .await?;
    Ok(rows.into_iter().map(|row| row.into_stats(anchor)).collect())
}

async fn upsert_stats(
    transaction: &mut db::DbTransaction<'_>,
    rows: &[RecallStatsRow],
) -> Result<(), RecallStoreError> {
    if rows.is_empty() {
        return Ok(());
    }
    let table = RecallStatsRow::table();
    db::from(&table)
        .upsert_many(rows, &table.memory_id)
        .update_only((
            &table.window_start,
            &table.sampled_retrieved_count,
            &table.estimated_retrieved_count,
            &table.accepted_count,
            &table.used_count,
            &table.dismissed_count,
            &table.corrected_count,
            &table.last_retrieved_at,
            &table.last_accepted_at,
            &table.last_used_at,
            &table.last_dismissed_at,
            &table.last_corrected_at,
            &table.decayed_use_mass,
            &table.decay_anchor,
            &table.updated_at,
        ))
        .exec(transaction)
        .await?;
    Ok(())
}

async fn mark_events_aggregated(
    transaction: &mut db::DbTransaction<'_>,
    event_ids: &[Uuid],
    aggregated_at: DateTime<Utc>,
) -> Result<(), RecallStoreError> {
    db::query(sql::MARK_EVENTS_AGGREGATED)
        .bind("aggregated_at", aggregated_at)
        .bind("event_ids", event_ids.to_vec())
        .execute(transaction)
        .await?;
    Ok(())
}

async fn claim_prunable(
    transaction: &mut db::DbTransaction<'_>,
    cutoff: DateTime<Utc>,
    batch_limit: i64,
) -> Result<Vec<ClaimedEvent>, RecallStoreError> {
    db::query(sql::CLAIM_PRUNABLE_EVENTS)
        .bind("cutoff", cutoff)
        .bind("batch_limit", batch_limit)
        .all(transaction)
        .await
        .map_err(Into::into)
}

async fn delete_prunable(
    transaction: &mut db::DbTransaction<'_>,
    claimed: &[ClaimedEvent],
) -> Result<u64, RecallStoreError> {
    let event_ids = claimed.iter().map(|event| event.id).collect::<Vec<_>>();
    let rows = db::query(sql::DELETE_PRUNABLE_EVENTS)
        .bind("event_ids", event_ids)
        .all::<(Uuid,)>(transaction)
        .await?;
    Ok(rows.len() as u64)
}

async fn lock_memory_stats(
    transaction: &mut db::DbTransaction<'_>,
    memory_ids: &[Uuid],
) -> Result<(), RecallStoreError> {
    db::query(sql::LOCK_MEMORY_STATS)
        .bind("memory_ids", memory_ids.to_vec())
        .all::<(bool,)>(transaction)
        .await?;
    Ok(())
}
async fn delete_empty_stats(
    transaction: &mut db::DbTransaction<'_>,
    affected: &[Uuid],
    retained: &[Uuid],
) -> Result<(), RecallStoreError> {
    db::query(sql::DELETE_EMPTY_STATS)
        .bind("affected_memory_ids", affected.to_vec())
        .bind("retained_memory_ids", retained.to_vec())
        .execute(transaction)
        .await?;
    Ok(())
}

async fn delete_scope_stats(
    transaction: &mut db::DbTransaction<'_>,
    user_key: &str,
    agent_key: &str,
) -> Result<(), RecallStoreError> {
    db::query(sql::DELETE_SCOPE_STATS)
        .bind("user_key", user_key.to_owned())
        .bind("agent_key", agent_key.to_owned())
        .execute(transaction)
        .await?;
    Ok(())
}

async fn mark_scope_aggregated(
    transaction: &mut db::DbTransaction<'_>,
    user_key: &str,
    agent_key: &str,
) -> Result<(), RecallStoreError> {
    db::query(sql::MARK_SCOPE_AGGREGATED)
        .bind("aggregated_at", Utc::now())
        .bind("user_key", user_key.to_owned())
        .bind("agent_key", agent_key.to_owned())
        .execute(transaction)
        .await?;
    Ok(())
}

async fn lock_events_for_rebuild(
    transaction: &mut db::DbTransaction<'_>,
) -> Result<(), RecallStoreError> {
    db::query(sql::LOCK_EVENTS_FOR_REBUILD)
        .execute(transaction)
        .await?;
    Ok(())
}

fn half_life_seconds(config: &RecallStoreConfig) -> f64 {
    config.use_decay_half_life.num_milliseconds() as f64 / 1_000.0
}

fn unique_memory_ids(ids: impl IntoIterator<Item = Uuid>) -> Vec<Uuid> {
    let mut seen = HashSet::new();
    let mut ids = ids
        .into_iter()
        .filter(|id| seen.insert(*id))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

fn validate_scope_key(name: &'static str, value: String) -> Result<String, RecallStoreError> {
    if value.trim().is_empty() {
        Err(RecallStoreError::EmptyScope(name))
    } else {
        Ok(value)
    }
}

fn validate_maintenance_limit(limit: usize) -> Result<i64, RecallStoreError> {
    if limit == 0 {
        return Err(RecallStoreError::InvalidMaintenanceLimit);
    }
    i64::try_from(limit).map_err(|_| RecallStoreError::InvalidMaintenanceLimit)
}

/// Emits text-free operation volume, latency, and success telemetry.
fn trace_operation(
    operation: &'static str,
    started: Instant,
    requested_items: usize,
    result: &Result<u64, RecallStoreError>,
) {
    tracing::debug!(
        operation,
        requested_items,
        affected_rows = result.as_ref().copied().unwrap_or_default(),
        success = result.is_ok(),
        elapsed_micros = started.elapsed().as_micros() as u64,
        "memory recall store operation completed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryId;

    fn pool() -> db::DbPool {
        let sqlx_pool = sqlx::postgres::PgPoolOptions::new()
            .connect_lazy("postgres://localhost/pravah_recall_test")
            .expect("valid lazy PostgreSQL URL");
        db::DbPool::from_pool(sqlx_pool)
    }

    fn receipt() -> RecallReceipt {
        RecallReceipt::new(
            "user",
            "agent",
            vec![crate::RecallCandidate {
                memory_id: MemoryId::new(),
                rank: 1,
            }],
        )
        .expect("valid receipt")
    }

    /// Verifies the hot event write remains one typed idempotent PostgreSQL statement.
    #[test]
    fn event_insert_plan_uses_scoped_conflict_identity() {
        let table = RecallEventRow::table();
        let row = RecallEventRow {
            id: Uuid::now_v7(),
            user_key: "user".to_owned(),
            agent_key: "agent".to_owned(),
            recall_id: Uuid::now_v7(),
            memory_id: Uuid::now_v7(),
            kind: "used".to_owned(),
            rank: None,
            occurred_at: Utc::now(),
            correction_evidence_id: None,
            sample_probability: 1.0,
            aggregated_at: None,
        };
        let plan = db::from(&table)
            .insert_many(&[row])
            .single_statement()
            .ignore_conflicts_on((
                &table.user_key,
                &table.agent_key,
                &table.recall_id,
                &table.memory_id,
                &table.kind,
            ))
            .plan()
            .expect("valid typed event insert");

        assert!(plan.sql.contains("ON CONFLICT"));
        assert!(
            plan.sql
                .contains("user_key, agent_key, recall_id, memory_id, kind")
        );
        assert!(plan.sql.ends_with("DO NOTHING"));
    }

    /// Verifies builder errors accumulate until the asynchronous terminal method.
    #[tokio::test]
    async fn builder_accumulates_configuration_errors() {
        let result = RecallStore::builder(pool())
            .retention(Duration::zero())
            .use_decay_half_life(Duration::zero())
            .retrieved_sampling(f64::NAN)
            .max_record_batch(0)
            .build()
            .await;
        assert!(matches!(
            result,
            Err(RecallStoreError::InvalidConfiguration(errors)) if errors.len() == 4
        ));
    }

    /// Verifies the default recorder creates no durable retrieved rows.
    #[tokio::test]
    async fn default_recorder_disables_retrieved_events() {
        let store = RecallStore::builder(pool())
            .build()
            .await
            .expect("default store");
        let recorder = store.recorder("user", "agent").expect("valid scope");
        let batch = recorder.retrieved(&receipt()).expect("valid receipt");

        assert!(batch.is_empty());
    }

    /// Verifies a recorder rejects a valid batch from another application scope.
    #[tokio::test]
    async fn recorder_rejects_cross_scope_batches_before_database_access() {
        let store = RecallStore::builder(pool())
            .build()
            .await
            .expect("default store");
        let recorder = store.recorder("other", "agent").expect("valid scope");
        let receipt = receipt();
        let batch =
            RecallBatch::used(&receipt, [receipt.candidates[0].memory_id]).expect("valid outcome");
        let result = recorder.record_many(&[batch]).await;

        assert!(matches!(result, Err(RecallStoreError::ScopeMismatch)));
    }

    /// Verifies recorder event bounds are enforced before opening a transaction.
    #[tokio::test]
    async fn recorder_rejects_oversized_batches_before_database_access() {
        let store = RecallStore::builder(pool())
            .max_record_batch(1)
            .retrieved_sampling(1.0)
            .build()
            .await
            .expect("valid store");
        let receipt = RecallReceipt::new(
            "user",
            "agent",
            vec![
                crate::RecallCandidate {
                    memory_id: MemoryId::new(),
                    rank: 1,
                },
                crate::RecallCandidate {
                    memory_id: MemoryId::new(),
                    rank: 2,
                },
            ],
        )
        .expect("valid receipt");
        let recorder = store.recorder("user", "agent").expect("valid scope");
        let batch = recorder.retrieved(&receipt).expect("valid batch");
        let result = recorder.record_many(&[batch]).await;

        assert!(matches!(
            result,
            Err(RecallStoreError::RecordBatchTooLarge {
                count: 2,
                maximum: 1
            })
        ));
    }
}
