use chrono::{DateTime, Utc};
use mool as db;
use uuid::Uuid;

/// One durable, text-free recall observation.
#[derive(Debug, Clone, db::Model)]
#[table(name = "pravah_memory_recall_events")]
pub(crate) struct RecallEventRow {
    #[column(primary_key)]
    pub id: Uuid,
    pub user_key: String,
    pub agent_key: String,
    pub recall_id: Uuid,
    pub memory_id: Uuid,
    pub kind: String,
    pub rank: Option<i32>,
    pub occurred_at: DateTime<Utc>,
    pub correction_evidence_id: Option<Uuid>,
    pub sample_probability: f64,
    pub aggregated_at: Option<DateTime<Utc>>,
}

/// Rebuildable rolling recall statistics for one memory claim.
#[derive(Debug, Clone, db::Model)]
#[table(name = "pravah_memory_recall_stats")]
pub(crate) struct RecallStatsRow {
    #[column(primary_key)]
    pub memory_id: Uuid,
    pub user_key: String,
    pub agent_key: String,
    pub window_start: DateTime<Utc>,
    pub sampled_retrieved_count: i64,
    pub estimated_retrieved_count: f64,
    pub accepted_count: i64,
    pub used_count: i64,
    pub dismissed_count: i64,
    pub corrected_count: i64,
    pub last_retrieved_at: Option<DateTime<Utc>>,
    pub last_accepted_at: Option<DateTime<Utc>>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub last_dismissed_at: Option<DateTime<Utc>>,
    pub last_corrected_at: Option<DateTime<Utc>>,
    pub decayed_use_mass: f64,
    pub decay_anchor: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
