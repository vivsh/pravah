pub(crate) const CLAIM_PENDING_EVENTS: &str = r#"
SELECT id, memory_id
FROM pravah_memory_recall_events
WHERE aggregated_at IS NULL
ORDER BY occurred_at, id
LIMIT :batch_limit
FOR UPDATE SKIP LOCKED
"#;

pub(crate) const AGGREGATE_MEMORY_IDS: &str = r#"
SELECT
    memory_id,
    MIN(user_key) AS user_key,
    MIN(agent_key) AS agent_key,
    MIN(occurred_at) AS window_start,
    COUNT(*) FILTER (WHERE kind = 'retrieved') AS sampled_retrieved_count,
    COALESCE(SUM(CASE WHEN kind = 'retrieved' THEN 1.0 / sample_probability ELSE 0.0 END), 0.0)::double precision AS estimated_retrieved_count,
    COUNT(*) FILTER (WHERE kind = 'accepted') AS accepted_count,
    COUNT(*) FILTER (WHERE kind = 'used') AS used_count,
    COUNT(*) FILTER (WHERE kind = 'dismissed') AS dismissed_count,
    COUNT(*) FILTER (WHERE kind = 'corrected') AS corrected_count,
    MAX(occurred_at) FILTER (WHERE kind = 'retrieved') AS last_retrieved_at,
    MAX(occurred_at) FILTER (WHERE kind = 'accepted') AS last_accepted_at,
    MAX(occurred_at) FILTER (WHERE kind = 'used') AS last_used_at,
    MAX(occurred_at) FILTER (WHERE kind = 'dismissed') AS last_dismissed_at,
    MAX(occurred_at) FILTER (WHERE kind = 'corrected') AS last_corrected_at,
    COALESCE(SUM(
        CASE WHEN kind = 'used' THEN POWER(
            2.0,
            -GREATEST(EXTRACT(EPOCH FROM (:decay_anchor - occurred_at)), 0.0)
                / :half_life_seconds
        ) ELSE 0.0 END
    ), 0.0)::double precision AS decayed_use_mass
FROM pravah_memory_recall_events
WHERE memory_id = ANY(CAST(:memory_ids AS uuid[]))
GROUP BY memory_id
"#;

pub(crate) const AGGREGATE_SCOPE: &str = r#"
SELECT
    memory_id,
    MIN(user_key) AS user_key,
    MIN(agent_key) AS agent_key,
    MIN(occurred_at) AS window_start,
    COUNT(*) FILTER (WHERE kind = 'retrieved') AS sampled_retrieved_count,
    COALESCE(SUM(CASE WHEN kind = 'retrieved' THEN 1.0 / sample_probability ELSE 0.0 END), 0.0)::double precision AS estimated_retrieved_count,
    COUNT(*) FILTER (WHERE kind = 'accepted') AS accepted_count,
    COUNT(*) FILTER (WHERE kind = 'used') AS used_count,
    COUNT(*) FILTER (WHERE kind = 'dismissed') AS dismissed_count,
    COUNT(*) FILTER (WHERE kind = 'corrected') AS corrected_count,
    MAX(occurred_at) FILTER (WHERE kind = 'retrieved') AS last_retrieved_at,
    MAX(occurred_at) FILTER (WHERE kind = 'accepted') AS last_accepted_at,
    MAX(occurred_at) FILTER (WHERE kind = 'used') AS last_used_at,
    MAX(occurred_at) FILTER (WHERE kind = 'dismissed') AS last_dismissed_at,
    MAX(occurred_at) FILTER (WHERE kind = 'corrected') AS last_corrected_at,
    COALESCE(SUM(
        CASE WHEN kind = 'used' THEN POWER(
            2.0,
            -GREATEST(EXTRACT(EPOCH FROM (:decay_anchor - occurred_at)), 0.0)
                / :half_life_seconds
        ) ELSE 0.0 END
    ), 0.0)::double precision AS decayed_use_mass
FROM pravah_memory_recall_events
WHERE user_key = :user_key AND agent_key = :agent_key
GROUP BY memory_id
"#;

pub(crate) const MARK_EVENTS_AGGREGATED: &str = r#"
UPDATE pravah_memory_recall_events
SET aggregated_at = :aggregated_at
WHERE id = ANY(CAST(:event_ids AS uuid[]))
"#;

pub(crate) const CLAIM_PRUNABLE_EVENTS: &str = r#"
SELECT id, memory_id
FROM pravah_memory_recall_events
WHERE occurred_at < :cutoff
ORDER BY occurred_at, id
LIMIT :batch_limit
FOR UPDATE SKIP LOCKED
"#;

pub(crate) const LOCK_MEMORY_STATS: &str = r#"
SELECT pg_advisory_xact_lock(
    hashtextextended('pravah-memory-recall:' || memory_id::text, 0)
) IS NULL AS acquired
FROM unnest(CAST(:memory_ids AS uuid[])) AS ids(memory_id)
ORDER BY memory_id
"#;

pub(crate) const DELETE_PRUNABLE_EVENTS: &str = r#"
DELETE FROM pravah_memory_recall_events
WHERE id = ANY(CAST(:event_ids AS uuid[]))
RETURNING memory_id
"#;

pub(crate) const DELETE_EMPTY_STATS: &str = r#"
DELETE FROM pravah_memory_recall_stats
WHERE memory_id = ANY(CAST(:affected_memory_ids AS uuid[]))
  AND NOT (memory_id = ANY(CAST(:retained_memory_ids AS uuid[])))
"#;

pub(crate) const DELETE_SCOPE_STATS: &str = r#"
DELETE FROM pravah_memory_recall_stats
WHERE user_key = :user_key AND agent_key = :agent_key
"#;

pub(crate) const MARK_SCOPE_AGGREGATED: &str = r#"
UPDATE pravah_memory_recall_events
SET aggregated_at = :aggregated_at
WHERE user_key = :user_key AND agent_key = :agent_key
"#;

pub(crate) const LOCK_EVENTS_FOR_REBUILD: &str =
    "LOCK TABLE pravah_memory_recall_events IN ACCESS EXCLUSIVE MODE";

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies pruning returns every deleted event so its result is a row count.
    #[test]
    fn prune_query_does_not_collapse_deleted_memories() {
        assert!(DELETE_PRUNABLE_EVENTS.contains("RETURNING memory_id"));
        assert!(!DELETE_PRUNABLE_EVENTS.contains("DISTINCT"));
    }

    /// Verifies pending aggregation uses lock-skipping ownership in stable order.
    #[test]
    fn pending_query_uses_skip_locked() {
        assert!(CLAIM_PENDING_EVENTS.contains("ORDER BY occurred_at, id"));
        assert!(CLAIM_PENDING_EVENTS.contains("FOR UPDATE SKIP LOCKED"));
    }

    /// Verifies retention claims are bounded and lock-skipping before deletion.
    #[test]
    fn retention_query_claims_before_deletion() {
        assert!(CLAIM_PRUNABLE_EVENTS.contains("LIMIT :batch_limit"));
        assert!(CLAIM_PRUNABLE_EVENTS.contains("FOR UPDATE SKIP LOCKED"));
    }

    /// Verifies the advisory-lock statement keeps typed named-array binding.
    #[test]
    fn memory_lock_query_accepts_named_uuid_array() {
        let statement = mool::query(LOCK_MEMORY_STATS)
            .bind("memory_ids", vec![uuid::Uuid::now_v7()])
            .to_statement();

        assert!(statement.is_ok());
    }
}
