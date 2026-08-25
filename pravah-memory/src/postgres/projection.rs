use mool as db;
use uuid::Uuid;

use super::repository::RepositoryError;

pub(super) async fn recompute_projection(
    session: &mut impl db::DbSession,
    user_key: &str,
    agent_key: &str,
    seed_ids: &[Uuid],
    max_nodes: u32,
) -> Result<bool, RepositoryError> {
    let ids = relation_neighbourhood(
        session,
        user_key,
        agent_key,
        seed_ids,
        max_nodes.saturating_add(1),
    )
    .await?;
    if ids.len() > max_nodes as usize {
        return Ok(false);
    }
    recompute_projection_ids(session, &ids).await?;
    Ok(true)
}

/// Expands seed claims to their bounded-by-storage relation neighbourhood.
pub(super) async fn relation_neighbourhood(
    session: &mut impl db::DbSession,
    user_key: &str,
    agent_key: &str,
    seed_ids: &[Uuid],
    limit: u32,
) -> Result<Vec<Uuid>, RepositoryError> {
    if seed_ids.is_empty() {
        return Ok(Vec::new());
    }
    db::query(
        "WITH RECURSIVE neighbourhood(id) AS (\
         SELECT UNNEST(:seed_ids) UNION SELECT CASE WHEN relation.from_memory_id = neighbourhood.id \
         THEN relation.to_memory_id ELSE relation.from_memory_id END \
         FROM pravah_memory_relations relation JOIN neighbourhood \
         ON relation.from_memory_id = neighbourhood.id OR relation.to_memory_id = neighbourhood.id \
         WHERE relation.user_key = :user_key AND relation.agent_key = :agent_key) \
         SELECT id FROM neighbourhood LIMIT :limit",
    )
    .bind("seed_ids", seed_ids.to_vec())
    .bind("user_key", user_key.to_owned())
    .bind("agent_key", agent_key.to_owned())
    .bind("limit", i64::from(limit))
    .all::<(Uuid,)>(session)
    .await
    .map(|rows| rows.into_iter().map(|(id,)| id).collect())
    .map_err(Into::into)
}

/// Rebuilds only the relation component affected by a stale or relation change.
pub(super) async fn recompute_projection_ids(
    session: &mut impl db::DbSession,
    memory_ids: &[Uuid],
) -> Result<(), RepositoryError> {
    if memory_ids.is_empty() {
        return Ok(());
    }
    for sql in PROJECTION_UPDATES {
        db::query(sql)
            .bind("memory_ids", memory_ids.to_vec())
            .execute(session)
            .await?;
    }
    Ok(())
}

const PROJECTION_UPDATES: [&str; 3] = [
    "UPDATE pravah_memories SET current_for_retrieval = NOT stale \
     WHERE id = ANY(:memory_ids)",
    "UPDATE pravah_memories AS older SET current_for_retrieval = FALSE \
     FROM pravah_memory_relations AS relation, pravah_memories AS newer \
     WHERE relation.kind = 'supersedes' AND relation.to_memory_id = older.id \
       AND relation.from_memory_id = newer.id AND newer.stale = FALSE \
       AND (relation.effective_at IS NULL OR relation.effective_at <= now()) \
       AND older.id = ANY(:memory_ids)",
    "WITH RECURSIVE active_edges(left_id, right_id) AS ( \
         SELECT relation.from_memory_id, relation.to_memory_id \
         FROM pravah_memory_relations relation \
         JOIN pravah_memories left_memory ON left_memory.id = relation.from_memory_id \
         JOIN pravah_memories right_memory ON right_memory.id = relation.to_memory_id \
         WHERE relation.kind = 'corroborates' AND NOT left_memory.stale AND NOT right_memory.stale \
           AND relation.from_memory_id = ANY(:memory_ids) AND relation.to_memory_id = ANY(:memory_ids) \
     ), reachable(root_id, memory_id) AS ( \
         SELECT memory.id, memory.id FROM pravah_memories memory \
         WHERE memory.id = ANY(:memory_ids) AND NOT memory.stale \
         UNION \
         SELECT reachable.root_id, CASE WHEN edge.left_id = reachable.memory_id \
             THEN edge.right_id ELSE edge.left_id END \
         FROM reachable JOIN active_edges edge \
           ON edge.left_id = reachable.memory_id OR edge.right_id = reachable.memory_id \
     ), representatives AS ( \
         SELECT memory_id, MIN(root_id::text)::uuid AS representative_id \
         FROM reachable GROUP BY memory_id \
     ) \
     UPDATE pravah_memories AS duplicate SET current_for_retrieval = FALSE \
     FROM representatives \
     WHERE duplicate.id = representatives.memory_id \
       AND duplicate.id <> representatives.representative_id",
];
