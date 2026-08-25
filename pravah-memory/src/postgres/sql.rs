pub(crate) const RECONCILIATION_CANDIDATES: &str = r#"
WITH incoming AS (
    SELECT memory.*
    FROM pravah_memories memory
    WHERE memory.id = ANY(:claim_ids)
      AND memory.user_key = :user_key
      AND memory.agent_key = :agent_key
), channel_hits AS (
    SELECT incoming.id AS incoming_id, hit.id AS existing_id,
           8.0 / (60.0 + hit.rank) AS score
    FROM incoming
    CROSS JOIN LATERAL (
        SELECT existing.id,
               ROW_NUMBER() OVER (ORDER BY existing.id)::double precision AS rank
        FROM pravah_memories existing
        WHERE existing.user_key = :user_key
          AND existing.agent_key = :agent_key
          AND existing.evidence_id <> :evidence_id
          AND NOT existing.stale
          AND (existing.current_for_retrieval OR EXISTS (
              SELECT 1 FROM pravah_memory_scopes scope
              WHERE scope.user_key = :user_key AND scope.agent_key = :agent_key
                AND scope.projection_pending
          ))
          AND existing.content_hash = incoming.content_hash
        ORDER BY existing.id
        LIMIT :per_claim_limit
    ) hit
    UNION ALL
    SELECT incoming.id, hit.id, 4.0 / (60.0 + hit.rank)
    FROM incoming
    CROSS JOIN LATERAL (
        SELECT existing_link.memory_id AS id,
               ROW_NUMBER() OVER (
                   ORDER BY COUNT(*) DESC, existing_link.memory_id
               )::double precision AS rank
        FROM pravah_memory_entities incoming_link
        JOIN pravah_memory_entities existing_link
          ON existing_link.entity_id = incoming_link.entity_id
        JOIN pravah_memories existing ON existing.id = existing_link.memory_id
        WHERE incoming_link.memory_id = incoming.id
          AND existing.user_key = :user_key
          AND existing.agent_key = :agent_key
          AND existing.evidence_id <> :evidence_id
          AND NOT existing.stale
          AND (existing.current_for_retrieval OR EXISTS (
              SELECT 1 FROM pravah_memory_scopes scope
              WHERE scope.user_key = :user_key AND scope.agent_key = :agent_key
                AND scope.projection_pending
          ))
        GROUP BY existing_link.memory_id
        ORDER BY COUNT(*) DESC, existing_link.memory_id
        LIMIT :per_claim_limit
    ) hit
    UNION ALL
    SELECT incoming.id, hit.id, 2.0 / (60.0 + hit.rank)
    FROM incoming
    CROSS JOIN LATERAL (
        SELECT existing.id,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank_cd(existing.search_vector, query) DESC, existing.id
               )::double precision AS rank
        FROM pravah_memories existing,
             plainto_tsquery(CAST(:text_search_configuration AS regconfig), incoming.text) query
        WHERE existing.search_vector @@ query
          AND existing.user_key = :user_key
          AND existing.agent_key = :agent_key
          AND existing.evidence_id <> :evidence_id
          AND NOT existing.stale
          AND (existing.current_for_retrieval OR EXISTS (
              SELECT 1 FROM pravah_memory_scopes scope
              WHERE scope.user_key = :user_key AND scope.agent_key = :agent_key
                AND scope.projection_pending
          ))
        ORDER BY ts_rank_cd(existing.search_vector, query) DESC, existing.id
        LIMIT :per_claim_limit
    ) hit
    UNION ALL
    SELECT incoming.id, hit.id, 1.0 / (60.0 + hit.rank)
    FROM incoming
    CROSS JOIN LATERAL (
        SELECT existing.id,
               ROW_NUMBER() OVER (
                   ORDER BY existing.embedding <=> incoming.embedding, existing.id
               )::double precision AS rank
        FROM pravah_memories existing
        WHERE existing.user_key = :user_key
          AND existing.agent_key = :agent_key
          AND existing.evidence_id <> :evidence_id
          AND NOT existing.stale
          AND (existing.current_for_retrieval OR EXISTS (
              SELECT 1 FROM pravah_memory_scopes scope
              WHERE scope.user_key = :user_key AND scope.agent_key = :agent_key
                AND scope.projection_pending
          ))
          AND (existing.embedding <=> incoming.embedding) <= :max_vector_distance
        ORDER BY existing.embedding <=> incoming.embedding, existing.id
        LIMIT :per_claim_limit
    ) hit
    UNION ALL
    SELECT incoming.id, hit.id, 1.5 / (60.0 + hit.rank)
    FROM incoming
    CROSS JOIN LATERAL (
        SELECT existing.id,
               ROW_NUMBER() OVER (
                   ORDER BY existing.created_at DESC, existing.id
               )::double precision AS rank
        FROM pravah_memories existing
        WHERE existing.user_key = :user_key
          AND existing.agent_key = :agent_key
          AND existing.evidence_id <> :evidence_id
          AND NOT existing.stale
          AND (existing.current_for_retrieval OR EXISTS (
              SELECT 1 FROM pravah_memory_scopes scope
              WHERE scope.user_key = :user_key AND scope.agent_key = :agent_key
                AND scope.projection_pending
          ))
          AND (existing.valid_from IS NOT NULL
               OR existing.valid_until IS NOT NULL
               OR existing.event_at IS NOT NULL)
          AND (incoming.valid_from IS NOT NULL
               OR incoming.valid_until IS NOT NULL
               OR incoming.event_at IS NOT NULL)
          AND COALESCE(existing.valid_from, existing.event_at, '-infinity')
              <= COALESCE(incoming.valid_until, incoming.event_at, 'infinity')
          AND COALESCE(incoming.valid_from, incoming.event_at, '-infinity')
              <= COALESCE(existing.valid_until, existing.event_at, 'infinity')
        ORDER BY existing.created_at DESC, existing.id
        LIMIT :per_claim_limit
    ) hit
), per_claim AS (
    SELECT incoming_id, existing_id, SUM(score) AS score,
           ROW_NUMBER() OVER (
               PARTITION BY incoming_id
               ORDER BY SUM(score) DESC, existing_id
           ) AS candidate_rank
    FROM channel_hits
    GROUP BY incoming_id, existing_id
)
SELECT existing_id
FROM per_claim
WHERE candidate_rank <= :per_claim_limit
GROUP BY existing_id
ORDER BY SUM(score) DESC, existing_id
LIMIT :limit
"#;
