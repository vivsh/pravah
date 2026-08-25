use super::*;

/// Creates a minimal non-stale row for relation-resolution tests.
fn memory_row(id: Uuid) -> MemoryRow {
    MemoryRow {
        id,
        user_key: "user".to_owned(),
        agent_key: "agent".to_owned(),
        evidence_id: Uuid::now_v7(),
        position: 0,
        text: "claim".to_owned(),
        content_hash: vec![0],
        kind: "fact".to_owned(),
        valid_from: None,
        valid_until: None,
        event_at: None,
        temporal_precision: "unknown".to_owned(),
        temporal_state: "unspecified".to_owned(),
        embedding: Vector::from([1.0]),
        metadata: serde_json::json!({}),
        created_at: chrono::Utc::now(),
        stale: false,
        current_for_retrieval: true,
    }
}

/// Creates one deterministic relation row for set-wise resolution tests.
fn relation(from: Uuid, to: Uuid, kind: &str) -> MemoryRelationRow {
    MemoryRelationRow {
        from_memory_id: from,
        to_memory_id: to,
        user_key: "user".to_owned(),
        agent_key: "agent".to_owned(),
        origin_evidence_id: Uuid::now_v7(),
        kind: kind.to_owned(),
        effective_at: None,
        reconciler_revision: "r1".to_owned(),
        created_at: chrono::Utc::now(),
    }
}

/// Creates a fixed current relation view so temporal assertions are deterministic.
fn current_relation_view() -> RelationView {
    RelationView {
        known_at: None,
        supersession_at: Some(chrono::Utc::now()),
    }
}

/// Verifies reciprocal-rank fusion rewards candidates present in multiple channels.
#[test]
fn reciprocal_rank_fusion_combines_channels() {
    let first = Uuid::now_v7();
    let shared = Uuid::now_v7();
    let fused =
        reciprocal_rank_fusion(vec![(1.0, vec![first, shared]), (1.0, vec![shared])], 60, 2);
    assert_eq!(fused[0].0, shared);
}

/// Verifies every temporal mode selects an explicit bounded predicate.
#[test]
fn temporal_modes_have_explicit_predicates() {
    let current = timeline_predicate(&SearchTimeline::default(), true);
    assert!(current.contains("memory.current_for_retrieval = TRUE"));
    let history = timeline_predicate(
        &SearchTimeline {
            view: ClaimView::AllVersions,
            valid_time: ValidTime::Any,
            ..Default::default()
        },
        true,
    );
    assert!(!history.contains("memory.current_for_retrieval = TRUE"));
}

/// Verifies unresolved conflict counterparts survive a fused top-K boundary.
#[test]
fn conflict_expansion_returns_both_claims() {
    let first = Uuid::from_u128(1);
    let second = Uuid::from_u128(2);
    let conflicts = BTreeMap::from([
        (first, BTreeSet::from([second])),
        (second, BTreeSet::from([first])),
    ]);
    let expanded = expand_conflicts(vec![(first, 1.0)], &conflicts);
    assert_eq!(expanded.len(), 2);
    assert!(expanded.iter().any(|(id, _)| *id == second));
}

/// Verifies a multi-hop corroboration chain collapses to one exact representative.
#[test]
fn corroboration_chain_has_exact_support() {
    let first = Uuid::from_u128(1);
    let second = Uuid::from_u128(2);
    let third = Uuid::from_u128(3);
    let memories = [first, second, third]
        .into_iter()
        .map(|id| (id, memory_row(id)))
        .collect();
    let relations = vec![
        relation(first, second, "corroborates"),
        relation(second, third, "corroborates"),
    ];
    let resolved = resolve_relations(&relations, &memories, &current_relation_view());
    let ranked = resolve_ranked(vec![(third, 1.0)], 0.0, 8, &resolved);
    assert_eq!(ranked, vec![(first, 1.0)]);
    assert_eq!(resolved.support.get(&first), Some(&3));
}

/// Verifies conflicts between corroborating components return both representatives.
#[test]
fn conflict_returns_both_complete_components() {
    let first = Uuid::from_u128(1);
    let first_support = Uuid::from_u128(2);
    let second = Uuid::from_u128(3);
    let second_support = Uuid::from_u128(4);
    let memories = [first, first_support, second, second_support]
        .into_iter()
        .map(|id| (id, memory_row(id)))
        .collect();
    let relations = vec![
        relation(first, first_support, "corroborates"),
        relation(second, second_support, "corroborates"),
        relation(first_support, second_support, "conflicts"),
    ];
    let resolved = resolve_relations(&relations, &memories, &current_relation_view());
    let ranked = resolve_ranked(vec![(first_support, 1.0)], 0.0, 1, &resolved);
    assert_eq!(ranked, vec![(first, 1.0), (second, 1.0)]);
    assert_eq!(resolved.support.get(&first), Some(&2));
    assert_eq!(resolved.support.get(&second), Some(&2));
}

/// Verifies applicable supersession chains suppress every older component.
#[test]
fn supersession_chain_suppresses_older_claims() {
    let oldest = Uuid::from_u128(1);
    let middle = Uuid::from_u128(2);
    let newest = Uuid::from_u128(3);
    let memories = [oldest, middle, newest]
        .into_iter()
        .map(|id| (id, memory_row(id)))
        .collect();
    let relations = vec![
        relation(middle, oldest, "supersedes"),
        relation(newest, middle, "supersedes"),
    ];
    let resolved = resolve_relations(&relations, &memories, &current_relation_view());
    let ranked = resolve_ranked(
        vec![(oldest, 1.0), (middle, 0.9), (newest, 0.8)],
        0.0,
        8,
        &resolved,
    );
    assert_eq!(ranked, vec![(newest, 0.8)]);
}

/// Verifies the raw query limits unique emitted edge rows at the exact sentinel.
#[test]
fn relation_query_is_recursive_and_bounded() {
    assert!(RELATION_NEIGHBOURHOOD_SQL.contains("WITH RECURSIVE walk"));
    assert!(RELATION_NEIGHBOURHOOD_SQL.contains("CROSS JOIN LATERAL"));
    assert!(RELATION_NEIGHBOURHOOD_SQL.contains("WHERE row_kind = 1"));
    assert!(RELATION_NEIGHBOURHOOD_SQL.contains("LIMIT :edge_limit"));
    assert!(RELATION_NEIGHBOURHOOD_SQL.contains("NOT walk.crossed_conflict"));
    assert!(!RELATION_NEIGHBOURHOOD_SQL.contains("UNION ALL"));
    assert!(!RELATION_NEIGHBOURHOOD_SQL.contains("EXISTS ("));
    assert_eq!(relation_query_limit(2_000), 2_001);
}
