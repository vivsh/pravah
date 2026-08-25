use super::*;

/// Verifies symmetric relations use deterministic UUID ordering.
#[test]
fn symmetric_relation_ids_are_canonical() {
    let first = Uuid::now_v7();
    let second = Uuid::now_v7();
    let (left, right) = canonical_relation_ids(second, first, true).unwrap();
    assert!(left < right);
}

/// Verifies evidence hashing is deterministic and content-sensitive.
#[test]
fn evidence_hashes_are_deterministic() {
    assert_eq!(sha256("same"), sha256("same"));
    assert_ne!(sha256("same"), sha256("different"));
}

/// Verifies deterministic claim normalization catches casing and whitespace duplicates.
#[test]
fn memory_hashes_normalize_casing_and_whitespace() {
    assert_eq!(
        normalized_sha256("The user likes tea."),
        normalized_sha256("  the USER likes   tea. ")
    );
}

/// Verifies the production evidence lookup query binds every scope component.
#[test]
fn evidence_lookup_query_is_fully_scoped() {
    let table = EvidenceRow::table();
    let plan = evidence_by_key_query(&table, "user", "agent", "evidence")
        .plan()
        .unwrap();

    assert!(plan.sql.contains("user_key ="));
    assert!(plan.sql.contains("agent_key ="));
    assert!(plan.sql.contains("evidence_key ="));
    assert_eq!(plan.total_bind_count, 3);
}
