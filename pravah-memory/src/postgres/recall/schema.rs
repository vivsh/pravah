use mool as db;

use super::models::{RecallEventRow, RecallStatsRow};

/// Adds the optional, independently deployable recall telemetry tables.
pub trait MemoryRecallSchemaExt {
    /// Registers exactly the recall event log and its rebuildable statistics table.
    fn with_memory_recall(self) -> Self;
}

impl MemoryRecallSchemaExt for db::SchemaBuilder {
    fn with_memory_recall(self) -> Self {
        self.model::<RecallEventRow>()
            .model::<RecallStatsRow>()
            .extend_table("pravah_memory_recall_events", event_schema)
            .extend_table("pravah_memory_recall_stats", stats_schema)
            .opaque(
                "CREATE INDEX pravah_memory_recall_events_pending_idx \
                 ON pravah_memory_recall_events (occurred_at, id) \
                 WHERE aggregated_at IS NULL",
            )
            .opaque(
                "CREATE INDEX pravah_memory_recall_events_occurred_brin_idx \
                 ON pravah_memory_recall_events USING BRIN (occurred_at)",
            )
    }
}

fn event_schema(table: db::schema::TableBuilder) -> db::schema::TableBuilder {
    table
        .index(unique_index(
            "pravah_memory_recall_events_identity_key",
            &["user_key", "agent_key", "recall_id", "memory_id", "kind"],
        ))
        .foreign_key_columns_with(
            &["user_key", "agent_key", "memory_id"],
            "pravah_memories",
            &["user_key", "agent_key", "id"],
            |foreign_key| foreign_key.on_delete("CASCADE"),
        )
        .foreign_key_columns_with(
            &["user_key", "agent_key", "correction_evidence_id"],
            "pravah_evidence",
            &["user_key", "agent_key", "id"],
            |foreign_key| foreign_key.on_delete("CASCADE"),
        )
        .index(index(
            "pravah_memory_recall_events_history_idx",
            &["user_key", "agent_key", "memory_id", "occurred_at"],
        ))
        .check(
            "pravah_memory_recall_events_kind_check",
            "kind IN ('retrieved', 'accepted', 'used', 'dismissed', 'corrected')",
        )
        .check(
            "pravah_memory_recall_events_probability_check",
            "sample_probability > 0 AND sample_probability <= 1",
        )
        .check(
            "pravah_memory_recall_events_rank_check",
            "(kind = 'retrieved' AND rank IS NOT NULL AND rank > 0) OR \
             (kind <> 'retrieved' AND rank IS NULL)",
        )
        .check(
            "pravah_memory_recall_events_correction_check",
            "(kind = 'corrected' AND correction_evidence_id IS NOT NULL) OR \
             (kind <> 'corrected' AND correction_evidence_id IS NULL)",
        )
        .check(
            "pravah_memory_recall_events_explicit_probability_check",
            "kind = 'retrieved' OR sample_probability = 1.0",
        )
}

fn stats_schema(table: db::schema::TableBuilder) -> db::schema::TableBuilder {
    table
        .foreign_key_columns_with(
            &["user_key", "agent_key", "memory_id"],
            "pravah_memories",
            &["user_key", "agent_key", "id"],
            |foreign_key| foreign_key.on_delete("CASCADE"),
        )
        .index(index(
            "pravah_memory_recall_stats_scope_idx",
            &["user_key", "agent_key"],
        ))
        .check(
            "pravah_memory_recall_stats_counts_check",
            "sampled_retrieved_count >= 0 AND estimated_retrieved_count >= 0 AND \
             accepted_count >= 0 AND used_count >= 0 AND dismissed_count >= 0 AND \
             corrected_count >= 0",
        )
        .check(
            "pravah_memory_recall_stats_mass_check",
            "decayed_use_mass >= 0 AND decayed_use_mass <> 'NaN'::double precision AND \
             decayed_use_mass < 'Infinity'::double precision AND \
             estimated_retrieved_count <> 'NaN'::double precision AND \
             estimated_retrieved_count < 'Infinity'::double precision",
        )
}

fn index(name: &str, columns: &[&str]) -> db::schema::Index {
    db::schema::Index::columns(columns.iter().copied()).named(name)
}

fn unique_index(name: &str, columns: &[&str]) -> db::schema::Index {
    index(name, columns).unique()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postgres::{MemoryProfile, MemorySchemaExt};

    #[derive(Debug, Clone, db::Model)]
    #[table(name = "pravah_evidence")]
    struct EvidenceStub {
        #[column(primary_key)]
        id: uuid::Uuid,
        user_key: String,
        agent_key: String,
    }

    #[derive(Debug, Clone, db::Model)]
    #[table(name = "pravah_memories")]
    struct MemoryStub {
        #[column(primary_key)]
        id: uuid::Uuid,
        user_key: String,
        agent_key: String,
    }

    /// Verifies the optional extension contributes only its two telemetry tables.
    #[test]
    fn recall_schema_adds_exactly_two_tables() {
        let base = db::schema()
            .model::<EvidenceStub>()
            .model::<MemoryStub>()
            .extend_table("pravah_evidence", |table| {
                table.index(unique_index(
                    "pravah_evidence_scope_id_key",
                    &["user_key", "agent_key", "id"],
                ))
            })
            .extend_table("pravah_memories", |table| {
                table.index(unique_index(
                    "pravah_memories_scope_id_key",
                    &["user_key", "agent_key", "id"],
                ))
            })
            .build()
            .expect("valid base schema");
        let extended = db::schema()
            .model::<EvidenceStub>()
            .model::<MemoryStub>()
            .extend_table("pravah_evidence", |table| {
                table.index(unique_index(
                    "pravah_evidence_scope_id_key",
                    &["user_key", "agent_key", "id"],
                ))
            })
            .extend_table("pravah_memories", |table| {
                table.index(unique_index(
                    "pravah_memories_scope_id_key",
                    &["user_key", "agent_key", "id"],
                ))
            })
            .with_memory_recall()
            .build()
            .expect("valid recall schema");

        assert_eq!(extended.tables.len(), base.tables.len() + 2);
        assert!(extended.tables.contains_key("pravah_memory_recall_events"));
        assert!(extended.tables.contains_key("pravah_memory_recall_stats"));
    }

    /// Verifies the event table owns idempotency, scope, and field-consistency constraints.
    #[test]
    fn recall_schema_registers_event_contract() {
        let schema = db::schema()
            .model::<EvidenceStub>()
            .model::<MemoryStub>()
            .extend_table("pravah_evidence", |table| {
                table.index(unique_index(
                    "pravah_evidence_scope_id_key",
                    &["user_key", "agent_key", "id"],
                ))
            })
            .extend_table("pravah_memories", |table| {
                table.index(unique_index(
                    "pravah_memories_scope_id_key",
                    &["user_key", "agent_key", "id"],
                ))
            })
            .with_memory_recall()
            .build()
            .expect("valid recall schema");
        let events = &schema.tables["pravah_memory_recall_events"];

        assert!(events.indexes.iter().any(|index| {
            index.name == "pravah_memory_recall_events_identity_key" && index.unique
        }));
        assert!(events.constraints.iter().any(|constraint| {
            constraint.name() == "pravah_memory_recall_events_explicit_probability_check"
        }));
    }

    /// Verifies recall composes additively with the real seven-table memory schema.
    #[test]
    fn recall_schema_composes_with_real_memory_family() {
        let profile = memory_profile();
        let base = db::schema()
            .with_memory(profile.clone())
            .build()
            .expect("valid memory schema");
        let extended = db::schema()
            .with_memory(profile)
            .with_memory_recall()
            .build()
            .expect("valid memory and recall schema");

        assert_eq!(base.tables.len(), 7);
        assert_eq!(extended.tables.len(), 9);
        assert!(extended.tables.contains_key("pravah_memory_recall_events"));
        assert!(extended.tables.contains_key("pravah_memory_recall_stats"));
    }

    /// Verifies the memory-v2 to recall migration creates only optional telemetry objects.
    #[test]
    fn recall_migration_does_not_mutate_base_memory_tables() {
        let profile = memory_profile();
        let base = db::schema()
            .with_memory(profile.clone())
            .build()
            .expect("valid memory schema");
        let desired = db::schema()
            .with_memory(profile)
            .with_memory_recall()
            .build()
            .expect("valid memory and recall schema");
        let empty = db::gaman::core::OfflinePlanner::new(db::gaman::core::Dialect::Postgres);
        let initial = make_reviewed_migration(&empty, base);
        let planner = db::gaman::core::OfflinePlanner::new(db::gaman::core::Dialect::Postgres)
            .from_migrations(vec![initial]);
        let recall = make_reviewed_migration(&planner, desired);
        let sql = planner
            .sql_migrate(&[recall])
            .expect("recall migration SQL")
            .join("\n");

        assert!(sql.contains("pravah_memory_recall_events"));
        assert!(sql.contains("pravah_memory_recall_stats"));
        assert!(!sql.contains("DROP "));
        for table in base_table_names() {
            assert!(!sql.contains(&format!("ALTER TABLE {table}")));
            assert!(!sql.contains(&format!("ALTER TABLE \"{table}\"")));
        }
    }

    fn memory_profile() -> MemoryProfile {
        MemoryProfile::new(
            "model",
            "r1",
            3,
            "document-v1",
            "extract-v1",
            "reconcile-v1",
        )
    }

    fn make_reviewed_migration(
        planner: &db::gaman::core::OfflinePlanner,
        schema: db::gaman::schema::Schema,
    ) -> db::gaman::Migration {
        let pending = planner
            .make_migration(schema.clone(), &[])
            .expect_err("opaque indexes require review");
        let db::gaman::core::OfflineError::NeedsInput(clarifications) = pending else {
            panic!("expected opaque-index review");
        };
        let decisions = clarifications
            .into_iter()
            .map(|clarification| db::gaman::core::Decision {
                clarification_id: clarification.id,
                answer: db::gaman::core::Answer::AcceptRisk,
            })
            .collect::<Vec<_>>();
        planner
            .make_migration(schema, &decisions)
            .expect("reviewed migration")
            .expect("non-empty migration")
    }

    fn base_table_names() -> [&'static str; 7] {
        [
            "pravah_evidence",
            "pravah_memories",
            "pravah_entities",
            "pravah_memory_entities",
            "pravah_memory_relations",
            "pravah_memory_scopes",
            "pravah_memory_profile",
        ]
    }
}
