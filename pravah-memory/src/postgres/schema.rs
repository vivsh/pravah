use mool as db;

use super::models::{
    EntityRow, EvidenceRow, MemoryEntityRow, MemoryProfileRow, MemoryRelationRow, MemoryRow,
    MemoryScopeRow,
};

/// Desired singleton database profile registered with the memory schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryProfile {
    /// Memory schema contract revision.
    pub schema_version: i32,
    /// Provider/model name.
    pub embedding_model: String,
    /// Provider deployment or model revision.
    pub embedding_revision: String,
    /// Exact active embedding dimensions.
    pub embedding_dimensions: i32,
    /// Text formatting revision passed to the embedder.
    pub document_format_revision: String,
    /// PostgreSQL text search configuration.
    pub text_search_configuration: String,
    /// Active extraction contract revision.
    pub extractor_revision: String,
    /// Active relation-classification revision.
    pub reconciler_revision: String,
    /// Rebuildable retrieval projection revision.
    pub derived_index_revision: i32,
}

impl MemoryProfile {
    /// Creates the v2 cosine-profile defaults around an embedding provider.
    pub fn new(
        embedding_model: impl Into<String>,
        embedding_revision: impl Into<String>,
        embedding_dimensions: i32,
        document_format_revision: impl Into<String>,
        extractor_revision: impl Into<String>,
        reconciler_revision: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: 2,
            embedding_model: embedding_model.into(),
            embedding_revision: embedding_revision.into(),
            embedding_dimensions,
            document_format_revision: document_format_revision.into(),
            text_search_configuration: "simple".to_owned(),
            extractor_revision: extractor_revision.into(),
            reconciler_revision: reconciler_revision.into(),
            derived_index_revision: 2,
        }
    }
}

/// Adds Pravah's fixed app-wide memory table family to a Mool schema builder.
pub trait MemorySchemaExt {
    /// Registers all tables, constraints, indexes, extension, and profile data.
    fn with_memory(self, profile: MemoryProfile) -> Self;
}

impl MemorySchemaExt for db::SchemaBuilder {
    fn with_memory(self, profile: MemoryProfile) -> Self {
        let dimensions = profile.embedding_dimensions;
        let text_config = profile.text_search_configuration.clone();
        self.extension("vector")
            .model::<EvidenceRow>()
            .model::<MemoryRow>()
            .model::<EntityRow>()
            .model::<MemoryEntityRow>()
            .model::<MemoryRelationRow>()
            .model::<MemoryScopeRow>()
            .model::<MemoryProfileRow>()
            .managed_rows::<MemoryProfileRow>([profile_row(profile)])
            .extend_table("pravah_evidence", evidence_schema)
            .extend_table("pravah_memories", move |table| {
                memory_schema(table, dimensions, &text_config)
            })
            .extend_table("pravah_entities", entity_schema)
            .extend_table("pravah_memory_entities", memory_entity_schema)
            .extend_table("pravah_memory_relations", memory_relation_schema)
            .extend_table("pravah_memory_scopes", |table| {
                table
                    .check(
                        "pravah_memory_scopes_revision_check",
                        "scope_revision >= 0 AND projection_revision >= 0 AND projection_revision <= scope_revision",
                    )
                    .check(
                        "pravah_memory_scopes_lease_pair_check",
                        "(reconciliation_token IS NULL) = (reconciliation_lease_until IS NULL)",
                    )
                    .index(index(
                        "pravah_memory_scopes_reconciliation_idx",
                        &["reconciliation_lease_until"],
                    ))
            })
            .opaque("CREATE INDEX pravah_memories_current_text_idx ON pravah_memories USING GIN (search_vector) WHERE stale = FALSE AND current_for_retrieval = TRUE")
            .opaque(format!("CREATE INDEX pravah_memories_current_embedding_idx ON pravah_memories USING HNSW ((embedding::vector({dimensions})) vector_cosine_ops) WHERE stale = FALSE AND current_for_retrieval = TRUE"))
            .opaque(format!("CREATE INDEX pravah_memories_archive_embedding_idx ON pravah_memories USING HNSW ((embedding::vector({dimensions})) vector_cosine_ops) WHERE stale = FALSE"))
            .opaque("CREATE INDEX pravah_entities_aliases_idx ON pravah_entities USING GIN (aliases)")
    }
}

fn profile_row(profile: MemoryProfile) -> MemoryProfileRow {
    MemoryProfileRow {
        id: 1,
        schema_version: profile.schema_version,
        embedding_model: profile.embedding_model,
        embedding_revision: profile.embedding_revision,
        embedding_dimensions: profile.embedding_dimensions,
        document_format_revision: profile.document_format_revision,
        distance_metric: "cosine".to_owned(),
        text_search_configuration: profile.text_search_configuration,
        extractor_revision: profile.extractor_revision,
        reconciler_revision: profile.reconciler_revision,
        derived_index_revision: profile.derived_index_revision,
    }
}

fn evidence_schema(table: db::schema::TableBuilder) -> db::schema::TableBuilder {
    table
        .index(unique_index(
            "pravah_evidence_scope_key_key",
            &["user_key", "agent_key", "evidence_key"],
        ))
        .index(unique_index(
            "pravah_evidence_scope_id_key",
            &["user_key", "agent_key", "id"],
        ))
        .index(index(
            "pravah_evidence_processing_idx",
            &["processing_state", "processing_lease_until", "created_at"],
        ))
        .index(index(
            "pravah_evidence_scope_hash_idx",
            &["user_key", "agent_key", "content_hash"],
        ))
        .check(
            "pravah_evidence_processing_lease_pair_check",
            "(processing_token IS NULL) = (processing_lease_until IS NULL)",
        )
        .check(
            "pravah_evidence_processing_attempts_check",
            "processing_attempts >= 0",
        )
        .check(
            "pravah_evidence_terminal_lease_check",
            "(processing_state NOT IN ('ready') AND stale = FALSE) OR processing_token IS NULL",
        )
}

fn memory_schema(
    table: db::schema::TableBuilder,
    dimensions: i32,
    text_config: &str,
) -> db::schema::TableBuilder {
    let escaped_text_config = text_config.replace('\'', "''");
    let generated = format!("to_tsvector(CAST('{escaped_text_config}' AS regconfig), text)");
    table
        .column("search_vector", "tsvector", |column| {
            column
                .generated(generated)
                .generated_storage(db::schema::GeneratedStorage::Stored)
        })
        .index(unique_index(
            "pravah_memories_evidence_position_key",
            &["evidence_id", "position"],
        ))
        .index(unique_index(
            "pravah_memories_scope_id_key",
            &["user_key", "agent_key", "id"],
        ))
        .foreign_key_columns_with(
            &["user_key", "agent_key", "evidence_id"],
            "pravah_evidence",
            &["user_key", "agent_key", "id"],
            |foreign_key| foreign_key.on_delete("CASCADE"),
        )
        .check(
            "pravah_memories_embedding_dimensions_check",
            format!("vector_dims(embedding) = {dimensions}"),
        )
        .index(index(
            "pravah_memories_scope_hash_idx",
            &["user_key", "agent_key", "content_hash"],
        ))
        .index(index(
            "pravah_memories_temporal_idx",
            &["user_key", "agent_key", "valid_from", "valid_until"],
        ))
        .index(index(
            "pravah_memories_event_idx",
            &["user_key", "agent_key", "event_at"],
        ))
}

fn entity_schema(table: db::schema::TableBuilder) -> db::schema::TableBuilder {
    table
        .index(unique_index(
            "pravah_entities_scope_key_key",
            &["user_key", "agent_key", "entity_key"],
        ))
        .index(unique_index(
            "pravah_entities_scope_id_key",
            &["user_key", "agent_key", "id"],
        ))
}

fn memory_entity_schema(table: db::schema::TableBuilder) -> db::schema::TableBuilder {
    table
        .foreign_key_columns_with(
            &["user_key", "agent_key", "memory_id"],
            "pravah_memories",
            &["user_key", "agent_key", "id"],
            |foreign_key| foreign_key.on_delete("CASCADE"),
        )
        .foreign_key_columns_with(
            &["user_key", "agent_key", "entity_id"],
            "pravah_entities",
            &["user_key", "agent_key", "id"],
            |foreign_key| foreign_key.on_delete("CASCADE"),
        )
        .index(index(
            "pravah_memory_entities_reverse_idx",
            &["entity_id", "memory_id"],
        ))
}

fn memory_relation_schema(table: db::schema::TableBuilder) -> db::schema::TableBuilder {
    table
        .foreign_key_columns_with(
            &["user_key", "agent_key", "from_memory_id"],
            "pravah_memories",
            &["user_key", "agent_key", "id"],
            |foreign_key| foreign_key.on_delete("CASCADE"),
        )
        .foreign_key_columns_with(
            &["user_key", "agent_key", "origin_evidence_id"],
            "pravah_evidence",
            &["user_key", "agent_key", "id"],
            |foreign_key| foreign_key.on_delete("CASCADE"),
        )
        .foreign_key_columns_with(
            &["user_key", "agent_key", "to_memory_id"],
            "pravah_memories",
            &["user_key", "agent_key", "id"],
            |foreign_key| foreign_key.on_delete("CASCADE"),
        )
        .check(
            "pravah_memory_relations_no_self_check",
            "from_memory_id <> to_memory_id",
        )
        .check(
            "pravah_memory_relations_symmetric_order_check",
            "kind = 'supersedes' OR from_memory_id < to_memory_id",
        )
        .index(index(
            "pravah_memory_relations_to_idx",
            &["to_memory_id", "kind", "from_memory_id"],
        ))
        .index(index(
            "pravah_memory_relations_origin_idx",
            &["origin_evidence_id"],
        ))
        .index(index(
            "pravah_memory_relations_effective_idx",
            &["effective_at"],
        ))
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

    /// Verifies one schema extension owns exactly the fixed seven-table family.
    #[test]
    fn memory_schema_registers_fixed_table_family() {
        let profile = MemoryProfile::new(
            "model",
            "r1",
            3,
            "document-v1",
            "extract-v1",
            "reconcile-v1",
        );
        let schema = db::schema().with_memory(profile).build().unwrap();

        assert_eq!(schema.tables.len(), 7);
        assert!(schema.tables.contains_key("pravah_evidence"));
        assert!(schema.tables.contains_key("pravah_memory_relations"));
        assert!(schema.extensions.contains_key("vector"));
    }

    /// Verifies generated text search and specialized current indexes remain migration-owned.
    #[test]
    fn memory_schema_registers_generated_and_opaque_indexes() {
        let profile = MemoryProfile::new(
            "model",
            "r1",
            1536,
            "document-v1",
            "extract-v1",
            "reconcile-v1",
        );
        let schema = db::schema().with_memory(profile).build().unwrap();
        let memories = &schema.tables["pravah_memories"];

        assert!(
            memories
                .columns
                .iter()
                .any(|column| column.name == "search_vector")
        );
        assert!(memories.indexes.iter().any(|index| index.name
            == "pravah_memories_current_embedding_idx"
            && index.is_opaque()));
    }

    /// Verifies the registered schema lowers to replayable PostgreSQL migration SQL.
    #[test]
    fn memory_schema_lowers_to_postgres_migration() {
        let profile = MemoryProfile::new(
            "model",
            "r1",
            1536,
            "document-v1",
            "extract-v1",
            "reconcile-v1",
        );
        let schema = db::schema().with_memory(profile).build().unwrap();
        let planner = db::gaman::core::OfflinePlanner::new(db::gaman::core::Dialect::Postgres);
        let pending = planner.make_migration(schema.clone(), &[]).unwrap_err();
        let db::gaman::core::OfflineError::NeedsInput(clarifications) = pending else {
            panic!("opaque indexes should require explicit migration acceptance");
        };
        let decisions = clarifications
            .into_iter()
            .map(|clarification| db::gaman::core::Decision {
                clarification_id: clarification.id,
                answer: db::gaman::core::Answer::AcceptRisk,
            })
            .collect::<Vec<_>>();
        let migration = planner
            .make_migration(schema, &decisions)
            .unwrap()
            .expect("memory schema must create an initial migration");
        let sql = planner.sql_migrate(&[migration]).unwrap().join("\n");

        assert!(sql.contains("CREATE EXTENSION"));
        assert!(sql.contains("pravah_memories_current_embedding_idx"));
        assert!(sql.contains("ON DELETE CASCADE"));
        assert!(sql.contains("GENERATED ALWAYS AS"));
    }
}
