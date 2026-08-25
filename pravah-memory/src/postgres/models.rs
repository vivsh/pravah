use chrono::{DateTime, Utc};
use mool as db;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use super::vector::PgVector;

#[derive(Debug, Clone, db::Model)]
#[table(name = "pravah_evidence")]
pub(crate) struct EvidenceRow {
    #[column(primary_key)]
    pub id: Uuid,
    pub user_key: String,
    pub agent_key: String,
    pub evidence_key: String,
    pub content: String,
    pub content_hash: Vec<u8>,
    #[column(type = "jsonb")]
    pub metadata: JsonValue,
    pub observed_at: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub processed_at: Option<DateTime<Utc>>,
    pub processing_state: String,
    pub processing_token: Option<Uuid>,
    pub processing_lease_until: Option<DateTime<Utc>>,
    pub processing_attempts: i32,
    pub published_revision: Option<i64>,
    pub reconciliation_state: String,
    pub extractor_revision: String,
    pub reconciler_revision: Option<String>,
    pub error_code: Option<String>,
    pub stale: bool,
}

#[derive(Debug, Clone, db::Model)]
#[table(name = "pravah_memories")]
pub(crate) struct MemoryRow {
    #[column(primary_key)]
    pub id: Uuid,
    pub user_key: String,
    pub agent_key: String,
    pub evidence_id: Uuid,
    pub position: i32,
    pub text: String,
    pub content_hash: Vec<u8>,
    pub kind: String,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_until: Option<DateTime<Utc>>,
    pub event_at: Option<DateTime<Utc>>,
    pub temporal_precision: String,
    pub temporal_state: String,
    #[column(type = "vector")]
    pub embedding: PgVector,
    #[column(type = "jsonb")]
    pub metadata: JsonValue,
    pub created_at: DateTime<Utc>,
    pub stale: bool,
    pub current_for_retrieval: bool,
}

#[derive(Debug, Clone, db::Model)]
#[table(name = "pravah_entities")]
pub(crate) struct EntityRow {
    #[column(primary_key)]
    pub id: Uuid,
    pub user_key: String,
    pub agent_key: String,
    pub entity_key: String,
    pub kind: String,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    #[column(type = "jsonb")]
    pub metadata: JsonValue,
}

#[derive(Debug, Clone, db::Model)]
#[table(
    name = "pravah_memory_entities",
    primary_key(name = "pravah_memory_entities_pkey", columns = ["memory_id", "entity_id"])
)]
pub(crate) struct MemoryEntityRow {
    pub memory_id: Uuid,
    pub entity_id: Uuid,
    pub user_key: String,
    pub agent_key: String,
}

#[derive(Debug, Clone, db::Model)]
#[table(
    name = "pravah_memory_relations",
    primary_key(
        name = "pravah_memory_relations_pkey",
        columns = ["from_memory_id", "to_memory_id"]
    )
)]
pub(crate) struct MemoryRelationRow {
    pub from_memory_id: Uuid,
    pub to_memory_id: Uuid,
    pub user_key: String,
    pub agent_key: String,
    pub origin_evidence_id: Uuid,
    pub kind: String,
    pub effective_at: Option<DateTime<Utc>>,
    pub reconciler_revision: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, db::Model)]
#[table(
    name = "pravah_memory_scopes",
    primary_key(name = "pravah_memory_scopes_pkey", columns = ["user_key", "agent_key"])
)]
pub(crate) struct MemoryScopeRow {
    pub user_key: String,
    pub agent_key: String,
    pub scope_revision: i64,
    pub projection_revision: i64,
    pub reconciliation_token: Option<Uuid>,
    pub reconciliation_lease_until: Option<DateTime<Utc>>,
    pub projection_pending: bool,
    pub projection_due_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, db::Model, db::ManagedRecord)]
#[table(name = "pravah_memory_profile")]
pub(crate) struct MemoryProfileRow {
    #[column(primary_key)]
    pub id: i16,
    pub schema_version: i32,
    pub embedding_model: String,
    pub embedding_revision: String,
    pub embedding_dimensions: i32,
    pub document_format_revision: String,
    pub distance_metric: String,
    pub text_search_configuration: String,
    pub extractor_revision: String,
    pub reconciler_revision: String,
    pub derived_index_revision: i32,
}

#[derive(Debug, Clone, db::Record)]
#[table(name = "pravah_evidence")]
pub(crate) struct EvidenceProcessingPatch {
    pub processing_state: String,
    pub processing_token: Option<Uuid>,
    pub processing_lease_until: Option<DateTime<Utc>>,
    pub processing_attempts: i32,
    pub error_code: Option<String>,
}

#[derive(Debug, Clone, db::Record)]
#[table(name = "pravah_evidence")]
pub(crate) struct EvidenceFailurePatch {
    pub processing_state: String,
    pub processing_token: Option<Uuid>,
    pub processing_lease_until: Option<DateTime<Utc>>,
    pub error_code: Option<String>,
}

#[derive(Debug, Clone, db::Record)]
#[table(name = "pravah_evidence")]
pub(crate) struct EvidenceReadyPatch {
    pub processed_at: Option<DateTime<Utc>>,
    pub processing_state: String,
    pub processing_token: Option<Uuid>,
    pub processing_lease_until: Option<DateTime<Utc>>,
    pub published_revision: Option<i64>,
    pub reconciliation_state: String,
    pub error_code: Option<String>,
}

#[derive(Debug, Clone, db::Record)]
#[table(name = "pravah_evidence")]
pub(crate) struct EvidenceStalePatch {
    pub stale: bool,
    pub processing_token: Option<Uuid>,
    pub processing_lease_until: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, db::Record)]
#[table(name = "pravah_evidence")]
pub(crate) struct EvidenceReconciliationPatch {
    pub reconciliation_state: String,
    pub error_code: Option<String>,
}

#[derive(Debug, Clone, db::Record)]
#[table(name = "pravah_memories")]
pub(crate) struct MemoryStalePatch {
    pub stale: bool,
    pub current_for_retrieval: bool,
}

#[derive(Debug, Clone, db::Record)]
#[table(name = "pravah_memory_scopes")]
pub(crate) struct ScopeRevisionPatch {
    pub scope_revision: i64,
    pub projection_pending: bool,
    pub projection_due_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, db::Record)]
#[table(name = "pravah_memory_scopes")]
pub(crate) struct ScopeLeasePatch {
    pub reconciliation_token: Option<Uuid>,
    pub reconciliation_lease_until: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, db::Record)]
#[table(name = "pravah_memory_scopes")]
pub(crate) struct ScopeProjectionPatch {
    pub projection_revision: i64,
    pub projection_pending: bool,
    pub projection_due_at: Option<DateTime<Utc>>,
}
