use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use mool as db;
use mool::Model;
use mool::backend::{IgnoreConflictsExt, RowLockExt};
use mool::types::Vector;
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

use super::models::{
    EntityRow, EvidenceFailurePatch, EvidenceProcessingPatch, EvidenceReadyPatch,
    EvidenceReconciliationPatch, EvidenceRow, EvidenceStalePatch, MemoryEntityRow,
    MemoryProfileRow, MemoryRelationRow, MemoryRow, MemoryScopeRow, MemoryStalePatch,
    ScopeLeasePatch, ScopeProjectionPatch, ScopeRevisionPatch,
};
use super::projection::{recompute_projection, recompute_projection_ids, relation_neighbourhood};
use crate::{
    EntityId, Evidence, ExtractedEntity, Memory, MemoryId, MemoryKind, ProcessingState,
    ReconciliationDecision, ReconciliationOutcome, ReconciliationState, TemporalMetadata,
    TemporalPrecision, TemporalState,
};

mod records;
pub(crate) use records::memory_from_row;
use records::*;

/// Prepared immutable claim and embedding awaiting one atomic persistence step.
pub(crate) struct PreparedMemory {
    pub id: MemoryId,
    pub position: u32,
    pub text: String,
    pub kind: MemoryKind,
    pub temporal: TemporalMetadata,
    pub embedding: crate::Embedding,
    pub entities: Vec<ExtractedEntity>,
    pub metadata: serde_json::Value,
}

/// Result of atomically attempting to own one evidence-processing attempt.
pub(crate) enum ProcessingStart {
    Acquired { evidence: EvidenceRow, token: Uuid },
    Ready(EvidenceRow),
    Stale(EvidenceRow),
    InProgress(EvidenceRow),
}

#[derive(Debug, Error)]
pub(crate) enum RepositoryError {
    #[error("database operation failed: {0}")]
    Database(#[from] db::DbError),
    #[error("stored memory data is invalid: {0}")]
    InvalidStoredData(String),
    #[error("evidence key is already bound to different content")]
    EvidenceKeyConflict,
    #[error("evidence was not found")]
    EvidenceNotFound,
    #[error("memory profile row is missing")]
    MissingProfile,
    #[error("scope revision changed during reconciliation")]
    ScopeRevisionChanged,
    #[error("evidence processing ownership changed")]
    ProcessingSuperseded,
    #[error("evidence became stale during processing")]
    EvidenceStale,
    #[error("memory scope is already being reconciled")]
    ReconciliationBusy,
    #[error("memory reconciliation ownership changed")]
    ReconciliationSuperseded,
    #[error("relation expansion exceeded its configured safety bound")]
    RelationExpansionLimit,
    #[error("projection rebuild contains {actual} claims, exceeding limit {limit}")]
    ProjectionRebuildLimit { actual: usize, limit: u32 },
}

#[derive(Clone)]
pub(crate) struct MemoryRepository {
    pool: db::DbPool,
}

impl MemoryRepository {
    /// Creates typed persistence over the application-owned pool.
    pub fn new(pool: db::DbPool) -> Self {
        Self { pool }
    }

    /// Loads the singleton schema/provider profile without modifying schema.
    pub async fn profile(&self) -> Result<MemoryProfileRow, RepositoryError> {
        let table = MemoryProfileRow::table();
        let mut pool = self.pool.clone();
        db::from(&table)
            .filter(table.id.eq(db::val(1_i16)))
            .first::<MemoryProfileRow>()
            .exec(&mut pool)
            .await?
            .ok_or(RepositoryError::MissingProfile)
    }

    /// Accepts immutable evidence or returns the existing idempotent row.
    pub async fn accept_evidence(
        &self,
        row: EvidenceRow,
    ) -> Result<(EvidenceRow, bool), RepositoryError> {
        let table = EvidenceRow::table();
        let mut pool = self.pool.clone();
        let rows = [row.clone()];
        let inserted = db::from(&table)
            .insert_many(&rows)
            .ignore_conflicts_on((&table.user_key, &table.agent_key, &table.evidence_key))
            .exec(&mut pool)
            .await?;
        let stored = self
            .evidence_row(&row.user_key, &row.agent_key, &row.evidence_key)
            .await?
            .ok_or(RepositoryError::EvidenceNotFound)?;
        if stored.content_hash != row.content_hash {
            return Err(RepositoryError::EvidenceKeyConflict);
        }
        Ok((stored, inserted == 1))
    }

    /// Loads one scoped evidence row by its application key.
    pub async fn evidence_row(
        &self,
        user_key: &str,
        agent_key: &str,
        evidence_key: &str,
    ) -> Result<Option<EvidenceRow>, RepositoryError> {
        let table = EvidenceRow::table();
        let mut pool = self.pool.clone();
        evidence_by_key_query(&table, user_key, agent_key, evidence_key)
            .exec(&mut pool)
            .await
            .map_err(Into::into)
    }

    /// Acquires one expiring processing attempt without duplicating active provider work.
    pub async fn try_begin_processing(
        &self,
        evidence: &EvidenceRow,
        token: Uuid,
        now: DateTime<Utc>,
        lease_until: DateTime<Utc>,
    ) -> Result<ProcessingStart, RepositoryError> {
        let table = EvidenceRow::table();
        let mut pool = self.pool.clone();
        let updated = db::from(&table)
            .filter(
                table
                    .id
                    .eq(db::val(evidence.id))
                    .and(table.stale.eq(db::val(false)))
                    .and(processing_available(&table, now)),
            )
            .update(&EvidenceProcessingPatch {
                processing_state: "processing".to_owned(),
                processing_token: Some(token),
                processing_lease_until: Some(lease_until),
                processing_attempts: evidence.processing_attempts.saturating_add(1),
                error_code: None,
            })
            .exec(&mut pool)
            .await?;
        let stored = evidence_by_id(&mut pool, evidence.id)
            .await?
            .ok_or(RepositoryError::EvidenceNotFound)?;
        Ok(classify_processing_start(stored, token, updated))
    }

    /// Records failure only while the caller still owns the processing attempt.
    pub async fn mark_failed(
        &self,
        id: Uuid,
        token: Uuid,
        error_code: String,
    ) -> Result<(), RepositoryError> {
        let table = EvidenceRow::table();
        let mut pool = self.pool.clone();
        db::from(&table)
            .filter(
                table
                    .id
                    .eq(db::val(id))
                    .and(table.processing_token.eq(db::val(Some(token))))
                    .and(table.processing_state.eq(db::val("processing".to_owned()))),
            )
            .update(&EvidenceFailurePatch {
                processing_state: "failed".to_owned(),
                processing_token: None,
                processing_lease_until: None,
                error_code: Some(error_code),
            })
            .exec(&mut pool)
            .await?;
        Ok(())
    }

    /// Atomically persists all claims, canonical entities, links, and scope revision.
    pub async fn persist_memories(
        &self,
        evidence: &EvidenceRow,
        processing_token: Uuid,
        memories: &[PreparedMemory],
        reconciliation_required: bool,
    ) -> Result<i64, RepositoryError> {
        let mut transaction = self.pool.begin().await?;
        let locked = lock_evidence(&mut transaction, evidence.id).await?;
        validate_processing_owner(&locked, processing_token, Utc::now())?;
        let existing = load_memories_for_evidence(&mut transaction, evidence.id).await?;
        let inserted = existing.is_empty() && !memories.is_empty();
        let advance = if inserted {
            increment_scope_revision(&mut transaction, evidence).await?
        } else {
            ScopeAdvance {
                revision: current_scope_revision(&mut transaction, evidence).await?,
                was_pending: false,
            }
        };
        if existing.is_empty() {
            insert_prepared(&mut transaction, evidence, memories).await?;
        }
        mark_ready(
            &mut transaction,
            evidence.id,
            processing_token,
            advance.revision,
            reconciliation_required && !memories.is_empty(),
        )
        .await?;
        if inserted && !advance.was_pending {
            mark_projection_current(&mut transaction, evidence, advance.revision).await?;
        }
        transaction.commit().await?;
        Ok(advance.revision)
    }

    /// Loads one scoped evidence item for the public lifecycle API.
    pub async fn evidence(
        &self,
        user_key: &str,
        agent_key: &str,
        evidence_key: &str,
    ) -> Result<Option<Evidence>, RepositoryError> {
        self.evidence_row(user_key, agent_key, evidence_key)
            .await?
            .map(evidence_from_row)
            .transpose()
    }

    /// Stales evidence and every directly derived memory in one transaction.
    pub async fn mark_stale(
        &self,
        user_key: &str,
        agent_key: &str,
        evidence_key: &str,
        max_projection_nodes: u32,
    ) -> Result<(), RepositoryError> {
        let evidence = self
            .evidence_row(user_key, agent_key, evidence_key)
            .await?
            .ok_or(RepositoryError::EvidenceNotFound)?;
        let mut transaction = self.pool.begin().await?;
        let evidence = lock_evidence(&mut transaction, evidence.id).await?;
        if evidence.stale {
            transaction.commit().await?;
            return Ok(());
        }
        let seed_ids = stale_evidence_rows(&mut transaction, evidence.id).await?;
        let advance = increment_scope_revision(&mut transaction, &evidence).await?;
        let current = recompute_projection(
            &mut transaction,
            user_key,
            agent_key,
            &seed_ids,
            max_projection_nodes,
        )
        .await?;
        if current && !advance.was_pending {
            mark_projection_current(&mut transaction, &evidence, advance.revision).await?;
        }
        transaction.commit().await?;
        Ok(())
    }

    /// Explicitly deletes evidence; database cascades remove derived rows.
    pub async fn delete_evidence(
        &self,
        user_key: &str,
        agent_key: &str,
        evidence_key: &str,
        max_projection_nodes: u32,
    ) -> Result<(), RepositoryError> {
        let evidence = self
            .evidence_row(user_key, agent_key, evidence_key)
            .await?
            .ok_or(RepositoryError::EvidenceNotFound)?;
        let table = EvidenceRow::table();
        let mut transaction = self.pool.begin().await?;
        let evidence = lock_evidence(&mut transaction, evidence.id).await?;
        let seed_ids = load_memories_for_evidence(&mut transaction, evidence.id)
            .await?
            .into_iter()
            .map(|memory| memory.id)
            .collect::<Vec<_>>();
        let neighbourhood = relation_neighbourhood(
            &mut transaction,
            user_key,
            agent_key,
            &seed_ids,
            max_projection_nodes.saturating_add(1),
        )
        .await?;
        let advance = increment_scope_revision(&mut transaction, &evidence).await?;
        let deleted = db::from(&table)
            .filter(table.id.eq(db::val(evidence.id)))
            .delete()
            .exec(&mut transaction)
            .await?;
        if deleted == 0 {
            return Err(RepositoryError::EvidenceNotFound);
        }
        if neighbourhood.len() <= max_projection_nodes as usize && !advance.was_pending {
            recompute_projection_ids(&mut transaction, &neighbourhood).await?;
            mark_projection_current(&mut transaction, &evidence, advance.revision).await?;
        }
        transaction.commit().await?;
        Ok(())
    }

    /// Loads pending evidence and its claims in stable order for one scope.
    pub async fn pending_claims(
        &self,
        user_key: &str,
        agent_key: &str,
        limit: u32,
    ) -> Result<Vec<(EvidenceRow, Vec<MemoryRow>)>, RepositoryError> {
        let evidence_table = EvidenceRow::table();
        let mut pool = self.pool.clone();
        let evidence = db::from(&evidence_table)
            .filter(
                evidence_table
                    .user_key
                    .eq(db::val(user_key.to_owned()))
                    .and(evidence_table.agent_key.eq(db::val(agent_key.to_owned())))
                    .and(evidence_table.stale.eq(db::val(false)))
                    .and(
                        evidence_table
                            .reconciliation_state
                            .in_values(["pending".to_owned(), "failed".to_owned()]),
                    ),
            )
            .sort(evidence_table.created_at.asc())
            .slice::<EvidenceRow>(0, limit as usize)
            .exec(&mut pool)
            .await?;
        load_pending_memories(&mut pool, evidence).await
    }

    /// Acquires exclusive expiring ownership of reconciliation for one scope.
    pub async fn try_acquire_reconciliation(
        &self,
        user_key: &str,
        agent_key: &str,
        token: Uuid,
        now: DateTime<Utc>,
        lease_until: DateTime<Utc>,
    ) -> Result<i64, RepositoryError> {
        let table = MemoryScopeRow::table();
        let mut pool = self.pool.clone();
        let available = table.reconciliation_token.is_null().or(table
            .reconciliation_lease_until
            .is_null()
            .or(table.reconciliation_lease_until.lte(db::val(Some(now)))));
        let updated = db::from(&table)
            .filter(scope_filter(&table, user_key, agent_key).and(available))
            .update(&ScopeLeasePatch {
                reconciliation_token: Some(token),
                reconciliation_lease_until: Some(lease_until),
            })
            .exec(&mut pool)
            .await?;
        if updated == 0 {
            return Err(RepositoryError::ReconciliationBusy);
        }
        let scope = scope_row(&mut pool, user_key, agent_key)
            .await?
            .ok_or_else(|| RepositoryError::InvalidStoredData("missing scope row".to_owned()))?;
        Ok(scope.scope_revision)
    }

    /// Releases a reconciliation lease without disturbing a newer owner.
    pub async fn release_reconciliation(
        &self,
        user_key: &str,
        agent_key: &str,
        token: Uuid,
    ) -> Result<(), RepositoryError> {
        let table = MemoryScopeRow::table();
        let mut pool = self.pool.clone();
        db::from(&table)
            .filter(
                scope_filter(&table, user_key, agent_key)
                    .and(table.reconciliation_token.eq(db::val(Some(token)))),
            )
            .update(&ScopeLeasePatch {
                reconciliation_token: None,
                reconciliation_lease_until: None,
            })
            .exec(&mut pool)
            .await?;
        Ok(())
    }

    /// Loads bounded current candidates excluding claims from the same evidence.
    pub async fn reconciliation_candidates(
        &self,
        evidence: &EvidenceRow,
        claims: &[MemoryRow],
        limit: u32,
        per_claim_limit: u32,
        max_vector_distance: f32,
        text_search_configuration: &str,
    ) -> Result<Vec<MemoryRow>, RepositoryError> {
        if claims.is_empty() {
            return Ok(Vec::new());
        }
        let ids = reconciliation_candidate_ids(
            &self.pool,
            evidence,
            claims,
            limit,
            per_claim_limit,
            max_vector_distance,
            text_search_configuration,
        )
        .await?;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let table = MemoryRow::table();
        let mut pool = self.pool.clone();
        let rows = db::from(&table)
            .filter(table.id.in_values(ids.clone()))
            .all::<MemoryRow>()
            .exec(&mut pool)
            .await?;
        let mut by_id = rows
            .into_iter()
            .map(|row| (row.id, row))
            .collect::<BTreeMap<_, _>>();
        Ok(ids.into_iter().filter_map(|id| by_id.remove(&id)).collect())
    }

    /// Reads the scope revision used as the optimistic reconciliation guard.
    pub async fn scope_revision(
        &self,
        user_key: &str,
        agent_key: &str,
    ) -> Result<i64, RepositoryError> {
        let table = MemoryScopeRow::table();
        let mut pool = self.pool.clone();
        let row = db::from(&table)
            .filter(scope_filter(&table, user_key, agent_key))
            .first::<MemoryScopeRow>()
            .exec(&mut pool)
            .await?;
        Ok(row.map_or(0, |scope| scope.scope_revision))
    }

    /// Counts immutable claims directly derived from one evidence row.
    pub async fn memory_count(&self, evidence_id: Uuid) -> Result<usize, RepositoryError> {
        let table = MemoryRow::table();
        let mut pool = self.pool.clone();
        let count = db::from(&table)
            .filter(table.evidence_id.eq(db::val(evidence_id)))
            .count()
            .exec(&mut pool)
            .await?;
        usize::try_from(count)
            .map_err(|_| RepositoryError::InvalidStoredData("negative memory count".to_owned()))
    }

    /// Records retryable reconciliation failure without affecting searchability.
    pub async fn mark_reconciliation_failed(
        &self,
        user_key: &str,
        agent_key: &str,
        evidence_id: Uuid,
        token: Uuid,
        error_code: String,
    ) -> Result<(), RepositoryError> {
        let table = EvidenceRow::table();
        let mut transaction = self.pool.begin().await?;
        let scope = lock_scope(&mut transaction, user_key, agent_key).await?;
        validate_reconciliation_owner(&scope, token, Utc::now())?;
        db::from(&table)
            .filter(table.id.eq(db::val(evidence_id)))
            .update(&EvidenceReconciliationPatch {
                reconciliation_state: "failed".to_owned(),
                error_code: Some(error_code),
            })
            .exec(&mut transaction)
            .await?;
        transaction.commit().await?;
        Ok(())
    }

    /// Commits validated derived relations only when the scope has not changed.
    pub async fn commit_reconciliation(
        &self,
        evidence: &EvidenceRow,
        reconciliation_token: Uuid,
        expected_revision: i64,
        decisions: &[ReconciliationDecision],
        revision: &str,
        max_projection_nodes: u32,
    ) -> Result<(), RepositoryError> {
        let mut transaction = self.pool.begin().await?;
        let scope = lock_scope(&mut transaction, &evidence.user_key, &evidence.agent_key).await?;
        validate_reconciliation_owner(&scope, reconciliation_token, Utc::now())?;
        if scope.scope_revision != expected_revision {
            return Err(RepositoryError::ScopeRevisionChanged);
        }
        let rows = relation_rows(evidence, decisions, revision)?;
        let seed_ids = replace_origin_relations(&mut transaction, evidence.id, &rows).await?;
        set_reconciliation_ready(&mut transaction, evidence.id).await?;
        let advance = advance_locked_scope_revision(&mut transaction, &scope).await?;
        let current = recompute_projection(
            &mut transaction,
            &evidence.user_key,
            &evidence.agent_key,
            &seed_ids,
            max_projection_nodes,
        )
        .await?;
        if current && !advance.was_pending {
            mark_projection_current(&mut transaction, evidence, advance.revision).await?;
        }
        transaction.commit().await?;
        Ok(())
    }

    /// Rebuilds a lagging scope projection under an existing reconciliation lease.
    pub async fn rebuild_projection(
        &self,
        user_key: &str,
        agent_key: &str,
        reconciliation_token: Uuid,
        claim_limit: u32,
    ) -> Result<usize, RepositoryError> {
        let mut transaction = self.pool.begin().await?;
        let scope = lock_scope(&mut transaction, user_key, agent_key).await?;
        validate_reconciliation_owner(&scope, reconciliation_token, Utc::now())?;
        let ids = scoped_memory_ids(&mut transaction, user_key, agent_key, claim_limit).await?;
        recompute_projection_ids(&mut transaction, &ids).await?;
        mark_scope_projection_current(&mut transaction, &scope).await?;
        transaction.commit().await?;
        Ok(ids.len())
    }

    pub(crate) fn pool(&self) -> db::DbPool {
        self.pool.clone()
    }
}

pub(crate) fn sha256(content: &str) -> Vec<u8> {
    Sha256::digest(content.as_bytes()).to_vec()
}

fn processing_available(
    table: &db::queries::ModelTable<EvidenceRow>,
    now: DateTime<Utc>,
) -> db::queries::Predicate {
    table
        .processing_state
        .in_values(["pending".to_owned(), "failed".to_owned()])
        .or(table
            .processing_state
            .eq(db::val("processing".to_owned()))
            .and(
                table
                    .processing_lease_until
                    .is_null()
                    .or(table.processing_lease_until.lte(db::val(Some(now)))),
            ))
}

fn classify_processing_start(stored: EvidenceRow, token: Uuid, updated: u64) -> ProcessingStart {
    if updated == 1 {
        ProcessingStart::Acquired {
            evidence: stored,
            token,
        }
    } else if stored.stale {
        ProcessingStart::Stale(stored)
    } else if stored.processing_state == "ready" {
        ProcessingStart::Ready(stored)
    } else {
        ProcessingStart::InProgress(stored)
    }
}

/// Hashes a case- and whitespace-normalized claim for deterministic duplicate detection.
fn normalized_sha256(content: &str) -> Vec<u8> {
    let normalized = content
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    sha256(&normalized)
}

/// Selects a bounded reconciliation set across entity, lexical, vector, and temporal channels.
/// Fuses bounded database-native candidate channels before semantic reconciliation.
async fn reconciliation_candidate_ids(
    pool: &db::DbPool,
    evidence: &EvidenceRow,
    claims: &[MemoryRow],
    limit: u32,
    per_claim_limit: u32,
    max_vector_distance: f32,
    text_search_configuration: &str,
) -> Result<Vec<Uuid>, RepositoryError> {
    let claim_ids = claims.iter().map(|claim| claim.id).collect::<Vec<_>>();
    let mut session = pool.clone();
    db::query(super::sql::RECONCILIATION_CANDIDATES)
        .bind("user_key", evidence.user_key.clone())
        .bind("agent_key", evidence.agent_key.clone())
        .bind("evidence_id", evidence.id)
        .bind("claim_ids", claim_ids)
        .bind("per_claim_limit", i64::from(per_claim_limit))
        .bind("max_vector_distance", max_vector_distance)
        .bind(
            "text_search_configuration",
            text_search_configuration.to_owned(),
        )
        .bind("limit", i64::from(limit))
        .all::<(Uuid,)>(&mut session)
        .await
        .map(|rows| rows.into_iter().map(|(id,)| id).collect())
        .map_err(Into::into)
}

/// Replaces one evidence's complete relation set and returns all affected endpoints.
async fn replace_origin_relations(
    session: &mut impl db::DbSession,
    evidence_id: Uuid,
    rows: &[MemoryRelationRow],
) -> Result<Vec<Uuid>, RepositoryError> {
    let table = MemoryRelationRow::table();
    let previous = db::from(&table)
        .filter(table.origin_evidence_id.eq(db::val(evidence_id)))
        .all::<MemoryRelationRow>()
        .exec(session)
        .await?;
    let seeds = rows
        .iter()
        .chain(&previous)
        .flat_map(|row| [row.from_memory_id, row.to_memory_id])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    db::from(&table)
        .filter(table.origin_evidence_id.eq(db::val(evidence_id)))
        .delete()
        .exec(session)
        .await?;
    if !rows.is_empty() {
        db::from(&table)
            .upsert_many(rows, (&table.from_memory_id, &table.to_memory_id))
            .exec(session)
            .await?;
    }
    Ok(seeds)
}

async fn load_memories_for_evidence(
    session: &mut impl db::DbSession,
    evidence_id: Uuid,
) -> Result<Vec<MemoryRow>, RepositoryError> {
    let table = MemoryRow::table();
    db::from(&table)
        .filter(table.evidence_id.eq(db::val(evidence_id)))
        .sort(table.position.asc())
        .all::<MemoryRow>()
        .exec(session)
        .await
        .map_err(Into::into)
}

/// Stales one evidence row and its direct claims while returning projection seeds.
async fn stale_evidence_rows(
    session: &mut impl db::DbSession,
    evidence_id: Uuid,
) -> Result<Vec<Uuid>, RepositoryError> {
    let memories = load_memories_for_evidence(session, evidence_id).await?;
    let evidence_table = EvidenceRow::table();
    db::from(&evidence_table)
        .filter(evidence_table.id.eq(db::val(evidence_id)))
        .update(&EvidenceStalePatch {
            stale: true,
            processing_token: None,
            processing_lease_until: None,
        })
        .exec(session)
        .await?;
    let memory_table = MemoryRow::table();
    db::from(&memory_table)
        .filter(memory_table.evidence_id.eq(db::val(evidence_id)))
        .update(&MemoryStalePatch {
            stale: true,
            current_for_retrieval: false,
        })
        .exec(session)
        .await?;
    Ok(memories.into_iter().map(|memory| memory.id).collect())
}

/// Batch-loads claims for pending evidence without one query per evidence item.
async fn load_pending_memories(
    session: &mut impl db::DbSession,
    evidence: Vec<EvidenceRow>,
) -> Result<Vec<(EvidenceRow, Vec<MemoryRow>)>, RepositoryError> {
    if evidence.is_empty() {
        return Ok(Vec::new());
    }
    let ids = evidence.iter().map(|row| row.id).collect::<Vec<_>>();
    let table = MemoryRow::table();
    let memories = db::from(&table)
        .filter(table.evidence_id.in_values(ids))
        .sort(table.evidence_id.asc())
        .sort(table.position.asc())
        .all::<MemoryRow>()
        .exec(session)
        .await?;
    let mut grouped = BTreeMap::<Uuid, Vec<MemoryRow>>::new();
    for memory in memories {
        grouped.entry(memory.evidence_id).or_default().push(memory);
    }
    Ok(evidence
        .into_iter()
        .map(|row| {
            let claims = grouped.remove(&row.id).unwrap_or_default();
            (row, claims)
        })
        .collect())
}

/// Revision transition retaining whether older projection work was already pending.
struct ScopeAdvance {
    revision: i64,
    was_pending: bool,
}

/// Creates and locks a scope before monotonically advancing its claim revision.
async fn increment_scope_revision(
    session: &mut impl db::DbSession,
    evidence: &EvidenceRow,
) -> Result<ScopeAdvance, RepositoryError> {
    let table = MemoryScopeRow::table();
    let initial = MemoryScopeRow {
        user_key: evidence.user_key.clone(),
        agent_key: evidence.agent_key.clone(),
        scope_revision: 0,
        projection_revision: 0,
        reconciliation_token: None,
        reconciliation_lease_until: None,
        projection_pending: false,
        projection_due_at: None,
    };
    let initial_rows = [initial];
    db::from(&table)
        .insert_many(&initial_rows)
        .ignore_conflicts_on((&table.user_key, &table.agent_key))
        .exec(session)
        .await?;
    let scope = lock_scope(session, &evidence.user_key, &evidence.agent_key).await?;
    advance_locked_scope_revision(session, &scope).await
}

/// Returns the current revision without advancing it during an idempotent retry.
async fn current_scope_revision(
    session: &mut impl db::DbSession,
    evidence: &EvidenceRow,
) -> Result<i64, RepositoryError> {
    let table = MemoryScopeRow::table();
    let row = db::from(&table)
        .filter(scope_filter(
            &table,
            &evidence.user_key,
            &evidence.agent_key,
        ))
        .first::<MemoryScopeRow>()
        .exec(session)
        .await?;
    Ok(row.map_or(0, |scope| scope.scope_revision))
}

/// Advances a previously locked scope and marks its derived projection behind.
async fn advance_locked_scope_revision(
    session: &mut impl db::DbSession,
    scope: &MemoryScopeRow,
) -> Result<ScopeAdvance, RepositoryError> {
    let revision = scope.scope_revision.saturating_add(1);
    let table = MemoryScopeRow::table();
    db::from(&table)
        .filter(scope_filter(&table, &scope.user_key, &scope.agent_key))
        .update(&ScopeRevisionPatch {
            scope_revision: revision,
            projection_pending: true,
            projection_due_at: Some(Utc::now()),
        })
        .exec(session)
        .await?;
    Ok(ScopeAdvance {
        revision,
        was_pending: scope.projection_pending,
    })
}

/// Locks one short-lived scope row inside a caller-owned transaction.
async fn lock_scope(
    session: &mut impl db::DbSession,
    user_key: &str,
    agent_key: &str,
) -> Result<MemoryScopeRow, RepositoryError> {
    let table = MemoryScopeRow::table();
    db::from(&table)
        .filter(scope_filter(&table, user_key, agent_key))
        .for_update()
        .first::<MemoryScopeRow>()
        .exec(session)
        .await?
        .ok_or_else(|| RepositoryError::InvalidStoredData("missing scope row".to_owned()))
}

/// Loads a scope row through the caller's session without acquiring a lock.
async fn scope_row(
    session: &mut impl db::DbSession,
    user_key: &str,
    agent_key: &str,
) -> Result<Option<MemoryScopeRow>, RepositoryError> {
    let table = MemoryScopeRow::table();
    db::from(&table)
        .filter(scope_filter(&table, user_key, agent_key))
        .first::<MemoryScopeRow>()
        .exec(session)
        .await
        .map_err(Into::into)
}

/// Loads a whole scope only when it fits the caller's explicit rebuild bound.
async fn scoped_memory_ids(
    session: &mut impl db::DbSession,
    user_key: &str,
    agent_key: &str,
    limit: u32,
) -> Result<Vec<Uuid>, RepositoryError> {
    let table = MemoryRow::table();
    let rows = db::from(&table)
        .filter(
            table
                .user_key
                .eq(db::val(user_key.to_owned()))
                .and(table.agent_key.eq(db::val(agent_key.to_owned()))),
        )
        .sort(table.id.asc())
        .slice::<MemoryRow>(0, limit.saturating_add(1) as usize)
        .exec(session)
        .await?;
    if rows.len() > limit as usize {
        return Err(RepositoryError::ProjectionRebuildLimit {
            actual: rows.len(),
            limit,
        });
    }
    Ok(rows.into_iter().map(|row| row.id).collect())
}

/// Locks one evidence row before validating a fenced lifecycle transition.
async fn lock_evidence(
    session: &mut impl db::DbSession,
    evidence_id: Uuid,
) -> Result<EvidenceRow, RepositoryError> {
    let table = EvidenceRow::table();
    db::from(&table)
        .filter(table.id.eq(db::val(evidence_id)))
        .for_update()
        .first::<EvidenceRow>()
        .exec(session)
        .await?
        .ok_or(RepositoryError::EvidenceNotFound)
}

/// Loads one evidence row by its internal identity through the supplied session.
async fn evidence_by_id(
    session: &mut impl db::DbSession,
    evidence_id: Uuid,
) -> Result<Option<EvidenceRow>, RepositoryError> {
    let table = EvidenceRow::table();
    db::from(&table)
        .filter(table.id.eq(db::val(evidence_id)))
        .first::<EvidenceRow>()
        .exec(session)
        .await
        .map_err(Into::into)
}

/// Rejects stale or expired processing work before any claim becomes visible.
fn validate_processing_owner(
    evidence: &EvidenceRow,
    token: Uuid,
    now: DateTime<Utc>,
) -> Result<(), RepositoryError> {
    if evidence.stale {
        return Err(RepositoryError::EvidenceStale);
    }
    let owns = evidence.processing_state == "processing"
        && evidence.processing_token == Some(token)
        && evidence
            .processing_lease_until
            .is_some_and(|lease_until| lease_until > now);
    if owns {
        Ok(())
    } else {
        Err(RepositoryError::ProcessingSuperseded)
    }
}

/// Rejects reconciliation results from an expired or replaced scope owner.
fn validate_reconciliation_owner(
    scope: &MemoryScopeRow,
    token: Uuid,
    now: DateTime<Utc>,
) -> Result<(), RepositoryError> {
    let owns = scope.reconciliation_token == Some(token)
        && scope
            .reconciliation_lease_until
            .is_some_and(|lease_until| lease_until > now);
    if owns {
        Ok(())
    } else {
        Err(RepositoryError::ReconciliationSuperseded)
    }
}

/// Marks the synchronous projection current for the completed scope revision.
async fn mark_projection_current(
    session: &mut impl db::DbSession,
    evidence: &EvidenceRow,
    revision: i64,
) -> Result<(), RepositoryError> {
    let table = MemoryScopeRow::table();
    db::from(&table)
        .filter(scope_filter(
            &table,
            &evidence.user_key,
            &evidence.agent_key,
        ))
        .update(&ScopeProjectionPatch {
            projection_revision: revision,
            projection_pending: false,
            projection_due_at: None,
        })
        .exec(session)
        .await?;
    Ok(())
}

/// Marks a locked scope's derived projection current at its claim revision.
async fn mark_scope_projection_current(
    session: &mut impl db::DbSession,
    scope: &MemoryScopeRow,
) -> Result<(), RepositoryError> {
    let table = MemoryScopeRow::table();
    db::from(&table)
        .filter(scope_filter(&table, &scope.user_key, &scope.agent_key))
        .update(&ScopeProjectionPatch {
            projection_revision: scope.scope_revision,
            projection_pending: false,
            projection_due_at: None,
        })
        .exec(session)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests;
