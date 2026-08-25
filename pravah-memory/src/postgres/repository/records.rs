use super::*;

/// Persists claims, canonical entities, and links through one caller-owned transaction.
pub(super) async fn insert_prepared(
    session: &mut impl db::DbSession,
    evidence: &EvidenceRow,
    prepared: &[PreparedMemory],
) -> Result<(), RepositoryError> {
    let now = Utc::now();
    let memories = prepared
        .iter()
        .map(|memory| memory_row(evidence, memory, now))
        .collect::<Result<Vec<_>, _>>()?;
    if !memories.is_empty() {
        db::from(MemoryRow::table())
            .insert_many(&memories)
            .exec(session)
            .await?;
    }
    let entities = canonical_entities(session, evidence, prepared).await?;
    let links = entity_links(evidence, prepared, &entities);
    if !links.is_empty() {
        db::from(MemoryEntityRow::table())
            .insert_many(&links)
            .ignore_conflicts()
            .exec(session)
            .await?;
    }
    Ok(())
}

fn memory_row(
    evidence: &EvidenceRow,
    memory: &PreparedMemory,
    now: DateTime<Utc>,
) -> Result<MemoryRow, RepositoryError> {
    let embedding = PgVector::from_embedding(&memory.embedding);
    Ok(MemoryRow {
        id: memory.id.as_uuid(),
        user_key: evidence.user_key.clone(),
        agent_key: evidence.agent_key.clone(),
        evidence_id: evidence.id,
        position: memory.position as i32,
        text: memory.text.clone(),
        content_hash: normalized_sha256(&memory.text),
        kind: memory.kind.as_str().to_owned(),
        valid_from: memory.temporal.valid_from,
        valid_until: memory.temporal.valid_until,
        event_at: memory.temporal.event_at,
        temporal_precision: memory.temporal.precision.as_str().to_owned(),
        temporal_state: memory.temporal.state.as_str().to_owned(),
        embedding,
        metadata: memory.metadata.clone(),
        created_at: now,
        stale: false,
        current_for_retrieval: true,
    })
}

async fn canonical_entities(
    session: &mut impl db::DbSession,
    evidence: &EvidenceRow,
    prepared: &[PreparedMemory],
) -> Result<BTreeMap<String, Uuid>, RepositoryError> {
    let mut distinct = BTreeMap::<String, ExtractedEntity>::new();
    for entity in prepared.iter().flat_map(|memory| &memory.entities) {
        distinct
            .entry(entity.entity_key.clone())
            .or_insert_with(|| entity.clone());
    }
    insert_entities(session, evidence, distinct.values()).await?;
    load_entity_ids(session, evidence, distinct.keys().cloned().collect()).await
}

async fn insert_entities<'a>(
    session: &mut impl db::DbSession,
    evidence: &EvidenceRow,
    entities: impl Iterator<Item = &'a ExtractedEntity>,
) -> Result<(), RepositoryError> {
    let rows = entities
        .map(|entity| EntityRow {
            id: EntityId::new().as_uuid(),
            user_key: evidence.user_key.clone(),
            agent_key: evidence.agent_key.clone(),
            entity_key: entity.entity_key.clone(),
            kind: entity.kind.clone(),
            canonical_name: entity.canonical_name.clone(),
            aliases: entity.aliases.clone(),
            metadata: entity.metadata.clone(),
        })
        .collect::<Vec<_>>();
    if !rows.is_empty() {
        let table = EntityRow::table();
        db::from(&table)
            .upsert_many(
                &rows,
                (&table.user_key, &table.agent_key, &table.entity_key),
            )
            .update_only((
                &table.kind,
                &table.canonical_name,
                &table.aliases,
                &table.metadata,
            ))
            .exec(session)
            .await?;
    }
    Ok(())
}

async fn load_entity_ids(
    session: &mut impl db::DbSession,
    evidence: &EvidenceRow,
    keys: Vec<String>,
) -> Result<BTreeMap<String, Uuid>, RepositoryError> {
    if keys.is_empty() {
        return Ok(BTreeMap::new());
    }
    let table = EntityRow::table();
    let rows = db::from(&table)
        .filter(
            table
                .user_key
                .eq(db::val(evidence.user_key.clone()))
                .and(table.agent_key.eq(db::val(evidence.agent_key.clone())))
                .and(table.entity_key.in_values(keys)),
        )
        .all::<EntityRow>()
        .exec(session)
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| (row.entity_key, row.id))
        .collect())
}

fn entity_links(
    evidence: &EvidenceRow,
    prepared: &[PreparedMemory],
    ids: &BTreeMap<String, Uuid>,
) -> Vec<MemoryEntityRow> {
    let unique = prepared
        .iter()
        .flat_map(|memory| {
            memory.entities.iter().filter_map(|entity| {
                ids.get(&entity.entity_key)
                    .map(|entity_id| (memory.id.as_uuid(), *entity_id))
            })
        })
        .collect::<BTreeSet<_>>();
    unique
        .into_iter()
        .map(|(memory_id, entity_id)| MemoryEntityRow {
            memory_id,
            entity_id,
            user_key: evidence.user_key.clone(),
            agent_key: evidence.agent_key.clone(),
        })
        .collect()
}

pub(super) async fn mark_ready(
    session: &mut impl db::DbSession,
    evidence_id: Uuid,
    processing_token: Uuid,
    published_revision: i64,
    reconciliation_required: bool,
) -> Result<(), RepositoryError> {
    let table = EvidenceRow::table();
    let updated = db::from(&table)
        .filter(
            table
                .id
                .eq(db::val(evidence_id))
                .and(table.processing_token.eq(db::val(Some(processing_token)))),
        )
        .update(&EvidenceReadyPatch {
            processed_at: Some(Utc::now()),
            processing_state: "ready".to_owned(),
            processing_token: None,
            processing_lease_until: None,
            published_revision: Some(published_revision),
            reconciliation_state: if reconciliation_required {
                "pending"
            } else {
                "not_required"
            }
            .to_owned(),
            error_code: None,
        })
        .exec(session)
        .await?;
    if updated != 1 {
        return Err(RepositoryError::ProcessingSuperseded);
    }
    Ok(())
}

pub(super) async fn set_reconciliation_ready(
    session: &mut impl db::DbSession,
    evidence_id: Uuid,
) -> Result<(), RepositoryError> {
    let table = EvidenceRow::table();
    db::from(&table)
        .filter(table.id.eq(db::val(evidence_id)))
        .update(&EvidenceReconciliationPatch {
            reconciliation_state: "ready".to_owned(),
            error_code: None,
        })
        .exec(session)
        .await?;
    Ok(())
}

pub(super) fn relation_rows(
    evidence: &EvidenceRow,
    decisions: &[ReconciliationDecision],
    revision: &str,
) -> Result<Vec<MemoryRelationRow>, RepositoryError> {
    let now = Utc::now();
    decisions
        .iter()
        .filter_map(|decision| relation_row(evidence, decision, revision, now).transpose())
        .collect()
}

fn relation_row(
    evidence: &EvidenceRow,
    decision: &ReconciliationDecision,
    revision: &str,
    now: DateTime<Utc>,
) -> Result<Option<MemoryRelationRow>, RepositoryError> {
    let (kind, effective_at, symmetric) = match decision.outcome {
        ReconciliationOutcome::Independent => return Ok(None),
        ReconciliationOutcome::Corroborates => ("corroborates", None, true),
        ReconciliationOutcome::Supersedes { effective_at } => ("supersedes", effective_at, false),
        ReconciliationOutcome::Conflicts => ("conflicts", None, true),
    };
    let (from, to) = canonical_relation_ids(
        decision.from_memory_id.as_uuid(),
        decision.to_memory_id.as_uuid(),
        symmetric,
    )?;
    Ok(Some(MemoryRelationRow {
        from_memory_id: from,
        to_memory_id: to,
        user_key: evidence.user_key.clone(),
        agent_key: evidence.agent_key.clone(),
        origin_evidence_id: evidence.id,
        kind: kind.to_owned(),
        effective_at,
        reconciler_revision: revision.to_owned(),
        created_at: now,
    }))
}

pub(super) fn canonical_relation_ids(
    from: Uuid,
    to: Uuid,
    symmetric: bool,
) -> Result<(Uuid, Uuid), RepositoryError> {
    if from == to {
        return Err(RepositoryError::InvalidStoredData(
            "self relation".to_owned(),
        ));
    }
    if symmetric && from > to {
        Ok((to, from))
    } else {
        Ok((from, to))
    }
}

pub(super) fn scope_evidence(
    table: &db::queries::ModelTable<EvidenceRow>,
    user_key: &str,
    agent_key: &str,
    evidence_key: &str,
) -> db::queries::Predicate {
    table
        .user_key
        .eq(db::val(user_key.to_owned()))
        .and(table.agent_key.eq(db::val(agent_key.to_owned())))
        .and(table.evidence_key.eq(db::val(evidence_key.to_owned())))
}

pub(super) fn evidence_by_key_query(
    table: &db::queries::ModelTable<EvidenceRow>,
    user_key: &str,
    agent_key: &str,
    evidence_key: &str,
) -> db::queries::First<EvidenceRow> {
    db::from(table)
        .filter(scope_evidence(table, user_key, agent_key, evidence_key))
        .first::<EvidenceRow>()
}

pub(super) fn scope_filter(
    table: &db::queries::ModelTable<MemoryScopeRow>,
    user_key: &str,
    agent_key: &str,
) -> db::queries::Predicate {
    table
        .user_key
        .eq(db::val(user_key.to_owned()))
        .and(table.agent_key.eq(db::val(agent_key.to_owned())))
}

pub(super) fn evidence_from_row(row: EvidenceRow) -> Result<Evidence, RepositoryError> {
    Ok(Evidence {
        id: row.id.to_string().parse().map_err(invalid_uuid)?,
        user_key: row.user_key,
        agent_key: row.agent_key,
        evidence_key: row.evidence_key,
        content: row.content,
        metadata: row.metadata,
        observed_at: row.observed_at,
        created_at: row.created_at,
        stale: row.stale,
        processing: processing_state(&row.processing_state)?,
        reconciliation: reconciliation_state(&row.reconciliation_state)?,
    })
}

pub(crate) fn memory_from_row(row: MemoryRow) -> Result<Memory, RepositoryError> {
    Ok(Memory {
        id: row.id.to_string().parse().map_err(invalid_uuid)?,
        evidence_id: row.evidence_id.to_string().parse().map_err(invalid_uuid)?,
        user_key: row.user_key,
        agent_key: row.agent_key,
        position: row.position as u32,
        text: row.text,
        kind: memory_kind(&row.kind)?,
        temporal: TemporalMetadata {
            valid_from: row.valid_from,
            valid_until: row.valid_until,
            event_at: row.event_at,
            precision: temporal_precision(&row.temporal_precision)?,
            state: temporal_state(&row.temporal_state)?,
        },
        metadata: row.metadata,
        stale: row.stale,
        current_for_retrieval: row.current_for_retrieval,
    })
}

fn invalid_uuid(error: uuid::Error) -> RepositoryError {
    RepositoryError::InvalidStoredData(error.to_string())
}

fn processing_state(value: &str) -> Result<ProcessingState, RepositoryError> {
    match value {
        "pending" => Ok(ProcessingState::Pending),
        "processing" => Ok(ProcessingState::Processing),
        "ready" => Ok(ProcessingState::Ready),
        "failed" => Ok(ProcessingState::Failed),
        _ => Err(invalid_enum("processing state", value)),
    }
}

fn reconciliation_state(value: &str) -> Result<ReconciliationState, RepositoryError> {
    match value {
        "not_required" => Ok(ReconciliationState::NotRequired),
        "pending" => Ok(ReconciliationState::Pending),
        "processing" => Ok(ReconciliationState::Processing),
        "ready" => Ok(ReconciliationState::Ready),
        "failed" => Ok(ReconciliationState::Failed),
        _ => Err(invalid_enum("reconciliation state", value)),
    }
}

fn memory_kind(value: &str) -> Result<MemoryKind, RepositoryError> {
    match value {
        "fact" => Ok(MemoryKind::Fact),
        "event" => Ok(MemoryKind::Event),
        "state" => Ok(MemoryKind::State),
        "plan" => Ok(MemoryKind::Plan),
        "preference" => Ok(MemoryKind::Preference),
        "relationship" => Ok(MemoryKind::Relationship),
        _ => Err(invalid_enum("memory kind", value)),
    }
}

fn temporal_precision(value: &str) -> Result<TemporalPrecision, RepositoryError> {
    match value {
        "unknown" => Ok(TemporalPrecision::Unknown),
        "year" => Ok(TemporalPrecision::Year),
        "month" => Ok(TemporalPrecision::Month),
        "day" => Ok(TemporalPrecision::Day),
        "instant" => Ok(TemporalPrecision::Instant),
        _ => Err(invalid_enum("temporal precision", value)),
    }
}

fn temporal_state(value: &str) -> Result<TemporalState, RepositoryError> {
    match value {
        "unspecified" => Ok(TemporalState::Unspecified),
        "ongoing" => Ok(TemporalState::Ongoing),
        "completed" => Ok(TemporalState::Completed),
        _ => Err(invalid_enum("temporal state", value)),
    }
}

fn invalid_enum(name: &str, value: &str) -> RepositoryError {
    RepositoryError::InvalidStoredData(format!("unknown {name}: {value}"))
}
