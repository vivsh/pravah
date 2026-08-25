use std::collections::{BTreeMap, BTreeSet, VecDeque};

use mool as db;
use mool::Model;
use mool::types::Vector;
use uuid::Uuid;

use super::models::{EvidenceRow, MemoryRelationRow, MemoryRow};
use super::repository::{MemoryRepository, RepositoryError, memory_from_row};
use crate::{
    ClaimView, Embedding, MemoryId, SearchRequest, SearchResult, SearchTimeline, StalePolicy,
    ValidTime,
};

pub(crate) async fn hybrid_search(
    repository: &MemoryRepository,
    scope: (&str, &str),
    request: &SearchRequest,
    embedding: Option<&Embedding>,
    dimensions: i32,
    text_search_configuration: &str,
    max_relation_edges: u32,
) -> Result<Vec<SearchResult>, RepositoryError> {
    let (user_key, agent_key) = scope;
    let pool = repository.pool();
    let mut transaction = pool.begin().await?;
    db::query("SET LOCAL hnsw.iterative_scan = 'strict_order'")
        .execute(&mut transaction)
        .await?;
    let hot_projection =
        projection_is_current(&mut transaction, user_key, agent_key, &request.timeline).await?;
    let context = CandidateContext {
        user_key,
        agent_key,
        request,
        embedding,
        dimensions,
        hot_projection,
        text_search_configuration,
    };
    let channels = candidate_channels(&mut transaction, &context).await?;
    let ranked =
        reciprocal_rank_fusion(channels, request.reciprocal_rank_k, request.candidate_limit);
    let results = hydrate_results(
        &mut transaction,
        ranked,
        request.minimum_fused_score,
        &request.timeline,
        request.limit,
        scope,
        max_relation_edges,
    )
    .await?;
    transaction.commit().await?;
    Ok(results)
}

/// Executes all enabled bounded retrieval channels in one database round trip.
async fn candidate_channels(
    session: &mut impl db::DbSession,
    context: &CandidateContext<'_>,
) -> Result<Vec<(f64, Vec<Uuid>)>, RepositoryError> {
    let plan = CandidatePlan::new(
        context.request,
        context.embedding,
        context.dimensions,
        context.hot_projection,
    );
    if plan.parts.is_empty() {
        return Ok(Vec::new());
    }
    let query = plan.bind(
        db::query(&plan.sql()),
        context.user_key,
        context.agent_key,
        context.request,
        context.embedding,
        context.text_search_configuration,
    )?;
    let rows = bind_timeline(query, &context.request.timeline)
        .all::<(String, Uuid, i64)>(session)
        .await?;
    Ok(assemble_channels(rows, context.request))
}

struct CandidateContext<'a> {
    user_key: &'a str,
    agent_key: &'a str,
    request: &'a SearchRequest,
    embedding: Option<&'a Embedding>,
    dimensions: i32,
    hot_projection: bool,
    text_search_configuration: &'a str,
}

struct CandidatePlan {
    parts: Vec<String>,
    lexical: bool,
    vector: bool,
    entity: bool,
    reference: Option<chrono::DateTime<chrono::Utc>>,
}

impl CandidatePlan {
    /// Builds only enabled channels so skipped providers and SQL work stay skipped.
    fn new(
        request: &SearchRequest,
        embedding: Option<&Embedding>,
        dimensions: i32,
        hot_projection: bool,
    ) -> Self {
        let lexical = request.weights.lexical > 0.0;
        let vector = request.weights.vector > 0.0 && embedding.is_some();
        let entity = request.weights.entity > 0.0 && !request.entity_keys.is_empty();
        let reference = temporal_reference(request).filter(|_| request.weights.temporal > 0.0);
        let filter = candidate_filter(request, hot_projection);
        let mut parts = Vec::with_capacity(4);
        if lexical {
            parts.push(lexical_sql(&filter));
        }
        if vector {
            parts.push(vector_sql(&filter, dimensions));
        }
        if entity {
            parts.push(entity_sql(&filter));
        }
        if reference.is_some() {
            parts.push(temporal_sql(&filter));
        }
        Self {
            parts,
            lexical,
            vector,
            entity,
            reference,
        }
    }

    fn sql(&self) -> String {
        self.parts.join(" UNION ALL ")
    }

    /// Binds only values referenced by the dynamically assembled channel statement.
    fn bind(
        &self,
        mut query: db::RawQuery,
        user_key: &str,
        agent_key: &str,
        request: &SearchRequest,
        embedding: Option<&Embedding>,
        text_search_configuration: &str,
    ) -> Result<db::RawQuery, RepositoryError> {
        query = query
            .bind("user_key", user_key.to_owned())
            .bind("agent_key", agent_key.to_owned())
            .bind("limit", i64::from(request.candidate_limit));
        if self.lexical {
            query = query.bind("text", request.text.clone()).bind(
                "text_search_configuration",
                text_search_configuration.to_owned(),
            );
        }
        if self.vector
            && let Some(embedding) = embedding
        {
            let vector = Vector::try_from_vec(embedding.values().to_vec())
                .map_err(|error| RepositoryError::InvalidStoredData(error.to_string()))?;
            query = query.bind("embedding", vector);
        }
        if self.entity {
            query = query.bind("entity_keys", request.entity_keys.clone());
        }
        if let Some(reference) = self.reference {
            query = query.bind("reference_time", reference);
        }
        Ok(query)
    }
}

fn candidate_filter(request: &SearchRequest, hot_projection: bool) -> String {
    format!(
        "memory.user_key = :user_key AND memory.agent_key = :agent_key AND {} AND {}",
        stale_predicate(request.stale),
        timeline_predicate(&request.timeline, hot_projection),
    )
}

fn lexical_sql(filter: &str) -> String {
    format!(
        "(SELECT 'lexical', memory.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(memory.search_vector, query) DESC, memory.id)::bigint \
         FROM pravah_memories memory, plainto_tsquery(CAST(:text_search_configuration AS regconfig), :text) query \
         WHERE {filter} AND memory.search_vector @@ query \
         ORDER BY ts_rank_cd(memory.search_vector, query) DESC, memory.id LIMIT :limit)"
    )
}

fn vector_sql(filter: &str, dimensions: i32) -> String {
    format!(
        "(SELECT 'vector', memory.id, ROW_NUMBER() OVER (ORDER BY memory.embedding::vector({dimensions}) <=> CAST(:embedding AS vector({dimensions})), memory.id)::bigint \
         FROM pravah_memories memory WHERE {filter} \
         ORDER BY memory.embedding::vector({dimensions}) <=> CAST(:embedding AS vector({dimensions})), memory.id LIMIT :limit)"
    )
}

fn entity_sql(filter: &str) -> String {
    format!(
        "(SELECT 'entity', memory.id, ROW_NUMBER() OVER (ORDER BY COUNT(DISTINCT entity.entity_key) DESC, memory.id)::bigint \
         FROM pravah_memories memory JOIN pravah_memory_entities link ON link.memory_id = memory.id \
         JOIN pravah_entities entity ON entity.id = link.entity_id \
         WHERE {filter} AND (entity.entity_key = ANY(:entity_keys) OR entity.aliases && :entity_keys) \
         GROUP BY memory.id \
         ORDER BY COUNT(DISTINCT entity.entity_key) DESC, memory.id LIMIT :limit)"
    )
}

fn temporal_sql(filter: &str) -> String {
    format!(
        "(SELECT 'temporal', memory.id, ROW_NUMBER() OVER (ORDER BY ABS(EXTRACT(EPOCH FROM (COALESCE(memory.event_at, memory.valid_from, memory.valid_until) - :reference_time))), memory.id)::bigint \
         FROM pravah_memories memory WHERE {filter} \
         AND COALESCE(memory.event_at, memory.valid_from, memory.valid_until) IS NOT NULL \
         ORDER BY ABS(EXTRACT(EPOCH FROM (COALESCE(memory.event_at, memory.valid_from, memory.valid_until) - :reference_time))), memory.id LIMIT :limit)"
    )
}

fn temporal_reference(request: &SearchRequest) -> Option<chrono::DateTime<chrono::Utc>> {
    request
        .timeline
        .reference_time
        .or(match request.timeline.valid_time {
            ValidTime::At(at) | ValidTime::Before(at) | ValidTime::After(at) => Some(at),
            ValidTime::Between { start, .. } => Some(start),
            ValidTime::Current | ValidTime::Any => None,
        })
}

fn assemble_channels(
    rows: Vec<(String, Uuid, i64)>,
    request: &SearchRequest,
) -> Vec<(f64, Vec<Uuid>)> {
    let mut channels = BTreeMap::<String, Vec<(i64, Uuid)>>::new();
    for (channel, id, rank) in rows {
        channels.entry(channel).or_default().push((rank, id));
    }
    [
        ("lexical", request.weights.lexical),
        ("vector", request.weights.vector),
        ("entity", request.weights.entity),
        ("temporal", request.weights.temporal),
    ]
    .into_iter()
    .filter_map(|(name, weight)| channels.remove(name).map(|rows| (weight, ranked_ids(rows))))
    .collect()
}

fn ranked_ids(mut rows: Vec<(i64, Uuid)>) -> Vec<Uuid> {
    rows.sort_by_key(|(rank, id)| (*rank, *id));
    rows.into_iter().map(|(_, id)| id).collect()
}

fn bind_timeline(mut query: db::RawQuery, timeline: &SearchTimeline) -> db::RawQuery {
    query = match timeline.valid_time {
        ValidTime::Current => query.bind(
            "valid_at",
            timeline.reference_time.unwrap_or_else(chrono::Utc::now),
        ),
        ValidTime::At(at) => query.bind("valid_at", at),
        ValidTime::Between { start, end } => {
            query.bind("valid_start", start).bind("valid_end", end)
        }
        ValidTime::Before(at) | ValidTime::After(at) => query.bind("valid_at", at),
        ValidTime::Any => query,
    };
    if let Some(known_at) = timeline.known_at {
        query = query.bind("known_at", known_at);
    }
    query
}

fn stale_predicate(stale: StalePolicy) -> &'static str {
    match stale {
        StalePolicy::Exclude => "memory.stale = FALSE",
        StalePolicy::Include => "TRUE",
        StalePolicy::Only => "memory.stale = TRUE",
    }
}

fn timeline_predicate(timeline: &SearchTimeline, hot_projection: bool) -> String {
    let projection = if hot_projection && uses_hot_projection(timeline) {
        "memory.current_for_retrieval = TRUE"
    } else {
        "TRUE"
    };
    let valid = match timeline.valid_time {
        ValidTime::Current | ValidTime::At(_) => {
            "((memory.event_at IS NOT NULL AND memory.event_at <= :valid_at) OR \
             (memory.event_at IS NULL AND (memory.valid_from IS NULL OR memory.valid_from <= :valid_at) \
              AND (memory.valid_until IS NULL OR memory.valid_until > :valid_at)))"
        }
        ValidTime::Between { .. } => {
            "((memory.event_at >= :valid_start AND memory.event_at < :valid_end) OR \
             (memory.event_at IS NULL AND (memory.valid_from IS NOT NULL OR memory.valid_until IS NOT NULL) \
              AND COALESCE(memory.valid_from, '-infinity') < :valid_end \
              AND COALESCE(memory.valid_until, 'infinity') > :valid_start))"
        }
        ValidTime::Before(_) => {
            "COALESCE(memory.event_at, memory.valid_until, memory.valid_from) < :valid_at"
        }
        ValidTime::After(_) => {
            "COALESCE(memory.event_at, memory.valid_from, memory.valid_until) >= :valid_at"
        }
        ValidTime::Any => "TRUE",
    };
    let known = if timeline.known_at.is_some() {
        "memory.created_at <= :known_at"
    } else {
        "TRUE"
    };
    format!("({projection}) AND ({valid}) AND ({known})")
}

/// Uses the hot partial indexes only when the scope projection is current.
async fn projection_is_current(
    session: &mut impl db::DbSession,
    user_key: &str,
    agent_key: &str,
    timeline: &SearchTimeline,
) -> Result<bool, RepositoryError> {
    if !uses_hot_projection(timeline) {
        return Ok(false);
    }
    let rows = db::query(
        "SELECT projection_pending FROM pravah_memory_scopes \
         WHERE user_key = :user_key AND agent_key = :agent_key",
    )
    .bind("user_key", user_key.to_owned())
    .bind("agent_key", agent_key.to_owned())
    .all::<(bool,)>(session)
    .await?;
    Ok(rows.first().is_none_or(|(pending,)| !pending))
}

fn uses_hot_projection(timeline: &SearchTimeline) -> bool {
    timeline.view == ClaimView::Current
        && timeline.valid_time == ValidTime::Current
        && timeline.known_at.is_none()
}

/// Combines independent bounded candidate orderings without score calibration.
fn reciprocal_rank_fusion(
    channels: Vec<(f64, Vec<Uuid>)>,
    reciprocal_rank_k: u32,
    limit: u32,
) -> Vec<(Uuid, f64)> {
    let mut scores = BTreeMap::<Uuid, f64>::new();
    for (weight, channel) in channels {
        for (position, id) in channel.into_iter().enumerate() {
            let rank = position as u32 + 1;
            *scores.entry(id).or_default() += weight / f64::from(reciprocal_rank_k + rank);
        }
    }
    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    ranked.truncate(limit as usize);
    ranked
}

/// Batch-hydrates candidates, provenance, and relation neighborhoods without N+1 queries.
async fn hydrate_results(
    session: &mut impl db::DbSession,
    mut ranked: Vec<(Uuid, f64)>,
    minimum_relevance: f64,
    timeline: &SearchTimeline,
    result_limit: u32,
    scope: (&str, &str),
    max_relation_edges: u32,
) -> Result<Vec<SearchResult>, RepositoryError> {
    let ids = ranked.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut memories = load_memories(session, ids.clone()).await?;
    let relation_view = RelationView::from_timeline(timeline);
    let (user_key, agent_key) = scope;
    let relations = load_relations(
        session,
        user_key,
        agent_key,
        ids,
        &relation_view,
        max_relation_edges,
    )
    .await?;
    let related_ids = relations
        .iter()
        .flat_map(|row| [row.from_memory_id, row.to_memory_id])
        .filter(|id| !memories.contains_key(id))
        .collect::<BTreeSet<_>>();
    if !related_ids.is_empty() {
        memories.extend(load_memories(session, related_ids.into_iter().collect()).await?);
    }
    let resolution = resolve_relations(&relations, &memories, &relation_view);
    ranked = resolve_ranked(ranked, minimum_relevance, result_limit, &resolution);
    let evidence_ids = selected_evidence_ids(&ranked, &memories)?;
    let evidence = load_evidence_keys(session, evidence_ids.into_iter()).await?;
    let annotations = resolution.annotations();
    assemble_results(ranked, memories, evidence, annotations)
}

/// Resolves final provenance identities before the evidence projection query.
fn selected_evidence_ids(
    ranked: &[(Uuid, f64)],
    memories: &BTreeMap<Uuid, MemoryRow>,
) -> Result<BTreeSet<Uuid>, RepositoryError> {
    ranked
        .iter()
        .map(|(id, _)| {
            memories.get(id).map(|row| row.evidence_id).ok_or_else(|| {
                RepositoryError::InvalidStoredData("candidate memory disappeared".to_owned())
            })
        })
        .collect()
}

/// Converts fully hydrated rows into ordered public search results.
fn assemble_results(
    ranked: Vec<(Uuid, f64)>,
    memories: BTreeMap<Uuid, MemoryRow>,
    evidence: BTreeMap<Uuid, String>,
    annotations: BTreeMap<Uuid, (u32, Vec<MemoryId>)>,
) -> Result<Vec<SearchResult>, RepositoryError> {
    let mut results = Vec::with_capacity(ranked.len());
    for (id, score) in ranked {
        let row = memories.get(&id).ok_or_else(|| {
            RepositoryError::InvalidStoredData("candidate memory disappeared".to_owned())
        })?;
        let evidence_key = evidence.get(&row.evidence_id).cloned().ok_or_else(|| {
            RepositoryError::InvalidStoredData("candidate evidence disappeared".to_owned())
        })?;
        let (support_count, conflicts) = annotations.get(&id).cloned().unwrap_or((1, Vec::new()));
        results.push(SearchResult {
            memory: memory_from_row(row.clone())?,
            evidence_key,
            score,
            rerank_score: None,
            support_count,
            conflicts,
        });
    }
    Ok(results)
}

/// Adds complete loaded conflict components without applying the ordinary result limit.
fn expand_conflicts(
    ranked: Vec<(Uuid, f64)>,
    conflicts: &BTreeMap<Uuid, BTreeSet<Uuid>>,
) -> Vec<(Uuid, f64)> {
    let mut scores = ranked.into_iter().collect::<BTreeMap<_, _>>();
    let mut pending = scores.keys().copied().collect::<VecDeque<_>>();
    while let Some(id) = pending.pop_front() {
        let Some(score) = scores.get(&id).copied() else {
            continue;
        };
        for conflict in conflicts.get(&id).into_iter().flatten() {
            if scores
                .get(conflict)
                .is_none_or(|existing| score > *existing)
            {
                scores.insert(*conflict, score);
                pending.push_back(*conflict);
            }
        }
    }
    let mut expanded = scores.into_iter().collect::<Vec<_>>();
    expanded.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    expanded
}

async fn load_memories(
    session: &mut impl db::DbSession,
    ids: Vec<Uuid>,
) -> Result<BTreeMap<Uuid, MemoryRow>, RepositoryError> {
    let table = MemoryRow::table();
    let rows = db::from(&table)
        .filter(table.id.in_values(ids))
        .all::<MemoryRow>()
        .exec(session)
        .await?;
    Ok(rows.into_iter().map(|row| (row.id, row)).collect())
}

/// Batch-loads evidence provenance keys without fetching full evidence content.
async fn load_evidence_keys(
    session: &mut impl db::DbSession,
    ids: impl Iterator<Item = Uuid>,
) -> Result<BTreeMap<Uuid, String>, RepositoryError> {
    let ids = ids.collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
    let table = EvidenceRow::table();
    let rows = db::from(&table)
        .filter(table.id.in_values(ids))
        .all::<EvidenceRow>()
        .exec(session)
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| (row.id, row.evidence_key))
        .collect())
}

/// Recursively loads the complete bounded semantic relation view in one statement.
async fn load_relations(
    session: &mut impl db::DbSession,
    user_key: &str,
    agent_key: &str,
    ids: Vec<Uuid>,
    view: &RelationView,
    max_edges: u32,
) -> Result<Vec<MemoryRelationRow>, RepositoryError> {
    let relations = db::query(RELATION_NEIGHBOURHOOD_SQL)
        .bind("memory_ids", ids)
        .bind("user_key", user_key.to_owned())
        .bind("agent_key", agent_key.to_owned())
        .bind("known_at_bounded", view.known_at.is_some())
        .bind("known_at", view.known_at.unwrap_or_else(chrono::Utc::now))
        .bind("include_supersedes", view.supersession_at.is_some())
        .bind(
            "supersession_at",
            view.supersession_at.unwrap_or_else(chrono::Utc::now),
        )
        .bind("edge_limit", relation_query_limit(max_edges))
        .all::<MemoryRelationRow>(session)
        .await?;
    if relations.len() > max_edges as usize {
        return Err(RepositoryError::RelationExpansionLimit);
    }
    Ok(relations)
}

fn relation_query_limit(max_edges: u32) -> i64 {
    i64::from(max_edges) + 1
}

/// Applies score thresholds, complete corroboration collapse, and atomic conflicts.
fn resolve_ranked(
    ranked: Vec<(Uuid, f64)>,
    minimum_relevance: f64,
    result_limit: u32,
    resolution: &RelationResolution,
) -> Vec<(Uuid, f64)> {
    let mut selected = Vec::new();
    let mut seen = BTreeSet::new();
    for (id, score) in ranked {
        let representative = resolution.representative(id);
        if score >= minimum_relevance
            && !resolution.suppressed.contains(&representative)
            && seen.insert(representative)
        {
            selected.push((representative, score));
        }
    }
    selected.truncate(result_limit as usize);
    expand_conflicts(selected, &resolution.conflicts)
}

struct RelationResolution {
    representatives: BTreeMap<Uuid, Uuid>,
    support: BTreeMap<Uuid, u32>,
    suppressed: BTreeSet<Uuid>,
    conflicts: BTreeMap<Uuid, BTreeSet<Uuid>>,
}

impl RelationResolution {
    fn representative(&self, id: Uuid) -> Uuid {
        self.representatives.get(&id).copied().unwrap_or(id)
    }

    fn annotations(&self) -> BTreeMap<Uuid, (u32, Vec<MemoryId>)> {
        self.support
            .iter()
            .map(|(id, count)| {
                let conflicts = self
                    .conflicts
                    .get(id)
                    .into_iter()
                    .flatten()
                    .copied()
                    .map(MemoryId::from)
                    .collect();
                (*id, (*count, conflicts))
            })
            .collect()
    }
}

fn resolve_relations(
    relations: &[MemoryRelationRow],
    memories: &BTreeMap<Uuid, MemoryRow>,
    view: &RelationView,
) -> RelationResolution {
    let (representatives, support) = corroboration_components(relations, memories, view);
    let suppressed = superseded_components(relations, memories, view, &representatives);
    let conflicts = conflict_components(relations, memories, view, &representatives, &suppressed);
    RelationResolution {
        representatives,
        support,
        suppressed,
        conflicts,
    }
}

fn corroboration_components(
    relations: &[MemoryRelationRow],
    memories: &BTreeMap<Uuid, MemoryRow>,
    view: &RelationView,
) -> (BTreeMap<Uuid, Uuid>, BTreeMap<Uuid, u32>) {
    let mut parents = memories
        .keys()
        .map(|id| (*id, *id))
        .collect::<BTreeMap<_, _>>();
    for relation in relations {
        if relation.kind == "corroborates" && relation_is_active(relation, memories, view) {
            union_components(&mut parents, relation.from_memory_id, relation.to_memory_id);
        }
    }
    let representatives = parents
        .keys()
        .map(|id| (*id, component_root(&parents, *id)))
        .collect::<BTreeMap<_, _>>();
    let mut support = BTreeMap::<Uuid, u32>::new();
    for representative in representatives.values() {
        let count = support.entry(*representative).or_default();
        *count = count.saturating_add(1);
    }
    (representatives, support)
}

fn union_components(parents: &mut BTreeMap<Uuid, Uuid>, left: Uuid, right: Uuid) {
    let left_root = component_root(parents, left);
    let right_root = component_root(parents, right);
    if left_root == right_root {
        return;
    }
    let (representative, duplicate) = if left_root < right_root {
        (left_root, right_root)
    } else {
        (right_root, left_root)
    };
    parents.insert(duplicate, representative);
}

fn component_root(parents: &BTreeMap<Uuid, Uuid>, mut id: Uuid) -> Uuid {
    while let Some(parent) = parents.get(&id) {
        if *parent == id {
            break;
        }
        id = *parent;
    }
    id
}

fn superseded_components(
    relations: &[MemoryRelationRow],
    memories: &BTreeMap<Uuid, MemoryRow>,
    view: &RelationView,
    representatives: &BTreeMap<Uuid, Uuid>,
) -> BTreeSet<Uuid> {
    relations
        .iter()
        .filter(|relation| {
            relation.kind == "supersedes" && relation_is_active(relation, memories, view)
        })
        .filter_map(|relation| {
            let newer = representatives.get(&relation.from_memory_id)?;
            let older = representatives.get(&relation.to_memory_id)?;
            (newer != older).then_some(*older)
        })
        .collect()
}

fn conflict_components(
    relations: &[MemoryRelationRow],
    memories: &BTreeMap<Uuid, MemoryRow>,
    view: &RelationView,
    representatives: &BTreeMap<Uuid, Uuid>,
    suppressed: &BTreeSet<Uuid>,
) -> BTreeMap<Uuid, BTreeSet<Uuid>> {
    let mut conflicts = BTreeMap::<Uuid, BTreeSet<Uuid>>::new();
    for relation in relations {
        if relation.kind != "conflicts" || !relation_is_active(relation, memories, view) {
            continue;
        }
        let Some(left) = representatives.get(&relation.from_memory_id).copied() else {
            continue;
        };
        let Some(right) = representatives.get(&relation.to_memory_id).copied() else {
            continue;
        };
        if left != right && !suppressed.contains(&left) && !suppressed.contains(&right) {
            conflicts.entry(left).or_default().insert(right);
            conflicts.entry(right).or_default().insert(left);
        }
    }
    conflicts
}

fn relation_is_active(
    relation: &MemoryRelationRow,
    memories: &BTreeMap<Uuid, MemoryRow>,
    view: &RelationView,
) -> bool {
    let endpoints_current = [relation.from_memory_id, relation.to_memory_id]
        .iter()
        .all(|id| memories.get(id).is_some_and(|memory| !memory.stale));
    let known = view
        .known_at
        .is_none_or(|known_at| relation.created_at <= known_at);
    let temporally_active = relation.kind != "supersedes"
        || view.supersession_at.is_some_and(|at| {
            relation
                .effective_at
                .is_none_or(|effective| effective <= at)
        });
    endpoints_current && known && temporally_active
}

#[derive(Clone, Copy)]
struct RelationView {
    known_at: Option<chrono::DateTime<chrono::Utc>>,
    supersession_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl RelationView {
    fn from_timeline(timeline: &SearchTimeline) -> Self {
        let supersession_at = if timeline.view == ClaimView::AllVersions {
            None
        } else {
            match timeline.valid_time {
                ValidTime::Current => {
                    Some(timeline.reference_time.unwrap_or_else(chrono::Utc::now))
                }
                ValidTime::At(at) | ValidTime::Before(at) | ValidTime::After(at) => Some(at),
                ValidTime::Between { end, .. } => Some(end),
                ValidTime::Any => None,
            }
        };
        Self {
            known_at: timeline.known_at,
            supersession_at,
        }
    }
}

const RELATION_NEIGHBOURHOOD_SQL: &str = r#"
WITH RECURSIVE walk(
    row_kind, memory_id, crossed_conflict,
    from_memory_id, to_memory_id, user_key, agent_key, origin_evidence_id,
    kind, effective_at, reconciler_revision, created_at
) AS (
    SELECT 0::SMALLINT, seed.id, FALSE,
           NULL::UUID, NULL::UUID, NULL::TEXT, NULL::TEXT, NULL::UUID,
           NULL::TEXT, NULL::TIMESTAMPTZ, NULL::TEXT, NULL::TIMESTAMPTZ
    FROM UNNEST(CAST(:memory_ids AS UUID[])) AS seed(id)
    UNION
    SELECT emitted.row_kind, emitted.memory_id, emitted.crossed_conflict,
           emitted.from_memory_id, emitted.to_memory_id, emitted.user_key,
           emitted.agent_key, emitted.origin_evidence_id, emitted.kind,
           emitted.effective_at, emitted.reconciler_revision, emitted.created_at
    FROM walk
    JOIN pravah_memory_relations relation
      ON walk.row_kind = 0
     AND (relation.from_memory_id = walk.memory_id
          OR relation.to_memory_id = walk.memory_id)
    JOIN pravah_memories left_memory ON left_memory.id = relation.from_memory_id
    JOIN pravah_memories right_memory ON right_memory.id = relation.to_memory_id
    CROSS JOIN LATERAL (
        VALUES
            (1::SMALLINT, NULL::UUID, FALSE,
             relation.from_memory_id, relation.to_memory_id, relation.user_key,
             relation.agent_key, relation.origin_evidence_id, relation.kind,
             relation.effective_at, relation.reconciler_revision, relation.created_at),
            (0::SMALLINT,
             CASE WHEN relation.from_memory_id = walk.memory_id
                  THEN relation.to_memory_id ELSE relation.from_memory_id END,
             walk.crossed_conflict OR relation.kind = 'conflicts',
             NULL::UUID, NULL::UUID, NULL::TEXT, NULL::TEXT, NULL::UUID,
             NULL::TEXT, NULL::TIMESTAMPTZ, NULL::TEXT, NULL::TIMESTAMPTZ)
    ) AS emitted(
        row_kind, memory_id, crossed_conflict,
        from_memory_id, to_memory_id, user_key, agent_key, origin_evidence_id,
        kind, effective_at, reconciler_revision, created_at
    )
    WHERE relation.user_key = :user_key AND relation.agent_key = :agent_key
      AND left_memory.user_key = :user_key AND left_memory.agent_key = :agent_key
      AND right_memory.user_key = :user_key AND right_memory.agent_key = :agent_key
      AND NOT left_memory.stale AND NOT right_memory.stale
      AND (NOT :known_at_bounded OR relation.created_at <= :known_at)
      AND (
          relation.kind = 'corroborates'
          OR (
              relation.kind = 'supersedes' AND :include_supersedes
              AND (relation.effective_at IS NULL OR relation.effective_at <= :supersession_at)
          )
          OR (relation.kind = 'conflicts' AND NOT walk.crossed_conflict)
      )
)
SELECT from_memory_id, to_memory_id, user_key, agent_key, origin_evidence_id,
       kind, effective_at, reconciler_revision, created_at
FROM walk
WHERE row_kind = 1
LIMIT :edge_limit
"#;

#[cfg(test)]
mod live_tests;
#[cfg(test)]
mod tests;
