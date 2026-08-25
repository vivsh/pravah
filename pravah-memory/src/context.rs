//! Deterministic, backend-neutral assembly of retrieved claims into model context.
//!
//! Assembly preserves retrieval order and treats each unresolved conflict component
//! as an atomic unit. It never loads evidence, calls a provider, or truncates claim
//! text.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use chrono::SecondsFormat;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};
use thiserror::Error;

use super::{MemoryId, SearchResult, TemporalMetadata, TemporalPrecision, TemporalState};

/// Unit used to bound the complete rendered context document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextBudget {
    /// Maximum number of Unicode scalar values in the rendered document.
    Characters(usize),
    /// Maximum model-token count reported by the configured [`TokenCounter`].
    Tokens(usize),
}

impl ContextBudget {
    const fn limit(self) -> usize {
        match self {
            Self::Characters(limit) | Self::Tokens(limit) => limit,
        }
    }
}

/// Policy for unresolved conflict components found in retrieval results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictPolicy {
    /// Render every side together when the complete component fits.
    IncludeAll,
    /// Omit every unresolved conflict component.
    OmitGroups,
}

/// Provenance fields exposed by the default compact renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourcePolicy {
    /// Do not render provenance identifiers.
    None,
    /// Render only the application-supplied evidence key.
    EvidenceKey,
    /// Render the evidence key and immutable memory identifier.
    EvidenceAndMemoryId,
}

/// Claim metadata exposed by the default compact renderer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetadataPolicy {
    /// Do not expose extractor metadata.
    None,
    /// Expose only the selected top-level object keys.
    Keys(BTreeSet<String>),
    /// Expose the complete extractor metadata value.
    All,
}

/// Deterministic context-selection and rendering options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextOptions {
    /// Maximum size of the complete rendered document.
    pub budget: ContextBudget,
    /// Maximum number of retrieved claims included in the document.
    pub max_claims: usize,
    /// Handling for unresolved conflict components.
    pub conflicts: ConflictPolicy,
    /// Provenance fields rendered with each claim.
    pub sources: SourcePolicy,
    /// Extractor metadata rendered with each claim.
    pub metadata: MetadataPolicy,
    /// Whether available temporal annotations are rendered.
    pub include_temporal: bool,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            budget: ContextBudget::Characters(8_000),
            max_claims: 8,
            conflicts: ConflictPolicy::IncludeAll,
            sources: SourcePolicy::EvidenceKey,
            metadata: MetadataPolicy::None,
            include_temporal: true,
        }
    }
}

/// One atomic selection unit passed to a [`ContextRenderer`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextGroup {
    claims: Vec<SearchResult>,
    conflict: bool,
}

impl ContextGroup {
    /// Borrows claims in their original retrieval order.
    pub fn claims(&self) -> &[SearchResult] {
        &self.claims
    }

    /// Reports whether the group is an unresolved conflict component.
    pub const fn is_conflict(&self) -> bool {
        self.conflict
    }

    /// Returns immutable claim identifiers in retrieval order.
    pub fn memory_ids(&self) -> Vec<MemoryId> {
        self.claims.iter().map(|result| result.memory.id).collect()
    }
}

/// Reason a complete context group was omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextOmissionReason {
    /// The configured policy excludes unresolved conflicts.
    ConflictPolicy,
    /// Including the group would exceed the maximum claim count.
    ClaimLimit,
    /// Including the group would exceed the complete-document budget.
    Budget,
}

/// Structured record of one atomically omitted group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextOmission {
    /// Immutable identifiers omitted together.
    pub memory_ids: Vec<MemoryId>,
    /// Constraint or policy responsible for the omission.
    pub reason: ContextOmissionReason,
}

/// Successful deterministic context assembly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssembledContext {
    /// Complete groups selected in retrieval order.
    pub groups: Vec<ContextGroup>,
    /// Rendered context document.
    pub rendered: String,
    /// Selected memory identifiers in retrieval order.
    pub selected_memory_ids: Vec<MemoryId>,
    /// Omitted memory identifiers in retrieval order.
    pub omitted_memory_ids: Vec<MemoryId>,
    /// Atomic omissions with machine-readable reasons.
    pub omissions: Vec<ContextOmission>,
    /// Number of unresolved conflict components omitted for any reason.
    pub omitted_conflict_groups: usize,
    /// Exact character or token count of `rendered`.
    pub budget_used: usize,
    /// Whether any retrieved claim was omitted.
    pub truncated: bool,
}

/// Model-specific token measurement used only for token-budget assembly.
pub trait TokenCounter: Send + Sync {
    /// Counts tokens in a complete rendered context document.
    fn count_tokens(&self, text: &str) -> Result<usize, ContextError>;
}

/// Pluggable deterministic representation of selected context groups.
pub trait ContextRenderer: Send + Sync {
    /// Renders one complete atomic group without truncating claim text.
    fn render_group(
        &self,
        group: &ContextGroup,
        options: &ContextOptions,
    ) -> Result<String, ContextError>;

    /// Combines rendered groups into one complete context document.
    fn render_document(&self, groups: &[String]) -> Result<String, ContextError>;
}

/// Errors raised before any partial context can be returned.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// Retrieval returned the same immutable claim more than once.
    #[error("retrieval contains duplicate memory {0}")]
    DuplicateMemoryId(MemoryId),
    /// A result references a conflict counterpart absent from the supplied results.
    #[error("memory {memory_id} references missing conflict counterpart {conflict_id}")]
    IncompleteConflictGroup {
        /// Claim containing the invalid conflict annotation.
        memory_id: MemoryId,
        /// Referenced claim missing from the supplied results.
        conflict_id: MemoryId,
    },
    /// A result incorrectly identifies itself as a conflict counterpart.
    #[error("memory {0} cannot conflict with itself")]
    SelfConflict(MemoryId),
    /// A token budget was requested without a model-specific counter.
    #[error("token budgets require a configured token counter")]
    MissingTokenCounter,
    /// A custom renderer could not produce a complete group or document.
    #[error("context renderer failed: {0}")]
    Renderer(String),
    /// A configured token counter could not measure a complete document.
    #[error("token counter failed: {0}")]
    TokenCounter(String),
    /// Metadata selected for rendering could not be serialized.
    #[error("context metadata serialization failed: {0}")]
    MetadataSerialization(String),
    /// A renderer produced a final document exceeding the declared budget.
    #[error("rendered context uses {used} budget units but the limit is {limit}")]
    RenderedBudgetExceeded {
        /// Measured budget units in the rendered document.
        used: usize,
        /// Configured maximum budget units.
        limit: usize,
    },
}

/// Deterministic selector and renderer over already retrieved structured claims.
pub struct ContextAssembler {
    renderer: Box<dyn ContextRenderer>,
    token_counter: Option<Box<dyn TokenCounter>>,
}

impl ContextAssembler {
    /// Creates an assembler using the compact deterministic renderer.
    pub fn compact() -> Self {
        Self {
            renderer: Box::new(CompactRenderer),
            token_counter: None,
        }
    }

    /// Replaces the renderer while retaining deterministic group selection.
    pub fn with_renderer(mut self, renderer: impl ContextRenderer + 'static) -> Self {
        self.renderer = Box::new(renderer);
        self
    }

    /// Configures the model-specific counter required by token budgets.
    pub fn with_token_counter(mut self, counter: impl TokenCounter + 'static) -> Self {
        self.token_counter = Some(Box::new(counter));
        self
    }

    /// Assembles complete conflict-safe groups within the declared limits.
    pub fn assemble(
        &self,
        results: &[SearchResult],
        options: ContextOptions,
    ) -> Result<AssembledContext, ContextError> {
        self.require_budget_support(options.budget)?;
        let groups = build_groups(results)?;
        let rendered_groups = self.render_groups(&groups, &options)?;
        let selection = self.select_groups(groups, rendered_groups, &options)?;
        self.finish(selection, options.budget)
    }

    fn require_budget_support(&self, budget: ContextBudget) -> Result<(), ContextError> {
        if matches!(budget, ContextBudget::Tokens(_)) && self.token_counter.is_none() {
            Err(ContextError::MissingTokenCounter)
        } else {
            Ok(())
        }
    }

    fn render_groups(
        &self,
        groups: &[ContextGroup],
        options: &ContextOptions,
    ) -> Result<Vec<String>, ContextError> {
        groups
            .iter()
            .map(|group| self.renderer.render_group(group, options))
            .collect()
    }

    /// Applies policy, claim, and complete-document limits in retrieval order.
    fn select_groups(
        &self,
        groups: Vec<ContextGroup>,
        rendered_groups: Vec<String>,
        options: &ContextOptions,
    ) -> Result<Selection, ContextError> {
        let mut selection = Selection::default();
        for (group, rendered_group) in groups.into_iter().zip(rendered_groups) {
            let reason = self.omission_reason(&selection, &group, &rendered_group, options)?;
            if let Some(reason) = reason {
                selection.omit(group, reason);
            } else {
                selection.include(group, rendered_group);
            }
        }
        Ok(selection)
    }

    /// Chooses the first policy or bound that excludes an otherwise valid group.
    fn omission_reason(
        &self,
        selection: &Selection,
        group: &ContextGroup,
        rendered_group: &str,
        options: &ContextOptions,
    ) -> Result<Option<ContextOmissionReason>, ContextError> {
        if group.is_conflict() && options.conflicts == ConflictPolicy::OmitGroups {
            return Ok(Some(ContextOmissionReason::ConflictPolicy));
        }
        if selection.claim_count + group.claims().len() > options.max_claims {
            return Ok(Some(ContextOmissionReason::ClaimLimit));
        }
        let document = selection.document_with(self.renderer.as_ref(), rendered_group)?;
        let used = self.measure(&document, options.budget)?;
        Ok((used > options.budget.limit()).then_some(ContextOmissionReason::Budget))
    }

    /// Produces and remeasures the complete document before exposing any result.
    fn finish(
        &self,
        selection: Selection,
        budget: ContextBudget,
    ) -> Result<AssembledContext, ContextError> {
        let rendered = self.renderer.render_document(&selection.rendered_groups)?;
        let budget_used = self.measure(&rendered, budget)?;
        if budget_used > budget.limit() {
            return Err(ContextError::RenderedBudgetExceeded {
                used: budget_used,
                limit: budget.limit(),
            });
        }
        Ok(selection.into_context(rendered, budget_used))
    }

    fn measure(&self, text: &str, budget: ContextBudget) -> Result<usize, ContextError> {
        match budget {
            ContextBudget::Characters(_) => Ok(text.chars().count()),
            ContextBudget::Tokens(_) => self
                .token_counter
                .as_ref()
                .ok_or(ContextError::MissingTokenCounter)?
                .count_tokens(text),
        }
    }
}

impl Default for ContextAssembler {
    fn default() -> Self {
        Self::compact()
    }
}

#[derive(Default)]
struct Selection {
    groups: Vec<ContextGroup>,
    rendered_groups: Vec<String>,
    omitted_memory_ids: Vec<MemoryId>,
    omissions: Vec<ContextOmission>,
    omitted_conflict_groups: usize,
    claim_count: usize,
}

impl Selection {
    fn include(&mut self, group: ContextGroup, rendered: String) {
        self.claim_count += group.claims().len();
        self.groups.push(group);
        self.rendered_groups.push(rendered);
    }

    fn omit(&mut self, group: ContextGroup, reason: ContextOmissionReason) {
        let memory_ids = group.memory_ids();
        self.omitted_conflict_groups += usize::from(group.is_conflict());
        self.omitted_memory_ids.extend(memory_ids.iter().copied());
        self.omissions.push(ContextOmission { memory_ids, reason });
    }

    fn document_with(
        &self,
        renderer: &dyn ContextRenderer,
        rendered_group: &str,
    ) -> Result<String, ContextError> {
        let mut groups = self.rendered_groups.clone();
        groups.push(rendered_group.to_owned());
        renderer.render_document(&groups)
    }

    /// Converts internal selection state into the stable public result shape.
    fn into_context(self, rendered: String, budget_used: usize) -> AssembledContext {
        let selected_memory_ids = self
            .groups
            .iter()
            .flat_map(ContextGroup::memory_ids)
            .collect();
        AssembledContext {
            groups: self.groups,
            rendered,
            selected_memory_ids,
            truncated: !self.omitted_memory_ids.is_empty(),
            omitted_memory_ids: self.omitted_memory_ids,
            omissions: self.omissions,
            omitted_conflict_groups: self.omitted_conflict_groups,
            budget_used,
        }
    }
}

/// Validates result identities and constructs conflict-connected atomic groups.
fn build_groups(results: &[SearchResult]) -> Result<Vec<ContextGroup>, ContextError> {
    let positions = result_positions(results)?;
    let adjacency = conflict_adjacency(results, &positions)?;
    let mut visited = vec![false; results.len()];
    let mut groups = Vec::with_capacity(results.len());
    for start in 0..results.len() {
        if !visited[start] {
            groups.push(build_component(results, &adjacency, &mut visited, start));
        }
    }
    Ok(groups)
}

fn result_positions(results: &[SearchResult]) -> Result<BTreeMap<MemoryId, usize>, ContextError> {
    let mut positions = BTreeMap::new();
    for (position, result) in results.iter().enumerate() {
        if positions.insert(result.memory.id, position).is_some() {
            return Err(ContextError::DuplicateMemoryId(result.memory.id));
        }
    }
    Ok(positions)
}

/// Builds a deduplicated undirected conflict graph over validated results.
fn conflict_adjacency(
    results: &[SearchResult],
    positions: &BTreeMap<MemoryId, usize>,
) -> Result<Vec<Vec<usize>>, ContextError> {
    let mut adjacency = vec![Vec::new(); results.len()];
    for (position, result) in results.iter().enumerate() {
        add_conflict_edges(position, result, positions, &mut adjacency)?;
    }
    for neighbours in &mut adjacency {
        neighbours.sort_unstable();
        neighbours.dedup();
    }
    Ok(adjacency)
}

/// Adds undirected edges while rejecting incomplete or self-referential annotations.
fn add_conflict_edges(
    position: usize,
    result: &SearchResult,
    positions: &BTreeMap<MemoryId, usize>,
    adjacency: &mut [Vec<usize>],
) -> Result<(), ContextError> {
    for conflict_id in &result.conflicts {
        if *conflict_id == result.memory.id {
            return Err(ContextError::SelfConflict(result.memory.id));
        }
        let counterpart =
            positions
                .get(conflict_id)
                .copied()
                .ok_or(ContextError::IncompleteConflictGroup {
                    memory_id: result.memory.id,
                    conflict_id: *conflict_id,
                })?;
        adjacency[position].push(counterpart);
        adjacency[counterpart].push(position);
    }
    Ok(())
}

/// Traverses one conflict component and restores retrieval order within the group.
fn build_component(
    results: &[SearchResult],
    adjacency: &[Vec<usize>],
    visited: &mut [bool],
    start: usize,
) -> ContextGroup {
    let mut queue = VecDeque::from([start]);
    let mut positions = Vec::new();
    visited[start] = true;
    while let Some(position) = queue.pop_front() {
        positions.push(position);
        for neighbour in &adjacency[position] {
            if !visited[*neighbour] {
                visited[*neighbour] = true;
                queue.push_back(*neighbour);
            }
        }
    }
    positions.sort_unstable();
    ContextGroup {
        conflict: positions.len() > 1,
        claims: positions
            .into_iter()
            .map(|position| results[position].clone())
            .collect(),
    }
}

struct CompactRenderer;

impl ContextRenderer for CompactRenderer {
    fn render_group(
        &self,
        group: &ContextGroup,
        options: &ContextOptions,
    ) -> Result<String, ContextError> {
        let mut lines = Vec::with_capacity(group.claims().len() + 1);
        if group.is_conflict() {
            lines.push("Unresolved conflict:".to_owned());
        }
        for result in group.claims() {
            lines.push(render_claim(result, options)?);
        }
        Ok(lines.join("\n"))
    }

    fn render_document(&self, groups: &[String]) -> Result<String, ContextError> {
        Ok(groups.join("\n\n"))
    }
}

/// Renders one complete immutable claim with only explicitly selected annotations.
fn render_claim(result: &SearchResult, options: &ContextOptions) -> Result<String, ContextError> {
    let mut annotations = source_annotations(result, options.sources);
    annotations.push(format!("support={}", result.support_count));
    if options.include_temporal {
        annotations.extend(temporal_annotations(&result.memory.temporal));
    }
    if let Some(metadata) = selected_metadata(&result.memory.metadata, &options.metadata) {
        let rendered = serde_json::to_string(&metadata)
            .map_err(|error| ContextError::MetadataSerialization(error.to_string()))?;
        annotations.push(format!("metadata={rendered}"));
    }
    Ok(format!(
        "- {} [{}]",
        result.memory.text,
        annotations.join("; ")
    ))
}

fn source_annotations(result: &SearchResult, policy: SourcePolicy) -> Vec<String> {
    match policy {
        SourcePolicy::None => Vec::new(),
        SourcePolicy::EvidenceKey => vec![format!("evidence={}", result.evidence_key)],
        SourcePolicy::EvidenceAndMemoryId => vec![
            format!("evidence={}", result.evidence_key),
            format!("memory={}", result.memory.id),
        ],
    }
}

/// Renders only temporal values that carry evidence-supported information.
fn temporal_annotations(temporal: &TemporalMetadata) -> Vec<String> {
    let mut values = Vec::new();
    add_time(&mut values, "valid_from", temporal.valid_from);
    add_time(&mut values, "valid_until", temporal.valid_until);
    add_time(&mut values, "event_at", temporal.event_at);
    if temporal.precision != TemporalPrecision::Unknown {
        values.push(format!("precision={}", precision_name(temporal.precision)));
    }
    if temporal.state != TemporalState::Unspecified {
        values.push(format!("state={}", state_name(temporal.state)));
    }
    values
}

fn add_time(values: &mut Vec<String>, name: &str, value: Option<chrono::DateTime<chrono::Utc>>) {
    if let Some(value) = value {
        values.push(format!(
            "{name}={}",
            value.to_rfc3339_opts(SecondsFormat::AutoSi, true)
        ));
    }
}

const fn precision_name(precision: TemporalPrecision) -> &'static str {
    match precision {
        TemporalPrecision::Unknown => "unknown",
        TemporalPrecision::Year => "year",
        TemporalPrecision::Month => "month",
        TemporalPrecision::Day => "day",
        TemporalPrecision::Instant => "instant",
    }
}

const fn state_name(state: TemporalState) -> &'static str {
    match state {
        TemporalState::Unspecified => "unspecified",
        TemporalState::Ongoing => "ongoing",
        TemporalState::Completed => "completed",
    }
}

fn selected_metadata(metadata: &JsonValue, policy: &MetadataPolicy) -> Option<JsonValue> {
    match policy {
        MetadataPolicy::None => None,
        MetadataPolicy::All => Some(metadata.clone()),
        MetadataPolicy::Keys(keys) => select_metadata_keys(metadata, keys),
    }
}

fn select_metadata_keys(metadata: &JsonValue, keys: &BTreeSet<String>) -> Option<JsonValue> {
    let object = metadata.as_object()?;
    let selected: JsonMap<String, JsonValue> = keys
        .iter()
        .filter_map(|key| object.get(key).cloned().map(|value| (key.clone(), value)))
        .collect();
    (!selected.is_empty()).then_some(JsonValue::Object(selected))
}

#[cfg(test)]
mod tests;
