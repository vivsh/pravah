use chrono::{TimeZone, Utc};
use serde_json::json;

use super::*;
use crate::{EvidenceId, Memory, MemoryKind};

fn result(text: &str) -> SearchResult {
    SearchResult {
        memory: Memory {
            id: MemoryId::new(),
            evidence_id: EvidenceId::new(),
            user_key: "user".to_owned(),
            agent_key: "agent".to_owned(),
            position: 0,
            text: text.to_owned(),
            kind: MemoryKind::Fact,
            temporal: TemporalMetadata::default(),
            metadata: json!({}),
            stale: false,
            current_for_retrieval: true,
        },
        evidence_key: format!("evidence-{text}"),
        score: 1.0,
        rerank_score: None,
        support_count: 1,
        conflicts: Vec::new(),
    }
}

fn linked_conflicts(texts: &[&str]) -> Vec<SearchResult> {
    let mut results: Vec<_> = texts.iter().map(|text| result(text)).collect();
    for position in 0..results.len().saturating_sub(1) {
        let next = results[position + 1].memory.id;
        results[position].conflicts.push(next);
    }
    results
}

/// Verifies compact defaults expose provenance and support without arbitrary metadata.
#[test]
fn compact_defaults_render_safe_claim_fields() {
    let mut input = result("The user prefers tea");
    input.support_count = 3;
    input.memory.metadata = json!({"private": "excluded"});
    let context = ContextAssembler::compact()
        .assemble(&[input], ContextOptions::default())
        .expect("default context should assemble");

    assert!(
        context
            .rendered
            .contains("evidence=evidence-The user prefers tea")
    );
    assert!(context.rendered.contains("support=3"));
    assert!(!context.rendered.contains("private"));
    assert_eq!(context.budget_used, context.rendered.chars().count());
}

/// Verifies duplicate immutable claim identities are rejected before rendering.
#[test]
fn duplicate_results_are_rejected() {
    let input = result("duplicate");
    let error = ContextAssembler::compact()
        .assemble(&[input.clone(), input.clone()], ContextOptions::default())
        .expect_err("duplicate memories must fail");

    assert_eq!(error, ContextError::DuplicateMemoryId(input.memory.id));
}

/// Verifies a missing conflict side never produces a partial context document.
#[test]
fn missing_conflict_counterpart_is_rejected() {
    let mut input = result("one side");
    let missing = MemoryId::new();
    input.conflicts.push(missing);
    let error = ContextAssembler::compact()
        .assemble(&[input.clone()], ContextOptions::default())
        .expect_err("incomplete conflicts must fail");

    assert_eq!(
        error,
        ContextError::IncompleteConflictGroup {
            memory_id: input.memory.id,
            conflict_id: missing,
        }
    );
}

/// Verifies transitive conflict components remain one atomic selection unit.
#[test]
fn transitive_conflicts_form_one_group() {
    let input = linked_conflicts(&["one", "two", "three"]);
    let context = ContextAssembler::compact()
        .assemble(&input, ContextOptions::default())
        .expect("complete conflict chain should assemble");

    assert_eq!(context.groups.len(), 1);
    assert!(context.groups[0].is_conflict());
    assert_eq!(context.groups[0].claims().len(), 3);
}

/// Verifies a budget that cannot hold every conflict side omits the entire component.
#[test]
fn oversized_conflict_group_is_omitted_atomically() {
    let input = linked_conflicts(&["one", "two"]);
    let full = ContextAssembler::compact()
        .assemble(&input, ContextOptions::default())
        .expect("full group should assemble");
    let options = ContextOptions {
        budget: ContextBudget::Characters(full.budget_used - 1),
        ..ContextOptions::default()
    };
    let context = ContextAssembler::compact()
        .assemble(&input, options)
        .expect("oversized group should be omitted");

    assert!(context.groups.is_empty());
    assert_eq!(context.omitted_memory_ids.len(), 2);
    assert_eq!(context.omitted_conflict_groups, 1);
    assert!(context.rendered.is_empty());
}

/// Verifies character budgets count Unicode scalar values instead of UTF-8 bytes.
#[test]
fn character_budget_uses_unicode_scalars() {
    let input = result("茶");
    let full = ContextAssembler::compact()
        .assemble(std::slice::from_ref(&input), ContextOptions::default())
        .expect("unicode claim should assemble");
    let context = ContextAssembler::compact()
        .assemble(
            &[input],
            ContextOptions {
                budget: ContextBudget::Characters(full.rendered.chars().count()),
                ..ContextOptions::default()
            },
        )
        .expect("exact scalar budget should fit");

    assert_eq!(context.groups.len(), 1);
    assert!(context.rendered.len() > context.budget_used);
}

struct WordCounter;

impl TokenCounter for WordCounter {
    fn count_tokens(&self, text: &str) -> Result<usize, ContextError> {
        Ok(text.split_whitespace().count())
    }
}

struct FailingCounter;

impl TokenCounter for FailingCounter {
    fn count_tokens(&self, _text: &str) -> Result<usize, ContextError> {
        Err(ContextError::TokenCounter("intentional".to_owned()))
    }
}

/// Verifies token budgets require an explicitly configured model counter.
#[test]
fn token_budget_without_counter_is_rejected() {
    let error = ContextAssembler::compact()
        .assemble(
            &[result("claim")],
            ContextOptions {
                budget: ContextBudget::Tokens(10),
                ..ContextOptions::default()
            },
        )
        .expect_err("missing token counter must fail");

    assert_eq!(error, ContextError::MissingTokenCounter);
}

/// Verifies a supplied token counter measures the complete rendered document.
#[test]
fn token_budget_uses_supplied_counter() {
    let context = ContextAssembler::compact()
        .with_token_counter(WordCounter)
        .assemble(
            &[result("short claim")],
            ContextOptions {
                budget: ContextBudget::Tokens(20),
                ..ContextOptions::default()
            },
        )
        .expect("configured counter should measure context");

    assert_eq!(
        context.budget_used,
        context.rendered.split_whitespace().count()
    );
}

/// Verifies token-counter failure aborts without returning a partial context.
#[test]
fn token_counter_failure_is_propagated() {
    let error = ContextAssembler::compact()
        .with_token_counter(FailingCounter)
        .assemble(
            &[result("claim")],
            ContextOptions {
                budget: ContextBudget::Tokens(10),
                ..ContextOptions::default()
            },
        )
        .expect_err("counter failure must abort assembly");

    assert_eq!(error, ContextError::TokenCounter("intentional".to_owned()));
}

/// Verifies claim limits never split an unresolved conflict component.
#[test]
fn claim_limit_omits_complete_conflict_group() {
    let input = linked_conflicts(&["one", "two"]);
    let context = ContextAssembler::compact()
        .assemble(
            &input,
            ContextOptions {
                max_claims: 1,
                ..ContextOptions::default()
            },
        )
        .expect("claim limit should omit instead of fail");

    assert!(context.selected_memory_ids.is_empty());
    assert_eq!(
        context.omissions[0].reason,
        ContextOmissionReason::ClaimLimit
    );
}

/// Verifies explicit conflict omission is distinguishable from budget truncation.
#[test]
fn conflict_policy_records_structured_omission() {
    let input = linked_conflicts(&["one", "two"]);
    let context = ContextAssembler::compact()
        .assemble(
            &input,
            ContextOptions {
                conflicts: ConflictPolicy::OmitGroups,
                ..ContextOptions::default()
            },
        )
        .expect("policy omission should succeed");

    assert_eq!(
        context.omissions[0].reason,
        ContextOmissionReason::ConflictPolicy
    );
    assert!(context.truncated);
}

/// Verifies selected metadata keys and temporal fields render deterministically.
#[test]
fn selected_metadata_and_temporal_fields_are_rendered() {
    let mut input = result("dated claim");
    input.memory.metadata = json!({"b": 2, "a": 1, "excluded": 3});
    input.memory.temporal.valid_from = Utc.with_ymd_and_hms(2026, 8, 24, 0, 0, 0).single();
    input.memory.temporal.precision = TemporalPrecision::Day;
    input.memory.temporal.state = TemporalState::Ongoing;
    let keys = BTreeSet::from(["a".to_owned(), "b".to_owned()]);
    let context = ContextAssembler::compact()
        .assemble(
            &[input],
            ContextOptions {
                sources: SourcePolicy::EvidenceAndMemoryId,
                metadata: MetadataPolicy::Keys(keys),
                ..ContextOptions::default()
            },
        )
        .expect("selected annotations should render");

    assert!(context.rendered.contains("valid_from=2026-08-24T00:00:00Z"));
    assert!(context.rendered.contains("precision=day"));
    assert!(context.rendered.contains("state=ongoing"));
    assert!(context.rendered.contains("metadata={\"a\":1,\"b\":2}"));
    assert!(!context.rendered.contains("excluded"));
}

/// Verifies source and temporal exposure can be disabled without changing claim text.
#[test]
fn source_and_temporal_annotations_can_be_disabled() {
    let mut input = result("minimal claim");
    input.memory.temporal.state = TemporalState::Ongoing;
    let context = ContextAssembler::compact()
        .assemble(
            &[input],
            ContextOptions {
                sources: SourcePolicy::None,
                include_temporal: false,
                ..ContextOptions::default()
            },
        )
        .expect("minimal annotations should render");

    assert_eq!(context.rendered, "- minimal claim [support=1]");
}

/// Verifies every character budget either includes or omits a conflict component wholly.
#[test]
fn all_character_budgets_preserve_conflict_atomicity() {
    let input = linked_conflicts(&["alpha", "beta", "gamma"]);
    let full = ContextAssembler::compact()
        .assemble(&input, ContextOptions::default())
        .expect("full conflict should assemble");
    for limit in 0..=full.budget_used {
        let context = ContextAssembler::compact()
            .assemble(
                &input,
                ContextOptions {
                    budget: ContextBudget::Characters(limit),
                    ..ContextOptions::default()
                },
            )
            .expect("every bounded selection should succeed");
        assert!(context.budget_used <= limit);
        assert!(matches!(context.selected_memory_ids.len(), 0 | 3));
    }
}

struct SeparatorRenderer;

impl ContextRenderer for SeparatorRenderer {
    fn render_group(
        &self,
        group: &ContextGroup,
        _options: &ContextOptions,
    ) -> Result<String, ContextError> {
        Ok(group
            .claims()
            .iter()
            .map(|claim| claim.memory.text.as_str())
            .collect::<Vec<_>>()
            .join(" | "))
    }

    fn render_document(&self, groups: &[String]) -> Result<String, ContextError> {
        Ok(format!("BEGIN\n{}\nEND", groups.join("\n---\n")))
    }
}

/// Verifies custom document prefixes, suffixes, and separators count toward the exact budget.
#[test]
fn custom_document_separators_are_budgeted_exactly() {
    let input = [result("first"), result("second")];
    let full = ContextAssembler::compact()
        .with_renderer(SeparatorRenderer)
        .assemble(&input, ContextOptions::default())
        .expect("custom document should assemble");
    let context = ContextAssembler::compact()
        .with_renderer(SeparatorRenderer)
        .assemble(
            &input,
            ContextOptions {
                budget: ContextBudget::Characters(full.budget_used - 1),
                ..ContextOptions::default()
            },
        )
        .expect("second group should be omitted when its separator does not fit");

    assert_eq!(full.budget_used, full.rendered.chars().count());
    assert_eq!(context.selected_memory_ids.len(), 1);
    assert_eq!(context.omitted_memory_ids.len(), 1);
    assert!(context.budget_used < full.budget_used);
    assert!(context.rendered.starts_with("BEGIN\nfirst"));
    assert!(context.rendered.ends_with("\nEND"));
}

struct FailingRenderer;

impl ContextRenderer for FailingRenderer {
    fn render_group(
        &self,
        _group: &ContextGroup,
        _options: &ContextOptions,
    ) -> Result<String, ContextError> {
        Err(ContextError::Renderer("intentional".to_owned()))
    }

    fn render_document(&self, _groups: &[String]) -> Result<String, ContextError> {
        Ok(String::new())
    }
}

/// Verifies custom renderer errors return no partially assembled value.
#[test]
fn custom_renderer_failure_is_propagated() {
    let error = ContextAssembler::compact()
        .with_renderer(FailingRenderer)
        .assemble(&[result("claim")], ContextOptions::default())
        .expect_err("renderer failure must abort assembly");

    assert_eq!(error, ContextError::Renderer("intentional".to_owned()));
}
