use std::borrow::Cow;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::config::AgentConfig;
use super::{AgentLoopMetrics, AgentPayload, GraphError};

/// Checkpointed invocation-local agent and tool budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AgentBudgetState {
    turn_limit: Option<u32>,
    tools: Vec<ToolBudgetState>,
}

/// Remaining calls for one configured tool in prepared order.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolBudgetState {
    name: String,
    limit: u32,
    remaining: u32,
}

impl AgentBudgetState {
    /// Resolves configured budgets for tools selected by the initial filter.
    pub(super) fn resolve(config: &AgentConfig, selected: &[String]) -> Option<Self> {
        let tools = selected
            .iter()
            .filter_map(|name| tool_budget(config, name))
            .collect::<Vec<_>>();
        if config.turn_budget.is_none() && tools.is_empty() {
            None
        } else {
            Some(Self {
                turn_limit: config.turn_budget,
                tools,
            })
        }
    }

    /// Returns ordinary model turns left before forced conclusion.
    pub(super) fn turns_remaining(&self, completed: u64) -> Option<u32> {
        self.turn_limit.map(|limit| {
            let remaining = u64::from(limit).saturating_sub(completed);
            u32::try_from(remaining).unwrap_or_default()
        })
    }

    /// Returns calls left for a specifically budgeted tool.
    pub(super) fn calls_remaining(&self, name: &str) -> Option<u32> {
        self.tools
            .iter()
            .find(|tool| tool.name == name)
            .map(|tool| tool.remaining)
    }

    /// Reserves one accepted attempt when the tool has remaining capacity.
    pub(super) fn admit(&mut self, name: &str) -> bool {
        let Some(tool) = self.tools.iter_mut().find(|tool| tool.name == name) else {
            return true;
        };
        if tool.remaining == 0 {
            return false;
        }
        tool.remaining -= 1;
        true
    }

    /// Returns whether a tool is currently available under its budget.
    fn allows(&self, name: &str) -> bool {
        self.calls_remaining(name)
            .is_none_or(|remaining| remaining > 0)
    }
}

/// Validates accumulated and graph-relative budget configuration errors.
pub(super) fn validate_budget_config(
    payload: &AgentPayload,
    config: &AgentConfig,
) -> Result<(), GraphError> {
    let mut errors = config.budget_errors.clone();
    for budget in &config.tool_budgets {
        if !payload.tools.iter().any(|tool| tool.name == budget.name) {
            errors.push(format!("unknown agent tool '{}'", budget.name));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(GraphError::AgentConfigValidation(errors.join("; ")))
    }
}

/// Produces the prepared-order intersection exposed to the model.
pub(super) fn effective_tools<'a>(
    configured: &'a [String],
    selected: &'a [String],
    budget: Option<&AgentBudgetState>,
) -> Cow<'a, [String]> {
    let Some(budget) = budget else {
        return Cow::Borrowed(selected);
    };
    Cow::Owned(
        configured
            .iter()
            .filter(|name| selected.contains(name))
            .filter(|name| budget.allows(name))
            .cloned()
            .collect(),
    )
}

/// Returns the remaining ordinary model turns for controller observation.
pub(super) fn turns_remaining(
    budget: Option<&AgentBudgetState>,
    metrics: &AgentLoopMetrics,
) -> Option<u32> {
    budget.and_then(|state| state.turns_remaining(metrics.model_turns()))
}

/// Returns whether another ordinary model dispatch is allowed.
pub(super) fn can_dispatch_normally(
    budget: Option<&AgentBudgetState>,
    metrics: &AgentLoopMetrics,
) -> bool {
    turns_remaining(budget, metrics).is_none_or(|remaining| remaining > 0)
}

/// Validates serialized budget identities, ordering, and remaining counts.
pub(super) fn validate_budget_state(
    configured: &[String],
    budget: Option<&AgentBudgetState>,
) -> Result<(), GraphError> {
    let Some(budget) = budget else {
        return Ok(());
    };
    validate_turn_limit(budget.turn_limit)?;
    validate_tool_budgets(configured, &budget.tools)?;
    if budget.turn_limit.is_none() && budget.tools.is_empty() {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint contains an empty budget state".into(),
        ));
    }
    Ok(())
}

fn tool_budget(config: &AgentConfig, name: &str) -> Option<ToolBudgetState> {
    config
        .tool_budgets
        .iter()
        .find(|budget| budget.name == name)
        .map(|budget| ToolBudgetState {
            name: name.to_owned(),
            limit: budget.limit,
            remaining: budget.limit,
        })
}

fn validate_turn_limit(limit: Option<u32>) -> Result<(), GraphError> {
    if limit == Some(0) {
        Err(GraphError::SnapshotValidation(
            "agent checkpoint turn budget is zero".into(),
        ))
    } else {
        Ok(())
    }
}

fn validate_tool_budgets(
    configured: &[String],
    budgets: &[ToolBudgetState],
) -> Result<(), GraphError> {
    let mut seen = BTreeSet::new();
    let mut last_position = None;
    for budget in budgets {
        validate_tool_budget(configured, budget, &mut seen, &mut last_position)?;
    }
    Ok(())
}

/// Validates one tool budget and advances the prepared-order cursor.
fn validate_tool_budget(
    configured: &[String],
    budget: &ToolBudgetState,
    seen: &mut BTreeSet<String>,
    last_position: &mut Option<usize>,
) -> Result<(), GraphError> {
    let position = configured.iter().position(|name| name == &budget.name);
    if budget.limit == 0 || budget.remaining > budget.limit || !seen.insert(budget.name.clone()) {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint tool budget is invalid or duplicated".into(),
        ));
    }
    let position = position.ok_or_else(|| {
        GraphError::SnapshotValidation("agent checkpoint budget names an unknown tool".into())
    })?;
    if last_position.is_some_and(|last| position <= last) {
        return Err(GraphError::SnapshotValidation(
            "agent checkpoint tool budgets are unordered".into(),
        ));
    }
    *last_position = Some(position);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;

    use crate::clients::Message;

    use super::*;

    /// Verifies absent budgets keep selection and resolution allocation-free.
    #[test]
    fn no_budget_helpers_allocate_nothing() {
        let config = AgentConfig::new("test:///model", "test", Message::user("test"));
        let configured = vec!["first".to_owned(), "second".to_owned()];
        let measured = allocation_counter::measure(|| {
            black_box(AgentBudgetState::resolve(&config, &configured));
            black_box(effective_tools(&configured, &configured, None));
        });

        assert_eq!(measured.count_total, 0);
        assert_eq!(measured.bytes_total, 0);
    }

    /// Verifies every malformed checkpoint budget relationship is rejected.
    #[test]
    fn serialized_budget_validation_rejects_malformed_relations() {
        let configured = vec!["first".to_owned(), "second".to_owned()];
        let valid = AgentBudgetState {
            turn_limit: Some(2),
            tools: vec![tool("first", 2, 1), tool("second", 3, 0)],
        };
        let cases = [
            malformed(&valid, |state| state.turn_limit = Some(0)),
            malformed(&valid, |state| state.tools[0].limit = 0),
            malformed(&valid, |state| state.tools[0].remaining = 3),
            malformed(&valid, |state| state.tools[1].name = "first".into()),
            malformed(&valid, |state| state.tools[0].name = "unknown".into()),
            malformed(&valid, |state| state.tools.reverse()),
            AgentBudgetState {
                turn_limit: None,
                tools: Vec::new(),
            },
        ];

        for state in cases {
            assert!(matches!(
                validate_budget_state(&configured, Some(&state)),
                Err(GraphError::SnapshotValidation(_))
            ));
        }
    }

    fn tool(name: &str, limit: u32, remaining: u32) -> ToolBudgetState {
        ToolBudgetState {
            name: name.to_owned(),
            limit,
            remaining,
        }
    }

    fn malformed(
        state: &AgentBudgetState,
        change: impl FnOnce(&mut AgentBudgetState),
    ) -> AgentBudgetState {
        let mut malformed = state.clone();
        change(&mut malformed);
        malformed
    }
}
