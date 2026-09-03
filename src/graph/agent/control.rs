use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::clients::{Message, TokenUsage};

use super::budget::AgentBudgetState;
use super::{ToolFilter, ToolInfo};
use crate::graph::error::GraphError;
use crate::graph::value::Value;

/// Point in an agent loop at which an application controller may intervene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentInterventionPoint {
    /// Before the initial model dispatch or a dispatch following a redirect.
    BeforeModel,
    /// After the model proposes tools but before any proposed tool executes.
    BeforeTools,
    /// After every accepted tool call has produced a result.
    AfterTools,
}

/// One tool call proposed by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolProposal {
    call_id: String,
    tool_name: String,
    arguments: Value,
}

impl AgentToolProposal {
    pub(crate) fn new(call_id: String, tool_name: String, arguments: Value) -> Self {
        Self {
            call_id,
            tool_name,
            arguments,
        }
    }

    /// Returns the provider-issued call identity.
    pub fn call_id(&self) -> &str {
        &self.call_id
    }

    /// Returns the requested model-facing tool name.
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Returns the canonical runtime value containing the tool arguments.
    pub fn arguments(&self) -> &Value {
        &self.arguments
    }
}

/// Result of one accepted tool call as observed by an agent controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolResult {
    call_id: String,
    tool_name: String,
    arguments: Value,
    value: Value,
    error: bool,
}

impl AgentToolResult {
    pub(crate) fn new(
        call_id: String,
        tool_name: String,
        arguments: Value,
        value: Value,
        error: bool,
    ) -> Self {
        Self {
            call_id,
            tool_name,
            arguments,
            value,
            error,
        }
    }

    /// Returns the provider-issued call identity.
    pub fn call_id(&self) -> &str {
        &self.call_id
    }

    /// Returns the requested model-facing tool name.
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Returns the canonical arguments supplied by the model.
    pub fn arguments(&self) -> &Value {
        &self.arguments
    }

    /// Returns the domain result or structured recoverable error value.
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Returns whether this is a recoverable tool failure.
    pub fn is_error(&self) -> bool {
        self.error
    }
}

/// Serializable invocation-local observations maintained by the agent runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentLoopMetrics {
    model_turns: u64,
    tool_rounds: u64,
    tool_calls: u64,
    recoverable_failures: u64,
    consecutive_tool_rounds: u64,
    repeated_proposals: u64,
    repeated_results: u64,
    input_tokens: u64,
    output_tokens: u64,
    calls_by_tool: BTreeMap<String, u64>,
    last_proposal_fingerprint: Option<String>,
    last_result_fingerprint: Option<String>,
}

impl AgentLoopMetrics {
    /// Returns model turns completed during this invocation.
    pub fn model_turns(&self) -> u64 {
        self.model_turns
    }

    /// Returns model turns that proposed at least one tool call.
    pub fn tool_rounds(&self) -> u64 {
        self.tool_rounds
    }

    /// Returns the total number of proposed tool calls.
    pub fn tool_calls(&self) -> u64 {
        self.tool_calls
    }

    /// Returns tool calls that produced recoverable errors.
    pub fn recoverable_failures(&self) -> u64 {
        self.recoverable_failures
    }

    /// Returns consecutive model turns that proposed tools.
    pub fn consecutive_tool_rounds(&self) -> u64 {
        self.consecutive_tool_rounds
    }

    /// Returns the current identical-proposal streak, including this proposal.
    pub fn repeated_proposals(&self) -> u64 {
        self.repeated_proposals
    }

    /// Returns the current identical-result-batch streak, including this batch.
    pub fn repeated_results(&self) -> u64 {
        self.repeated_results
    }

    /// Returns cumulative provider-reported input tokens.
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
    }

    /// Returns cumulative provider-reported output tokens.
    pub fn output_tokens(&self) -> u64 {
        self.output_tokens
    }

    /// Returns calls proposed for the named tool.
    pub fn calls_for(&self, tool_name: &str) -> u64 {
        self.calls_by_tool
            .get(tool_name)
            .copied()
            .unwrap_or_default()
    }

    /// Records one semantic tool proposal and its provider usage.
    pub(crate) fn record_proposal(
        &mut self,
        proposals: &[AgentToolProposal],
        usage: Option<TokenUsage>,
    ) -> Result<(), GraphError> {
        checked_increment(&mut self.model_turns, "agent model turn")?;
        checked_increment(&mut self.tool_rounds, "agent tool round")?;
        checked_increment(
            &mut self.consecutive_tool_rounds,
            "consecutive agent tool rounds",
        )?;
        checked_add(
            &mut self.tool_calls,
            u64::try_from(proposals.len()).map_err(|_| {
                GraphError::Invalid("agent tool proposal count is out of range".into())
            })?,
            "agent tool calls",
        )?;
        for proposal in proposals {
            let count = self
                .calls_by_tool
                .entry(proposal.tool_name.clone())
                .or_default();
            checked_increment(count, "agent calls by tool")?;
        }
        self.record_usage(usage)?;
        let fingerprint = proposal_fingerprint(proposals);
        update_streak(
            &mut self.last_proposal_fingerprint,
            &mut self.repeated_proposals,
            fingerprint,
            "agent proposal repetition",
        )
    }

    /// Records recoverable failures and semantic result repetition.
    pub(crate) fn record_results(&mut self, results: &[AgentToolResult]) -> Result<(), GraphError> {
        let failures = results.iter().filter(|result| result.error).count();
        checked_add(
            &mut self.recoverable_failures,
            u64::try_from(failures).map_err(|_| {
                GraphError::Invalid("agent recoverable failure count is out of range".into())
            })?,
            "agent recoverable failures",
        )?;
        let fingerprint = result_fingerprint(results);
        update_streak(
            &mut self.last_result_fingerprint,
            &mut self.repeated_results,
            fingerprint,
            "agent result repetition",
        )
    }

    /// Records a final model output and ends the consecutive tool streak.
    pub(crate) fn record_output(&mut self, usage: Option<TokenUsage>) -> Result<(), GraphError> {
        checked_increment(&mut self.model_turns, "agent model turn")?;
        self.consecutive_tool_rounds = 0;
        self.record_usage(usage)
    }

    /// Adds provider usage with checked counters.
    fn record_usage(&mut self, usage: Option<TokenUsage>) -> Result<(), GraphError> {
        let Some(usage) = usage else {
            return Ok(());
        };
        if let Some(input) = usage.input {
            checked_add(
                &mut self.input_tokens,
                u64::from(input),
                "agent input tokens",
            )?;
        }
        if let Some(output) = usage.output {
            checked_add(
                &mut self.output_tokens,
                u64::from(output),
                "agent output tokens",
            )?;
        }
        Ok(())
    }
}

/// Read-only invocation state presented to an asynchronous agent controller.
pub struct AgentLoop<T> {
    input: T,
    point: AgentInterventionPoint,
    agent_id: String,
    session_id: String,
    configured_tools: Vec<ToolInfo>,
    active_tools: Vec<ToolInfo>,
    history: Vec<Message>,
    proposal: Vec<AgentToolProposal>,
    results: Vec<AgentToolResult>,
    metrics: AgentLoopMetrics,
    budget: Option<AgentBudgetState>,
    control_state: Option<Value>,
}

impl<T> AgentLoop<T> {
    /// Returns the typed invocation input.
    pub fn input(&self) -> &T {
        &self.input
    }

    /// Returns the boundary currently awaiting a decision.
    pub fn point(&self) -> AgentInterventionPoint {
        self.point
    }

    /// Returns the stable prepared agent identity.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Returns the active history session identity.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns tools selected by the invocation configuration.
    pub fn configured_tools(&self) -> &[ToolInfo] {
        &self.configured_tools
    }

    /// Returns tools exposed on the next model dispatch.
    pub fn active_tools(&self) -> &[ToolInfo] {
        &self.active_tools
    }

    /// Returns live, possibly compacted conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Returns the staged proposal at `BeforeTools` and an empty slice otherwise.
    pub fn proposal(&self) -> &[AgentToolProposal] {
        &self.proposal
    }

    /// Returns the completed batch at `AfterTools` and an empty slice otherwise.
    pub fn results(&self) -> &[AgentToolResult] {
        &self.results
    }

    /// Returns invocation-local progress and usage observations.
    pub fn metrics(&self) -> &AgentLoopMetrics {
        &self.metrics
    }

    /// Returns ordinary model turns left before automatic conclusion.
    ///
    /// `Some(0)` means the turn budget is exhausted. `None` means this
    /// invocation has no turn budget.
    pub fn turns_remaining(&self) -> Option<u32> {
        super::turns_remaining(self.budget.as_ref(), &self.metrics)
    }

    /// Returns accepted calls left for a specifically budgeted tool.
    ///
    /// `Some(0)` means the tool budget is exhausted. `None` means the tool is
    /// unknown or has no budget; use [`Self::configured_tools`] to distinguish
    /// those cases.
    pub fn calls_remaining(&self, tool_name: &str) -> Option<u32> {
        self.budget
            .as_ref()
            .and_then(|budget| budget.calls_remaining(tool_name))
    }

    /// Returns application-owned serializable controller state.
    pub fn control_state(&self) -> Option<&Value> {
        self.control_state.as_ref()
    }
}

#[derive(Clone)]
pub(crate) struct AgentLoopData {
    pub(crate) input: Value,
    pub(crate) point: AgentInterventionPoint,
    pub(crate) agent_id: String,
    pub(crate) session_id: String,
    pub(crate) configured_tools: Vec<ToolInfo>,
    pub(crate) active_tools: Vec<ToolInfo>,
    pub(crate) history: Vec<Message>,
    pub(crate) proposal: Vec<AgentToolProposal>,
    pub(crate) results: Vec<AgentToolResult>,
    pub(crate) metrics: AgentLoopMetrics,
    pub(crate) budget: Option<AgentBudgetState>,
    pub(crate) control_state: Option<Value>,
}

impl<T> AgentLoop<T> {
    pub(crate) fn from_data(input: T, data: AgentLoopData) -> Self {
        Self {
            input,
            point: data.point,
            agent_id: data.agent_id,
            session_id: data.session_id,
            configured_tools: data.configured_tools,
            active_tools: data.active_tools,
            history: data.history,
            proposal: data.proposal,
            results: data.results,
            metrics: data.metrics,
            budget: data.budget,
            control_state: data.control_state,
        }
    }
}

/// Guidance and tool visibility applied before a later model dispatch.
#[derive(Debug, Clone, Default)]
pub struct AgentDirective {
    pub(crate) guidance: Option<String>,
    pub(crate) tool_filter: Option<ToolFilter>,
    pub(crate) tool_names: Option<Vec<String>>,
}

/// Application-selected behavior for one agent-loop boundary.
#[derive(Debug, Clone)]
pub struct AgentDecision {
    pub(crate) kind: AgentDecisionKind,
    pub(crate) state: ControlStateUpdate,
}

#[derive(Debug, Clone)]
pub(crate) enum AgentDecisionKind {
    Continue,
    Redirect(AgentDirective),
    Conclude(String),
    Suspend(Value),
    Abort(String),
}

#[derive(Debug, Clone, Default)]
pub(crate) enum ControlStateUpdate {
    #[default]
    Preserve,
    Replace(Value),
    Clear,
}

impl AgentDecision {
    /// Continues normally from the current intervention point.
    pub fn continue_() -> Self {
        Self::new(AgentDecisionKind::Continue)
    }

    /// Redirects the next model turn with guidance and/or another tool subset.
    pub fn redirect() -> Self {
        Self::new(AgentDecisionKind::Redirect(AgentDirective::default()))
    }

    /// Adds one-shot system guidance to a redirect decision.
    pub fn guidance(mut self, guidance: impl Into<String>) -> Self {
        if let AgentDecisionKind::Redirect(directive) = &mut self.kind {
            directive.guidance = Some(guidance.into());
        }
        self
    }

    /// Selects tools exposed after a redirect decision.
    pub fn tools(mut self, filter: ToolFilter) -> Self {
        if let AgentDecisionKind::Redirect(directive) = &mut self.kind {
            directive.tool_filter = Some(filter);
        }
        self
    }

    /// Requests exactly one final tool-disabled model turn.
    pub fn conclude(guidance: impl Into<String>) -> Self {
        Self::new(AgentDecisionKind::Conclude(guidance.into()))
    }

    /// Suspends the workflow with an application-owned payload.
    pub fn suspend(payload: Value) -> Self {
        Self::new(AgentDecisionKind::Suspend(payload))
    }

    /// Deliberately aborts this step without committing further runtime state.
    pub fn abort(reason: impl Into<String>) -> Self {
        Self::new(AgentDecisionKind::Abort(reason.into()))
    }

    /// Replaces controller-owned state after this decision succeeds.
    pub fn with_state(mut self, state: Value) -> Self {
        self.state = ControlStateUpdate::Replace(state);
        self
    }

    /// Clears controller-owned state after this decision succeeds.
    pub fn clear_state(mut self) -> Self {
        self.state = ControlStateUpdate::Clear;
        self
    }

    fn new(kind: AgentDecisionKind) -> Self {
        Self {
            kind,
            state: ControlStateUpdate::Preserve,
        }
    }

    pub(crate) fn redirect_names(guidance: Option<String>, tools: Option<Vec<String>>) -> Self {
        Self::new(AgentDecisionKind::Redirect(AgentDirective {
            guidance,
            tool_filter: None,
            tool_names: tools,
        }))
    }
}

/// Serializable envelope returned when an agent controller suspends execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSuspension {
    agent_id: String,
    session_id: String,
    point: AgentInterventionPoint,
    payload: Value,
}

impl AgentSuspension {
    pub(crate) fn new(
        agent_id: String,
        session_id: String,
        point: AgentInterventionPoint,
        payload: Value,
    ) -> Self {
        Self {
            agent_id,
            session_id,
            point,
            payload,
        }
    }

    /// Returns the stable prepared agent identity.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Returns the active history session identity.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the boundary at which intervention was requested.
    pub fn point(&self) -> AgentInterventionPoint {
        self.point
    }

    /// Returns the application-owned intervention request payload.
    pub fn payload(&self) -> &Value {
        &self.payload
    }
}

/// External decision supplied when resuming an agent-controller suspension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AgentResume {
    /// Continues from the suspended intervention point.
    Continue,
    /// Redirects with optional guidance and explicit stable tool names.
    Redirect {
        /// One-shot system guidance for the next model dispatch.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        guidance: Option<String>,
        /// Replacement subset of originally configured tools.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tools: Option<Vec<String>>,
    },
    /// Requests one final tool-disabled model turn.
    Conclude {
        /// System guidance supplied to the final turn.
        guidance: String,
    },
    /// Deliberately aborts the resumed step without further mutation.
    Abort {
        /// Application-facing reason for the policy abort.
        reason: String,
    },
}

fn checked_increment(value: &mut u64, label: &str) -> Result<(), GraphError> {
    checked_add(value, 1, label)
}

fn checked_add(value: &mut u64, amount: u64, label: &str) -> Result<(), GraphError> {
    *value = value
        .checked_add(amount)
        .ok_or_else(|| GraphError::Invalid(format!("{label} overflowed")))?;
    Ok(())
}

/// Updates one deterministic repetition streak with checked arithmetic.
fn update_streak(
    previous: &mut Option<String>,
    streak: &mut u64,
    current: String,
    label: &str,
) -> Result<(), GraphError> {
    if previous.as_deref() == Some(current.as_str()) {
        checked_increment(streak, label)?;
    } else {
        *previous = Some(current);
        *streak = 1;
    }
    Ok(())
}

fn proposal_fingerprint(proposals: &[AgentToolProposal]) -> String {
    let mut digest = Sha256::new();
    for proposal in proposals {
        hash_bytes(&mut digest, proposal.tool_name.as_bytes());
        hash_bytes(&mut digest, proposal.arguments.to_string().as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn result_fingerprint(results: &[AgentToolResult]) -> String {
    let mut digest = Sha256::new();
    for result in results {
        hash_bytes(&mut digest, result.tool_name.as_bytes());
        hash_bytes(&mut digest, result.arguments.to_string().as_bytes());
        hash_bytes(&mut digest, result.value.to_string().as_bytes());
        digest.update([u8::from(result.error)]);
    }
    format!("{:x}", digest.finalize())
}

fn hash_bytes(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u128).to_be_bytes());
    digest.update(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal(call_id: &str) -> AgentToolProposal {
        AgentToolProposal::new(
            call_id.into(),
            "search".into(),
            Value::object([("query", Value::from("pravah"))])
                .expect("test arguments should be valid"),
        )
    }

    fn result(call_id: &str) -> AgentToolResult {
        AgentToolResult::new(
            call_id.into(),
            "search".into(),
            proposal(call_id).arguments().clone(),
            Value::from("found"),
            false,
        )
    }

    /// Verifies repetition ignores provider call identities but retains semantic values.
    #[test]
    fn repetition_metrics_use_canonical_semantic_batches() {
        let mut metrics = AgentLoopMetrics::default();
        metrics
            .record_proposal(&[proposal("provider-a")], None)
            .expect("first proposal should record");
        metrics
            .record_proposal(&[proposal("provider-b")], None)
            .expect("equivalent proposal should record");
        metrics
            .record_results(&[result("provider-a")])
            .expect("first result should record");
        metrics
            .record_results(&[result("provider-b")])
            .expect("equivalent result should record");

        assert_eq!(metrics.repeated_proposals(), 2);
        assert_eq!(metrics.repeated_results(), 2);
        assert_eq!(metrics.calls_for("search"), 2);
    }

    /// Verifies a final output counts as a model turn and ends the tool-round streak.
    #[test]
    fn output_metrics_reset_consecutive_tool_rounds() {
        let mut metrics = AgentLoopMetrics::default();
        metrics
            .record_proposal(&[proposal("provider-a")], None)
            .expect("proposal should record");
        metrics
            .record_output(Some(TokenUsage {
                input: Some(5),
                output: Some(3),
            }))
            .expect("output should record");

        assert_eq!(metrics.model_turns(), 2);
        assert_eq!(metrics.consecutive_tool_rounds(), 0);
        assert_eq!(metrics.input_tokens(), 5);
        assert_eq!(metrics.output_tokens(), 3);
    }
}
