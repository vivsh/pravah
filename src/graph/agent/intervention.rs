use super::execution::normal_dispatch_phase;
use super::*;
use crate::graph::model::TypeSpec;
use crate::graph::registry::ContinuationSuspension;

/// Validates text and redirect requirements before checkpoint mutation.
pub(super) fn validate_decision(decision: &AgentDecision) -> Result<(), GraphError> {
    match &decision.kind {
        AgentDecisionKind::Redirect(directive)
            if directive.guidance.is_none()
                && directive.tool_filter.is_none()
                && directive.tool_names.is_none() =>
        {
            Err(GraphError::AgentControlValidation(
                "redirect requires guidance or a tool selection".into(),
            ))
        }
        AgentDecisionKind::Redirect(directive) => {
            validate_optional_text(directive.guidance.as_deref(), "redirect guidance")
        }
        AgentDecisionKind::Conclude(guidance) => {
            validate_required_text(guidance, "conclusion guidance")
        }
        AgentDecisionKind::Abort(reason) => validate_required_text(reason, "abort reason"),
        AgentDecisionKind::Continue | AgentDecisionKind::Suspend(_) => Ok(()),
    }
}

fn validate_optional_text(value: Option<&str>, label: &str) -> Result<(), GraphError> {
    if value.is_some_and(|value| value.trim().is_empty()) {
        Err(GraphError::AgentControlValidation(format!(
            "{label} must not be empty"
        )))
    } else {
        Ok(())
    }
}

fn validate_required_text(value: &str, label: &str) -> Result<(), GraphError> {
    if value.trim().is_empty() {
        Err(GraphError::AgentControlValidation(format!(
            "{label} must not be empty"
        )))
    } else {
        Ok(())
    }
}

pub(super) fn apply_control_state(
    checkpoint: &mut EdgeAgentCheckpoint,
    update: ControlStateUpdate,
) {
    match update {
        ControlStateUpdate::Preserve => {}
        ControlStateUpdate::Replace(value) => checkpoint.control_state = Some(value),
        ControlStateUpdate::Clear => checkpoint.control_state = None,
    }
}

/// Applies guidance and a deterministic subset of configured tools.
pub(super) fn apply_directive(
    payload: &AgentPayload,
    checkpoint: &mut EdgeAgentCheckpoint,
    directive: AgentDirective,
) -> Result<(), GraphError> {
    if let Some(guidance) = directive.guidance {
        checkpoint.guidance = Some(guidance);
    }
    if let Some(filter) = directive.tool_filter {
        checkpoint.selected_tools = payload
            .tools
            .iter()
            .filter(|tool| checkpoint.resolved.tools.contains(&tool.name) && filter.allows(tool))
            .map(|tool| tool.name.clone())
            .collect();
    }
    if let Some(names) = directive.tool_names {
        checkpoint.selected_tools = validate_explicit_tools(payload, checkpoint, names)?;
    }
    Ok(())
}

/// Validates resume-selected identities and restores prepared tool order.
fn validate_explicit_tools(
    payload: &AgentPayload,
    checkpoint: &EdgeAgentCheckpoint,
    names: Vec<String>,
) -> Result<Vec<String>, GraphError> {
    let name_count = names.len();
    let requested = names.into_iter().collect::<BTreeSet<_>>();
    if requested.len() != name_count
        || requested
            .iter()
            .any(|name| !checkpoint.resolved.tools.contains(name))
    {
        return Err(GraphError::AgentResumeValidation(
            "redirect tools must be unique configured tool names".into(),
        ));
    }
    Ok(payload
        .tools
        .iter()
        .filter(|tool| requested.contains(&tool.name))
        .map(|tool| tool.name.clone())
        .collect())
}

pub(super) fn redirect_phase(
    point: AgentInterventionPoint,
    checkpoint: &EdgeAgentCheckpoint,
) -> EdgeAgentPhase {
    match point {
        AgentInterventionPoint::BeforeTools => EdgeAgentPhase::BeforeModel,
        AgentInterventionPoint::BeforeModel | AgentInterventionPoint::AfterTools => {
            normal_dispatch_phase(checkpoint)
        }
    }
}

pub(super) fn checkpoint_point(phase: &EdgeAgentPhase) -> Option<AgentInterventionPoint> {
    match phase {
        EdgeAgentPhase::BeforeModel => Some(AgentInterventionPoint::BeforeModel),
        EdgeAgentPhase::BeforeTools { .. } => Some(AgentInterventionPoint::BeforeTools),
        EdgeAgentPhase::AfterTools { .. } => Some(AgentInterventionPoint::AfterTools),
        EdgeAgentPhase::Dispatch { .. }
        | EdgeAgentPhase::AcceptedTools { .. }
        | EdgeAgentPhase::PendingTool { .. } => None,
    }
}

/// Builds a continuation-owned suspension retaining the current checkpoint.
pub(super) fn suspend_agent(
    payload: &AgentPayload,
    checkpoint: &EdgeAgentCheckpoint,
    point: AgentInterventionPoint,
    value: Value,
) -> Result<ContinuationTransition, GraphError> {
    let envelope = AgentSuspension::new(
        payload.agent_id.clone(),
        checkpoint.session_id.clone(),
        point,
        value,
    );
    let payload = to_value(envelope).map_err(|err| GraphError::ValueConversion {
        target: "agent suspension".into(),
        reason: err.to_string(),
    })?;
    let resume_type = TypeSpec::new(AgentResume::schema_name(), schema_for::<AgentResume>());
    Ok(ContinuationTransition {
        checkpoint: Some(
            to_value(checkpoint).map_err(|err| GraphError::ValueConversion {
                target: "agent checkpoint".into(),
                reason: err.to_string(),
            })?,
        ),
        state: None,
        outputs: Vec::new(),
        writes: Vec::new(),
        child_calls: Vec::new(),
        suspension: Some(ContinuationSuspension {
            resume_type,
            payload,
        }),
    })
}
