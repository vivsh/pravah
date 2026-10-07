//! Asynchronous callbacks executed by the agent worker, never by the VM.

use super::effects::*;
use super::*;
use crate::graph::agent_request::AgentOperation;
use crate::graph::{AgentRequest, DynAgentHandler};

impl DynAgentHandler for AgentHandler {
    fn execute<'a>(
        &'a self,
        request: &'a AgentRequest,
        context: Context,
    ) -> BoxFuture<'a, Result<Value, GraphError>> {
        async move {
            match request.operation.as_ref() {
                AgentOperation::Configure {
                    definition,
                    input,
                    execution_id,
                    ..
                } => {
                    let payload = self.validated_payload(definition)?;
                    let config = self
                        .configure
                        .configure(
                            input.clone(),
                            payload.configuration,
                            *execution_id,
                            context.clone(),
                        )
                        .await?;
                    encode(resolve_agent_config(&payload, config, &context).await?)
                }
                AgentOperation::Control {
                    definition,
                    observation,
                    ..
                } => {
                    let payload = self.validated_payload(definition)?;
                    let observation = super::effect_values::read_observation(observation)?;
                    let configured = observation
                        .configured_tools
                        .iter()
                        .map(|tool| tool.name().to_owned())
                        .collect::<Vec<_>>();
                    let controller = self.controller.as_ref().ok_or_else(|| {
                        GraphError::AgentRequestValidation("missing controller".into())
                    })?;
                    let decision = controller.control(observation, context).await?;
                    encode(resolve_decision(decision, &payload, &configured)?)
                }
                _ => Err(GraphError::AgentRequestValidation(
                    "wrong callback operation".into(),
                )),
            }
        }
        .boxed()
    }
}

/// Resolves predicates to portable ordered names; no executable closure crosses the boundary.
fn resolve_decision(
    decision: AgentDecision,
    payload: &AgentPayloadView<'_>,
    configured: &[String],
) -> Result<ResolvedDecision, GraphError> {
    super::intervention::validate_decision(&decision)?;
    let kind = match decision.kind {
        AgentDecisionKind::Continue => DecisionKind::Continue,
        AgentDecisionKind::Conclude(text) => DecisionKind::Conclude(text),
        AgentDecisionKind::Suspend(value) => DecisionKind::Suspend(value),
        AgentDecisionKind::Abort(reason) => DecisionKind::Abort(reason),
        AgentDecisionKind::Redirect(directive) => {
            let tools = if let Some(filter) = directive.tool_filter {
                filter
                    .validate_names(configured.iter().map(String::as_str))
                    .map_err(GraphError::AgentControlValidation)?;
                Some(
                    payload
                        .tools
                        .iter()
                        .filter(|tool| configured.contains(&tool.name) && filter.allows(tool))
                        .map(|tool| tool.name.clone())
                        .collect(),
                )
            } else {
                directive.tool_names
            };
            DecisionKind::Redirect {
                guidance: directive.guidance,
                tools,
            }
        }
    };
    Ok(ResolvedDecision {
        kind,
        state: decision.state,
    })
}
