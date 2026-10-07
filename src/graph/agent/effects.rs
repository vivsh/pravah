//! Durable agent boundaries and atomic history transitions.

use super::execution::normal_dispatch_phase;
use super::intervention::checkpoint_point;
use super::*;
use crate::graph::agent_request::AgentOperation;
use crate::graph::registry::HistoryChange;
use crate::graph::{AgentError, AgentRequest, AgentResponse};

#[derive(Serialize, Deserialize)]
#[serde(tag = "effect", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum AgentEffectCheckpoint {
    Configure { version: u32, input: Value },
    Control { version: u32, checkpoint: Value },
    Generate { version: u32, checkpoint: Value },
    Flush { version: u32, transition: Value },
}

#[derive(Serialize, Deserialize)]
pub(super) struct Configured {
    pub key: Option<String>,
    pub resolved: ResolvedAgentConfig,
    pub message: Message,
    pub budget: Option<AgentBudgetState>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct ResolvedDecision {
    pub kind: DecisionKind,
    pub state: ControlStateUpdate,
}

#[derive(Serialize, Deserialize)]
pub(super) enum DecisionKind {
    Continue,
    Redirect {
        guidance: Option<String>,
        tools: Option<Vec<String>>,
    },
    Conclude(String),
    Suspend(Value),
    Abort(String),
}

/// Retains the checkpoint and immutable operation under separate durable owners.
pub(super) fn effect(
    checkpoint: AgentEffectCheckpoint,
    operation: AgentOperation,
) -> Result<ContinuationTransition, GraphError> {
    Ok(ContinuationTransition {
        checkpoint: Some(checkpoint.into_value()?),
        agent: Some(AgentRequest::new(Uuid::nil(), operation)),
        ..Default::default()
    })
}

pub(super) fn encode(data: impl Serialize) -> Result<Value, GraphError> {
    to_value(data).map_err(|err| GraphError::ValueConversion {
        target: "agent effect".into(),
        reason: err.to_string(),
    })
}

pub(super) fn decode<T: DeserializeOwned>(value: &Value) -> Result<T, GraphError> {
    from_value(value.clone())
        .map_err(|_| GraphError::AgentRequestValidation("invalid agent outcome".into()))
}

pub(super) fn success(response: AgentResponse) -> Result<Value, GraphError> {
    response
        .outcome
        .map_err(|source| GraphError::AgentFailed { source })
}

/// Appends accepted messages once; final output waits for required persistence acknowledgement.
pub(super) fn record(
    ctx: ContinuationContext<'_>,
    session: &str,
    agent: &str,
    messages: Vec<Message>,
    mut next: ContinuationTransition,
) -> Result<ContinuationTransition, GraphError> {
    let entries = ctx
        .history()
        .stage_entries(ctx.execution_id(), session, agent, messages)?;
    if ctx.history_policy().persist && !next.outputs.is_empty() {
        let mut flushed = effect(
            AgentEffectCheckpoint::Flush {
                version: CHECKPOINT_VERSION,
                transition: encode(next)?,
            },
            AgentOperation::PersistHistory,
        )?;
        if let Some(request) = &mut flushed.agent {
            request.persist = Some(entries.clone().into());
        }
        flushed.history.push(HistoryChange::Append(entries));
        return Ok(flushed);
    }
    next.history.push(HistoryChange::Append(entries));
    Ok(next)
}

impl AgentHandler {
    /// Interprets only the recorded completion; failure never redispatches the completed operation.
    pub(super) fn advance_effect(
        &self,
        payload_value: &Value,
        state: AgentEffectCheckpoint,
        event: ContinuationEvent,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let ContinuationEvent::Agent { request, response } = event else {
            return Err(GraphError::Invalid(
                "agent effect has no accepted completion".into(),
            ));
        };
        if let Err(error) = response.outcome() {
            return Err(delivered_error(payload_value, error)?);
        }
        let value = success(response)?;
        match state {
            AgentEffectCheckpoint::Configure { version, input } => {
                check_version(version)?;
                self.accept_configuration(
                    &AgentPayloadView::read(payload_value)?,
                    input,
                    request.id(),
                    &value,
                    ctx,
                )
            }
            AgentEffectCheckpoint::Control {
                version,
                checkpoint,
            } => {
                check_version(version)?;
                let checkpoint = EdgeAgentCheckpoint::from_value(&checkpoint)?;
                let point = checkpoint_point(&checkpoint.phase).ok_or_else(|| {
                    GraphError::AgentControlValidation("invalid controller phase".into())
                })?;
                self.apply_decision(
                    &AgentPayloadView::read(payload_value)?,
                    checkpoint,
                    point,
                    decode::<ResolvedDecision>(&value)?.into_decision(),
                )
            }
            AgentEffectCheckpoint::Generate {
                version,
                checkpoint,
            } => {
                check_version(version)?;
                let checkpoint = EdgeAgentCheckpoint::from_value(&checkpoint)?;
                let response = crate::graph::agent_request::client_response::Response::<JsonValue>::deserialize(&value)
                    .map_err(|_| GraphError::AgentRequestValidation("invalid generation outcome".into()))?.into_response();
                self.accept_generation(
                    &AgentPayloadView::read(payload_value)?,
                    checkpoint,
                    response,
                    ctx,
                )
            }
            AgentEffectCheckpoint::Flush {
                version,
                transition,
            } => {
                check_version(version)?;
                decode(&transition)
            }
        }
    }

    /// Commits resolved configuration and the initial user message at one VM boundary.
    fn accept_configuration(
        &self,
        payload: &AgentPayloadView<'_>,
        input: Value,
        id: Uuid,
        value: &Value,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let Configured {
            key,
            resolved,
            message,
            budget,
        } = decode(value)?;
        if !matches!(message.role, Role::User) {
            return Err(GraphError::AgentConfigValidation(
                "initial message must be user-role".into(),
            ));
        }
        validate_resolved_config(&resolved)?;
        let session_id = super::conversation::select_session(key, id, ctx.history())?;
        let mut checkpoint = EdgeAgentCheckpoint {
            version: CHECKPOINT_VERSION,
            phase: EdgeAgentPhase::BeforeModel,
            session_id: session_id.clone(),
            input,
            selected_tools: resolved.tools.clone(),
            resolved: value
                .get("resolved")
                .ok_or_else(|| {
                    GraphError::AgentRequestValidation("missing resolved configuration".into())
                })?
                .clone(),
            budget,
            guidance: None,
            metrics: AgentLoopMetrics::default(),
            control_state: None,
        };
        validate_checkpoint_progress(&payload.tools, &checkpoint)?;
        if self.controller.is_none() {
            checkpoint.phase = normal_dispatch_phase(&checkpoint);
        }
        record(
            ctx,
            &session_id,
            payload.agent_id,
            vec![message],
            persist_checkpoint(checkpoint)?,
        )
    }
}

/// Classifies portable output-limit failures without fabricating a local Rath source.
fn delivered_error(payload: &Value, error: &AgentError) -> Result<GraphError, GraphError> {
    if error.code() == "rath"
        && error
            .details()
            .and_then(|value| value.get("rath"))
            .and_then(|value| value.get("kind"))
            .and_then(Value::as_str)
            == Some("output_limit_reached")
    {
        let provider = error
            .details()
            .and_then(|value| value.get("rath"))
            .and_then(|value| value.get("provider"))
            .ok_or_else(|| {
                GraphError::AgentRequestValidation("missing output-limit provider".into())
            })?;
        return Ok(GraphError::AgentOutputLimit {
            agent: super::payload::validate_identity(payload)?.into(),
            provider: decode(provider)?,
        });
    }
    Ok(GraphError::AgentFailed {
        source: error.clone(),
    })
}

impl ResolvedDecision {
    fn into_decision(self) -> AgentDecision {
        let mut decision = match self.kind {
            DecisionKind::Continue => AgentDecision::continue_(),
            DecisionKind::Redirect { guidance, tools } => {
                AgentDecision::redirect_names(guidance, tools)
            }
            DecisionKind::Conclude(text) => AgentDecision::conclude(text),
            DecisionKind::Suspend(value) => AgentDecision::suspend(value),
            DecisionKind::Abort(reason) => AgentDecision::abort(reason),
        };
        decision.state = self.state;
        decision
    }
}

fn check_version(got: u32) -> Result<(), GraphError> {
    if got == CHECKPOINT_VERSION {
        return Ok(());
    }
    Err(GraphError::UnsupportedVersion {
        format: "agent checkpoint",
        got,
        expected: CHECKPOINT_VERSION,
    })
}

/// Checks durable effect phases before a restored runtime can expose or interpret them.
pub(super) fn validate_effect_checkpoint(
    tools: &[AgentToolPayload],
    value: &Value,
) -> Result<(), GraphError> {
    let checkpoint = AgentEffectCheckpoint::from_value(value)?;
    match checkpoint {
        AgentEffectCheckpoint::Flush { version, .. }
        | AgentEffectCheckpoint::Configure { version, .. } => {
            check_version(version)?;
        }
        AgentEffectCheckpoint::Control {
            version,
            checkpoint,
        }
        | AgentEffectCheckpoint::Generate {
            version,
            checkpoint,
        } => {
            check_version(version)?;
            let checkpoint = EdgeAgentCheckpoint::from_value(&checkpoint)?;
            check_version(checkpoint.version)?;
            validate_checkpoint(tools, &checkpoint)?;
        }
    }
    Ok(())
}
