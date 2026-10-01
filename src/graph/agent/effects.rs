//! Durable agent boundaries; resolved state stays in the continuation checkpoint.

use super::execution::normal_dispatch_phase;
use super::intervention::checkpoint_point;
use super::*;
use crate::graph::registry::HistoryChange;
use crate::graph::{FetchBody, FetchError, FetchRequest, FetchResponse};

pub(super) const AGENT_HOOK_VERSION: u32 = 2;

/// Rejects configure hooks lacking the execution identity required for Chat defaults.
pub(super) fn check_agent_hook_version(got: u32) -> Result<(), GraphError> {
    if got == AGENT_HOOK_VERSION {
        return Ok(());
    }
    Err(GraphError::UnsupportedVersion {
        format: "agent hook",
        got,
        expected: AGENT_HOOK_VERSION,
    })
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "effect", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum AgentEffectCheckpoint {
    Configure { version: u32, input: Value },
    Control { version: u32, checkpoint: Value },
    Prepare { version: u32, checkpoint: Value },
    Generate { version: u32, checkpoint: Value },
}

#[derive(Serialize, Deserialize)]
pub(super) struct AgentHook {
    pub version: u32,
    pub handler: String,
    pub payload: Value,
    pub operation: AgentHookOperation,
}

#[derive(Serialize, Deserialize)]
#[expect(
    clippy::large_enum_variant,
    reason = "operation-local observations avoid another heap allocation"
)]
pub(super) enum AgentHookOperation {
    Configure { input: Value, execution_id: Uuid },
    Control { observation: AgentLoopData },
}

#[derive(Serialize, Deserialize)]
pub(super) struct Configured {
    pub key: Option<String>,
    pub resolved: ResolvedAgentConfig,
    pub message: Message,
    pub budget: Option<AgentBudgetState>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct PreparationRequest {
    pub version: u32,
    pub request: crate::graph::fetch::rath::RathRequest,
    pub guidance: Vec<Message>,
    pub budget_conclusion: bool,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Prepared {
    pub version: u32,
    pub generation: Value,
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

/// Encodes an explicit boundary; the checkpoint and request have separate durable ownership.
pub(super) fn effect(
    checkpoint: AgentEffectCheckpoint,
    request: FetchRequest,
) -> Result<ContinuationTransition, GraphError> {
    Ok(ContinuationTransition {
        checkpoint: Some(checkpoint.into_value()?),
        fetch: Some(request),
        ..Default::default()
    })
}

pub(super) fn response(data: impl Serialize) -> Result<FetchResponse, GraphError> {
    Ok(FetchResponse::new(200).body(FetchBody::Value(encode(data)?)))
}

pub(super) fn encode(data: impl Serialize) -> Result<Value, GraphError> {
    to_value(data).map_err(|err| GraphError::ValueConversion {
        target: "agent effect".into(),
        reason: err.to_string(),
    })
}

pub(super) fn decode<T: DeserializeOwned>(body: Option<&FetchBody>) -> Result<T, GraphError> {
    let Some(FetchBody::Value(value)) = body else {
        return Err(GraphError::FetchValidation(
            "expected structured hook payload".into(),
        ));
    };
    from_value(value.clone())
        .map_err(|_| GraphError::FetchValidation("invalid hook payload".into()))
}

pub(super) fn success(
    outcome: Result<FetchResponse, FetchError>,
) -> Result<FetchResponse, GraphError> {
    let response = outcome.map_err(|source| GraphError::FetchFailed { source })?;
    if response.status() != 200 {
        return Err(GraphError::FetchValidation(
            "invalid hook acknowledgement".into(),
        ));
    }
    Ok(response)
}

/// Stages a synchronous append; the caller persists committed rows outside the VM.
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
    next.history.push(HistoryChange::Append(entries));
    Ok(next)
}

impl AgentHandler {
    /// Consumes an already recorded external outcome; failures leave the inbox untouched.
    pub(super) fn advance_effect(
        &self,
        payload_value: &Value,
        state: AgentEffectCheckpoint,
        event: ContinuationEvent,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let ContinuationEvent::Fetch { fetch, outcome } = event else {
            return Err(GraphError::Invalid(
                "agent effect has no accepted outcome".into(),
            ));
        };
        let response = match outcome {
            Err(error)
                if error.code() == "rath"
                    && error
                        .details()
                        .and_then(|v| v.get("rath"))
                        .and_then(|v| v.get("kind"))
                        .and_then(Value::as_str)
                        == Some("output_limit_reached") =>
            {
                let provider = error
                    .details()
                    .and_then(|v| v.get("rath"))
                    .and_then(|v| v.get("provider"))
                    .cloned()
                    .ok_or_else(|| {
                        GraphError::FetchValidation("missing output-limit provider".into())
                    })?;
                return Err(GraphError::AgentOutputLimit {
                    agent: super::payload::validate_identity(payload_value)?.into(),
                    provider: from_value(provider).map_err(|_| {
                        GraphError::FetchValidation("invalid output-limit provider".into())
                    })?,
                });
            }
            outcome => success(outcome)?,
        };
        match state {
            AgentEffectCheckpoint::Configure { version, input } => {
                check_version(version)?;
                self.accept_configuration(
                    &AgentPayloadView::read(payload_value)?,
                    input,
                    fetch.id(),
                    &response,
                    ctx,
                )
            }
            AgentEffectCheckpoint::Prepare {
                version,
                checkpoint,
            } => {
                check_version(version)?;
                let prepared = Prepared::from_response(&response)?;
                check_protocol(prepared.version)?;
                let generation = FetchRequest::from_value(&prepared.generation)?;
                effect(
                    AgentEffectCheckpoint::Generate {
                        version,
                        checkpoint,
                    },
                    generation,
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
                    decode::<ResolvedDecision>(response.body_ref())?.into_decision(),
                )
            }
            AgentEffectCheckpoint::Generate {
                version,
                checkpoint,
            } => {
                check_version(version)?;
                let checkpoint = EdgeAgentCheckpoint::from_value(&checkpoint)?;
                let response =
                    crate::graph::fetch::rath::RathResponse::from_fetch_response(&response)?
                        .into_response();
                self.accept_generation(
                    &AgentPayloadView::read(payload_value)?,
                    checkpoint,
                    response,
                    ctx,
                )
            }
        }
    }

    /// Commits invocation state and its initial message in one synchronous transition.
    fn accept_configuration(
        &self,
        payload: &AgentPayloadView<'_>,
        input: Value,
        id: Uuid,
        response: &FetchResponse,
        ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let Configured {
            key,
            resolved,
            message,
            budget,
        } = decode(response.body_ref())?;
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
            resolved: super::effect_values::body_field(response.body_ref(), "resolved")?.clone(),
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

pub(super) fn check_protocol(got: u32) -> Result<(), GraphError> {
    if got == 1 {
        return Ok(());
    }
    Err(GraphError::UnsupportedVersion {
        format: "Pravah hook",
        got,
        expected: 1,
    })
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
        AgentEffectCheckpoint::Configure { version, .. } => {
            check_version(version)?;
        }
        AgentEffectCheckpoint::Control {
            version,
            checkpoint,
        }
        | AgentEffectCheckpoint::Prepare {
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
