//! Checks durable work against its authored continuation and checkpoint during restore.

use super::*;
use crate::graph::agent_request::{AgentOperation, AgentRequest};

/// Validates stable handler, phase and invocation relationships without introducing runtime state.
pub(crate) fn validate_operation(
    payload: &Value,
    checkpoint: &Value,
    request: &AgentRequest,
    execution_id: Uuid,
    output_validator: Option<&jsonschema::Validator>,
    retained_input: Option<&Value>,
) -> Result<(), GraphError> {
    if let Some(key) = payload.get("tool_handler_key").and_then(Value::as_str) {
        return validate_tool_operation(key, payload, checkpoint, request, retained_input);
    }
    if payload.get("agent_id").is_none() {
        return Ok(());
    }
    let state = effects::AgentEffectCheckpoint::from_value(checkpoint)?;
    match (state, request.operation.as_ref()) {
        (
            effects::AgentEffectCheckpoint::Configure { input, .. },
            operation @ AgentOperation::Configure { .. },
        ) => validate_configuration(payload, &input, operation, execution_id),
        (
            effects::AgentEffectCheckpoint::Control { checkpoint, .. },
            AgentOperation::Control {
                handler,
                definition,
                observation,
            },
        ) if Some(handler.as_str()) == payload.get("agent_id").and_then(Value::as_str)
            && definition == payload =>
        {
            validate_control(&checkpoint, observation)
        }
        (
            effects::AgentEffectCheckpoint::Generate { checkpoint, .. },
            operation @ AgentOperation::Generate { .. },
        ) => validate_generation(payload, &checkpoint, operation),
        (
            effects::AgentEffectCheckpoint::Flush { transition, .. },
            AgentOperation::PersistHistory,
        ) => validate_flush(&transition, output_validator),
        _ => Err(invalid()),
    }
}

/// Configuration work must retain the exact authored definition and invocation identity.
fn validate_configuration(
    payload: &Value,
    input: &Value,
    operation: &AgentOperation,
    execution_id: Uuid,
) -> Result<(), GraphError> {
    let AgentOperation::Configure {
        execution_id: owner,
        handler,
        definition,
        input: requested,
        ..
    } = operation
    else {
        return Err(invalid());
    };
    if *owner != execution_id
        || Some(handler.as_str()) != payload.get("agent_id").and_then(Value::as_str)
        || definition != payload
        || requested != input
    {
        return Err(invalid());
    }
    Ok(())
}

/// Function-tool work must retain its authored handler and waiting checkpoint.
fn validate_tool_operation(
    key: &str,
    payload: &Value,
    checkpoint: &Value,
    request: &AgentRequest,
    retained_input: Option<&Value>,
) -> Result<(), GraphError> {
    match request.operation.as_ref() {
        AgentOperation::Tool { handler, input }
            if handler.as_str() == key && checkpoint.as_u64() == Some(1) =>
        {
            if payload.get("json_contract").is_some() && retained_input != Some(input) {
                return Err(invalid());
            }
            super::json_tool::validate_restored_json_input(payload, input)
        }
        _ => Err(invalid()),
    }
}

/// A final persistence checkpoint can only release one validated output after acknowledgement.
fn validate_flush(
    value: &Value,
    validator: Option<&jsonschema::Validator>,
) -> Result<(), GraphError> {
    let transition: ContinuationTransition = effects::decode(value)?;
    if transition.outputs.len() != 1
        || transition.agent.is_some()
        || transition.checkpoint.is_some()
        || transition.state.is_some()
        || transition.suspension.is_some()
        || !transition.history.is_empty()
        || !transition.writes.is_empty()
        || !transition.child_calls.is_empty()
    {
        return Err(invalid());
    }
    let output = transition.outputs.first().ok_or_else(invalid)?;
    if let Some(validator) = validator {
        let json = serde_json::to_value(output).map_err(|_| invalid())?;
        if !validator.is_valid(&json) {
            return Err(invalid());
        }
    }
    Ok(())
}

/// Pending generation options must match the checkpointed model, tool surface and authored output schema.
fn validate_generation(
    payload: &Value,
    state: &Value,
    operation: &AgentOperation,
) -> Result<(), GraphError> {
    let AgentOperation::Generate {
        session_id,
        model,
        options,
        budget_conclusion,
        ..
    } = operation
    else {
        return Err(invalid());
    };
    let checkpoint = EdgeAgentCheckpoint::from_value(state)?;
    let EdgeAgentPhase::Dispatch { conclusion } = checkpoint.phase else {
        return Err(invalid());
    };
    let expected = request::generation(
        &AgentPayloadView::read(payload)?,
        &checkpoint,
        conclusion.is_some(),
    )?;
    if session_id != &checkpoint.session_id
        || expected.get("model").and_then(Value::as_str) != Some(model.as_str())
        || expected.get("options") != Some(options)
        || *budget_conclusion != matches!(conclusion, Some(ConclusionCause::TurnBudget))
    {
        return Err(invalid());
    }
    Ok(())
}

/// Controller observations retain the same input, session and intervention point as their checkpoint.
fn validate_control(state: &Value, observation: &Value) -> Result<(), GraphError> {
    let checkpoint = EdgeAgentCheckpoint::from_value(state)?;
    let point = intervention::checkpoint_point(&checkpoint.phase).ok_or_else(invalid)?;
    let observation = effect_values::read_observation(observation)?;
    if observation.input != checkpoint.input
        || observation.session_id != checkpoint.session_id
        || observation.point != point
    {
        return Err(invalid());
    }
    Ok(())
}

/// Import acknowledgement must carry a valid configuration before it can change working history.
pub(crate) fn validate_loaded_configuration(
    payload: &Value,
    value: &Value,
) -> Result<(), GraphError> {
    let configured: effects::Configured = effects::decode(value)?;
    if !matches!(configured.message.role, Role::User) {
        return Err(GraphError::AgentConfigValidation(
            "initial message must be user-role".into(),
        ));
    }
    validate_resolved_config(&configured.resolved)?;
    let tools = &AgentPayloadView::read(payload)?.tools;
    let checkpoint = EdgeAgentCheckpoint {
        version: CHECKPOINT_VERSION,
        phase: EdgeAgentPhase::BeforeModel,
        session_id: format!("key:{}", configured.key.as_deref().ok_or_else(invalid)?),
        input: Value::NULL,
        selected_tools: configured.resolved.tools.clone(),
        resolved: value.get("resolved").ok_or_else(invalid)?.clone(),
        budget: configured.budget,
        guidance: None,
        metrics: AgentLoopMetrics::default(),
        control_state: None,
    };
    validate_checkpoint_progress(tools, &checkpoint)
}

fn invalid() -> GraphError {
    GraphError::SnapshotValidation("agent work does not match its continuation checkpoint".into())
}
