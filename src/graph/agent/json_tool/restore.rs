//! JSON-specific restore checks derived from authored data, without retained state.

use super::*;
use crate::graph::AgentResponse;

/// Validates a pending JSON-tool input before exposing a restored external operation.
pub(in crate::graph::agent) fn validate_input(
    payload: &Value,
    input: &Value,
) -> Result<(), GraphError> {
    if let Some(definition) = payload.get("json_contract") {
        JsonToolContract::from_definition(definition)?.validate_input(input)?;
    }
    Ok(())
}

/// Checks accepted worker evidence without dispatching it or treating failures as successes.
pub(crate) fn validate_json_outcome(
    payload: &Value,
    response: &AgentResponse,
) -> Result<(), GraphError> {
    if let Some(definition) = payload.get("json_contract")
        && let Ok(output) = response.outcome()
    {
        JsonToolContract::from_definition(definition)?.validate_envelope(output)?;
    }
    Ok(())
}

/// Validates retained input/output and admitted calls, while preserving invalid staged proposals.
pub(crate) fn validate_json_snapshot(
    payload: &Value,
    children: &[UntypedGraph],
    checkpoint: Option<&Value>,
    input: Option<&Value>,
    output: Option<&Value>,
) -> Result<(), GraphError> {
    if let Some(definition) = payload.get("json_contract") {
        let contract = JsonToolContract::from_definition(definition)?;
        if let Some(input) = input {
            contract.validate_input(input)?;
        }
        if let Some(output) = output {
            contract.validate_envelope(output)?;
        }
    }
    if payload.get("version").and_then(Value::as_u64) == Some(u64::from(JSON_PAYLOAD_VERSION))
        && let Some(checkpoint) = checkpoint
    {
        validate_checkpoint_values(payload, children, checkpoint)?;
    }
    Ok(())
}

/// Follows existing effect wrappers and validates only calls that have already passed admission.
fn validate_checkpoint_values(
    payload: &Value,
    children: &[UntypedGraph],
    checkpoint: &Value,
) -> Result<(), GraphError> {
    if let Some(nested) = checkpoint.get("checkpoint") {
        return validate_checkpoint_values(payload, children, nested);
    }
    if checkpoint.get("effect").is_some() {
        return Ok(());
    }
    let state: EdgeAgentCheckpoint = from_value(checkpoint.clone())
        .map_err(|_| GraphError::SnapshotValidation("invalid JSON agent checkpoint".into()))?;
    let agent = decode_payload(payload)?;
    match state.phase {
        EdgeAgentPhase::PendingTool {
            active,
            waiting,
            results,
        } => {
            for call in active {
                validate_call(children, call.child_index, &call.args)?;
            }
            for call in waiting {
                validate_call(children, call.child_index, &call.args)?;
                if child_contract(&children[call.child_index]).is_some() && call.args != call.input
                {
                    return Err(GraphError::SnapshotValidation(
                        "JSON tool queued input differs from arguments".into(),
                    ));
                }
            }
            for result in results {
                validate_result(&agent.tools, children, &result.result)?;
            }
        }
        EdgeAgentPhase::AfterTools { results } => {
            for result in results {
                validate_result(&agent.tools, children, &result)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_call(children: &[UntypedGraph], index: usize, args: &Value) -> Result<(), GraphError> {
    let child = children
        .get(index)
        .ok_or_else(|| GraphError::SnapshotValidation("invalid tool index".into()))?;
    if let Some(definition) = child_contract(child) {
        JsonToolContract::from_definition(definition)?.validate_input(args)?;
    }
    Ok(())
}

/// Error results may intentionally describe invalid arguments and have no success-schema obligation.
fn validate_result(
    tools: &[AgentToolPayload],
    children: &[UntypedGraph],
    result: &AgentToolResult,
) -> Result<(), GraphError> {
    if result.is_error() {
        return Ok(());
    }
    if let Some(tool) = tools.iter().find(|tool| tool.name == result.tool_name()) {
        let child = children
            .get(tool.child_index)
            .ok_or_else(|| GraphError::SnapshotValidation("invalid result child index".into()))?;
        if let Some(definition) = child_contract(child) {
            let contract = JsonToolContract::from_definition(definition)?;
            contract.validate_input(result.arguments())?;
            let output = serde_json::to_value(result.value())
                .map_err(|e| GraphError::Invalid(e.to_string()))?;
            contract
                .validate(&output, true)
                .map_err(GraphError::SnapshotValidation)?;
        }
    }
    Ok(())
}
