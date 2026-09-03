use super::*;

/// Validates and converts one complete provider proposal into checkpoint data.
pub(super) fn stage_tool_calls(
    calls: Vec<ToolCall>,
) -> Result<Vec<EdgeProposedToolCall>, GraphError> {
    let mut seen = BTreeSet::new();
    calls
        .into_iter()
        .map(|call| {
            if call.id.is_empty() || call.name.is_empty() || !seen.insert(call.id.clone()) {
                return Err(GraphError::AgentControlValidation(
                    "model tool call ids and names must be non-empty, and ids must be unique"
                        .into(),
                ));
            }
            let arguments = to_value(call.args).map_err(|err| GraphError::ValueConversion {
                target: format!("tool '{}' arguments", call.name),
                reason: err.to_string(),
            })?;
            Ok(EdgeProposedToolCall {
                proposal: AgentToolProposal::new(call.id, call.name, arguments),
                thought_signatures: call.thought_signatures,
            })
        })
        .collect()
}

/// Reconstructs the accepted assistant tool-call message from staged values.
pub(super) fn assistant_tool_call_message(
    thought: Option<String>,
    calls: &[EdgeProposedToolCall],
    usage: Option<crate::clients::TokenUsage>,
) -> Result<Message, GraphError> {
    let calls = calls
        .iter()
        .map(|call| {
            let args = serde_json::to_value(call.proposal.arguments()).map_err(|err| {
                GraphError::ValueConversion {
                    target: format!("tool '{}' arguments", call.proposal.tool_name()),
                    reason: err.to_string(),
                }
            })?;
            Ok(ToolCall {
                id: call.proposal.call_id().to_owned(),
                name: call.proposal.tool_name().to_owned(),
                args,
                thought_signatures: call.thought_signatures.clone(),
            })
        })
        .collect::<Result<Vec<_>, GraphError>>()?;
    Ok(Message {
        role: Role::AssistantToolCalls { calls },
        content: thought.unwrap_or_default(),
        attachments: Vec::new(),
        usage,
    })
}

/// Adds a generic recoverable result without revealing hidden tool membership.
pub(super) fn add_unavailable_result(
    proposal: &AgentToolProposal,
    position: usize,
    prepared: &mut PreparedToolCalls,
) -> Result<(), GraphError> {
    let value = Value::object([("error", Value::from("tool unavailable for this turn"))])
        .map_err(|err| GraphError::Invalid(err.to_string()))?;
    prepared.recoverable_messages.push(Message::tool_output(
        proposal.call_id().to_owned(),
        value.to_string(),
    ));
    prepared.results.push(EdgeCompletedToolCall {
        position,
        result: AgentToolResult::new(
            proposal.call_id().to_owned(),
            proposal.tool_name().to_owned(),
            proposal.arguments().clone(),
            value,
            true,
        ),
    });
    Ok(())
}

/// Adds a recoverable argument-decoding failure to the accepted result batch.
pub(super) fn add_decode_error_result(
    proposal: &AgentToolProposal,
    position: usize,
    err: ToolError,
    prepared: &mut PreparedToolCalls,
) -> Result<(), GraphError> {
    let error_json = err.to_json(proposal.tool_name());
    let value = to_value(error_json).map_err(|encode| GraphError::ValueConversion {
        target: "recoverable tool error".into(),
        reason: encode.to_string(),
    })?;
    prepared.recoverable_messages.push(
        err.into_error_message(proposal.tool_name())
            .with_call_id(proposal.call_id().to_owned()),
    );
    prepared.results.push(EdgeCompletedToolCall {
        position,
        result: AgentToolResult::new(
            proposal.call_id().to_owned(),
            proposal.tool_name().to_owned(),
            proposal.arguments().clone(),
            value,
            true,
        ),
    });
    Ok(())
}

/// Schedules one call per tool and queues repeated calls in proposal order.
pub(super) fn add_executable_call(
    tool: &AgentToolPayload,
    proposal: &AgentToolProposal,
    position: usize,
    input: Value,
    prepared: &mut PreparedToolCalls,
) {
    if prepared.running_tools.insert(tool.name.clone()) {
        prepared.child_calls.push(ContinuationChildCall {
            child_index: tool.child_index,
            call_id: proposal.call_id().to_owned(),
            input,
        });
        prepared.active.push(EdgeActiveToolCall {
            position,
            call_id: proposal.call_id().to_owned(),
            tool_name: tool.name.clone(),
            child_index: tool.child_index,
            args: proposal.arguments().clone(),
        });
    } else {
        prepared.waiting.push(EdgeWaitingToolCall {
            position,
            call_id: proposal.call_id().to_owned(),
            tool_name: tool.name.clone(),
            child_index: tool.child_index,
            args: proposal.arguments().clone(),
            input,
        });
    }
}

/// Removes the active call matching one returned continuation child identity.
pub(super) fn take_active_call(
    payload: &AgentPayload,
    checkpoint: &mut EdgeAgentCheckpoint,
    call_id: &str,
) -> Result<EdgeActiveToolCall, GraphError> {
    let EdgeAgentPhase::PendingTool { active, .. } = &mut checkpoint.phase else {
        return Err(GraphError::Invalid(format!(
            "agent '{}' received a child result outside a tool round",
            payload.agent_id
        )));
    };
    let position = active
        .iter()
        .position(|call| call.call_id == call_id)
        .ok_or_else(|| {
            GraphError::Invalid(format!(
                "agent '{}' received unknown tool child result '{call_id}'",
                payload.agent_id
            ))
        })?;
    Ok(active.remove(position))
}

/// Converts a recoverable output-rendering failure into controller-visible data.
pub(super) fn recoverable_rendered_result(
    err: ToolError,
    tool_name: &str,
) -> Result<EdgeRenderedToolResult, GraphError> {
    let value = to_value(err.to_json(tool_name)).map_err(|encode| GraphError::ValueConversion {
        target: "recoverable tool output error".into(),
        reason: encode.to_string(),
    })?;
    Ok(EdgeRenderedToolResult {
        message: err.into_error_message(tool_name),
        value,
        error: true,
    })
}

/// Records one completed call and prepares the next queued call for that tool.
pub(super) fn complete_active_call(
    checkpoint: &mut EdgeAgentCheckpoint,
    active_call: EdgeActiveToolCall,
    result: AgentToolResult,
) -> Result<Option<(EdgeActiveToolCall, ContinuationChildCall)>, GraphError> {
    let EdgeAgentPhase::PendingTool {
        waiting, results, ..
    } = &mut checkpoint.phase
    else {
        return Err(GraphError::Invalid("agent tool phase disappeared".into()));
    };
    results.push(EdgeCompletedToolCall {
        position: active_call.position,
        result,
    });
    let Some(position) = waiting
        .iter()
        .position(|call| call.tool_name == active_call.tool_name)
    else {
        return Ok(None);
    };
    let next = waiting.remove(position);
    let child = ContinuationChildCall {
        child_index: next.child_index,
        call_id: next.call_id.clone(),
        input: next.input,
    };
    let active = EdgeActiveToolCall {
        position: next.position,
        call_id: next.call_id,
        tool_name: next.tool_name,
        child_index: next.child_index,
        args: next.args,
    };
    Ok(Some((active, child)))
}

/// Adds a newly scheduled queued call to the active checkpoint set.
pub(super) fn add_active_call(
    checkpoint: &mut EdgeAgentCheckpoint,
    active_call: EdgeActiveToolCall,
) -> Result<(), GraphError> {
    let EdgeAgentPhase::PendingTool { active, .. } = &mut checkpoint.phase else {
        return Err(GraphError::Invalid("agent tool phase disappeared".into()));
    };
    active.push(active_call);
    Ok(())
}

/// Sorts a completed batch back into original model proposal order.
pub(super) fn finish_results(results: &mut Vec<EdgeCompletedToolCall>) -> Vec<AgentToolResult> {
    results.sort_by_key(|result| result.position);
    std::mem::take(results)
        .into_iter()
        .map(|result| result.result)
        .collect()
}
