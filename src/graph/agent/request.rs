//! Provider request construction from the checkpoint's authoritative shared values.

use super::{AgentPayloadView, EdgeAgentCheckpoint, GraphError, Value, effective_tools};
use crate::graph::fetch::rath::RathRequest;

/// Freezes the active tool surface and options without rebuilding immutable JSON trees.
pub(super) fn generation(
    payload: &AgentPayloadView<'_>,
    checkpoint: &EdgeAgentCheckpoint,
    conclude: bool,
) -> Result<Value, GraphError> {
    let configured = checkpoint.configured_tools()?;
    let selected = effective_tools(
        &configured,
        &checkpoint.selected_tools,
        checkpoint.budget.as_ref(),
    );
    let mut tools = Vec::new();
    for tool in field(payload.raw, "tools")?
        .as_array()
        .ok_or_else(invalid)?
    {
        if !conclude
            && selected
                .iter()
                .any(|name| Some(name.as_str()) == tool.get("name").and_then(Value::as_str))
        {
            tools.push(Value::array([
                field(tool, "name")?.clone(),
                field(tool, "description")?.clone(),
                field(tool, "parameters")?.clone(),
            ]));
        }
    }
    let resolved = &checkpoint.resolved;
    RathRequest::agent_value(
        field(resolved, "model")?.clone(),
        payload.agent_id,
        preamble(resolved)?,
        tools,
        payload.output_schema().clone(),
        resolved
            .get("provider_config")
            .cloned()
            .unwrap_or(Value::NULL),
        resolved
            .get("max_output_tokens")
            .cloned()
            .unwrap_or(Value::NULL),
    )
}

/// Renders exactly the existing section order and separators into one operation-local string.
fn preamble(resolved: &Value) -> Result<String, GraphError> {
    let mut text = text_field(resolved, "instructions")?.to_owned();
    if let Some(memory) = resolved.get("memory").filter(|value| !value.is_null()) {
        separate(&mut text);
        text.push_str("<memory>\n");
        text.push_str(memory.as_str().ok_or_else(invalid)?);
        text.push_str("\n</memory>");
    }
    for resource in field(resolved, "resources")?
        .as_array()
        .ok_or_else(invalid)?
    {
        separate(&mut text);
        text.push_str("<resource server=\"");
        text.push_str(text_field(resource, "server")?);
        text.push_str("\" uri=\"");
        text.push_str(text_field(resource, "uri")?);
        text.push_str("\">\n");
        text.push_str(text_field(resource, "text")?);
        text.push_str("\n</resource>");
    }
    Ok(text)
}

fn separate(text: &mut String) {
    if !text.is_empty() {
        text.push_str("\n\n");
    }
}

fn invalid() -> GraphError {
    GraphError::SnapshotValidation("invalid agent request fields".into())
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, GraphError> {
    value.get(name).ok_or_else(invalid)
}

fn text_field<'a>(value: &'a Value, name: &str) -> Result<&'a str, GraphError> {
    field(value, name)?.as_str().ok_or_else(invalid)
}
