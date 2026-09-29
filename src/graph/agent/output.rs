use std::sync::Arc;

use super::{GraphError, JsonValue, Value, from_value};

/// Compiles the authored agent schema once per prepared node, never per invocation or response.
pub(crate) fn prepare_output_validator(
    payload: &Value,
) -> Result<Option<Arc<jsonschema::Validator>>, GraphError> {
    if payload.get("agent_id").is_none() {
        return Ok(None);
    }
    let Some(schema) = payload.get("output_schema") else {
        return Ok(None);
    };
    let schema: JsonValue = from_value(schema.clone())
        .map_err(|error| GraphError::GraphValidation(format!("invalid output schema: {error}")))?;
    jsonschema::validator_for(&schema)
        .map(|validator| Some(Arc::new(validator)))
        .map_err(|error| {
            GraphError::GraphValidation(format!("invalid agent output schema: {error}"))
        })
}
