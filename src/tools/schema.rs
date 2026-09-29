use schemars::JsonSchema;

use crate::clients::ToolDefinition;
mod normalize;
use normalize::sanitize_strict;

use super::base::pascal_to_snake;

/// Builds the canonical model-facing definition for a typed tool input.
pub(crate) fn tool_definition<T: JsonSchema>() -> Result<ToolDefinition, String> {
    let raw = serde_json::to_value(schemars::SchemaGenerator::default().root_schema_for::<T>())
        .map_err(|error| format!("schema serialization failed: {error}"))?;
    let description = raw
        .get("description")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_owned();
    Ok(ToolDefinition::new(
        pascal_to_snake(&T::schema_name()),
        description,
        sanitize_strict(raw)?,
    ))
}
