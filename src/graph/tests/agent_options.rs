use super::*;
use crate::clients::{CacheControl, ResponseFormat, ThinkingLevel, ToolChoice};

/// Direct option decoding preserves all supported Rath fields without applying provider defaults.
#[test]
fn normalized_options_preserve_every_field() -> Result<(), crate::graph::GraphError> {
    let wire = serde_json::json!({
        "max_output_tokens": 2000, "name": "agent", "preamble": "instructions",
        "tools": [["search", "find sources", {"type":"object"}]],
        "thinking": "high", "tool_choice": "required", "input_schema": {"type":"string"},
        "response_format": {"kind":"json_schema", "schema":{"type":"object"}},
        "temperature": 0.25, "provider_config": {"custom":true}, "cache":"1h"
    });
    let value = crate::graph::to_value(wire.clone()).map_err(codec_error)?;
    let options = options::deserialize(&value).map_err(codec_error)?;
    assert_eq!(options.max_output_tokens, Some(2000));
    assert_eq!(options.name.as_deref(), Some("agent"));
    assert_eq!(options.preamble.as_deref(), Some("instructions"));
    let tool = options
        .tools
        .first()
        .ok_or_else(|| codec_error("missing tool"))?;
    assert_eq!(tool.name, "search");
    assert_eq!(tool.description, "find sources");
    assert_eq!(tool.parameters, wire["tools"][0][2]);
    assert_eq!(options.thinking, Some(ThinkingLevel::High));
    assert_eq!(options.tool_choice, ToolChoice::Required);
    assert_eq!(options.input_schema, Some(wire["input_schema"].clone()));
    assert!(
        matches!(options.response_format, ResponseFormat::JsonSchema { schema } if schema == wire["response_format"]["schema"])
    );
    assert_eq!(options.temperature, Some(0.25));
    assert_eq!(
        options.provider_config,
        Some(wire["provider_config"].clone())
    );
    assert_eq!(options.cache, Some(CacheControl::Ephemeral1h));
    Ok(())
}

/// Borrowed validation and owned decoding reject identical malformed protocol controls.
#[test]
fn normalized_controls_reject_invalid_fields() -> Result<(), crate::graph::GraphError> {
    let base = serde_json::json!({
        "max_output_tokens":null,"name":null,"preamble":null,"tools":[],
        "thinking":null,"tool_choice":"auto","input_schema":null,
        "response_format":{"kind":"text"},"temperature":null,"provider_config":null,"cache":null
    });
    for (field, invalid) in [
        ("tool_choice", "sometimes"),
        ("thinking", "unknown"),
        ("cache", "forever"),
    ] {
        let mut wire = base.clone();
        wire[field] = serde_json::json!(invalid);
        let value = crate::graph::to_value(wire).map_err(codec_error)?;
        let controls: options::Options<serde::de::IgnoredAny> =
            serde::Deserialize::deserialize(&value).map_err(codec_error)?;
        assert!(controls.controls().is_err());
        assert!(options::deserialize(&value).is_err());
    }
    Ok(())
}

fn codec_error(error: impl std::fmt::Display) -> crate::graph::GraphError {
    crate::graph::GraphError::AgentRequestValidation(error.to_string())
}
