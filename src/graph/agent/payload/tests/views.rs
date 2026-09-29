use super::super::AgentPayloadView;
use crate::graph::agent::{PAYLOAD_VERSION, support::decode_payload};
use crate::graph::{GraphError, Value, ValueError, to_value};

fn payload() -> serde_json::Value {
    serde_json::json!({
        "version": PAYLOAD_VERSION,
        "agent_id": "reviewer", "configure_handler_key": "reviewer",
        "control_handler_key": "reviewer::control",
        "input_schema": {"type": "string"},
        "output_type_name": "String", "output_schema": {"type": "string"},
        "tools": [],
        "configuration": {"value": [1, 2], "schema": {"type": "array"}}
    })
}

/// Runtime views borrow names/data and preserve all provider-facing metadata.
#[test]
fn view_matches_full_payload() -> Result<(), ValueError> {
    let mut json = payload();
    json["tools"] = serde_json::json!([{
        "name": "search", "child_index": 0, "description": "Search",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
    }]);
    let value = to_value(json)?;
    let view = AgentPayloadView::read(&value).expect("execution view");
    let full = decode_payload(&value).expect("full payload");
    assert_eq!(view.agent_id, full.agent_id);
    assert_eq!(view.output_schema(), &to_value(full.output_schema)?);
    assert_eq!(to_value(&view.tools)?, to_value(&full.tools)?);
    assert!(std::ptr::eq(
        view.agent_id,
        value.get("agent_id").and_then(Value::as_str).expect("name")
    ));
    assert!(std::ptr::eq(
        view.configuration.expect("configuration"),
        value
            .get("configuration")
            .and_then(|data| data.get("value"))
            .expect("value")
    ));
    Ok(())
}

/// Unused schemas, including structured output, do not increase execution-view allocations.
#[test]
fn schema_size_does_not_affect_view_allocations() -> Result<(), ValueError> {
    let small = to_value(payload())?;
    let mut json = payload();
    let schema = serde_json::json!({"enum": vec!["large-schema".repeat(1000); 100]});
    json["input_schema"] = schema.clone();
    json["output_schema"] = schema.clone();
    json["configuration"]["schema"] = schema;
    let large = to_value(json)?;
    let measure = |value: &Value| {
        allocation_counter::measure(|| {
            std::hint::black_box(AgentPayloadView::read(value).expect("valid payload"));
        })
    };
    let small = measure(&small);
    let large = measure(&large);
    assert_eq!(small.count_total, large.count_total);
    assert_eq!(small.bytes_total, large.bytes_total);
    Ok(())
}

/// Both decoding paths reject malformed version, identity and executable metadata.
#[test]
fn invalid_runtime_fields_are_rejected() -> Result<(), ValueError> {
    for (field, invalid) in [
        ("version", serde_json::json!(PAYLOAD_VERSION - 1)),
        ("agent_id", serde_json::json!("")),
        ("configure_handler_key", serde_json::json!("other")),
        ("control_handler_key", serde_json::json!("other::control")),
        ("output_type_name", serde_json::json!(false)),
        ("tools", serde_json::json!([{"name": "incomplete"}])),
    ] {
        let mut json = payload();
        json[field] = invalid;
        let value = to_value(json)?;
        assert!(AgentPayloadView::read(&value).is_err(), "{field}");
        assert!(decode_payload(&value).is_err(), "{field}");
        if field == "version" {
            assert!(matches!(
                AgentPayloadView::read(&value),
                Err(GraphError::UnsupportedVersion { .. })
            ));
        }
    }
    Ok(())
}

/// Boundary decoding still requires metadata deliberately unused by execution views.
#[test]
fn full_decode_retains_definition_checks() -> Result<(), ValueError> {
    let mut json = payload();
    json.as_object_mut().expect("object").remove("input_schema");
    let value = to_value(json)?;
    assert!(decode_payload(&value).is_err());
    assert!(AgentPayloadView::read(&value).is_ok());
    Ok(())
}
