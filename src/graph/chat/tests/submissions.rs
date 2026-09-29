use super::super::ChatRequest;
use crate::graph::{Value, ValueError, from_value, to_value};

/// Envelope decoding agrees with Serde for accepted and malformed submissions.
#[test]
fn submission_checks_match_serde() -> Result<(), ValueError> {
    for json in [
        serde_json::json!({"input":null}),
        serde_json::json!({"input":[1, 2], "key": null}),
        serde_json::json!({"input":"hello", "key": "message-1"}),
        serde_json::json!({"input":"hello", "key": 42}),
        serde_json::json!({"input":"hello", "extra":true}),
        serde_json::json!({"input":"hello", "memory":42}),
        serde_json::json!({"input":"hello", "tools":[42]}),
        serde_json::json!({"input":"hello", "resources":[42]}),
        serde_json::json!({"memory":null}),
        serde_json::json!({"input": null}),
        serde_json::json!({"key": "message-1"}),
        serde_json::json!([]),
    ] {
        let value = to_value(json)?;
        let decoded = ChatRequest::decode(&value);
        let serde = from_value::<ChatRequest<Value>>(value);
        assert_eq!(decoded.is_ok(), serde.is_ok());
        if let (Ok(decoded), Ok(serde)) = (decoded, serde) {
            assert_eq!(decoded, serde);
        }
    }
    Ok(())
}

/// Extracting a large shared submission allocates no recursive copy of its input.
#[test]
fn submission_input_remains_shared() -> Result<(), ValueError> {
    let input = Value::array([Value::from("large".repeat(200_000))]);
    let value = Value::object([("input", input.clone()), ("key", Value::NULL)])?;
    let mut decoded = None;
    let allocations = allocation_counter::measure(|| {
        decoded = Some(ChatRequest::decode(&value));
    });
    let decoded = decoded
        .transpose()?
        .ok_or_else(|| ValueError::Unsupported("missing result".into()))?;
    assert_eq!(allocations.count_total, 0);
    assert!(std::ptr::eq(
        decoded
            .input
            .as_array()
            .ok_or_else(|| ValueError::Unsupported("not array".into()))?,
        input
            .as_array()
            .ok_or_else(|| ValueError::Unsupported("not array".into()))?
    ));
    Ok(())
}
