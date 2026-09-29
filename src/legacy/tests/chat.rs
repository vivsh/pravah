use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
struct DraftInput {
    topic: String,
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
struct DraftOutput {
    answer: String,
}

/// String chat stays in plain-text mode and sends unquoted content.
#[test]
fn string_chat_type_stays_text() {
    assert_eq!(String::wire_kind(), ChatWireKind::Text);
    assert!(String::schema().unwrap().is_none());
    assert_eq!("hello", "hello".to_string().encode_input().unwrap());
}

/// Non-string chat types use JSON mode and derive a schema.
#[test]
fn json_chat_type_uses_json_wire_format() {
    let input = DraftInput {
        topic: "ownership".into(),
    };
    assert_eq!(DraftInput::wire_kind(), ChatWireKind::Json);
    assert!(DraftInput::schema().unwrap().is_some());
    assert_eq!(input.encode_input().unwrap(), r#"{"topic":"ownership"}"#);
}

/// Typed options use the output type to decide provider response mode.
#[test]
fn typed_options_use_output_type_for_response_mode() {
    let text = build_typed_options::<DraftInput, String>(ClientOptions::default()).unwrap();
    assert!(text.input_schema.is_some());
    assert!(matches!(text.response_format, ResponseFormat::Text));
    assert_eq!(text.response_format, ResponseFormat::Text);

    let json = build_typed_options::<String, DraftOutput>(ClientOptions::default()).unwrap();
    assert!(json.input_schema.is_none());
    assert!(matches!(
        json.response_format,
        ResponseFormat::JsonSchema { .. }
    ));
    if let ResponseFormat::JsonSchema { schema } = json.response_format {
        assert_eq!(Some(schema), DraftOutput::schema().unwrap());
    }
}

/// Text output rejects structured provider values.
#[test]
fn text_output_rejects_structured_values() {
    let err = String::decode_output(serde_json::json!({ "answer": "ok" })).unwrap_err();
    assert!(matches!(err, ChatError::UnexpectedOutput));
}

/// JSON output decodes from the provider value into the typed result.
#[test]
fn json_output_decodes_from_value() {
    let output = DraftOutput::decode_output(serde_json::json!({ "answer": "ok" })).unwrap();
    assert_eq!(
        output,
        DraftOutput {
            answer: "ok".into(),
        }
    );
}
