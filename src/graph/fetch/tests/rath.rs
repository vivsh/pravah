use super::*;
use crate::clients::{
    CacheControl, ClientOutput, Provider, ResponseFormat, ThinkingLevel, ToolCall, ToolChoice,
    ToolDefinition,
};

/// All construction fields survive direct Value, JSON and CBOR codecs without normalization.
#[test]
fn option_and_message_fidelity() -> Result<(), GraphError> {
    let mut options = ClientOptions::default()
        .with_max_output_tokens(1234)
        .with_name("label")
        .with_preamble("instructions")
        .with_thinking(Some(ThinkingLevel::Off))
        .with_tool_choice(ToolChoice::Required)
        .with_cache(CacheControl::Ephemeral1h)
        .with_input_schema(serde_json::json!({"type":"string"}))
        .with_output_schema(serde_json::json!({"type":"object"}))
        .with_temperature(0.4)
        .with_provider_config(serde_json::json!({"custom":true}));
    options.tools.push(ToolDefinition::new(
        "search".into(),
        "Find things".into(),
        serde_json::json!({"type":"object"}),
    ));
    let message = Message::user("hello")
        .with_key("message-1")
        .with_inline("image/png", [1, 2, 3]);
    let request = RathRequest::new("openai:///test?temperature=0.1", options, vec![message]);
    let expected = to_value(&request).map_err(convert)?;
    let fetch = request.into_fetch_request()?;
    let restored = RathRequest::from_fetch_request(&fetch)?;
    assert_eq!(expected, to_value(&restored).map_err(convert)?);
    assert_eq!(restored.options().temperature, Some(0.4));
    assert_eq!(restored.options().thinking, Some(ThinkingLevel::Off));
    assert_eq!(
        restored
            .messages()
            .first()
            .and_then(|message| message.key.as_deref()),
        Some("message-1")
    );
    let json = serde_json::to_vec(&restored).map_err(convert)?;
    let decoded: RathRequest = serde_json::from_slice(&json).map_err(convert)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&decoded, &mut cbor).map_err(convert)?;
    let decoded: RathRequest = ciborium::from_reader(cbor.as_slice()).map_err(convert)?;
    assert_eq!(expected, to_value(decoded).map_err(convert)?);
    Ok(())
}

/// Rath output preserves provider metadata and Gemini continuation signatures.
#[test]
fn response_preserves_thought_signatures() -> Result<(), GraphError> {
    let call = ToolCall::new(
        "call-1".into(),
        "search".into(),
        serde_json::json!({"q":"hello"}),
    )
    .with_thought_signatures(vec!["opaque-signature".into()]);
    let response = ClientResponse::new(
        Provider::Gemini,
        ClientOutput::ToolCalls {
            text: Some("thinking".into()),
            calls: vec![call],
        },
    )
    .with_provider_model(Some("gemini-test".into()))
    .with_raw_metadata(Some(serde_json::json!({"finish_reason":"tools"})));
    let response = RathResponse::new(response);
    let expected = to_value(&response).map_err(convert)?;
    let response = RathResponse::from_fetch_response(&response.into_fetch_response()?)?;
    assert_eq!(to_value(response).map_err(convert)?, expected);
    Ok(())
}

/// Unset defaults stay unset; incompatible protocol versions fail before client creation.
#[test]
fn defaults_and_version_rejection() -> Result<(), GraphError> {
    let request = RathRequest::new("openai:///test", ClientOptions::default(), Vec::new());
    let restored = RathRequest::from_fetch_request(&request.into_fetch_request()?)?;
    assert_eq!(restored.options().thinking, None);
    assert_eq!(restored.options().cache, None);
    assert_eq!(restored.options().response_format, ResponseFormat::Text);
    let mut invalid = restored;
    invalid.version = 0;
    assert!(matches!(
        invalid.into_fetch_request(),
        Err(GraphError::UnsupportedVersion { .. })
    ));
    Ok(())
}

fn convert(error: impl std::fmt::Display) -> GraphError {
    GraphError::ValueConversion {
        target: "test codec".into(),
        reason: error.to_string(),
    }
}
