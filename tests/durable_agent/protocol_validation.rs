use super::*;
use serde_json::{Value as Json, json};

/// All response fields retain normalized Rath metadata through both snapshot codecs.
#[tokio::test]
async fn response_protocol_and_codec_fidelity() -> Result<(), GraphError> {
    let factory = pravah::testing::ScriptedFactory::new().then(
        ClientResponse::new(
            Provider::Gemini,
            ClientOutput::ToolCalls {
                text: Some("thinking".into()),
                calls: vec![
                    pravah::clients::ToolCall::new(
                        "call".into(),
                        "unknown".into(),
                        json!({"query":"question"}),
                    )
                    .with_thought_signatures(vec!["signature".into()]),
                ],
            },
        )
        .with_usage(Some(
            pravah::clients::TokenUsage::new()
                .with_input(10)
                .with_output(20),
        ))
        .with_provider_model(Some("model".into()))
        .with_raw_metadata(Some(json!({"raw":[true,"text"]}))),
    );
    let mut chat = builder().build(
        Context::default().with_providers(ProviderRegistry::with_builtin_factory(factory)),
    )?;
    chat.submit("question")?;
    let request = preparation::pending(&mut chat).await?;
    let response = chat.executor().execute(&request).await;
    let expected = serde_json::to_value(&response).map_err(codec)?;
    assert_eq!(expected["outcome"]["Ok"]["provider_model"], "model");
    assert_eq!(expected["outcome"]["Ok"]["usage"]["output"], 20);
    assert_eq!(
        expected["outcome"]["Ok"]["output"]["calls"][0]["thought_signatures"][0],
        "signature"
    );
    let restored: AgentResponse = serde_json::from_value(expected.clone()).map_err(codec)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&restored, &mut cbor).map_err(codec)?;
    let restored: AgentResponse = ciborium::from_reader(cbor.as_slice()).map_err(codec)?;
    assert_eq!(serde_json::to_value(restored).map_err(codec)?, expected);
    Ok(())
}

/// Malformed normalized responses fail atomically, including unknown fields and invalid nested types.
#[tokio::test]
async fn response_validation_matches_protocol_decoding() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().build(context(&calls))?;
    chat.submit("question")?;
    let request = preparation::pending(&mut chat).await?;
    let response = chat.executor().execute(&request).await;
    let wire = serde_json::to_value(&response).map_err(codec)?;
    let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    for (path, value) in [
        ("/outcome/Ok/provider", json!(false)),
        ("/outcome/Ok/provider_model", json!(42)),
        ("/outcome/Ok/usage", json!({"input":-1})),
        ("/outcome/Ok/output", json!({"kind":"unknown"})),
        (
            "/outcome/Ok/output",
            json!({"kind":"output","value":"answer","extra":null}),
        ),
    ] {
        let mut invalid = wire.clone();
        *invalid.pointer_mut(path).ok_or_else(|| codec(path))? = value;
        let response = serde_json::from_value(invalid).map_err(codec)?;
        assert!(chat.resume_agent(response).is_err(), "{path}");
        assert_eq!(
            before,
            serde_json::to_value(chat.snapshot()?).map_err(codec)?
        );
    }
    chat.resume_agent(response)?;
    Ok(())
}

/// Protocol versions and closed operation fields reject obsolete or ambiguous requests.
#[test]
fn obsolete_and_unknown_request_formats_are_rejected() -> Result<(), GraphError> {
    let workflow = compile(agent_flow)?;
    let mut runtime = workflow.start("question".into(), Uuid::nil())?;
    let request = next_agent(&mut runtime)?;
    let wire = serde_json::to_value(request).map_err(codec)?;
    for (key, value) in [
        ("version", json!(0)),
        ("operation", json!("unknown")),
        ("input", Json::Null),
    ] {
        let mut invalid = wire.clone();
        invalid[key] = value;
        if key == "input" {
            continue;
        } // Input semantics remain the configured handler's typed boundary.
        assert!(serde_json::from_value::<AgentRequest>(invalid).is_err());
    }
    let mut invalid = wire;
    invalid["unexpected"] = true.into();
    assert!(serde_json::from_value::<AgentRequest>(invalid).is_err());
    Ok(())
}

/// Response validation borrows diagnostic JSON subtrees instead of allocating their contents.
#[tokio::test]
async fn response_validation_does_not_copy_metadata() -> Result<(), GraphError> {
    let mut measured = Vec::new();
    for count in [1, 1000] {
        let factory = pravah::testing::ScriptedFactory::new().then(
            ClientResponse::new(Provider::OpenAi, ClientOutput::Output(json!("answer")))
                .with_raw_metadata(Some(json!({"data": vec!["value".repeat(100); count]}))),
        );
        let mut chat = builder().build(
            Context::default().with_providers(ProviderRegistry::with_builtin_factory(factory)),
        )?;
        chat.submit("question")?;
        let request = preparation::pending(&mut chat).await?;
        let response = chat.executor().execute(&request).await;
        let mut result = Ok(());
        let counts = allocation_counter::measure(|| result = chat.resume_agent(response));
        result?;
        measured.push((counts.count_total, counts.bytes_total));
    }
    assert_eq!(measured[0], measured[1]);
    Ok(())
}
