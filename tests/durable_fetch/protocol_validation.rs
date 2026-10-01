use super::*;
use pravah::clients::{
    CacheControl, Role, ThinkingLevel, TokenUsage, ToolCall, ToolChoice, ToolDefinition,
};
use pravah::graph::{
    FetchBody,
    fetch::rath::{RathRequest, RathResponse},
    to_value,
};
use serde_json::{Value as Json, json};

/// Exercises every current option plus nested roles, attachment forms and usage.
fn request() -> Result<FetchRequest, GraphError> {
    let options = ClientOptions::default()
        .with_name("name")
        .with_preamble("instructions")
        .with_temperature(0.4)
        .with_thinking(Some(ThinkingLevel::High))
        .with_cache(CacheControl::Ephemeral1h)
        .with_max_output_tokens(42)
        .with_tool_choice(ToolChoice::Required)
        .with_input_schema(json!({"type":"string"}))
        .with_output_schema(json!({"type":"string"}))
        .with_provider_config(json!({"opaque":[null, true, 1.5, u64::MAX]}))
        .with_tools(vec![ToolDefinition::new(
            "tool".into(),
            "description".into(),
            json!({"type":"object"}),
        )]);
    RathRequest::new(
        "openai:///recorded",
        options,
        vec![
            Message::user("input")
                .with_key("key")
                .with_inline("image/png", [1, 2, 3])
                .with_file("text/plain", "file.txt")
                .with_url("image/png", "https://example.invalid"),
            Message::new(
                Role::AssistantToolCalls {
                    calls: vec![call()],
                },
                "thinking".into(),
            ),
            Message::tool_output("call".into(), "result")
                .with_usage(TokenUsage::new().with_input(1).with_output(2)),
        ],
    )
    .into_fetch_request()
}

fn call() -> ToolCall {
    ToolCall::new("call".into(), "tool".into(), json!({"query":"question"}))
        .with_thought_signatures(vec!["signature".into()])
}

/// Includes optional diagnostics so their types participate in mutation coverage.
fn response(output: ClientOutput) -> Result<FetchResponse, GraphError> {
    RathResponse::new(
        ClientResponse::new(Provider::Gemini, output)
            .with_usage(Some(TokenUsage::new().with_input(10).with_output(20)))
            .with_provider_model(Some("model".into()))
            .with_raw_metadata(Some(json!({"raw":[true, "text"]}))),
    )
    .into_fetch_response()
}

/// The shared decoder preserves options, attachments, keys, usage and thought signatures in both codecs.
#[test]
fn rath_codecs_preserve_full_protocol_values() -> Result<(), GraphError> {
    let request = RathRequest::from_fetch_request(&request()?)?;
    let expected = to_value(&request).map_err(codec)?;
    let json: RathRequest =
        serde_json::from_slice(&serde_json::to_vec(&request).map_err(codec)?).map_err(codec)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&json, &mut cbor).map_err(codec)?;
    let restored: RathRequest = ciborium::from_reader(cbor.as_slice()).map_err(codec)?;
    assert_eq!(to_value(restored).map_err(codec)?, expected);
    let response = RathResponse::from_fetch_response(&response(ClientOutput::ToolCalls {
        text: Some("thinking".into()),
        calls: vec![call()],
    })?)?;
    let expected = to_value(&response).map_err(codec)?;
    let json: RathResponse =
        serde_json::from_slice(&serde_json::to_vec(&response).map_err(codec)?).map_err(codec)?;
    cbor.clear();
    ciborium::into_writer(&json, &mut cbor).map_err(codec)?;
    let restored: RathResponse = ciborium::from_reader(cbor.as_slice()).map_err(codec)?;
    assert_eq!(to_value(restored).map_err(codec)?, expected);
    Ok(())
}

/// Unsupported protocol versions retain structured errors rather than becoming generic decode failures.
#[test]
fn obsolete_rath_versions_remain_explicit() -> Result<(), GraphError> {
    let workflow = compile(direct_fetch)?;
    let request = request()?;
    let mut old_request = encoded(request.body_ref())?;
    old_request["version"] = json!(0);
    let old_request = request
        .clone()
        .body(FetchBody::Value(to_value(old_request).map_err(codec)?));
    let mut runtime = workflow.start(old_request, Uuid::nil())?;
    assert!(matches!(
        runtime.next(),
        Err(GraphError::UnsupportedVersion {
            format: "Rath request",
            got: 0,
            ..
        })
    ));
    let mut runtime = workflow.start(request, Uuid::nil())?;
    let fetch = next_fetch(&mut runtime)?;
    let response = response(ClientOutput::Output(json!("answer")))?;
    let mut old_response = encoded(response.body_ref())?;
    old_response["version"] = json!(0);
    let response =
        FetchResponse::new(200).body(FetchBody::Value(to_value(old_response).map_err(codec)?));
    assert!(matches!(
        runtime.resume_fetch(fetch.id(), Ok(response)),
        Err(GraphError::UnsupportedVersion {
            format: "Rath response",
            got: 0,
            ..
        })
    ));
    Ok(())
}

fn encoded(body: Option<&FetchBody>) -> Result<Json, GraphError> {
    match body {
        Some(FetchBody::Value(value)) => serde_json::to_value(value).map_err(codec),
        _ => Err(codec("expected structured body")),
    }
}

/// Installation uses exactly the owned codec's field, enum, numeric and attachment acceptance rules.
#[test]
fn request_validation_matches_owned_decoding() -> Result<(), GraphError> {
    let workflow = compile(direct_fetch)?;
    let request = request()?;
    for candidate in variants(&encoded(request.body_ref())?) {
        let request = request.clone().body(FetchBody::Value(
            to_value(candidate.clone()).map_err(codec)?,
        ));
        let expected = RathRequest::from_fetch_request(&request).is_ok();
        let mut runtime = workflow.start(request, Uuid::nil())?;
        let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
        assert_eq!(runtime.next().is_ok(), expected, "{candidate}");
        if !expected {
            assert_eq!(
                before,
                serde_json::to_value(runtime.snapshot()?).map_err(codec)?
            );
        }
    }
    Ok(())
}

/// Delivery validates complete outputs/metadata without changing rejected waits or source rules.
#[test]
fn response_validation_matches_owned_decoding() -> Result<(), GraphError> {
    let workflow = compile(direct_fetch)?;
    for output in [
        ClientOutput::Output(json!({"answer":[1,true,null]})),
        ClientOutput::ToolCalls {
            text: Some("thinking".into()),
            calls: vec![call()],
        },
    ] {
        let response = response(output)?;
        let mut runtime = workflow.start(request()?, Uuid::nil())?;
        let fetch = next_fetch(&mut runtime)?;
        let snapshot = runtime.snapshot()?;
        for candidate in variants(&encoded(response.body_ref())?) {
            let response = FetchResponse::new(200).body(FetchBody::Value(
                to_value(candidate.clone()).map_err(codec)?,
            ));
            let expected = RathResponse::from_fetch_response(&response).is_ok();
            let mut runtime = workflow.restore(snapshot.clone())?;
            assert_eq!(
                runtime.resume_fetch(fetch.id(), Ok(response)).is_ok(),
                expected,
                "{candidate}"
            );
            if !expected {
                assert_eq!(
                    serde_json::to_value(&snapshot).map_err(codec)?,
                    serde_json::to_value(runtime.snapshot()?).map_err(codec)?
                );
            }
        }
    }
    Ok(())
}

/// Generates malformed and valid field variations, including missing and unknown nested fields.
fn variants(source: &Json) -> Vec<Json> {
    let mut paths = Vec::new();
    paths_of(source, "", &mut paths);
    let mut variants = vec![source.clone()];
    for path in paths {
        for replacement in [
            Json::Null,
            json!(true),
            json!(-1),
            json!(u64::MAX),
            json!(1.5),
            json!(""),
            json!([]),
            json!({}),
        ] {
            let mut value = source.clone();
            if let Some(field) = value.pointer_mut(&path) {
                *field = replacement;
                variants.push(value);
            }
        }
        if let Some(object) = source.pointer(&path).and_then(Json::as_object) {
            let mut changed = source.clone();
            if let Some(fields) = changed.pointer_mut(&path).and_then(Json::as_object_mut) {
                fields.insert("unexpected".into(), json!(true));
                variants.push(changed);
            }
            for key in object.keys() {
                let mut changed = source.clone();
                if let Some(fields) = changed.pointer_mut(&path).and_then(Json::as_object_mut) {
                    fields.remove(key);
                    variants.push(changed);
                }
            }
        }
    }
    variants
}

/// Enumerates stable JSON pointers for protocol shapes without depending on fixture field order.
fn paths_of(value: &Json, path: &str, paths: &mut Vec<String>) {
    paths.push(path.into());
    match value {
        Json::Object(fields) => {
            for (key, value) in fields {
                let key = key.replace('~', "~0").replace('/', "~1");
                paths_of(value, &format!("{path}/{key}"), paths);
            }
        }
        Json::Array(values) => {
            for (i, value) in values.iter().enumerate() {
                paths_of(value, &format!("{path}/{i}"), paths);
            }
        }
        _ => {}
    }
}

/// Validating opaque provider configuration does not allocate a second JSON tree.
#[test]
fn request_validation_does_not_copy_provider_json() -> Result<(), GraphError> {
    let mut measurements = Vec::new();
    for size in [1, 1000] {
        let options =
            ClientOptions::default().with_provider_config(json!({"data": vec!["payload"; size]}));
        let request =
            RathRequest::new("openai:///test", options, Vec::new()).into_fetch_request()?;
        let mut graph = compile(direct_fetch)?.graph().clone();
        graph.nodes.first_mut().ok_or_else(|| codec("node"))?.kind =
            pravah::graph::NodeKind::Continuation {
                key: pravah::graph::HandlerKey::new("request"),
                payload: Value::NULL,
                children: vec![],
            };
        let mut registry = pravah::graph::HandlerRegistry::new();
        registry.insert_continuation("request", RequestFixture(request))?;
        let prepared = pravah::graph::PreparedGraph::new(graph, registry)?;
        let input = to_value(FetchRequest::new("GET", "task://unused")).map_err(codec)?;
        let mut runtime = prepared.start(input, Uuid::nil())?;
        let mut outcome = Ok(Step::Continue);
        let count = allocation_counter::measure(|| outcome = runtime.next());
        assert!(matches!(outcome?, Step::Fetch(_)));
        measurements.push((count.count_total, count.bytes_total));
    }
    assert_eq!(measurements.first(), measurements.last());
    Ok(())
}

struct RequestFixture(FetchRequest);

impl ContinuationHandler for RequestFixture {
    /// Shares an already encoded request, isolating protocol validation from Fetch-node input decoding.
    fn start<'a>(
        &'a self,
        _: &'a Value,
        _: Option<Value>,
        _: Vec<Value>,
        _: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        Ok(ContinuationTransition {
            checkpoint: Some(true.into()),
            fetch: Some(self.0.clone()),
            ..Default::default()
        })
    }

    fn advance<'a>(
        &'a self,
        _: &'a Value,
        _: Value,
        _: ContinuationEvent,
        _: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        Err(codec("fixture does not execute responses"))
    }
}

/// Response metadata is validated without a discarded owned JSON tree before inbox acceptance.
#[tokio::test]
async fn response_validation_does_not_copy_metadata() -> Result<(), GraphError> {
    let mut measurements = Vec::new();
    for size in [1, 1000] {
        let mut chat = builder().build(Context::default())?;
        chat.submit("question")?;
        let fetch = super::preparation::pending(&mut chat).await?;
        assert_eq!(fetch.request().url(), "rath://generate");
        let response = RathResponse::new(
            ClientResponse::new(Provider::OpenAi, ClientOutput::Output(json!("answer")))
                .with_raw_metadata(Some(json!({"data": vec!["payload"; size]}))),
        )
        .into_fetch_response()?;
        let mut outcome = Ok(());
        let count =
            allocation_counter::measure(|| outcome = chat.resume_fetch(fetch.id(), Ok(response)));
        outcome?;
        measurements.push((count.count_total, count.bytes_total));
    }
    assert_eq!(measurements.first(), measurements.last());
    Ok(())
}
