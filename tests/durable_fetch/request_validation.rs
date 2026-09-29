use super::*;
use pravah::graph::{
    FetchBody,
    fetch::rath::{RathRequest, RathResponse},
};

fn request(size: usize) -> Result<FetchRequest, GraphError> {
    RathRequest::new(
        "openai:///test",
        ClientOptions::default(),
        vec![Message::user("x".repeat(size))],
    )
    .into_fetch_request()
}

fn response() -> Result<FetchResponse, GraphError> {
    RathResponse::new(ClientResponse::new(
        Provider::OpenAi,
        ClientOutput::Output(serde_json::json!("answer")),
    ))
    .into_fetch_response()
}

/// Malformed requests fail before installation or sequence changes, for ordinary Fetch nodes too.
#[test]
fn request_installation_is_validated_atomically() -> Result<(), GraphError> {
    let workflow = compile(direct_fetch)?;
    for input in [
        FetchRequest::new("GET", "rath://generate"),
        FetchRequest::new("POST", "rath://generate"),
        request(1)?.header("unexpected", b"header"),
        request(1)?.body(FetchBody::Value(Value::NULL)),
    ] {
        let mut runtime = workflow.start(input, Uuid::nil())?;
        let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
        assert!(runtime.next().is_err());
        assert!(runtime.pending_fetch().is_none());
        assert_eq!(
            before,
            serde_json::to_value(runtime.snapshot()?).map_err(codec)?
        );
    }
    Ok(())
}

/// Delivery cost is independent of the already validated pending prompt's size.
#[test]
fn response_delivery_does_not_reconstruct_pending_request() -> Result<(), GraphError> {
    let workflow = compile(direct_fetch)?;
    let mut measured = Vec::new();
    for size in [1, 100_000] {
        let mut runtime = workflow.start(request(size)?, Uuid::nil())?;
        let fetch = next_fetch(&mut runtime)?;
        let reply = response()?;
        let mut result = Ok(());
        let count = allocation_counter::measure(|| {
            result = runtime.resume_fetch(fetch.id(), Ok(reply));
        });
        result?;
        measured.push((count.count_total, count.bytes_total));
    }
    assert_eq!(measured.first(), measured.last());
    Ok(())
}

/// An invalid response is rejected atomically; restoration retains the valid pending request.
#[test]
fn response_validation_and_pending_restore_remain_strict() -> Result<(), GraphError> {
    let workflow = compile(direct_fetch)?;
    let mut runtime = workflow.start(request(100)?, Uuid::nil())?;
    let fetch = next_fetch(&mut runtime)?;
    let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    assert!(
        runtime
            .resume_fetch(fetch.id(), Ok(FetchResponse::new(200)))
            .is_err()
    );
    assert_eq!(
        before,
        serde_json::to_value(runtime.snapshot()?).map_err(codec)?
    );
    let mut runtime = workflow.restore(cbor_roundtrip(json_roundtrip(runtime.snapshot()?)?)?)?;
    runtime.resume_fetch(fetch.id(), Ok(response()?))?;
    assert!(runtime.pending_fetch().is_none());
    Ok(())
}

/// Restoration checks immutable pending requests even when no response has arrived yet.
#[test]
fn restore_rejects_malformed_pending_request() -> Result<(), GraphError> {
    let workflow = compile(direct_fetch)?;
    let mut runtime = workflow.start(request(100)?, Uuid::nil())?;
    next_fetch(&mut runtime)?;
    let mut encoded = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    let pending = encoded
        .pointer_mut("/state/waiting/fetch/request/method")
        .ok_or_else(|| GraphError::Invalid("missing pending method".into()))?;
    *pending = serde_json::json!("GET");
    let snapshot = serde_json::from_value(encoded).map_err(codec)?;
    assert!(workflow.restore(snapshot).is_err());
    Ok(())
}
