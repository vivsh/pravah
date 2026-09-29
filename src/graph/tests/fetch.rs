use super::*;

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Value(#[from] super::super::ValueError),
    #[error("CBOR encoding failed: {0}")]
    Encode(#[from] ciborium::ser::Error<std::io::Error>),
    #[error("CBOR decoding failed: {0}")]
    Decode(#[from] ciborium::de::Error<std::io::Error>),
}

/// Requests preserve opaque bytes, duplicate headers and shared structured data in both codecs.
#[test]
fn request_roundtrips() -> Result<(), TestError> {
    let request = FetchRequest::new("POST", "task://run")
        .header("x-repeat", vec![0, 255])
        .header("x-repeat", b"second".to_vec())
        .body(FetchBody::Bytes(Arc::from([0, 255, 10])));
    let fetch = Fetch::new(Uuid::nil(), Arc::new(request));
    let json = serde_json::to_vec(&fetch)?;
    assert_eq!(fetch, serde_json::from_slice::<Fetch>(&json)?);
    let mut cbor = Vec::new();
    ciborium::into_writer(&fetch, &mut cbor)?;
    assert_eq!(fetch, ciborium::from_reader::<Fetch, _>(cbor.as_slice())?);
    let value = super::super::to_value(&fetch)?;
    assert_eq!(fetch, super::super::from_value::<Fetch>(value)?);
    Ok(())
}

/// HTTP error status and portable execution failure remain different typed outcomes.
#[test]
fn response_and_failure_are_distinct() -> Result<(), TestError> {
    let response = Ok::<_, FetchError>(FetchResponse::new(503));
    let failure = Err::<FetchResponse, _>(
        FetchError::new("transport", "disconnected")
            .with_details(Value::from("retained diagnostic")),
    );
    for outcome in [response, failure] {
        let encoded = super::super::to_value(&outcome)?;
        assert_eq!(outcome, super::super::from_value(encoded)?);
    }
    Ok(())
}

/// Default diagnostics never include user-supplied URLs, bodies, headers or failure text.
#[test]
fn formatting_omits_request_data() {
    let request = FetchRequest::new("secret-method", "https://secret-url")
        .header("secret-header", b"secret-value".to_vec())
        .body(FetchBody::Value(Value::from("secret-body")));
    let fetch = Fetch::new(Uuid::nil(), Arc::new(request));
    let error = FetchError::new("secret-code", "secret-message");
    assert!(!format!("{fetch:?} {:?} {error:?} {error}", fetch.request()).contains("secret"));
}

/// Cloning a pending request shares its immutable allocation.
#[test]
fn pending_request_clone_is_shared() {
    let fetch = Fetch::new(
        Uuid::nil(),
        Arc::new(FetchRequest::new("GET", "test://resource")),
    );
    assert!(Arc::ptr_eq(&fetch.request, &fetch.clone().request));
}
