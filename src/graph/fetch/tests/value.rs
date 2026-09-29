use super::*;

/// Embedded requests preserve the public Serde shape for every body kind and repeated headers.
#[test]
fn embedding_matches_serde() -> Result<(), GraphError> {
    let values = [
        None,
        Some(FetchBody::Bytes(Arc::from([1, 2, 3]))),
        Some(FetchBody::Value(Value::array([Value::from("nested")]))),
        Some(FetchBody::Value(Value::NULL)),
    ];
    for body in values {
        let mut request = FetchRequest::new("POST", "test://request")
            .header("repeat", [0, 255])
            .header("repeat", [42]);
        request.body = body;
        let expected = to_value(&request).map_err(|_| invalid())?;
        let value = request.clone().into_value()?;
        assert_eq!(value, expected);
        assert_eq!(FetchRequest::from_value(&value)?, request);
        let roundtrip: FetchRequest = from_value(value).map_err(|_| invalid())?;
        assert_eq!(roundtrip, request);
    }
    Ok(())
}

/// Encoding and decoding an embedded structured request share its nested payload allocation.
#[test]
fn embedding_shares_structured_body() -> Result<(), GraphError> {
    let body = Value::array([Value::from("large".repeat(10_000))]);
    let request = FetchRequest::new("POST", "test://request").body(FetchBody::Value(body.clone()));
    let decoded = FetchRequest::from_value(&request.into_value()?)?;
    let Some(FetchBody::Value(retained)) = decoded.body_ref() else {
        return Err(invalid());
    };
    assert!(std::ptr::eq(
        body.as_array().ok_or_else(invalid)?,
        retained.as_array().ok_or_else(invalid)?
    ));
    Ok(())
}

/// Invalid closed-envelope fields and body tags are rejected, not silently interpreted.
#[test]
fn malformed_embedding_is_rejected() -> Result<(), GraphError> {
    for json in [
        serde_json::json!({"method":"POST","url":"test:","headers":[],"extra":0}),
        serde_json::json!({"method":"POST","url":"test:","headers":[],"body":{"kind":"value"}}),
        serde_json::json!({"method":"POST","url":"test:","headers":[],"body":{"kind":"unknown","data":0}}),
    ] {
        assert!(FetchRequest::from_value(&to_value(json).map_err(|_| invalid())?).is_err());
    }
    Ok(())
}
