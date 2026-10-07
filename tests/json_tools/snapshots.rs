use super::*;

/// Pending operation input cannot substitute another value, even one valid under the same schema.
#[tokio::test]
async fn pending_input_corruption_is_rejected() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_tool_calls(vec![staff_call()]);
    let proxy = Arc::new(Proxy::default());
    let ctx = context(&factory, &proxy)?;
    let mut chat = builder(catalogue()).build(ctx.clone())?;
    chat.submit("Find staff 42")?;
    advance_to_tool(&mut chat).await?;
    let snapshot = serde_json::to_value(chat.snapshot()?)?;
    for input in [json!({"staff_id":0}), json!({"staff_id":43})] {
        let mut changed = snapshot.clone();
        changed["state"]["waiting"]["request"]["input"] = input;
        assert!(
            builder(catalogue())
                .restore::<()>(serde_json::from_value(changed)?, ctx.clone())
                .is_err()
        );
    }
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

/// Binary snapshot transport preserves a pending native JSON call and its one-execution boundary.
#[tokio::test]
async fn cbor_snapshot_resumes_json_tool() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_tool_calls(vec![staff_call()]);
    let proxy = Arc::new(Proxy::default());
    let mut chat = builder(catalogue()).build(context(&factory, &proxy)?)?;
    chat.submit("Find staff 42")?;
    advance_to_tool(&mut chat).await?;
    let mut bytes = Vec::new();
    ciborium::into_writer(&chat.snapshot()?, &mut bytes)?;
    let snapshot: Snapshot = ciborium::from_reader(bytes.as_slice())?;
    let next = ScriptedFactory::new().then_output(json!("Done"));
    let mut restored = builder(catalogue()).restore::<()>(snapshot, context(&next, &proxy)?)?;
    assert_eq!(finish(&mut restored).await?, "Done");
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Invalid model proposals remain restorable before admission and become ordinary validation errors.
#[tokio::test]
async fn staged_invalid_arguments_remain_correctable() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![ToolCall::new(
            "bad".into(),
            "find_staff_member".into(),
            json!({"staff_id":0}),
        )])
        .then_output(json!("Please correct the ID"));
    let proxy = Arc::new(Proxy::default());
    let ctx = context(&factory, &proxy)?;
    let mut chat = builder(catalogue()).build(ctx.clone())?;
    chat.submit("Find staff")?;
    for _ in 0..100 {
        let step = chat.next()?;
        if let ChatStep::Agent(request) = step {
            assert_ne!(request.kind(), "tool");
            chat.resume_agent(chat.executor().execute(&request).await)?;
        }
        let snapshot = chat.snapshot()?;
        if contains_phase(&serde_json::to_value(&snapshot)?, "accepted_tools") {
            let mut restored = builder(catalogue()).restore::<()>(snapshot, ctx)?;
            assert_eq!(finish(&mut restored).await?, "Please correct the ID");
            assert_eq!(
                tool_outputs(&restored.snapshot()?)[0]["error_kind"],
                "Validation"
            );
            assert_eq!(proxy.calls.load(Ordering::SeqCst), 0);
            return Ok(());
        }
    }
    Err(TestError::Missing("staged proposal"))
}

/// Locates a phase inside existing checkpoint/effect wrappers without assuming frame positions.
fn contains_phase(value: &serde_json::Value, phase: &str) -> bool {
    if value
        .get("phase")
        .and_then(|value| value.get("kind"))
        .and_then(serde_json::Value::as_str)
        == Some(phase)
    {
        return true;
    }
    match value {
        serde_json::Value::Object(object) => object.values().any(|v| contains_phase(v, phase)),
        serde_json::Value::Array(array) => array.iter().any(|v| contains_phase(v, phase)),
        _ => false,
    }
}
