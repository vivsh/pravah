use super::*;

/// A pending tool resumes with fresh Context, while changed canonical schemas reject continuation.
#[tokio::test]
async fn pending_snapshot_restores_only_matching_definitions() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_tool_calls(vec![staff_call()]);
    let original = Arc::new(Proxy::default());
    let mut chat = builder(catalogue()).build(context(&factory, &original)?)?;
    chat.submit("Find staff 42")?;
    advance_to_tool(&mut chat).await?;
    let snapshot = chat.snapshot()?;
    let fresh = Arc::new(Proxy::default());
    let next = ScriptedFactory::new().then_output(json!("Done"));
    let copy: Snapshot = serde_json::from_value(serde_json::to_value(&snapshot)?)?;
    let mut restored = builder(catalogue()).restore::<()>(copy, context(&next, &fresh)?)?;
    assert_eq!(original.calls.load(Ordering::SeqCst), 0);
    assert_eq!(fresh.calls.load(Ordering::SeqCst), 0);
    assert_eq!(finish(&mut restored).await?, "Done");
    assert_eq!(fresh.calls.load(Ordering::SeqCst), 1);
    for field in ["input_schema", "output_schema", "alias", "description"] {
        let mut changed = catalogue();
        changed[0][field] = match field {
            "input_schema" => json!({"type":"object"}),
            "output_schema" => json!(true),
            _ => json!("changed"),
        };
        assert!(matches!(
            builder(changed).restore::<()>(snapshot.clone(), context(&next, &fresh)?),
            Err(GraphError::GraphMismatch { .. })
        ));
    }
    let mut reordered = catalogue();
    reordered.as_array_mut().expect("catalogue").reverse();
    assert!(matches!(
        builder(reordered).restore::<()>(snapshot, context(&next, &fresh)?),
        Err(GraphError::GraphMismatch { .. })
    ));
    Ok(())
}

/// Accepted success resumes after restore without executing the application handler again.
#[tokio::test]
async fn accepted_success_is_not_replayed() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![staff_call()])
        .then_output(json!("Done"));
    let proxy = Arc::new(Proxy::default());
    let ctx = context(&factory, &proxy)?;
    let mut chat = builder(catalogue()).build(ctx.clone())?;
    chat.submit("Find staff 42")?;
    let request = advance_to_tool(&mut chat).await?;
    let response = chat.executor().execute(&request).await;
    chat.resume_agent(response)?;
    let mut restored = builder(catalogue()).restore::<()>(chat.snapshot()?, ctx)?;
    assert_eq!(finish(&mut restored).await?, "Done");
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 1);
    assert_eq!(tool_outputs(&restored.snapshot()?).len(), 1);
    Ok(())
}

/// Restore rejects externally accepted malformed successes before returning an executable Chat.
#[tokio::test]
async fn invalid_accepted_success_is_rejected_on_restore() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_tool_calls(vec![staff_call()]);
    let proxy = Arc::new(Proxy::default());
    let ctx = context(&factory, &proxy)?;
    let mut chat = builder(catalogue()).build(ctx.clone())?;
    chat.submit("Find staff 42")?;
    let request = advance_to_tool(&mut chat).await?;
    chat.resume_agent(AgentResponse::new(
        request.id(),
        Ok(pravah::graph::to_value(
            json!({"kind":"success","value":{"staff_id":"bad","email":null}}),
        )?),
    ))?;
    assert!(matches!(
        builder(catalogue()).restore::<()>(chat.snapshot()?, ctx),
        Err(GraphError::SnapshotValidation(_))
    ));
    assert_eq!(proxy.calls.load(Ordering::SeqCst), 0);
    Ok(())
}
