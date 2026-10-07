use super::*;
use pravah::clients::ErrorKind;
use pravah::clients::Provider;

/// Stops at generation and inspects the local typed failure before durable delivery.
async fn execution_error(
    runtime: &mut Runtime,
    executor: &AgentExecutor,
) -> Result<pravah::AgentResponse, TestError> {
    for _ in 0..100 {
        if let Some(fetch) = runtime.pending_agent()
            && fetch.kind() == "generate"
        {
            let response = executor.execute(fetch).await;
            assert!(response.outcome().is_err());
            return Ok(response);
        }
        host::step(runtime, executor).await?;
    }
    Err(TestError::Missing("generation boundary"))
}

/// Records a portable failure once; later VM errors must retain this exact accepted input.
fn accept_failure(
    runtime: &mut Runtime,
    response: pravah::AgentResponse,
) -> Result<Value, TestError> {
    runtime.resume_agent(response)?;
    Ok(serde_json::to_value(runtime.snapshot()?)?)
}

/// Output-limit classification uses Rath metadata, falling back to the actual client provider.
#[tokio::test]
async fn output_limit_provider_uses_metadata_or_client() -> Result<(), TestError> {
    for provider in [None, Some(Provider::Anthropic)] {
        let mut error = ClientError::new(ErrorKind::OutputLimitReached, "partial output");
        if let Some(provider) = &provider {
            error = error.with_context(provider.clone(), "llm.execute");
        }
        let expected = provider.unwrap_or_else(|| Provider::External("test".into()));
        let script = ScriptedFactory::new().then_err(error);
        let flow = compile(workflow)?;
        let executor = flow.prepared().executor(context(script, Some(2048))?);
        let mut runtime = flow.start(Request::capped(), uuid::Uuid::nil())?;
        let error = execution_error(&mut runtime, &executor).await?;
        let before = accept_failure(&mut runtime, error)?;
        assert!(matches!(
            runtime.next(),
            Err(GraphError::AgentOutputLimit { provider, .. }) if provider == expected
        ));
        assert_eq!(before, serde_json::to_value(runtime.snapshot()?)?);
    }
    Ok(())
}

/// Zero and duplicate caps fail before history or checkpoints change or clients execute.
#[tokio::test]
async fn invalid_caps_are_atomic_activation_errors() -> Result<(), TestError> {
    let flow = compile(workflow)?;
    for request in [
        Request {
            cap: Some(0),
            ..Request::capped()
        },
        Request {
            duplicate: true,
            ..Request::capped()
        },
    ] {
        let script = ScriptedFactory::new();
        let executor = flow.prepared().executor(context(script.clone(), None)?);
        let mut runtime = flow.start(request, uuid::Uuid::nil())?;
        let mut rejected = false;
        for _ in 0..20 {
            let before = serde_json::to_value(runtime.snapshot()?.history())?;
            match host::step(&mut runtime, &executor).await {
                Err(GraphError::AgentFailed { .. }) => {
                    assert_eq!(before, serde_json::to_value(runtime.snapshot()?.history())?);
                    rejected = true;
                    break;
                }
                other => {
                    other?;
                }
            }
        }
        assert!(rejected);
        assert!(runtime.snapshot()?.history().entries().is_empty());
        assert!(script.calls().is_empty());
    }
    Ok(())
}

/// Accepted provider exhaustion is distinct and never redispatches a completed generation.
#[tokio::test]
async fn exhausted_output_never_becomes_a_partial_answer() -> Result<(), TestError> {
    let script = ScriptedFactory::new()
        .then_err(
            ClientError::new(ErrorKind::OutputLimitReached, "partial-answer")
                .with_context(Provider::OpenAi, "llm.execute"),
        )
        .then_output(json!("complete"));
    let flow = compile(workflow)?;
    let executor = flow
        .prepared()
        .executor(context(script.clone(), Some(2048))?);
    let mut runtime = flow.start(Request::capped(), uuid::Uuid::nil())?;
    let local = execution_error(&mut runtime, &executor).await?;
    let before = accept_failure(&mut runtime, local)?;
    let error = runtime
        .next()
        .err()
        .ok_or(TestError::Missing("output limit error"))?;
    assert!(matches!(
        error,
        GraphError::AgentOutputLimit {
            provider: Provider::OpenAi,
            ..
        }
    ));
    assert!(!format!("{error:?} {error}").contains("partial-answer"));
    assert_eq!(before, serde_json::to_value(runtime.snapshot()?)?);
    let mut restored = flow.restore(runtime.snapshot()?)?;
    assert!(matches!(
        restored.next(),
        Err(GraphError::AgentOutputLimit { .. })
    ));
    assert_eq!(before, serde_json::to_value(restored.snapshot()?)?);
    assert_eq!(script.calls().len(), 1);
    assert!(!serde_json::to_string(runtime.snapshot()?.history())?.contains("partial-answer"));
    Ok(())
}

/// A non-limit client error remains a typed execution failure.
#[tokio::test]
async fn other_client_errors_remain_client_errors() -> Result<(), TestError> {
    let script =
        ScriptedFactory::new().then_err(ClientError::new(ErrorKind::Validation, "invalid request"));
    let flow = compile(workflow)?;
    let executor = flow.prepared().executor(context(script, Some(2048))?);
    let mut runtime = flow.start(Request::capped(), uuid::Uuid::nil())?;
    let error = execution_error(&mut runtime, &executor).await?;
    assert_eq!(error.outcome().unwrap_err().code(), "rath");
    Ok(())
}

/// Forced conclusion preserves output-exhaustion errors instead of masking them as invalid output.
#[tokio::test]
async fn conclusion_output_exhaustion_remains_distinct() -> Result<(), TestError> {
    let script = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "lookup-1",
            "lookup",
            json!({"query":"evidence"}),
        )])
        .then_err(
            ClientError::new(ErrorKind::OutputLimitReached, "partial-conclusion")
                .with_context(Provider::OpenAi, "llm.execute"),
        );
    let request = Request {
        turns: Some(1),
        ..Request::capped()
    };
    let flow = compile(workflow)?;
    let executor = flow
        .prepared()
        .executor(context(script.clone(), Some(2048))?);
    let mut runtime = flow.start(request, uuid::Uuid::nil())?;
    // The first generation succeeds; continue until the later conclusion request fails.
    loop {
        if let Some(fetch) = runtime.pending_agent() {
            let response = executor.execute(fetch).await;
            if response.outcome().is_err() {
                let before = accept_failure(&mut runtime, response)?;
                assert!(matches!(
                    runtime.next(),
                    Err(GraphError::AgentOutputLimit { .. })
                ));
                assert_eq!(before, serde_json::to_value(runtime.snapshot()?)?);
                assert_eq!(script.calls().len(), 2);
                return Ok(());
            }
            runtime.resume_agent(response)?;
        } else {
            runtime.next()?;
        }
    }
}
