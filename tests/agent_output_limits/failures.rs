use super::*;
use pravah::clients::Provider;

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
        let mut runtime = flow.start(request, context(script.clone(), None))?;
        let mut rejected = false;
        for _ in 0..20 {
            let before = serde_json::to_value(runtime.snapshot()?)?;
            match runtime.next().await {
                Err(GraphError::AgentConfigValidation(_)) => {
                    assert_eq!(before, serde_json::to_value(runtime.snapshot()?)?);
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

/// Provider exhaustion is distinct, redacted, and leaves the dispatch checkpoint retryable.
#[tokio::test]
async fn exhausted_output_never_becomes_a_partial_answer() -> Result<(), TestError> {
    let script = ScriptedFactory::new()
        .then_err(ClientError::OutputLimitReached {
            provider: Provider::OpenAi,
            response: json!({"secret":"partial-answer"}),
        })
        .then_output(json!("complete"));
    let mut runtime =
        compile(workflow)?.start(Request::capped(), context(script.clone(), Some(2048)))?;
    let before = serde_json::to_value(activated(&mut runtime).await?)?;
    let error = runtime
        .next()
        .await
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
    finish(&mut runtime).await?;
    assert_eq!(script.calls().len(), 2);
    assert!(!serde_json::to_string(runtime.snapshot()?.history())?.contains("partial-answer"));
    Ok(())
}

/// A non-limit client error preserves its existing generic error contract.
#[tokio::test]
async fn other_client_errors_remain_client_errors() -> Result<(), TestError> {
    let script = ScriptedFactory::new().then_err(ClientError::Validation("invalid request".into()));
    let mut runtime = compile(workflow)?.start(Request::capped(), context(script, Some(2048)))?;
    activated(&mut runtime).await?;
    assert!(matches!(
        runtime.next().await,
        Err(GraphError::AgentClient(_))
    ));
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
        .then_err(ClientError::OutputLimitReached {
            provider: Provider::OpenAi,
            response: json!({"secret":"partial-conclusion"}),
        });
    let request = Request {
        turns: Some(1),
        ..Request::capped()
    };
    let mut runtime = compile(workflow)?.start(request, context(script.clone(), Some(2048)))?;
    for _ in 0..50 {
        let before = serde_json::to_value(runtime.snapshot()?)?;
        match runtime.next().await {
            Err(GraphError::AgentOutputLimit { .. }) => {
                assert_eq!(before, serde_json::to_value(runtime.snapshot()?)?);
                assert_eq!(script.calls().len(), 2);
                return Ok(());
            }
            other => {
                other?;
            }
        }
    }
    Err(TestError::Missing("conclusion did not reach output limit"))
}
