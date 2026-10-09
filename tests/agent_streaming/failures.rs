use super::chat::builder;
use super::*;
use std::sync::atomic::Ordering;

/// Creation, startup and mid-stream errors retain portable metadata without committing previews.
#[tokio::test]
async fn failures_preserve_diagnostics_and_stage_identity() -> Result<(), GraphError> {
    for (mode, operation, previews) in [
        (Mode::CreationFailure, "creation", 0),
        (Mode::StartupFailure, "execution", 0),
        (Mode::FailureAfterProgress, "execution", 1),
    ] {
        let stats = Arc::new(Stats::default());
        let mut chat = builder().build(context(mode, stats.clone())?)?;
        let mut seen = 0;
        let error = chat
            .send_stream_with_key("question", "message-42", |_, _| {
                seen += 1;
                std::future::ready(())
            })
            .await
            .err()
            .ok_or_else(|| GraphError::Invalid("missing failure".into()))?;
        assert_eq!(seen, previews);
        let GraphError::AgentFailed { source } = &error else {
            return Err(GraphError::Invalid("expected portable failure".into()));
        };
        assert_diagnostics(source, operation)?;
        assert!(!format!("{error} {error:?}").contains("private diagnostic"));
        assert_eq!(chat.snapshot()?.history().entries().len(), 1);
        assert!(chat.pending_agent().is_none());
        assert_eq!(stats.ordinary.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

/// Reads the existing portable failure fields without relying on safe Display text.
fn assert_diagnostics(source: &pravah::AgentError, operation: &str) -> Result<(), GraphError> {
    let details = source
        .details()
        .ok_or_else(|| GraphError::Invalid("missing diagnostics".into()))?;
    assert_eq!(
        details
            .get("operation")
            .and_then(pravah::graph::Value::as_str),
        Some(operation)
    );
    let rath = details
        .get("rath")
        .ok_or_else(|| GraphError::Invalid("missing Rath diagnostics".into()))?;
    assert_eq!(
        rath.get("kind").and_then(pravah::graph::Value::as_str),
        Some("http")
    );
    // Rath wraps creation context around the original diagnostic and retains it as a cause.
    let rath = if operation == "creation" {
        rath.get("cause")
            .ok_or_else(|| GraphError::Invalid("missing original creation failure".into()))?
    } else {
        rath
    };
    assert_provider_diagnostics(rath);
    Ok(())
}

/// Original provider metadata and nested causes survive streaming failure conversion.
fn assert_provider_diagnostics(rath: &pravah::graph::Value) {
    for (key, value) in [
        ("kind", "http"),
        ("provider_code", "overloaded"),
        ("request_id", "request-stream"),
        ("retry_after", "5"),
    ] {
        assert_eq!(
            rath.get(key).and_then(pravah::graph::Value::as_str),
            Some(value)
        );
    }
    assert_eq!(
        rath.get("http_status")
            .and_then(pravah::graph::Value::as_u64),
        Some(429)
    );
    assert_eq!(
        rath.get("cause")
            .and_then(|v| v.get("kind"))
            .and_then(pravah::graph::Value::as_str),
        Some("timeout")
    );
}

/// EOF and terminal validation failures remain distinct from successfully displayed progress.
#[tokio::test]
async fn incomplete_invalid_and_token_limited_outputs_fail() -> Result<(), GraphError> {
    for mode in [
        Mode::Eof,
        Mode::EmptyTools,
        Mode::WrongOutput,
        Mode::OutputLimit,
    ] {
        let stats = Arc::new(Stats::default());
        let mut chat = builder().build(context(mode, stats.clone())?)?;
        let error = chat
            .send_stream("question", |_, _| std::future::ready(()))
            .await
            .err()
            .ok_or_else(|| GraphError::Invalid("missing terminal failure".into()))?;
        match mode {
            Mode::Eof => assert!(matches!(&error, GraphError::AgentFailed { source } if
                source.details().and_then(|v| v.get("rath")).and_then(|v| v.get("kind")).and_then(pravah::graph::Value::as_str) == Some("invalid_response"))),
            Mode::OutputLimit => assert!(matches!(error, GraphError::AgentOutputLimit { .. })),
            Mode::WrongOutput => assert!(matches!(error, GraphError::Schema { .. })),
            _ => assert!(matches!(error, GraphError::AgentResponseValidation(_))),
        }
        assert_eq!(chat.snapshot()?.history().entries().len(), 1);
        assert_eq!(stats.starts.load(Ordering::SeqCst), 1);
        assert_eq!(stats.ordinary.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

/// Custom backends inherit Rath's unsupported-streaming error, without executing ordinary fallback.
#[tokio::test]
async fn unsupported_streaming_does_not_fall_back() -> Result<(), GraphError> {
    let script = pravah::testing::ScriptedFactory::new().then_output(serde_json::json!("unused"));
    let mut chat = builder()
        .build(Context::default().with_providers(pravah::testing::providers(script.clone())?))?;
    let error = chat
        .send_stream("question", |_, _| std::future::ready(()))
        .await
        .err()
        .ok_or_else(|| GraphError::Invalid("missing unsupported failure".into()))?;
    assert!(
        matches!(error, GraphError::AgentFailed { source } if source.details()
        .and_then(|v| v.get("rath")).and_then(|v| v.get("kind")).and_then(pravah::graph::Value::as_str) == Some("unsupported_capability"))
    );
    assert!(script.calls().is_empty());
    assert_eq!(script.remaining(), 1);
    Ok(())
}
