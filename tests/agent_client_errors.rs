use std::error::Error;

use pravah::clients::{ClientError, ErrorKind, Provider};
use pravah::testing::ScriptedFactory;
use pravah::{AgentClientOperation, Chat, Context, GraphError};

#[path = "agent_client_errors/fixtures.rs"]
mod fixtures;
use fixtures::{TestError, client_context, keyed_error};

/// Async sends retain portable diagnostics and durably accept its portable failure exactly once.
#[tokio::test]
async fn local_error_and_portable_restore_share_one_completed_request() -> Result<(), TestError> {
    let script = ScriptedFactory::new().then_err(ClientError::new(ErrorKind::Transport, "offline"));
    let definition = || Chat::builder::<String, String>().model("test:///test");
    let mut chat = definition()
        .build(Context::default().with_providers(pravah::testing::providers(script.clone())?))?;
    let error = chat
        .send_with_key("question", "durable-key")
        .await
        .err()
        .ok_or(TestError::Missing("client failure"))?;
    assert!(matches!(&error, GraphError::AgentFailed { source } if
        source.details().and_then(|value| value.get("rath")).and_then(|value| value.get("kind")).and_then(pravah::graph::Value::as_str) == Some("transport")));
    assert!(chat.pending_agent().is_none());
    let snapshot = chat.snapshot()?;
    let before = serde_json::to_value(&snapshot)?;
    let mut restored = definition().restore::<()>(snapshot, Context::default())?;
    assert!(
        matches!(restored.next(), Err(GraphError::AgentFailed { source }) if source.code() == "rath")
    );
    assert_eq!(before, serde_json::to_value(restored.snapshot()?)?);
    assert_eq!(script.calls().len(), 1);
    Ok(())
}

/// Both durable client boundaries preserve classification and normalized cause chains.
#[tokio::test]
async fn keyed_chat_preserves_creation_and_execution_errors() -> Result<(), GraphError> {
    for operation in [AgentClientOperation::Create, AgentClientOperation::Execute] {
        let source = ClientError::new(ErrorKind::Transport, "private generated content")
            .with_context(Provider::OpenAi, "provider operation")
            .with_source(ClientError::new(ErrorKind::Timeout, "private prompt"));
        let error = keyed_error(client_context(operation, source)?).await?;
        let GraphError::AgentFailed { source } = &error else {
            return Err(GraphError::Invalid("missing portable failure".into()));
        };
        let details = source
            .details()
            .ok_or_else(|| GraphError::Invalid("missing diagnostics".into()))?;
        assert_eq!(
            details
                .get("operation")
                .and_then(pravah::graph::Value::as_str),
            Some(operation.to_string().as_str())
        );
        let rath = details
            .get("rath")
            .ok_or_else(|| GraphError::Invalid("missing Rath fields".into()))?;
        assert_eq!(
            rath.get("kind").and_then(pravah::graph::Value::as_str),
            Some("transport")
        );
        let diagnostic = if operation == AgentClientOperation::Create {
            rath.get("cause").unwrap()
        } else {
            rath
        };
        assert_eq!(
            diagnostic
                .get("cause")
                .and_then(|v| v.get("kind"))
                .and_then(pravah::graph::Value::as_str),
            Some("timeout")
        );
        assert!(
            Error::source(&error)
                .and_then(|v| v.downcast_ref::<pravah::AgentError>())
                .is_some()
        );
        assert!(!format!("{error} {error:?}").contains("private"));
        assert!(error.client_error().is_none());
    }
    Ok(())
}

/// Portable diagnostics explicitly retain HTTP metadata and bodies without displaying them.
#[tokio::test]
async fn keyed_chat_retains_http_diagnostics() -> Result<(), TestError> {
    let diagnostic = fixtures::http_diagnostic().await?;
    for operation in [AgentClientOperation::Create, AgentClientOperation::Execute] {
        let error = keyed_error(client_context(operation, diagnostic.clone())?).await?;
        let GraphError::AgentFailed { source } = &error else {
            return Err(TestError::Missing("portable error"));
        };
        let rath = source
            .details()
            .and_then(|v| v.get("rath"))
            .ok_or(TestError::Missing("Rath diagnostic"))?;
        let rath = if operation == AgentClientOperation::Create {
            rath.get("cause").unwrap()
        } else {
            rath
        };
        assert_eq!(
            rath.get("http_status")
                .and_then(pravah::graph::Value::as_u64),
            Some(400)
        );
        for (key, expected) in [
            ("provider_code", "test_code"),
            ("request_id", "request-42"),
            ("retry_after", "17"),
        ] {
            assert_eq!(
                rath.get(key).and_then(pravah::graph::Value::as_str),
                Some(expected)
            );
        }
        let bytes: Vec<u8> = serde_json::from_value(serde_json::to_value(
            rath.get("response_body")
                .and_then(|v| v.get("bytes"))
                .unwrap(),
        )?)?;
        assert_eq!(bytes, fixtures::PRIVATE_BODY.as_bytes());
        assert_eq!(
            rath.get("cause")
                .and_then(|v| v.get("kind"))
                .and_then(pravah::graph::Value::as_str),
            Some("transport")
        );
        assert!(!format!("{error} {error:?}").contains(fixtures::PRIVATE_BODY));
    }
    Ok(())
}

/// Runtime response validation does not fabricate a Rath source or provider failure.
#[tokio::test]
async fn empty_tool_batch_is_a_runtime_response_error() -> Result<(), GraphError> {
    let script = ScriptedFactory::new().then_tool_calls(Vec::new());
    let error =
        keyed_error(Context::default().with_providers(pravah::testing::providers(script.clone())?))
            .await?;
    assert!(
        matches!(&error, GraphError::AgentResponseValidation(reason) if reason == "model returned an empty tool-call batch")
    );
    assert!(error.client_error().is_none());
    assert!(Error::source(&error).is_none());
    assert_eq!(script.calls().len(), 1);
    Ok(())
}

/// Output exhaustion through keyed Chat retains its existing dedicated classification.
#[tokio::test]
async fn keyed_chat_output_limit_remains_distinct() -> Result<(), GraphError> {
    let source = ClientError::new(ErrorKind::OutputLimitReached, "private partial answer")
        .with_context(Provider::Anthropic, "generation");
    let error = keyed_error(client_context(AgentClientOperation::Execute, source)?).await?;
    assert!(matches!(
        &error,
        GraphError::AgentOutputLimit {
            provider: Provider::Anthropic,
            ..
        }
    ));
    assert!(error.client_error().is_none());
    assert!(Error::source(&error).is_none());
    assert!(!error.to_string().contains("private partial answer"));
    Ok(())
}

/// Moving Rath's already-boxed error into GraphError adds no allocation or wrapper box.
#[test]
fn retaining_client_source_adds_no_allocation() {
    let mut source = Some(ClientError::new(ErrorKind::Validation, "invalid request"));
    let allocations = allocation_counter::measure(|| {
        std::hint::black_box(source.take().map(|source| GraphError::AgentClient {
            operation: AgentClientOperation::Create,
            source,
        }));
    });
    assert_eq!(allocations.count_total, 0);
}

/// Client failures preserve the exact pre-dispatch snapshot and remain errors on retry.
#[tokio::test]
async fn graph_client_failure_preserves_checkpoint_and_history() -> Result<(), TestError> {
    for operation in [AgentClientOperation::Create, AgentClientOperation::Execute] {
        let error = ClientError::new(ErrorKind::Validation, "invalid configuration");
        let context = client_context(operation, error)?;
        let flow = pravah::compile(fixtures::flow)?;
        let executor = flow.prepared().executor(context);
        let mut runtime = flow.start("question".into(), uuid::Uuid::nil())?;
        let mut failed = false;
        for _ in 0..20 {
            let pravah::Step::Agent(fetch) = runtime.next()? else {
                continue;
            };
            let before = serde_json::to_value(runtime.snapshot()?)?;
            let response = executor.execute(&fetch).await;
            if response.outcome().is_err() {
                assert_eq!(response.outcome().unwrap_err().code(), "rath");
                assert_eq!(before, serde_json::to_value(runtime.snapshot()?)?);
                runtime.resume_agent(response)?;
                let accepted = serde_json::to_value(runtime.snapshot()?)?;
                assert!(matches!(
                    runtime.next(),
                    Err(GraphError::AgentFailed { .. })
                ));
                assert_eq!(accepted, serde_json::to_value(runtime.snapshot()?)?);
                failed = true;
                break;
            }
            runtime.resume_agent(response)?;
        }
        assert!(failed, "client boundary was not reached");
    }
    Ok(())
}
