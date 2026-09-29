use std::error::Error;

use pravah::clients::{ClientError, ErrorKind, Provider};
use pravah::testing::ScriptedFactory;
use pravah::{AgentClientOperation, Chat, Context, GraphError};

#[path = "agent_client_errors/fixtures.rs"]
mod fixtures;
use fixtures::{TestError, client_context, keyed_error};

/// Async sends retain the local source and durably accept its portable failure exactly once.
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
    assert_eq!(
        error.client_error().map(ClientError::kind),
        Some(ErrorKind::Transport)
    );
    assert!(chat.pending_fetch().is_none());
    let snapshot = chat.snapshot()?;
    let before = serde_json::to_value(&snapshot)?;
    let mut restored = definition().restore::<()>(snapshot, Context::default())?;
    assert!(
        matches!(restored.next(), Err(GraphError::FetchFailed { source }) if source.code() == "rath")
    );
    assert_eq!(before, serde_json::to_value(restored.snapshot()?)?);
    assert_eq!(script.calls().len(), 1);
    Ok(())
}

/// Keyed Chat submissions expose the same typed source at both client boundaries.
#[tokio::test]
async fn keyed_chat_preserves_creation_and_execution_errors() -> Result<(), GraphError> {
    for operation in [AgentClientOperation::Create, AgentClientOperation::Execute] {
        let source = ClientError::new(ErrorKind::Transport, "private generated content")
            .with_context(Provider::OpenAi, "provider operation")
            .with_source(ClientError::new(ErrorKind::Timeout, "private prompt"));
        let error = keyed_error(client_context(operation, source)?).await?;
        assert!(
            matches!(&error, GraphError::AgentClient { operation: actual, .. } if *actual == operation)
        );
        let source = error
            .client_error()
            .ok_or_else(|| GraphError::Invalid("missing client source".into()))?;
        assert_eq!(source.kind(), ErrorKind::Transport);
        let (provider, stage) = match operation {
            AgentClientOperation::Create => {
                (Provider::External("test".into()), "client construction")
            }
            AgentClientOperation::Execute => (Provider::OpenAi, "provider operation"),
        };
        assert_eq!(source.provider(), Some(&provider));
        assert_eq!(source.operation(), Some(stage));
        let diagnostic = match operation {
            AgentClientOperation::Create => source
                .source()
                .ok_or_else(|| GraphError::Invalid("missing original cause".into()))?,
            AgentClientOperation::Execute => source,
        };
        assert_eq!(diagnostic.provider(), Some(&Provider::OpenAi));
        assert_eq!(diagnostic.operation(), Some("provider operation"));
        assert_eq!(
            diagnostic.source().map(ClientError::kind),
            Some(ErrorKind::Timeout)
        );
        let chained = Error::source(&error).and_then(|source| source.downcast_ref::<ClientError>());
        assert!(chained.is_some_and(|chained| std::ptr::eq(chained, source)));
        assert_eq!(
            error.to_string(),
            format!("agent client {operation} failed")
        );
    }
    Ok(())
}

/// Every supplied HTTP diagnostic and nested source survives keyed Chat propagation.
#[tokio::test]
async fn keyed_chat_retains_http_diagnostics() -> Result<(), TestError> {
    let diagnostic = fixtures::http_diagnostic().await?;
    for operation in [AgentClientOperation::Create, AgentClientOperation::Execute] {
        let error = keyed_error(client_context(operation, diagnostic.clone())?).await?;
        let retained = error
            .client_error()
            .ok_or(TestError::Missing("client error"))?;
        assert_eq!(retained.kind(), ErrorKind::Http);
        let (provider, stage) = match operation {
            AgentClientOperation::Create => {
                (Provider::External("test".into()), "client construction")
            }
            AgentClientOperation::Execute => (Provider::Ollama, "generation"),
        };
        assert_eq!(retained.provider(), Some(&provider));
        assert_eq!(retained.operation(), Some(stage));
        let retained = match operation {
            AgentClientOperation::Create => retained
                .source()
                .ok_or(TestError::Missing("original HTTP cause"))?,
            AgentClientOperation::Execute => retained,
        };
        assert_eq!(retained.provider(), Some(&Provider::Ollama));
        assert_eq!(retained.operation(), Some("generation"));
        assert_eq!(retained.http_status(), Some(400));
        assert_eq!(retained.provider_code(), Some("test_code"));
        assert_eq!(retained.request_id(), Some("request-42"));
        assert_eq!(retained.retry_after(), Some("17"));
        let body = retained
            .response_body()
            .ok_or(TestError::Missing("response body"))?;
        assert!(body.is_complete());
        assert_eq!(body.bytes(), fixtures::PRIVATE_BODY.as_bytes());
        let cause = Error::source(retained)
            .and_then(|cause| cause.downcast_ref::<ClientError>())
            .ok_or(TestError::Missing("nested cause"))?;
        assert_eq!(cause.kind(), ErrorKind::Transport);
        assert_eq!(
            cause.source().map(ClientError::kind),
            Some(ErrorKind::Timeout)
        );
        assert_eq!(
            error.to_string(),
            format!("agent client {operation} failed")
        );
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
            let pravah::Step::Fetch(fetch) = runtime.next()? else {
                continue;
            };
            let before = serde_json::to_value(runtime.snapshot()?)?;
            match executor.execute(&fetch).await {
                Err(error) => {
                    assert!(
                        matches!(error, GraphError::AgentClient { operation: actual, .. } if actual == operation)
                    );
                    assert_eq!(before, serde_json::to_value(runtime.snapshot()?)?);
                    assert!(matches!(
                        executor.execute(&fetch).await,
                        Err(GraphError::AgentClient { .. })
                    ));
                    assert_eq!(before, serde_json::to_value(runtime.snapshot()?)?);
                    failed = true;
                    break;
                }
                Ok(response) => runtime.resume_fetch(fetch.id(), Ok(response))?,
            }
        }
        assert!(failed, "client boundary was not reached");
    }
    Ok(())
}
