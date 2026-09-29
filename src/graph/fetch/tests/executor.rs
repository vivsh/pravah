use super::*;
use crate::clients::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, ErrorKind, LlmBackend,
    Message, ModelUrl, Provider, ProviderFactory, ProviderRegistry,
};
use crate::graph::FetchError;
use std::sync::atomic::{AtomicUsize, Ordering};
use uuid::Uuid;

struct Factory(Arc<AtomicUsize>);
struct Backend {
    model: ModelUrl,
    options: ClientOptions,
}

impl ProviderFactory for Factory {
    async fn llm(&self, model: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        assert_eq!(model.model(), "recorded");
        assert_eq!(options.temperature, Some(0.2));
        Ok(Client::from_backend(Backend {
            model: model.clone(),
            options,
        }))
    }
}

impl LlmBackend for Backend {
    fn model_url(&self) -> &ModelUrl {
        &self.model
    }
    fn options(&self) -> &ClientOptions {
        &self.options
    }
    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        assert_eq!(
            messages.first().map(|message| message.content.as_str()),
            Some("input")
        );
        Ok(ClientResponse::new(
            Provider::OpenAi,
            ClientOutput::Output(serde_json::json!({"answer":42})),
        ))
    }
}

/// Recorded providers use normal model identities and Rath URL precedence, exactly once.
#[tokio::test]
async fn executes_recorded_rath_without_network() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let context = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
        Factory(calls.clone()),
    ));
    let executor = FetchExecutor::new(context, Arc::new(HandlerRegistry::new()));
    let request = RathRequest::new(
        "openai:///recorded?temperature=0.2",
        ClientOptions::default().with_temperature(0.9),
        vec![Message::user("input")],
    );
    let fetch = Fetch::new(Uuid::nil(), Arc::new(request.into_fetch_request()?));
    let response = executor.execute(&fetch).await?;
    assert!(matches!(
        RathResponse::from_fetch_response(&response)?
            .response()
            .output,
        ClientOutput::Output(_)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

struct FailingFactory;
impl ProviderFactory for FailingFactory {
    async fn llm(&self, _model: &ModelUrl, _options: ClientOptions) -> Result<Client, ClientError> {
        Err(ClientError::new(ErrorKind::Timeout, "private diagnostic")
            .with_context(Provider::OpenAi, "construction")
            .with_source(ClientError::new(ErrorKind::Transport, "nested cause")))
    }
}

/// Executor errors retain original sources; portable delivery explicitly preserves normalized causes.
#[tokio::test]
async fn failure_is_typed_without_fallback() -> Result<(), GraphError> {
    let context =
        Context::default().with_providers(ProviderRegistry::with_builtin_factory(FailingFactory));
    let executor = FetchExecutor::new(context, Arc::new(HandlerRegistry::new()));
    let request = RathRequest::new("openai:///recorded", ClientOptions::default(), Vec::new())
        .into_fetch_request()?;
    let result = executor
        .execute(&Fetch::new(Uuid::nil(), Arc::new(request)))
        .await;
    let error = result
        .err()
        .ok_or_else(|| GraphError::Invalid("expected factory failure".into()))?;
    assert!(matches!(
        error,
        GraphError::AgentClient {
            operation: AgentClientOperation::Create,
            ..
        }
    ));
    assert_eq!(
        error.client_error().map(ClientError::kind),
        Some(ErrorKind::Timeout)
    );
    assert!(std::error::Error::source(&error).is_some());
    let portable = FetchError::from_execution_error(&error)
        .map_err(|err| GraphError::Invalid(err.to_string()))?;
    assert_eq!(portable.code(), "rath");
    assert!(!format!("{portable:?} {portable}").contains("private"));
    assert_eq!(
        portable
            .details()
            .and_then(|value| value.get("rath"))
            .and_then(|value| value.get("kind"))
            .and_then(|value| value.as_str()),
        Some("timeout")
    );
    Ok(())
}

/// Unknown protocols never fall through to HTTP or a provider.
#[tokio::test]
async fn rejects_unknown_scheme() {
    let executor = FetchExecutor::new(Context::default(), Arc::new(HandlerRegistry::new()));
    let fetch = Fetch::new(
        Uuid::nil(),
        Arc::new(super::super::FetchRequest::new("POST", "unknown://job")),
    );
    assert!(matches!(
        executor.execute(&fetch).await,
        Err(GraphError::FetchValidation(_))
    ));
}
