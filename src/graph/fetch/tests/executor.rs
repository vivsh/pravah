use super::*;
use crate::clients::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, ErrorKind, LlmBackend,
    Message, ModelUrl, Provider, ProviderFactory, ProviderRegistry,
};
use crate::graph::FetchError;
use crate::history::{CompactionRequest, CompactionResult, Compactor, HistoryEntry, HistoryStore};
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
    let executor = FetchExecutor::new(context);
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
    let executor = FetchExecutor::new(context);
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
    let executor = FetchExecutor::new(Context::default());
    let fetch = Fetch::new(
        Uuid::nil(),
        Arc::new(super::super::FetchRequest::new("POST", "unknown://job")),
    );
    assert!(matches!(
        executor.execute(&fetch).await,
        Err(GraphError::FetchValidation(_))
    ));
}

struct RecordedTask;

impl DynFetchHandler for RecordedTask {
    fn execute<'a>(
        &'a self,
        fetch: &'a Fetch,
        _context: Context,
    ) -> BoxFuture<'a, Result<FetchResponse, GraphError>> {
        Box::pin(async move {
            assert_eq!(fetch.request().url(), "task://recorded");
            Ok(FetchResponse::new(202))
        })
    }
}

/// A worker can decode and execute a persisted request without any graph or VM.
#[tokio::test]
async fn graph_free_worker_executes_serialized_fetch() -> Result<(), GraphError> {
    let fetch = Fetch::new(
        Uuid::from_u128(7),
        Arc::new(super::super::FetchRequest::new("POST", "task://recorded")),
    );
    let encoded = serde_json::to_vec(&fetch)
        .map_err(|error| GraphError::FetchValidation(error.to_string()))?;
    let restored: Fetch = serde_json::from_slice(&encoded)
        .map_err(|error| GraphError::FetchValidation(error.to_string()))?;
    let mut executor = FetchExecutor::new(Context::default());
    executor.register("task", RecordedTask)?;

    assert_eq!(executor.execute(&restored).await?.status(), 202);
    assert_eq!(restored.id(), fetch.id());
    Ok(())
}

struct CountingCompactor(Arc<AtomicUsize>);

impl Compactor for CountingCompactor {
    type Error = std::convert::Infallible;

    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _context: Context,
    ) -> Result<CompactionResult, Self::Error> {
        assert_eq!(request.session_id(), "session");
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(CompactionResult::default())
    }
}

struct CountingStore(Arc<AtomicUsize>);

impl HistoryStore for CountingStore {
    type Error = std::convert::Infallible;

    async fn record(&self, entry: &HistoryEntry) -> Result<(), Self::Error> {
        assert_eq!(entry.position, 0);
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Builds a provider-backed preparation hook without any graph state.
fn preparation_fetch() -> Result<Fetch, GraphError> {
    let request = RathRequest::new(
        "openai:///recorded?temperature=0.2",
        ClientOptions::default(),
        Vec::new(),
    );
    let body = crate::graph::to_value(serde_json::json!({
        "version": 1,
        "session": "session",
        "execution": Uuid::from_u128(4),
        "request": request,
        "entries": [],
        "guidance": [],
        "budget_conclusion": false,
    }))
    .map_err(|error| GraphError::FetchValidation(error.to_string()))?;
    Ok(Fetch::new(
        Uuid::from_u128(5),
        Arc::new(
            super::super::FetchRequest::new("POST", "pravah://prepare")
                .body(FetchBody::Value(body)),
        ),
    ))
}

/// Builds a history acknowledgement hook using a fixed, reproducible entry ID.
fn history_fetch() -> Result<Fetch, GraphError> {
    let entry = HistoryEntry {
        id: Uuid::from_u128(6),
        position: 0,
        session_id: "session".into(),
        agent_id: "agent".into(),
        evicted: false,
        message: Message::user("input"),
    };
    let body = crate::graph::to_value(serde_json::json!({
        "version": 1,
        "entries": [entry],
    }))
    .map_err(|error| GraphError::FetchValidation(error.to_string()))?;
    Ok(Fetch::new(
        Uuid::from_u128(7),
        Arc::new(
            super::super::FetchRequest::new("POST", "pravah://history")
                .body(FetchBody::Value(body)),
        ),
    ))
}

/// Framework preparation and persistence use worker-installed services without a graph.
#[tokio::test]
async fn graph_free_worker_uses_history_services() -> Result<(), GraphError> {
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let compactor_calls = Arc::new(AtomicUsize::new(0));
    let store_calls = Arc::new(AtomicUsize::new(0));
    let context = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
        Factory(factory_calls.clone()),
    ));
    let executor = FetchExecutor::new(context)
        .with_compactor(CountingCompactor(compactor_calls.clone()))
        .with_store(CountingStore(store_calls.clone()));
    assert_eq!(executor.execute(&preparation_fetch()?).await?.status(), 200);
    assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
    assert_eq!(compactor_calls.load(Ordering::SeqCst), 1);
    assert_eq!(executor.execute(&history_fetch()?).await?.status(), 200);
    assert_eq!(store_calls.load(Ordering::SeqCst), 1);
    Ok(())
}
