//! Deterministic first migration slice: delivery, restoration and Chat parity.

#[path = "durable_fetch/agent.rs"]
mod agent;
#[path = "durable_fetch/client_preparation.rs"]
mod client_preparation;
#[path = "durable_fetch/history.rs"]
mod history;
#[path = "durable_fetch/intervention.rs"]
mod intervention;
#[path = "durable_fetch/preparation.rs"]
mod preparation;
#[path = "durable_fetch/preparation_failures.rs"]
mod preparation_failures;
#[path = "durable_fetch/prepared_output.rs"]
mod prepared_output;
#[path = "durable_fetch/protocol_validation.rs"]
mod protocol_validation;
#[path = "durable_fetch/request_construction.rs"]
mod request_construction;
#[path = "durable_fetch/request_validation.rs"]
mod request_validation;
#[path = "durable_fetch/shared_configuration.rs"]
mod shared_configuration;
#[path = "durable_fetch/tool_values.rs"]
mod tool_values;
#[path = "durable_fetch/validation.rs"]
mod validation;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use pravah::graph::{
    ContinuationContext, ContinuationEvent, ContinuationHandler, ContinuationTransition, Snapshot,
    TypedGraphBuilder, Value,
};
use pravah::{
    Chat, ChatStep, Context, Fetch, FetchError, FetchRequest, FetchResponse, Flow, GraphError,
    Step, compile,
};
use uuid::Uuid;

fn fetch_flow(input: Flow<String>) -> Flow<bool> {
    input
        .map(|_| FetchRequest::new("GET", "task://example"))
        .fetch()
        .map(|outcome| outcome.is_ok())
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::ValueConversion {
        target: "test codec".into(),
        reason: error.to_string(),
    }
}

fn json_roundtrip(snapshot: Snapshot) -> Result<Snapshot, GraphError> {
    serde_json::from_slice(&serde_json::to_vec(&snapshot).map_err(codec)?).map_err(codec)
}

fn cbor_roundtrip(snapshot: Snapshot) -> Result<Snapshot, GraphError> {
    let mut bytes = Vec::new();
    ciborium::into_writer(&snapshot, &mut bytes).map_err(codec)?;
    ciborium::from_reader(bytes.as_slice()).map_err(codec)
}

fn next_fetch(runtime: &mut pravah::Runtime) -> Result<Fetch, GraphError> {
    loop {
        match runtime.next()? {
            Step::Continue => {}
            Step::Fetch(fetch) => return Ok(fetch),
            _ => return Err(GraphError::Invalid("expected Fetch".into())),
        }
    }
}

/// Synchronous flows require no Tokio runtime and can branch on response or external failure.
#[test]
fn fetch_restore_and_branch_without_executor() -> Result<(), GraphError> {
    let workflow = compile(fetch_flow)?;
    for succeeds in [true, false] {
        let mut runtime = workflow.start("input".into(), Uuid::from_u128(42))?;
        let fetch = next_fetch(&mut runtime)?;
        let snapshot = cbor_roundtrip(json_roundtrip(runtime.snapshot()?)?)?;
        let mut runtime = workflow.restore(snapshot)?;
        assert_eq!(runtime.pending_fetch().map(Fetch::id), Some(fetch.id()));
        let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
        assert!(
            runtime
                .resume_fetch(Uuid::nil(), Ok(FetchResponse::new(200)))
                .is_err()
        );
        assert_eq!(
            before,
            serde_json::to_value(runtime.snapshot()?).map_err(codec)?
        );
        let outcome = if succeeds {
            Ok(FetchResponse::new(503))
        } else {
            Err(FetchError::new("transport", "offline"))
        };
        runtime.resume_fetch(fetch.id(), outcome)?;
        assert!(
            runtime
                .resume_fetch(fetch.id(), Ok(FetchResponse::new(200)))
                .is_err()
        );
        loop {
            match runtime.next()? {
                Step::Continue => {}
                Step::Done(value) => {
                    assert_eq!(value.as_bool(), Some(succeeds));
                    break;
                }
                _ => return Err(GraphError::Invalid("unexpected external boundary".into())),
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct FailsAfterDelivery;
impl ContinuationHandler for FailsAfterDelivery {
    fn start<'a>(
        &'a self,
        _payload: &'a Value,
        _state: Option<Value>,
        _inputs: Vec<Value>,
        _ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        Ok(ContinuationTransition {
            checkpoint: Some(Value::from(true)),
            fetch: Some(FetchRequest::new("POST", "task://operation")),
            ..Default::default()
        })
    }
    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _checkpoint: Value,
        event: ContinuationEvent,
        _ctx: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        assert!(matches!(event, ContinuationEvent::Fetch { .. }));
        Err(GraphError::Invalid("intentional processing error".into()))
    }
}

/// Accepted responses remain in the continuation inbox after processing errors and restoration.
#[test]
fn accepted_outcome_is_not_redispatched() -> Result<(), GraphError> {
    let builder = TypedGraphBuilder::<()>::new();
    let output = builder.continuation::<(), (), FailsAfterDelivery, _>(builder.root(), ());
    let workflow = builder.finish(output)?;
    let mut runtime = workflow.start((), Uuid::from_u128(7))?;
    let fetch = next_fetch(&mut runtime)?;
    runtime.resume_fetch(fetch.id(), Ok(FetchResponse::new(200)))?;
    let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    assert!(runtime.next().is_err());
    assert_eq!(
        before,
        serde_json::to_value(runtime.snapshot()?).map_err(codec)?
    );
    let mut restored = workflow.restore(json_roundtrip(runtime.snapshot()?)?)?;
    assert!(restored.pending_fetch().is_none());
    assert!(restored.next().is_err());
    assert_eq!(
        before,
        serde_json::to_value(restored.snapshot()?).map_err(codec)?
    );
    Ok(())
}

use pravah::clients::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, LlmBackend, Message,
    ModelUrl, Provider, ProviderFactory, ProviderRegistry,
};

struct Factory(Arc<AtomicUsize>);
struct Model {
    model: ModelUrl,
    options: ClientOptions,
    calls: Arc<AtomicUsize>,
}
impl ProviderFactory for Factory {
    async fn llm(&self, model: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        Ok(Client::from_backend(Model {
            model: model.clone(),
            options,
            calls: self.0.clone(),
        }))
    }
}
impl LlmBackend for Model {
    fn model_url(&self) -> &ModelUrl {
        &self.model
    }
    fn options(&self) -> &ClientOptions {
        &self.options
    }
    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        assert!(!messages.is_empty());
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ClientResponse::new(
            Provider::OpenAi,
            ClientOutput::Output(serde_json::json!("answer")),
        ))
    }
}

fn context(calls: &Arc<AtomicUsize>) -> Context {
    Context::default().with_providers(ProviderRegistry::with_builtin_factory(Factory(
        calls.clone(),
    )))
}

fn builder() -> pravah::ChatBuilder<String, String> {
    Chat::builder::<String, String>()
        .model("openai:///recorded")
        .instructions("Answer.")
}

/// Every agent boundary survives a snapshot, with no constructor-time or repeated generation.
#[tokio::test]
async fn chat_manual_restore_at_every_fetch() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().state(42_u32).build(context(&calls))?;
    chat.submit_with_key("question", "key-1")?;
    loop {
        match chat.next()? {
            ChatStep::Continue => {}
            ChatStep::Fetch(fetch) => {
                let snapshot = cbor_roundtrip(json_roundtrip(chat.snapshot()?)?)?;
                chat = builder().restore(snapshot, context(&calls))?;
                let response = chat.executor().execute(&fetch).await?;
                chat.resume_fetch(fetch.id(), Ok(response))?;
            }
            ChatStep::Done(turn) => {
                assert_eq!(turn.output, "answer");
                break;
            }
            ChatStep::Suspend(_) => return Err(GraphError::ChatSuspended),
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(chat.get()?, 42);
    assert_eq!(
        chat.snapshot()?
            .history()
            .entries()
            .first()
            .and_then(|entry| entry.message.key.as_deref()),
        Some("key-1")
    );
    assert_eq!(chat.send("again").await?.output, "answer");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    Ok(())
}

/// Identical explicit execution identities and inputs produce stable pending request UUIDs.
#[test]
fn deterministic_identity() -> Result<(), GraphError> {
    let workflow = compile(fetch_flow)?;
    let mut left = workflow.start("input".into(), Uuid::from_u128(7))?;
    let mut right = workflow.start("input".into(), Uuid::from_u128(7))?;
    assert_eq!(next_fetch(&mut left)?.id(), next_fetch(&mut right)?.id());
    assert_eq!(
        serde_json::to_value(left.snapshot()?).map_err(codec)?,
        serde_json::to_value(right.snapshot()?).map_err(codec)?
    );
    Ok(())
}

fn direct_fetch(input: Flow<FetchRequest>) -> Flow<Result<FetchResponse, FetchError>> {
    input.fetch()
}

/// Exhausting the request sequence fails before installing a wait or changing VM state.
#[test]
fn fetch_sequence_overflow_is_atomic() -> Result<(), GraphError> {
    let workflow = compile(direct_fetch)?;
    let runtime = workflow.start(FetchRequest::new("GET", "task://test"), Uuid::nil())?;
    let mut encoded = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    encoded["state"]["next_fetch_sequence"] = serde_json::json!(u64::MAX);
    let mut runtime = workflow.restore(serde_json::from_value(encoded.clone()).map_err(codec)?)?;
    assert!(matches!(
        runtime.next(),
        Err(GraphError::FetchValidation(_))
    ));
    assert_eq!(
        encoded,
        serde_json::to_value(runtime.snapshot()?).map_err(codec)?
    );
    assert!(runtime.pending_fetch().is_none());
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error("intentional store failure")]
struct StoreFailure;

struct RejectAssistant;
impl pravah::HistoryStore for RejectAssistant {
    type Error = StoreFailure;

    async fn record(&self, entry: &pravah::HistoryEntry) -> Result<(), Self::Error> {
        if matches!(entry.message.role, pravah::clients::Role::Assistant) {
            Err(StoreFailure)
        } else {
            Ok(())
        }
    }
}

/// An accepted store failure after generation cannot cause another model dispatch after restore.
#[tokio::test]
async fn history_failure_retains_completed_generation() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder()
        .build(context(&calls))?
        .with_store(RejectAssistant);
    assert!(matches!(
        chat.send("question").await,
        Err(GraphError::HistoryPersistence(_))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(chat.pending_fetch().is_none());
    let snapshot = chat.snapshot()?;
    assert_eq!(snapshot.history().entries().len(), 1);
    let before = serde_json::to_value(&snapshot).map_err(codec)?;
    let mut restored = builder().restore::<()>(json_roundtrip(snapshot)?, context(&calls))?;
    assert!(matches!(
        restored.next(),
        Err(GraphError::FetchFailed { .. })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        before,
        serde_json::to_value(restored.snapshot()?).map_err(codec)?
    );
    Ok(())
}
