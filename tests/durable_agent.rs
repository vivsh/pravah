//! Deterministic first migration slice: delivery, restoration and Chat parity.

#[path = "durable_agent/agent.rs"]
mod agent;
#[path = "durable_agent/client_preparation.rs"]
mod client_preparation;
#[path = "durable_agent/history.rs"]
mod history;
#[path = "durable_agent/intervention.rs"]
mod intervention;
#[path = "durable_agent/preparation.rs"]
mod preparation;
#[path = "durable_agent/preparation_failures.rs"]
mod preparation_failures;
#[path = "durable_agent/prepared_output.rs"]
mod prepared_output;
#[path = "durable_agent/protocol_validation.rs"]
mod protocol_validation;
#[path = "durable_agent/request_construction.rs"]
mod request_construction;
#[path = "durable_agent/request_validation.rs"]
mod request_validation;
#[path = "durable_agent/shared_configuration.rs"]
mod shared_configuration;
#[path = "durable_agent/tool_values.rs"]
mod tool_values;
#[path = "durable_agent/validation.rs"]
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
    AgentError, AgentRequest, AgentResponse, Chat, ChatStep, Context, Flow, GraphError, Step,
    compile,
};
use uuid::Uuid;

fn agent_flow(input: Flow<String>) -> Flow<String> {
    input.agent(|root| root.configure(configure_agent))
}
async fn configure_agent(input: String, _: Context) -> Result<pravah::AgentConfig, GraphError> {
    Ok(pravah::AgentConfig::new(
        "openai:///recorded",
        "Answer",
        Message::user(input),
    ))
}
fn external_flow(input: Flow<String>) -> Flow<bool> {
    input
        .suspend::<Result<(), String>>()
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

fn next_agent(runtime: &mut pravah::Runtime) -> Result<AgentRequest, GraphError> {
    loop {
        match runtime.next()? {
            Step::Continue => {}
            Step::Agent(fetch) => return Ok(fetch),
            _ => return Err(GraphError::Invalid("expected AgentRequest".into())),
        }
    }
}

/// Non-agent work uses ordinary synchronous suspension and portable typed outcomes.
#[test]
fn suspend_restore_and_branch_without_executor() -> Result<(), GraphError> {
    let workflow = compile(external_flow)?;
    for succeeds in [true, false] {
        let mut runtime = workflow.start("input".into(), Uuid::from_u128(42))?;
        assert!(matches!(runtime.next()?, Step::Suspend(_)));
        let snapshot = cbor_roundtrip(json_roundtrip(runtime.snapshot()?)?)?;
        let mut runtime = workflow.restore(snapshot)?;
        let outcome: Result<(), String> = if succeeds {
            Ok(())
        } else {
            Err("offline".into())
        };
        runtime.resume(outcome)?;
        loop {
            match runtime.next()? {
                Step::Continue => {}
                Step::Done(value) => {
                    assert_eq!(value.as_bool(), Some(succeeds));
                    break;
                }
                _ => return Err(GraphError::Invalid("unexpected boundary".into())),
            }
        }
    }
    Ok(())
}

/// Accepted portable failures remain in the inbox through processing errors and restoration.
#[test]
fn accepted_outcome_is_not_redispatched() -> Result<(), GraphError> {
    let workflow = compile(agent_flow)?;
    let mut runtime = workflow.start("question".into(), Uuid::from_u128(7))?;
    let request = next_agent(&mut runtime)?;
    runtime.resume_agent(AgentResponse::new(
        request.id(),
        Err(AgentError::new("test", "offline")),
    ))?;
    let before = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    assert!(matches!(
        runtime.next(),
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(
        before,
        serde_json::to_value(runtime.snapshot()?).map_err(codec)?
    );
    let mut restored = workflow.restore(json_roundtrip(runtime.snapshot()?)?)?;
    assert!(restored.pending_agent().is_none());
    assert!(matches!(
        restored.next(),
        Err(GraphError::AgentFailed { .. })
    ));
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
async fn chat_manual_restore_at_every_agent() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().state(42_u32).build(context(&calls))?;
    chat.submit_with_key("question", "key-1")?;
    loop {
        match chat.next()? {
            ChatStep::Continue => {}
            ChatStep::Agent(fetch) => {
                let snapshot = cbor_roundtrip(json_roundtrip(chat.snapshot()?)?)?;
                chat = builder().restore(snapshot, context(&calls))?;
                let response = chat.executor().execute(&fetch).await;
                chat.resume_agent(response)?;
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
    let workflow = compile(agent_flow)?;
    let mut left = workflow.start("input".into(), Uuid::from_u128(7))?;
    let mut right = workflow.start("input".into(), Uuid::from_u128(7))?;
    assert_eq!(next_agent(&mut left)?.id(), next_agent(&mut right)?.id());
    assert_eq!(
        serde_json::to_value(left.snapshot()?).map_err(codec)?,
        serde_json::to_value(right.snapshot()?).map_err(codec)?
    );
    Ok(())
}

/// Exhausting the request sequence fails before installing a wait or changing VM state.
#[test]
fn agent_sequence_overflow_is_atomic() -> Result<(), GraphError> {
    let workflow = compile(agent_flow)?;
    let runtime = workflow.start("question".into(), Uuid::nil())?;
    let mut encoded = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    encoded["state"]["next_agent_sequence"] = serde_json::json!(u64::MAX);
    let mut runtime = workflow.restore(serde_json::from_value(encoded.clone()).map_err(codec)?)?;
    assert!(matches!(
        runtime.next(),
        Err(GraphError::AgentRequestValidation(_))
    ));
    assert_eq!(
        encoded,
        serde_json::to_value(runtime.snapshot()?).map_err(codec)?
    );
    assert!(runtime.pending_agent().is_none());
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error("intentional store failure")]
struct StoreFailure;

struct RejectAssistant;
impl pravah::HistoryStore for RejectAssistant {
    type Error = StoreFailure;

    async fn load(&self, _key: &str) -> Result<Vec<pravah::HistoryEntry>, Self::Error> {
        Ok(Vec::new())
    }

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
    let result = chat.send("question").await;
    assert!(
        matches!(result, Err(GraphError::AgentFailed { .. })),
        "{result:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(chat.pending_agent().is_none());
    let snapshot = chat.snapshot()?;
    assert_eq!(snapshot.history().entries().len(), 2);
    let before = serde_json::to_value(snapshot.history()).map_err(codec)?;
    let mut restored = builder().restore::<()>(json_roundtrip(snapshot)?, context(&calls))?;
    assert!(matches!(
        restored.next(),
        Err(GraphError::AgentFailed { .. })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        before,
        serde_json::to_value(restored.snapshot()?.history()).map_err(codec)?
    );
    Ok(())
}
