use std::convert::Infallible;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use pravah::clients::{Message, Role};
use pravah::history::HistoryEntry;
use pravah::testing::ScriptedFactory;
use pravah::{
    Agent, AgentConfig, Chat, CompactionRequest, CompactionResult, Compactor, Context, Flow,
    GraphError, HistoryStore, Snapshot, Step, compile,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Input {
    text: String,
    key: String,
}

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    CborWrite(#[from] ciborium::ser::Error<std::io::Error>),
    #[error(transparent)]
    CborRead(#[from] ciborium::de::Error<std::io::Error>),
}

fn agent(root: Agent<Input>) -> Agent<String> {
    root.configure(configure)
}

async fn configure(input: Input, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test",
        "Answer briefly.",
        Message::user(input.text).with_key(input.key),
    )
    .keep_alive())
}

fn workflow(root: Flow<Input>) -> Flow<String> {
    root.agent(agent)
}

fn input() -> Input {
    Input {
        text: "question".into(),
        key: "db:42".into(),
    }
}

fn context() -> Result<Context, pravah::GraphError> {
    Ok(
        Context::default().with_providers(pravah::testing::providers(
            ScriptedFactory::new().then_output(serde_json::json!("answer")),
        )?),
    )
}

#[derive(Clone, Default)]
struct ObserveKey(Arc<AtomicUsize>);

impl HistoryStore for ObserveKey {
    type Error = Infallible;

    async fn record(&self, entry: &HistoryEntry) -> Result<(), Self::Error> {
        assert_message(&entry.message);
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl Compactor for ObserveKey {
    type Error = Infallible;

    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        for entry in request.committed().iter().chain(request.protected()) {
            assert_message(&entry.message);
        }
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(CompactionResult::default())
    }
}

fn assert_message(message: &Message) {
    let expected = matches!(message.role, Role::User).then_some("db:42");
    assert_eq!(message.key.as_deref(), expected);
}

fn copies(snapshot: &Snapshot) -> Result<[Snapshot; 2], TestError> {
    let json = serde_json::to_vec(snapshot)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(snapshot, &mut cbor)?;
    Ok([
        serde_json::from_slice(&json)?,
        ciborium::from_reader(cbor.as_slice())?,
    ])
}

/// Chat preserves the submitted message key in policies, stores and both snapshot codecs.
#[tokio::test]
async fn chat_preserves_message_keys() -> Result<(), TestError> {
    let observer = ObserveKey::default();
    let mut chat = Chat::new(agent, context()?)?
        .with_store(observer.clone())
        .with_compactor(observer.clone());
    assert_eq!(chat.send(input()).await?.output, "answer");
    assert_eq!(observer.0.load(Ordering::SeqCst), 3);
    for snapshot in copies(&chat.snapshot()?)? {
        let restored = Chat::<Input, String>::from_snapshot(agent, snapshot, context()?)?;
        let snapshot = restored.snapshot()?;
        assert_eq!(snapshot.history().entries().len(), 2);
        for entry in snapshot.history().entries() {
            assert_message(&entry.message);
        }
    }
    Ok(())
}

/// Explicit graph execution retains keys before activation and after history commit on restore.
#[tokio::test]
async fn graph_preserves_message_keys() -> Result<(), TestError> {
    let flow = compile(workflow)?;
    let runtime = flow.start(input(), uuid::Uuid::nil())?;
    for snapshot in copies(&runtime.snapshot()?)? {
        let observer = ObserveKey::default();
        let executor = pravah::graph::FetchExecutor::new(context()?)
            .with_registry(Arc::new(flow.registry().clone()))
            .with_store(observer.clone())
            .with_compactor(observer.clone());
        let mut runtime = flow.restore(snapshot)?;
        loop {
            match runtime.next()? {
                Step::Continue => {}
                Step::Fetch(fetch) => {
                    runtime.resume_fetch(fetch.id(), Ok(executor.execute(&fetch).await?))?;
                }
                Step::Suspend(_) => return Err(GraphError::ChatSuspended.into()),
                Step::Done(output) => {
                    assert_eq!(flow.decode_output(output)?, "answer");
                    break;
                }
            }
            // Restore every committed instruction without rerunning configuration.
            let [snapshot, _] = copies(&runtime.snapshot()?)?;
            runtime = flow.restore(snapshot)?;
        }
        assert_eq!(observer.0.load(Ordering::SeqCst), 3);
        for entry in runtime.snapshot()?.history().entries() {
            assert_message(&entry.message);
        }
    }
    Ok(())
}

/// Old serialized messages without a key remain readable and do not acquire an identity.
#[test]
fn missing_key_defaults_to_none() -> Result<(), serde_json::Error> {
    let message: Message = serde_json::from_value(serde_json::json!({
        "role": {"role": "user"}, "content": "old message"
    }))?;
    assert!(message.key.is_none());
    assert!(serde_json::to_value(message)?.get("key").is_none());
    Ok(())
}
