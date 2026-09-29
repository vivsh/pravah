use pravah::clients::{
    Client, ClientError, ClientOptions, Message, ModelUrl, ProviderFactory, ProviderRegistry,
};
use pravah::testing::ScriptedFactory;
use pravah::{Chat, Context, GraphError};

#[path = "rath_registry/builtin.rs"]
mod builtin;
#[path = "rath_registry/routing.rs"]
mod routing;
#[path = "rath_registry/tools.rs"]
mod tools;

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error(transparent)]
    Client(#[from] ClientError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Encode(#[from] ciborium::ser::Error<std::io::Error>),
    #[error(transparent)]
    Decode(#[from] ciborium::de::Error<std::io::Error>),
    #[error("missing {0}")]
    Missing(&'static str),
}

struct PendingFactory;

impl ProviderFactory for PendingFactory {
    async fn llm(&self, _: &ModelUrl, _: ClientOptions) -> Result<Client, ClientError> {
        std::future::pending().await
    }
}

fn flow(root: pravah::Flow<String>) -> pravah::Flow<String> {
    root.agent(agent)
}

fn agent(root: pravah::Agent<String>) -> pravah::Agent<String> {
    root.configure(configure)
}

async fn configure(input: String, _: Context) -> Result<pravah::AgentConfig, GraphError> {
    Ok(pravah::AgentConfig::new(
        "pending:///test",
        "Answer.",
        Message::user(input),
    ))
}

/// Cancelling asynchronous provider construction cannot commit a dispatch or alter history.
#[tokio::test]
async fn cancelled_construction_preserves_dispatch_snapshot() -> Result<(), TestError> {
    let registry = ProviderRegistry::new().register("pending", PendingFactory)?;
    let flow = pravah::compile(flow)?;
    let executor = flow
        .prepared()
        .executor(Context::default().with_providers(registry));
    let mut runtime = flow.start("question".into(), uuid::Uuid::nil())?;
    for _ in 0..20 {
        let pravah::Step::Fetch(fetch) = runtime.next()? else {
            continue;
        };
        let before = serde_json::to_value(runtime.snapshot()?)?;
        let poll = {
            let future = executor.execute(&fetch);
            futures::pin_mut!(future);
            futures::poll!(future)
        };
        match poll {
            std::task::Poll::Ready(result) => runtime.resume_fetch(fetch.id(), Ok(result?))?,
            std::task::Poll::Pending => {
                assert_eq!(before, serde_json::to_value(runtime.snapshot()?)?);
                assert_eq!(runtime.snapshot()?.history().entries().len(), 1);
                return Ok(());
            }
        }
    }
    Err(GraphError::Invalid("provider construction was not reached".into()).into())
}

/// Registry-backed Chat preserves keyed history and restores with fresh provider dependencies.
#[tokio::test]
async fn registry_chat_round_trip() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("first"));
    let registry = ProviderRegistry::new().register("scripted", factory.clone())?;
    let mut chat = Chat::builder::<String, String>()
        .model("scripted:///test")
        .state(7u32)
        .build(Context::default().with_providers(registry))?;
    assert!(factory.calls().is_empty());
    assert_eq!(
        chat.send_with_key("question", "message-42").await?.output,
        "first"
    );
    let snapshot = serde_json::from_slice(&serde_json::to_vec(&chat.snapshot()?)?)?;
    let next = ScriptedFactory::new().then_output(serde_json::json!("second"));
    let registry = ProviderRegistry::new().register("scripted", next.clone())?;
    let mut restored = Chat::builder::<String, String>()
        .model("scripted:///test")
        .restore::<u32>(snapshot, Context::default().with_providers(registry))?;
    assert!(next.calls().is_empty());
    assert_eq!(restored.get()?, 7);
    assert_eq!(restored.send("next").await?.output, "second");
    let snapshot = restored.snapshot()?;
    assert_eq!(
        snapshot
            .history()
            .entries()
            .first()
            .and_then(|entry| entry.message.key.as_deref()),
        Some("message-42")
    );
    assert_eq!(snapshot.history().entries().len(), 4);
    Ok(())
}
