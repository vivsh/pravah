use pravah::clients::Role;
use pravah::testing::ScriptedFactory;
use pravah::{Chat, ChatBuilder, ChatRequest, Context, GraphError, Snapshot};

#[path = "chat_builder/construction.rs"]
mod construction;
#[path = "chat_builder/inputs.rs"]
mod inputs;
#[path = "chat_builder/recovery.rs"]
mod recovery;
#[path = "chat_builder/tools.rs"]
mod tools;

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error(transparent)]
    Value(#[from] pravah::graph::ValueError),
    #[error("missing {0}")]
    Missing(&'static str),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    CborWrite(#[from] ciborium::ser::Error<std::io::Error>),
    #[error(transparent)]
    CborRead(#[from] ciborium::de::Error<std::io::Error>),
}

fn builder() -> ChatBuilder<String, String> {
    Chat::builder()
        .model("test:///test")
        .instructions("Answer briefly.")
}

fn context(factory: &ScriptedFactory) -> Result<Context, pravah::GraphError> {
    Ok(Context::default().with_providers(pravah::testing::providers(factory.clone())?))
}

struct NoMemory;
impl pravah::Compactor for NoMemory {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        request: pravah::CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<pravah::CompactionResult, Self::Error> {
        assert!(
            !request
                .options()
                .preamble
                .as_deref()
                .unwrap_or_default()
                .contains("per-turn memory")
        );
        Ok(pravah::CompactionResult::default())
    }
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

/// The builder owns no live state: keys, history and application data restore through the graph.
#[tokio::test]
async fn builder_keyed_stateful_round_trip() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("first"));
    let mut chat = builder().state(7u32).build(context(&factory)?)?;
    assert!(factory.calls().is_empty());
    assert!(chat.snapshot()?.history().entries().is_empty());
    assert_eq!(chat.get()?, 7);
    chat.set(8)?;
    let request = ChatRequest::from(String::from("question")).memory("per-turn memory");
    assert_eq!(chat.send_with_key(request, "db:42").await?.output, "first");
    let snapshot = chat.snapshot()?;
    let entries = snapshot.history().entries();
    assert_eq!(entries[0].message.key.as_deref(), Some("db:42"));
    assert!(entries[1].message.key.is_none());
    for snapshot in copies(&snapshot)? {
        let factory = ScriptedFactory::new().then_output(serde_json::json!("second"));
        let mut restored = builder()
            .compactor(NoMemory)
            .restore::<u32>(snapshot, context(&factory)?)?;
        assert!(factory.calls().is_empty());
        assert_eq!(restored.get()?, 8);
        assert_eq!(restored.send("next").await?.output, "second");
        assert_eq!(restored.snapshot()?.history().entries().len(), 4);
    }
    Ok(())
}

/// Infallible request construction does not accept invalid selections into the VM.
#[tokio::test]
async fn invalid_selection_is_atomic_and_correctable() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let mut chat = builder().build(context(&factory)?)?;
    let before = serde_json::to_value(chat.snapshot()?)?;
    let request = ChatRequest::from(String::from("not a submission")).tools(["unknown"]);
    assert!(serde_json::to_value(&request).is_ok());
    assert!(matches!(
        chat.send(request).await,
        Err(GraphError::ChatRequestValidation { .. })
    ));
    assert_eq!(serde_json::to_value(chat.snapshot()?)?, before);
    assert!(factory.calls().is_empty());
    assert_eq!(chat.send("valid").await?.output, "answer");
    assert!(matches!(
        chat.snapshot()?.history().entries()[0].message.role,
        Role::User
    ));
    Ok(())
}
