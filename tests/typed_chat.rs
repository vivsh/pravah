use pravah::testing::ScriptedFactory;
use pravah::{Chat, ChatRequest, Context, GraphError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[path = "typed_chat/inputs.rs"]
mod inputs;
#[path = "typed_chat/options.rs"]
mod options;

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

#[derive(Serialize, Deserialize, JsonSchema)]
struct Question {
    topic: String,
    depth: u32,
}

/// Non-Clone domain inputs, state and keyed history survive a typed builder round trip.
#[tokio::test]
async fn typed_keyed_restore() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("first"));
    let mut chat = Chat::builder::<Question, String>()
        .model("test:///test")
        .state(7u32)
        .build(Context::default().with_providers(pravah::testing::providers(factory.clone())?))?;
    chat.send_with_key(
        ChatRequest::from(Question {
            topic: "durability".into(),
            depth: 2,
        })
        .memory("Be concise"),
        "db:42",
    )
    .await?;
    let snapshot = chat.snapshot()?;
    let entry = &snapshot.history().entries()[0];
    assert_eq!(entry.message.content, r#"{"topic":"durability","depth":2}"#);
    assert_eq!(entry.message.key.as_deref(), Some("db:42"));
    let json = serde_json::from_slice(&serde_json::to_vec(&snapshot)?)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor)?;
    let cbor = ciborium::from_reader(cbor.as_slice())?;
    for snapshot in [json, cbor] {
        let factory = ScriptedFactory::new().then_output(serde_json::json!("second"));
        let mut restored = Chat::builder::<Question, String>()
            .model("test:///test")
            .restore::<u32>(
                snapshot,
                Context::default().with_providers(pravah::testing::providers(factory)?),
            )?;
        assert_eq!(restored.get()?, 7);
        assert_eq!(
            restored
                .send(Question {
                    topic: "recovery".into(),
                    depth: 3
                })
                .await?
                .output,
            "second"
        );
    }
    Ok(())
}
