use pravah::clients::Message;
use pravah::testing::ScriptedFactory;
use pravah::{Agent, AgentConfig, Chat, Context, GraphError, Snapshot};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[path = "chat_state/conversions.rs"]
mod conversions;
#[path = "chat_state/failures.rs"]
mod failures;
#[path = "chat_state/lifecycle.rs"]
mod lifecycle;
#[path = "chat_state/restore.rs"]
mod restore;
#[path = "chat_state/tools.rs"]
mod tools;

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Encode(#[from] ciborium::ser::Error<std::io::Error>),
    #[error(transparent)]
    Decode(#[from] ciborium::de::Error<std::io::Error>),
    #[error("{0}")]
    Missing(&'static str),
}

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Session {
    project: String,
    visits: u64,
}

fn assistant(root: Agent<String>) -> Agent<String> {
    root.configure(configure)
}

async fn configure(input: String, ctx: Context) -> Result<AgentConfig, GraphError> {
    if let Some(calls) = ctx.deps().get::<std::sync::atomic::AtomicUsize>() {
        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    Ok(AgentConfig::new("openai:///test", "Answer.", Message::user(input)).keep_alive())
}

/// Makes codec round trips usable for pristine, idle and unfinished executions.
fn roundtrips(snapshot: &Snapshot) -> Result<[Snapshot; 2], TestError> {
    let mut cbor = Vec::new();
    ciborium::into_writer(snapshot, &mut cbor)?;
    Ok([
        serde_json::from_slice(&serde_json::to_vec(snapshot)?)?,
        ciborium::from_reader(cbor.as_slice())?,
    ])
}

fn initial_state() -> Session {
    Session {
        project: "private".into(),
        visits: 0,
    }
}

fn context(script: ScriptedFactory) -> Context {
    Context::default().with_client_factory(script)
}

/// Verifies the complete typed-state path without requiring Default or Clone on state.
#[tokio::test]
async fn state_is_part_of_the_existing_snapshot() -> Result<(), TestError> {
    let script = ScriptedFactory::new().then_output(json!("first"));
    let mut chat = Chat::with_state(
        assistant,
        Session {
            project: "private".into(),
            visits: 0,
        },
        context(script.clone()),
    )
    .await?;
    assert!(script.calls().is_empty());
    assert!(chat.snapshot()?.history().entries().is_empty());
    assert_eq!(chat.get()?.visits, 0);
    chat.set(Session {
        project: "private".into(),
        visits: 1,
    })?;
    assert_eq!(chat.send("question").await?.output, "first");
    let snapshot: Snapshot = serde_json::from_slice(&serde_json::to_vec(&chat.snapshot()?)?)?;
    let fresh = ScriptedFactory::new().then_output(json!("second"));
    let mut restored = Chat::<String, String, Session>::from_snapshot(
        assistant,
        snapshot,
        context(fresh.clone()),
    )?;
    assert_eq!(restored.get()?.visits, 1);
    assert_eq!(restored.send("again").await?.output, "second");
    assert_eq!(restored.get()?.project, "private");
    assert!(!serde_json::to_string(restored.snapshot()?.history())?.contains("private"));
    assert_eq!(script.calls().len(), 1);
    assert_eq!(fresh.calls().len(), 1);
    Ok(())
}
