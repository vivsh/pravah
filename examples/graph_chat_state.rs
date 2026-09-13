//! Typed application state, messages and JSON/CBOR checkpoints in one chat session.
//!
//! Run with `cargo run --example graph_chat_state --features testing`. No credentials needed.

#[cfg(feature = "testing")]
mod support;

#[cfg(feature = "testing")]
mod example {
    use super::support::ExampleError;
    use pravah::clients::Message;
    use pravah::testing::ScriptedFactory;
    use pravah::{Agent, AgentConfig, Chat, Context, GraphError, Snapshot};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct Session {
        selected_project: String,
        questions: u32,
    }

    fn assistant(root: Agent<String>) -> Agent<String> {
        root.configure(configure)
    }

    async fn configure(question: String, _ctx: Context) -> Result<AgentConfig, GraphError> {
        Ok(AgentConfig::new(
            "openai:///scripted",
            "Answer briefly.",
            Message::user(question),
        )
        .keep_alive())
    }

    /// Persists state before input and restores a conversation using both supported example codecs.
    pub(super) async fn run() -> Result<(), ExampleError> {
        let state = Session {
            selected_project: "Pravah".into(),
            questions: 0,
        };
        let first =
            ScriptedFactory::new().then_output(serde_json::json!("Let's review the project."));
        let mut chat = Chat::with_state(
            assistant,
            state,
            Context::default().with_client_factory(first),
        )
        .await?;
        let initial_checkpoint = serde_json::to_vec(&chat.snapshot()?)?;
        println!(
            "Snapshot before first message: {} bytes",
            initial_checkpoint.len()
        );
        let mut state = chat.get()?;
        state.questions += 1;
        let question = format!("Help me review {}.", state.selected_project);
        chat.set(state)?;
        println!("{}", chat.send(question).await?.output);
        let snapshot = chat.snapshot()?;
        let json = serde_json::to_vec(&snapshot)?;
        let mut cbor = Vec::new();
        ciborium::into_writer(&snapshot, &mut cbor)
            .map_err(|error| ExampleError::unexpected(error.to_string()))?;
        let copies: [Snapshot; 2] = [
            serde_json::from_slice(&json)?,
            ciborium::from_reader(cbor.as_slice())
                .map_err(|error| ExampleError::unexpected(error.to_string()))?,
        ];
        for copy in copies {
            continue_chat(copy).await?;
        }
        Ok(())
    }

    /// Binds new services while retaining application state and conversation history.
    async fn continue_chat(snapshot: Snapshot) -> Result<(), ExampleError> {
        let next =
            ScriptedFactory::new().then_output(serde_json::json!("Start with the user journey."));
        let mut chat = Chat::<String, String, Session>::from_snapshot(
            assistant,
            snapshot,
            Context::default().with_client_factory(next),
        )?;
        let mut state = chat.get()?;
        state.questions += 1;
        chat.set(state)?;
        println!(
            "{}",
            chat.send("What should I review first?".into())
                .await?
                .output
        );
        println!("Application state: {:?}", chat.get()?);
        Ok(())
    }
}

#[cfg(feature = "testing")]
#[tokio::main]
async fn main() -> Result<(), support::ExampleError> {
    example::run().await
}

#[cfg(not(feature = "testing"))]
fn main() {
    eprintln!("enable the 'testing' feature to run this deterministic example");
}
