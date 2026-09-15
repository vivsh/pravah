//! Callback-free chat, per-request memory, keyed messages and durable application state.
//!
//! Run with `cargo run --example graph_chat_builder --features testing`. No credentials needed.

#[cfg(feature = "testing")]
mod support;

#[cfg(feature = "testing")]
mod example {
    use super::support::ExampleError;
    use pravah::testing::{CapturingHistoryStore, ScriptedFactory};
    use pravah::{
        Chat, ChatBuilder, ChatRequest, CompactionRequest, CompactionResult, Compactor, Context,
        Snapshot,
    };

    struct WorkingMemory;

    impl Compactor for WorkingMemory {
        type Error = std::convert::Infallible;
        /// Inspects completed history without replacing any messages in this example.
        async fn compact(
            &self,
            request: CompactionRequest<'_>,
            _ctx: Context,
        ) -> Result<CompactionResult, Self::Error> {
            println!(
                "Completed turns available for compaction: {}",
                request.turn_count()
            );
            Ok(CompactionResult::default())
        }
    }

    fn assistant() -> ChatBuilder<String> {
        Chat::builder()
            .model("openai:///scripted")
            .instructions("Help the user review their project.")
            .turn_budget(4)
            .max_output_tokens(512)
    }

    /// Saves a keyed exchange and restores it with fresh services using JSON and CBOR.
    pub(super) async fn run() -> Result<(), ExampleError> {
        let factory =
            ScriptedFactory::new().then_output(serde_json::json!("Start with usability."));
        let ctx = Context::default().with_client_factory(factory);
        let mut chat = assistant()
            .state("Pravah".to_owned())
            .compactor(WorkingMemory)
            .store(CapturingHistoryStore::new())
            .build(ctx)
            .await?;
        println!(
            "Initial snapshot: {} bytes",
            serde_json::to_vec(&chat.snapshot()?)?.len()
        );
        chat.set("Pravah chat".to_owned())?;
        let request = ChatRequest::from("What should I review?").memory(format!(
            "Current project: {}. Prefer concise answers.",
            chat.get()?
        ));
        println!(
            "{}",
            chat.send_with_key(request, "message-42").await?.output
        );
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
        for snapshot in copies {
            continue_chat(snapshot).await?;
        }
        Ok(())
    }

    /// Reuses the definition, not live clients; memory is explicitly supplied per invocation.
    async fn continue_chat(snapshot: Snapshot) -> Result<(), ExampleError> {
        let factory =
            ScriptedFactory::new().then_output(serde_json::json!("Test the first conversation."));
        let ctx = Context::default().with_client_factory(factory);
        let mut chat = assistant()
            .compactor(WorkingMemory)
            .store(CapturingHistoryStore::new())
            .restore::<String>(snapshot, ctx)?;
        println!("Restored project: {}", chat.get()?);
        println!("{}", chat.send("How do I start?").await?.output);
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
