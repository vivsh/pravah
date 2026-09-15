//! Fallible history compaction with message inspection, a deterministic client and summary policy.
//!
//! Run with `cargo run --example graph_chat_working_memory --features testing`.
//! No credentials or network services are required. Supply your own summarizer in an application.

#[cfg(feature = "testing")]
mod example {
    use pravah::clients::{Message, Role};
    use pravah::deps::{Deps, DepsError};
    use pravah::testing::ScriptedFactory;
    use pravah::{
        Agent, AgentConfig, Chat, CompactionRequest, CompactionResult, Compactor, Context,
        GraphError,
    };
    use std::sync::Arc;

    #[derive(Debug, thiserror::Error)]
    enum MemoryError {
        #[error(transparent)]
        Dependency(#[from] DepsError),
        #[error(transparent)]
        Size(#[from] serde_json::Error),
        #[error("completed history contains no user question")]
        MissingQuestion,
    }

    struct WorkingMemory;

    impl Compactor for WorkingMemory {
        type Error = MemoryError;

        async fn compact(
            &self,
            request: CompactionRequest<'_>,
            ctx: Context,
        ) -> Result<CompactionResult, Self::Error> {
            if request.turn_count() == 0 {
                return Ok(CompactionResult::default());
            }
            println!(
                "Compacting {} completed turns ({} JSON bytes)",
                request.turn_count(),
                request.byte_size()?
            );
            let summary = ctx
                .deps()
                .require::<Summarizer>()?
                .summarize(request.enum_messages(0).map(|(_, message)| message))
                .await?;
            Ok(CompactionResult {
                evict_indices: (0..request.committed().len()).collect(),
                summary: Some(summary),
            })
        }
    }

    struct Summarizer;

    impl Summarizer {
        /// Keeps one useful fact for this demo; applications supply their own fallible summarizer.
        async fn summarize<'a>(
            &self,
            messages: impl Iterator<Item = &'a Message> + Send,
        ) -> Result<String, MemoryError> {
            let question = messages
                .filter(|message| matches!(message.role, Role::User))
                .last()
                .ok_or(MemoryError::MissingQuestion)?;
            Ok(format!(
                "The previous user request was: {}",
                question.content
            ))
        }
    }

    fn context(client: ScriptedFactory) -> Context {
        let mut deps = Deps::default();
        deps.insert(Arc::new(Summarizer));
        Context::default()
            .with_deps(deps)
            .with_client_factory(client)
    }

    fn assistant(root: Agent<String>) -> Agent<String> {
        root.configure(configure)
    }

    async fn configure(input: String, _ctx: Context) -> Result<AgentConfig, GraphError> {
        Ok(AgentConfig::new(
            "openai:///scripted",
            "Help plan the trip.",
            Message::user(input),
        )
        .keep_alive())
    }

    /// Sends a question, restores the session with fresh dependencies, then prepares the next request.
    pub(super) async fn run() -> Result<(), GraphError> {
        let first =
            ScriptedFactory::new().then_output(serde_json::json!("We can plan a Kyoto trip."));
        let mut chat = Chat::new(assistant, context(first))
            .await?
            .with_compactor(WorkingMemory);
        println!("{}", chat.send("I'd like to visit Kyoto.").await?.output);
        let checkpoint = chat.snapshot()?;
        let next = ScriptedFactory::new()
            .then_output(serde_json::json!("Start with the eastern temples."));
        let mut chat = Chat::<_, _>::from_snapshot(assistant, checkpoint, context(next.clone()))?
            .with_compactor(WorkingMemory);
        println!("{}", chat.send("What should I see first?").await?.output);
        if let Some((_, messages)) = next.calls().first() {
            for message in messages {
                println!("{:?}: {}", message.role, message.content);
            }
        }
        Ok(())
    }
}

#[cfg(feature = "testing")]
#[tokio::main]
async fn main() -> Result<(), pravah::GraphError> {
    example::run().await
}

#[cfg(not(feature = "testing"))]
fn main() {
    eprintln!("Run with --features testing to enable the deterministic client.");
}
