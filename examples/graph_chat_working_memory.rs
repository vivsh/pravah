//! Fallible working-memory preparation with a deterministic client and summary policy.
//!
//! Run with `cargo run --example graph_chat_working_memory --features testing`.
//! No credentials or network services are required. Supply your own summarizer in an application.

#[cfg(feature = "testing")]
mod example {
    use pravah::clients::{Message, Role};
    use pravah::testing::ScriptedFactory;
    use pravah::{
        Agent, AgentConfig, Chat, Context, GraphError, HistoryEntry, HistoryPreparation,
        HistoryPreparer, HistoryReplacement,
    };

    #[derive(Debug, thiserror::Error)]
    enum MemoryError {
        #[error("completed history contains no user question")]
        MissingQuestion,
    }

    struct WorkingMemory;

    impl HistoryPreparer for WorkingMemory {
        type Error = MemoryError;

        async fn prepare(
            &self,
            request: HistoryPreparation<'_>,
        ) -> Result<HistoryReplacement, Self::Error> {
            if request.committed().is_empty() {
                return Ok(HistoryReplacement::default());
            }
            // The policy can also inspect request.model(), options(), protected(), and guidance.
            let summary = summarize(request.committed()).await?;
            Ok(HistoryReplacement {
                evict_indices: (0..request.committed().len()).collect(),
                summary: Some(summary),
            })
        }
    }

    /// Keeps one useful fact for this demo; applications supply their own fallible summarizer.
    async fn summarize(entries: &[&HistoryEntry]) -> Result<String, MemoryError> {
        let question = entries
            .iter()
            .rev()
            .find(|entry| matches!(entry.message.role, Role::User))
            .ok_or(MemoryError::MissingQuestion)?;
        Ok(format!(
            "The previous user request was: {}",
            question.message.content
        ))
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
        let mut chat = Chat::new(assistant, Context::default().with_client_factory(first))
            .with_history_preparer(WorkingMemory);
        println!(
            "{}",
            chat.send("I'd like to visit Kyoto.".into()).await?.output
        );
        let checkpoint = chat.snapshot()?;
        let next = ScriptedFactory::new()
            .then_output(serde_json::json!("Start with the eastern temples."));
        let mut chat = Chat::from_snapshot(
            assistant,
            checkpoint,
            Context::default().with_client_factory(next.clone()),
        )?
        .with_history_preparer(WorkingMemory);
        println!(
            "{}",
            chat.send("What should I see first?".into()).await?.output
        );
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
