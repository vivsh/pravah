//! Replace completed exchanges with working memory before the next model request.
//!
//! Run with --features testing. The small policy retains the first user request;
//! a real application supplies its own summarizer or fact extractor.

use pravah::clients::Role;
use pravah::testing::ScriptedFactory;
use pravah::{
    Chat, CompactionRequest, CompactionResult, Compactor, Context, GraphError, HistoryManager,
    MessageHistory,
};

#[derive(Debug, thiserror::Error)]
#[error("completed history contains no user question")]
struct MissingQuestion;

struct WorkingMemory;

impl Compactor for WorkingMemory {
    type Error = MissingQuestion;

    /// Skip preparation until there is a completed exchange to consolidate.
    fn needs_compaction(&self, history: &MessageHistory, session_id: &str) -> bool {
        history.turn_count(session_id) > 0
    }

    /// Keeps existing memory or extracts it from the first completed user message.
    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        if request.turn_count() == 0 {
            return Ok(CompactionResult::default());
        }
        let summary = match request.summary() {
            Some(previous) => previous.to_owned(),
            None => {
                let (_, message) = request
                    .enum_messages(0)
                    .find(|(_, message)| matches!(message.role, Role::User))
                    .ok_or(MissingQuestion)?;
                format!("Remember the user's initial request: {}", message.content)
            }
        };
        println!("Working memory: {summary}");

        // Replace the whole committed prefix, including any previous summary.
        // The pending input is protected and is not in committed().
        Ok(CompactionResult {
            evict_indices: (0..request.committed().len()).collect(),
            summary: Some(summary),
        })
    }
}

/// Registers the policy before build; each send prepares history before model execution.
#[tokio::main]
async fn main() -> Result<(), GraphError> {
    let client = ScriptedFactory::new()
        .then_output(serde_json::json!("Let's plan a Kyoto trip."))
        .then_output(serde_json::json!("Start with the eastern temples."))
        .then_output(serde_json::json!("Allow three days to explore."));
    let ctx = Context::default().with_providers(pravah::testing::providers(client)?);
    let mut chat = Chat::builder::<String, String>()
        .model("test:///scripted")
        .instructions("Help plan the trip.")
        .history_manager(HistoryManager::new().with_compactor(WorkingMemory))
        .build(ctx)?;

    for question in [
        "I'd like to visit Kyoto.",
        "What should I see first?",
        "How long should I stay?",
    ] {
        println!("{}", chat.send(question).await?.output);
    }
    Ok(())
}
