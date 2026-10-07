use pravah::clients::{Message, Role};
use pravah::testing::ScriptedFactory;
use pravah::{
    Agent, AgentConfig, Chat, CompactionRequest, CompactionResult, Compactor, Context, GraphError,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[path = "history_preparation/context.rs"]
mod context;
#[path = "history_preparation/failures.rs"]
mod failures;
#[path = "history_preparation/inspection.rs"]
mod inspection;
#[path = "history_preparation/lifecycle.rs"]
mod lifecycle;
#[path = "history_preparation/summary.rs"]
mod summary;
#[path = "history_preparation/tools.rs"]
mod tools;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Question {
    text: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Answer {
    text: String,
}

struct Summarize;

impl Compactor for Summarize {
    type Error = std::convert::Infallible;

    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        assert_eq!(request.model(), "test:///test");
        assert!(
            request
                .options()
                .preamble
                .as_deref()
                .is_some_and(|p| p.contains("Answer briefly."))
        );
        assert!(matches!(
            request.protected().first().map(|e| &e.message.role),
            Some(Role::User)
        ));
        Ok(if request.committed().is_empty() {
            CompactionResult::default()
        } else {
            CompactionResult {
                evict_indices: (0..request.committed().len()).collect(),
                summary: Some("Prior conversation memory".into()),
            }
        })
    }
}

fn tutor(root: Agent<Question>) -> Agent<Answer> {
    root.configure(configure_tutor)
}

/// Configures a persistent deterministic chat session.
async fn configure_tutor(question: Question, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test",
        "Answer briefly.",
        Message::user(question.text),
    )
    .key("conversation"))
}

/// Verifies preparation replaces old exchanges before execution and protects the new input.
#[tokio::test]
async fn replacement_reaches_client_and_bounds_history() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!({"text":"first"}))
        .then_output(serde_json::json!({"text":"second"}));
    let mut chat = Chat::new(
        tutor,
        Context::default().with_providers(pravah::testing::providers(factory.clone())?),
    )?
    .with_compactor(Summarize);
    chat.send(Question {
        text: "first question".into(),
    })
    .await?;
    chat.send(Question {
        text: "second question".into(),
    })
    .await?;
    let calls = factory.calls();
    assert_eq!(calls.len(), 2);
    let messages = &calls[1].1;
    assert_eq!(messages.len(), 2);
    assert!(matches!(messages[0].role, Role::System));
    assert!(messages[0].content.contains("Prior conversation memory"));
    assert_eq!(messages[1].content, "second question");
    assert_eq!(chat.snapshot()?.history().entries().len(), 3);
    Ok(())
}
#[path = "support/host.rs"]
mod host;
use pravah::graph::AgentExecutor;
