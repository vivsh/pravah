//! Graph-backed typed chat with snapshot restoration between turns.
//!
//! Requires `GEMINI_API_KEY`, or change the model URL in `configure_tutor`.

use pravah::clients::Message;
use pravah::{Agent, AgentConfig, Chat, Context, GraphError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Question {
    text: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Answer {
    text: String,
}

fn tutor(root: Agent<Question>) -> Agent<Answer> {
    root.configure(configure_tutor)
}

/// Configures one typed conversational turn while retaining session history.
async fn configure_tutor(question: Question, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "gemini:///gemini-2.5-flash-lite",
        "You are a concise Rust tutor. Keep answers to two sentences.",
        Message::user(question.text),
    )
    .keep_alive())
}

#[tokio::main]
async fn main() -> Result<(), GraphError> {
    dotenvy::dotenv().ok();
    let ctx = Context::default();
    let mut chat = Chat::new(tutor, ctx);

    let first = chat
        .send(Question {
            text: "What is ownership in Rust?".into(),
        })
        .await?;
    println!("{}", first.output.text);

    let snapshot = chat.snapshot()?;
    let mut restored = Chat::from_snapshot(tutor, snapshot, Context::default())?;
    let second = restored
        .send(Question {
            text: "Give me one short example of that rule.".into(),
        })
        .await?;
    println!("{}", second.output.text);
    Ok(())
}
