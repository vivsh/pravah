//! A typed chat with request memory, message keys, and persistent application state.
//!
//! Run with --features testing. Scripted replies make this example fully offline.

mod support;

use pravah::testing::ScriptedFactory;
use pravah::{Chat, ChatBuilder, ChatRequest, Context};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use support::ExampleError;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Question {
    topic: String,
}

fn assistant() -> ChatBuilder<Question, String> {
    Chat::builder()
        .model("test:///scripted")
        .instructions("Suggest one practical next step.")
}

fn context(reply: &str) -> Result<Context, pravah::GraphError> {
    let client = ScriptedFactory::new().then_output(serde_json::json!(reply));
    Ok(Context::default().with_providers(pravah::testing::providers(client)?))
}

/// Saves one conversation, restores it with fresh dependencies, then sends another question.
#[tokio::main]
async fn main() -> Result<(), ExampleError> {
    let mut chat = assistant()
        .state("Pravah".to_owned())
        .build(context("Start by reviewing the examples.")?)?;
    chat.set("Pravah documentation".to_owned())?;

    let question = ChatRequest::from(Question { topic: chat.get()? })
        .memory("The reader prefers short, runnable examples.");
    let reply = chat.send_with_key(question, "question-42").await?;
    println!("{}", reply.output);

    let saved = serde_json::to_vec(&chat.snapshot()?)?;
    let mut chat = assistant().restore::<String>(
        serde_json::from_slice(&saved)?,
        context("Keep each example focused on one feature.")?,
    )?;
    println!("Restored project: {}", chat.get()?);

    let reply = chat.send(Question { topic: chat.get()? }).await?;
    println!("{}", reply.output);
    Ok(())
}
