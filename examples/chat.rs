//! Simple multi-turn chat using [`Chat`].
//!
//! Demonstrates: builder, two turns, snapshot/restore, third turn confirming history continuity.
//!
//! Requires a provider API key: set `GEMINI_API_KEY` or change the URL to another provider.

use pravah::{Chat, Context, FlowConf};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let ctx = Context::new(FlowConf::default());

    let mut session: Chat = Chat::builder("gemini:///gemini-2.5-flash-lite")
        .preamble("You are a concise Rust tutor. Keep answers to two sentences.")
        .build()?;

    println!("--- turn 1 ---");
    let t1 = session
        .send(ctx.clone(), "What is ownership in Rust?")
        .await?;
    println!("{}", t1.text());
    if let Some(u) = t1.usage {
        println!("(tokens: in={:?} out={:?})", u.input, u.output);
    }

    println!("\n--- turn 2 ---");
    let t2 = session
        .send(
            ctx.clone(),
            "Give me one short example of the rule you just described.",
        )
        .await?;
    println!("{}", t2.text());

    // Take a snapshot and restore to a fresh session.
    let snap = session.snapshot();
    println!("\n--- restored from snapshot ---");
    let mut restored = Chat::from_snapshot(snap)?;

    let t3 = restored
        .send(
            ctx.clone(),
            "Summarise everything we discussed in one sentence.",
        )
        .await?;
    println!("{}", t3.text());

    println!("\nsession_id: {}", restored.session_id());
    println!("history entries: {}", restored.history().entries().len());

    Ok(())
}
