//! Pause for approval, retain application state, and resume a saved checkpoint.
//!
//! No async runtime or external services are required.

mod support;

use pravah::{Flow, Step, compile};
use support::ExampleError;

#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct ReviewState {
    reviewer: String,
}

fn approval(root: Flow<String>) -> Flow<String> {
    root.suspend::<bool>()
        .map(|approved| if approved { "Approved" } else { "Declined" }.to_owned())
}

/// Round-trips the pending approval through JSON before delivering the decision.
fn main() -> Result<(), ExampleError> {
    let workflow = compile(approval)?;
    let mut execution = workflow.start_with_state(
        "Publish the report?".into(),
        ReviewState {
            reviewer: "Alice".into(),
        },
        uuid::Uuid::now_v7(),
    )?;
    loop {
        match execution.next()? {
            Step::Continue => {}
            Step::Suspend(question) => {
                println!("Approval request: {question}");
                break;
            }
            _ => return Err(ExampleError::from("expected approval suspension")),
        }
    }

    let saved = serde_json::to_vec(&execution.snapshot()?)?;
    let mut execution = workflow.restore(serde_json::from_slice(&saved)?)?;
    execution.set_state(ReviewState {
        reviewer: "Bob".into(),
    })?;
    execution.resume(true)?;

    loop {
        match execution.next()? {
            Step::Continue => {}
            Step::Done(value) => {
                let state: ReviewState = execution.get_state()?;
                println!("{} by {}", workflow.decode_output(value)?, state.reviewer);
                return Ok(());
            }
            _ => return Err(ExampleError::from("expected approval result")),
        }
    }
}
