//! Pause for approval, save a checkpoint, and resume after restoring it.
//!
//! No async runtime or external services are required.

mod support;

use pravah::{Flow, Step, compile};
use support::ExampleError;

fn approval(root: Flow<String>) -> Flow<String> {
    root.suspend::<bool>()
        .map(|approved| if approved { "Approved" } else { "Declined" }.to_owned())
}

/// Round-trips the pending approval through JSON before delivering the decision.
fn main() -> Result<(), ExampleError> {
    let workflow = compile(approval)?;
    let mut execution = workflow.start("Publish the report?".into(), uuid::Uuid::now_v7())?;
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
    execution.resume(true)?;

    loop {
        match execution.next()? {
            Step::Continue => {}
            Step::Done(value) => {
                println!("{}", workflow.decode_output(value)?);
                return Ok(());
            }
            _ => return Err(ExampleError::from("expected approval result")),
        }
    }
}
