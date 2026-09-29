//! Synchronous external requests, pending snapshots, and response/failure delivery.
//!
//! No runtime or network is needed: the host supplies deterministic outcomes.
//! Real hosts persist the pending snapshot before dispatch and durably record
//! outcomes before delivery; this example only round-trips the snapshot in memory.

mod support;

use pravah::{FetchError, FetchRequest, FetchResponse, Flow, Step, compile};
use support::ExampleError;

fn status(root: Flow<FetchRequest>) -> Flow<String> {
    root.fetch().map(|outcome| match outcome {
        Ok(response) => format!("HTTP status {}", response.status()),
        Err(error) => format!("External failure: {}", error.code()),
    })
}

/// Demonstrates that both HTTP error statuses and transport failures reach the flow.
fn main() -> Result<(), ExampleError> {
    for outcome in [
        Ok(FetchResponse::new(200)),
        Ok(FetchResponse::new(503)),
        Err(FetchError::new("offline", "No connection")),
    ] {
        run(outcome)?;
    }
    Ok(())
}

/// Restores the pending request, delivers its recorded outcome, and finishes execution.
fn run(outcome: Result<FetchResponse, FetchError>) -> Result<(), ExampleError> {
    let workflow = compile(status)?;
    let mut execution = workflow.start(
        FetchRequest::new("GET", "https://example.com/status"),
        uuid::Uuid::now_v7(),
    )?;
    loop {
        match execution.next()? {
            Step::Continue => {}
            Step::Fetch(_) => break,
            _ => return Err(ExampleError::from("expected an external request")),
        }
    }
    let saved = serde_json::to_vec(&execution.snapshot()?)?;
    let mut execution = workflow.restore(serde_json::from_slice(&saved)?)?;
    let id = execution
        .pending_fetch()
        .ok_or_else(|| ExampleError::from("missing restored request"))?
        .id();
    execution.resume_fetch(id, outcome)?;
    loop {
        match execution.next()? {
            Step::Continue => {}
            Step::Done(value) => {
                println!("{}", workflow.decode_output(value)?);
                return Ok(());
            }
            _ => {
                return Err(ExampleError::from("expected completion after delivery"));
            }
        }
    }
}
