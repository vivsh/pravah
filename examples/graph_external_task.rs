//! Typed external work uses ordinary suspension; no network service is required.
//! Persist the waiting snapshot before scheduling work and record its completion
//! before resuming. This example uses an in-memory host result.

mod support;

use pravah::{Flow, Step, compile};
use support::ExampleError;

fn external(root: Flow<String>) -> Flow<String> {
    root.suspend::<Result<String, String>>()
        .map(|result| match result {
            Ok(value) => value,
            Err(reason) => format!("External failure: {reason}"),
        })
}

/// Snapshots a typed request, restores it, and delivers a recorded host response.
fn main() -> Result<(), ExampleError> {
    let workflow = compile(external)?;
    let mut runtime = workflow.start("Process this document".into(), uuid::Uuid::now_v7())?;
    let Step::Suspend(request) = runtime.next()? else {
        return Err(ExampleError::from("expected task request"));
    };
    println!(
        "Task: {}",
        request
            .as_str()
            .ok_or_else(|| ExampleError::from("invalid task payload"))?
    );
    let bytes = serde_json::to_vec(&runtime.snapshot()?)?;
    let mut runtime = workflow.restore(serde_json::from_slice(&bytes)?)?;
    runtime.resume(Ok::<_, String>("Document processed".to_owned()))?;
    loop {
        match runtime.next()? {
            Step::Continue => {}
            Step::Done(value) => {
                println!("{}", workflow.decode_output(value)?);
                return Ok(());
            }
            _ => return Err(ExampleError::from("unexpected task boundary")),
        }
    }
}
