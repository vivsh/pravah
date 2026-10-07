//! Compose reusable functions, select a branch, and process a collection.
//!
//! No async runtime, provider credentials, or external services are needed.

mod support;

use either::Either;
use pravah::{Flow, Step, compile};
use support::ExampleError;

fn add_three(root: Flow<u32>) -> Flow<u32> {
    root.map(|score| score.saturating_add(3))
}

/// Reuses one subflow twice, then labels the result through two typed branches.
fn score(root: Flow<u32>) -> Flow<String> {
    root.flow(add_three)
        .flow(add_three)
        .either(|score| {
            if score >= 10 {
                Either::Left(score)
            } else {
                Either::Right(score)
            }
        })
        .branch(
            |high| high.map(|score| format!("{score}: high")),
            |low| low.map(|score| format!("{score}: low")),
        )
}

fn score_batch(root: Flow<Vec<u32>>) -> Flow<Vec<String>> {
    root.each(score)
}

/// Drives the workflow explicitly; every input is processed by the same reusable flow.
fn main() -> Result<(), ExampleError> {
    let workflow = compile(score_batch)?;
    let mut execution = workflow.start(vec![1, 5, 10], uuid::Uuid::now_v7())?;

    loop {
        match execution.next()? {
            Step::Continue => {}
            Step::Done(value) => {
                for score in workflow.decode_output(value)? {
                    println!("{score}");
                }
                return Ok(());
            }
            Step::Agent(_) | Step::Suspend(_) => {
                return Err(ExampleError::from("pure workflow requested input"));
            }
        }
    }
}
