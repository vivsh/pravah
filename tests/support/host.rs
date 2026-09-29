use pravah::graph::FetchExecutor;
use pravah::{GraphError, Runtime, Step};

/// Performs one VM operation or acknowledges one pending local effect, without retries.
pub async fn step(runtime: &mut Runtime, executor: &FetchExecutor) -> Result<Step, GraphError> {
    if let Some(fetch) = runtime.pending_fetch() {
        let id = fetch.id();
        let response = executor.execute(fetch).await?;
        runtime.resume_fetch(id, Ok(response))?;
        return Ok(Step::Continue);
    }
    match runtime.next()? {
        Step::Fetch(_) => Ok(Step::Continue),
        step => Ok(step),
    }
}

/// Completes a deterministic test flow, preserving errors and external suspension.
#[allow(dead_code)]
pub async fn finish(runtime: &mut Runtime, executor: &FetchExecutor) -> Result<Step, GraphError> {
    for _ in 0..200 {
        match step(runtime, executor).await? {
            Step::Continue => {}
            step => return Ok(step),
        }
    }
    Err(GraphError::Invalid(
        "test execution did not complete".into(),
    ))
}
