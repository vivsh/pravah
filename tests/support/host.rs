use pravah::{AgentExecutor, GraphError, Runtime, Step};

/// Performs one synchronous instruction or accepts one complete worker response, without retries.
pub async fn step(runtime: &mut Runtime, executor: &AgentExecutor) -> Result<Step, GraphError> {
    if let Some(request) = runtime.pending_agent() {
        let response = executor.execute(request).await;
        runtime.resume_agent(response)?;
        return Ok(Step::Continue);
    }
    match runtime.next()? {
        Step::Agent(_) => Ok(Step::Continue),
        step => Ok(step),
    }
}

/// Completes a deterministic flow without hiding suspensions or retrying worker failures.
#[allow(dead_code)]
pub async fn finish(runtime: &mut Runtime, executor: &AgentExecutor) -> Result<Step, GraphError> {
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
