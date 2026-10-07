use crate::graph::{AgentExecutor, GraphError, Runtime, Step};

/// Executes workers until the next VM-only observation, preserving failed accepted outcomes.
pub(crate) async fn step(
    runtime: &mut Runtime,
    executor: &AgentExecutor,
) -> Result<Step, GraphError> {
    loop {
        let next = match runtime.pending_agent() {
            Some(request) => Step::Agent(request.clone()),
            None => runtime.next()?,
        };
        match next {
            Step::Agent(request) => runtime.resume_agent(executor.execute(&request).await)?,
            step => return Ok(step),
        }
    }
}

/// Drives deterministic execution without hiding external input or adding retries.
pub(crate) async fn finish(
    runtime: &mut Runtime,
    executor: &AgentExecutor,
) -> Result<Step, GraphError> {
    loop {
        match step(runtime, executor).await? {
            Step::Continue => {}
            step => return Ok(step),
        }
    }
}
