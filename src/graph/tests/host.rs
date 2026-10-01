use crate::graph::{FetchExecutor, GraphError, Runtime, Step};

/// Keeps maintenance external while preserving the unit-test host's single-observation stepping.
pub(crate) async fn step_with_manager(
    runtime: &mut Runtime,
    executor: &FetchExecutor,
    manager: &mut crate::HistoryManager,
) -> Result<Step, GraphError> {
    manager
        .maintain(runtime, executor.context().clone())
        .await?;
    let outcome = step(runtime, executor).await?;
    if matches!(outcome, Step::Done(_)) {
        manager
            .maintain(runtime, executor.context().clone())
            .await?;
    }
    Ok(outcome)
}

/// Executes external requests locally until the next VM-only observation.
/// A failed executor call remains pending; the test chooses whether to retry it.
pub(crate) async fn step(
    runtime: &mut Runtime,
    executor: &FetchExecutor,
) -> Result<Step, GraphError> {
    loop {
        let next = match runtime.pending_fetch() {
            Some(fetch) => Step::Fetch(fetch.clone()),
            None => runtime.next()?,
        };
        match next {
            Step::Fetch(fetch) => {
                let response = executor.execute(&fetch).await?;
                runtime.resume_fetch(fetch.id(), Ok(response))?;
            }
            step => return Ok(step),
        }
    }
}

/// Drives a complete test operation without hiding suspensions or retrying failures.
pub(crate) async fn finish(
    runtime: &mut Runtime,
    executor: &FetchExecutor,
) -> Result<Step, GraphError> {
    loop {
        match step(runtime, executor).await? {
            Step::Continue => {}
            step => return Ok(step),
        }
    }
}
