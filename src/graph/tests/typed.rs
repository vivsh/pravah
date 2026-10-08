use super::*;
use crate::{AgentConfig, Context, clients::Message, testing::ScriptedFactory};
use serde::Deserialize;

#[derive(Serialize, Deserialize, JsonSchema)]
struct Number(u32);

/// Cloning requires neither input nor output Clone and shares preparation without allocating.
#[test]
fn clones_share_preparation_without_allocations() -> Result<(), GraphError> {
    let flow = compile(|root: Flow<Number>| root.map(|number| number))?;
    let measured = allocation_counter::measure(|| {
        let cloned = flow.clone();
        assert!(std::ptr::eq(flow.graph(), cloned.graph()));
        assert!(std::ptr::eq(flow.registry(), cloned.registry()));
        assert_eq!(
            flow.prepared().fingerprint(),
            cloned.prepared().fingerprint()
        );
    });
    assert_eq!(measured.count_total, 0);
    assert_eq!(measured.bytes_total, 0);
    Ok(())
}

/// A clone restores the original snapshot after its definition drops without sharing progress.
#[test]
fn cloned_definition_restores_isolated_executions() -> Result<(), GraphError> {
    let flow = compile(|root: Flow<Number>| root.suspend::<Number>())?;
    let cloned = flow.clone();
    let mut first = flow.start(Number(1), Uuid::from_u128(1))?;
    assert!(matches!(first.next()?, crate::Step::Suspend(_)));
    let snapshot = first.snapshot()?;
    let encoded = serde_json::to_vec(&snapshot).map_err(codec)?;
    drop(flow);
    let snapshot = serde_json::from_slice(&encoded).map_err(codec)?;
    let mut restored = cloned.restore(snapshot)?;
    let mut second = cloned.start(Number(2), Uuid::from_u128(2))?;
    assert!(matches!(second.next()?, crate::Step::Suspend(_)));
    restored.resume(Number(10))?;
    assert!(second.suspension().is_some());
    second.resume(Number(20))?;
    assert_eq!(completed(&cloned, &mut restored)?.0, 10);
    assert_eq!(completed(&cloned, &mut second)?.0, 20);
    assert!(first.suspension().is_some());
    assert_eq!(restored.state().execution_id(), Uuid::from_u128(1));
    assert_eq!(second.state().execution_id(), Uuid::from_u128(2));
    Ok(())
}

/// One factory preparation supplies captured callbacks to the executor and cloned executions.
#[tokio::test]
async fn cloned_flow_uses_original_agent_callbacks() -> Result<(), GraphError> {
    let mut preparations = 0;
    let mut factory = |root| counted_agent(root, &mut preparations);
    let flow = factory(Flow::root()).finish::<Number>()?;
    let script = ScriptedFactory::new()
        .then_output(serde_json::json!(10))
        .then_output(serde_json::json!(20));
    let context = Context::default().with_providers(crate::testing::providers(script.clone())?);
    let executor = flow.prepared().executor(context);
    let registration = move || flow.clone();
    for (input, expected) in [(1, 10), (2, 20)] {
        let definition = registration();
        let mut runtime = definition.start(Number(input), Uuid::from_u128(input.into()))?;
        let request = next_agent(&mut runtime)?;
        assert_eq!(request.kind(), "configure");
        let snapshot = serde_json::to_vec(&runtime.snapshot()?).map_err(codec)?;
        let mut runtime = definition.restore(serde_json::from_slice(&snapshot).map_err(codec)?)?;
        runtime.resume_agent(executor.execute(&request).await)?;
        let crate::Step::Done(output) =
            crate::graph::tests::host::finish(&mut runtime, &executor).await?
        else {
            return Err(GraphError::Invalid("expected completed agent".into()));
        };
        assert_eq!(definition.decode_output(output)?.0, expected);
    }
    assert_eq!(preparations, 1);
    let calls = script.calls();
    assert_eq!(calls.len(), 2);
    for ((_, messages), expected) in calls.into_iter().zip(["2", "3"]) {
        assert!(messages.iter().any(|message| message.content == expected));
    }
    Ok(())
}

/// Captures the construction count so worker calls reveal accidental factory rebuilding.
fn counted_agent(root: Flow<Number>, preparations: &mut usize) -> Flow<Number> {
    *preparations += 1;
    let offset = *preparations as u32;
    root.map(move |input| Number(input.0 + offset))
        .agent(|agent| agent.configure(configure))
}

async fn configure(input: Number, _: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///model",
        "Return a number",
        Message::user(input.0.to_string()),
    ))
}

/// Finds the worker boundary after the captured pure transform.
fn next_agent(runtime: &mut Runtime) -> Result<crate::AgentRequest, GraphError> {
    loop {
        match runtime.next()? {
            crate::Step::Continue => {}
            crate::Step::Agent(request) => return Ok(request),
            _ => return Err(GraphError::Invalid("expected configure work".into())),
        }
    }
}

/// Extracts the completed output of a pure test flow.
fn completed(
    flow: &CompiledFlow<Number, Number>,
    runtime: &mut Runtime,
) -> Result<Number, GraphError> {
    match runtime.next()? {
        crate::Step::Done(output) => flow.decode_output(output),
        _ => Err(GraphError::Invalid("expected completed flow".into())),
    }
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::ValueConversion {
        target: "typed clone test checkpoint".into(),
        reason: error.to_string(),
    }
}
