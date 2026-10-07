use super::*;
use crate::graph::agent_request::AgentOperation;

struct ContextDelta(i64);

/// Keeps asynchronous dependencies outside the prepared graph and execution snapshots.
fn executor(flow: &PreparedGraph, delta: i64) -> AgentExecutor {
    let mut deps = Deps::default();
    deps.insert(Arc::new(ContextDelta(delta)));
    flow.executor(ctx().with_deps(deps))
}

#[derive(Default)]
struct AddDelta;
impl ContinuationHandler for AddDelta {
    fn start<'a>(
        &'a self,
        _: &'a Value,
        _: Option<Value>,
        inputs: Vec<Value>,
        _: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let input = inputs
            .into_iter()
            .next()
            .ok_or_else(|| GraphError::Invalid("missing amount".into()))?;
        Ok(ContinuationTransition {
            checkpoint: Some(1_u32.into()),
            agent: Some(AgentRequest::new(
                uuid::Uuid::nil(),
                AgentOperation::Tool {
                    handler: HandlerKey::new("delta"),
                    input,
                },
            )),
            ..Default::default()
        })
    }
    fn advance<'a>(
        &'a self,
        _: &'a Value,
        _: Value,
        event: ContinuationEvent,
        _: ContinuationContext<'_>,
    ) -> Result<ContinuationTransition, GraphError> {
        let ContinuationEvent::Agent { response, .. } = event else {
            return Err(GraphError::Invalid("missing completion".into()));
        };
        let output = response
            .outcome
            .map_err(|source| GraphError::AgentFailed { source })?;
        Ok(ContinuationTransition {
            outputs: vec![output],
            ..Default::default()
        })
    }
}
impl DynAgentHandler for AddDelta {
    fn execute<'a>(
        &'a self,
        request: &'a AgentRequest,
        context: Context,
    ) -> BoxFuture<'a, Result<Value, GraphError>> {
        Box::pin(async move {
            let AgentOperation::Tool { input, .. } = request.operation.as_ref() else {
                return Err(GraphError::Invalid("missing tool input".into()));
            };
            let mut amount: TypedAmount = from_value(input.clone())
                .map_err(|error| GraphError::Invalid(error.to_string()))?;
            let delta = context
                .require::<ContextDelta>()
                .map_err(|error| GraphError::Invalid(error.to_string()))?;
            amount.value = amount
                .value
                .checked_add(delta.0)
                .ok_or_else(|| GraphError::Invalid("amount overflow".into()))?;
            to_value(amount).map_err(|error| GraphError::Invalid(error.to_string()))
        })
    }
}

/// Compiles two callback boundaries without embedding runtime dependencies or generic HTTP effects.
fn context_bound_amount() -> Result<PreparedGraph, GraphError> {
    let builder = TypedGraphBuilder::<TypedAmount>::new();
    let first = builder.continuation::<TypedAmount, TypedAmount, AddDelta, _>(builder.root(), ());
    let second = builder.continuation::<TypedAmount, TypedAmount, AddDelta, _>(first, ());
    let (graph, mut registry) = builder.finish(second)?.into_parts();
    registry.insert_agent("delta", AddDelta)?;
    PreparedGraph::new(graph, registry)
}

/// Every effect uses the selected host dependencies, independently across runtimes.
#[tokio::test]
async fn execution_dependencies_are_external_and_isolated() {
    let flow = context_bound_amount().unwrap();
    for (delta, expected) in [(2, 5), (3, 7)] {
        let executor = executor(&flow, delta);
        let mut runtime = flow
            .start(
                to_value(TypedAmount { value: 1 }).unwrap(),
                uuid::Uuid::nil(),
            )
            .unwrap();
        let Step::Done(output) = host::finish(&mut runtime, &executor).await.unwrap() else {
            panic!("execution should complete");
        };
        assert_eq!(from_value::<TypedAmount>(output).unwrap().value, expected);
    }
}

/// A pending request restores with new host dependencies, not serialized dependencies.
#[tokio::test]
async fn restore_uses_fresh_executor() {
    let flow = context_bound_amount().unwrap();
    let mut runtime = flow
        .start(
            to_value(TypedAmount { value: 1 }).unwrap(),
            uuid::Uuid::nil(),
        )
        .unwrap();
    assert!(matches!(runtime.next().unwrap(), Step::Agent(_)));
    let snapshot = runtime.snapshot().unwrap();
    assert!(
        !serde_json::to_string(&snapshot)
            .unwrap()
            .contains("ContextDelta")
    );
    let mut restored = flow.restore(snapshot).unwrap();
    let Step::Done(output) = host::finish(&mut restored, &executor(&flow, 10))
        .await
        .unwrap()
    else {
        panic!("execution should complete");
    };
    assert_eq!(from_value::<TypedAmount>(output).unwrap().value, 21);
}

/// Constructing different executors has no effect on deterministic snapshot encoding.
#[test]
fn snapshots_are_independent_of_executor_dependencies() {
    let flow = context_bound_amount().unwrap();
    let snapshots = [2, 9].map(|delta| {
        let _executor = executor(&flow, delta);
        flow.start(
            to_value(TypedAmount { value: 1 }).unwrap(),
            uuid::Uuid::nil(),
        )
        .unwrap()
        .snapshot()
        .unwrap()
    });
    assert_eq!(
        serde_json::to_vec(&snapshots[0]).unwrap(),
        serde_json::to_vec(&snapshots[1]).unwrap()
    );
    let mut first = Vec::new();
    let mut second = Vec::new();
    ciborium::into_writer(&snapshots[0], &mut first).unwrap();
    ciborium::into_writer(&snapshots[1], &mut second).unwrap();
    assert_eq!(first, second);
}
