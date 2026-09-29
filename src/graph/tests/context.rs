use super::*;
use crate::graph::fetch::DynFetchHandler;

struct ContextDelta(i64);

/// Keeps the arithmetic dependency outside runtime state.
fn executor(flow: &CompiledFlow<TypedAmount, TypedAmount>, delta: i64) -> FetchExecutor {
    let mut deps = Deps::default();
    deps.insert(Arc::new(ContextDelta(delta)));
    let mut executor = FetchExecutor::new(ctx().with_deps(deps), Arc::new(flow.registry().clone()));
    executor.register("delta", AddDelta).unwrap();
    assert!(executor.register("delta", AddDelta).is_err());
    executor
}

struct AddDelta;

impl DynFetchHandler for AddDelta {
    fn execute<'a>(
        &'a self,
        fetch: &'a Fetch,
        context: Context,
    ) -> BoxFuture<'a, Result<FetchResponse, GraphError>> {
        Box::pin(async move {
            let Some(FetchBody::Value(value)) = fetch.request().body_ref() else {
                return Err(GraphError::FetchValidation("missing amount".into()));
            };
            let mut amount: TypedAmount = from_value(value.clone()).unwrap();
            let delta = context
                .require::<ContextDelta>()
                .map_err(|error| GraphError::Invalid(error.to_string()))?;
            amount.value += delta.0;
            Ok(FetchResponse::new(200).body(FetchBody::Value(to_value(amount).unwrap())))
        })
    }
}

fn request(amount: TypedAmount) -> FetchRequest {
    FetchRequest::new("POST", "delta://add").body(FetchBody::Value(to_value(amount).unwrap()))
}

fn response(outcome: Result<FetchResponse, FetchError>) -> TypedAmount {
    let response = outcome.unwrap();
    let Some(FetchBody::Value(value)) = response.body_ref() else {
        panic!("missing amount")
    };
    from_value(value.clone()).unwrap()
}

fn context_bound_amount(root: Flow<TypedAmount>) -> Flow<TypedAmount> {
    root.map(request)
        .fetch()
        .map(response)
        .map(request)
        .fetch()
        .map(response)
}

/// Every effect uses the selected host dependencies, independently across runtimes.
#[tokio::test]
async fn execution_dependencies_are_external_and_isolated() {
    let flow = compile(context_bound_amount).unwrap();
    for (delta, expected) in [(2, 5), (3, 7)] {
        let executor = executor(&flow, delta);
        let mut runtime = flow
            .start(TypedAmount { value: 1 }, uuid::Uuid::nil())
            .unwrap();
        let Step::Done(output) = host::finish(&mut runtime, &executor).await.unwrap() else {
            panic!("execution should complete");
        };
        assert_eq!(flow.decode_output(output).unwrap().value, expected);
    }
}

/// A pending request restores with new host dependencies, not serialized dependencies.
#[tokio::test]
async fn restore_uses_fresh_executor() {
    let flow = compile(context_bound_amount).unwrap();
    let mut runtime = flow
        .start(TypedAmount { value: 1 }, uuid::Uuid::nil())
        .unwrap();
    runtime.next().unwrap();
    assert!(matches!(runtime.next().unwrap(), Step::Fetch(_)));
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
    assert_eq!(flow.decode_output(output).unwrap().value, 21);
}

/// Constructing different executors has no effect on deterministic snapshot encoding.
#[test]
fn snapshots_are_independent_of_executor_dependencies() {
    let flow = compile(context_bound_amount).unwrap();
    let snapshots = [2, 9].map(|delta| {
        let _executor = executor(&flow, delta);
        flow.start(TypedAmount { value: 1 }, uuid::Uuid::nil())
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
