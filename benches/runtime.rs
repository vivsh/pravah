use std::hint::black_box;
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use pravah::graph::{
    BuiltinNode, ContinuationContext, ContinuationEvent, ContinuationHandler,
    ContinuationTransition, EdgeId, GraphError, HandlerKey, HandlerRegistry, JSON_WIRE_VERSION,
    JsonInvoker, JsonRequest, JsonResponse, NodeKind, PreparedGraph, Snapshot, Step, TypeSpec,
    UntypedGraph, UntypedGraphBuilder, Value, VarId, VarKey, VarScope, from_value, to_value,
};
use pravah::{Context, FlowConf};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[cfg(not(debug_assertions))]
const FAST_ITERATIONS: usize = 100_000;
#[cfg(debug_assertions)]
const FAST_ITERATIONS: usize = 100;
#[cfg(not(debug_assertions))]
const VM_ITERATIONS: usize = 10_000;
#[cfg(debug_assertions)]
const VM_ITERATIONS: usize = 10;
#[cfg(not(debug_assertions))]
const SYNC_SAMPLES: usize = 5;
#[cfg(debug_assertions)]
const SYNC_SAMPLES: usize = 1;
#[cfg(not(debug_assertions))]
const VM_SAMPLES: usize = 3;
#[cfg(debug_assertions)]
const VM_SAMPLES: usize = 1;

#[derive(Clone, Serialize, Deserialize)]
struct TypedFixture {
    id: u64,
    name: String,
    values: Vec<TypedItem>,
}

#[derive(Clone, Serialize, Deserialize)]
struct TypedItem {
    code: String,
    amount: i64,
}

#[derive(Default)]
struct AddContinuation;

impl ContinuationHandler for AddContinuation {
    fn start<'a>(
        &'a self,
        payload: &'a Value,
        _state: Option<Value>,
        inputs: Vec<Value>,
        _ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            let input = required_i64(inputs.first(), "continuation input")?;
            let add = required_i64(payload.get("add"), "continuation payload")?;
            Ok(ContinuationTransition {
                outputs: vec![Value::from(input + add)],
                ..ContinuationTransition::default()
            })
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _checkpoint: Value,
        _event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async {
            Err(GraphError::Invalid(
                "benchmark continuation cannot advance".into(),
            ))
        })
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("runtime benchmark failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), GraphError> {
    report_value_benchmarks()?;
    report_typed_benchmarks()?;
    report_runtime_benchmarks().await
}

fn report_value_benchmarks() -> Result<(), GraphError> {
    println!(
        "value/layout: {} byte(s); serde_json::Value baseline: {} byte(s)",
        std::mem::size_of::<Value>(),
        std::mem::size_of::<serde_json::Value>()
    );
    let nested = nested_value()?;
    let nested_json = serde_json::to_value(&nested).map_err(|error| GraphError::JsonEncode {
        target: "benchmark baseline value".into(),
        reason: error.to_string(),
    })?;
    report_allocations("value/scalar_clone", || {
        black_box(Value::from(7_i64).clone())
    });
    report_allocations("value/nested_clone", || black_box(nested.clone()));
    report_allocations("baseline/serde_json_nested_clone", || {
        black_box(nested_json.clone())
    });
    report_sync("value/scalar_clone", FAST_ITERATIONS, || {
        black_box(Value::from(7_i64).clone())
    });
    report_sync("value/nested_clone", FAST_ITERATIONS, || {
        black_box(nested.clone())
    });
    report_sync("baseline/serde_json_nested_clone", FAST_ITERATIONS, || {
        black_box(nested_json.clone())
    });
    Ok(())
}

fn typed_fixture() -> TypedFixture {
    TypedFixture {
        id: 42,
        name: "pravah".repeat(4),
        values: (0_i64..32)
            .map(|amount| TypedItem {
                code: format!("item-{amount}"),
                amount,
            })
            .collect(),
    }
}

fn report_typed_benchmarks() -> Result<(), GraphError> {
    let fixture = typed_fixture();
    report_sync("value/typed_roundtrip", FAST_ITERATIONS, || {
        to_value(black_box(&fixture)).and_then(from_value::<TypedFixture>)
    });
    report_sync(
        "baseline/serde_json_typed_roundtrip",
        FAST_ITERATIONS,
        || {
            serde_json::to_value(black_box(&fixture))
                .and_then(serde_json::from_value::<TypedFixture>)
        },
    );
    let fixture_value = to_value(&fixture).map_err(|error| GraphError::ValueConversion {
        target: "benchmark typed fixture".into(),
        reason: error.to_string(),
    })?;
    let fixture_json = serde_json::to_value(&fixture).map_err(|error| GraphError::JsonEncode {
        target: "benchmark typed fixture".into(),
        reason: error.to_string(),
    })?;
    report_sync("vm/typed_handler_roundtrip", FAST_ITERATIONS, || {
        from_value::<TypedFixture>(fixture_value.clone()).and_then(to_value)
    });
    report_sync(
        "baseline/serde_json_typed_handler_roundtrip",
        FAST_ITERATIONS,
        || {
            serde_json::from_value::<TypedFixture>(fixture_json.clone())
                .and_then(serde_json::to_value)
        },
    );
    Ok(())
}

async fn report_runtime_benchmarks() -> Result<(), GraphError> {
    let fanout_graph = fanout_tuple_graph()?;
    let fanout = PreparedGraph::new(fanout_graph.clone(), HandlerRegistry::new())?;
    report_vm("vm/fanout_tuple", VM_ITERATIONS, &fanout).await?;
    report_prepare_each_start(
        "baseline/vm_fanout_compile_each_start",
        VM_ITERATIONS / 10,
        &fanout_graph,
    )
    .await?;

    let live_identity = PreparedGraph::new(identity_graph()?, HandlerRegistry::new())?;
    let dead_shaping = PreparedGraph::new(dead_shaping_graph()?, HandlerRegistry::new())?;
    report_vm("dce/live_identity", VM_ITERATIONS, &live_identity).await?;
    report_vm("dce/dead_shaping_identity", VM_ITERATIONS, &dead_shaping).await?;

    let mixed = prepared_mixed_graph()?;
    report_vm("vm/variable_subflow_continuation", VM_ITERATIONS, &mixed).await?;
    report_snapshot_restore("vm/snapshot_restore", VM_ITERATIONS, &mixed).await?;
    report_json("json/stateless_start", VM_ITERATIONS).await?;
    report_json_next("json/stateless_next", VM_ITERATIONS).await?;
    report_json_resume("json/stateless_resume", VM_ITERATIONS).await?;
    Ok(())
}

async fn report_prepare_each_start(
    name: &str,
    iterations: usize,
    graph: &UntypedGraph,
) -> Result<(), GraphError> {
    let mut samples = Vec::with_capacity(VM_SAMPLES);
    for _ in 0..VM_SAMPLES {
        let start = Instant::now();
        for _ in 0..iterations {
            let prepared = PreparedGraph::new(graph.clone(), HandlerRegistry::new())?;
            let mut runtime = prepared.start(Value::from(7_i64))?;
            while matches!(runtime.next(context()).await?, Step::Continue) {}
            black_box(runtime);
        }
        samples.push(ns_per_iteration(start.elapsed(), iterations));
    }
    print_median(name, iterations, VM_SAMPLES, &mut samples);
    Ok(())
}

fn nested_value() -> Result<Value, GraphError> {
    let items = (0_i64..32)
        .map(|index| {
            Value::object([
                ("id", Value::from(index)),
                ("name", Value::from("shared payload")),
            ])
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(value_error("nested benchmark value"))?;
    Value::object([("items", Value::array(items))]).map_err(value_error("nested benchmark object"))
}

fn fanout_tuple_graph() -> Result<UntypedGraph, GraphError> {
    let mut builder = UntypedGraphBuilder::new("benchmark_fanout_tuple");
    let input = builder.edge("input", number_type());
    let left = builder.edge("left", number_type());
    let right = builder.edge("right", number_type());
    let output = builder.edge("output", array_type());
    builder.set_entry(input).set_exit(output);
    builder.node(
        "fanout",
        NodeKind::Builtin {
            op: BuiltinNode::FanOut,
        },
        vec![input],
        vec![left, right],
    );
    builder.node(
        "tuple",
        NodeKind::Builtin {
            op: BuiltinNode::PackTuple,
        },
        vec![left, right],
        vec![output],
    );
    builder.build()
}

fn prepared_mixed_graph() -> Result<PreparedGraph, GraphError> {
    PreparedGraph::new(mixed_graph()?, mixed_registry()?)
}

fn mixed_graph() -> Result<UntypedGraph, GraphError> {
    let child = child_graph()?;
    let mut builder = UntypedGraphBuilder::new("benchmark_mixed");
    let input = builder.edge("input", number_type());
    let loaded = builder.edge("loaded", number_type());
    let child_output = builder.edge("child_output", number_type());
    let continued = builder.edge("continued", number_type());
    let output = builder.edge("output", number_type());
    let variable = builder.variable_with_value(
        VarKey::new("benchmark", "Counter"),
        number_type(),
        VarScope::Local,
        Value::from(5_i64),
    );
    builder.set_entry(input).set_exit(output);
    add_mixed_prefix(&mut builder, input, loaded, child_output, variable, child);
    add_mixed_suffix(&mut builder, child_output, continued, output, variable)?;
    builder.build()
}

fn add_mixed_prefix(
    builder: &mut UntypedGraphBuilder,
    input: EdgeId,
    loaded: EdgeId,
    child_output: EdgeId,
    variable: VarId,
    child: UntypedGraph,
) {
    builder.node(
        "load",
        NodeKind::Load {
            var: variable,
            key: HandlerKey::new("add_state"),
        },
        vec![input],
        vec![loaded],
    );
    builder.node(
        "subflow",
        NodeKind::Subflow {
            graph: Box::new(child),
        },
        vec![loaded],
        vec![child_output],
    );
}

fn add_mixed_suffix(
    builder: &mut UntypedGraphBuilder,
    child_output: EdgeId,
    continued: EdgeId,
    output: EdgeId,
    variable: VarId,
) -> Result<(), GraphError> {
    builder.node(
        "continuation",
        NodeKind::Continuation {
            key: HandlerKey::new("add_continuation"),
            payload: Value::object([("add", Value::from(3_i64))])
                .map_err(value_error("continuation payload"))?,
            children: Vec::new(),
        },
        vec![child_output],
        vec![continued],
    );
    builder.node(
        "store",
        NodeKind::Store {
            var: variable,
            key: HandlerKey::new("store"),
        },
        vec![continued],
        vec![output],
    );
    Ok(())
}

fn mixed_registry() -> Result<HandlerRegistry, GraphError> {
    let mut registry = HandlerRegistry::new();
    registry.insert_value("add_state", |inputs: Vec<Value>| {
        let input = required_i64(inputs.first(), "load input")?;
        let state = required_i64(inputs.get(1), "load state")?;
        Ok(vec![Value::from(input + state)])
    })?;
    registry.insert_value("double", |inputs: Vec<Value>| {
        Ok(vec![Value::from(
            required_i64(inputs.first(), "child input")? * 2,
        )])
    })?;
    registry.insert_value("store", |inputs: Vec<Value>| {
        Ok(vec![inputs.first().cloned().unwrap_or(Value::NULL)])
    })?;
    registry.insert_continuation("add_continuation", AddContinuation)?;
    Ok(registry)
}

fn child_graph() -> Result<UntypedGraph, GraphError> {
    let mut builder = UntypedGraphBuilder::new("benchmark_child");
    let input = builder.edge("input", number_type());
    let output = builder.edge("output", number_type());
    builder.set_entry(input).set_exit(output);
    builder.node(
        "double",
        NodeKind::PureHandler {
            key: HandlerKey::new("double"),
        },
        vec![input],
        vec![output],
    );
    builder.build()
}

async fn report_vm(
    name: &str,
    iterations: usize,
    prepared: &PreparedGraph,
) -> Result<(), GraphError> {
    let mut samples = Vec::with_capacity(VM_SAMPLES);
    for _ in 0..VM_SAMPLES {
        let start = Instant::now();
        for _ in 0..iterations {
            let mut runtime = prepared.start(Value::from(7_i64))?;
            loop {
                match runtime.next(context()).await? {
                    Step::Continue => {}
                    Step::Done(output) => {
                        black_box(output);
                        break;
                    }
                    Step::Suspend(_) => {
                        return Err(GraphError::Invalid(
                            "benchmark unexpectedly suspended".into(),
                        ));
                    }
                }
            }
        }
        samples.push(ns_per_iteration(start.elapsed(), iterations));
    }
    print_median(name, iterations, VM_SAMPLES, &mut samples);
    Ok(())
}

async fn report_snapshot_restore(
    name: &str,
    iterations: usize,
    prepared: &PreparedGraph,
) -> Result<(), GraphError> {
    let mut runtime = prepared.start(Value::from(7_i64))?;
    let _ = runtime.next(context()).await?;
    let snapshot = runtime.snapshot()?;
    report_allocations("vm/snapshot_clone_restore", || {
        prepared.restore(snapshot.clone())
    });
    let encoded = serde_json::to_vec(&snapshot).map_err(|error| GraphError::JsonEncode {
        target: "benchmark snapshot".into(),
        reason: error.to_string(),
    })?;
    report_allocations("vm/snapshot_decode_restore", || {
        let decoded: Snapshot = serde_json::from_slice(&encoded).expect("snapshot should decode");
        prepared.restore(decoded)
    });
    let mut samples = Vec::with_capacity(VM_SAMPLES);
    for _ in 0..VM_SAMPLES {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(prepared.restore(snapshot.clone())?);
        }
        samples.push(ns_per_iteration(start.elapsed(), iterations));
    }
    print_median(name, iterations, VM_SAMPLES, &mut samples);
    Ok(())
}

async fn report_json(name: &str, iterations: usize) -> Result<(), GraphError> {
    let invoker = JsonInvoker::new(identity_graph()?, HandlerRegistry::new())?;
    let mut samples = Vec::with_capacity(VM_SAMPLES);
    for _ in 0..VM_SAMPLES {
        let start = Instant::now();
        for _ in 0..iterations {
            let response = invoker
                .invoke(
                    JsonRequest::Start {
                        version: JSON_WIRE_VERSION,
                        input: json!(7),
                    },
                    context(),
                )
                .await?;
            black_box(response);
        }
        samples.push(ns_per_iteration(start.elapsed(), iterations));
    }
    print_median(name, iterations, VM_SAMPLES, &mut samples);
    Ok(())
}

async fn report_json_next(name: &str, iterations: usize) -> Result<(), GraphError> {
    let invoker = JsonInvoker::new(two_step_json_graph()?, HandlerRegistry::new())?;
    let response = invoker
        .invoke(
            JsonRequest::Start {
                version: JSON_WIRE_VERSION,
                input: json!(7),
            },
            context(),
        )
        .await?;
    let JsonResponse::Continue { snapshot, .. } = response else {
        return Err(GraphError::Invalid(
            "two-step start did not continue".into(),
        ));
    };
    let request = encode_request(&JsonRequest::Next {
        version: JSON_WIRE_VERSION,
        snapshot,
    })?;
    report_json_string(name, iterations, &invoker, &request).await
}

async fn report_json_resume(name: &str, iterations: usize) -> Result<(), GraphError> {
    let invoker = JsonInvoker::new(suspend_json_graph()?, HandlerRegistry::new())?;
    let response = invoker
        .invoke(
            JsonRequest::Start {
                version: JSON_WIRE_VERSION,
                input: json!(7),
            },
            context(),
        )
        .await?;
    let JsonResponse::Suspend { snapshot, .. } = response else {
        return Err(GraphError::Invalid("suspend start did not suspend".into()));
    };
    let request = encode_request(&JsonRequest::Resume {
        version: JSON_WIRE_VERSION,
        snapshot,
        input: json!(9),
    })?;
    report_json_string(name, iterations, &invoker, &request).await
}

async fn report_json_string(
    name: &str,
    iterations: usize,
    invoker: &JsonInvoker,
    request: &str,
) -> Result<(), GraphError> {
    let mut samples = Vec::with_capacity(VM_SAMPLES);
    for _ in 0..VM_SAMPLES {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(invoker.invoke_str(request, context()).await?);
        }
        samples.push(ns_per_iteration(start.elapsed(), iterations));
    }
    print_median(name, iterations, VM_SAMPLES, &mut samples);
    Ok(())
}

fn encode_request(request: &JsonRequest) -> Result<String, GraphError> {
    serde_json::to_string(request).map_err(|error| GraphError::JsonEncode {
        target: "benchmark request".into(),
        reason: error.to_string(),
    })
}

fn identity_graph() -> Result<UntypedGraph, GraphError> {
    let mut builder = UntypedGraphBuilder::new("benchmark_json");
    let input = builder.edge("input", number_type());
    let output = builder.edge("output", number_type());
    builder.set_entry(input).set_exit(output);
    builder.node(
        "identity",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![input],
        vec![output],
    );
    builder.build()
}

fn two_step_json_graph() -> Result<UntypedGraph, GraphError> {
    let mut builder = UntypedGraphBuilder::new("benchmark_json_next");
    let input = builder.edge("input", number_type());
    let middle = builder.edge("middle", number_type());
    let output = builder.edge("output", number_type());
    builder.set_entry(input).set_exit(output);
    for (name, source, target) in [("first", input, middle), ("second", middle, output)] {
        builder.node(
            name,
            NodeKind::Builtin {
                op: BuiltinNode::Identity,
            },
            vec![source],
            vec![target],
        );
    }
    builder.build()
}

fn suspend_json_graph() -> Result<UntypedGraph, GraphError> {
    let mut builder = UntypedGraphBuilder::new("benchmark_json_resume");
    let input = builder.edge("input", number_type());
    let output = builder.edge("output", number_type());
    builder.set_entry(input).set_exit(output);
    builder.node(
        "wait",
        NodeKind::Suspend {
            resume_type: "Number".into(),
            payload: Value::NULL,
        },
        vec![input],
        vec![output],
    );
    builder.build()
}

fn dead_shaping_graph() -> Result<UntypedGraph, GraphError> {
    let mut builder = UntypedGraphBuilder::new("benchmark_dead_shaping");
    let input = builder.edge("input", number_type());
    let left = builder.edge("left", number_type());
    let right = builder.edge("right", number_type());
    let packed = builder.edge("packed", array_type());
    let output = builder.edge("output", number_type());
    builder.set_entry(input).set_exit(output);
    builder.node(
        "dead_fanout",
        NodeKind::Builtin {
            op: BuiltinNode::FanOut,
        },
        vec![input],
        vec![left, right],
    );
    builder.node(
        "dead_pack",
        NodeKind::Builtin {
            op: BuiltinNode::PackTuple,
        },
        vec![left, right],
        vec![packed],
    );
    builder.node(
        "live_identity",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![input],
        vec![output],
    );
    builder.build()
}

fn report_sync<T>(name: &str, iterations: usize, mut operation: impl FnMut() -> T) {
    let mut samples = Vec::with_capacity(SYNC_SAMPLES);
    for _ in 0..SYNC_SAMPLES {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(operation());
        }
        samples.push(ns_per_iteration(start.elapsed(), iterations));
    }
    print_median(name, iterations, SYNC_SAMPLES, &mut samples);
}

fn report_allocations<T>(name: &str, operation: impl FnOnce() -> T) {
    let measured = allocation_counter::measure(|| {
        black_box(operation());
    });
    println!(
        "{name}: {} allocation(s), {} byte(s)",
        measured.count_total, measured.bytes_total
    );
}

fn print_median(name: &str, iterations: usize, sample_count: usize, samples: &mut [f64]) {
    samples.sort_by(f64::total_cmp);
    let median = samples.get(sample_count / 2).copied().unwrap_or_default();
    println!(
        "{name}: {:.2} ns/op median ({sample_count} x {iterations} iterations)",
        median
    );
}

fn ns_per_iteration(elapsed: Duration, iterations: usize) -> f64 {
    if iterations == 0 {
        return 0.0;
    }
    elapsed.as_secs_f64() * 1_000_000_000.0 / iterations as f64
}

fn number_type() -> TypeSpec {
    TypeSpec::new("Number", json!({"type": "number"}))
}

fn array_type() -> TypeSpec {
    TypeSpec::new("Array", json!({"type": "array"}))
}

fn required_i64(value: Option<&Value>, label: &str) -> Result<i64, GraphError> {
    value
        .and_then(Value::as_i64)
        .ok_or_else(|| GraphError::Invalid(format!("{label} must be an integer")))
}

fn value_error(target: &'static str) -> impl FnOnce(pravah::graph::ValueError) -> GraphError {
    move |error| GraphError::ValueConversion {
        target: target.into(),
        reason: error.to_string(),
    }
}

fn context() -> Context {
    Context::new(FlowConf::default())
}
