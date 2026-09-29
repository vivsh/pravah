use crate::clients::ErrorKind;
use ::serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use either::Either;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde_json::Value as JsonValue;

use crate::clients::{
    Client, ClientError, ClientOptions, ClientOutput, ClientResponse, LlmBackend, Message,
    ModelUrl, Provider, ProviderFactory, Role, ToolCall,
};
use crate::tools::ToolError;
use crate::{Context, FlowConf, deps::Deps};

use super::state::ReturnTarget;
use super::*;

#[path = "tests/host.rs"]
pub(crate) mod host;

macro_rules! rv {
    ($($tokens:tt)*) => {{
        to_value(serde_json::json!($($tokens)*)).expect("test value should enter runtime domain")
    }};
}

fn test_runtime<T: Serialize>(
    graph: UntypedGraph,
    input: T,
    registry: HandlerRegistry,
) -> Result<Runtime, GraphError> {
    let input = to_value(input).map_err(|err| GraphError::ValueConversion {
        target: "test input".into(),
        reason: err.to_string(),
    })?;
    PreparedGraph::new(graph, registry)?.start(input, uuid::Uuid::nil())
}

fn any_type(name: &str) -> TypeSpec {
    TypeSpec::new(name, serde_json::json!({ "type": "object" }))
}

fn number_type(name: &str) -> TypeSpec {
    TypeSpec::new(name, serde_json::json!({ "type": "number" }))
}

fn bool_type(name: &str) -> TypeSpec {
    TypeSpec::new(name, serde_json::json!({ "type": "boolean" }))
}

fn array_type(name: &str) -> TypeSpec {
    TypeSpec::new(name, serde_json::json!({ "type": "array" }))
}

fn ctx() -> Context {
    Context::new(FlowConf::default())
}

fn run_value_handler(f: fn(Vec<Value>) -> Result<Vec<Value>, GraphError>) -> impl ValueHandler {
    move |inputs| f(inputs)
}

/// Verifies graph round trips exact json.
#[test]
fn graph_round_trips_exact_json() {
    let mut builder = UntypedGraphBuilder::new("roundtrip");
    let input = builder.edge("input", any_type("Payload"));
    let output = builder.edge("output", any_type("Payload"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "copy",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");

    let json = serde::to_json_pretty(&graph).expect("graph should serialize");
    let restored = serde::from_json(&json).expect("graph should deserialize");

    assert_eq!(graph, restored);
}

/// Verifies runtime executes one node per next.
#[tokio::test]
async fn runtime_executes_one_node_per_next() {
    let mut builder = UntypedGraphBuilder::new("chain");
    let input = builder.edge("input", number_type("Number"));
    let middle = builder.edge("middle", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "plus_one",
        NodeKind::PureHandler {
            key: HandlerKey::new("plus_one"),
        },
        vec![input],
        vec![middle],
    );
    builder.node(
        "double",
        NodeKind::PureHandler {
            key: HandlerKey::new("double"),
        },
        vec![middle],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value(
            "plus_one",
            run_value_handler(|inputs| Ok(vec![rv!(inputs[0].as_i64().unwrap_or_default() + 1)])),
        )
        .expect("handler should insert");
    registry
        .insert_value(
            "double",
            run_value_handler(|inputs| Ok(vec![rv!(inputs[0].as_i64().unwrap_or_default() * 2)])),
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(2), registry).expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!(6)));
}

/// Verifies mark goto reenters edge with new generation.
#[tokio::test]
async fn mark_goto_reenters_edge_with_new_generation() {
    let mut builder = UntypedGraphBuilder::new("mark_goto_suspend_loop");
    let input = builder.edge("input", number_type("Number"));
    let resumed = builder.edge("resumed", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    let mark = builder.mark(input);
    builder.set_entry(input).set_exit(output);
    builder.node(
        "wait",
        NodeKind::Suspend {
            resume_type: "Number".into(),
            payload: rv!({"need": "number"}),
        },
        vec![input],
        vec![resumed],
    );
    builder.goto("repeat", resumed, mark);
    builder.node(
        "exit_copy",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut runtime =
        test_runtime(graph, rv!(1), HandlerRegistry::new()).expect("runtime should build");

    assert_eq!(
        runtime.next().unwrap(),
        Step::Suspend(rv!({"need": "number"}))
    );
    runtime.resume_value(rv!(2)).unwrap();
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.state().frames[0].values[input.0], Some(rv!(2)));
    assert_eq!(
        runtime.next().unwrap(),
        Step::Suspend(rv!({"need": "number"}))
    );
}

/// Verifies typed resume validation leaves a suspension unchanged and retryable.
#[tokio::test]
async fn typed_mark_goto_is_string_free_and_builder_checked() {
    let root = Flow::<i64>::root();
    let start = root.mark();
    let _loop_edge = root.clone().suspend::<i64>().goto(start);
    let exit = root.map(|value| value);
    let flow = exit.finish::<i64>().expect("typed mark/goto graph builds");
    let mut runtime = flow
        .start(1, uuid::Uuid::nil())
        .expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Suspend(rv!(1)));
    let before = serde_json::to_vec(&runtime.snapshot().unwrap()).unwrap();
    assert!(runtime.resume("not a number").is_err());
    let after = serde_json::to_vec(&runtime.snapshot().unwrap()).unwrap();
    assert_eq!(before, after);
    runtime.resume(3_i64).unwrap();
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Suspend(rv!(3)));
}

/// Verifies typed goto rejects cross builder mark at finish.
#[test]
fn typed_goto_rejects_cross_builder_mark_at_finish() {
    let first = Flow::<i64>::root();
    let mark = first.mark();
    let second = Flow::<i64>::root();
    let result = second.goto(mark).finish::<i64>();

    assert!(
        result.is_err(),
        "cross-builder mark usage should be surfaced at finish"
    );
}

/// Verifies untyped goto rejects type mismatch.
#[test]
fn untyped_goto_rejects_type_mismatch() {
    let mut builder = UntypedGraphBuilder::new("bad_goto");
    let input = builder.edge("input", number_type("Number"));
    let text = builder.edge(
        "text",
        TypeSpec::new("Text", serde_json::json!({"type": "string"})),
    );
    let mark = builder.mark(text);
    builder.set_entry(input).set_exit(text);
    builder.goto("bad", input, mark);

    let err = builder
        .build()
        .expect_err("goto type mismatch should fail build");
    assert!(
        err.to_string().contains("does not match mark target type"),
        "unexpected error: {err}"
    );
}

/// Verifies graph diagram renders mark and goto.
#[test]
fn graph_diagram_renders_mark_and_goto() {
    let mut builder = UntypedGraphBuilder::new("diagram_loop");
    let input = builder.edge("input", number_type("Number"));
    let resumed = builder.edge("resumed", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    let mark = builder.mark(input);
    builder.set_entry(input).set_exit(output);
    builder.node(
        "wait",
        NodeKind::Suspend {
            resume_type: "Number".into(),
            payload: rv!({"need": "number"}),
        },
        vec![input],
        vec![resumed],
    );
    builder.goto("repeat", resumed, mark);
    builder.node(
        "exit_copy",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let diagram = GraphDiagram::from_graph(&graph);
    let mermaid = diagram.mermaid();
    let dot = diagram.dot();
    let tree = diagram.render_tree();

    assert!(mermaid.contains("mark_0"));
    assert!(mermaid.contains("|goto|"));
    assert!(mermaid.contains("-.->|goto|"));
    assert!(mermaid.contains("|reenter|"));
    assert!(mermaid.contains("Number"));
    assert!(dot.contains("constraint=false"));
    assert!(tree.contains("(goto)"));
    assert!(
        diagram
            .nodes()
            .iter()
            .any(|node| node.kind == DiagramNodeKind::Mark)
    );
}

/// Verifies subflow pushes frame and returns to parent edge.
#[tokio::test]
async fn subflow_pushes_frame_and_returns_to_parent_edge() {
    let mut child_builder = UntypedGraphBuilder::new("child");
    let child_in = child_builder.edge("child_in", number_type("Number"));
    let child_out = child_builder.edge("child_out", number_type("Number"));
    child_builder.set_entry(child_in).set_exit(child_out);
    child_builder.node(
        "child_plus_one",
        NodeKind::PureHandler {
            key: HandlerKey::new("plus_one"),
        },
        vec![child_in],
        vec![child_out],
    );
    let child = child_builder.build().expect("child graph should build");

    let mut parent_builder = UntypedGraphBuilder::new("parent");
    let parent_in = parent_builder.edge("parent_in", number_type("Number"));
    let after_child = parent_builder.edge("after_child", number_type("Number"));
    let parent_out = parent_builder.edge("parent_out", number_type("Number"));
    parent_builder.set_entry(parent_in).set_exit(parent_out);
    parent_builder.node(
        "call_child",
        NodeKind::Subflow {
            graph: Box::new(child),
        },
        vec![parent_in],
        vec![after_child],
    );
    parent_builder.node(
        "double",
        NodeKind::PureHandler {
            key: HandlerKey::new("double"),
        },
        vec![after_child],
        vec![parent_out],
    );
    let parent = parent_builder.build().expect("parent graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value(
            "plus_one",
            run_value_handler(|inputs| Ok(vec![rv!(inputs[0].as_i64().unwrap_or_default() + 1)])),
        )
        .expect("handler should insert");
    registry
        .insert_value(
            "double",
            run_value_handler(|inputs| Ok(vec![rv!(inputs[0].as_i64().unwrap_or_default() * 2)])),
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(parent, rv!(4), registry).expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.state().frames.len(), 2);
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(
        runtime.state().frames.len(),
        1,
        "child exit should cascade to parent"
    );
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!(10)));
}

/// Verifies multi consumer subflow input is not moved from parent edge.
#[tokio::test]
async fn multi_consumer_subflow_input_is_not_moved_from_parent_edge() {
    let mut child_builder = UntypedGraphBuilder::new("child_identity");
    let child_in = child_builder.edge("child_in", number_type("Number"));
    let child_out = child_builder.edge("child_out", number_type("Number"));
    child_builder.set_entry(child_in).set_exit(child_out);
    child_builder.node(
        "child_identity",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![child_in],
        vec![child_out],
    );
    let child = child_builder.build().expect("child graph should build");

    let mut parent_builder = UntypedGraphBuilder::new("shared_parent_input");
    let parent_in = parent_builder.edge("parent_in", number_type("Number"));
    let subflow_out = parent_builder.edge("subflow_out", number_type("Number"));
    let doubled = parent_builder.edge("doubled", number_type("Number"));
    let parent_out = parent_builder.edge("parent_out", array_type("Tuple"));
    parent_builder.set_entry(parent_in).set_exit(parent_out);
    parent_builder.node(
        "call_child",
        NodeKind::Subflow {
            graph: Box::new(child),
        },
        vec![parent_in],
        vec![subflow_out],
    );
    parent_builder.node(
        "double_original",
        NodeKind::PureHandler {
            key: HandlerKey::new("double"),
        },
        vec![parent_in],
        vec![doubled],
    );
    parent_builder.node(
        "pack_outputs",
        NodeKind::Builtin {
            op: BuiltinNode::PackTuple,
        },
        vec![subflow_out, doubled],
        vec![parent_out],
    );
    let graph = parent_builder.build().expect("parent graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value(
            "double",
            run_value_handler(|inputs| Ok(vec![rv!(inputs[0].as_i64().unwrap_or_default() * 2)])),
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(5), registry).expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!([5, 10])));
}

/// Verifies local load and store are pure vm state nodes.
#[tokio::test]
async fn local_load_and_store_are_pure_vm_state_nodes() {
    let mut builder = UntypedGraphBuilder::new("vars");
    let input = builder.edge("input", number_type("Number"));
    let loaded = builder.edge("loaded", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    let state = builder.variable_with_value(
        VarKey::new("rust", "Counter"),
        number_type("Counter"),
        VarScope::Local,
        rv!(10),
    );
    builder.set_entry(input).set_exit(output);
    builder.node(
        "load_counter",
        NodeKind::Load {
            var: state,
            key: HandlerKey::new("load_add"),
        },
        vec![input],
        vec![loaded],
    );
    builder.node(
        "store_counter",
        NodeKind::Store {
            var: state,
            key: HandlerKey::new("store_input"),
        },
        vec![loaded],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value(
            "load_add",
            run_value_handler(|inputs| {
                Ok(vec![rv!(
                    inputs[0].as_i64().unwrap_or_default() + inputs[1].as_i64().unwrap_or_default()
                )])
            }),
        )
        .expect("handler should insert");
    registry
        .insert_value(
            "store_input",
            run_value_handler(|inputs| Ok(vec![inputs[0].clone()])),
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(5), registry).expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!(15)));
    let root = runtime.state().frames.first();
    assert!(root.is_none(), "done pops the root frame");
}

/// Verifies failed store output validation does not mutate variable.
#[tokio::test]
async fn failed_store_output_validation_does_not_mutate_variable() {
    let mut builder = UntypedGraphBuilder::new("store_rollback");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge(
        "output",
        TypeSpec::new("Text", serde_json::json!({"type": "string"})),
    );
    let state = builder.variable_with_value(
        VarKey::new("rust", "Counter"),
        number_type("Counter"),
        VarScope::Local,
        rv!(10),
    );
    builder.set_entry(input).set_exit(output);
    builder.node(
        "store_counter",
        NodeKind::Store {
            var: state,
            key: HandlerKey::new("store_new_value"),
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value("store_new_value", run_value_handler(|_| Ok(vec![rv!(99)])))
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(5), registry).expect("runtime should build");

    let err = runtime
        .next()
        .expect_err("passthrough output schema should fail");
    match err {
        GraphError::Schema { expected, .. } => assert_eq!(expected, "Text"),
        other => panic!("expected schema error, got {other:?}"),
    }
    let frame = runtime.state().frames.first().expect("frame should remain");
    assert_eq!(frame.variables[state.0], Some(rv!(10)));
    assert!(frame.values[output.0].is_none());
}

/// Verifies suspend node resume preserves frame stack.
#[tokio::test]
async fn suspend_node_resume_preserves_frame_stack() {
    let mut builder = UntypedGraphBuilder::new("suspend");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "suspend",
        NodeKind::Suspend {
            resume_type: "Number".into(),
            payload: rv!({"need": "resume"}),
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let registry = HandlerRegistry::new();
    let mut runtime = test_runtime(graph, rv!(7), registry).expect("runtime should build");

    assert_eq!(
        runtime.next().unwrap(),
        Step::Suspend(rv!({"need": "resume"}))
    );
    assert_eq!(runtime.state().frames.len(), 1);
    assert!(runtime.next().is_err(), "next while suspended must fail");
    runtime.resume_value(rv!(5)).unwrap();
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!(5)));
}

/// Verifies snapshot rejects suspension graph frame mismatch.
#[tokio::test]
async fn snapshot_rejects_suspension_graph_frame_mismatch() {
    let mut builder = UntypedGraphBuilder::new("bad_suspension");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "suspend",
        NodeKind::Suspend {
            resume_type: "Number".into(),
            payload: rv!({"need": "resume"}),
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let registry = HandlerRegistry::new();
    let prepared = PreparedGraph::new(graph, registry).expect("graph should prepare");
    let mut runtime = prepared
        .start(rv!(7), uuid::Uuid::nil())
        .expect("runtime should build");
    assert!(matches!(runtime.next().unwrap(), Step::Suspend(_)));

    let mut snapshot = runtime.snapshot().expect("snapshot should build");
    let mut encoded = serde_json::to_value(&snapshot).unwrap();
    encoded["state"]["waiting"]["suspension"]["frame_depth"] = serde_json::json!(999);
    snapshot = serde_json::from_value(encoded).unwrap();
    let err = match prepared.restore(snapshot) {
        Ok(_) => panic!("bad suspension graph should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("frame depth"));
}

/// Verifies snapshot rejects continuation inbox without checkpoint.
#[tokio::test]
async fn snapshot_rejects_continuation_inbox_without_checkpoint() {
    let mut child_builder = UntypedGraphBuilder::new("continuation_child");
    let child_in = child_builder.edge("child_in", number_type("Number"));
    let child_out = child_builder.edge("child_out", number_type("Number"));
    child_builder.set_entry(child_in).set_exit(child_out);
    child_builder.node(
        "copy",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![child_in],
        vec![child_out],
    );
    let child = child_builder.build().expect("child should build");

    let mut builder = UntypedGraphBuilder::new("bad_continuation_inbox");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "continuation",
        NodeKind::Continuation {
            key: HandlerKey::new("continuation"),
            payload: Value::NULL,
            children: vec![child],
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_continuation("continuation", StartChildThenError)
        .expect("handler should insert");
    let prepared = PreparedGraph::new(graph, registry).expect("graph should prepare");
    let mut runtime = prepared
        .start(rv!(5), uuid::Uuid::nil())
        .expect("runtime should build");
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Continue);

    let mut snapshot = runtime.snapshot().expect("snapshot should build");
    let frame = snapshot
        .state
        .frame_mut(0)
        .expect("parent frame should remain");
    assert_eq!(frame.continuation_inboxes[0].values.len(), 1);
    frame.checkpoints = Arc::default();
    let err = match prepared.restore(snapshot) {
        Ok(_) => panic!("inbox without checkpoint should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("without checkpoint"));
}

/// Verifies snapshot restore round trips edge vm state.
#[tokio::test]
async fn snapshot_restore_round_trips_edge_vm_state() {
    let mut builder = UntypedGraphBuilder::new("snapshot");
    let input = builder.edge("input", number_type("Number"));
    let middle = builder.edge("middle", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "plus_one",
        NodeKind::PureHandler {
            key: HandlerKey::new("plus_one"),
        },
        vec![input],
        vec![middle],
    );
    builder.node(
        "double",
        NodeKind::PureHandler {
            key: HandlerKey::new("double"),
        },
        vec![middle],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value(
            "plus_one",
            run_value_handler(|inputs| Ok(vec![rv!(inputs[0].as_i64().unwrap_or_default() + 1)])),
        )
        .expect("handler should insert");
    registry
        .insert_value(
            "double",
            run_value_handler(|inputs| Ok(vec![rv!(inputs[0].as_i64().unwrap_or_default() * 2)])),
        )
        .expect("handler should insert");
    let prepared = PreparedGraph::new(graph, registry).expect("graph should prepare");
    let mut runtime = prepared
        .start(rv!(3), uuid::Uuid::nil())
        .expect("runtime should build");
    assert_eq!(runtime.next().unwrap(), Step::Continue);

    let snapshot = runtime.snapshot().expect("snapshot should build");
    let mut restored = prepared.restore(snapshot).expect("snapshot should restore");

    assert_eq!(restored.next().unwrap(), Step::Done(rv!(8)));
}

/// Verifies handler output schema mismatch is fatal.
#[tokio::test]
async fn handler_output_schema_mismatch_is_fatal() {
    let mut builder = UntypedGraphBuilder::new("schema_mismatch");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "bad",
        NodeKind::PureHandler {
            key: HandlerKey::new("bad"),
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value("bad", run_value_handler(|_| Ok(vec![rv!("not a number")])))
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(1), registry).expect("runtime should build");

    let err = runtime.next().expect_err("schema mismatch should fail");
    assert!(matches!(err, GraphError::Schema { .. }));
}

/// Verifies runtime shape checks are minimal schema hints.
#[test]
fn runtime_shape_checks_are_minimal_schema_hints() {
    let direct_number = TypeSpec::new("Number", serde_json::json!({ "type": "number" }));
    let err = schema::validate_value(&direct_number, &rv!("not a number"), "direct")
        .expect_err("direct primitive mismatch should fail");
    assert!(matches!(err, GraphError::Schema { .. }));

    let direct_object = TypeSpec::new(
        "Payload",
        serde_json::json!({
            "type": "object",
            "required": ["count"],
            "properties": {
                "count": { "type": "integer" }
            }
        }),
    );
    schema::validate_value(&direct_object, &rv!({"count": 3}), "direct object")
        .expect("direct object shape should pass");
    assert!(
        schema::validate_value(&direct_object, &rv!({"count": "three"}), "direct object").is_err(),
        "direct object property mismatch should fail"
    );

    let referenced = TypeSpec::new(
        "Referenced",
        serde_json::json!({
            "$ref": "#/$defs/Payload",
            "$defs": {
                "Payload": {
                    "type": "object",
                    "required": ["count"]
                }
            }
        }),
    );
    schema::validate_value(&referenced, &rv!({"not_count": "metadata only"}), "ref")
        .expect("runtime does not pretend to resolve complex JSON Schema references");
}

#[derive(Clone)]
struct StaticStartContinuation {
    transition: ContinuationTransition,
}

impl ContinuationHandler for StaticStartContinuation {
    fn start<'a>(
        &'a self,
        _payload: &'a Value,
        _state: Option<Value>,
        _inputs: Vec<Value>,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let transition = self.transition.clone();
        Ok(transition)
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        checkpoint: Value,
        _event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        Ok(ContinuationTransition {
            fetch: None,
            history: Vec::new(),
            checkpoint: None,
            state: None,
            outputs: vec![checkpoint],
            writes: Vec::new(),
            child_calls: Vec::new(),
            suspension: None,
        })
    }
}

struct SuspendOnceContinuation;

impl ContinuationHandler for SuspendOnceContinuation {
    fn start<'a>(
        &'a self,
        _payload: &'a Value,
        _state: Option<Value>,
        inputs: Vec<Value>,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        Ok(ContinuationTransition {
            fetch: None,
            history: Vec::new(),
            checkpoint: inputs.into_iter().next(),
            state: None,
            outputs: Vec::new(),
            writes: Vec::new(),
            child_calls: Vec::new(),
            suspension: Some(ContinuationSuspension {
                resume_type: number_type("Number"),
                payload: rv!({"prompt": "replacement number"}),
            }),
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _checkpoint: Value,
        event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let ContinuationEvent::Resume { input } = event else {
            return Err(GraphError::Invalid("expected continuation resume".into()));
        };
        Ok(ContinuationTransition {
            fetch: None,
            history: Vec::new(),
            checkpoint: None,
            state: None,
            outputs: vec![input],
            writes: Vec::new(),
            child_calls: Vec::new(),
            suspension: None,
        })
    }
}

fn suspension_continuation_graph() -> UntypedGraph {
    let mut builder = UntypedGraphBuilder::new("continuation_suspension");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "continuation",
        NodeKind::Continuation {
            key: HandlerKey::new("continuation"),
            payload: Value::NULL,
            children: Vec::new(),
        },
        vec![input],
        vec![output],
    );
    builder.build().expect("graph should build")
}

/// Verifies a non-agent continuation can suspend, restore, and receive resume input.
#[tokio::test]
async fn continuation_owned_suspension_round_trips_through_snapshot() {
    let graph = suspension_continuation_graph();
    let mut registry = HandlerRegistry::new();
    registry
        .insert_continuation("continuation", SuspendOnceContinuation)
        .expect("handler should insert");
    let prepared = PreparedGraph::new(graph, registry).expect("graph should prepare");
    let mut runtime = prepared
        .start(rv!(1), uuid::Uuid::nil())
        .expect("runtime should build");

    assert_eq!(
        runtime.next().expect("continuation should run"),
        Step::Suspend(rv!({"prompt": "replacement number"}))
    );
    let snapshot = runtime.snapshot().expect("snapshot should encode");
    let mut runtime = prepared.restore(snapshot).expect("snapshot should restore");
    runtime
        .resume_value(rv!(9))
        .expect("continuation should resume");
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!(9)));
}

async fn run_static_continuation_transition(
    transition: ContinuationTransition,
) -> Result<Step, GraphError> {
    let mut builder = UntypedGraphBuilder::new("continuation_transition");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "continuation",
        NodeKind::Continuation {
            key: HandlerKey::new("continuation"),
            payload: Value::NULL,
            children: Vec::new(),
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_continuation("continuation", StaticStartContinuation { transition })
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(1), registry).expect("runtime should build");
    runtime.next()
}

struct AssertNoServiceSmuggling;

impl ContinuationHandler for AssertNoServiceSmuggling {
    fn start<'a>(
        &'a self,
        _payload: &'a Value,
        _state: Option<Value>,
        _inputs: Vec<Value>,
        ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        assert_eq!(ctx.execution_id(), uuid::Uuid::nil());
        Ok(ContinuationTransition {
            fetch: None,
            history: Vec::new(),
            checkpoint: None,
            state: None,
            outputs: vec![rv!(ctx.history().entries().is_empty())],
            writes: Vec::new(),
            child_calls: Vec::new(),
            suspension: None,
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _continuation: Value,
        _event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        Err(GraphError::Invalid("unexpected resume".into()))
    }
}

/// Verifies continuation context does not smuggle runtime services into context.
#[tokio::test]
async fn continuation_context_does_not_smuggle_runtime_services_into_context() {
    let mut builder = UntypedGraphBuilder::new("continuation_context_services");
    let input = builder.edge("input", any_type("Input"));
    let output = builder.edge("output", bool_type("Bool"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "continuation",
        NodeKind::Continuation {
            key: HandlerKey::new("continuation"),
            payload: Value::NULL,
            children: Vec::new(),
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_continuation("continuation", AssertNoServiceSmuggling)
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!({}), registry).expect("runtime should build");

    let step = runtime.next().expect("continuation should run");
    assert_eq!(step, Step::Done(rv!(true)));
}

/// Verifies continuation rejects outputs with checkpoint.
#[tokio::test]
async fn continuation_rejects_outputs_with_checkpoint() {
    let err = run_static_continuation_transition(ContinuationTransition {
        fetch: None,
        history: Vec::new(),
        checkpoint: Some(rv!({"state": true})),
        state: None,
        outputs: vec![rv!(1)],
        writes: Vec::new(),
        child_calls: Vec::new(),
        suspension: None,
    })
    .await
    .expect_err("outputs plus checkpoint should fail");
    assert!(matches!(
        err,
        GraphError::InvalidContinuationTransition { .. }
    ));
}

/// Verifies a continuation suspension cannot combine external pause with output mutation.
#[tokio::test]
async fn continuation_rejects_suspension_with_outputs() {
    let err = run_static_continuation_transition(ContinuationTransition {
        fetch: None,
        history: Vec::new(),
        checkpoint: Some(rv!({"state": true})),
        state: None,
        outputs: vec![rv!(1)],
        writes: Vec::new(),
        child_calls: Vec::new(),
        suspension: Some(ContinuationSuspension {
            resume_type: number_type("Number"),
            payload: rv!({"prompt": "number"}),
        }),
    })
    .await
    .expect_err("suspension plus outputs should fail");
    assert!(matches!(
        err,
        GraphError::InvalidContinuationTransition { .. }
    ));
}

/// Verifies failed continuation write plan does not partially write edges.
#[tokio::test]
async fn failed_continuation_write_plan_does_not_partially_write_edges() {
    let mut builder = UntypedGraphBuilder::new("continuation_write_rollback");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "continuation",
        NodeKind::Continuation {
            key: HandlerKey::new("continuation"),
            payload: Value::NULL,
            children: Vec::new(),
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_continuation(
            "continuation",
            StaticStartContinuation {
                transition: ContinuationTransition {
                    fetch: None,
                    history: Vec::new(),
                    checkpoint: None,
                    state: None,
                    outputs: vec![rv!(2)],
                    writes: vec![EdgeWrite {
                        edge: output,
                        value: rv!(1),
                    }],
                    child_calls: Vec::new(),
                    suspension: None,
                },
            },
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(0), registry).expect("runtime should build");

    let err = runtime
        .next()
        .expect_err("duplicate continuation write should fail");
    assert!(err.to_string().contains("written more than once"));
    let frame = runtime.state().frames.first().expect("frame should remain");
    assert!(frame.values[output.0].is_none());
    assert_eq!(frame.node_epochs[0], 0);
}

struct PollThenComplete;

impl ContinuationHandler for PollThenComplete {
    fn start<'a>(
        &'a self,
        _payload: &'a Value,
        _state: Option<Value>,
        inputs: Vec<Value>,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        Ok(ContinuationTransition {
            fetch: None,
            history: Vec::new(),
            checkpoint: inputs.into_iter().next(),
            state: None,
            outputs: Vec::new(),
            writes: Vec::new(),
            child_calls: Vec::new(),
            suspension: None,
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        checkpoint: Value,
        event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let ContinuationEvent::Poll = event else {
            return Err(GraphError::Invalid("expected poll event".into()));
        };
        Ok(ContinuationTransition {
            fetch: None,
            history: Vec::new(),
            checkpoint: None,
            state: None,
            outputs: vec![rv!(checkpoint.as_i64().unwrap_or_default() + 1)],
            writes: Vec::new(),
            child_calls: Vec::new(),
            suspension: None,
        })
    }
}

struct StartChildThenError;

impl ContinuationHandler for StartChildThenError {
    fn start<'a>(
        &'a self,
        _payload: &'a Value,
        _state: Option<Value>,
        inputs: Vec<Value>,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        Ok(ContinuationTransition {
            fetch: None,
            history: Vec::new(),
            checkpoint: Some(rv!({"started": true})),
            state: None,
            outputs: Vec::new(),
            writes: Vec::new(),
            child_calls: vec![ContinuationChildCall {
                child_index: 0,
                call_id: "child-1".into(),
                input: inputs.into_iter().next().unwrap_or(Value::NULL),
            }],
            suspension: None,
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _continuation: Value,
        event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        match event {
            ContinuationEvent::ChildResult { .. } => {
                Err(GraphError::Invalid("child result handling failed".into()))
            }
            other => Err(GraphError::Invalid(format!("unexpected event {other:?}"))),
        }
    }
}

struct StartInvalidChildInput;

impl ContinuationHandler for StartInvalidChildInput {
    fn start<'a>(
        &'a self,
        _payload: &'a Value,
        _state: Option<Value>,
        _inputs: Vec<Value>,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        Ok(ContinuationTransition {
            fetch: None,
            history: Vec::new(),
            checkpoint: Some(rv!({"started": true})),
            state: Some(rv!({"mutated": true})),
            outputs: Vec::new(),
            writes: Vec::new(),
            child_calls: vec![ContinuationChildCall {
                child_index: 0,
                call_id: "child-1".into(),
                input: rv!("not a number"),
            }],
            suspension: None,
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _continuation: Value,
        _event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        Err(GraphError::Invalid("unexpected resume".into()))
    }
}

/// Verifies continuation child result error preserves checkpoint and inbox.
#[tokio::test]
async fn continuation_child_result_error_preserves_checkpoint_and_inbox() {
    let mut child_builder = UntypedGraphBuilder::new("continuation_child");
    let child_in = child_builder.edge("child_in", number_type("Number"));
    let child_out = child_builder.edge("child_out", number_type("Number"));
    child_builder.set_entry(child_in).set_exit(child_out);
    child_builder.node(
        "copy",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![child_in],
        vec![child_out],
    );
    let child = child_builder.build().expect("child should build");

    let mut builder = UntypedGraphBuilder::new("continuation_preserve");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "continuation",
        NodeKind::Continuation {
            key: HandlerKey::new("continuation"),
            payload: Value::NULL,
            children: vec![child],
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_continuation("continuation", StartChildThenError)
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(5), registry).expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.state().frames.len(), 2);
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.state().frames.len(), 1);
    let frame = runtime
        .state()
        .frames
        .first()
        .expect("parent frame remains");
    assert!(frame.checkpoints[0].is_some());
    assert_eq!(frame.continuation_inboxes[0].len(), 1);

    let err = runtime.next().expect_err("child-result poll should fail");
    assert!(err.to_string().contains("child result handling failed"));
    let frame = runtime
        .state()
        .frames
        .first()
        .expect("parent frame remains");
    assert!(frame.checkpoints[0].is_some());
    assert_eq!(frame.continuation_inboxes[0].len(), 1);
}

/// Verifies failed continuation child preflight does not mutate parent state.
#[tokio::test]
async fn failed_continuation_child_preflight_does_not_mutate_parent_state() {
    let mut child_builder = UntypedGraphBuilder::new("continuation_child");
    let child_in = child_builder.edge("child_in", number_type("Number"));
    let child_out = child_builder.edge("child_out", number_type("Number"));
    child_builder.set_entry(child_in).set_exit(child_out);
    child_builder.node(
        "copy",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![child_in],
        vec![child_out],
    );
    let child = child_builder.build().expect("child should build");

    let mut builder = UntypedGraphBuilder::new("continuation_child_preflight");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "continuation",
        NodeKind::Continuation {
            key: HandlerKey::new("continuation"),
            payload: Value::NULL,
            children: vec![child],
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_continuation("continuation", StartInvalidChildInput)
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(5), registry).expect("runtime should build");

    let err = runtime
        .next()
        .expect_err("invalid child input should fail before parent mutation");
    assert!(matches!(err, GraphError::Schema { .. }));
    let frame = runtime
        .state()
        .frames
        .first()
        .expect("parent frame remains");
    assert_eq!(runtime.state().frames.len(), 1);
    assert!(frame.checkpoints[0].is_none());
    assert!(frame.continuation_states[0].is_none());
    assert!(frame.continuation_child_queues[0].is_empty());
    assert!(frame.values[output.0].is_none());
    assert_eq!(frame.node_epochs[0], 0);
}

/// Verifies continuation checkpoint only polls later.
#[tokio::test]
async fn continuation_checkpoint_only_polls_later() {
    let mut builder = UntypedGraphBuilder::new("poll_continuation");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "poll_then_complete",
        NodeKind::Continuation {
            key: HandlerKey::new("poll_then_complete"),
            payload: Value::NULL,
            children: Vec::new(),
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_continuation("poll_then_complete", PollThenComplete)
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(4), registry).expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!(5)));
}

/// Verifies registry rejects duplicate keys within same handler class.
#[test]
fn registry_rejects_duplicate_keys_within_same_handler_class() {
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value("dup", run_value_handler(|_| Ok(Vec::new())))
        .expect("first value handler should insert");
    assert!(
        registry
            .insert_value("dup", run_value_handler(|_| Ok(Vec::new())))
            .is_err()
    );

    registry
        .insert_continuation("dup", PollThenComplete)
        .expect("first continuation handler should insert");
    assert!(
        registry
            .insert_continuation("dup", PollThenComplete)
            .is_err()
    );
}

/// Verifies validation rejects invalid builtin arities.
#[test]
fn validation_rejects_invalid_builtin_arities() {
    assert_invalid_builtin(BuiltinNode::Identity, 1, 2);
    assert_invalid_builtin(BuiltinNode::FanOut, 2, 1);
    assert_invalid_builtin(BuiltinNode::PackTuple, 0, 1);
    assert_invalid_builtin(BuiltinNode::UnpackTuple, 1, 0);
}

fn assert_invalid_builtin(op: BuiltinNode, input_count: usize, output_count: usize) {
    let mut builder = UntypedGraphBuilder::new("bad_builtin");
    let inputs = (0..input_count)
        .map(|index| builder.edge(format!("input_{index}"), number_type("Number")))
        .collect::<Vec<_>>();
    let outputs = (0..output_count)
        .map(|index| builder.edge(format!("output_{index}"), number_type("Number")))
        .collect::<Vec<_>>();
    let entry = inputs
        .first()
        .copied()
        .unwrap_or_else(|| builder.edge("entry", number_type("Number")));
    let exit = outputs
        .first()
        .copied()
        .unwrap_or_else(|| builder.edge("exit", number_type("Number")));
    builder.set_entry(entry).set_exit(exit);
    builder.node("bad_builtin", NodeKind::Builtin { op }, inputs, outputs);
    let err = builder
        .build()
        .expect_err("invalid builtin arity should fail validation");
    assert!(
        err.to_string().contains("invalid arity"),
        "unexpected error: {err}"
    );
}

/// Verifies inherit is visible in child frame.
#[tokio::test]
async fn inherit_is_visible_in_child_frame() {
    let child_in = EdgeId(0);
    let child_out = EdgeId(1);
    let inherited = VarId(0);
    let child = UntypedGraph {
        schema_version: UNTYPED_GRAPH_SCHEMA_VERSION,
        name: "child_inherit".into(),
        edges: vec![
            Edge {
                id: child_in,
                label: Some("child_in".into()),
                type_spec: number_type("Number"),
                producer: None,
                consumers: vec![NodeId(0)],
            },
            Edge {
                id: child_out,
                label: Some("child_out".into()),
                type_spec: number_type("Number"),
                producer: Some(NodeId(0)),
                consumers: Vec::new(),
            },
        ],
        variables: vec![Variable {
            id: inherited,
            key: VarKey::new("rust", "Bonus"),
            type_spec: number_type("Bonus"),
            scope: VarScope::Inherit,
            init: VarInit::Value(rv!(99)),
        }],
        marks: Vec::new(),
        nodes: vec![model::Node {
            id: NodeId(0),
            name: "load_parent_bonus".into(),
            kind: NodeKind::Load {
                var: inherited,
                key: HandlerKey::new("add"),
            },
            inputs: vec![child_in],
            outputs: vec![child_out],
        }],
        entry: child_in,
        exit: child_out,
    };

    let mut parent_builder = UntypedGraphBuilder::new("parent_variable");
    let parent_in = parent_builder.edge("parent_in", number_type("Number"));
    let after_child = parent_builder.edge("after_child", number_type("Number"));
    let parent_out = parent_builder.edge("parent_out", number_type("Number"));
    parent_builder.variable_with_value(
        VarKey::new("rust", "Bonus"),
        number_type("Bonus"),
        VarScope::Local,
        rv!(10),
    );
    parent_builder.set_entry(parent_in).set_exit(parent_out);
    parent_builder.node(
        "call_child",
        NodeKind::Subflow {
            graph: Box::new(child),
        },
        vec![parent_in],
        vec![after_child],
    );
    parent_builder.node(
        "copy",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![after_child],
        vec![parent_out],
    );
    let parent = parent_builder
        .build()
        .expect("parent variable should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value(
            "add",
            run_value_handler(|inputs| {
                Ok(vec![rv!(
                    inputs[0].as_i64().unwrap_or_default() + inputs[1].as_i64().unwrap_or_default()
                )])
            }),
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(parent, rv!(5), registry).expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!(15)));
}

/// Verifies child inherit uses default when parent variable is missing.
#[tokio::test]
async fn child_inherit_uses_default_when_parent_variable_is_missing() {
    let child_in = EdgeId(0);
    let child_out = EdgeId(1);
    let child_inherit = VarId(0);
    let child = UntypedGraph {
        schema_version: UNTYPED_GRAPH_SCHEMA_VERSION,
        name: "child_inherit_default".into(),
        edges: vec![
            Edge {
                id: child_in,
                label: Some("child_in".into()),
                type_spec: number_type("Number"),
                producer: None,
                consumers: vec![NodeId(0)],
            },
            Edge {
                id: child_out,
                label: Some("child_out".into()),
                type_spec: number_type("Number"),
                producer: Some(NodeId(0)),
                consumers: Vec::new(),
            },
        ],
        variables: vec![Variable {
            id: child_inherit,
            key: VarKey::new("rust", "Bonus"),
            type_spec: number_type("Bonus"),
            scope: VarScope::Inherit,
            init: VarInit::Value(rv!(7)),
        }],
        marks: Vec::new(),
        nodes: vec![model::Node {
            id: NodeId(0),
            name: "load_default_bonus".into(),
            kind: NodeKind::Load {
                var: child_inherit,
                key: HandlerKey::new("add"),
            },
            inputs: vec![child_in],
            outputs: vec![child_out],
        }],
        entry: child_in,
        exit: child_out,
    };

    let mut parent_builder = UntypedGraphBuilder::new("parent_without_variable");
    let parent_in = parent_builder.edge("parent_in", number_type("Number"));
    let parent_out = parent_builder.edge("parent_out", number_type("Number"));
    parent_builder.set_entry(parent_in).set_exit(parent_out);
    parent_builder.node(
        "call_child",
        NodeKind::Subflow {
            graph: Box::new(child),
        },
        vec![parent_in],
        vec![parent_out],
    );
    let parent = parent_builder.build().expect("parent should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value(
            "add",
            run_value_handler(|inputs| {
                Ok(vec![rv!(
                    inputs[0].as_i64().unwrap_or_default() + inputs[1].as_i64().unwrap_or_default()
                )])
            }),
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(parent, rv!(5), registry).expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!(12)));
}

/// Verifies child inherit copies parent variable when available.
#[tokio::test]
async fn child_inherit_copies_parent_variable_when_available() {
    let child_in = EdgeId(0);
    let child_out = EdgeId(1);
    let child_inherit = VarId(0);
    let child = UntypedGraph {
        schema_version: UNTYPED_GRAPH_SCHEMA_VERSION,
        name: "child_inherit_copy".into(),
        edges: vec![
            Edge {
                id: child_in,
                label: Some("child_in".into()),
                type_spec: number_type("Number"),
                producer: None,
                consumers: vec![NodeId(0)],
            },
            Edge {
                id: child_out,
                label: Some("child_out".into()),
                type_spec: number_type("Number"),
                producer: Some(NodeId(0)),
                consumers: Vec::new(),
            },
        ],
        variables: vec![Variable {
            id: child_inherit,
            key: VarKey::new("rust", "Bonus"),
            type_spec: number_type("Bonus"),
            scope: VarScope::Inherit,
            init: VarInit::Value(rv!(99)),
        }],
        marks: Vec::new(),
        nodes: vec![model::Node {
            id: NodeId(0),
            name: "load_parent_bonus_copy".into(),
            kind: NodeKind::Load {
                var: child_inherit,
                key: HandlerKey::new("add"),
            },
            inputs: vec![child_in],
            outputs: vec![child_out],
        }],
        entry: child_in,
        exit: child_out,
    };

    let mut parent_builder = UntypedGraphBuilder::new("parent_with_variable");
    let parent_in = parent_builder.edge("parent_in", number_type("Number"));
    let parent_out = parent_builder.edge("parent_out", number_type("Number"));
    parent_builder.variable_with_value(
        VarKey::new("rust", "Bonus"),
        number_type("Bonus"),
        VarScope::Local,
        rv!(10),
    );
    parent_builder.set_entry(parent_in).set_exit(parent_out);
    parent_builder.node(
        "call_child",
        NodeKind::Subflow {
            graph: Box::new(child),
        },
        vec![parent_in],
        vec![parent_out],
    );
    let parent = parent_builder.build().expect("parent should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value(
            "add",
            run_value_handler(|inputs| {
                Ok(vec![rv!(
                    inputs[0].as_i64().unwrap_or_default() + inputs[1].as_i64().unwrap_or_default()
                )])
            }),
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(parent, rv!(5), registry).expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Done(rv!(15)));
}

/// Verifies child inherit writes do not update parent frame.
#[tokio::test]
async fn child_inherit_writes_do_not_update_parent_frame() {
    let child_in = EdgeId(0);
    let child_out = EdgeId(1);
    let child_inherit = VarId(0);
    let child = UntypedGraph {
        schema_version: UNTYPED_GRAPH_SCHEMA_VERSION,
        name: "child_store_inherit_copy".into(),
        edges: vec![
            Edge {
                id: child_in,
                label: Some("child_in".into()),
                type_spec: number_type("Number"),
                producer: None,
                consumers: vec![NodeId(0)],
            },
            Edge {
                id: child_out,
                label: Some("child_out".into()),
                type_spec: number_type("Number"),
                producer: Some(NodeId(0)),
                consumers: Vec::new(),
            },
        ],
        variables: vec![Variable {
            id: child_inherit,
            key: VarKey::new("rust", "Bonus"),
            type_spec: number_type("Bonus"),
            scope: VarScope::Inherit,
            init: VarInit::Value(rv!(99)),
        }],
        marks: Vec::new(),
        nodes: vec![model::Node {
            id: NodeId(0),
            name: "store_child_bonus".into(),
            kind: NodeKind::Store {
                var: child_inherit,
                key: HandlerKey::new("store_input"),
            },
            inputs: vec![child_in],
            outputs: vec![child_out],
        }],
        entry: child_in,
        exit: child_out,
    };

    let mut parent_builder = UntypedGraphBuilder::new("parent_inherit_write");
    let parent_in = parent_builder.edge("parent_in", number_type("Number"));
    let after_child = parent_builder.edge("after_child", number_type("Number"));
    let parent_out = parent_builder.edge("parent_out", number_type("Number"));
    let parent_var = parent_builder.variable_with_value(
        VarKey::new("rust", "Bonus"),
        number_type("Bonus"),
        VarScope::Local,
        rv!(10),
    );
    parent_builder.set_entry(parent_in).set_exit(parent_out);
    parent_builder.node(
        "call_child",
        NodeKind::Subflow {
            graph: Box::new(child),
        },
        vec![parent_in],
        vec![after_child],
    );
    parent_builder.node(
        "load_updated_bonus",
        NodeKind::Load {
            var: parent_var,
            key: HandlerKey::new("add"),
        },
        vec![after_child],
        vec![parent_out],
    );
    let parent = parent_builder.build().expect("parent should build");
    let mut registry = HandlerRegistry::new();
    registry
        .insert_value(
            "store_input",
            run_value_handler(|inputs| Ok(vec![inputs[0].clone()])),
        )
        .expect("handler should insert");
    registry
        .insert_value(
            "add",
            run_value_handler(|inputs| {
                Ok(vec![rv!(
                    inputs[0].as_i64().unwrap_or_default() + inputs[1].as_i64().unwrap_or_default()
                )])
            }),
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(parent, rv!(5), registry).expect("runtime should build");

    let done = loop {
        match runtime.next().unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            other => panic!("expected continue or done, got {other:?}"),
        }
    };
    assert_eq!(done, rv!(15));
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TypedAmount {
    value: i64,
}

#[path = "tests/context.rs"]
mod context;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct LeftAmount {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct RightAmount {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ThirdAmount {
    value: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TypedBonus {
    value: i64,
}

/// Verifies typed variable handles drive load and store.
#[tokio::test]
async fn typed_variable_handles_drive_load_and_store() {
    let root = Flow::<TypedAmount>::root();
    let bonus = root.local(TypedBonus { value: 3 });
    let flow = root
        .load(&bonus, |mut amount, bonus| {
            amount.value += bonus.value;
            amount
        })
        .store(&bonus, |amount, _bonus| TypedBonus {
            value: amount.value,
        })
        .finish::<TypedAmount>()
        .expect("flow should compile");
    let mut runtime = flow
        .start(TypedAmount { value: 4 }, uuid::Uuid::nil())
        .expect("runtime should build");
    let done = loop {
        match runtime.next().unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    assert_eq!(output.value, 7);
}

/// Verifies typed load can change output type.
#[tokio::test]
async fn typed_load_can_change_output_type() {
    let root = Flow::<TypedAmount>::root();
    let bonus = root.local(TypedBonus { value: 5 });
    let flow = root
        .load(&bonus, |amount, bonus| LeftAmount {
            value: amount.value + bonus.value,
        })
        .finish::<TypedAmount>()
        .expect("flow should compile");
    let mut runtime = flow
        .start(TypedAmount { value: 4 }, uuid::Uuid::nil())
        .expect("runtime should build");
    let done = loop {
        match runtime.next().unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    assert_eq!(output.value, 9);
}

/// Verifies typed variable handle from other builder fails at finish.
#[test]
fn typed_variable_handle_from_other_builder_fails_at_finish() {
    let builder = TypedGraphBuilder::<TypedAmount>::new();
    let root = builder.root();
    let other = TypedGraphBuilder::<TypedAmount>::new();
    let foreign = other.local(TypedBonus { value: 1 });
    let output = builder.load(root, &foreign, |amount: TypedAmount, _bonus: TypedBonus| {
        amount
    });
    let err = match builder.finish(output) {
        Ok(_) => panic!("cross-builder variable handle should fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("variable must belong"));
}

/// Verifies typed fluent api supports current style map split merge.
#[tokio::test]
async fn typed_fluent_api_supports_current_style_map_split_merge() {
    let root = Flow::<TypedAmount>::root();
    let (left, right) = root
        .map(|mut amount| {
            amount.value += 1;
            amount
        })
        .split(|amount| {
            (
                LeftAmount {
                    value: amount.value,
                },
                RightAmount {
                    value: amount.value * 2,
                },
            )
        });
    let flow = left
        .merge(right, |(left, right)| TypedAmount {
            value: left.value + right.value,
        })
        .finish::<TypedAmount>()
        .expect("flow should finish");

    let mut runtime = flow
        .start(TypedAmount { value: 3 }, uuid::Uuid::nil())
        .expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    let done = runtime.next().unwrap();
    let Step::Done(value) = done else {
        panic!("expected done, got {done:?}");
    };
    let output = flow
        .decode_output(value)
        .expect("typed output should decode");
    assert_eq!(output.value, 12);
}

/// Verifies typed fluent api supports nary split merge.
#[tokio::test]
async fn typed_fluent_api_supports_nary_split_merge() {
    let root = Flow::<TypedAmount>::root();
    let (left, right, third) = root.split(|amount| {
        (
            LeftAmount {
                value: amount.value,
            },
            RightAmount {
                value: amount.value * 2,
            },
            ThirdAmount {
                value: amount.value * 3,
            },
        )
    });
    let flow = left
        .merge((right, third), |(left, right, third)| TypedAmount {
            value: left.value + right.value + third.value,
        })
        .finish::<TypedAmount>()
        .expect("flow should finish");

    let mut runtime = flow
        .start(TypedAmount { value: 2 }, uuid::Uuid::nil())
        .expect("runtime should build");

    assert_eq!(runtime.next().unwrap(), Step::Continue);
    let done = runtime.next().unwrap();
    let Step::Done(value) = done else {
        panic!("expected done, got {done:?}");
    };
    let output = flow
        .decode_output(value)
        .expect("typed output should decode");
    assert_eq!(output.value, 12);
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TypedChoice {
    value: i64,
}

fn typed_choice(root: Flow<TypedChoice>) -> Flow<TypedAmount> {
    root.either(|input| {
        if input.value < 0 {
            Either::Left(LeftAmount { value: input.value })
        } else {
            Either::Right(RightAmount { value: input.value })
        }
    })
    .branch(
        |left| {
            left.map(|input| TypedAmount {
                value: input.value.abs(),
            })
        },
        |right| {
            right.map(|input| TypedAmount {
                value: input.value * 2,
            })
        },
    )
}

/// Verifies typed fluent api supports either branch.
#[tokio::test]
async fn typed_fluent_api_supports_either_branch() {
    let flow = compile(typed_choice).expect("flow should compile");

    let mut runtime = flow
        .start(TypedChoice { value: -7 }, uuid::Uuid::nil())
        .expect("runtime should build");
    let done = loop {
        match runtime.next().unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    assert_eq!(output.value, 7);

    let mut runtime = flow
        .start(TypedChoice { value: 8 }, uuid::Uuid::nil())
        .expect("runtime should build");
    let done = loop {
        match runtime.next().unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    assert_eq!(output.value, 16);
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TypedItem {
    value: i64,
}

fn typed_item(root: Flow<TypedItem>) -> Flow<TypedAmount> {
    root.map(|input| TypedAmount {
        value: input.value + 10,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct TypedBatch {
    values: Vec<TypedItem>,
}

fn typed_batch(root: Flow<TypedBatch>) -> Flow<Vec<TypedAmount>> {
    root.map(|input| input.values).each(typed_item)
}

/// Verifies typed fluent api supports each.
#[tokio::test]
async fn typed_fluent_api_supports_each() {
    let flow = compile(typed_batch).expect("flow should compile");
    let mut runtime = flow
        .start(
            TypedBatch {
                values: vec![TypedItem { value: 1 }, TypedItem { value: 2 }],
            },
            uuid::Uuid::nil(),
        )
        .expect("runtime should build");
    let done = loop {
        match runtime.next().unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    let values: Vec<i64> = output.into_iter().map(|amount| amount.value).collect();
    assert_eq!(values, vec![11, 12]);
}

#[derive(Default)]
struct AddPayloadContinuation;

impl ContinuationHandler for AddPayloadContinuation {
    fn start<'a>(
        &'a self,
        payload: &'a Value,
        _state: Option<Value>,
        inputs: Vec<Value>,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        let input = decode_test_value::<TypedAmount>(inputs, "payload_effect")?;
        let add = payload
            .get("add")
            .and_then(Value::as_i64)
            .ok_or_else(|| GraphError::Invalid("missing payload add".into()))?;
        Ok(ContinuationTransition {
            fetch: None,
            history: Vec::new(),
            checkpoint: None,
            state: None,
            outputs: vec![rv!(TypedAmount {
                value: input.value + add
            })],
            writes: Vec::new(),
            child_calls: Vec::new(),
            suspension: None,
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _continuation: Value,
        _event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> Result<ContinuationTransition, GraphError> {
        Err(GraphError::Invalid(
            "payload continuation does not advance".into(),
        ))
    }
}

fn decode_test_value<T: for<'de> Deserialize<'de>>(
    mut inputs: Vec<Value>,
    node: &str,
) -> Result<T, GraphError> {
    if inputs.len() != 1 {
        return Err(GraphError::Invalid(format!("{node} expected one input")));
    }
    from_value(inputs.remove(0))
        .map_err(|err| GraphError::Invalid(format!("{node} decode failed: {err}")))
}

/// Verifies typed builder builds maps and continuation without fluent api.
#[tokio::test]
async fn typed_builder_builds_maps_and_continuation_without_fluent_api() {
    let builder = TypedGraphBuilder::<TypedAmount>::new();
    let root = builder.root();
    let mapped = builder.map(root, |input: TypedAmount| TypedAmount {
        value: input.value + 1,
    });
    let worked = builder.map(mapped, |input: TypedAmount| TypedAmount {
        value: input.value * 2,
    });
    let continued = builder.continuation::<TypedAmount, TypedAmount, AddPayloadContinuation, _>(
        worked,
        rv!({"add": 5}),
    );
    let flow = builder
        .finish(continued)
        .expect("typed builder should finish");
    let mut runtime = flow
        .start(TypedAmount { value: 3 }, uuid::Uuid::nil())
        .expect("runtime should build");

    let done = loop {
        match runtime.next().unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    assert_eq!(output.value, 13);
}

struct ExternalNode<I, T> {
    builder: TypedGraphBuilder<I>,
    edge: TypedEdge<T>,
}

impl<I, T> ExternalNode<I, T>
where
    I: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    T: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
{
    fn map<P>(self, func: impl Fn(T) -> P + Send + Sync + 'static) -> ExternalNode<I, P>
    where
        P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let edge = self.builder.map(self.edge, func);
        ExternalNode {
            builder: self.builder,
            edge,
        }
    }

    fn custom_continuation<P>(self, payload: Value) -> ExternalNode<I, P>
    where
        P: 'static + Serialize + DeserializeOwned + JsonSchema + Send + Sync,
    {
        let edge = self
            .builder
            .continuation::<T, P, AddPayloadContinuation, _>(self.edge, payload);
        ExternalNode {
            builder: self.builder,
            edge,
        }
    }

    fn finish(self) -> Result<CompiledFlow<I, T>, GraphError> {
        self.builder.finish(self.edge)
    }
}

/// Verifies external facade can wrap typed builder without edge node.
#[tokio::test]
async fn external_facade_can_wrap_typed_builder_without_edge_node() {
    let builder = TypedGraphBuilder::<TypedAmount>::new();
    let root = ExternalNode {
        edge: builder.root(),
        builder,
    };
    let flow = root
        .map(|input| TypedAmount {
            value: input.value + 2,
        })
        .custom_continuation::<TypedAmount>(rv!({"add": 4}))
        .finish()
        .expect("external facade should finish");
    let mut runtime = flow
        .start(TypedAmount { value: 1 }, uuid::Uuid::nil())
        .expect("runtime should build");

    let done = loop {
        match runtime.next().unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    assert_eq!(output.value, 7);
}

struct EdgeScriptedInner {
    responses: VecDeque<Result<ClientResponse, ClientError>>,
    calls: Vec<Vec<Message>>,
    creates: Vec<String>,
    options: Vec<ClientOptions>,
}

#[derive(Clone)]
struct EdgeScriptedFactory {
    inner: Arc<Mutex<EdgeScriptedInner>>,
}

impl EdgeScriptedFactory {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(EdgeScriptedInner {
                responses: VecDeque::new(),
                calls: Vec::new(),
                creates: Vec::new(),
                options: Vec::new(),
            })),
        }
    }

    fn then_output(self, value: JsonValue) -> Self {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .responses
            .push_back(Ok(ClientResponse::new(
                Provider::OpenAi,
                ClientOutput::Output(value),
            )));
        self
    }

    fn then_tool_calls(self, calls: Vec<ToolCall>) -> Self {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .responses
            .push_back(Ok(ClientResponse::new(
                Provider::OpenAi,
                ClientOutput::ToolCalls { text: None, calls },
            )));
        self
    }

    fn calls(&self) -> Vec<Vec<Message>> {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .calls
            .clone()
    }

    fn creates(&self) -> Vec<String> {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .creates
            .clone()
    }

    fn options(&self) -> Vec<ClientOptions> {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .options
            .clone()
    }
}

struct EdgeScriptedClient {
    inner: Arc<Mutex<EdgeScriptedInner>>,
    url: ModelUrl,
    options: ClientOptions,
}

impl LlmBackend for EdgeScriptedClient {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }

    fn options(&self) -> &ClientOptions {
        &self.options
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        inner.calls.push(messages.to_vec());
        match inner.responses.pop_front() {
            Some(response) => response,
            None => Err(ClientError::new(
                ErrorKind::Provider,
                "edge scripted response queue exhausted",
            )),
        }
    }
}

impl ProviderFactory for EdgeScriptedFactory {
    async fn llm(
        &self,
        model_url: &ModelUrl,
        options: ClientOptions,
    ) -> Result<Client, ClientError> {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .creates
            .push(format!(
                "{}:///{}",
                model_url.provider().as_str(),
                model_url.model()
            ));
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .options
            .push(options.clone());
        Ok(Client::from_backend(EdgeScriptedClient {
            inner: Arc::clone(&self.inner),
            url: model_url.clone(),
            options,
        }))
    }
}

fn edge_tool_call(id: &str, name: &str, args: JsonValue) -> ToolCall {
    ToolCall::new(id.into(), name.into(), args)
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct EdgeAgentInput {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct EdgeAgentOutput {
    text: String,
}

async fn configure_edge_agent(
    input: EdgeAgentInput,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test-model",
        "answer",
        Message::user(input.text),
    ))
}

async fn configure_edge_chat_agent(
    input: EdgeAgentInput,
    ctx: Context,
) -> Result<AgentConfig, GraphError> {
    configure_edge_agent(input, ctx)
        .await
        .map(AgentConfig::keep_alive)
}

fn edge_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.configure(configure_edge_agent)
}

fn edge_chat_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.configure(configure_edge_chat_agent)
}

struct ConfigureCalls(AtomicUsize);

async fn configure_counted_agent(
    input: EdgeAgentInput,
    ctx: Context,
) -> Result<AgentConfig, GraphError> {
    ctx.require::<ConfigureCalls>()
        .map_err(|err| GraphError::AgentConfigValidation(err.to_string()))?
        .0
        .fetch_add(1, Ordering::SeqCst);
    Ok(AgentConfig::new(
        "test:///test-model",
        "answer carefully",
        Message::user(input.text),
    )
    .memory("stable private memory"))
}

fn counted_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.configure(configure_counted_agent)
}

async fn configure_output_agent(
    input: EdgeAgentOutput,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test-model",
        "answer",
        Message::user(input.text),
    ))
}

fn missing_configure(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentInput> {
    root
}

fn repeated_configure(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.configure(configure_edge_agent)
        .configure(configure_output_agent)
}

fn tools_after_configure(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.configure(configure_edge_agent).tools(echo_tools)
}

fn counted_agent_flow(root: Flow<EdgeAgentInput>) -> Flow<EdgeAgentOutput> {
    root.agent(counted_agent)
}

fn missing_configure_flow(root: Flow<EdgeAgentInput>) -> Flow<EdgeAgentInput> {
    root.agent(missing_configure)
}

fn repeated_configure_flow(root: Flow<EdgeAgentInput>) -> Flow<EdgeAgentOutput> {
    root.agent(repeated_configure)
}

fn tools_after_configure_flow(root: Flow<EdgeAgentInput>) -> Flow<EdgeAgentOutput> {
    root.agent(tools_after_configure)
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct EdgeProviderConfigAgentInput {
    text: String,
}

async fn configure_provider_agent(
    input: EdgeProviderConfigAgentInput,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///gemini-2.5-flash",
        "answer",
        Message::user(input.text),
    )
    .provider_config(serde_json::json!({
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    })))
}

fn provider_agent(root: Agent<EdgeProviderConfigAgentInput>) -> Agent<EdgeAgentOutput> {
    root.configure(configure_provider_agent)
}

/// Verifies typed edge agent without tools uses structured output.
#[tokio::test]
async fn typed_edge_agent_without_tools_uses_structured_output() -> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");

    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    let done = match host::step(&mut runtime, &executor).await.unwrap() {
        Step::Done(value) => flow.decode_output(value).unwrap(),
        other => panic!("expected done, got {other:?}"),
    };

    assert_eq!(
        done,
        EdgeAgentOutput {
            text: "done".into()
        }
    );
    assert_eq!(factory.calls().len(), 1);
    let snapshot = runtime.snapshot().expect("snapshot should include history");
    let session_id = snapshot
        .history
        .entries()
        .first()
        .map(|entry| entry.session_id.as_str())
        .expect("agent history should retain a session");
    let all_messages = snapshot.history.for_session(session_id);
    assert!(
        all_messages.len() >= 2,
        "completed agent history should survive in runtime snapshot"
    );
    Ok(())
}

/// Verifies function-defined agents infer their output and compile symmetrically with flows.
#[test]
fn symmetric_agent_function_infers_output_type() {
    let flow: CompiledFlow<EdgeAgentInput, EdgeAgentOutput> =
        compile(counted_agent_flow).expect("symmetric agent flow should compile");

    let graph = flow.graph();
    assert_eq!(
        graph.edges[graph.entry.0].type_spec.name,
        EdgeAgentInput::schema_name()
    );
    assert_eq!(
        graph.edges[graph.exit.0].type_spec.name,
        EdgeAgentOutput::schema_name()
    );
}

/// Verifies missing, repeated, and non-terminal configuration errors surface at compile time.
#[test]
fn agent_definition_errors_accumulate_until_compile() {
    let missing = match compile(missing_configure_flow) {
        Ok(_) => panic!("configure should be required"),
        Err(err) => err,
    };
    let repeated = match compile(repeated_configure_flow) {
        Ok(_) => panic!("configure should be terminal"),
        Err(err) => err,
    };
    let late_tools = match compile(tools_after_configure_flow) {
        Ok(_) => panic!("tools after configure should fail"),
        Err(err) => err,
    };

    assert!(
        missing
            .to_string()
            .contains("configure function is required")
    );
    assert!(repeated.to_string().contains("only be declared once"));
    assert!(late_tools.to_string().contains("before configure"));
}

/// Verifies controller declarations are unique and must precede terminal configuration.
#[test]
fn agent_control_definition_errors_accumulate_until_compile() {
    let repeated = match compile(repeated_control_flow) {
        Ok(_) => panic!("control should be declared only once"),
        Err(err) => err,
    };
    let late = match compile(control_after_configure_flow) {
        Ok(_) => panic!("control after configure should fail"),
        Err(err) => err,
    };

    assert!(repeated.to_string().contains("control"));
    assert!(repeated.to_string().contains("only be declared once"));
    assert!(late.to_string().contains("control"));
    assert!(late.to_string().contains("before configure"));
}

/// Verifies result-aware control narrows and later widens tools in prepared order.
#[tokio::test]
async fn adaptive_agent_control_observes_boundaries_and_changes_tool_visibility()
-> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(adaptive_agent)
        .finish::<EdgeAgentInput>()
        .expect("controlled agent should compile");
    let factory = EdgeScriptedFactory::new()
        .then_tool_calls(vec![edge_tool_call(
            "echo-call",
            "echo_in",
            serde_json::json!({ "text": "first" }),
        )])
        .then_tool_calls(vec![edge_tool_call(
            "suffix-call",
            "suffix_in",
            serde_json::json!({ "text": "second" }),
        )])
        .then_output(serde_json::json!({ "text": "done" }));
    let trace = Arc::new(ControlTrace::default());
    let mut deps = Deps::default();
    deps.insert(Arc::clone(&trace));
    let ctx = ctx()
        .with_deps(deps)
        .with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");

    let output = loop {
        match host::step(&mut runtime, &executor)
            .await
            .expect("agent step should run")
        {
            Step::Continue => {}
            Step::Done(value) => break flow.decode_output(value).expect("output should decode"),
            Step::Fetch(_) => panic!("unexpected undelivered fetch"),
            Step::Suspend(_) => panic!("controller should not suspend"),
        }
    };

    assert_eq!(output.text, "done");
    let options = factory.options();
    let exposed = options
        .iter()
        .map(|option| {
            option
                .tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        exposed,
        vec![
            vec!["echo_in", "suffix_in"],
            vec!["suffix_in"],
            vec!["echo_in", "suffix_in"]
        ]
    );
    let observations = trace.0.lock().unwrap_or_else(|err| err.into_inner());
    let points = observations
        .iter()
        .map(|observation| observation.point)
        .collect::<Vec<_>>();
    assert_eq!(
        points,
        vec![
            AgentInterventionPoint::BeforeModel,
            AgentInterventionPoint::BeforeTools,
            AgentInterventionPoint::AfterTools,
            AgentInterventionPoint::BeforeTools,
            AgentInterventionPoint::AfterTools,
        ]
    );
    assert_eq!(observations[2].model_turns, 1);
    assert_eq!(observations[4].model_turns, 2);
    assert_eq!(observations[3].active_tools, vec!["suffix_in"]);
    assert_eq!(observations[4].result_errors, vec![false]);
    Ok(())
}

/// Verifies inactive calls are not executed and forced conclusion disables every tool.
#[tokio::test]
async fn agent_control_recovers_from_hidden_tool_calls_and_forces_conclusion()
-> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(hidden_tool_agent)
        .finish::<EdgeAgentInput>()
        .expect("hidden-tool agent should compile");
    let factory = EdgeScriptedFactory::new()
        .then_tool_calls(vec![
            edge_tool_call(
                "hidden-call",
                "suffix_in",
                serde_json::json!({ "text": "hidden" }),
            ),
            edge_tool_call(
                "valid-call",
                "echo_in",
                serde_json::json!({ "text": "valid" }),
            ),
        ])
        .then_output(serde_json::json!({ "text": "concluded" }));
    let observed = Arc::new(HiddenToolResults::default());
    let mut deps = Deps::default();
    deps.insert(Arc::clone(&observed));
    let ctx = ctx()
        .with_deps(deps)
        .with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");

    let output = loop {
        match host::step(&mut runtime, &executor)
            .await
            .expect("agent step should run")
        {
            Step::Continue => {}
            Step::Done(value) => break flow.decode_output(value).expect("output should decode"),
            Step::Fetch(_) => panic!("unexpected undelivered fetch"),
            Step::Suspend(_) => panic!("controller should not suspend"),
        }
    };

    assert_eq!(output.text, "concluded");
    assert_eq!(
        *observed.0.lock().unwrap_or_else(|err| err.into_inner()),
        vec![("suffix_in".into(), true), ("echo_in".into(), false)]
    );
    let options = factory.options();
    assert_eq!(
        options[0]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        vec!["echo_in"]
    );
    assert!(options[1].tools.is_empty());
    let calls = factory.calls();
    assert!(calls[1].iter().any(|message| {
        matches!(message.role, Role::Tool { .. })
            && message.content.contains("tool unavailable for this turn")
    }));
    assert!(calls[1].iter().any(|message| {
        matches!(message.role, Role::Tool { .. }) && message.content.contains("VALID")
    }));
    Ok(())
}

/// Verifies a policy abort is retryable and commits no checkpoint or history changes.
#[tokio::test]
async fn agent_policy_abort_leaves_runtime_retryable() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(aborting_agent)
        .finish::<EdgeAgentInput>()
        .expect("aborting agent should compile");
    let executor = FetchExecutor::new(ctx()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");
    assert_eq!(
        host::step(&mut runtime, &executor)
            .await
            .expect("configure should run"),
        Step::Continue
    );
    assert!(matches!(runtime.next().unwrap(), Step::Fetch(_)));
    let history = serde_json::to_value(runtime.snapshot().unwrap().history()).unwrap();
    let err = host::step(&mut runtime, &executor)
        .await
        .expect_err("policy should abort the boundary");
    assert!(matches!(err, GraphError::AgentPolicyAbort { .. }));
    let accepted = serde_json::to_value(runtime.snapshot().unwrap()).unwrap();
    assert_eq!(
        history,
        serde_json::to_value(runtime.snapshot().unwrap().history()).unwrap()
    );
    assert!(matches!(
        runtime.next(),
        Err(GraphError::AgentPolicyAbort { .. })
    ));
    assert_eq!(
        accepted,
        serde_json::to_value(runtime.snapshot().unwrap()).unwrap()
    );
}

/// Verifies forced conclusion rejects invalid structured output before history mutation.
#[tokio::test]
async fn forced_agent_conclusion_validates_output_before_commit() -> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(concluding_agent)
        .finish::<EdgeAgentInput>()
        .expect("concluding agent should compile");
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "wrong": true }));
    let ctx = ctx().with_providers(crate::testing::providers(factory)?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");
    assert_eq!(
        host::step(&mut runtime, &executor)
            .await
            .expect("configure should run"),
        Step::Continue
    );
    assert_eq!(
        host::step(&mut runtime, &executor)
            .await
            .expect("conclusion decision should commit"),
        Step::Continue
    );
    let history = serde_json::to_value(runtime.snapshot()?.history()).unwrap();
    let err = host::finish(&mut runtime, &executor)
        .await
        .expect_err("invalid conclusion output should fail");
    assert!(matches!(err, GraphError::AgentConclusion { .. }));
    let accepted = serde_json::to_value(runtime.snapshot()?).unwrap();
    assert_eq!(
        history,
        serde_json::to_value(runtime.snapshot()?.history()).unwrap()
    );
    assert!(runtime.pending_fetch().is_none());
    assert!(matches!(
        runtime.next(),
        Err(GraphError::AgentConclusion { .. })
    ));
    assert_eq!(accepted, serde_json::to_value(runtime.snapshot()?).unwrap());
    Ok(())
}

/// Verifies an agent suspension survives JSON and CBOR restore and resumes at its owner.
#[tokio::test]
async fn agent_controller_suspension_restores_and_accepts_typed_resume()
-> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(suspending_agent)
        .finish::<EdgeAgentInput>()
        .expect("suspending agent should compile");
    let factory = EdgeScriptedFactory::new()
        .then_tool_calls(vec![edge_tool_call(
            "echo-call",
            "echo_in",
            serde_json::json!({ "text": "hello" }),
        )])
        .then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory)?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");

    let payload = loop {
        match host::step(&mut runtime, &executor)
            .await
            .expect("agent step should run")
        {
            Step::Continue => {}
            Step::Suspend(payload) => break payload,
            Step::Done(_) => panic!("agent should suspend before its tool"),
            Step::Fetch(_) => panic!("host must deliver fetch"),
        }
    };
    let suspension: AgentSuspension = from_value(payload).expect("agent suspension should decode");
    assert_eq!(suspension.point(), AgentInterventionPoint::BeforeTools);
    assert_eq!(suspension.payload(), &rv!({ "reason": "approval" }));

    let snapshot = runtime.snapshot().expect("snapshot should encode");
    let json = serde_json::to_vec(&snapshot).expect("snapshot should encode as JSON");
    let restored_json: Snapshot =
        serde_json::from_slice(&json).expect("snapshot should decode from JSON");
    let mut cbor = Vec::new();
    ciborium::into_writer(&restored_json, &mut cbor).expect("snapshot should encode as CBOR");
    let restored_cbor: Snapshot =
        ciborium::from_reader(cbor.as_slice()).expect("snapshot should decode from CBOR");
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .restore(restored_cbor)
        .expect("snapshot should restore");

    let before_invalid = serde_json::to_vec(&runtime.snapshot().expect("snapshot should encode"))
        .expect("snapshot should encode as JSON");
    let invalid = AgentResume::Redirect {
        guidance: None,
        tools: Some(vec!["echo_in".into(), "echo_in".into()]),
    };
    let mut invalid_runtime = flow.restore(runtime.snapshot()?)?;
    invalid_runtime.resume(invalid)?;
    let accepted = serde_json::to_value(invalid_runtime.snapshot()?).unwrap();
    let err = invalid_runtime
        .next()
        .expect_err("duplicate resume tools should fail on execution");
    assert!(matches!(err, GraphError::AgentResumeValidation(_)));
    assert_eq!(
        accepted,
        serde_json::to_value(invalid_runtime.snapshot()?).unwrap()
    );
    let after_invalid = serde_json::to_vec(&runtime.snapshot().expect("snapshot should encode"))
        .expect("snapshot should encode as JSON");
    assert_eq!(before_invalid, after_invalid);

    runtime
        .resume(AgentResume::Continue)
        .expect("resume should succeed");
    let output = loop {
        match host::step(&mut runtime, &executor)
            .await
            .expect("agent step should run")
        {
            Step::Continue => {}
            Step::Done(value) => break flow.decode_output(value).expect("output should decode"),
            Step::Fetch(_) => panic!("unexpected undelivered fetch"),
            Step::Suspend(_) => panic!("committed boundary should not be reevaluated"),
        }
    };
    assert_eq!(output.text, "done");
    Ok(())
}

/// Verifies JSON invocation routes an `AgentResume` value to a controlled agent.
#[tokio::test]
async fn json_invoker_resumes_agent_controller_suspension() -> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(suspending_agent)
        .finish::<EdgeAgentInput>()
        .expect("suspending agent should compile");
    let (graph, registry) = flow.into_parts();
    let invoker = JsonInvoker::new(graph, registry.clone()).expect("invoker should prepare");
    let factory = EdgeScriptedFactory::new()
        .then_tool_calls(vec![edge_tool_call(
            "echo-call",
            "echo_in",
            serde_json::json!({ "text": "hello" }),
        )])
        .then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory)?);
    let mut response = invoker
        .invoke(JsonRequest::Start {
            execution_id: uuid::Uuid::nil(),
            version: JSON_WIRE_VERSION,
            input: serde_json::json!({"text": "hi"}),
        })
        .expect("agent should start");

    let executor = FetchExecutor::new(ctx).with_registry(Arc::new(registry));
    let snapshot = advance_json_until_agent_suspend(&invoker, response, &executor).await;

    response = invoker
        .invoke(JsonRequest::Resume {
            version: JSON_WIRE_VERSION,
            snapshot,
            input: serde_json::to_value(AgentResume::Continue)
                .expect("resume should encode as JSON"),
        })
        .expect("agent should resume");
    let output = advance_json_until_done(&invoker, response, &executor).await;
    assert_eq!(output, serde_json::json!({"text": "done"}));
    Ok(())
}

/// Advances stateless JSON requests until the controlled agent suspends.
async fn advance_json_until_agent_suspend(
    invoker: &JsonInvoker,
    mut response: JsonResponse,
    executor: &FetchExecutor,
) -> Snapshot {
    loop {
        match response {
            JsonResponse::Fetch {
                fetch, snapshot, ..
            } => {
                let outcome = Ok(executor.execute(&fetch).await.unwrap());
                response = invoker
                    .invoke(JsonRequest::ResumeFetch {
                        version: JSON_WIRE_VERSION,
                        snapshot,
                        id: fetch.id(),
                        outcome,
                    })
                    .unwrap();
            }
            JsonResponse::Continue { snapshot, .. } => {
                response = invoker
                    .invoke(JsonRequest::Next {
                        version: JSON_WIRE_VERSION,
                        snapshot,
                    })
                    .expect("agent should advance");
            }
            JsonResponse::Suspend {
                payload,
                resume_type,
                snapshot,
                ..
            } => {
                let suspension: AgentSuspension = serde_json::from_value(payload)
                    .expect("agent suspension should decode from JSON");
                assert_eq!(suspension.point(), AgentInterventionPoint::BeforeTools);
                assert_eq!(resume_type, AgentResume::schema_name());
                return snapshot;
            }
            JsonResponse::Done { .. } => panic!("agent should suspend before completion"),
        }
    }
}

/// Advances resumed stateless JSON requests until the agent completes.
async fn advance_json_until_done(
    invoker: &JsonInvoker,
    mut response: JsonResponse,
    executor: &FetchExecutor,
) -> JsonValue {
    loop {
        match response {
            JsonResponse::Fetch {
                fetch, snapshot, ..
            } => {
                let outcome = Ok(executor.execute(&fetch).await.unwrap());
                response = invoker
                    .invoke(JsonRequest::ResumeFetch {
                        version: JSON_WIRE_VERSION,
                        snapshot,
                        id: fetch.id(),
                        outcome,
                    })
                    .unwrap();
            }
            JsonResponse::Continue { snapshot, .. } => {
                response = invoker
                    .invoke(JsonRequest::Next {
                        version: JSON_WIRE_VERSION,
                        snapshot,
                    })
                    .expect("agent should advance after resume");
            }
            JsonResponse::Done { output, .. } => return output,
            JsonResponse::Suspend { .. } => panic!("committed boundary should not suspend again"),
        }
    }
}

/// Verifies activation is checkpointed once and memory remains outside conversation history.
#[tokio::test]
async fn agent_configuration_runs_once_across_snapshot_restore() -> Result<(), crate::GraphError> {
    let flow = compile(counted_agent_flow).expect("counted agent should compile");
    let calls = Arc::new(ConfigureCalls(AtomicUsize::new(0)));
    let mut deps = Deps::default();
    deps.insert(Arc::clone(&calls));
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx()
        .with_deps(deps)
        .with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");

    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert_eq!(calls.0.load(Ordering::SeqCst), 1);
    let snapshot = runtime
        .snapshot()
        .expect("configured state should snapshot");
    let json = serde_json::to_string(&snapshot).expect("snapshot should encode as JSON");
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor).expect("snapshot should encode as CBOR");
    assert!(json.contains("stable private memory"));
    assert!(
        !snapshot
            .history()
            .entries()
            .iter()
            .any(|entry| { entry.message.content.contains("stable private memory") })
    );

    let snapshot =
        ciborium::from_reader(cbor.as_slice()).expect("snapshot should decode from CBOR");
    let mut restored = flow
        .restore(snapshot)
        .expect("configured runtime should restore");
    assert!(matches!(
        host::finish(&mut restored, &executor).await.unwrap(),
        Step::Done(_)
    ));
    assert_eq!(calls.0.load(Ordering::SeqCst), 1);
    let preamble = factory.options()[0]
        .preamble
        .clone()
        .expect("agent preamble should be set");
    assert!(preamble.contains("<memory>\nstable private memory\n</memory>"));
    Ok(())
}

/// Verifies graph chat uses one runtime across turns.
#[tokio::test]
async fn graph_chat_uses_one_runtime_across_turns() -> Result<(), crate::GraphError> {
    let factory = EdgeScriptedFactory::new()
        .then_output(serde_json::json!({ "text": "first" }))
        .then_output(serde_json::json!({ "text": "second" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory.clone())?);
    let mut chat = Chat::<EdgeAgentInput, EdgeAgentOutput>::new(edge_chat_agent, ctx)
        .expect("chat initializes");

    let first = chat
        .send(EdgeAgentInput { text: "hi".into() })
        .await
        .expect("first chat turn should run");
    assert_eq!(
        first.output,
        EdgeAgentOutput {
            text: "first".into()
        }
    );

    let second = chat
        .send(EdgeAgentInput {
            text: "again".into(),
        })
        .await
        .expect("second chat turn should run");
    assert_eq!(
        second.output,
        EdgeAgentOutput {
            text: "second".into()
        }
    );
    assert_eq!(factory.calls().len(), 2);
    assert!(
        chat.snapshot()
            .expect("chat snapshot should exist")
            .history()
            .entries()
            .len()
            >= 4
    );
    Ok(())
}

/// Verifies services attached after chat restoration are applied to its runtime.
#[tokio::test]
async fn restored_graph_chat_uses_reattached_history_store() -> Result<(), crate::GraphError> {
    let factory = EdgeScriptedFactory::new()
        .then_output(serde_json::json!({ "text": "first" }))
        .then_output(serde_json::json!({ "text": "second" }));
    let calls = Arc::new(AtomicUsize::new(0));
    let store = FailAtHistoryRecord {
        calls: Arc::clone(&calls),
        fail_at: usize::MAX,
    };
    let ctx = ctx().with_providers(crate::testing::providers(factory)?);
    let mut chat = Chat::new(edge_chat_agent, ctx.clone())
        .expect("chat initializes")
        .with_store(store.clone());

    chat.send(EdgeAgentInput { text: "hi".into() })
        .await
        .expect("first chat turn should run");
    let recorded_before_restore = calls.load(Ordering::SeqCst);
    let snapshot = chat.snapshot().expect("chat should snapshot");
    let mut restored = Chat::<_, _>::from_snapshot(edge_chat_agent, snapshot, ctx)
        .expect("chat should restore")
        .with_store(store);

    restored
        .send(EdgeAgentInput {
            text: "again".into(),
        })
        .await
        .expect("restored chat turn should run");
    assert!(calls.load(Ordering::SeqCst) > recorded_before_restore);
    Ok(())
}

/// Verifies typed edge agent provider config reaches client options.
#[tokio::test]
async fn typed_edge_agent_provider_config_reaches_client_options() -> Result<(), crate::GraphError>
{
    let flow = Flow::<EdgeProviderConfigAgentInput>::root()
        .agent(provider_agent)
        .finish::<EdgeProviderConfigAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(
            EdgeProviderConfigAgentInput { text: "hi".into() },
            uuid::Uuid::nil(),
        )
        .expect("runtime should build");

    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert!(matches!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Done(_)
    ));

    let options = factory.options();
    let provider_config = options
        .first()
        .and_then(|opts| opts.provider_config.as_ref())
        .expect("provider config should reach graph agent client options");
    assert_eq!(
        provider_config["safety_settings"][0]["category"],
        "HARM_CATEGORY_DANGEROUS_CONTENT"
    );
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct EchoIn {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct EchoOut {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SuffixIn {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SuffixOut {
    text: String,
}

async fn echo_tool(input: EchoIn, _ctx: Context) -> Result<EchoOut, ToolError> {
    Ok(EchoOut {
        text: input.text.to_uppercase(),
    })
}

async fn suffix_tool(input: SuffixIn, _ctx: Context) -> Result<SuffixOut, ToolError> {
    Ok(SuffixOut {
        text: format!("{}!", input.text),
    })
}

fn echo_tools(tools: Toolset) -> Toolset {
    tools.tool(echo_tool)
}

fn two_tools(tools: Toolset) -> Toolset {
    echo_tools(tools).tool(suffix_tool)
}

fn duplicate_tools(tools: Toolset) -> Toolset {
    tools.tool(echo_tool).tool(echo_tool)
}

fn edge_agent_with_echo(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.tools(echo_tools).configure(configure_edge_agent)
}

fn edge_agent_with_two_tools(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.tools(two_tools).configure(configure_edge_agent)
}

fn edge_agent_with_duplicate_tools(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.tools(duplicate_tools).configure(configure_edge_agent)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ControlObservation {
    point: AgentInterventionPoint,
    active_tools: Vec<String>,
    result_errors: Vec<bool>,
    model_turns: u64,
}

#[derive(Default)]
struct ControlTrace(Mutex<Vec<ControlObservation>>);

async fn adaptive_controller(
    loop_: AgentLoop<EdgeAgentInput>,
    ctx: Context,
) -> Result<AgentDecision, GraphError> {
    let observation = ControlObservation {
        point: loop_.point(),
        active_tools: loop_
            .active_tools()
            .iter()
            .map(|tool| tool.name().to_owned())
            .collect(),
        result_errors: loop_
            .results()
            .iter()
            .map(AgentToolResult::is_error)
            .collect(),
        model_turns: loop_.metrics().model_turns(),
    };
    ctx.require::<ControlTrace>()
        .map_err(|err| GraphError::AgentControl {
            agent: loop_.agent_id().to_owned(),
            reason: err.to_string(),
        })?
        .0
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .push(observation);
    let result = match (loop_.point(), loop_.control_state().and_then(Value::as_u64)) {
        (AgentInterventionPoint::AfterTools, None) => AgentDecision::redirect()
            .guidance("Use the suffix tool next")
            .tools(ToolFilter::new(|tool| tool.name() == "suffix_in"))
            .with_state(Value::from(1_u64)),
        (AgentInterventionPoint::AfterTools, Some(1)) => AgentDecision::redirect()
            .tools(ToolFilter::all())
            .with_state(Value::from(2_u64)),
        _ => AgentDecision::continue_(),
    };
    Ok(result)
}

fn adaptive_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.tools(two_tools)
        .control(adaptive_controller)
        .configure(configure_edge_agent)
}

async fn suspend_before_tools_controller(
    loop_: AgentLoop<EdgeAgentInput>,
    _ctx: Context,
) -> Result<AgentDecision, GraphError> {
    if loop_.point() == AgentInterventionPoint::BeforeTools {
        Ok(AgentDecision::suspend(rv!({ "reason": "approval" })))
    } else {
        Ok(AgentDecision::continue_())
    }
}

fn suspending_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.tools(echo_tools)
        .control(suspend_before_tools_controller)
        .configure(configure_edge_agent)
}

#[derive(Default)]
struct HiddenToolResults(Mutex<Vec<(String, bool)>>);

async fn hidden_tool_controller(
    loop_: AgentLoop<EdgeAgentInput>,
    ctx: Context,
) -> Result<AgentDecision, GraphError> {
    if loop_.point() != AgentInterventionPoint::AfterTools {
        return Ok(AgentDecision::continue_());
    }
    let results: Vec<(String, bool)> = loop_
        .results()
        .iter()
        .map(|result| (result.tool_name().to_owned(), result.is_error()))
        .collect();
    ctx.require::<HiddenToolResults>()
        .map_err(|err| GraphError::AgentControl {
            agent: loop_.agent_id().to_owned(),
            reason: err.to_string(),
        })?
        .0
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .extend(results);
    Ok(AgentDecision::conclude(
        "Return the structured answer from the available evidence",
    ))
}

async fn configure_echo_only_agent(
    input: EdgeAgentInput,
    ctx: Context,
) -> Result<AgentConfig, GraphError> {
    configure_edge_agent(input, ctx)
        .await
        .map(|config| config.tool_filter(ToolFilter::new(|tool| tool.name() == "echo_in")))
}

fn hidden_tool_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.tools(two_tools)
        .control(hidden_tool_controller)
        .configure(configure_echo_only_agent)
}

async fn abort_controller(
    _loop: AgentLoop<EdgeAgentInput>,
    _ctx: Context,
) -> Result<AgentDecision, GraphError> {
    Ok(AgentDecision::abort(
        "application policy declined this turn",
    ))
}

fn aborting_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.control(abort_controller)
        .configure(configure_edge_agent)
}

async fn conclude_controller(
    _loop: AgentLoop<EdgeAgentInput>,
    _ctx: Context,
) -> Result<AgentDecision, GraphError> {
    Ok(AgentDecision::conclude(
        "Return the final structured answer",
    ))
}

fn concluding_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.control(conclude_controller)
        .configure(configure_edge_agent)
}

async fn continue_controller(
    _loop: AgentLoop<EdgeAgentInput>,
    _ctx: Context,
) -> Result<AgentDecision, GraphError> {
    Ok(AgentDecision::continue_())
}

async fn continue_output_controller(
    _loop: AgentLoop<EdgeAgentOutput>,
    _ctx: Context,
) -> Result<AgentDecision, GraphError> {
    Ok(AgentDecision::continue_())
}

fn repeated_control_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.control(continue_controller)
        .control(continue_controller)
        .configure(configure_edge_agent)
}

fn control_after_configure_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.configure(configure_edge_agent)
        .control(continue_output_controller)
}

fn repeated_control_flow(root: Flow<EdgeAgentInput>) -> Flow<EdgeAgentOutput> {
    root.agent(repeated_control_agent)
}

fn control_after_configure_flow(root: Flow<EdgeAgentInput>) -> Flow<EdgeAgentOutput> {
    root.agent(control_after_configure_agent)
}

async fn configure_filtered_agent(
    input: EdgeAgentInput,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    let selected = input.text;
    Ok(AgentConfig::new(
        "test:///test-model",
        "answer",
        Message::user("filtered request"),
    )
    .tool_filter(ToolFilter::new(move |tool| tool.name() == selected)))
}

fn filtered_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.tools(two_tools).configure(configure_filtered_agent)
}

async fn configure_failing_agent(
    _input: EdgeAgentInput,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    Err(GraphError::Invalid("configuration unavailable".into()))
}

fn failing_agent(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.configure(configure_failing_agent)
}

/// Verifies a standalone tool function executes through the shared graph VM.
#[tokio::test]
async fn typed_edge_agent_tool_function_round_trips_through_same_vm()
-> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent_with_echo)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new()
        .then_tool_calls(vec![edge_tool_call(
            "c1",
            "echo_in",
            serde_json::json!({ "text": "hi" }),
        )])
        .then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");

    let done = loop {
        match host::step(&mut runtime, &executor).await.unwrap() {
            Step::Continue => {}
            Step::Done(value) => break flow.decode_output(value).unwrap(),
            other => panic!("expected continue or done, got {other:?}"),
        }
    };

    assert_eq!(
        done,
        EdgeAgentOutput {
            text: "done".into()
        }
    );
    let calls = factory.calls();
    assert_eq!(calls.len(), 2);
    assert!(
        calls[1]
            .iter()
            .any(|message| matches!(message.role, Role::Tool { .. })
                && message.content.contains("HI"))
    );
    Ok(())
}

/// Verifies a capturing runtime filter can expose only statically prepared tools.
#[tokio::test]
async fn typed_edge_agent_filters_tools_in_prepared_order() -> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(filtered_agent)
        .finish::<EdgeAgentInput>()
        .expect("filtered agent should compile");
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(
            EdgeAgentInput {
                text: "suffix_in".into(),
            },
            uuid::Uuid::nil(),
        )
        .expect("runtime should build");

    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert!(matches!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Done(_)
    ));
    let options = factory.options();
    let names = options[0]
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["suffix_in"]);
    Ok(())
}

/// Verifies configuration failure leaves the runtime snapshot retryable and history empty.
#[tokio::test]
async fn typed_edge_agent_configuration_failure_does_not_mutate_runtime() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(failing_agent)
        .finish::<EdgeAgentInput>()
        .expect("failing agent definition should compile");
    let executor = FetchExecutor::new(ctx()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");
    assert!(matches!(runtime.next().unwrap(), Step::Fetch(_)));
    let before = serde_json::to_value(runtime.snapshot().unwrap()).unwrap();

    let err = host::step(&mut runtime, &executor)
        .await
        .expect_err("configuration should fail");
    let after = serde_json::to_value(runtime.snapshot().unwrap()).unwrap();

    assert!(matches!(err, GraphError::AgentConfiguration { .. }));
    assert_eq!(before, after);
    assert!(runtime.snapshot().unwrap().history().entries().is_empty());
}

/// Verifies typed edge agent provider is resolved at dispatch.
#[tokio::test]
async fn typed_edge_agent_provider_is_resolved_at_dispatch() -> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");
    assert!(factory.creates().is_empty());

    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert!(factory.creates().is_empty());

    let done = match host::step(&mut runtime, &executor).await.unwrap() {
        Step::Done(value) => flow.decode_output(value).unwrap(),
        other => panic!("expected done, got {other:?}"),
    };

    assert_eq!(
        done,
        EdgeAgentOutput {
            text: "done".into()
        }
    );
    assert_eq!(factory.creates(), vec!["test:///test-model".to_string()]);
    Ok(())
}

/// Verifies typed edge agent multiple tool calls are queued on single vm stack.
#[tokio::test]
async fn typed_edge_agent_multiple_tool_calls_are_queued_on_single_vm_stack()
-> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent_with_two_tools)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new()
        .then_tool_calls(vec![
            edge_tool_call("c1", "echo_in", serde_json::json!({ "text": "hi" })),
            edge_tool_call("c2", "suffix_in", serde_json::json!({ "text": "bye" })),
        ])
        .then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");

    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert_eq!(runtime.state().frames.len(), 2);

    let done = loop {
        match runtime.next().unwrap() {
            Step::Continue => {}
            Step::Fetch(fetch) => {
                let encoded = serde_json::to_vec(&runtime.snapshot()?).unwrap();
                runtime = flow.restore(serde_json::from_slice(&encoded).unwrap())?;
                assert_eq!(runtime.pending_fetch().map(Fetch::id), Some(fetch.id()));
                runtime.resume_fetch(fetch.id(), Ok(executor.execute(&fetch).await?))?;
            }
            Step::Done(value) => break flow.decode_output(value).unwrap(),
            other => panic!("expected continue or done, got {other:?}"),
        }
    };

    assert_eq!(
        done,
        EdgeAgentOutput {
            text: "done".into()
        }
    );
    let calls = factory.calls();
    assert_eq!(calls.len(), 2);
    let tool_messages = calls[1]
        .iter()
        .filter_map(|message| match &message.role {
            Role::Tool { call_id } => Some((call_id.as_str(), message.content.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_messages.len(), 2);
    assert_eq!(tool_messages[0].0, "c1");
    assert!(tool_messages[0].1.contains("HI"));
    assert_eq!(tool_messages[1].0, "c2");
    assert!(tool_messages[1].1.contains("bye!"));
    Ok(())
}

/// Verifies typed edge agent same tool calls run in deterministic queue order.
#[tokio::test]
async fn typed_edge_agent_same_tool_calls_run_in_deterministic_queue_order()
-> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent_with_echo)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new()
        .then_tool_calls(vec![
            edge_tool_call("c1", "echo_in", serde_json::json!({ "text": "one" })),
            edge_tool_call("c2", "echo_in", serde_json::json!({ "text": "two" })),
        ])
        .then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_providers(crate::testing::providers(factory.clone())?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");

    let done = loop {
        match host::step(&mut runtime, &executor).await.unwrap() {
            Step::Continue => {}
            Step::Done(value) => break flow.decode_output(value).unwrap(),
            other => panic!("expected continue or done, got {other:?}"),
        }
    };

    assert_eq!(
        done,
        EdgeAgentOutput {
            text: "done".into()
        }
    );
    let calls = factory.calls();
    assert_eq!(calls.len(), 2);
    let tool_messages = calls[1]
        .iter()
        .filter_map(|message| match &message.role {
            Role::Tool { call_id } => Some((call_id.as_str(), message.content.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_messages.len(), 2);
    assert_eq!(tool_messages[0].0, "c1");
    assert!(tool_messages[0].1.contains("ONE"));
    assert_eq!(tool_messages[1].0, "c2");
    assert!(tool_messages[1].1.contains("TWO"));
    Ok(())
}

/// Verifies typed edge agent duplicate tool names fail at finish.
#[test]
fn typed_edge_agent_duplicate_tool_names_fail_at_finish() {
    let err = match Flow::<EdgeAgentInput>::root()
        .agent(edge_agent_with_duplicate_tools)
        .finish::<EdgeAgentInput>()
    {
        Ok(_) => panic!("duplicate tool names should fail"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("duplicate agent tool name"));
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ReusableStep(i64);

fn reusable_step(root: Flow<ReusableStep>) -> Flow<ReusableStep> {
    root.map(|ReusableStep(value)| ReusableStep(value + 1))
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct RepeatedFlowInput(i64);

fn repeated_flow(root: Flow<RepeatedFlowInput>) -> Flow<i64> {
    root.map(|RepeatedFlowInput(value)| ReusableStep(value))
        .flow(reusable_step)
        .flow(reusable_step)
        .map(|ReusableStep(value)| value)
}

/// Verifies one typed subflow can be embedded repeatedly without registry collisions.
#[tokio::test]
async fn typed_flow_reuses_same_subflow_with_namespaced_handlers() {
    let flow = compile(repeated_flow).expect("repeated subflow should compile");
    let executor = FetchExecutor::new(ctx()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(RepeatedFlowInput(1), uuid::Uuid::nil())
        .expect("runtime should build");

    loop {
        match host::step(&mut runtime, &executor)
            .await
            .expect("step should succeed")
        {
            Step::Continue => {}
            Step::Done(value) => {
                assert_eq!(flow.decode_output(value).unwrap(), 3);
                break;
            }
            Step::Fetch(_) => panic!("unexpected undelivered fetch"),
            Step::Suspend(_) => panic!("repeated subflow should not suspend"),
        }
    }
}

/// Verifies explicit typed node names reach the canonical graph and diagrams.
#[test]
fn typed_named_nodes_preserve_supplied_names() {
    let flow = Flow::<i64>::root()
        .map_named("meaningful_map", |value| value + 1)
        .map_named("meaningful_double", |value| value * 2)
        .finish::<i64>()
        .expect("named flow should compile");
    let names = flow
        .graph()
        .nodes
        .iter()
        .map(|node| node.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(names, vec!["meaningful_map", "meaningful_double"]);
}

/// Verifies validation compares every pair of writers to one frame variable.
#[test]
fn validation_rejects_non_adjacent_competing_variable_writers() {
    let mut builder = UntypedGraphBuilder::new("competing_writers");
    let number = number_type("Number");
    let input = builder.edge("input", number.clone());
    let left = builder.edge("left", number.clone());
    let right = builder.edge("right", number.clone());
    let left_written = builder.edge("left_written", number.clone());
    let merged = builder.edge("merged", array_type("Pair"));
    let output = builder.edge("output", array_type("Pair"));
    let right_written = builder.edge("right_written", number.clone());
    let var = builder.variable_with_value(
        VarKey::new("test", "Number"),
        number,
        VarScope::Local,
        rv!(0),
    );
    builder.set_entry(input).set_exit(output);
    builder.node(
        "fan_out",
        NodeKind::Builtin {
            op: BuiltinNode::FanOut,
        },
        vec![input],
        vec![left, right],
    );
    builder.node(
        "writer_a",
        NodeKind::Store {
            var,
            key: HandlerKey::new("a"),
        },
        vec![left],
        vec![left_written],
    );
    builder.node(
        "writer_b",
        NodeKind::Store {
            var,
            key: HandlerKey::new("b"),
        },
        vec![merged],
        vec![output],
    );
    builder.node(
        "writer_c",
        NodeKind::Store {
            var,
            key: HandlerKey::new("c"),
        },
        vec![right],
        vec![right_written],
    );
    builder.node(
        "join",
        NodeKind::Builtin {
            op: BuiltinNode::PackTuple,
        },
        vec![left_written, right_written],
        vec![merged],
    );

    let err = builder
        .build()
        .expect_err("unordered writers should be rejected");
    assert!(err.to_string().contains("competing unordered stores"));
}

/// Verifies snapshot restoration rejects every return-target kind when it does not match the parent call.
#[tokio::test]
async fn snapshot_rejects_corrupted_frame_return_chain() {
    let mut child_builder = UntypedGraphBuilder::new("snapshot_child");
    let child_in = child_builder.edge("input", number_type("Number"));
    let child_out = child_builder.edge("output", number_type("Number"));
    child_builder.set_entry(child_in).set_exit(child_out);
    child_builder.node(
        "copy",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![child_in],
        vec![child_out],
    );
    let child = child_builder.build().expect("child should build");
    let mut parent_builder = UntypedGraphBuilder::new("snapshot_parent");
    let parent_in = parent_builder.edge("input", number_type("Number"));
    let parent_out = parent_builder.edge("output", number_type("Number"));
    parent_builder.set_entry(parent_in).set_exit(parent_out);
    parent_builder.node(
        "child",
        NodeKind::Subflow {
            graph: Box::new(child),
        },
        vec![parent_in],
        vec![parent_out],
    );
    let graph = parent_builder.build().expect("parent should build");
    let prepared = PreparedGraph::new(graph, HandlerRegistry::new()).expect("graph should prepare");
    let mut runtime = prepared
        .start(rv!(1), uuid::Uuid::nil())
        .expect("runtime should build");
    assert_eq!(runtime.next().unwrap(), Step::Continue);
    let snapshot = runtime.snapshot().expect("snapshot should build");

    let mut wrong_root = snapshot.clone();
    wrong_root
        .state
        .frame_mut(0)
        .expect("root frame")
        .return_target = Some(ReturnTarget::Edge {
        parent_edge: parent_out,
    });
    assert!(matches!(
        prepared.restore(wrong_root),
        Err(GraphError::SnapshotValidation(_))
    ));

    let corrupt_targets = [
        ReturnTarget::Edge {
            parent_edge: parent_in,
        },
        ReturnTarget::Either {
            parent_node: NodeId(0),
        },
        ReturnTarget::Each {
            parent_node: NodeId(0),
        },
        ReturnTarget::Continuation {
            parent_node: NodeId(0),
            call_id: "unknown-child-call".to_owned(),
        },
    ];
    for target in corrupt_targets {
        let mut wrong_child = snapshot.clone();
        wrong_child
            .state
            .frame_mut(1)
            .expect("child frame")
            .return_target = Some(target);
        assert!(matches!(
            prepared.restore(wrong_child),
            Err(GraphError::SnapshotValidation(_))
        ));
    }
}

/// Verifies obsolete agent payloads are rejected during graph preparation.
#[test]
fn agent_rejects_obsolete_payload_version() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let (mut graph, registry) = flow.into_parts();
    let NodeKind::Continuation { payload, .. } = &mut graph.nodes[0].kind else {
        panic!("agent should compile to continuation");
    };
    let mut encoded = serde_json::to_value(&*payload).expect("payload should encode");
    encoded["version"] = serde_json::json!(2);
    *payload = to_value(encoded).expect("payload should enter runtime domain");
    let err = match test_runtime(graph, rv!({"text": "hello"}), registry) {
        Ok(_) => panic!("obsolete payload should fail preparation"),
        Err(err) => err,
    };

    assert!(matches!(err, GraphError::GraphValidation(_)));
    assert!(
        err.to_string()
            .contains("unsupported agent payload version 2")
    );
}

/// Verifies preparation checks controller payload presence against its runtime handler.
#[test]
fn agent_rejects_missing_registered_controller() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let (mut graph, registry) = flow.into_parts();
    let NodeKind::Continuation { key, payload, .. } = &mut graph.nodes[0].kind else {
        panic!("agent should compile to continuation");
    };
    let mut encoded = serde_json::to_value(&*payload).expect("payload should encode");
    encoded["control_handler_key"] = serde_json::json!(format!("{}::control", key.as_str()));
    *payload = to_value(encoded).expect("payload should enter runtime domain");

    let err = match PreparedGraph::new(graph, registry) {
        Ok(_) => panic!("missing controller should fail preparation"),
        Err(err) => err,
    };
    assert!(matches!(err, GraphError::MissingHandler(_)));
}

/// Verifies an agent payload cannot substitute its configure-handler identity.
#[test]
fn agent_rejects_mismatched_configure_handler_identity() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let (mut graph, registry) = flow.into_parts();
    let NodeKind::Continuation { payload, .. } = &mut graph.nodes[0].kind else {
        panic!("agent should compile to continuation");
    };
    let mut encoded = serde_json::to_value(&*payload).expect("payload should encode");
    encoded["configure_handler_key"] = serde_json::json!("substituted");
    encoded["agent_id"] = serde_json::json!("substituted");
    *payload = to_value(encoded).expect("payload should enter runtime domain");

    let err = match test_runtime(graph, rv!({"text": "hello"}), registry) {
        Ok(_) => panic!("mismatched handler identity should fail preparation"),
        Err(err) => err,
    };
    assert!(matches!(err, GraphError::GraphValidation(_)));
}

/// Verifies agent continuation payloads render as agent nodes in every graph diagram.
#[test]
fn graph_diagram_classifies_agent_payload() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let diagram = GraphDiagram::from_graph(flow.graph());

    assert!(
        diagram
            .nodes()
            .iter()
            .any(|node| node.kind == DiagramNodeKind::Agent)
    );
}

/// Verifies snapshots reject an incompatible agent checkpoint before restoration.
#[tokio::test]
async fn snapshot_rejects_obsolete_agent_checkpoint_version() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let executor = FetchExecutor::new(ctx()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");
    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    let mut snapshot = runtime.snapshot().expect("snapshot should build");
    let frame = snapshot.state.frame_mut(0).expect("root frame");
    let checkpoint = &mut Arc::make_mut(&mut frame.checkpoints)[0].value;
    let mut encoded = serde_json::to_value(&*checkpoint).expect("checkpoint should encode");
    encoded["version"] = serde_json::json!(3);
    *checkpoint = to_value(encoded).expect("checkpoint should enter runtime domain");

    assert!(matches!(
        flow.restore(snapshot),
        Err(GraphError::UnsupportedVersion { .. })
    ));
}

/// Verifies the externally seeded entry edge can never also be node-produced.
#[test]
fn validation_rejects_entry_edge_producer() {
    let mut builder = UntypedGraphBuilder::new("entry_producer");
    let input = builder.edge("input", number_type("Number"));
    let output = builder.edge("output", number_type("Number"));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "copy",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![input],
        vec![output],
    );
    let mut graph = builder.build().expect("baseline graph should build");
    graph.edges[input.0].producer = Some(NodeId(0));
    graph.nodes[0].outputs.push(input);

    let err = validation::validate_graph_shape(&graph).expect_err("entry producer should fail");
    assert!(err.to_string().contains("entry edge"));
}

/// Verifies repeated inline either branches receive independent handler namespaces.
#[test]
fn typed_flow_reuses_inline_either_branches() {
    let flow = Flow::<i64>::root()
        .either(|value| {
            if value >= 0 {
                Either::Left(value)
            } else {
                Either::Right(value)
            }
        })
        .branch(
            |left| left.map(|value| value + 1),
            |right| right.map(|value| -value),
        )
        .either(|value| {
            if value % 2 == 0 {
                Either::Left(value)
            } else {
                Either::Right(value)
            }
        })
        .branch(
            |left| left.map(|value| value),
            |right| right.map(|value| value + 1),
        )
        .finish::<i64>();

    assert!(flow.is_ok(), "repeated either branches should compile");
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct EchoToolFlow {
    text: String,
}

fn echo_tool_flow(root: Flow<EchoToolFlow>) -> Flow<EchoOut> {
    root.map(|input| EchoOut { text: input.text })
}

fn echo_flow_tools(tools: Toolset) -> Toolset {
    tools.flow(echo_tool_flow)
}

fn edge_agent_with_echo_flow(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.tools(echo_flow_tools).configure(configure_edge_agent)
}

fn controlled_agent_with_echo_flow(root: Agent<EdgeAgentInput>) -> Agent<EdgeAgentOutput> {
    root.tools(echo_flow_tools)
        .control(continue_controller)
        .configure(configure_edge_agent)
}

/// Verifies two agent nodes can embed the same tool flow without registry collisions.
#[test]
fn typed_flow_reuses_tool_flow_across_agents() {
    let root = Flow::<EdgeAgentInput>::root();
    let (left, right) = root.split(|input| (input.clone(), input));
    let left = left.agent(edge_agent_with_echo_flow);
    let right = right.agent(edge_agent_with_echo_flow);
    let flow = left
        .merge(right, |(left, right)| EdgeAgentOutput {
            text: format!("{} {}", left.text, right.text),
        })
        .finish::<EdgeAgentInput>();

    assert!(flow.is_ok(), "repeated agent tool flows should compile");
}

/// Verifies reused controlled agents receive independent controller identities.
#[test]
fn typed_flow_reuses_controlled_agent_with_namespaced_handlers() {
    let root = Flow::<EdgeAgentInput>::root();
    let (left, right) = root.split(|input| (input.clone(), input));
    let left = left.agent(controlled_agent_with_echo_flow);
    let right = right.agent(controlled_agent_with_echo_flow);
    let flow = left
        .merge(right, |(left, right)| EdgeAgentOutput {
            text: format!("{} {}", left.text, right.text),
        })
        .finish::<EdgeAgentInput>();

    assert!(flow.is_ok(), "reused controlled agents should compile");
}

#[derive(Debug)]
struct EdgeHistoryRecordError;

impl std::fmt::Display for EdgeHistoryRecordError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("record failed")
    }
}

impl std::error::Error for EdgeHistoryRecordError {}

#[derive(Clone)]
struct FailAtHistoryRecord {
    calls: Arc<std::sync::atomic::AtomicUsize>,
    fail_at: usize,
}

impl crate::history::HistoryStore for FailAtHistoryRecord {
    type Error = EdgeHistoryRecordError;

    async fn record(&self, _entry: &crate::history::HistoryEntry) -> Result<(), Self::Error> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == self.fail_at {
            Err(EdgeHistoryRecordError)
        } else {
            Ok(())
        }
    }
}

/// Verifies a failed multi-message store batch leaves runtime history unchanged.
#[tokio::test]
async fn agent_history_batch_failure_does_not_commit_a_prefix() -> Result<(), crate::GraphError> {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new().then_tool_calls(vec![edge_tool_call(
        "unknown-1",
        "unknown_tool",
        serde_json::json!({}),
    )]);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let store = FailAtHistoryRecord {
        calls: Arc::clone(&calls),
        fail_at: 2,
    };
    let ctx = ctx().with_providers(crate::testing::providers(factory)?);
    let executor = FetchExecutor::new(ctx.clone()).with_registry(Arc::new(flow.registry().clone()));
    let mut runtime = flow
        .start(EdgeAgentInput { text: "hi".into() }, uuid::Uuid::nil())
        .expect("runtime should build");
    let executor = executor.with_store(store);
    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );
    assert_eq!(
        host::step(&mut runtime, &executor).await.unwrap(),
        Step::Continue
    );

    let err = host::step(&mut runtime, &executor)
        .await
        .expect_err("second message in tool-call batch should fail");
    assert!(matches!(err, GraphError::HistoryPersistence(_)));
    let snapshot = runtime
        .snapshot()
        .expect("snapshot should remain available");
    assert_eq!(
        snapshot.history().entries().len(),
        1,
        "assistant tool-call prefix must not enter runtime history"
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 3);
    Ok(())
}
