use ::serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use either::Either;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde_json::Value as JsonValue;

use crate::clients::{
    Client, ClientError, ClientFactory, ClientOptions, ClientOutput, ClientResponse, Message,
    ModelUrl, Provider, Role, ToolCall,
};
use crate::tools::ToolOutput;
use crate::{Context, FlowConf, deps::Deps};

use super::state::ReturnTarget;
use super::*;

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
    PreparedGraph::new(graph, registry)?.start(input)
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

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Done(rv!(6)));
}

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
        runtime.next(ctx()).await.unwrap(),
        Step::Suspend(rv!({"need": "number"}))
    );
    assert_eq!(runtime.resume(rv!(2), ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.state().frames[0].values[input.0], Some(rv!(2)));
    assert_eq!(
        runtime.next(ctx()).await.unwrap(),
        Step::Suspend(rv!({"need": "number"}))
    );
}

#[tokio::test]
async fn typed_mark_goto_is_string_free_and_builder_checked() {
    let root = Flow::<i64>::root();
    let start = root.mark();
    let _loop_edge = root.clone().suspend::<i64>().goto(start);
    let exit = root.map(|value| value);
    let flow = exit.finish::<i64>().expect("typed mark/goto graph builds");
    let mut runtime = flow.runtime(1).expect("runtime should build");

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Suspend(rv!(1)));
    assert_eq!(runtime.resume(rv!(3), ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Suspend(rv!(3)));
}

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

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.state().frames.len(), 2);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(
        runtime.state().frames.len(),
        1,
        "child exit should cascade to parent"
    );
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Done(rv!(10)));
}

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

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Done(rv!([5, 10])));
}

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

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Done(rv!(15)));
    let root = runtime.state().frames.first();
    assert!(root.is_none(), "done pops the root frame");
}

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
        .next(ctx())
        .await
        .expect_err("passthrough output schema should fail");
    match err {
        GraphError::Schema { expected, .. } => assert_eq!(expected, "Text"),
        other => panic!("expected schema error, got {other:?}"),
    }
    let frame = runtime.state().frames.first().expect("frame should remain");
    assert_eq!(frame.variables[state.0], Some(rv!(10)));
    assert!(frame.values[output.0].is_none());
}

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
        runtime.next(ctx()).await.unwrap(),
        Step::Suspend(rv!({"need": "resume"}))
    );
    assert_eq!(runtime.state().frames.len(), 1);
    assert!(
        runtime.next(ctx()).await.is_err(),
        "next while suspended must fail"
    );
    assert_eq!(
        runtime.resume(rv!(5), ctx()).await.unwrap(),
        Step::Done(rv!(5))
    );
}

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
    let mut runtime = prepared.start(rv!(7)).expect("runtime should build");
    assert!(matches!(
        runtime.next(ctx()).await.unwrap(),
        Step::Suspend(_)
    ));

    let mut snapshot = runtime.snapshot().expect("snapshot should build");
    snapshot
        .state
        .suspension
        .as_mut()
        .expect("snapshot should be suspended")
        .frame_depth = 999;
    let err = match prepared.restore(snapshot) {
        Ok(_) => panic!("bad suspension graph should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("frame depth"));
}

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
    let mut runtime = prepared.start(rv!(5)).expect("runtime should build");
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);

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
    let mut runtime = prepared.start(rv!(3)).expect("runtime should build");
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);

    let snapshot = runtime.snapshot().expect("snapshot should build");
    let mut restored = prepared.restore(snapshot).expect("snapshot should restore");

    assert_eq!(restored.next(ctx()).await.unwrap(), Step::Done(rv!(8)));
}

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

    let err = runtime
        .next(ctx())
        .await
        .expect_err("schema mismatch should fail");
    assert!(matches!(err, GraphError::Schema { .. }));
}

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
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        let transition = self.transition.clone();
        Box::pin(async move { Ok(transition) })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        checkpoint: Value,
        _event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            Ok(ContinuationTransition {
                checkpoint: None,
                state: None,
                outputs: vec![checkpoint],
                writes: Vec::new(),
                child_calls: Vec::new(),
            })
        })
    }
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
    runtime.next(ctx()).await
}

struct AssertNoServiceSmuggling;

impl ContinuationHandler for AssertNoServiceSmuggling {
    fn start<'a>(
        &'a self,
        _payload: &'a Value,
        _state: Option<Value>,
        _inputs: Vec<Value>,
        ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            let _provider = ctx.context().client_factory();
            let service_was_smuggled = ctx.context().deps().get::<RuntimeServices>().is_some();
            Ok(ContinuationTransition {
                checkpoint: None,
                state: None,
                outputs: vec![rv!(!service_was_smuggled)],
                writes: Vec::new(),
                child_calls: Vec::new(),
            })
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _continuation: Value,
        _event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move { Err(GraphError::Invalid("unexpected resume".into())) })
    }
}

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

    let step = runtime.next(ctx()).await.expect("continuation should run");
    assert_eq!(step, Step::Done(rv!(true)));
}

#[tokio::test]
async fn continuation_rejects_outputs_with_checkpoint() {
    let err = run_static_continuation_transition(ContinuationTransition {
        checkpoint: Some(rv!({"state": true})),
        state: None,
        outputs: vec![rv!(1)],
        writes: Vec::new(),
        child_calls: Vec::new(),
    })
    .await
    .expect_err("outputs plus checkpoint should fail");
    assert!(matches!(
        err,
        GraphError::InvalidContinuationTransition { .. }
    ));
}

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
                    checkpoint: None,
                    state: None,
                    outputs: vec![rv!(2)],
                    writes: vec![EdgeWrite {
                        edge: output,
                        value: rv!(1),
                    }],
                    child_calls: Vec::new(),
                },
            },
        )
        .expect("handler should insert");
    let mut runtime = test_runtime(graph, rv!(0), registry).expect("runtime should build");

    let err = runtime
        .next(ctx())
        .await
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
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            Ok(ContinuationTransition {
                checkpoint: inputs.into_iter().next(),
                state: None,
                outputs: Vec::new(),
                writes: Vec::new(),
                child_calls: Vec::new(),
            })
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        checkpoint: Value,
        event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            let ContinuationEvent::Poll = event else {
                return Err(GraphError::Invalid("expected poll event".into()));
            };
            Ok(ContinuationTransition {
                checkpoint: None,
                state: None,
                outputs: vec![rv!(checkpoint.as_i64().unwrap_or_default() + 1)],
                writes: Vec::new(),
                child_calls: Vec::new(),
            })
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
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            Ok(ContinuationTransition {
                checkpoint: Some(rv!({"started": true})),
                state: None,
                outputs: Vec::new(),
                writes: Vec::new(),
                child_calls: vec![ContinuationChildCall {
                    child_index: 0,
                    call_id: "child-1".into(),
                    input: inputs.into_iter().next().unwrap_or(Value::NULL),
                }],
            })
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _continuation: Value,
        event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            match event {
                ContinuationEvent::ChildResult { .. } => {
                    Err(GraphError::Invalid("child result handling failed".into()))
                }
                other => Err(GraphError::Invalid(format!("unexpected event {other:?}"))),
            }
        })
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
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            Ok(ContinuationTransition {
                checkpoint: Some(rv!({"started": true})),
                state: Some(rv!({"mutated": true})),
                outputs: Vec::new(),
                writes: Vec::new(),
                child_calls: vec![ContinuationChildCall {
                    child_index: 0,
                    call_id: "child-1".into(),
                    input: rv!("not a number"),
                }],
            })
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _continuation: Value,
        _event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move { Err(GraphError::Invalid("unexpected resume".into())) })
    }
}

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

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.state().frames.len(), 2);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.state().frames.len(), 1);
    let frame = runtime
        .state()
        .frames
        .first()
        .expect("parent frame remains");
    assert!(frame.checkpoints[0].is_some());
    assert_eq!(frame.continuation_inboxes[0].len(), 1);

    let err = runtime
        .next(ctx())
        .await
        .expect_err("child-result poll should fail");
    assert!(err.to_string().contains("child result handling failed"));
    let frame = runtime
        .state()
        .frames
        .first()
        .expect("parent frame remains");
    assert!(frame.checkpoints[0].is_some());
    assert_eq!(frame.continuation_inboxes[0].len(), 1);
}

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
        .next(ctx())
        .await
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

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Done(rv!(5)));
}

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
        .insert_work("dup", |_inputs, _ctx| {
            let fut: BoxFuture<'static, Result<Vec<Value>, GraphError>> =
                Box::pin(async { Ok(Vec::new()) });
            fut
        })
        .expect("first work handler should insert");
    assert!(
        registry
            .insert_work("dup", |_inputs, _ctx| {
                let fut: BoxFuture<'static, Result<Vec<Value>, GraphError>> =
                    Box::pin(async { Ok(Vec::new()) });
                fut
            })
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

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Done(rv!(15)));
}

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

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Done(rv!(12)));
}

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

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Done(rv!(15)));
}

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
        match runtime.next(ctx()).await.unwrap() {
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

#[tokio::test]
async fn typed_variable_handles_drive_load_and_store() {
    let root = Flow::<TypedAmount>::root();
    let bonus = root.local(TypedBonus { value: 3 });
    let flow = root
        .load(bonus.clone(), |mut amount, bonus| {
            amount.value += bonus.value;
            amount
        })
        .store(bonus, |amount, _bonus| TypedBonus {
            value: amount.value,
        })
        .finish::<TypedAmount>()
        .expect("flow should compile");
    let mut runtime = flow
        .runtime(TypedAmount { value: 4 })
        .expect("runtime should build");
    let done = loop {
        match runtime.next(ctx()).await.unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    assert_eq!(output.value, 7);
}

#[tokio::test]
async fn typed_load_can_change_output_type() {
    let root = Flow::<TypedAmount>::root();
    let bonus = root.local(TypedBonus { value: 5 });
    let flow = root
        .load(bonus, |amount, bonus| LeftAmount {
            value: amount.value + bonus.value,
        })
        .finish::<TypedAmount>()
        .expect("flow should compile");
    let mut runtime = flow
        .runtime(TypedAmount { value: 4 })
        .expect("runtime should build");
    let done = loop {
        match runtime.next(ctx()).await.unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    assert_eq!(output.value, 9);
}

#[test]
fn typed_variable_handle_from_other_builder_fails_at_finish() {
    let builder = TypedGraphBuilder::<TypedAmount>::new();
    let root = builder.root();
    let other = TypedGraphBuilder::<TypedAmount>::new();
    let foreign = other.local(TypedBonus { value: 1 });
    let output = builder.load(root, foreign, |amount: TypedAmount, _bonus: TypedBonus| {
        amount
    });
    let err = match builder.finish(output) {
        Ok(_) => panic!("cross-builder variable handle should fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("variable must belong"));
}

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
        .runtime(TypedAmount { value: 3 })
        .expect("runtime should build");

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    let done = runtime.next(ctx()).await.unwrap();
    let Step::Done(value) = done else {
        panic!("expected done, got {done:?}");
    };
    let output = flow
        .decode_output(value)
        .expect("typed output should decode");
    assert_eq!(output.value, 12);
}

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
        .runtime(TypedAmount { value: 2 })
        .expect("runtime should build");

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    let done = runtime.next(ctx()).await.unwrap();
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

#[tokio::test]
async fn typed_fluent_api_supports_either_branch() {
    let flow = compile(typed_choice).expect("flow should compile");

    let mut runtime = flow
        .runtime(TypedChoice { value: -7 })
        .expect("runtime should build");
    let done = loop {
        match runtime.next(ctx()).await.unwrap() {
            Step::Continue => {}
            Step::Done(value) => break value,
            step => panic!("expected done, got {step:?}"),
        }
    };
    let output = flow.decode_output(done).expect("output should decode");
    assert_eq!(output.value, 7);

    let mut runtime = flow
        .runtime(TypedChoice { value: 8 })
        .expect("runtime should build");
    let done = loop {
        match runtime.next(ctx()).await.unwrap() {
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

#[tokio::test]
async fn typed_fluent_api_supports_each() {
    let flow = compile(typed_batch).expect("flow should compile");
    let mut runtime = flow
        .runtime(TypedBatch {
            values: vec![TypedItem { value: 1 }, TypedItem { value: 2 }],
        })
        .expect("runtime should build");
    let done = loop {
        match runtime.next(ctx()).await.unwrap() {
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
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            let input = decode_test_value::<TypedAmount>(inputs, "payload_effect")?;
            let add = payload
                .get("add")
                .and_then(Value::as_i64)
                .ok_or_else(|| GraphError::Invalid("missing payload add".into()))?;
            Ok(ContinuationTransition {
                checkpoint: None,
                state: None,
                outputs: vec![rv!(TypedAmount {
                    value: input.value + add
                })],
                writes: Vec::new(),
                child_calls: Vec::new(),
            })
        })
    }

    fn advance<'a>(
        &'a self,
        _payload: &'a Value,
        _continuation: Value,
        _event: ContinuationEvent,
        _ctx: ContinuationContext,
    ) -> BoxFuture<'a, Result<ContinuationTransition, GraphError>> {
        Box::pin(async move {
            Err(GraphError::Invalid(
                "payload continuation does not advance".into(),
            ))
        })
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

#[tokio::test]
async fn typed_builder_builds_map_work_and_continuation_without_fluent_api() {
    let builder = TypedGraphBuilder::<TypedAmount>::new();
    let root = builder.root();
    let mapped = builder.map(root, |input: TypedAmount| TypedAmount {
        value: input.value + 1,
    });
    let worked = builder.work(mapped, |input: TypedAmount, _ctx| async move {
        Ok(TypedAmount {
            value: input.value * 2,
        })
    });
    let continued = builder.continuation::<TypedAmount, TypedAmount, AddPayloadContinuation, _>(
        worked,
        rv!({"add": 5}),
    );
    let flow = builder
        .finish(continued)
        .expect("typed builder should finish");
    let mut runtime = flow
        .runtime(TypedAmount { value: 3 })
        .expect("runtime should build");

    let done = loop {
        match runtime.next(ctx()).await.unwrap() {
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
        .runtime(TypedAmount { value: 1 })
        .expect("runtime should build");

    let done = loop {
        match runtime.next(ctx()).await.unwrap() {
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
                ClientOutput::ToolCalls {
                    thought: None,
                    calls,
                },
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

#[async_trait]
impl Client for EdgeScriptedClient {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }

    fn options(&self) -> &ClientOptions {
        &self.options
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        inner.calls.push(messages.to_vec());
        inner.responses.pop_front().unwrap_or_else(|| {
            Err(ClientError::Provider(
                "edge scripted response queue exhausted".into(),
            ))
        })
    }
}

impl ClientFactory for EdgeScriptedFactory {
    fn create(
        &self,
        model_url: &str,
        options: ClientOptions,
    ) -> Result<Box<dyn Client>, ClientError> {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .creates
            .push(model_url.to_owned());
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .options
            .push(options.clone());
        Ok(Box::new(EdgeScriptedClient {
            inner: Arc::clone(&self.inner),
            url: ModelUrl::parse(model_url).or_else(|_| ModelUrl::parse("openai:///test-model"))?,
            options,
        }))
    }
}

fn edge_tool_call(id: &str, name: &str, args: JsonValue) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        args,
        thought_signatures: None,
    }
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
        "openai:///test-model",
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
        "openai:///test-model",
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
        "openai:///test-model",
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
        "gemini:///gemini-2.5-flash",
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

#[derive(Debug)]
struct EdgeHistoryRecordError;

impl std::fmt::Display for EdgeHistoryRecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("record failed")
    }
}

impl std::error::Error for EdgeHistoryRecordError {}

struct FailingEdgeHistoryStore;

impl crate::legacy::HistoryStore for FailingEdgeHistoryStore {
    type Error = EdgeHistoryRecordError;

    async fn record(&self, _entry: &crate::legacy::HistoryEntry) -> Result<(), Self::Error> {
        Err(EdgeHistoryRecordError)
    }
}

#[tokio::test]
async fn typed_edge_agent_without_tools_uses_structured_output() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");
    let ctx = ctx().with_client_factory(factory.clone());

    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    let done = match runtime.next(ctx).await.unwrap() {
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

/// Verifies activation is checkpointed once and memory remains outside conversation history.
#[tokio::test]
async fn agent_configuration_runs_once_across_snapshot_restore() {
    let flow = compile(counted_agent_flow).expect("counted agent should compile");
    let calls = Arc::new(ConfigureCalls(AtomicUsize::new(0)));
    let mut deps = Deps::default();
    deps.insert(Arc::clone(&calls));
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let ctx = ctx().with_deps(deps).with_client_factory(factory.clone());
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");

    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
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
        .prepared()
        .restore(snapshot)
        .expect("configured runtime should restore");
    assert!(matches!(restored.next(ctx).await.unwrap(), Step::Done(_)));
    assert_eq!(calls.0.load(Ordering::SeqCst), 1);
    let preamble = factory.options()[0]
        .preamble
        .clone()
        .expect("agent preamble should be set");
    assert!(preamble.contains("<memory>\nstable private memory\n</memory>"));
}

#[tokio::test]
async fn graph_chat_uses_one_runtime_across_turns() {
    let factory = EdgeScriptedFactory::new()
        .then_output(serde_json::json!({ "text": "first" }))
        .then_output(serde_json::json!({ "text": "second" }));
    let mut chat = Chat::<EdgeAgentInput, EdgeAgentOutput>::new(edge_chat_agent);
    let ctx = ctx().with_client_factory(factory.clone());

    let first = chat
        .send(EdgeAgentInput { text: "hi".into() }, ctx.clone())
        .await
        .expect("first chat turn should run");
    assert_eq!(
        first.output,
        EdgeAgentOutput {
            text: "first".into()
        }
    );

    let second = chat
        .send(
            EdgeAgentInput {
                text: "again".into(),
            },
            ctx,
        )
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
}

#[tokio::test]
async fn typed_edge_agent_provider_config_reaches_client_options() {
    let flow = Flow::<EdgeProviderConfigAgentInput>::root()
        .agent(provider_agent)
        .finish::<EdgeProviderConfigAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let mut runtime = flow
        .runtime(EdgeProviderConfigAgentInput { text: "hi".into() })
        .expect("runtime should build");
    let ctx = ctx().with_client_factory(factory.clone());

    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    assert!(matches!(runtime.next(ctx).await.unwrap(), Step::Done(_)));

    let options = factory.options();
    let provider_config = options
        .first()
        .and_then(|opts| opts.provider_config.as_ref())
        .expect("provider config should reach graph agent client options");
    assert_eq!(
        provider_config["safety_settings"][0]["category"],
        "HARM_CATEGORY_DANGEROUS_CONTENT"
    );
}

#[tokio::test]
async fn graph_runtime_inject_message_appends_at_dispatch_boundary() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    runtime
        .inject_message("extra context")
        .await
        .expect("inject should succeed at dispatch boundary");

    let snapshot = runtime.snapshot().expect("snapshot should build");
    assert!(
        snapshot
            .history()
            .entries()
            .iter()
            .any(|entry| entry.message.content == "extra context")
    );
}

#[tokio::test]
async fn graph_runtime_inject_message_store_failure_does_not_commit() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");

    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    runtime = runtime.with_store(FailingEdgeHistoryStore);
    let err = runtime
        .inject_message("extra context")
        .await
        .expect_err("injected message record should fail");
    assert!(matches!(err, GraphError::HistoryPersistence(_)));
    let snapshot = runtime.snapshot().expect("snapshot should build");
    assert!(
        snapshot
            .history()
            .entries()
            .iter()
            .all(|entry| entry.message.content != "extra context"),
        "failed injected record must not commit history"
    );
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct EchoIn {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct EchoOut {
    text: String,
}

impl ToolOutput for EchoOut {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SuffixIn {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SuffixOut {
    text: String,
}

impl ToolOutput for SuffixOut {}

fn echo_tools(tools: Toolset) -> Toolset {
    tools.tool_handler(|input: EchoIn, _ctx| async move {
        Ok(EchoOut {
            text: input.text.to_uppercase(),
        })
    })
}

fn two_tools(tools: Toolset) -> Toolset {
    echo_tools(tools).tool_handler(|input: SuffixIn, _ctx| async move {
        Ok(SuffixOut {
            text: format!("{}!", input.text),
        })
    })
}

fn duplicate_tools(tools: Toolset) -> Toolset {
    tools
        .tool_handler(|input: EchoIn, _ctx| async move { Ok(EchoOut { text: input.text }) })
        .tool_handler(|input: EchoIn, _ctx| async move { Ok(EchoOut { text: input.text }) })
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

async fn configure_filtered_agent(
    input: EdgeAgentInput,
    _ctx: Context,
) -> Result<AgentConfig, GraphError> {
    let selected = input.text;
    Ok(AgentConfig::new(
        "openai:///test-model",
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

#[tokio::test]
async fn typed_edge_agent_tool_handler_round_trips_through_same_vm() {
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
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");
    let ctx = ctx().with_client_factory(factory.clone());

    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    let done = match runtime.next(ctx).await.unwrap() {
        Step::Done(value) => flow.decode_output(value).unwrap(),
        other => panic!("expected done, got {other:?}"),
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
}

/// Verifies a capturing runtime filter can expose only statically prepared tools.
#[tokio::test]
async fn typed_edge_agent_filters_tools_in_prepared_order() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(filtered_agent)
        .finish::<EdgeAgentInput>()
        .expect("filtered agent should compile");
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let mut runtime = flow
        .runtime(EdgeAgentInput {
            text: "suffix_in".into(),
        })
        .expect("runtime should build");
    let ctx = ctx().with_client_factory(factory.clone());

    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    assert!(matches!(runtime.next(ctx).await.unwrap(), Step::Done(_)));
    let options = factory.options();
    let names = options[0]
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["suffix_in"]);
}

/// Verifies configuration failure leaves the runtime snapshot retryable and history empty.
#[tokio::test]
async fn typed_edge_agent_configuration_failure_does_not_mutate_runtime() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(failing_agent)
        .finish::<EdgeAgentInput>()
        .expect("failing agent definition should compile");
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");
    let before = serde_json::to_value(runtime.snapshot().unwrap()).unwrap();

    let err = runtime
        .next(ctx())
        .await
        .expect_err("configuration should fail");
    let after = serde_json::to_value(runtime.snapshot().unwrap()).unwrap();

    assert!(matches!(err, GraphError::AgentConfiguration { .. }));
    assert_eq!(before, after);
    assert!(runtime.snapshot().unwrap().history().entries().is_empty());
}

#[tokio::test]
async fn typed_edge_agent_provider_is_resolved_at_dispatch() {
    let flow = Flow::<EdgeAgentInput>::root()
        .agent(edge_agent)
        .finish::<EdgeAgentInput>()
        .expect("agent flow should compile");
    let factory = EdgeScriptedFactory::new().then_output(serde_json::json!({ "text": "done" }));
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");
    let ctx = ctx().with_client_factory(factory.clone());
    assert!(factory.creates().is_empty());

    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    assert!(factory.creates().is_empty());

    let done = match runtime.next(ctx).await.unwrap() {
        Step::Done(value) => flow.decode_output(value).unwrap(),
        other => panic!("expected done, got {other:?}"),
    };

    assert_eq!(
        done,
        EdgeAgentOutput {
            text: "done".into()
        }
    );
    assert_eq!(factory.creates(), vec!["openai:///test-model".to_string()]);
}

#[tokio::test]
async fn typed_edge_agent_multiple_tool_calls_are_queued_on_single_vm_stack() {
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
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");
    let ctx = ctx().with_client_factory(factory.clone());

    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);
    assert_eq!(runtime.state().frames.len(), 2);

    let done = loop {
        match runtime.next(ctx.clone()).await.unwrap() {
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
    assert!(tool_messages[0].1.contains("HI"));
    assert_eq!(tool_messages[1].0, "c2");
    assert!(tool_messages[1].1.contains("bye!"));
}

#[tokio::test]
async fn typed_edge_agent_same_tool_calls_run_in_deterministic_queue_order() {
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
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");
    let ctx = ctx().with_client_factory(factory.clone());

    let done = loop {
        match runtime.next(ctx.clone()).await.unwrap() {
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
}

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
    let mut runtime = flow
        .runtime(RepeatedFlowInput(1))
        .expect("runtime should build");

    loop {
        match runtime.next(ctx()).await.expect("step should succeed") {
            Step::Continue => {}
            Step::Done(value) => {
                assert_eq!(flow.decode_output(value).unwrap(), 3);
                break;
            }
            Step::Suspend(_) => panic!("repeated subflow should not suspend"),
        }
    }
}

/// Verifies explicit typed node names reach the canonical graph and diagrams.
#[test]
fn typed_named_nodes_preserve_supplied_names() {
    let flow = Flow::<i64>::root()
        .map_named("meaningful_map", |value| value + 1)
        .work_named(
            "meaningful_work",
            |value, _ctx| async move { Ok(value * 2) },
        )
        .finish::<i64>()
        .expect("named flow should compile");
    let names = flow
        .graph()
        .nodes
        .iter()
        .map(|node| node.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(names, vec!["meaningful_map", "meaningful_work"]);
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
    let mut runtime = prepared.start(rv!(1)).expect("runtime should build");
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
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
    encoded["version"] = serde_json::json!(1);
    *payload = to_value(encoded).expect("payload should enter runtime domain");
    let err = match test_runtime(graph, rv!({"text": "hello"}), registry) {
        Ok(_) => panic!("obsolete payload should fail preparation"),
        Err(err) => err,
    };

    assert!(matches!(err, GraphError::GraphValidation(_)));
    assert!(
        err.to_string()
            .contains("unsupported agent payload version 1")
    );
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
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build");
    assert_eq!(runtime.next(ctx()).await.unwrap(), Step::Continue);
    let mut snapshot = runtime.snapshot().expect("snapshot should build");
    let frame = snapshot.state.frame_mut(0).expect("root frame");
    let checkpoint = &mut Arc::make_mut(&mut frame.checkpoints)[0].value;
    let mut encoded = serde_json::to_value(&*checkpoint).expect("checkpoint should encode");
    encoded["version"] = serde_json::json!(1);
    *checkpoint = to_value(encoded).expect("checkpoint should enter runtime domain");

    assert!(matches!(
        flow.prepared().restore(snapshot),
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

#[derive(Clone)]
struct FailAtHistoryRecord {
    calls: Arc<std::sync::atomic::AtomicUsize>,
    fail_at: usize,
}

impl crate::legacy::HistoryStore for FailAtHistoryRecord {
    type Error = EdgeHistoryRecordError;

    async fn record(&self, _entry: &crate::legacy::HistoryEntry) -> Result<(), Self::Error> {
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
async fn agent_history_batch_failure_does_not_commit_a_prefix() {
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
    let mut runtime = flow
        .runtime(EdgeAgentInput { text: "hi".into() })
        .expect("runtime should build")
        .with_store(store);
    let ctx = ctx().with_client_factory(factory);
    assert_eq!(runtime.next(ctx.clone()).await.unwrap(), Step::Continue);

    let err = runtime
        .next(ctx)
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
}
