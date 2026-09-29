use serde_json::json;

use super::*;
use crate::graph::{NodeKind, TypeSpec, UntypedGraphBuilder};

/// Verifies a value with two readers is cleared only after the second commits.
#[tokio::test]
async fn multiple_readers_release_after_last_successful_reader() {
    let prepared = prepare_two_reader_graph();
    let mut runtime = prepared
        .start(Value::from(7_i64), uuid::Uuid::nil())
        .expect("runtime should start");

    runtime.next().expect("first reader");
    assert!(runtime.state.frames[0].values[0].is_some());

    runtime.next().expect("second reader");
    assert!(runtime.state.frames[0].values[0].is_none());
}

/// Verifies restored reader counters preserve completed-reader progress.
#[tokio::test]
async fn restore_rebuilds_reader_counters() {
    let prepared = prepare_two_reader_graph();
    let mut runtime = prepared
        .start(Value::from(7_i64), uuid::Uuid::nil())
        .expect("runtime should start");
    runtime.next().expect("first reader");
    let snapshot = runtime.snapshot().expect("snapshot should build");
    let mut restored = prepared.restore(snapshot).expect("snapshot should restore");

    restored.next().expect("second reader");
    assert!(restored.state.frames[0].values[0].is_none());
}

/// Verifies a failing instruction leaves its input available for retry.
#[tokio::test]
async fn failed_instruction_does_not_release_input() {
    let prepared = prepare_failing_unpack_graph();
    let mut runtime = prepared
        .start(Value::array([Value::from(1_i64)]), uuid::Uuid::nil())
        .expect("runtime should start");

    runtime.next().expect_err("unpack should fail");
    assert!(runtime.state.frames[0].values[0].is_some());
    assert_eq!(runtime.state.frames[0].node_epochs[0], 0);
}

/// Verifies epoch overflow fails without committing output or releasing input.
#[tokio::test]
async fn write_epoch_overflow_preserves_retry_state() {
    let prepared = prepare_two_reader_graph();
    let mut runtime = prepared
        .start(Value::from(7_i64), uuid::Uuid::nil())
        .expect("runtime should start");
    runtime.state.frames[0].write_epoch = u64::MAX;

    let error = runtime
        .next()
        .expect_err("write should reject epoch overflow");
    assert!(error.to_string().contains("epoch overflowed"));
    assert!(runtime.state.frames[0].values[0].is_some());
    assert!(runtime.state.frames[0].values[1].is_none());
    assert_eq!(runtime.state.frames[0].node_epochs[0], 0);
}

/// Builds an acyclic graph whose entry has two independent readers.
fn prepare_two_reader_graph() -> PreparedGraph {
    let mut builder = UntypedGraphBuilder::new("two_readers");
    let number = TypeSpec::new("Number", json!({"type": "number"}));
    let pair = TypeSpec::new("Pair", json!({"type": "array"}));
    let input = builder.edge("input", number.clone());
    let left = builder.edge("left", number.clone());
    let right = builder.edge("right", number);
    let output = builder.edge("output", pair);
    builder.set_entry(input).set_exit(output);
    add_builtin(
        &mut builder,
        "left",
        BuiltinNode::Identity,
        vec![input],
        vec![left],
    );
    add_builtin(
        &mut builder,
        "right",
        BuiltinNode::Identity,
        vec![input],
        vec![right],
    );
    add_builtin(
        &mut builder,
        "join",
        BuiltinNode::PackTuple,
        vec![left, right],
        vec![output],
    );
    let graph = builder.build().expect("graph should build");
    PreparedGraph::new(graph, HandlerRegistry::new()).expect("graph should prepare")
}

/// Builds a graph whose unpack instruction fails for a one-item input.
fn prepare_failing_unpack_graph() -> PreparedGraph {
    let mut builder = UntypedGraphBuilder::new("failing_unpack");
    let array = TypeSpec::new("Array", json!({"type": "array"}));
    let number = TypeSpec::new("Number", json!({"type": "number"}));
    let input = builder.edge("input", array);
    let left = builder.edge("left", number.clone());
    let output = builder.edge("output", number);
    builder.set_entry(input).set_exit(output);
    add_builtin(
        &mut builder,
        "unpack",
        BuiltinNode::UnpackTuple,
        vec![input],
        vec![left, output],
    );
    let graph = builder.build().expect("graph should build");
    PreparedGraph::new(graph, HandlerRegistry::new()).expect("graph should prepare")
}

fn add_builtin(
    builder: &mut UntypedGraphBuilder,
    name: &str,
    op: BuiltinNode,
    inputs: Vec<EdgeId>,
    outputs: Vec<EdgeId>,
) {
    builder.node(name, NodeKind::Builtin { op }, inputs, outputs);
}
