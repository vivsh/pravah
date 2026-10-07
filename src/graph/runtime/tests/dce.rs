use serde_json::json;

use super::*;
use crate::graph::{TypeSpec, UntypedGraphBuilder};

/// Verifies dead infallible shaping nodes are omitted in ascending ID order.
#[test]
fn eliminates_only_dead_infallible_nodes() {
    let graph = graph_with_dead_shaping_nodes();
    let plan = prepare_dce(&graph).expect("DCE should prepare");
    assert_eq!(plan.instructions.as_ref(), &[NodeId(3)]);
}

/// Verifies potentially failing unpack nodes remain prepared when unused.
#[test]
fn preserves_dead_unpack_tuple() {
    let mut builder = UntypedGraphBuilder::new("preserve_unpack");
    let array = TypeSpec::new("Array", json!({"type": "array"}));
    let number = TypeSpec::new("Number", json!({"type": "number"}));
    let input = builder.edge("input", array.clone());
    let unpacked = builder.edge("unpacked", number);
    let output = builder.edge("output", array);
    builder.set_entry(input).set_exit(output);
    add_builtin(
        &mut builder,
        "unpack",
        BuiltinNode::UnpackTuple,
        input,
        unpacked,
    );
    add_builtin(&mut builder, "live", BuiltinNode::Identity, input, output);
    let graph = builder.build().expect("graph should build");

    let plan = prepare_dce(&graph).expect("DCE should prepare");
    assert_eq!(plan.instructions.as_ref(), &[NodeId(0), NodeId(1)]);
}

/// Verifies execution skips eliminated nodes while the authored graph remains intact.
#[tokio::test]
async fn runtime_executes_only_surviving_instructions() {
    let graph = graph_with_dead_shaping_nodes();
    let authored_nodes = graph.nodes.len();
    let prepared = PreparedGraph::new(graph, HandlerRegistry::new()).expect("graph should prepare");
    let mut runtime = prepared
        .start(Value::from(7_i64), uuid::Uuid::nil())
        .expect("runtime should start");

    assert!(
        matches!(runtime.next().expect("runtime step"), Step::Done(value) if value == Value::from(7_i64))
    );
    assert_eq!(prepared.graph().nodes.len(), authored_nodes);
}

/// Builds a graph containing a dead fan-out and tuple chain beside its exit.
fn graph_with_dead_shaping_nodes() -> UntypedGraph {
    let mut builder = UntypedGraphBuilder::new("dead_shaping");
    let number = TypeSpec::new("Number", json!({"type": "number"}));
    let array = TypeSpec::new("Array", json!({"type": "array"}));
    let input = builder.edge("input", number.clone());
    let left = builder.edge("left", number.clone());
    let right = builder.edge("right", number.clone());
    let packed = builder.edge("packed", array);
    let copied = builder.edge("copied", number.clone());
    let output = builder.edge("output", number);
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
    add_builtin(
        &mut builder,
        "dead_copy",
        BuiltinNode::Identity,
        input,
        copied,
    );
    add_builtin(
        &mut builder,
        "live_copy",
        BuiltinNode::Identity,
        input,
        output,
    );
    builder.build().expect("graph should build")
}

fn add_builtin(
    builder: &mut UntypedGraphBuilder,
    name: &str,
    op: BuiltinNode,
    input: EdgeId,
    output: EdgeId,
) {
    builder.node(name, NodeKind::Builtin { op }, vec![input], vec![output]);
}
