use std::collections::VecDeque;
use std::sync::Arc;

use super::*;
use crate::graph::model::Node;

/// Deterministic set of authored nodes retained as prepared instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DcePlan {
    active: Arc<[bool]>,
    pub(super) instructions: Arc<[NodeId]>,
}

impl DcePlan {
    pub(super) fn is_active(&self, node: NodeId) -> bool {
        self.active.get(node.0).copied().unwrap_or(false)
    }
}

/// Retains effectful nodes and traces their data dependencies backwards.
pub(super) fn prepare_dce(graph: &UntypedGraph) -> Result<DcePlan, GraphError> {
    let mut active = vec![false; graph.nodes.len()];
    let mut queue = VecDeque::new();
    for node in &graph.nodes {
        if !is_removable(node) {
            mark_node(node.id, &mut active, &mut queue)?;
        }
    }
    if let Some(producer) = graph.edge(graph.exit).and_then(|edge| edge.producer) {
        mark_node(producer, &mut active, &mut queue)?;
    }
    trace_dependencies(graph, &mut active, &mut queue)?;
    let instructions = active
        .iter()
        .enumerate()
        .filter(|(_, live)| **live)
        .map(|(index, _)| NodeId(index))
        .collect::<Vec<_>>();
    Ok(DcePlan {
        active: active.into(),
        instructions: instructions.into(),
    })
}

fn trace_dependencies(
    graph: &UntypedGraph,
    active: &mut [bool],
    queue: &mut VecDeque<NodeId>,
) -> Result<(), GraphError> {
    while let Some(node_id) = queue.pop_front() {
        let node = graph
            .node(node_id)
            .ok_or(GraphError::MissingNode(node_id))?;
        for edge_id in node.inputs.iter().copied() {
            let edge = graph
                .edge(edge_id)
                .ok_or(GraphError::MissingEdge(edge_id))?;
            if let Some(producer) = edge.producer {
                mark_node(producer, active, queue)?;
            }
        }
    }
    Ok(())
}

fn mark_node(
    node: NodeId,
    active: &mut [bool],
    queue: &mut VecDeque<NodeId>,
) -> Result<(), GraphError> {
    let slot = active
        .get_mut(node.0)
        .ok_or(GraphError::MissingNode(node))?;
    if !*slot {
        *slot = true;
        queue.push_back(node);
    }
    Ok(())
}

fn is_removable(node: &Node) -> bool {
    matches!(
        node.kind,
        NodeKind::Builtin {
            op: BuiltinNode::Identity | BuiltinNode::FanOut | BuiltinNode::PackTuple
        }
    )
}

#[cfg(test)]
mod tests {
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
        let prepared =
            PreparedGraph::new(graph, HandlerRegistry::new()).expect("graph should prepare");
        let mut runtime = prepared
            .start(Value::from(7_i64))
            .expect("runtime should start");

        assert_eq!(
            runtime
                .next(Context::default())
                .await
                .expect("runtime step"),
            Step::Done(Value::from(7_i64))
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
}
