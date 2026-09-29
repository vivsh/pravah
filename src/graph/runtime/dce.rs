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
#[path = "tests/dce.rs"]
mod tests;
