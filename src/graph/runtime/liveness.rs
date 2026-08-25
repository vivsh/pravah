use std::collections::VecDeque;
use std::sync::Arc;

use super::*;
use crate::graph::model::Node;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Retention {
    Dead,
    OneReader,
    ManyReaders(usize),
    Pinned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LiveValue {
    Edge(EdgeId),
    Variable(VarId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ReaderCounterPlan {
    pub(super) value: LiveValue,
    pub(super) readers: Arc<[NodeId]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReleaseAction {
    ReadEdge {
        edge: EdgeId,
        counter: Option<usize>,
    },
    ReadVariable {
        variable: VarId,
        counter: Option<usize>,
    },
    ClearEdge(EdgeId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LivenessPlan {
    pub(super) edges: Arc<[Retention]>,
    pub(super) variables: Arc<[Retention]>,
    pub(super) counters: Arc<[ReaderCounterPlan]>,
}

impl LivenessPlan {
    pub(super) fn initial_counters(&self) -> Result<Vec<u32>, GraphError> {
        self.counters
            .iter()
            .map(|counter| {
                u32::try_from(counter.readers.len()).map_err(|_| {
                    GraphError::Invalid("reader counter exceeds the supported range".into())
                })
            })
            .collect()
    }
}

pub(super) fn prepare_liveness(
    graph: &UntypedGraph,
    dce: &DcePlan,
) -> Result<LivenessPlan, GraphError> {
    let readers = edge_readers(graph, dce);
    let adjacency = node_adjacency(graph, &readers, dce)?;
    let cyclic = cyclic_nodes(&adjacency);
    let (dynamic_nodes, pinned_edges) = dynamic_values(graph, &readers, &cyclic, dce)?;
    let pin_frame = graph
        .nodes
        .iter()
        .any(|node| dce.is_active(node.id) && matches!(node.kind, NodeKind::Continuation { .. }));
    let mut edges = classify_edges(graph, &readers, &pinned_edges, pin_frame);
    let variable_readers = variable_readers(graph, dce);
    let child_captures = child_variable_captures(graph, dce);
    let mut variables = classify_variables(
        graph,
        &variable_readers,
        &cyclic,
        &dynamic_nodes,
        &child_captures,
        pin_frame,
    );
    let counters = assign_counters(&mut edges, &mut variables, &readers, &variable_readers)?;
    Ok(LivenessPlan {
        edges: Arc::from(edges),
        variables: Arc::from(variables),
        counters: Arc::from(counters),
    })
}

pub(super) fn release_actions(
    node: &Node,
    plan: &LivenessPlan,
) -> Result<Arc<[ReleaseAction]>, GraphError> {
    let mut actions = Vec::new();
    let mut inputs = node.inputs.clone();
    inputs.sort_unstable_by_key(|edge| edge.0);
    inputs.dedup();
    for edge in inputs {
        if let Some(action) = edge_read_action(edge, plan)? {
            actions.push(action);
        }
    }
    if let NodeKind::Load { var, .. } | NodeKind::Store { var, .. } = node.kind
        && let Some(action) = variable_read_action(var, plan)?
    {
        actions.push(action);
    }
    for edge in &node.outputs {
        if matches!(plan.edges.get(edge.0), Some(Retention::Dead)) {
            actions.push(ReleaseAction::ClearEdge(*edge));
        }
    }
    actions.sort_by_key(release_order);
    Ok(Arc::from(actions))
}

fn edge_readers(graph: &UntypedGraph, dce: &DcePlan) -> Vec<Vec<NodeId>> {
    graph
        .edges
        .iter()
        .map(|edge| {
            let mut readers = edge.consumers.clone();
            readers.retain(|node| dce.is_active(*node));
            readers.sort_unstable_by_key(|node| node.0);
            readers.dedup();
            readers
        })
        .collect()
}

fn variable_readers(graph: &UntypedGraph, dce: &DcePlan) -> Vec<Vec<NodeId>> {
    let mut readers = vec![Vec::new(); graph.variables.len()];
    for node in &graph.nodes {
        if !dce.is_active(node.id) {
            continue;
        }
        let (NodeKind::Load { var, .. } | NodeKind::Store { var, .. }) = node.kind else {
            continue;
        };
        if let Some(found) = readers.get_mut(var.0) {
            found.push(node.id);
        }
    }
    readers
}

fn node_adjacency(
    graph: &UntypedGraph,
    readers: &[Vec<NodeId>],
    dce: &DcePlan,
) -> Result<Vec<Vec<usize>>, GraphError> {
    let mut adjacency = vec![Vec::new(); graph.nodes.len()];
    for edge in &graph.edges {
        let Some(producer) = edge.producer else {
            continue;
        };
        let targets = readers
            .get(edge.id.0)
            .ok_or(GraphError::MissingEdge(edge.id))?;
        adjacency
            .get_mut(producer.0)
            .ok_or(GraphError::MissingNode(producer))?
            .extend(targets.iter().map(|node| node.0));
    }
    for node in &graph.nodes {
        if !dce.is_active(node.id) {
            continue;
        }
        let NodeKind::Goto { mark } = node.kind else {
            continue;
        };
        let target = graph
            .mark(mark)
            .ok_or(GraphError::MissingNode(node.id))?
            .target;
        let targets = readers
            .get(target.0)
            .ok_or(GraphError::MissingEdge(target))?;
        adjacency
            .get_mut(node.id.0)
            .ok_or(GraphError::MissingNode(node.id))?
            .extend(targets.iter().map(|target| target.0));
    }
    for targets in &mut adjacency {
        targets.sort_unstable();
        targets.dedup();
    }
    Ok(adjacency)
}

fn cyclic_nodes(adjacency: &[Vec<usize>]) -> Vec<bool> {
    let components = stable_components(adjacency);
    let mut cyclic = vec![false; adjacency.len()];
    for component in components {
        let is_cycle = component.len() > 1
            || component.first().is_some_and(|node| {
                adjacency
                    .get(*node)
                    .is_some_and(|targets| targets.binary_search(node).is_ok())
            });
        if is_cycle {
            for node in component {
                if let Some(slot) = cyclic.get_mut(node) {
                    *slot = true;
                }
            }
        }
    }
    cyclic
}

fn stable_components(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut visited = vec![false; adjacency.len()];
    let mut order = Vec::with_capacity(adjacency.len());
    for node in 0..adjacency.len() {
        finish_from(node, adjacency, &mut visited, &mut order);
    }
    let reverse = reverse_adjacency(adjacency);
    visited.fill(false);
    let mut components = Vec::new();
    for node in order.into_iter().rev() {
        if visited.get(node).copied().unwrap_or(true) {
            continue;
        }
        components.push(collect_component(node, &reverse, &mut visited));
    }
    components.sort_by_key(|component| component.first().copied().unwrap_or(usize::MAX));
    components
}

fn finish_from(
    start: usize,
    adjacency: &[Vec<usize>],
    visited: &mut [bool],
    order: &mut Vec<usize>,
) {
    if visited.get(start).copied().unwrap_or(true) {
        return;
    }
    visited[start] = true;
    let mut stack = vec![(start, 0_usize)];
    while let Some((node, next)) = stack.pop() {
        let Some(targets) = adjacency.get(node) else {
            continue;
        };
        if let Some(target) = targets.get(next).copied() {
            stack.push((node, next + 1));
            if visited.get(target).is_some_and(|found| !*found) {
                visited[target] = true;
                stack.push((target, 0));
            }
        } else {
            order.push(node);
        }
    }
}

fn reverse_adjacency(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut reverse = vec![Vec::new(); adjacency.len()];
    for (source, targets) in adjacency.iter().enumerate() {
        for target in targets {
            if let Some(inputs) = reverse.get_mut(*target) {
                inputs.push(source);
            }
        }
    }
    for inputs in &mut reverse {
        inputs.sort_unstable();
    }
    reverse
}

fn collect_component(start: usize, reverse: &[Vec<usize>], visited: &mut [bool]) -> Vec<usize> {
    let mut component = Vec::new();
    let mut stack = vec![start];
    visited[start] = true;
    while let Some(node) = stack.pop() {
        component.push(node);
        if let Some(inputs) = reverse.get(node) {
            for input in inputs.iter().rev() {
                if visited.get(*input).is_some_and(|found| !*found) {
                    visited[*input] = true;
                    stack.push(*input);
                }
            }
        }
    }
    component.sort_unstable();
    component
}

fn dynamic_values(
    graph: &UntypedGraph,
    readers: &[Vec<NodeId>],
    cyclic: &[bool],
    dce: &DcePlan,
) -> Result<(Vec<bool>, Vec<bool>), GraphError> {
    let mut dynamic_nodes = cyclic.to_vec();
    let mut pinned_edges = vec![false; graph.edges.len()];
    let mut queue = VecDeque::new();
    for mark in &graph.marks {
        mark_edge(mark.target, &mut pinned_edges, &mut queue)?;
    }
    pin_latest_value_inputs(graph, dce, &mut dynamic_nodes, &mut pinned_edges);
    pin_cyclic_edges(graph, cyclic, &mut pinned_edges);
    while let Some(edge) = queue.pop_front() {
        for node in readers.get(edge.0).ok_or(GraphError::MissingEdge(edge))? {
            if let Some(slot) = dynamic_nodes.get_mut(node.0) {
                *slot = true;
            }
            let found = graph.node(*node).ok_or(GraphError::MissingNode(*node))?;
            for input in &found.inputs {
                mark_edge(*input, &mut pinned_edges, &mut queue)?;
            }
            for output in &found.outputs {
                mark_edge(*output, &mut pinned_edges, &mut queue)?;
            }
        }
    }
    Ok((dynamic_nodes, pinned_edges))
}

fn pin_latest_value_inputs(
    graph: &UntypedGraph,
    dce: &DcePlan,
    dynamic: &mut [bool],
    pinned: &mut [bool],
) {
    for node in &graph.nodes {
        if !dce.is_active(node.id) || node.inputs.len() < 2 {
            continue;
        }
        if let Some(slot) = dynamic.get_mut(node.id.0) {
            *slot = true;
        }
        for edge in &node.inputs {
            if let Some(slot) = pinned.get_mut(edge.0) {
                *slot = true;
            }
        }
    }
}

fn mark_edge(
    edge: EdgeId,
    pinned: &mut [bool],
    queue: &mut VecDeque<EdgeId>,
) -> Result<(), GraphError> {
    let slot = pinned
        .get_mut(edge.0)
        .ok_or(GraphError::MissingEdge(edge))?;
    if !*slot {
        *slot = true;
        queue.push_back(edge);
    }
    Ok(())
}

fn pin_cyclic_edges(graph: &UntypedGraph, cyclic: &[bool], pinned: &mut [bool]) {
    for node in &graph.nodes {
        if !cyclic.get(node.id.0).copied().unwrap_or(false) {
            continue;
        }
        for edge in node.inputs.iter().chain(&node.outputs) {
            if let Some(slot) = pinned.get_mut(edge.0) {
                *slot = true;
            }
        }
    }
}

fn classify_edges(
    graph: &UntypedGraph,
    readers: &[Vec<NodeId>],
    pinned: &[bool],
    pin_frame: bool,
) -> Vec<Retention> {
    graph
        .edges
        .iter()
        .map(|edge| {
            if pin_frame || edge.id == graph.exit || pinned.get(edge.id.0) == Some(&true) {
                return Retention::Pinned;
            }
            match readers.get(edge.id.0).map(Vec::len).unwrap_or_default() {
                0 => Retention::Dead,
                1 => Retention::OneReader,
                _ => Retention::ManyReaders(usize::MAX),
            }
        })
        .collect()
}

fn classify_variables(
    graph: &UntypedGraph,
    readers: &[Vec<NodeId>],
    cyclic: &[bool],
    dynamic: &[bool],
    child_captures: &[bool],
    pin_frame: bool,
) -> Vec<Retention> {
    graph
        .variables
        .iter()
        .map(|variable| {
            let found = readers.get(variable.id.0).map(Vec::as_slice).unwrap_or(&[]);
            let stores = found
                .iter()
                .filter(|node| {
                    matches!(
                        graph.node(**node).map(|node| &node.kind),
                        Some(NodeKind::Store { .. })
                    )
                })
                .count();
            let unstable = found.iter().any(|node| {
                cyclic.get(node.0).copied().unwrap_or(false)
                    || dynamic.get(node.0).copied().unwrap_or(false)
            });
            if pin_frame
                || variable.scope == VarScope::Inherit
                || child_captures.get(variable.id.0) == Some(&true)
                || stores > 1
                || unstable
            {
                Retention::Pinned
            } else {
                match found.len() {
                    0 => Retention::Dead,
                    1 => Retention::OneReader,
                    _ => Retention::ManyReaders(usize::MAX),
                }
            }
        })
        .collect()
}

fn child_variable_captures(graph: &UntypedGraph, dce: &DcePlan) -> Vec<bool> {
    graph
        .variables
        .iter()
        .map(|variable| {
            graph
                .nodes
                .iter()
                .filter(|node| dce.is_active(node.id))
                .any(|node| child_captures_key(node, &variable.key))
        })
        .collect()
}

fn child_captures_key(node: &Node, key: &VarKey) -> bool {
    match &node.kind {
        NodeKind::Subflow { graph } | NodeKind::Each { graph } => graph_captures_key(graph, key),
        NodeKind::Either { left, right, .. } => {
            graph_captures_key(left, key) || graph_captures_key(right, key)
        }
        NodeKind::Continuation { children, .. } => {
            children.iter().any(|child| graph_captures_key(child, key))
        }
        _ => false,
    }
}

fn graph_captures_key(graph: &UntypedGraph, key: &VarKey) -> bool {
    graph
        .variables
        .iter()
        .any(|variable| variable.scope == VarScope::Inherit && &variable.key == key)
}

fn assign_counters(
    edges: &mut [Retention],
    variables: &mut [Retention],
    edge_readers: &[Vec<NodeId>],
    variable_readers: &[Vec<NodeId>],
) -> Result<Vec<ReaderCounterPlan>, GraphError> {
    let mut counters = Vec::new();
    assign_counter_group(edges, edge_readers, true, &mut counters)?;
    assign_counter_group(variables, variable_readers, false, &mut counters)?;
    Ok(counters)
}

fn assign_counter_group(
    retention: &mut [Retention],
    readers: &[Vec<NodeId>],
    edges: bool,
    counters: &mut Vec<ReaderCounterPlan>,
) -> Result<(), GraphError> {
    for (index, item) in retention.iter_mut().enumerate() {
        if !matches!(item, Retention::ManyReaders(_)) {
            continue;
        }
        let found = readers
            .get(index)
            .ok_or_else(|| GraphError::Invalid("liveness reader metadata is missing".into()))?;
        let counter = counters.len();
        *item = Retention::ManyReaders(counter);
        counters.push(ReaderCounterPlan {
            value: if edges {
                LiveValue::Edge(EdgeId(index))
            } else {
                LiveValue::Variable(VarId(index))
            },
            readers: Arc::from(found.clone()),
        });
    }
    Ok(())
}

fn edge_read_action(
    edge: EdgeId,
    plan: &LivenessPlan,
) -> Result<Option<ReleaseAction>, GraphError> {
    let retention = plan
        .edges
        .get(edge.0)
        .ok_or(GraphError::MissingEdge(edge))?;
    Ok(match retention {
        Retention::OneReader => Some(ReleaseAction::ReadEdge {
            edge,
            counter: None,
        }),
        Retention::ManyReaders(counter) => Some(ReleaseAction::ReadEdge {
            edge,
            counter: Some(*counter),
        }),
        Retention::Dead | Retention::Pinned => None,
    })
}

fn variable_read_action(
    variable: VarId,
    plan: &LivenessPlan,
) -> Result<Option<ReleaseAction>, GraphError> {
    let retention = plan
        .variables
        .get(variable.0)
        .ok_or(GraphError::MissingVariable(variable))?;
    Ok(match retention {
        Retention::OneReader => Some(ReleaseAction::ReadVariable {
            variable,
            counter: None,
        }),
        Retention::ManyReaders(counter) => Some(ReleaseAction::ReadVariable {
            variable,
            counter: Some(*counter),
        }),
        Retention::Dead | Retention::Pinned => None,
    })
}

fn release_order(action: &ReleaseAction) -> (usize, usize) {
    match action {
        ReleaseAction::ReadEdge { edge, .. } => (edge.0, 0),
        ReleaseAction::ClearEdge(edge) => (edge.0, 1),
        ReleaseAction::ReadVariable { variable, .. } => (variable.0, 2),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::graph::{BuiltinNode, TypeSpec, UntypedGraphBuilder};

    fn number() -> TypeSpec {
        TypeSpec::new("Number", json!({"type": "number"}))
    }

    /// Verifies repeated preparation yields exactly the same liveness metadata.
    #[test]
    fn preparation_is_deterministic() {
        let graph = fanout_graph();
        let dce = dce::prepare_dce(&graph).expect("DCE should prepare");
        let first = prepare_liveness(&graph, &dce).expect("first liveness plan should prepare");
        let second = prepare_liveness(&graph, &dce).expect("second liveness plan should prepare");
        assert_eq!(first, second);
    }

    /// Verifies latest-value join inputs stay pinned for deterministic reactivation.
    #[test]
    fn fanout_input_is_reclaimed_and_join_inputs_are_pinned() {
        let graph = fanout_graph();
        let dce = dce::prepare_dce(&graph).expect("DCE should prepare");
        let plan = prepare_liveness(&graph, &dce).expect("liveness should prepare");
        assert_eq!(plan.edges[graph.entry.0], Retention::OneReader);
        assert_eq!(plan.edges[1], Retention::Pinned);
        assert_eq!(plan.edges[2], Retention::Pinned);
        assert_eq!(plan.edges[graph.exit.0], Retention::Pinned);
    }

    /// Verifies arbitrary-writing continuations conservatively pin their frame.
    #[test]
    fn continuation_pins_all_frame_values() {
        let mut builder = UntypedGraphBuilder::new("continuation_liveness");
        let input = builder.edge("input", number());
        let output = builder.edge("output", number());
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
        let dce = dce::prepare_dce(&graph).expect("DCE should prepare");
        let plan = prepare_liveness(&graph, &dce).expect("liveness should prepare");
        assert!(plan.edges.iter().all(|item| *item == Retention::Pinned));
    }

    /// Verifies typed re-entry values remain pinned across future activations.
    #[test]
    fn reentry_mark_pins_target_value() {
        let mut builder = UntypedGraphBuilder::new("mark_liveness");
        let input = builder.edge("input", number());
        let resumed = builder.edge("resumed", number());
        let output = builder.edge("output", number());
        let mark = builder.mark(input);
        builder.set_entry(input).set_exit(output);
        builder.node(
            "wait",
            NodeKind::Suspend {
                resume_type: "Number".into(),
                payload: Value::NULL,
            },
            vec![input],
            vec![resumed],
        );
        builder.goto("repeat", resumed, mark);
        builder.node(
            "copy",
            NodeKind::Builtin {
                op: BuiltinNode::Identity,
            },
            vec![input],
            vec![output],
        );
        let graph = builder.build().expect("graph should build");
        let dce = dce::prepare_dce(&graph).expect("DCE should prepare");
        let plan = prepare_liveness(&graph, &dce).expect("liveness should prepare");
        assert_eq!(plan.edges[input.0], Retention::Pinned);
    }

    fn fanout_graph() -> UntypedGraph {
        let mut builder = UntypedGraphBuilder::new("liveness_fanout");
        let input = builder.edge("input", number());
        let left = builder.edge("left", number());
        let right = builder.edge("right", number());
        let output = builder.edge("output", TypeSpec::new("Pair", json!({"type": "array"})));
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
            "join",
            NodeKind::Builtin {
                op: BuiltinNode::PackTuple,
            },
            vec![left, right],
            vec![output],
        );
        builder.build().expect("graph should build")
    }
}
