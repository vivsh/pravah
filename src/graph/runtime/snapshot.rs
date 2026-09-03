use super::*;
use crate::graph::Value;

pub(super) fn validate_snapshot_state(
    callables: &[CompiledGraph],
    root_index: usize,
    state: &State,
) -> Result<(), GraphError> {
    if state.frames.is_empty() {
        if state.suspension.is_some() {
            return Err(GraphError::SnapshotValidation(
                "completed snapshot cannot retain a suspension".into(),
            ));
        }
        return Ok(());
    }
    validate_snapshot_frame_chain(callables, root_index, state)?;
    for (frame_index, frame) in state.frames.iter().enumerate() {
        let graph = callables.get(frame.graph_index).ok_or_else(|| {
            GraphError::Invalid(format!(
                "snapshot frame {frame_index} references missing graph index {}",
                frame.graph_index
            ))
        })?;
        if frame.values.len() != graph.graph.edges.len() {
            return Err(GraphError::Invalid(format!(
                "snapshot frame {frame_index} has {} edge values, expected {}",
                frame.values.len(),
                graph.graph.edges.len()
            )));
        }
        if frame.variables.len() != graph.graph.variables.len() {
            return Err(GraphError::Invalid(format!(
                "snapshot frame {frame_index} has {} variables, expected {}",
                frame.variables.len(),
                graph.graph.variables.len()
            )));
        }
        if frame.checkpoints.len() != graph.graph.nodes.len()
            || frame.continuation_states.len() != graph.graph.nodes.len()
            || frame.continuation_inboxes.len() != graph.graph.nodes.len()
            || frame.continuation_child_queues.len() != graph.graph.nodes.len()
            || frame.edge_epochs.len() != graph.graph.edges.len()
            || frame.variable_epochs.len() != graph.graph.variables.len()
            || frame.node_epochs.len() != graph.graph.nodes.len()
        {
            return Err(GraphError::Invalid(format!(
                "snapshot frame {frame_index} node state does not match graph"
            )));
        }
        for (edge_index, value) in frame.values.iter().enumerate() {
            if let Some(value) = value {
                validate_edge_value(
                    &graph.graph,
                    EdgeId(edge_index),
                    value,
                    &format!("snapshot frame {frame_index}"),
                )?;
            }
        }
        for (edge_index, epoch) in frame.edge_epochs.iter().enumerate() {
            if *epoch > frame.write_epoch {
                return Err(GraphError::Invalid(format!(
                    "snapshot frame {frame_index} edge {:?} epoch exceeds its frame epoch",
                    EdgeId(edge_index)
                )));
            }
        }
        for (node_index, epoch) in frame.node_epochs.iter().enumerate() {
            if *epoch > frame.write_epoch {
                return Err(GraphError::Invalid(format!(
                    "snapshot frame {frame_index} node {:?} epoch exceeds its frame epoch",
                    NodeId(node_index)
                )));
            }
        }
        for (var_index, value) in frame.variables.iter().enumerate() {
            if let Some(value) = value {
                if frame
                    .variable_epochs
                    .get(var_index)
                    .copied()
                    .unwrap_or_default()
                    == 0
                {
                    return Err(GraphError::Invalid(format!(
                        "snapshot frame {frame_index} variable {:?} has a value but no epoch",
                        VarId(var_index)
                    )));
                }
                let variable = graph
                    .graph
                    .variable(VarId(var_index))
                    .ok_or(GraphError::MissingVariable(VarId(var_index)))?;
                validate_value(
                    &variable.type_spec,
                    value,
                    &format!(
                        "snapshot frame {frame_index} variable '{}::{}'",
                        variable.key.namespace, variable.key.type_name
                    ),
                )?;
            }
        }
        for (node_index, checkpoint) in frame.checkpoints.iter().enumerate() {
            let node_id = NodeId(node_index);
            let node = graph
                .graph
                .node(node_id)
                .ok_or(GraphError::MissingNode(node_id))?;
            if checkpoint.is_none() {
                continue;
            }
            if frame
                .node_epochs
                .get(node_index)
                .copied()
                .unwrap_or_default()
                == 0
            {
                return Err(GraphError::Invalid(format!(
                    "snapshot frame {frame_index} has checkpoint for node '{}' without activation",
                    node.name
                )));
            }
            let compiled_node = graph
                .nodes
                .get(node_index)
                .ok_or(GraphError::MissingNode(node_id))?;
            if !compiled_node.can_continue {
                return Err(GraphError::Invalid(format!(
                    "snapshot frame {frame_index} has checkpoint for node '{}' which cannot continue",
                    node.name
                )));
            }
            if matches!(compiled_node.kind, CompiledNodeKind::Each { .. }) {
                validate_each_snapshot_checkpoint(
                    checkpoint.as_ref().ok_or_else(|| {
                        GraphError::SnapshotValidation(format!(
                            "snapshot frame {frame_index} each checkpoint disappeared"
                        ))
                    })?,
                    frame_index,
                    &node.name,
                )?;
            }
        }
        for (node_index, compiled_node) in graph.nodes.iter().enumerate() {
            let CompiledNodeKind::Continuation { payload, .. } = &compiled_node.kind else {
                continue;
            };
            validate_agent_snapshot_state(
                payload,
                frame.checkpoints.get(node_index).and_then(Option::as_ref),
                frame
                    .continuation_states
                    .get(node_index)
                    .and_then(Option::as_ref),
            )?;
        }
        for (node_index, state) in frame.continuation_states.iter().enumerate() {
            if state.is_none() {
                continue;
            }
            let node_id = NodeId(node_index);
            let node = graph
                .graph
                .node(node_id)
                .ok_or(GraphError::MissingNode(node_id))?;
            if !matches!(node.kind, NodeKind::Continuation { .. }) {
                return Err(GraphError::Invalid(format!(
                    "snapshot frame {frame_index} has continuation state for non-continuation node '{}'",
                    node.name
                )));
            }
        }
        for (node_index, inbox) in frame.continuation_inboxes.iter().enumerate() {
            if inbox.is_empty() {
                continue;
            }
            let node_id = NodeId(node_index);
            let node = graph
                .graph
                .node(node_id)
                .ok_or(GraphError::MissingNode(node_id))?;
            if !matches!(node.kind, NodeKind::Continuation { .. }) {
                return Err(GraphError::Invalid(format!(
                    "snapshot frame {frame_index} has continuation inbox for non-continuation node '{}'",
                    node.name
                )));
            }
            if frame
                .checkpoints
                .get(node_index)
                .is_none_or(Option::is_none)
            {
                return Err(GraphError::Invalid(format!(
                    "snapshot frame {frame_index} has continuation inbox for node '{}' without checkpoint",
                    node.name
                )));
            }
        }
        for (node_index, queue) in frame.continuation_child_queues.iter().enumerate() {
            if queue.is_empty() {
                continue;
            }
            let node = graph
                .graph
                .node(NodeId(node_index))
                .ok_or(GraphError::MissingNode(NodeId(node_index)))?;
            if !matches!(node.kind, NodeKind::Continuation { .. }) {
                return Err(GraphError::Invalid(format!(
                    "snapshot frame {frame_index} has continuation child queue for non-continuation node '{}'",
                    node.name
                )));
            }
            if frame
                .checkpoints
                .get(node_index)
                .is_none_or(Option::is_none)
            {
                return Err(GraphError::Invalid(format!(
                    "snapshot frame {frame_index} has continuation child queue for node '{}' without checkpoint",
                    node.name
                )));
            }
            let children = graph.child_indices.get(node_index).ok_or_else(|| {
                GraphError::Invalid(format!(
                    "snapshot frame {frame_index} node state does not match compiled graph"
                ))
            })?;
            for call in queue {
                if children.continuation.get(call.child_index).is_none() {
                    return Err(GraphError::Invalid(format!(
                        "snapshot frame {frame_index} continuation node '{}' queued missing child index {}",
                        node.name, call.child_index
                    )));
                }
            }
        }
    }
    if let Some(suspension) = &state.suspension {
        validate_snapshot_suspension(callables, state, suspension)?;
    }
    Ok(())
}

pub(super) fn validate_each_snapshot_checkpoint(
    value: &Value,
    frame_index: usize,
    node_name: &str,
) -> Result<(), GraphError> {
    let checkpoint: EachVmCheckpoint = crate::graph::from_value(value.clone()).map_err(|err| {
        GraphError::SnapshotValidation(format!(
            "snapshot frame {frame_index} each node '{node_name}' has invalid checkpoint: {err}"
        ))
    })?;
    if checkpoint.items.is_empty()
        || checkpoint.index >= checkpoint.items.len()
        || checkpoint.outputs.len() != checkpoint.index
    {
        return Err(GraphError::SnapshotValidation(format!(
            "snapshot frame {frame_index} each node '{node_name}' checkpoint is inconsistent"
        )));
    }
    Ok(())
}

pub(super) fn validate_snapshot_frame_chain(
    callables: &[CompiledGraph],
    root_index: usize,
    state: &State,
) -> Result<(), GraphError> {
    let root = state
        .frames
        .first()
        .ok_or_else(|| GraphError::SnapshotValidation("snapshot root frame is missing".into()))?;
    if root.graph_index != root_index || root.return_target.is_some() {
        return Err(GraphError::SnapshotValidation(format!(
            "root frame must reference graph {root_index} and have no return target"
        )));
    }
    for frame_index in 1..state.frames.len() {
        validate_snapshot_child_frame(callables, state, frame_index)?;
    }
    Ok(())
}

pub(super) fn validate_snapshot_child_frame(
    callables: &[CompiledGraph],
    state: &State,
    frame_index: usize,
) -> Result<(), GraphError> {
    let parent = state.frames.get(frame_index - 1).ok_or_else(|| {
        GraphError::SnapshotValidation(format!("snapshot frame {frame_index} has no parent frame"))
    })?;
    let child = state.frames.get(frame_index).ok_or_else(|| {
        GraphError::SnapshotValidation(format!("snapshot frame {frame_index} is missing"))
    })?;
    let parent_graph = callables.get(parent.graph_index).ok_or_else(|| {
        GraphError::SnapshotValidation(format!(
            "snapshot parent frame references missing graph {}",
            parent.graph_index
        ))
    })?;
    let target = child.return_target.as_ref().ok_or_else(|| {
        GraphError::SnapshotValidation(format!(
            "snapshot child frame {frame_index} has no return target"
        ))
    })?;
    match target {
        ReturnTarget::Edge { parent_edge } => {
            let edge = parent_graph.graph.edge(*parent_edge).ok_or_else(|| {
                GraphError::SnapshotValidation(format!(
                    "snapshot child frame {frame_index} returns to missing edge {parent_edge:?}"
                ))
            })?;
            let producer = edge.producer.ok_or_else(|| {
                GraphError::SnapshotValidation(format!(
                    "snapshot child frame {frame_index} return edge has no subflow producer"
                ))
            })?;
            let node = parent_graph.nodes.get(producer.0).ok_or_else(|| {
                GraphError::SnapshotValidation(format!(
                    "snapshot child frame {frame_index} return producer is missing"
                ))
            })?;
            if !matches!(
                node.kind,
                CompiledNodeKind::Subflow { child_index } if child_index == child.graph_index
            ) {
                return Err(GraphError::SnapshotValidation(format!(
                    "snapshot child frame {frame_index} does not match its parent subflow"
                )));
            }
        }
        ReturnTarget::Either { parent_node } => {
            let node = snapshot_parent_node(parent_graph, *parent_node, frame_index)?;
            if !matches!(
                node.kind,
                CompiledNodeKind::Either { left_index, right_index, .. }
                    if left_index == child.graph_index || right_index == child.graph_index
            ) {
                return Err(GraphError::SnapshotValidation(format!(
                    "snapshot child frame {frame_index} does not match its parent either node"
                )));
            }
        }
        ReturnTarget::Each { parent_node } => {
            let node = snapshot_parent_node(parent_graph, *parent_node, frame_index)?;
            if !matches!(node.kind, CompiledNodeKind::Each { child_index } if child_index == child.graph_index)
                || parent
                    .checkpoints
                    .get(parent_node.0)
                    .is_none_or(Option::is_none)
            {
                return Err(GraphError::SnapshotValidation(format!(
                    "snapshot child frame {frame_index} does not match an active parent each node"
                )));
            }
        }
        ReturnTarget::Continuation {
            parent_node,
            call_id,
        } => {
            let node = snapshot_parent_node(parent_graph, *parent_node, frame_index)?;
            let valid_child = match &node.kind {
                CompiledNodeKind::Continuation { children, .. } => {
                    children.contains(&child.graph_index)
                }
                _ => false,
            };
            if call_id.is_empty()
                || !valid_child
                || parent
                    .checkpoints
                    .get(parent_node.0)
                    .is_none_or(Option::is_none)
            {
                return Err(GraphError::SnapshotValidation(format!(
                    "snapshot child frame {frame_index} does not match an active continuation call"
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn snapshot_parent_node(
    parent_graph: &CompiledGraph,
    node: NodeId,
    frame_index: usize,
) -> Result<&CompiledNode, GraphError> {
    parent_graph
        .nodes
        .get(node.0)
        .filter(|found| found.id == node)
        .ok_or_else(|| {
            GraphError::SnapshotValidation(format!(
                "snapshot child frame {frame_index} references missing parent node {node:?}"
            ))
        })
}

pub(super) fn validate_snapshot_suspension(
    callables: &[CompiledGraph],
    state: &State,
    suspension: &Suspension,
) -> Result<(), GraphError> {
    if suspension.frame_depth == 0 || suspension.frame_depth > state.frames.len() {
        return Err(GraphError::Invalid(format!(
            "snapshot suspension frame depth {} is invalid for stack depth {}",
            suspension.frame_depth,
            state.frames.len()
        )));
    }
    if suspension.frame_depth != state.frames.len() {
        return Err(GraphError::Invalid(format!(
            "snapshot suspension frame depth {} is not the active frame depth {}",
            suspension.frame_depth,
            state.frames.len()
        )));
    }
    let frame_index = suspension.frame_depth - 1;
    let frame = state
        .frames
        .get(frame_index)
        .ok_or_else(|| GraphError::Invalid("snapshot suspension frame is missing".into()))?;
    if frame.graph_index != suspension.graph_index {
        return Err(GraphError::Invalid(format!(
            "snapshot suspension graph index {} does not match frame graph index {}",
            suspension.graph_index, frame.graph_index
        )));
    }
    let graph = callables
        .get(suspension.graph_index)
        .ok_or_else(|| GraphError::Invalid("snapshot suspension graph index is invalid".into()))?;
    let node = graph
        .graph
        .node(suspension.node)
        .ok_or(GraphError::MissingNode(suspension.node))?;
    let compiled_node = graph
        .nodes
        .get(suspension.node.0)
        .ok_or(GraphError::MissingNode(suspension.node))?;
    if !compiled_node.can_suspend {
        return Err(GraphError::Invalid(format!(
            "snapshot suspension node '{}' cannot suspend",
            node.name
        )));
    }
    match suspension.target {
        SuspensionTarget::Node => {
            let CompiledNodeKind::Suspend { .. } = &compiled_node.kind else {
                return Err(GraphError::SnapshotValidation(format!(
                    "snapshot suspension node '{}' is not a suspend node",
                    node.name
                )));
            };
            let output = compiled_node.outputs.first().copied().ok_or_else(|| {
                GraphError::SnapshotValidation(format!(
                    "snapshot suspension node '{}' has no output",
                    node.name
                ))
            })?;
            let expected = graph
                .graph
                .edge(output)
                .map(|edge| &edge.type_spec)
                .ok_or_else(|| {
                    GraphError::SnapshotValidation(format!(
                        "snapshot suspension node '{}' output type is missing",
                        node.name
                    ))
                })?;
            if expected != &suspension.resume_type || !inputs_ready(frame, compiled_node)? {
                return Err(GraphError::SnapshotValidation(format!(
                    "snapshot suspension node '{}' has inconsistent resume state",
                    node.name
                )));
            }
        }
        SuspensionTarget::Continuation => {
            let CompiledNodeKind::Continuation { payload, .. } = &compiled_node.kind else {
                return Err(GraphError::SnapshotValidation(format!(
                    "snapshot suspension node '{}' is not a continuation",
                    node.name
                )));
            };
            let checkpoint = frame
                .checkpoints
                .get(suspension.node.0)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    GraphError::SnapshotValidation(format!(
                        "snapshot suspension continuation '{}' has no checkpoint",
                        node.name
                    ))
                })?;
            if frame
                .continuation_inboxes
                .get(suspension.node.0)
                .is_none_or(|inbox| !inbox.is_empty())
                || frame
                    .continuation_child_queues
                    .get(suspension.node.0)
                    .is_none_or(|queue| !queue.is_empty())
            {
                return Err(GraphError::SnapshotValidation(format!(
                    "snapshot suspension continuation '{}' has pending child work",
                    node.name
                )));
            }
            validate_agent_suspension(
                payload,
                checkpoint,
                &suspension.payload,
                &suspension.resume_type,
            )?;
        }
    }
    Ok(())
}
