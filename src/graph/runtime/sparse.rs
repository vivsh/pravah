use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SparseState {
    pub(crate) frames: Arc<[SparseFrame]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) suspension: Option<SparseSuspension>,
}

#[cfg(test)]
impl SparseState {
    pub(crate) fn frame_mut(&mut self, index: usize) -> Option<&mut SparseFrame> {
        Arc::make_mut(&mut self.frames).get_mut(index)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SparseFrame {
    pub(crate) graph_path: GraphPath,
    pub(crate) write_epoch: u64,
    #[serde(default, skip_serializing_if = "arc_is_empty")]
    pub(crate) edges: Arc<[SparseEdge]>,
    #[serde(default, skip_serializing_if = "arc_is_empty")]
    pub(crate) variables: Arc<[SparseVariable]>,
    #[serde(default, skip_serializing_if = "arc_is_empty")]
    pub(crate) node_epochs: Arc<[SparseNodeEpoch]>,
    #[serde(default, skip_serializing_if = "arc_is_empty")]
    pub(crate) checkpoints: Arc<[SparseNodeValue]>,
    #[serde(default, skip_serializing_if = "arc_is_empty")]
    pub(crate) continuation_states: Arc<[SparseNodeValue]>,
    #[serde(default, skip_serializing_if = "arc_is_empty")]
    pub(crate) continuation_inboxes: Arc<[SparseNodeInbox]>,
    #[serde(default, skip_serializing_if = "arc_is_empty")]
    pub(crate) continuation_child_queues: Arc<[SparseNodeQueue]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) return_target: Option<ReturnTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SparseEdge {
    pub(crate) edge: EdgeId,
    pub(crate) epoch: u64,
    pub(crate) value: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SparseVariable {
    pub(crate) variable: VarId,
    pub(crate) epoch: u64,
    pub(crate) value: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SparseNodeEpoch {
    pub(crate) node: NodeId,
    pub(crate) epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SparseNodeValue {
    pub(crate) node: NodeId,
    pub(crate) value: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SparseNodeInbox {
    pub(crate) node: NodeId,
    pub(crate) values: Arc<[ContinuationChildResult]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SparseNodeQueue {
    pub(crate) node: NodeId,
    pub(crate) values: Arc<[ContinuationChildCall]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SparseSuspension {
    pub(crate) frame_depth: usize,
    pub(crate) graph_path: GraphPath,
    pub(crate) node: NodeId,
    pub(crate) resume_type: String,
    pub(crate) payload: Value,
}

pub(super) fn sparse_state(
    callables: &[CompiledGraph],
    state: &State,
) -> Result<SparseState, GraphError> {
    let frames = state
        .frames
        .iter()
        .enumerate()
        .map(|(index, frame)| sparse_frame(callables, frame, index))
        .collect::<Result<Vec<_>, _>>()?
        .into();
    let suspension = state
        .suspension
        .as_ref()
        .map(|value| sparse_suspension(callables, value))
        .transpose()?;
    Ok(SparseState { frames, suspension })
}

pub(super) fn expand_state(
    callables: &[CompiledGraph],
    sparse: SparseState,
) -> Result<State, GraphError> {
    let frames = sparse
        .frames
        .iter()
        .enumerate()
        .map(|(index, frame)| expand_frame(callables, frame, index))
        .collect::<Result<Vec<_>, _>>()?;
    let suspension = sparse
        .suspension
        .map(|value| expand_suspension(callables, value))
        .transpose()?;
    Ok(State { frames, suspension })
}

fn sparse_frame(
    callables: &[CompiledGraph],
    frame: &Frame,
    frame_index: usize,
) -> Result<SparseFrame, GraphError> {
    let graph = callable_by_index(callables, frame.graph_index, frame_index)?;
    validate_dense_lengths(graph, frame, frame_index)?;
    Ok(SparseFrame {
        graph_path: graph.path.clone(),
        write_epoch: frame.write_epoch,
        edges: sparse_edges(frame, frame_index)?,
        variables: sparse_variables(frame, frame_index)?,
        node_epochs: sparse_node_epochs(frame),
        checkpoints: sparse_node_values(&frame.checkpoints),
        continuation_states: sparse_node_values(&frame.continuation_states),
        continuation_inboxes: sparse_node_inboxes(frame),
        continuation_child_queues: sparse_node_queues(frame),
        return_target: frame.return_target.clone(),
    })
}

fn expand_frame(
    callables: &[CompiledGraph],
    sparse: &SparseFrame,
    frame_index: usize,
) -> Result<Frame, GraphError> {
    let (graph_index, graph) = callable_by_path(callables, &sparse.graph_path, frame_index)?;
    let mut frame = empty_frame(
        graph_index,
        graph,
        sparse.write_epoch,
        sparse.return_target.clone(),
    );
    expand_edges(&mut frame, &sparse.edges, frame_index)?;
    expand_variables(&mut frame, &sparse.variables, frame_index)?;
    expand_node_epochs(&mut frame, &sparse.node_epochs, frame_index)?;
    expand_node_values(
        &mut frame.checkpoints,
        &sparse.checkpoints,
        frame_index,
        "checkpoint",
    )?;
    expand_node_values(
        &mut frame.continuation_states,
        &sparse.continuation_states,
        frame_index,
        "continuation state",
    )?;
    expand_node_inboxes(&mut frame, &sparse.continuation_inboxes, frame_index)?;
    expand_node_queues(&mut frame, &sparse.continuation_child_queues, frame_index)?;
    Ok(frame)
}

fn empty_frame(
    graph_index: usize,
    graph: &CompiledGraph,
    write_epoch: u64,
    return_target: Option<ReturnTarget>,
) -> Frame {
    Frame {
        graph_index,
        values: vec![None; graph.graph.edges.len()],
        write_epoch,
        edge_epochs: vec![0; graph.graph.edges.len()],
        variables: vec![None; graph.graph.variables.len()],
        variable_epochs: vec![0; graph.graph.variables.len()],
        checkpoints: vec![None; graph.graph.nodes.len()],
        continuation_states: vec![None; graph.graph.nodes.len()],
        continuation_inboxes: vec![Vec::new(); graph.graph.nodes.len()],
        continuation_child_queues: vec![Vec::new(); graph.graph.nodes.len()],
        node_epochs: vec![0; graph.graph.nodes.len()],
        reader_counts: Vec::new(),
        return_target,
    }
}

fn sparse_edges(frame: &Frame, frame_index: usize) -> Result<Arc<[SparseEdge]>, GraphError> {
    let entries = frame
        .values
        .iter()
        .zip(&frame.edge_epochs)
        .enumerate()
        .filter_map(|(index, (value, epoch))| {
            value.as_ref().map(|value| {
                require_live_epoch(*epoch, frame.write_epoch, frame_index, "edge")?;
                Ok(SparseEdge {
                    edge: EdgeId(index),
                    epoch: *epoch,
                    value: value.clone(),
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::from(entries))
}

fn sparse_variables(
    frame: &Frame,
    frame_index: usize,
) -> Result<Arc<[SparseVariable]>, GraphError> {
    let entries = frame
        .variables
        .iter()
        .zip(&frame.variable_epochs)
        .enumerate()
        .filter_map(|(index, (value, epoch))| {
            value.as_ref().map(|value| {
                require_live_epoch(*epoch, frame.write_epoch, frame_index, "variable")?;
                Ok(SparseVariable {
                    variable: VarId(index),
                    epoch: *epoch,
                    value: value.clone(),
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::from(entries))
}

fn sparse_node_epochs(frame: &Frame) -> Arc<[SparseNodeEpoch]> {
    frame
        .node_epochs
        .iter()
        .enumerate()
        .filter(|(_, epoch)| **epoch > 0)
        .map(|(index, epoch)| SparseNodeEpoch {
            node: NodeId(index),
            epoch: *epoch,
        })
        .collect::<Vec<_>>()
        .into()
}

fn sparse_node_values(values: &[Option<Value>]) -> Arc<[SparseNodeValue]> {
    values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            value.as_ref().map(|value| SparseNodeValue {
                node: NodeId(index),
                value: value.clone(),
            })
        })
        .collect::<Vec<_>>()
        .into()
}

fn sparse_node_inboxes(frame: &Frame) -> Arc<[SparseNodeInbox]> {
    frame
        .continuation_inboxes
        .iter()
        .enumerate()
        .filter(|(_, values)| !values.is_empty())
        .map(|(index, values)| SparseNodeInbox {
            node: NodeId(index),
            values: Arc::from(values.clone()),
        })
        .collect::<Vec<_>>()
        .into()
}

fn sparse_node_queues(frame: &Frame) -> Arc<[SparseNodeQueue]> {
    frame
        .continuation_child_queues
        .iter()
        .enumerate()
        .filter(|(_, values)| !values.is_empty())
        .map(|(index, values)| SparseNodeQueue {
            node: NodeId(index),
            values: Arc::from(values.clone()),
        })
        .collect::<Vec<_>>()
        .into()
}

fn expand_edges(
    frame: &mut Frame,
    entries: &[SparseEdge],
    frame_index: usize,
) -> Result<(), GraphError> {
    let mut previous = None;
    for entry in entries.iter() {
        validate_id(
            previous,
            entry.edge.0,
            frame.values.len(),
            frame_index,
            "edge",
        )?;
        require_live_epoch(entry.epoch, frame.write_epoch, frame_index, "edge")?;
        frame.values[entry.edge.0] = Some(entry.value.clone());
        frame.edge_epochs[entry.edge.0] = entry.epoch;
        previous = Some(entry.edge.0);
    }
    Ok(())
}

fn expand_variables(
    frame: &mut Frame,
    entries: &[SparseVariable],
    frame_index: usize,
) -> Result<(), GraphError> {
    let mut previous = None;
    for entry in entries.iter() {
        validate_id(
            previous,
            entry.variable.0,
            frame.variables.len(),
            frame_index,
            "variable",
        )?;
        require_live_epoch(entry.epoch, frame.write_epoch, frame_index, "variable")?;
        frame.variables[entry.variable.0] = Some(entry.value.clone());
        frame.variable_epochs[entry.variable.0] = entry.epoch;
        previous = Some(entry.variable.0);
    }
    Ok(())
}

fn expand_node_epochs(
    frame: &mut Frame,
    entries: &[SparseNodeEpoch],
    frame_index: usize,
) -> Result<(), GraphError> {
    let mut previous = None;
    for entry in entries.iter() {
        validate_id(
            previous,
            entry.node.0,
            frame.node_epochs.len(),
            frame_index,
            "node epoch",
        )?;
        require_live_epoch(entry.epoch, frame.write_epoch, frame_index, "node epoch")?;
        frame.node_epochs[entry.node.0] = entry.epoch;
        previous = Some(entry.node.0);
    }
    Ok(())
}

fn expand_node_values(
    slots: &mut [Option<Value>],
    entries: &[SparseNodeValue],
    frame_index: usize,
    label: &str,
) -> Result<(), GraphError> {
    let mut previous = None;
    for entry in entries.iter() {
        validate_id(previous, entry.node.0, slots.len(), frame_index, label)?;
        slots[entry.node.0] = Some(entry.value.clone());
        previous = Some(entry.node.0);
    }
    Ok(())
}

fn expand_node_inboxes(
    frame: &mut Frame,
    entries: &[SparseNodeInbox],
    frame_index: usize,
) -> Result<(), GraphError> {
    let mut previous = None;
    for entry in entries.iter() {
        validate_id(
            previous,
            entry.node.0,
            frame.continuation_inboxes.len(),
            frame_index,
            "inbox",
        )?;
        if entry.values.is_empty() {
            return Err(sparse_error(frame_index, "inbox entry is empty"));
        }
        frame.continuation_inboxes[entry.node.0] = entry.values.to_vec();
        previous = Some(entry.node.0);
    }
    Ok(())
}

fn expand_node_queues(
    frame: &mut Frame,
    entries: &[SparseNodeQueue],
    frame_index: usize,
) -> Result<(), GraphError> {
    let mut previous = None;
    for entry in entries.iter() {
        validate_id(
            previous,
            entry.node.0,
            frame.continuation_child_queues.len(),
            frame_index,
            "queue",
        )?;
        if entry.values.is_empty() {
            return Err(sparse_error(frame_index, "queue entry is empty"));
        }
        frame.continuation_child_queues[entry.node.0] = entry.values.to_vec();
        previous = Some(entry.node.0);
    }
    Ok(())
}

fn sparse_suspension(
    callables: &[CompiledGraph],
    suspension: &Suspension,
) -> Result<SparseSuspension, GraphError> {
    let graph = callables
        .get(suspension.graph_index)
        .ok_or_else(|| GraphError::SnapshotValidation("suspension graph is missing".into()))?;
    Ok(SparseSuspension {
        frame_depth: suspension.frame_depth,
        graph_path: graph.path.clone(),
        node: suspension.node,
        resume_type: suspension.resume_type.clone(),
        payload: suspension.payload.clone(),
    })
}

fn expand_suspension(
    callables: &[CompiledGraph],
    sparse: SparseSuspension,
) -> Result<Suspension, GraphError> {
    let (graph_index, _) = callable_by_path(callables, &sparse.graph_path, sparse.frame_depth)?;
    Ok(Suspension {
        frame_depth: sparse.frame_depth,
        graph_index,
        node: sparse.node,
        resume_type: sparse.resume_type,
        payload: sparse.payload,
    })
}

fn callable_by_index(
    callables: &[CompiledGraph],
    graph_index: usize,
    frame_index: usize,
) -> Result<&CompiledGraph, GraphError> {
    callables.get(graph_index).ok_or_else(|| {
        sparse_error(
            frame_index,
            &format!("compiled graph index {graph_index} is missing"),
        )
    })
}

fn callable_by_path<'a>(
    callables: &'a [CompiledGraph],
    path: &GraphPath,
    frame_index: usize,
) -> Result<(usize, &'a CompiledGraph), GraphError> {
    callables
        .iter()
        .enumerate()
        .find(|(_, graph)| graph.path == *path)
        .ok_or_else(|| sparse_error(frame_index, "graph path does not exist in prepared graph"))
}

fn validate_dense_lengths(
    graph: &CompiledGraph,
    frame: &Frame,
    frame_index: usize,
) -> Result<(), GraphError> {
    let edge_count = graph.graph.edges.len();
    let variable_count = graph.graph.variables.len();
    let node_count = graph.graph.nodes.len();
    if frame.values.len() != edge_count
        || frame.edge_epochs.len() != edge_count
        || frame.variables.len() != variable_count
        || frame.variable_epochs.len() != variable_count
        || frame.node_epochs.len() != node_count
        || frame.checkpoints.len() != node_count
        || frame.continuation_states.len() != node_count
        || frame.continuation_inboxes.len() != node_count
        || frame.continuation_child_queues.len() != node_count
    {
        return Err(sparse_error(
            frame_index,
            "dense frame shape does not match graph",
        ));
    }
    Ok(())
}

fn validate_id(
    previous: Option<usize>,
    id: usize,
    limit: usize,
    frame_index: usize,
    label: &str,
) -> Result<(), GraphError> {
    if id >= limit {
        return Err(sparse_error(
            frame_index,
            &format!("{label} id {id} is out of range"),
        ));
    }
    if previous.is_some_and(|previous| id <= previous) {
        return Err(sparse_error(
            frame_index,
            &format!("{label} entries are duplicate or unordered"),
        ));
    }
    Ok(())
}

fn require_live_epoch(
    epoch: u64,
    write_epoch: u64,
    frame_index: usize,
    label: &str,
) -> Result<(), GraphError> {
    if epoch == 0 || epoch > write_epoch {
        return Err(sparse_error(
            frame_index,
            &format!("{label} epoch {epoch} is inconsistent with frame epoch {write_epoch}"),
        ));
    }
    Ok(())
}

fn sparse_error(frame_index: usize, reason: &str) -> GraphError {
    GraphError::SnapshotValidation(format!("snapshot frame {frame_index}: {reason}"))
}

fn arc_is_empty<T>(values: &Arc<[T]>) -> bool {
    values.is_empty()
}
