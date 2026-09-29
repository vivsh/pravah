use super::*;
use crate::graph::runtime::liveness::{LiveValue, ReaderCounterPlan};

impl Runtime {
    pub(super) fn complete_node(
        &mut self,
        frame_index: usize,
        node: &CompiledNode,
    ) -> Result<(), GraphError> {
        remember_node_activation(self.frame_mut(frame_index)?, node)?;
        for action in node.release_actions.iter().copied() {
            self.apply_release(frame_index, action)?;
        }
        Ok(())
    }

    fn apply_release(
        &mut self,
        frame_index: usize,
        action: ReleaseAction,
    ) -> Result<(), GraphError> {
        match action {
            ReleaseAction::ReadEdge { edge, counter } => {
                if self.reader_finished(frame_index, counter)? {
                    self.clear_edge(frame_index, edge)?;
                }
            }
            ReleaseAction::ReadVariable { variable, counter } => {
                if self.reader_finished(frame_index, counter)? {
                    self.clear_variable(frame_index, variable)?;
                }
            }
            ReleaseAction::ClearEdge(edge) => self.clear_edge(frame_index, edge)?,
        }
        Ok(())
    }

    fn reader_finished(
        &mut self,
        frame_index: usize,
        counter: Option<usize>,
    ) -> Result<bool, GraphError> {
        let Some(counter) = counter else {
            return Ok(true);
        };
        let remaining = self
            .state
            .frames
            .get_mut(frame_index)
            .and_then(|frame| frame.reader_counts.get_mut(counter))
            .ok_or_else(|| GraphError::Invalid("reader counter is missing".into()))?;
        *remaining = remaining
            .checked_sub(1)
            .ok_or_else(|| GraphError::Invalid("reader counter underflowed".into()))?;
        Ok(*remaining == 0)
    }

    fn clear_edge(&mut self, frame_index: usize, edge: EdgeId) -> Result<(), GraphError> {
        let slot = self
            .state
            .frames
            .get_mut(frame_index)
            .and_then(|frame| frame.values.get_mut(edge.0))
            .ok_or(GraphError::MissingEdge(edge))?;
        *slot = None;
        Ok(())
    }

    fn clear_variable(&mut self, frame_index: usize, variable: VarId) -> Result<(), GraphError> {
        let slot = self
            .state
            .frames
            .get_mut(frame_index)
            .and_then(|frame| frame.variables.get_mut(variable.0))
            .ok_or(GraphError::MissingVariable(variable))?;
        *slot = None;
        Ok(())
    }
}

pub(super) fn rebuild_reader_counts(
    callables: &[CompiledGraph],
    state: &mut State,
) -> Result<(), GraphError> {
    for frame in &mut state.frames {
        let graph = callables
            .get(frame.graph_index)
            .ok_or_else(|| GraphError::SnapshotValidation("frame graph is missing".into()))?;
        let mut counts = Vec::with_capacity(graph.liveness.counters.len());
        for counter in graph.liveness.counters.iter() {
            counts.push(remaining_readers(graph, frame, counter)?);
        }
        frame.reader_counts = counts;
    }
    Ok(())
}

fn remaining_readers(
    graph: &CompiledGraph,
    frame: &Frame,
    counter: &ReaderCounterPlan,
) -> Result<u32, GraphError> {
    let mut remaining = 0_u32;
    for reader in counter.readers.iter().copied() {
        if !reader_has_consumed(graph, frame, reader, counter.value)? {
            remaining = remaining.checked_add(1).ok_or_else(|| {
                GraphError::SnapshotValidation("reader counter overflowed".into())
            })?;
        }
    }
    Ok(remaining)
}

fn reader_has_consumed(
    graph: &CompiledGraph,
    frame: &Frame,
    reader: NodeId,
    value: LiveValue,
) -> Result<bool, GraphError> {
    let node = graph
        .nodes
        .get(reader.0)
        .filter(|node| node.id == reader)
        .ok_or(GraphError::MissingNode(reader))?;
    let activation = frame
        .node_epochs
        .get(reader.0)
        .ok_or(GraphError::MissingNode(reader))?;
    if *activation == 0 {
        return Ok(false);
    }
    match value {
        LiveValue::Edge(edge) => reader_consumed_edge(frame, *activation, edge),
        LiveValue::Variable(_) => reader_consumed_all_inputs(frame, node, *activation),
    }
}

fn reader_consumed_edge(frame: &Frame, activation: u64, edge: EdgeId) -> Result<bool, GraphError> {
    let epoch = frame
        .edge_epochs
        .get(edge.0)
        .ok_or(GraphError::MissingEdge(edge))?;
    Ok(activation >= *epoch)
}

fn reader_consumed_all_inputs(
    frame: &Frame,
    node: &CompiledNode,
    activation: u64,
) -> Result<bool, GraphError> {
    for edge in node.inputs.iter().copied() {
        let epoch = frame
            .edge_epochs
            .get(edge.0)
            .ok_or(GraphError::MissingEdge(edge))?;
        if activation < *epoch {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[path = "tests/reclaim.rs"]
mod tests;
