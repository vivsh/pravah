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
mod tests {
    use serde_json::json;

    use super::*;
    use crate::graph::{NodeKind, TypeSpec, UntypedGraphBuilder};

    /// Verifies a value with two readers is cleared only after the second commits.
    #[tokio::test]
    async fn multiple_readers_release_after_last_successful_reader() {
        let prepared = prepare_two_reader_graph();
        let mut runtime = prepared
            .start(Value::from(7_i64))
            .expect("runtime should start");

        runtime
            .next(Context::default())
            .await
            .expect("first reader");
        assert!(runtime.state.frames[0].values[0].is_some());

        runtime
            .next(Context::default())
            .await
            .expect("second reader");
        assert!(runtime.state.frames[0].values[0].is_none());
    }

    /// Verifies restored reader counters preserve completed-reader progress.
    #[tokio::test]
    async fn restore_rebuilds_reader_counters() {
        let prepared = prepare_two_reader_graph();
        let mut runtime = prepared
            .start(Value::from(7_i64))
            .expect("runtime should start");
        runtime
            .next(Context::default())
            .await
            .expect("first reader");
        let snapshot = runtime.snapshot().expect("snapshot should build");
        let mut restored = prepared.restore(snapshot).expect("snapshot should restore");

        restored
            .next(Context::default())
            .await
            .expect("second reader");
        assert!(restored.state.frames[0].values[0].is_none());
    }

    /// Verifies a failing instruction leaves its input available for retry.
    #[tokio::test]
    async fn failed_instruction_does_not_release_input() {
        let prepared = prepare_failing_unpack_graph();
        let mut runtime = prepared
            .start(Value::array([Value::from(1_i64)]))
            .expect("runtime should start");

        runtime
            .next(Context::default())
            .await
            .expect_err("unpack should fail");
        assert!(runtime.state.frames[0].values[0].is_some());
        assert_eq!(runtime.state.frames[0].node_epochs[0], 0);
    }

    /// Verifies epoch overflow fails without committing output or releasing input.
    #[tokio::test]
    async fn write_epoch_overflow_preserves_retry_state() {
        let prepared = prepare_two_reader_graph();
        let mut runtime = prepared
            .start(Value::from(7_i64))
            .expect("runtime should start");
        runtime.state.frames[0].write_epoch = u64::MAX;

        let error = runtime
            .next(Context::default())
            .await
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
}
