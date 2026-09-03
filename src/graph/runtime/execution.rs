use super::*;
use crate::graph::Value;

impl Runtime {
    pub(super) async fn step_inner(&mut self, ctx: Context) -> Result<Step, GraphError> {
        let frame_index = self
            .state
            .frames
            .len()
            .checked_sub(1)
            .ok_or_else(|| GraphError::Invalid("runtime has no active frame".into()))?;
        let graph_index = self
            .state
            .frames
            .get(frame_index)
            .map(|frame| frame.graph_index)
            .ok_or_else(|| GraphError::Invalid("runtime has no active frame".into()))?;
        let compiled = self
            .callables
            .get(graph_index)
            .ok_or_else(|| GraphError::Invalid(format!("graph index {graph_index} is invalid")))?;

        for node_id in compiled.instructions.iter().copied() {
            let node = compiled
                .nodes
                .get(node_id.0)
                .filter(|node| node.id == node_id)
                .ok_or(GraphError::MissingNode(node_id))?;
            let frame = self
                .state
                .frames
                .get(frame_index)
                .ok_or_else(|| GraphError::Invalid("active frame disappeared".into()))?;
            if has_continuation(frame, node.id)? {
                if matches!(node.kind, CompiledNodeKind::Continuation { .. }) {
                    return self.poll_continuation(frame_index, node.clone(), ctx).await;
                }
                if node.can_continue {
                    continue;
                }
                return Err(GraphError::Invalid(format!(
                    "node '{}' has checkpoint but cannot continue",
                    node.name
                )));
            }
            let frame = self
                .state
                .frames
                .get(frame_index)
                .ok_or_else(|| GraphError::Invalid("active frame disappeared".into()))?;
            if !inputs_ready_with_new_epoch(frame, node)? {
                continue;
            }
            return self.execute_node(frame_index, node.clone(), ctx).await;
        }

        let frame = self
            .state
            .frames
            .get(frame_index)
            .ok_or_else(|| GraphError::Invalid("active frame disappeared".into()))?;
        if edge_ready(frame, compiled.graph.exit)? {
            return Ok(Step::Continue);
        }
        Err(GraphError::Deadlock(describe_waiting(
            &compiled.graph,
            &compiled.instructions,
            frame,
        )))
    }

    pub(super) async fn execute_node(
        &mut self,
        frame_index: usize,
        node: CompiledNode,
        ctx: Context,
    ) -> Result<Step, GraphError> {
        match &node.kind {
            CompiledNodeKind::Builtin { op } => {
                let frame = self.frame(frame_index)?;
                let inputs = read_inputs(frame, &node)?;
                let outputs = run_builtin(op, inputs, node.outputs.len(), node.name.as_ref())?;
                self.write_outputs(frame_index, &node, outputs)?;
                self.complete_node(frame_index, &node)?;
                Ok(Step::Continue)
            }
            CompiledNodeKind::PureHandler { key } => {
                let frame = self.frame(frame_index)?;
                let inputs = read_inputs(frame, &node)?;
                let handler = self
                    .registry
                    .value(key)
                    .ok_or_else(|| GraphError::MissingHandler(key.as_str().into()))?;
                let outputs = handler.call(inputs)?;
                self.write_outputs(frame_index, &node, outputs)?;
                self.complete_node(frame_index, &node)?;
                Ok(Step::Continue)
            }
            CompiledNodeKind::WorkHandler { key } => {
                let frame = self.frame(frame_index)?;
                let inputs = read_inputs(frame, &node)?;
                let handler = self
                    .registry
                    .work(key)
                    .ok_or_else(|| GraphError::MissingHandler(key.as_str().into()))?;
                let outputs = handler.call(inputs, ctx).await?;
                self.write_outputs(frame_index, &node, outputs)?;
                self.complete_node(frame_index, &node)?;
                Ok(Step::Continue)
            }
            CompiledNodeKind::Load { var, key } => {
                let frame = self.frame(frame_index)?;
                let input = read_single_input(frame, &node)?;
                let state = self.read_variable(frame_index, *var)?;
                let handler = self
                    .registry
                    .value(key)
                    .ok_or_else(|| GraphError::MissingHandler(key.as_str().into()))?;
                let outputs = handler.call(vec![input, state])?;
                self.write_outputs(frame_index, &node, outputs)?;
                self.complete_node(frame_index, &node)?;
                Ok(Step::Continue)
            }
            CompiledNodeKind::Store { var, key } => {
                let frame = self.frame(frame_index)?;
                let input = read_single_input(frame, &node)?;
                let state = self.read_variable(frame_index, *var)?;
                let handler = self
                    .registry
                    .value(key)
                    .ok_or_else(|| GraphError::MissingHandler(key.as_str().into()))?;
                let outputs = handler.call(vec![input.clone(), state])?;
                if outputs.len() != 1 {
                    return Err(GraphError::OutputArity {
                        node: node.name.to_string(),
                        expected: 1,
                        got: outputs.len(),
                    });
                }
                let variable_value = outputs[0].clone();
                let variable = self.validate_variable_write(frame_index, *var, &variable_value)?;
                let output_edge = node.outputs.first().copied().ok_or_else(|| {
                    GraphError::Invalid(format!("store node '{}' has no output", node.name))
                })?;
                self.validate_edge_write_value(
                    frame_index,
                    output_edge,
                    &input,
                    &format!("node '{}'", node.name),
                )?;
                ensure_write_capacity(self.frame(frame_index)?, 2)?;
                self.commit_variable_write(frame_index, variable, variable_value)?;
                self.commit_edge_write(frame_index, output_edge, input)?;
                self.complete_node(frame_index, &node)?;
                Ok(Step::Continue)
            }
            CompiledNodeKind::Continuation { key, payload, .. } => {
                let frame = self.frame(frame_index)?;
                let inputs = read_inputs(frame, &node)?;
                let state = frame
                    .continuation_states
                    .get(node.id.0)
                    .cloned()
                    .ok_or(GraphError::MissingNode(node.id))?;
                let handler = self
                    .registry
                    .continuation(key)
                    .ok_or_else(|| GraphError::MissingHandler(key.as_str().into()))?;
                let ctx = self.continuation_context(ctx);
                let transition = handler.start(payload.as_ref(), state, inputs, ctx).await?;
                let suspension =
                    self.apply_continuation_transition(frame_index, &node, transition)?;
                self.complete_node(frame_index, &node)?;
                Ok(suspension.map_or(Step::Continue, Step::Suspend))
            }
            CompiledNodeKind::Suspend { payload, .. } => {
                if node.inputs.len() != 1 || node.outputs.len() != 1 {
                    return Err(GraphError::Invalid(format!(
                        "suspend node '{}' requires one input and one output",
                        node.name
                    )));
                }
                let frame = self.frame(frame_index)?;
                let input_edge = single_input_edge(&node)?;
                let input_ref = peek_edge(frame, input_edge)?;
                let payload_value = suspend_payload(payload.as_ref(), input_ref);
                let output_edge = node.outputs.first().copied().ok_or_else(|| {
                    GraphError::Invalid(format!("suspend node '{}' has no output", node.name))
                })?;
                let resume_type = self
                    .callables
                    .get(frame.graph_index)
                    .and_then(|graph| graph.graph.edge(output_edge))
                    .map(|edge| edge.type_spec.clone())
                    .ok_or_else(|| {
                        GraphError::Invalid(format!(
                            "suspend node '{}' output type is missing",
                            node.name
                        ))
                    })?;
                self.state.suspension = Some(Suspension {
                    frame_depth: frame_index + 1,
                    graph_index: frame.graph_index,
                    node: node.id,
                    target: SuspensionTarget::Node,
                    resume_type,
                    payload: payload_value.clone(),
                });
                Ok(Step::Suspend(payload_value))
            }
            CompiledNodeKind::Subflow { child_index } => {
                let parent_edge = node.outputs.first().copied().ok_or_else(|| {
                    GraphError::Invalid(format!("subflow node '{}' has no output", node.name))
                })?;
                let child_graph = self
                    .callables
                    .get(*child_index)
                    .ok_or_else(|| GraphError::Invalid("child graph index is invalid".into()))?;
                let entry = child_graph.graph.entry;
                let input_edge = single_input_edge(&node)?;
                let frame = self.frame(frame_index)?;
                let input_ref = peek_edge(frame, input_edge)?;
                validate_edge_value(&child_graph.graph, entry, input_ref, "child entry input")?;
                let input = self.read_single_input_for_handoff(frame_index, &node)?;
                let mut child = new_frame(
                    &self.callables,
                    &self.state.frames,
                    *child_index,
                    Some(ReturnTarget::Edge { parent_edge }),
                )?;
                write_edge(&mut child, entry, input)?;
                self.complete_node(frame_index, &node)?;
                self.state.frames.push(child);
                Ok(Step::Continue)
            }
            CompiledNodeKind::Either {
                key,
                left_index,
                right_index,
            } => {
                let frame = self.frame(frame_index)?;
                let input = read_single_input(frame, &node)?;
                let handler = self
                    .registry
                    .value(key)
                    .ok_or_else(|| GraphError::MissingHandler(key.as_str().into()))?;
                let outputs = handler.call(vec![input])?;
                if outputs.len() != 1 {
                    return Err(GraphError::OutputArity {
                        node: node.name.to_string(),
                        expected: 1,
                        got: outputs.len(),
                    });
                }
                let choice: BranchChoice = crate::graph::from_value(
                    outputs
                        .into_iter()
                        .next()
                        .ok_or_else(|| GraphError::Invalid("either returned no choice".into()))?,
                )
                .map_err(|err| {
                    GraphError::Invalid(format!(
                        "either node '{}' returned invalid branch choice: {err}",
                        node.name
                    ))
                })?;
                let child_index = match choice.side {
                    BranchSide::Left => *left_index,
                    BranchSide::Right => *right_index,
                };
                let mut child = new_frame(
                    &self.callables,
                    &self.state.frames,
                    child_index,
                    Some(ReturnTarget::Either {
                        parent_node: node.id,
                    }),
                )?;
                let child_graph = self.callables.get(child_index).ok_or_else(|| {
                    GraphError::Invalid("either branch graph index is invalid".into())
                })?;
                let entry = child_graph.graph.entry;
                validate_edge_value(
                    &child_graph.graph,
                    entry,
                    &choice.value,
                    "either branch input",
                )?;
                write_edge(&mut child, entry, choice.value)?;
                self.complete_node(frame_index, &node)?;
                self.state.frames.push(child);
                Ok(Step::Continue)
            }
            CompiledNodeKind::Each { child_index } => {
                let frame = self.frame(frame_index)?;
                let input = read_single_input(frame, &node)?;
                let items = input
                    .as_array()
                    .ok_or_else(|| {
                        GraphError::Invalid(format!(
                            "each node '{}' expected array input, got {input}",
                            node.name
                        ))
                    })?
                    .to_vec();
                if items.is_empty() {
                    self.write_outputs(frame_index, &node, vec![Value::array([])])?;
                    self.complete_node(frame_index, &node)?;
                    return Ok(Step::Continue);
                }
                let checkpoint = EachVmCheckpoint {
                    items,
                    outputs: Vec::new(),
                    index: 0,
                };
                self.set_node_checkpoint(frame_index, node.id, &checkpoint)?;
                self.push_each_child(frame_index, node.id, *child_index, &checkpoint)?;
                self.complete_node(frame_index, &node)?;
                Ok(Step::Continue)
            }
            CompiledNodeKind::Goto { target } => {
                let frame = self.frame(frame_index)?;
                let input = read_single_input(frame, &node)?;
                self.validate_edge_write_value(
                    frame_index,
                    *target,
                    &input,
                    &format!("goto node '{}'", node.name),
                )?;
                self.commit_edge_write(frame_index, *target, input)?;
                self.complete_node(frame_index, &node)?;
                Ok(Step::Continue)
            }
        }
    }

    pub(super) fn write_outputs(
        &mut self,
        frame_index: usize,
        node: &CompiledNode,
        outputs: Vec<Value>,
    ) -> Result<(), GraphError> {
        if outputs.len() != node.outputs.len() {
            return Err(GraphError::OutputArity {
                node: node.name.to_string(),
                expected: node.outputs.len(),
                got: outputs.len(),
            });
        }
        let writes = node
            .outputs
            .iter()
            .copied()
            .zip(outputs)
            .map(|(edge, value)| (edge, value, format!("node '{}'", node.name)))
            .collect::<Vec<_>>();
        self.validate_edge_write_plan(frame_index, &writes)?;
        self.commit_edge_write_plan(frame_index, writes)?;
        Ok(())
    }

    pub(super) fn validate_edge_write_plan(
        &self,
        frame_index: usize,
        writes: &[(EdgeId, Value, String)],
    ) -> Result<(), GraphError> {
        let mut seen = HashSet::new();
        for (edge, value, label) in writes {
            if !seen.insert(*edge) {
                return Err(GraphError::Invalid(format!(
                    "edge {:?} is written more than once in the same step",
                    edge
                )));
            }
            self.validate_edge_write_value(frame_index, *edge, value, label)?;
        }
        ensure_write_capacity(self.frame(frame_index)?, writes.len())?;
        Ok(())
    }

    pub(super) fn validate_edge_write_value(
        &self,
        frame_index: usize,
        edge: EdgeId,
        value: &Value,
        label: &str,
    ) -> Result<(), GraphError> {
        let frame = self.frame(frame_index)?;
        frame
            .values
            .get(edge.0)
            .ok_or(GraphError::MissingEdge(edge))?;
        let graph = self
            .callables
            .get(frame.graph_index)
            .ok_or_else(|| GraphError::Invalid("frame graph index is invalid".into()))?;
        validate_edge_value(&graph.graph, edge, value, label)
    }

    pub(super) fn commit_edge_write_plan(
        &mut self,
        frame_index: usize,
        writes: Vec<(EdgeId, Value, String)>,
    ) -> Result<(), GraphError> {
        for (edge, value, _) in writes {
            self.commit_edge_write(frame_index, edge, value)?;
        }
        Ok(())
    }

    pub(super) fn commit_edge_write(
        &mut self,
        frame_index: usize,
        edge: EdgeId,
        value: Value,
    ) -> Result<(), GraphError> {
        write_edge(self.frame_mut(frame_index)?, edge, value)
    }

    pub(super) fn read_single_input_for_handoff(
        &mut self,
        frame_index: usize,
        node: &CompiledNode,
    ) -> Result<Value, GraphError> {
        let edge = single_input_edge(node)?;
        read_edge(self.frame(frame_index)?, edge)
    }

    pub(super) fn read_variable(
        &self,
        frame_index: usize,
        var: VarId,
    ) -> Result<Value, GraphError> {
        self.state
            .frames
            .get(frame_index)
            .and_then(|frame| frame.variables.get(var.0))
            .and_then(|value| value.clone())
            .ok_or(GraphError::MissingVariable(var))
    }

    pub(super) fn validate_variable_write(
        &self,
        frame_index: usize,
        var: VarId,
        value: &Value,
    ) -> Result<VarId, GraphError> {
        let frame = self.frame(frame_index)?;
        let graph = self
            .callables
            .get(frame.graph_index)
            .ok_or_else(|| GraphError::Invalid("variable owner graph index is invalid".into()))?;
        let variable = graph
            .graph
            .variable(var)
            .ok_or(GraphError::MissingVariable(var))?;
        validate_value(
            &variable.type_spec,
            value,
            &format!(
                "variable '{}::{}'",
                variable.key.namespace, variable.key.type_name
            ),
        )?;
        self.state
            .frames
            .get(frame_index)
            .and_then(|frame| frame.variables.get(var.0))
            .ok_or(GraphError::MissingVariable(var))?;
        Ok(var)
    }

    pub(super) fn commit_variable_write(
        &mut self,
        frame_index: usize,
        var: VarId,
        value: Value,
    ) -> Result<(), GraphError> {
        write_variable(self.frame_mut(frame_index)?, var, value)
    }

    pub(super) fn try_exit_frames(&mut self) -> Result<Step, GraphError> {
        loop {
            let Some(frame) = self.state.frames.last() else {
                return Err(GraphError::Invalid("runtime has no active frame".into()));
            };
            let graph = self
                .callables
                .get(frame.graph_index)
                .ok_or_else(|| GraphError::Invalid("frame graph index is invalid".into()))?;
            let exit = graph.graph.exit;
            if !edge_ready(frame, exit)? {
                return Ok(Step::Continue);
            }
            let mut frame = self
                .state
                .frames
                .pop()
                .ok_or_else(|| GraphError::Invalid("frame stack unexpectedly empty".into()))?;
            let output = take_edge(&mut frame, exit)?;
            let return_target = frame.return_target;
            if let Some(target) = return_target {
                let parent_index = self
                    .state
                    .frames
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| GraphError::Invalid("child frame has no parent".into()))?;
                match target {
                    ReturnTarget::Edge { parent_edge } => {
                        let parent_graph_index = self.frame(parent_index)?.graph_index;
                        let parent_graph =
                            self.callables.get(parent_graph_index).ok_or_else(|| {
                                GraphError::Invalid("parent graph index is invalid".into())
                            })?;
                        validate_edge_value(
                            &parent_graph.graph,
                            parent_edge,
                            &output,
                            "subflow return",
                        )?;
                        let parent = self.frame_mut(parent_index)?;
                        write_edge(parent, parent_edge, output)?;
                    }
                    ReturnTarget::Either { parent_node } => {
                        self.complete_either_child(parent_index, parent_node, output)?;
                    }
                    ReturnTarget::Each { parent_node } => {
                        self.complete_each_child(parent_index, parent_node, output)?;
                    }
                    ReturnTarget::Continuation {
                        parent_node,
                        call_id,
                    } => {
                        self.complete_continuation_child(
                            parent_index,
                            parent_node,
                            call_id,
                            output,
                        )?;
                    }
                }
            } else {
                if !self.state.frames.is_empty() {
                    return Err(GraphError::Invalid(
                        "root frame exited while parent frames remain".into(),
                    ));
                }
                return Ok(Step::Done(output));
            }
        }
    }

    pub(super) fn complete_either_child(
        &mut self,
        parent_index: usize,
        parent_node: NodeId,
        output: Value,
    ) -> Result<(), GraphError> {
        let node = self.compiled_node(parent_index, parent_node)?.clone();
        self.write_outputs(parent_index, &node, vec![output])
    }

    pub(super) fn complete_each_child(
        &mut self,
        parent_index: usize,
        parent_node: NodeId,
        output: Value,
    ) -> Result<(), GraphError> {
        let node = self.compiled_node(parent_index, parent_node)?.clone();
        let mut checkpoint = self.take_each_checkpoint(parent_index, parent_node)?;
        checkpoint.outputs.push(output);
        checkpoint.index = checkpoint
            .index
            .checked_add(1)
            .ok_or_else(|| GraphError::Invalid("each checkpoint index overflowed".into()))?;
        if checkpoint.index >= checkpoint.items.len() {
            self.write_outputs(parent_index, &node, vec![Value::array(checkpoint.outputs)])?;
            return Ok(());
        }
        let CompiledNodeKind::Each { child_index } = &node.kind else {
            return Err(GraphError::Invalid(format!(
                "node '{}' is not an each node",
                node.name
            )));
        };
        self.set_node_checkpoint(parent_index, parent_node, &checkpoint)?;
        self.push_each_child(parent_index, parent_node, *child_index, &checkpoint)
    }

    pub(super) fn complete_continuation_child(
        &mut self,
        parent_index: usize,
        parent_node: NodeId,
        call_id: String,
        output: Value,
    ) -> Result<(), GraphError> {
        let slot = self
            .state
            .frames
            .get_mut(parent_index)
            .and_then(|frame| frame.continuation_inboxes.get_mut(parent_node.0))
            .ok_or(GraphError::MissingNode(parent_node))?;
        slot.push(ContinuationChildResult { call_id, output });
        Ok(())
    }

    pub(super) fn compiled_node(
        &self,
        frame_index: usize,
        node: NodeId,
    ) -> Result<&CompiledNode, GraphError> {
        let graph_index = self.frame(frame_index)?.graph_index;
        let graph = self
            .callables
            .get(graph_index)
            .ok_or_else(|| GraphError::Invalid("parent graph index is invalid".into()))?;
        graph
            .nodes
            .get(node.0)
            .filter(|item| item.id == node)
            .ok_or(GraphError::MissingNode(node))
    }

    pub(super) fn set_node_checkpoint<TValue: Serialize>(
        &mut self,
        frame_index: usize,
        node: NodeId,
        value: &TValue,
    ) -> Result<(), GraphError> {
        let value = crate::graph::to_value(value).map_err(|err| {
            GraphError::Invalid(format!("failed to encode node  checkpoint: {err}"))
        })?;
        let slot = self
            .state
            .frames
            .get_mut(frame_index)
            .and_then(|frame| frame.checkpoints.get_mut(node.0))
            .ok_or(GraphError::MissingNode(node))?;
        *slot = Some(value);
        Ok(())
    }

    pub(super) fn take_each_checkpoint(
        &mut self,
        frame_index: usize,
        node: NodeId,
    ) -> Result<EachVmCheckpoint, GraphError> {
        let value = self
            .state
            .frames
            .get_mut(frame_index)
            .and_then(|frame| frame.checkpoints.get_mut(node.0))
            .and_then(Option::take)
            .ok_or_else(|| GraphError::Invalid("each checkpoint is missing".into()))?;
        crate::graph::from_value(value)
            .map_err(|err| GraphError::Invalid(format!("failed to decode each  checkpoint: {err}")))
    }

    pub(super) fn push_each_child(
        &mut self,
        parent_index: usize,
        parent_node: NodeId,
        child_index: usize,
        checkpoint: &EachVmCheckpoint,
    ) -> Result<(), GraphError> {
        let item = checkpoint
            .items
            .get(checkpoint.index)
            .cloned()
            .ok_or_else(|| GraphError::Invalid("each index is out of range".into()))?;
        let mut child = new_frame(
            &self.callables,
            &self.state.frames,
            child_index,
            Some(ReturnTarget::Each { parent_node }),
        )?;
        let child_graph = self
            .callables
            .get(child_index)
            .ok_or_else(|| GraphError::Invalid("each child graph index is invalid".into()))?;
        let entry = child_graph.graph.entry;
        validate_edge_value(&child_graph.graph, entry, &item, "each item input")?;
        write_edge(&mut child, entry, item)?;
        if parent_index + 1 != self.state.frames.len() {
            return Err(GraphError::Invalid(
                "each parent is not at the top of the frame stack".into(),
            ));
        }
        self.state.frames.push(child);
        Ok(())
    }

    pub(super) fn frame(&self, frame_index: usize) -> Result<&Frame, GraphError> {
        self.state
            .frames
            .get(frame_index)
            .ok_or_else(|| GraphError::Invalid(format!("frame index {frame_index} is invalid")))
    }

    pub(super) fn frame_mut(&mut self, frame_index: usize) -> Result<&mut Frame, GraphError> {
        self.state
            .frames
            .get_mut(frame_index)
            .ok_or_else(|| GraphError::Invalid(format!("frame index {frame_index} is invalid")))
    }
}
