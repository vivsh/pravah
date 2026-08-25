use super::*;

impl Runtime {
    pub(super) async fn poll_continuation(
        &mut self,
        frame_index: usize,
        node: CompiledNode,
        ctx: Context,
    ) -> Result<Step, GraphError> {
        let CompiledNodeKind::Continuation { key, payload, .. } = &node.kind else {
            return Err(GraphError::Invalid(format!(
                "node '{}' is not a continuation",
                node.name
            )));
        };
        let event = self.peek_continuation_event(frame_index, node.id)?;
        let checkpoint = self
            .state
            .frames
            .get(frame_index)
            .and_then(|frame| frame.checkpoints.get(node.id.0))
            .and_then(Clone::clone)
            .ok_or_else(|| GraphError::Invalid("continuation checkpoint disappeared".into()))?;
        let handler = self
            .registry
            .continuation(key)
            .ok_or_else(|| GraphError::MissingHandler(key.as_str().into()))?;
        let ctx = self.continuation_context(ctx);
        let transition = handler
            .advance(payload.as_ref(), checkpoint, event, ctx)
            .await?;
        let node_id = node.id;
        self.apply_continuation_transition(frame_index, &node, transition)?;
        self.consume_continuation_event(frame_index, node_id)?;
        Ok(Step::Continue)
    }

    pub(super) fn apply_continuation_transition(
        &mut self,
        frame_index: usize,
        node: &CompiledNode,
        transition: ContinuationTransition,
    ) -> Result<(), GraphError> {
        let ContinuationTransition {
            checkpoint,
            state,
            outputs,
            writes,
            child_calls,
        } = transition;
        let has_outputs = !outputs.is_empty();
        let has_checkpoint = checkpoint.is_some();
        let has_child_calls = !child_calls.is_empty();

        if has_outputs && has_checkpoint {
            return Err(GraphError::InvalidContinuationTransition {
                node: node.name.to_string(),
                reason: "completion outputs cannot be combined with checkpoint state".into(),
            });
        }
        if has_outputs && has_child_calls {
            return Err(GraphError::InvalidContinuationTransition {
                node: node.name.to_string(),
                reason: "completion outputs cannot be combined with child calls".into(),
            });
        }
        if has_child_calls && !has_checkpoint {
            return Err(GraphError::InvalidContinuationTransition {
                node: node.name.to_string(),
                reason: "child calls require checkpoint state".into(),
            });
        }
        self.validate_continuation_child_calls(frame_index, node, &child_calls)?;
        let prepared_child =
            self.prepare_next_continuation_child_call(frame_index, node, &child_calls)?;
        let completing = has_outputs;
        let mut edge_writes = writes
            .into_iter()
            .map(|write| (write.edge, write.value, "continuation write".to_string()))
            .collect::<Vec<_>>();
        if has_outputs {
            if outputs.len() != node.outputs.len() {
                return Err(GraphError::OutputArity {
                    node: node.name.to_string(),
                    expected: node.outputs.len(),
                    got: outputs.len(),
                });
            }
            edge_writes.extend(
                node.outputs
                    .iter()
                    .copied()
                    .zip(outputs)
                    .map(|(edge, value)| (edge, value, format!("node '{}'", node.name))),
            );
        }
        self.validate_edge_write_plan(frame_index, &edge_writes)?;
        self.commit_edge_write_plan(frame_index, edge_writes)?;
        {
            let state_slot = self
                .state
                .frames
                .get_mut(frame_index)
                .and_then(|frame| frame.continuation_states.get_mut(node.id.0))
                .ok_or(GraphError::MissingNode(node.id))?;
            if let Some(state) = state {
                *state_slot = Some(state);
            } else if completing {
                *state_slot = None;
            }
        }
        let slot = self
            .state
            .frames
            .get_mut(frame_index)
            .and_then(|frame| frame.checkpoints.get_mut(node.id.0))
            .ok_or(GraphError::MissingNode(node.id))?;
        if let Some(checkpoint) = checkpoint {
            *slot = Some(checkpoint);
        } else if completing {
            *slot = None;
        }
        self.queue_continuation_child_calls(frame_index, node.id, child_calls)?;
        if let Some(prepared_child) = prepared_child {
            self.push_prepared_continuation_child_call(frame_index, node.id, prepared_child)?;
        }
        Ok(())
    }

    pub(super) fn peek_continuation_event(
        &self,
        frame_index: usize,
        node: NodeId,
    ) -> Result<ContinuationEvent, GraphError> {
        let inbox = self
            .state
            .frames
            .get(frame_index)
            .and_then(|frame| frame.continuation_inboxes.get(node.0))
            .ok_or(GraphError::MissingNode(node))?;
        if inbox.is_empty() {
            Ok(ContinuationEvent::Poll)
        } else {
            let result = inbox
                .first()
                .ok_or_else(|| GraphError::Invalid("continuation inbox disappeared".into()))?;
            Ok(ContinuationEvent::ChildResult {
                call_id: result.call_id.clone(),
                output: result.output.clone(),
            })
        }
    }

    pub(super) fn consume_continuation_event(
        &mut self,
        frame_index: usize,
        node: NodeId,
    ) -> Result<(), GraphError> {
        let inbox = self
            .state
            .frames
            .get_mut(frame_index)
            .and_then(|frame| frame.continuation_inboxes.get_mut(node.0))
            .ok_or(GraphError::MissingNode(node))?;
        if !inbox.is_empty() {
            inbox.remove(0);
        }
        Ok(())
    }

    pub(super) fn queue_continuation_child_calls(
        &mut self,
        frame_index: usize,
        node: NodeId,
        calls: Vec<ContinuationChildCall>,
    ) -> Result<(), GraphError> {
        if calls.is_empty() {
            return Ok(());
        }
        let queue = self
            .state
            .frames
            .get_mut(frame_index)
            .and_then(|frame| frame.continuation_child_queues.get_mut(node.0))
            .ok_or(GraphError::MissingNode(node))?;
        queue.extend(calls);
        Ok(())
    }

    pub(super) fn validate_continuation_child_calls(
        &self,
        _parent_index: usize,
        node: &CompiledNode,
        calls: &[ContinuationChildCall],
    ) -> Result<(), GraphError> {
        if calls.is_empty() {
            return Ok(());
        }
        let CompiledNodeKind::Continuation { children, .. } = &node.kind else {
            return Err(GraphError::Invalid(format!(
                "node '{}' cannot request continuation child calls",
                node.name
            )));
        };
        for call in calls {
            if children.get(call.child_index).is_none() {
                return Err(GraphError::Invalid(format!(
                    "continuation node '{}' requested missing child index {}",
                    node.name, call.child_index
                )));
            }
        }
        Ok(())
    }

    pub(super) fn prepare_next_continuation_child_call(
        &self,
        parent_index: usize,
        node: &CompiledNode,
        new_calls: &[ContinuationChildCall],
    ) -> Result<Option<PreparedContinuationChild>, GraphError> {
        let call = self
            .state
            .frames
            .get(parent_index)
            .and_then(|frame| frame.continuation_child_queues.get(node.id.0))
            .and_then(|queue| queue.first())
            .cloned()
            .or_else(|| new_calls.first().cloned());
        let Some(call) = call else {
            return Ok(None);
        };
        if parent_index + 1 != self.state.frames.len() {
            return Err(GraphError::Invalid(
                "continuation parent is not at the top of the frame stack".into(),
            ));
        }
        let child_index =
            self.continuation_child_graph_index(parent_index, node, call.child_index)?;
        let mut child = new_frame(
            &self.callables,
            &self.state.frames,
            child_index,
            Some(ReturnTarget::Continuation {
                parent_node: node.id,
                call_id: call.call_id,
            }),
        )?;
        let child_graph = self.callables.get(child_index).ok_or_else(|| {
            GraphError::Invalid("continuation child graph index is invalid".into())
        })?;
        let entry = child_graph.graph.entry;
        validate_edge_value(
            &child_graph.graph,
            entry,
            &call.input,
            "continuation child input",
        )?;
        write_edge(&mut child, entry, call.input)?;
        Ok(Some(PreparedContinuationChild { frame: child }))
    }

    pub(super) fn continuation_child_graph_index(
        &self,
        parent_index: usize,
        node: &CompiledNode,
        child_call_index: usize,
    ) -> Result<usize, GraphError> {
        let _ = self.frame(parent_index)?;
        let CompiledNodeKind::Continuation { children, .. } = &node.kind else {
            return Err(GraphError::Invalid(format!(
                "node '{}' cannot request continuation child calls",
                node.name
            )));
        };
        children.get(child_call_index).copied().ok_or_else(|| {
            GraphError::Invalid(format!(
                "continuation node '{}' requested missing child index {}",
                node.name, child_call_index
            ))
        })
    }

    pub(super) fn push_prepared_continuation_child_call(
        &mut self,
        parent_index: usize,
        node: NodeId,
        prepared: PreparedContinuationChild,
    ) -> Result<(), GraphError> {
        if parent_index + 1 != self.state.frames.len() {
            return Err(GraphError::Invalid(
                "continuation parent is not at the top of the frame stack".into(),
            ));
        }
        let queue = self
            .state
            .frames
            .get_mut(parent_index)
            .and_then(|frame| frame.continuation_child_queues.get_mut(node.0))
            .ok_or(GraphError::MissingNode(node))?;
        if queue.is_empty() {
            return Err(GraphError::Invalid(
                "continuation child call queue disappeared".into(),
            ));
        }
        queue.remove(0);
        self.state.frames.push(prepared.frame);
        Ok(())
    }
}
