use super::*;

impl Runtime {
    pub(super) fn poll_continuation(
        &mut self,
        frame_index: usize,
        node: CompiledNode,
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
        let ctx = self.continuation_context(&node);
        let transition = handler.advance(payload.as_ref(), checkpoint, event, ctx)?;
        let node_id = node.id;
        let suspension = self.apply_continuation_transition(frame_index, &node, transition)?;
        self.consume_continuation_event(frame_index, node_id)?;
        Ok(suspension)
    }

    pub(super) fn apply_continuation_transition(
        &mut self,
        frame_index: usize,
        node: &CompiledNode,
        transition: ContinuationTransition,
    ) -> Result<Step, GraphError> {
        let ContinuationTransition {
            checkpoint,
            state,
            outputs,
            writes,
            child_calls,
            suspension,
            agent,
            history,
        } = transition;
        if agent.is_some()
            && (checkpoint.is_none()
                || suspension.is_some()
                || !outputs.is_empty()
                || !child_calls.is_empty()
                || !writes.is_empty())
        {
            return Err(GraphError::InvalidContinuationTransition {
                node: node.name.to_string(),
                reason: "agent request requires an exclusive checkpointed external boundary".into(),
            });
        }
        let agent = agent
            .map(|request| self.prepare_agent(request))
            .transpose()?;
        let history = self.prepare_history_changes(frame_index, history)?;
        let has_outputs = !outputs.is_empty();
        let has_checkpoint = checkpoint.is_some();
        let has_child_calls = !child_calls.is_empty();
        let has_suspension = suspension.is_some();

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
        if has_suspension && !has_checkpoint {
            return Err(GraphError::InvalidContinuationTransition {
                node: node.name.to_string(),
                reason: "external suspension requires checkpoint state".into(),
            });
        }
        if has_suspension && (has_outputs || has_child_calls || !writes.is_empty()) {
            return Err(GraphError::InvalidContinuationTransition {
                node: node.name.to_string(),
                reason:
                    "external suspension cannot be combined with outputs, writes, or child calls"
                        .into(),
            });
        }
        if let Some(suspension) = &suspension {
            validate_continuation_suspension(node, suspension)?;
        }
        self.validate_continuation_child_calls(frame_index, node, &child_calls)?;
        let prepared_child = if agent.is_none() && !has_suspension {
            self.prepare_next_continuation_child_call(frame_index, node, &child_calls)?
        } else {
            None
        };
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
        let payload = if let Some(suspension) = suspension {
            let payload = suspension.payload.clone();
            let graph_index = self
                .state
                .frames
                .get(frame_index)
                .ok_or_else(|| GraphError::Invalid("continuation frame disappeared".into()))?
                .graph_index;
            self.state.waiting = Some(Waiting::Suspend(Suspension {
                frame_depth: frame_index + 1,
                graph_index,
                node: node.id,
                target: SuspensionTarget::Continuation,
                resume_type: Arc::new(suspension.resume_type),
                payload: suspension.payload,
            }));
            Some(payload)
        } else {
            None
        };
        self.commit_history_changes(frame_index, history)?;
        if let Some(agent) = agent {
            return Ok(self.commit_agent(frame_index, node.id, agent));
        }
        Ok(payload.map_or(Step::Continue, Step::Suspend))
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
            Ok(match result {
                ContinuationInput::Child { call_id, output } => ContinuationEvent::ChildResult {
                    call_id: call_id.clone(),
                    output: output.clone(),
                },
                ContinuationInput::Resume { input } => ContinuationEvent::Resume {
                    input: input.clone(),
                },
                ContinuationInput::Agent { request, response } => ContinuationEvent::Agent {
                    request: request.clone(),
                    response: response.clone(),
                },
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

    /// Resumes a first-class suspend node after validating its output edge.
    pub(super) fn resume_suspend_node(
        &mut self,
        frame_index: usize,
        node: &CompiledNode,
        value: Value,
    ) -> Result<(), GraphError> {
        let CompiledNodeKind::Suspend { .. } = &node.kind else {
            return Err(GraphError::Invalid(format!(
                "suspended node '{}' is not a suspend node",
                node.name
            )));
        };
        let output_edge = node.outputs.first().copied().ok_or_else(|| {
            GraphError::Invalid(format!("suspend node '{}' has no output", node.name))
        })?;
        self.validate_edge_write_value(
            frame_index,
            output_edge,
            &value,
            &format!("suspend node '{}'", node.name),
        )?;
        self.commit_edge_write(frame_index, output_edge, value)?;
        self.complete_node(frame_index, node)?;
        self.state.waiting = None;
        Ok(())
    }

    /// Accepts input into the existing inbox without executing the owning continuation.
    pub(super) fn resume_continuation(
        &mut self,
        index: usize,
        node: CompiledNode,
        input: Value,
    ) -> Result<(), GraphError> {
        let frame = self
            .state
            .frames
            .get_mut(index)
            .ok_or(GraphError::MissingNode(node.id))?;
        if frame.checkpoints.get(node.id.0).is_none_or(Option::is_none) {
            return Err(GraphError::SnapshotValidation(
                "suspended continuation has no checkpoint".into(),
            ));
        }
        let inbox = frame
            .continuation_inboxes
            .get_mut(node.id.0)
            .ok_or(GraphError::MissingNode(node.id))?;
        if !inbox.is_empty() {
            return Err(GraphError::SnapshotValidation(
                "suspended continuation has an accepted input".into(),
            ));
        }
        inbox.push(ContinuationInput::Resume { input });
        self.state.waiting = None;
        Ok(())
    }
}

fn validate_continuation_suspension(
    node: &CompiledNode,
    suspension: &crate::graph::registry::ContinuationSuspension,
) -> Result<(), GraphError> {
    if suspension.resume_type.name.trim().is_empty() {
        return Err(GraphError::InvalidContinuationTransition {
            node: node.name.to_string(),
            reason: "external suspension resume type is empty".into(),
        });
    }
    jsonschema::validator_for(&suspension.resume_type.schema).map_err(|error| {
        GraphError::InvalidContinuationTransition {
            node: node.name.to_string(),
            reason: format!("external suspension resume schema is invalid: {error}"),
        }
    })?;
    Ok(())
}
