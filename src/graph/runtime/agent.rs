//! Atomic installation and delivery of durable agent operations; no external code runs here.

use super::*;
use crate::graph::agent_request::{AGENT_REQUEST_VERSION, AgentOperation};

impl Runtime {
    /// Borrows the exact durable work waiting for its worker completion.
    pub fn pending_agent(&self) -> Option<&AgentRequest> {
        match &self.state.waiting {
            Some(Waiting::Agent { request, .. }) => Some(request),
            _ => None,
        }
    }

    /// Accepts one completion and its acknowledged stages without executing the owner.
    /// Invalid identities, acknowledgements or history replacements leave all state unchanged.
    pub fn resume_agent(&mut self, mut response: AgentResponse) -> Result<(), GraphError> {
        let Some(Waiting::Agent {
            frame_depth,
            node,
            request,
        }) = &self.state.waiting
        else {
            return Err(GraphError::AgentRequestValidation(
                "no agent request is pending".into(),
            ));
        };
        validate_response(request, &response)?;
        let index = frame_depth
            .checked_sub(1)
            .ok_or_else(|| GraphError::AgentRequestValidation("invalid owner depth".into()))?;
        self.validate_agent_owner(index, *node)?;
        let history = self.prepare_agent_history(request, &response)?;
        let inbox = self
            .state
            .frames
            .get_mut(index)
            .and_then(|frame| frame.continuation_inboxes.get_mut(node.0))
            .ok_or(GraphError::MissingNode(*node))?;
        if !inbox.is_empty() {
            return Err(GraphError::AgentRequestValidation(
                "owner already has accepted input".into(),
            ));
        }
        let node = *node;
        let Some(Waiting::Agent { request, .. }) = self.state.waiting.take() else {
            return Err(GraphError::Invalid(
                "validated agent wait disappeared".into(),
            ));
        };
        self.commit_agent_history(history, &response);
        response.loaded = None;
        response.compaction = None;
        response.persisted_through = None;
        self.state
            .frames
            .get_mut(index)
            .and_then(|frame| frame.continuation_inboxes.get_mut(node.0))
            .ok_or(GraphError::MissingNode(node))?
            .push(ContinuationInput::Agent { request, response });
        Ok(())
    }

    /// Resolves deterministic identity and accepted persistence rows before transition commit.
    pub(super) fn prepare_agent(
        &self,
        mut request: AgentRequest,
    ) -> Result<(AgentRequest, u64), GraphError> {
        if self.state.waiting.is_some() {
            return Err(GraphError::ResumeRequired);
        }
        let sequence = self
            .state
            .next_agent_sequence
            .checked_add(1)
            .ok_or_else(|| {
                GraphError::AgentRequestValidation("agent request sequence exhausted".into())
            })?;
        if self.state.history_policy.persist {
            let mut entries = self
                .history
                .entries()
                .iter()
                .filter(|entry| {
                    entry.position >= self.state.persisted_history_position
                        && entry.agent_id != crate::history::SUMMARY_AGENT_ID
                })
                .cloned()
                .collect::<Vec<_>>();
            if let Some(staged) = &request.persist {
                entries.extend(staged.iter().cloned());
            }
            request.persist = (!entries.is_empty()).then(|| entries.into());
        }
        Ok((
            request.with_id(agent_uuid(
                self.state.execution_id,
                self.state.next_agent_sequence,
            )),
            sequence,
        ))
    }

    /// Installs the one external wait only after the owning transition is validated.
    pub(super) fn commit_agent(
        &mut self,
        frame_index: usize,
        node: NodeId,
        prepared: (AgentRequest, u64),
    ) -> Step {
        let (request, sequence) = prepared;
        self.state.next_agent_sequence = sequence;
        self.state.waiting = Some(Waiting::Agent {
            frame_depth: frame_index + 1,
            node,
            request: request.clone(),
        });
        Step::Agent(request)
    }

    /// Checks an external operation can only belong to the active checkpointed continuation.
    fn validate_agent_owner(&self, index: usize, node: NodeId) -> Result<(), GraphError> {
        let frame = self.frame(index)?;
        let compiled = self
            .callables
            .get(frame.graph_index)
            .and_then(|graph| graph.nodes.get(node.0))
            .ok_or(GraphError::MissingNode(node))?;
        if index.checked_add(1) != Some(self.state.frames.len())
            || !matches!(compiled.kind, CompiledNodeKind::Continuation { .. })
            || frame.checkpoints.get(node.0).is_none_or(Option::is_none)
        {
            return Err(GraphError::AgentRequestValidation(
                "invalid agent request owner".into(),
            ));
        }
        Ok(())
    }
}

/// Derives request identities without randomness or allocation in the VM.
pub(super) fn agent_uuid(execution: Uuid, sequence: u64) -> Uuid {
    let mut name = *b"pravah.agent.v1\0\0\0\0\0\0\0\0";
    let offset = name.len() - 8;
    name[offset..].copy_from_slice(&sequence.to_be_bytes());
    Uuid::new_v5(&execution, &name)
}

/// Validates protocol identity and normalized generation structure at the external delivery boundary.
fn validate_response(request: &AgentRequest, response: &AgentResponse) -> Result<(), GraphError> {
    if response.version != AGENT_REQUEST_VERSION || response.id() != request.id() {
        return Err(GraphError::AgentRequestValidation(
            "agent completion version or identity mismatch".into(),
        ));
    }
    if let AgentOperation::Generate { .. } = request.operation.as_ref()
        && let Ok(value) = response.outcome()
    {
        <crate::graph::agent_request::client_response::Response<serde::de::IgnoredAny> as Deserialize>::deserialize(value)
            .map_err(|_| GraphError::AgentRequestValidation("invalid generation completion".into()))?;
    }
    Ok(())
}

/// Checks pending and already accepted request ownership before a restored runtime is exposed.
pub(super) fn validate_agent_state(
    callables: &[CompiledGraph],
    state: &State,
    history: &MessageHistory,
) -> Result<(), GraphError> {
    let mut accepted = 0;
    for (depth, frame) in state.frames.iter().enumerate() {
        for (node, inbox) in frame.continuation_inboxes.iter().enumerate() {
            for input in inbox {
                if let ContinuationInput::Agent { request, response } = input {
                    accepted += 1;
                    validate_identity(state, request)?;
                    validate_response(request, response)?;
                    validate_owner(callables, state, depth + 1, NodeId(node), request)?;
                    if let Some(CompiledNode {
                        kind: CompiledNodeKind::Continuation { payload, .. },
                        ..
                    }) = callables
                        .get(frame.graph_index)
                        .and_then(|graph| graph.nodes.get(node))
                    {
                        crate::graph::agent::validate_json_outcome(payload, response)?;
                    }
                }
            }
        }
    }
    if accepted > 1 || (accepted > 0 && state.waiting.is_some()) {
        return Err(GraphError::SnapshotValidation(
            "multiple agent operation owners".into(),
        ));
    }
    if let Some(Waiting::Agent {
        frame_depth,
        node,
        request,
    }) = &state.waiting
    {
        validate_identity(state, request)?;
        validate_owner(callables, state, *frame_depth, *node, request)?;
        validate_pending_history(request, history)?;
    }
    Ok(())
}

/// Pending history inputs must be the exact accepted rows owned by this runtime, not substituted messages.
fn validate_pending_history(
    request: &AgentRequest,
    history: &MessageHistory,
) -> Result<(), GraphError> {
    let expected = match request.operation.as_ref() {
        AgentOperation::Generate {
            session_id,
            entries,
            ..
        } => Some((entries.as_slice(), history.session_entries(session_id))),
        _ => None,
    };
    if let Some((entries, current)) = expected {
        if entries.len() != current.len() {
            return Err(GraphError::SnapshotValidation(
                "generation history differs from runtime".into(),
            ));
        }
        for (entry, current) in entries.iter().zip(current) {
            validate_history_row(entry, current)?;
        }
    }
    for entry in request.persist.as_deref().unwrap_or_default() {
        let current = history
            .entries()
            .iter()
            .find(|current| current.id == entry.id)
            .ok_or_else(|| {
                GraphError::SnapshotValidation("persistence row is absent from runtime".into())
            })?;
        validate_history_row(entry, current)?;
    }
    Ok(())
}

/// Full message fidelity is checked only at snapshot restoration, outside steady-state stepping.
fn validate_history_row(
    entry: &crate::HistoryEntry,
    current: &crate::HistoryEntry,
) -> Result<(), GraphError> {
    let matches = entry.id == current.id
        && entry.position == current.position
        && entry.session_id == current.session_id
        && entry.agent_id == current.agent_id
        && entry.evicted == current.evicted;
    let same_message = crate::graph::to_value(&entry.message)
        .map_err(|error| GraphError::SnapshotValidation(error.to_string()))?
        == crate::graph::to_value(&current.message)
            .map_err(|error| GraphError::SnapshotValidation(error.to_string()))?;
    if !matches || !same_message {
        return Err(GraphError::SnapshotValidation(
            "agent request history row differs from runtime".into(),
        ));
    }
    Ok(())
}

fn validate_identity(state: &State, request: &AgentRequest) -> Result<(), GraphError> {
    let sequence = state
        .next_agent_sequence
        .checked_sub(1)
        .ok_or_else(|| GraphError::SnapshotValidation("agent request without a sequence".into()))?;
    if request.id() != agent_uuid(state.execution_id, sequence) {
        return Err(GraphError::SnapshotValidation(
            "invalid deterministic agent request identity".into(),
        ));
    }
    Ok(())
}

fn validate_owner(
    callables: &[CompiledGraph],
    state: &State,
    depth: usize,
    node: NodeId,
    request: &AgentRequest,
) -> Result<(), GraphError> {
    if depth == 0 || depth != state.frames.len() {
        return Err(GraphError::SnapshotValidation(
            "invalid agent owner depth".into(),
        ));
    }
    let frame = state
        .frames
        .last()
        .ok_or_else(|| GraphError::SnapshotValidation("missing agent frame".into()))?;
    let owner = callables
        .get(frame.graph_index)
        .and_then(|graph| graph.nodes.get(node.0))
        .ok_or(GraphError::MissingNode(node))?;
    if !matches!(owner.kind, CompiledNodeKind::Continuation { .. })
        || frame.checkpoints.get(node.0).is_none_or(Option::is_none)
    {
        return Err(GraphError::SnapshotValidation(
            "invalid checkpointed agent owner".into(),
        ));
    }
    validate_owned_operation(owner, frame, request, state.execution_id)
}

/// Checks the restored request against its immutable payload and retained invocation input.
fn validate_owned_operation(
    owner: &CompiledNode,
    frame: &Frame,
    request: &AgentRequest,
    execution_id: Uuid,
) -> Result<(), GraphError> {
    if let CompiledNodeKind::Continuation {
        payload,
        output_validator,
        ..
    } = &owner.kind
    {
        let checkpoint = frame
            .checkpoints
            .get(owner.id.0)
            .and_then(Option::as_ref)
            .ok_or_else(|| GraphError::SnapshotValidation("missing agent checkpoint".into()))?;
        crate::graph::agent::validate_operation(
            payload,
            checkpoint,
            request,
            execution_id,
            output_validator.as_deref(),
            owner
                .inputs
                .first()
                .and_then(|id| frame.values.get(id.0))
                .and_then(Option::as_ref),
        )?;
    }
    Ok(())
}
