//! Pending-request creation and atomic outcome acceptance; no external code runs here.

use super::*;
use crate::graph::from_value;

impl Runtime {
    /// Borrows the durable request currently awaiting an external outcome.
    pub fn pending_fetch(&self) -> Option<&Fetch> {
        match &self.state.waiting {
            Some(Waiting::Fetch { fetch, .. }) => Some(fetch),
            _ => None,
        }
    }

    /// Accepts one outcome without advancing the requesting continuation.
    /// Wrong identities, targets or malformed protocol envelopes leave the wait unchanged.
    pub fn resume_fetch(
        &mut self,
        id: Uuid,
        outcome: Result<FetchResponse, FetchError>,
    ) -> Result<(), GraphError> {
        let Some(Waiting::Fetch {
            frame_depth,
            node,
            fetch,
        }) = &self.state.waiting
        else {
            return Err(GraphError::FetchValidation("no Fetch is pending".into()));
        };
        if id != fetch.id() {
            return Err(GraphError::FetchValidation(
                "Fetch identity does not match".into(),
            ));
        }
        let index = frame_depth
            .checked_sub(1)
            .ok_or_else(|| GraphError::FetchValidation("invalid Fetch frame".into()))?;
        let node = self.fetch_node(index, *node)?;
        validate_outcome(fetch, &outcome)?;
        match node.kind {
            CompiledNodeKind::Fetch => {
                let value = to_value(outcome).map_err(|err| GraphError::ValueConversion {
                    target: "Fetch outcome".into(),
                    reason: err.to_string(),
                })?;
                self.write_outputs(index, &node, vec![value])?;
                self.complete_node(index, &node)?;
                self.state.waiting = None;
            }
            CompiledNodeKind::Continuation { .. } => {
                let inbox = self
                    .state
                    .frames
                    .get_mut(index)
                    .and_then(|frame| frame.continuation_inboxes.get_mut(node.id.0))
                    .ok_or(GraphError::MissingNode(node.id))?;
                if !inbox.is_empty() {
                    return Err(GraphError::FetchValidation(
                        "Fetch owner has another accepted input".into(),
                    ));
                }
                let Some(Waiting::Fetch { fetch, .. }) = self.state.waiting.take() else {
                    return Err(GraphError::Invalid("validated Fetch disappeared".into()));
                };
                inbox.push(ContinuationInput::Fetch { fetch, outcome });
            }
            _ => return Err(GraphError::FetchValidation("node cannot own Fetch".into())),
        }
        Ok(())
    }

    /// Constructs an operation-local request without consuming its sequence before commit.
    pub(super) fn prepare_fetch(&self, request: FetchRequest) -> Result<(Fetch, u64), GraphError> {
        if self.state.waiting.is_some() {
            return Err(GraphError::ResumeRequired);
        }
        validate_request(&request)?;
        let next_sequence = self
            .state
            .next_fetch_sequence
            .checked_add(1)
            .ok_or_else(|| GraphError::FetchValidation("Fetch sequence exhausted".into()))?;
        Ok((
            Fetch::new(
                fetch_uuid(self.state.execution_id, self.state.next_fetch_sequence),
                Arc::new(request),
            ),
            next_sequence,
        ))
    }

    pub(super) fn commit_fetch(
        &mut self,
        frame_index: usize,
        node: NodeId,
        prepared: (Fetch, u64),
    ) -> Step {
        let (fetch, next_sequence) = prepared;
        self.state.next_fetch_sequence = next_sequence;
        self.state.waiting = Some(Waiting::Fetch {
            frame_depth: frame_index + 1,
            node,
            fetch: fetch.clone(),
        });
        Step::Fetch(fetch)
    }

    pub(super) fn execute_fetch(
        &mut self,
        index: usize,
        node: &CompiledNode,
    ) -> Result<Step, GraphError> {
        let input = read_single_input(self.frame(index)?, node)?;
        let request = from_value(input)
            .map_err(|_| GraphError::FetchValidation("invalid Fetch request input".into()))?;
        let fetch = self.prepare_fetch(request)?;
        Ok(self.commit_fetch(index, node.id, fetch))
    }

    /// Resolves only the active frame and validates the pending owner's state before delivery.
    fn fetch_node(&self, index: usize, node: NodeId) -> Result<CompiledNode, GraphError> {
        if index.checked_add(1) != Some(self.state.frames.len()) {
            return Err(GraphError::FetchValidation(
                "Fetch owner is not the active frame".into(),
            ));
        }
        let frame = self.frame(index)?;
        let compiled = self
            .callables
            .get(frame.graph_index)
            .and_then(|graph| graph.nodes.get(node.0))
            .ok_or(GraphError::MissingNode(node))?;
        if matches!(compiled.kind, CompiledNodeKind::Continuation { .. })
            && frame.checkpoints.get(node.0).is_none_or(Option::is_none)
        {
            return Err(GraphError::FetchValidation(
                "Fetch owner has no checkpoint".into(),
            ));
        }
        Ok(compiled.clone())
    }
}

pub(super) fn fetch_uuid(execution: Uuid, sequence: u64) -> Uuid {
    let mut name = *b"pravah.fetch.v1\0\0\0\0\0\0\0\0\0";
    let offset = name.len() - 8;
    name[offset..].copy_from_slice(&sequence.to_be_bytes());
    Uuid::new_v5(&execution, &name)
}

fn validate_outcome(
    fetch: &Fetch,
    outcome: &Result<FetchResponse, FetchError>,
) -> Result<(), GraphError> {
    if fetch.request().url() == "rath://generate"
        && let Ok(response) = outcome
    {
        crate::graph::fetch::rath::validate_response(response)?;
    }
    Ok(())
}

/// Checks immutable protocol inputs at installation and restore, never again during delivery.
fn validate_request(request: &FetchRequest) -> Result<(), GraphError> {
    if request.url() == "rath://generate" {
        crate::graph::fetch::rath::validate_request(request)?;
    }
    Ok(())
}

/// Rejects malformed ownership and identities for both pending and accepted external work.
pub(super) fn validate_fetch_state(
    callables: &[CompiledGraph],
    state: &State,
) -> Result<(), GraphError> {
    let mut accepted = 0;
    for frame in &state.frames {
        for inbox in &frame.continuation_inboxes {
            for input in inbox {
                if let ContinuationInput::Fetch { fetch, outcome } = input {
                    accepted += 1;
                    validate_identity(state, fetch)?;
                    validate_request(fetch.request())?;
                    validate_outcome(fetch, outcome)?;
                }
            }
        }
    }
    if accepted > 1 || (accepted > 0 && state.waiting.is_some()) {
        return Err(GraphError::SnapshotValidation(
            "multiple external request owners".into(),
        ));
    }
    let Some(Waiting::Fetch {
        frame_depth,
        node,
        fetch,
    }) = &state.waiting
    else {
        return Ok(());
    };
    if *frame_depth != state.frames.len() || *frame_depth == 0 {
        return Err(GraphError::SnapshotValidation(
            "invalid Fetch frame depth".into(),
        ));
    }
    validate_identity(state, fetch)?;
    validate_request(fetch.request())?;
    let frame = state
        .frames
        .last()
        .ok_or_else(|| GraphError::SnapshotValidation("missing Fetch frame".into()))?;
    let compiled = callables
        .get(frame.graph_index)
        .and_then(|graph| graph.nodes.get(node.0))
        .ok_or(GraphError::MissingNode(*node))?;
    match compiled.kind {
        CompiledNodeKind::Fetch if inputs_ready_with_new_epoch(frame, compiled)? => Ok(()),
        CompiledNodeKind::Continuation { .. }
            if frame.checkpoints.get(node.0).is_some_and(Option::is_some)
                && frame
                    .continuation_inboxes
                    .get(node.0)
                    .is_some_and(Vec::is_empty) =>
        {
            // Unstarted children may wait behind this Fetch (for example, while
            // persisting the preceding tool result). Frame-depth validation
            // above excludes an active child; queue targets are checked separately.
            Ok(())
        }
        _ => Err(GraphError::SnapshotValidation(
            "invalid Fetch owner state".into(),
        )),
    }
}

fn validate_identity(state: &State, fetch: &Fetch) -> Result<(), GraphError> {
    let sequence = state
        .next_fetch_sequence
        .checked_sub(1)
        .ok_or_else(|| GraphError::SnapshotValidation("Fetch without a sequence".into()))?;
    if fetch.id() != fetch_uuid(state.execution_id, sequence) {
        return Err(GraphError::SnapshotValidation(
            "invalid deterministic Fetch identity".into(),
        ));
    }
    Ok(())
}
