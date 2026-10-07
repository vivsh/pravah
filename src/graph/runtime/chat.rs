use super::*;

impl Runtime {
    /// Opts Chat into installed worker services; the VM remains the sole policy owner.
    pub(crate) fn enable_chat_history(&mut self, store: bool, compactor: bool) {
        self.state.history_policy.persist |= store;
        self.state.history_policy.load |= store;
        self.state.history_policy.compact |= compactor;
    }
    /// Locates the Chat agent through the bootstrap edge's authored consumer relationship.
    fn chat_agent_node(
        &self,
        boundaries: &[NodeId; 2],
    ) -> Result<&crate::graph::model::Node, GraphError> {
        let frame = self.frame(0)?;
        let graph = &self
            .callables
            .get(frame.graph_index)
            .ok_or_else(|| GraphError::SnapshotValidation("missing Chat root graph".into()))?
            .graph;
        let bootstrap = graph
            .node(boundaries[0])
            .ok_or_else(|| GraphError::SnapshotValidation("missing Chat bootstrap".into()))?;
        let edge = bootstrap
            .outputs
            .first()
            .and_then(|edge| graph.edge(*edge))
            .ok_or_else(|| GraphError::SnapshotValidation("missing Chat input edge".into()))?;
        let mut agents = edge
            .consumers
            .iter()
            .filter_map(|id| graph.node(*id))
            .filter(|node| matches!(node.kind, NodeKind::Continuation { .. }));
        let agent = agents
            .next()
            .ok_or_else(|| GraphError::SnapshotValidation("missing Chat agent".into()))?;
        if agents.next().is_some() {
            return Err(GraphError::SnapshotValidation(
                "ambiguous Chat agent".into(),
            ));
        }
        Ok(agent)
    }

    /// Borrows authored configuration/tool metadata without caching a second copy.
    pub(crate) fn chat_agent_payload(
        &self,
        boundaries: &[NodeId; 2],
    ) -> Result<&Value, GraphError> {
        match &self.chat_agent_node(boundaries)?.kind {
            NodeKind::Continuation { payload, .. } => Ok(payload),
            _ => Err(GraphError::SnapshotValidation(
                "Chat agent is not a continuation".into(),
            )),
        }
    }

    /// Checks present boundary values and active invocation input without advancing execution.
    pub(crate) fn validate_chat_inputs(
        &self,
        boundaries: &[NodeId; 2],
        validate: impl Fn(&Value) -> Result<(), GraphError>,
    ) -> Result<(), GraphError> {
        let frame = self.frame(0)?;
        let graph = &self
            .callables
            .get(frame.graph_index)
            .ok_or_else(|| GraphError::SnapshotValidation("missing Chat graph".into()))?
            .graph;
        for boundary in boundaries {
            let edge = graph
                .node(*boundary)
                .and_then(|node| node.outputs.first())
                .ok_or_else(|| {
                    GraphError::SnapshotValidation("missing Chat boundary output".into())
                })?;
            if let Some(Some(value)) = frame.values.get(edge.0) {
                validate(value)?;
            }
        }
        let agent = self.chat_agent_node(boundaries)?;
        if let Some(Some(checkpoint)) = frame.checkpoints.get(agent.id.0)
            && let Some(input) = checkpoint_input(checkpoint)?
        {
            validate(input)?;
        }
        Ok(())
    }

    /// Recognizes an idle root chat boundary without retaining another lifecycle flag.
    pub(crate) fn chat_ready(&self, boundaries: &[NodeId]) -> bool {
        let Some(frame) = self.state.frames.first() else {
            return false;
        };
        let Some(suspension) = self.state.suspension() else {
            return false;
        };
        self.state.frames.len() == 1
            && frame.return_target.is_none()
            && suspension.frame_depth == 1
            && suspension.graph_index == frame.graph_index
            && suspension.target == SuspensionTarget::Node
            && boundaries.contains(&suspension.node)
            && frame.checkpoints.iter().all(Option::is_none)
            && frame.continuation_inboxes.iter().all(Vec::is_empty)
            && frame.continuation_child_queues.iter().all(Vec::is_empty)
    }

    /// Clones only the shared graph value for typed application-state decoding.
    pub(crate) fn chat_state(&self, variable: VarId) -> Result<Value, GraphError> {
        self.read_variable(0, variable)
    }

    /// Preflights every fallible write check before changing the root variable or its epoch.
    pub(crate) fn write_chat_state(
        &mut self,
        variable: VarId,
        value: Value,
    ) -> Result<(), GraphError> {
        self.validate_variable_write(0, variable, &value)?;
        let frame = self.frame(0)?;
        frame
            .variable_epochs
            .get(variable.0)
            .ok_or(GraphError::MissingVariable(variable))?;
        ensure_write_capacity(frame, 1)?;
        self.commit_variable_write(0, variable, value)
    }
}

/// Borrows invocation data through explicit effect phases without retaining a decoded copy.
fn checkpoint_input(checkpoint: &Value) -> Result<Option<&Value>, GraphError> {
    match checkpoint.get("effect").and_then(Value::as_str) {
        // A flush retains only an already completed transition; graph input remains validated above.
        Some("flush") => Ok(None),
        Some("control" | "generate") => checkpoint
            .get("checkpoint")
            .ok_or_else(|| GraphError::SnapshotValidation("missing effect checkpoint".into()))
            .and_then(checkpoint_input),
        _ => checkpoint
            .get("input")
            .map(Some)
            .ok_or_else(|| GraphError::SnapshotValidation("missing Chat checkpoint input".into())),
    }
}
