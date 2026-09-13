use super::*;

impl Runtime {
    /// Recognizes an idle root chat boundary without retaining another lifecycle flag.
    pub(crate) fn chat_ready(&self, boundaries: &[NodeId]) -> bool {
        let Some(frame) = self.state.frames.first() else {
            return false;
        };
        let Some(suspension) = &self.state.suspension else {
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
