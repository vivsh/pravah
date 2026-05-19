use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::clients::Attachment;
use crate::flows::interner::NodeId;
use crate::flows::phase::Phase;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Callable {
    /// Parent slot that supplies this callable's input.
    pub parent_entry: NodeId,
    /// Parent slot that receives this callable's output.
    pub parent_exit: NodeId,
    /// Child slot that marks this frame as complete.
    pub exit: NodeId,
    /// Child slot that receives the input.
    pub entry: NodeId,
    /// Index into the runtime's callable table.
    pub index: usize,
    /// When true, repeated invocations of this agent reuse a stable session id
    /// that is stored in the *parent* frame's `keep_alive_sessions` map (keyed
    /// by this `entry` NodeId). This keeps conversation history continuous across
    /// loop iterations while ensuring that multiple agents in the same parent flow
    /// each have their own independent session.
    #[serde(default)]
    pub keep_alive: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct Suspension {
    pub src: NodeId,
    pub dst: NodeId,
    /// Schema name required by `resume()`.
    pub output_type: String,
}

/// One execution frame on the call stack.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Frame {
    /// State owned by this frame.
    pub(crate) states: IndexMap<NodeId, Value>,
    /// Attachments produced by tool calls, keyed by the tool's exit slot.
    /// Populated by `handle_tool` when a tool produces non-empty attachments,
    /// consumed by `handle_agent` when building the tool-result history message.
    /// Skipped during serialization when empty so existing snapshots remain valid.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub(crate) tool_attachments: IndexMap<NodeId, Vec<Attachment>>,
    /// Phase for the active multi-step node, if any.
    pub(crate) phase: Option<Phase>,

    /// Callable that owns this frame.
    pub(crate) callable: Callable,

    /// Session id used by agent history.
    pub(crate) session_id: String,

    /// Maps a child callable's entry `NodeId` to a stable session id.
    /// Populated only for children whose `Callable::keep_alive` is true, so
    /// repeated re-entries (e.g. an agent in a loop) reuse the same session
    /// and see the full conversation history. Multiple agents in one flow each
    /// get their own entry here and thus their own independent session.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub(crate) keep_alive_sessions: IndexMap<NodeId, String>,
}

impl Frame {
    fn new(call: Callable) -> Self {
        Self {
            states: IndexMap::new(),
            tool_attachments: IndexMap::new(),
            phase: None,
            callable: call,
            session_id: Uuid::now_v7().to_string(),
            keep_alive_sessions: IndexMap::new(),
        }
    }

    pub(crate) fn can_exit(&self) -> bool {
        self.states.contains_key(&self.callable.exit)
    }
}

/// Mutable runtime state.
/// An empty stack means execution is finished.
#[derive(Serialize, Deserialize, Clone)]
pub struct FlowState {
    frames: Vec<Frame>,
    suspension: Option<Suspension>,
}

impl FlowState {
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            suspension: None,
        }
    }

    fn top(&self) -> Option<&Frame> {
        self.frames.last()
    }

    fn top_mut(&mut self) -> Option<&mut Frame> {
        self.frames.last_mut()
    }

    /// Pushes a child frame and moves the parent entry value into it.
    pub(crate) fn call_enter(&mut self, callable: Callable) {
        let entry_value = self
            .top_mut()
            .and_then(|frame| frame.states.shift_remove(&callable.parent_entry));

        let mut child = Frame::new(callable);

        if child.callable.keep_alive {
            // Look up (or mint) a stable session id for this agent slot.
            // Keyed by the child's entry NodeId so each agent in a flow has its
            // own independent session even when multiple agents share one parent.
            let session_id = self
                .frames
                .last_mut()
                .map(|parent| {
                    parent
                        .keep_alive_sessions
                        .entry(child.callable.entry)
                        .or_insert_with(|| child.session_id.clone())
                        .clone()
                })
                .unwrap_or_else(|| child.session_id.clone());
            child.session_id = session_id;
        }

        if let Some(value) = entry_value {
            child.states.insert(child.callable.entry, value);
        }

        self.frames.push(child);
    }

    /// Pops every frame whose exit slot is ready.
    /// Returns the final value when the root frame exits.
    pub(crate) fn call_exit(&mut self) -> Option<Value> {
        loop {
            let (exited, return_value) = self.handle_exit();
            if exited {
                if let Some(v) = return_value {
                    return Some(v);
                }
            } else {
                break;
            }
        }
        None
    }

    fn handle_exit(&mut self) -> (bool, Option<Value>) {
        if let Some(frame) = self.top_mut() {
            let parent_exit = frame.callable.parent_exit;
            if let Some(exit_value) = frame.states.shift_remove(&frame.callable.exit) {
                self.frames.pop();
                if let Some(parent) = self.top_mut() {
                    parent.states.insert(parent_exit, exit_value);
                    return (true, None);
                } else {
                    return (true, Some(exit_value));
                }
            }
        }
        (false, None)
    }

    pub fn suspension(&self) -> Option<&Suspension> {
        self.suspension.as_ref()
    }

    pub fn suspend(&mut self, src: NodeId, dst: NodeId, output_type: String) {
        self.suspension = Some(Suspension { src, dst, output_type });
    }

    pub fn resume(&mut self, value: Value) -> bool {
        if let Some(suspension) = self.suspension.take() {
            if let Some(frame) = self.top_mut() {
                frame.states.shift_remove(&suspension.src);
                frame.states.insert(suspension.dst, value);
                return true;
            }
        }
        false
    }

    /// Returns the phase of the top frame.
    pub(crate) fn phase(&self) -> Option<&Phase> {
        self.top().and_then(|f| f.phase.as_ref())
    }

    /// Sets the phase of the top frame.
    /// Returns `false` when the stack is empty.
    pub(crate) fn set_phase(&mut self, phase: Phase) -> bool {
        match self.top_mut() {
            Some(f) => {
                f.phase = Some(phase);
                true
            }
            None => false,
        }
    }

    pub fn contains_state(&self, id: NodeId) -> bool {
        self.top().is_some_and(|f| f.states.contains_key(&id))
    }

    /// Writes a value into the top frame.
    /// Returns `false` when the stack is empty.
    pub fn set_state(&mut self, new_node: NodeId, value: Value, old_node: Option<NodeId>) -> bool {
        match self.top_mut() {
            Some(frame) => {
                frame.states.insert(new_node, value);
                if let Some(old) = old_node {
                    frame.states.shift_remove(&old);
                }
                true
            }
            None => false,
        }
    }

    /// Removes a state slot from the top frame.
    /// Returns `false` when the stack is empty.
    pub fn remove_state(&mut self, id: NodeId) -> bool {
        match self.top_mut() {
            Some(frame) => {
                frame.states.shift_remove(&id);
                true
            }
            None => false,
        }
    }

    pub fn get_state(&self, id: NodeId) -> Option<&Value> {
        self.top().and_then(|f| f.states.get(&id))
    }

    pub fn take_state(&mut self, id: NodeId) -> Option<Value> {
        self.top_mut().and_then(|f| f.states.shift_remove(&id))
    }

    /// Stores tool attachments for the given exit slot.
    /// Does nothing if the stack is empty or the vec is empty.
    pub(crate) fn set_tool_attachments(&mut self, id: NodeId, attachments: Vec<Attachment>) {
        if attachments.is_empty() {
            return;
        }
        if let Some(frame) = self.top_mut() {
            frame.tool_attachments.insert(id, attachments);
        }
    }

    /// Removes and returns tool attachments for the given exit slot.
    /// Returns an empty vec when none were stored.
    pub(crate) fn take_tool_attachments(&mut self, id: NodeId) -> Vec<Attachment> {
        self.top_mut()
            .and_then(|f| f.tool_attachments.shift_remove(&id))
            .unwrap_or_default()
    }

    /// Moves a state slot to the end of the top-frame ordering.
    pub fn reinsert_state(&mut self, id: NodeId)  {
        if let Some(frame) = self.top_mut() {
            if let Some(value) = frame.states.shift_remove(&id) {
                frame.states.insert(id, value);
            }
        }        
    }

    pub fn len(&self) -> usize {
        self.top().map_or(0, |f| f.states.len())
    }

    pub fn get_index(&self, i: usize) -> Option<(NodeId, &Value)> {
        self.top()
            .and_then(|f| f.states.get_index(i).map(|(&k, v)| (k, v)))
    }

    pub fn keys(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.top()
            .into_iter()
            .flat_map(|f| f.states.keys().copied())
    }

    pub fn callable_index(&self) -> Option<usize> {
        self.top().map(|f| f.callable.index)
    }

    /// Returns the entry slot for the top frame's callable.
    pub(crate) fn callable_entry(&self) -> Option<NodeId> {
        self.top().map(|f| f.callable.entry)
    }

    /// Returns the session id of the top frame.
    pub(crate) fn top_session_id(&self) -> &str {
        self.top().map_or("", |f| &f.session_id)
    }

    /// Returns session ids for every frame, from bottom to top.
    pub fn active_session_ids(&self) -> Vec<&str> {
        self.frames.iter().map(|f| f.session_id.as_str()).collect()
    }

    /// Returns the current call-stack depth.
    pub(crate) fn depth(&self) -> usize {
        self.frames.len()
    }

    pub(crate) fn frames_slice(&self) -> &[Frame] {
        &self.frames
    }
}
