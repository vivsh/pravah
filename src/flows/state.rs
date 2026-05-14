use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::flows::interner::NodeId;
use crate::flows::phase::Phase;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Callable {
    /// key where entry for this callable is expected in the parent frame;
    pub parent_entry: NodeId,
    /// key where exit value from this callable should be placed in the parent frame;
    pub parent_exit: NodeId,
    /// key where exit value from this callable is expected in this frame;
    pub exit: NodeId,
    /// key where entry value for this callable is expected in this frame;
    pub entry: NodeId,
    /// index of this callable in the runtime's callables list;
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct Suspension {
    pub src: NodeId,
    pub dst: NodeId,
}

/// A single execution frame — one per active call on the stack.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Frame {
    /// Key–value state for nodes executing in this frame.
    states: IndexMap<NodeId, Value>,
    /// Current execution phase for the active multi-phase node in this frame.
    phase: Option<Phase>, // discriminates between child(agent) and parent(flow) frames

    /// NodeId of the node that started this frame;
    callable: Callable,
}

impl Frame {
    fn new(call: Callable) -> Self {
        Self {
            states: IndexMap::new(),
            phase: None,
            callable: call,
        }
    }

    pub(crate) fn can_exit(&self) -> bool {
        self.states.contains_key(&self.callable.exit)
    }
}

/// Mutable runtime state: a call stack of frames.
///
/// An empty stack is a valid terminal state — it means execution finished.
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

    /// Pushes a new frame for `callable`, transferring state from the parent entry to the child entry.
    pub(crate) fn call_enter(&mut self, callable: Callable) {
        let entry_value = self
            .top_mut()
            .and_then(|frame| frame.states.shift_remove(&callable.parent_entry));

        let mut child = Frame::new(callable);

        if let Some(value) = entry_value {
            child.states.insert(child.callable.entry, value);
        }

        self.frames.push(child);
    }

    /// Checks if the top frame can exit, and if so pops
    /// it and transfers state from the child exit to the parent.
    /// if parent does not exist, the state is discarded. Returns the popped value
    /// This should be done for all frames till exit is not possible, to bubble up the exit through the stack.     
    /// This must be called at the end of every step to ensure the stack is in a consistent state
    /// for the next step.
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

    // ── suspension ────────────────────────────────────────────────────────

    pub fn suspension(&self) -> Option<&Suspension> {
        self.suspension.as_ref()
    }

    pub fn suspend(&mut self, src: NodeId, dst: NodeId) {
        self.suspension = Some(Suspension { src, dst });
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

    // ── phase ─────────────────────────────────────────────────────────────

    /// Returns the phase of the top frame, or `None` if the stack is empty.
    pub(crate) fn phase(&self) -> Option<&Phase> {
        self.top().and_then(|f| f.phase.as_ref())
    }

    /// Sets the phase of the top frame. Returns `false` if the stack is empty.
    pub(crate) fn set_phase(&mut self, phase: Phase) -> bool {
        match self.top_mut() {
            Some(f) => {
                f.phase = Some(phase);
                true
            }
            None => false,
        }
    }

    // ── state accessors (delegate to top frame) ───────────────────────────

    pub fn contains_state(&self, id: NodeId) -> bool {
        self.top().is_some_and(|f| f.states.contains_key(&id))
    }

    /// Writes `value` under `new_node` and optionally removes `old_node`.
    /// Returns `false` if the stack is empty.
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

    /// Removes `id` from the top frame. Returns `false` if the stack is empty.
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
}
