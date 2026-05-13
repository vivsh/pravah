use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::flows::phase::Phase;

/// A single execution frame — one per active call on the stack.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct Frame {
    /// Key–value state for nodes executing in this frame.
    pub(crate) states: IndexMap<String, Value>,
    /// Current execution phase for the active multi-phase node in this frame.
    pub(crate) phase: Phase,
    /// Schema name of the sub-flow node that started this frame.
    pub(crate) entry: String,
    /// Key in the *parent* frame to write the sub-flow result into on completion.
    pub(crate) exit_name: String,
}

impl Frame {
    pub(crate) fn new(entry: &str, exit_name: &str) -> Self {
        Self {
            states: IndexMap::new(),
            phase: Phase::Entry,
            entry: entry.to_string(),
            exit_name: exit_name.to_string(),
        }
    }
}

/// Mutable runtime state: a call stack of frames.
///
/// An empty stack is a valid terminal state — it means execution finished.
#[derive(Serialize, Deserialize, Clone)]
pub struct FlowState {
    frames: Vec<Frame>,
    suspension: Option<String>,
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

    // ── frame management ──────────────────────────────────────────────────

    /// Pushes a new frame for an agent or sub-flow call.
    pub(crate) fn push_frame(&mut self, entry: &str, exit_name: &str) {
        self.frames.push(Frame::new(entry, exit_name));
    }

    /// Pops the top frame. Returns `None` if the stack is already empty.
    pub(crate) fn pop_frame(&mut self) -> Option<Frame> {
        self.frames.pop()
    }

    /// Depth of the frame stack (0 = empty / execution finished).
    pub(crate) fn frame_depth(&self) -> usize {
        self.frames.len()
    }

    /// Returns the `entry` field of the frame at `index` (0-based), or `None` if out of range.
    pub(crate) fn frame_entry_at(&self, index: usize) -> Option<&str> {
        self.frames.get(index).map(|f| f.entry.as_str())
    }

    /// Returns the `exit_name` field of the frame at `index` (0-based), or `None` if out of range.
    pub(crate) fn frame_exit_name_at(&self, index: usize) -> Option<&str> {
        self.frames.get(index).map(|f| f.exit_name.as_str())
    }

    // ── suspension ────────────────────────────────────────────────────────

    pub fn suspension(&self) -> Option<&String> {
        self.suspension.as_ref()
    }

    pub fn clear_suspension(&mut self) {
        self.suspension = None;
    }

    pub fn suspend(&mut self, node_id: &str) {
        self.suspension = Some(node_id.to_string());
    }

    // ── phase ─────────────────────────────────────────────────────────────

    /// Returns the phase of the top frame, or `None` if the stack is empty.
    pub(crate) fn phase(&self) -> Option<&Phase> {
        self.top().map(|f| &f.phase)
    }

    /// Sets the phase of the top frame. Returns `false` if the stack is empty.
    pub(crate) fn set_phase(&mut self, phase: Phase) -> bool {
        match self.top_mut() {
            Some(f) => { f.phase = phase; true }
            None => false,
        }
    }

    // ── state accessors (delegate to top frame) ───────────────────────────

    pub fn contains_state(&self, state_name: &str) -> bool {
        self.top().is_some_and(|f| f.states.contains_key(state_name))
    }

    /// Writes `value` under `new_node` and optionally removes `old_node`.
    /// Returns `false` if the stack is empty.
    pub fn set_state(&mut self, new_node: &str, value: Value, old_node: Option<&str>) -> bool {
        match self.top_mut() {
            Some(frame) => {
                frame.states.insert(new_node.to_string(), value);
                if let Some(old) = old_node {
                    frame.states.shift_remove(old);
                }
                true
            }
            None => false,
        }
    }

    /// Removes `state_name` from the top frame. Returns `false` if the stack is empty.
    pub fn remove_state(&mut self, state_name: &str) -> bool {
        match self.top_mut() {
            Some(frame) => { frame.states.shift_remove(state_name); true }
            None => false,
        }
    }

    pub fn get_state(&self, node_id: &str) -> Option<&Value> {
        self.top().and_then(|f| f.states.get(node_id))
    }

    pub fn len(&self) -> usize {
        self.top().map_or(0, |f| f.states.len())
    }

    pub fn get_index(&self, i: usize) -> Option<(&String, &Value)> {
        self.top().and_then(|f| f.states.get_index(i))
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.top()
            .into_iter()
            .flat_map(|f| f.states.keys())
    }
}
