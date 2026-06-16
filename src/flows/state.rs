use std::collections::{HashMap, HashSet, VecDeque};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::flows::interner::NodeId;

/// A tool call that is queued behind a currently-active call to the same tool slot.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WaitingCall {
    pub call_id: String,
    pub args: Value,
    pub call_name: String,
    pub entry_id: NodeId,
}

/// Per-agent continuation state tracked inside the parent frame.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum AgentContinuation {
    /// Agent is ready to call the LLM.
    Dispatch,
    /// The LLM issued tool calls; waiting for work nodes to finish.
    PendingTool {
        /// One running call per tool exit slot (exit_id → (call_id, tool_name)).
        active: HashMap<NodeId, (String, String)>,
        /// Calls queued for each slot that is already occupied (exit_id → queue).
        waiting: HashMap<NodeId, VecDeque<WaitingCall>>,
        /// call_ids whose exit slots contain a [`ToolError`] payload rather than
        /// a typed output value. Checked in `handle_pending_tools` before
        /// calling `to_message`. Empty at every suspension point.
        #[serde(default, skip_serializing_if = "HashSet::is_empty")]
        error_exits: HashSet<String>,
    },
}

/// Per-item state tracked while an `each` node is fanning out.
///
/// # Known limitations
///
/// - `keep_alive` agents inside the sub-flow do not carry conversation history
///   across items. Each item's child frame is fresh, so repeated iterations see
///   independent sessions even when `keep_alive: true` is set on the agent.
///   To share history across items, the sub-flow would need to store session ids
///   in the parent frame's `keep_alive_sessions` map.
///
/// - Completed sub-flow sessions are never fed through the history compactor.
///   `run_compaction_and_flush` only visits live frames, so a long fan-out over
///   agent-heavy sub-flows grows `FlowHistory` without bound until the outer
///   runtime is dropped.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub(crate) struct EachState {
    /// Items yet to be processed.
    pub(crate) remaining: VecDeque<Value>,
    /// Outputs collected so far from completed sub-flow runs.
    pub(crate) results: Vec<Value>,
}

/// State for one agent node tracked inside the parent frame.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct AgentState {
    /// Session ID used for LLM history.
    pub session_id: String,
    /// Original input value, retained for use in `environment(&self)` at dispatch time.
    pub input: serde_json::Value,
    /// Effective preamble computed on the first dispatch and reused for all subsequent
    /// turns within the same invocation, preserving a stable KV-cache prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_preamble: Option<String>,
    pub continuation: AgentContinuation,
}

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
    /// Per-agent continuation state for all agents active in this frame.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub(crate) agent_states: IndexMap<NodeId, AgentState>,

    /// Callable that owns this frame.
    pub(crate) callable: Callable,

    /// Session id of this frame (used for sub-flow history when applicable).
    pub(crate) session_id: String,

    /// Maps a child agent's `NodeId` to a stable session id for keep-alive agents.
    /// Populated on first visit and reused on subsequent iterations.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub(crate) keep_alive_sessions: IndexMap<NodeId, String>,

    /// Per-each-node fan-out state, keyed by the each node's input `NodeId`.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub(crate) each_queues: IndexMap<NodeId, EachState>,
}

impl Frame {
    fn new(call: Callable) -> Self {
        Self {
            states: IndexMap::new(),
            agent_states: IndexMap::new(),
            callable: call,
            session_id: Uuid::now_v7().to_string(),
            keep_alive_sessions: IndexMap::new(),
            each_queues: IndexMap::new(),
        }
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

        let child = Frame::new(callable);

        if let Some(value) = entry_value {
            let entry = child.callable.entry;
            let mut child = child;
            child.states.insert(entry, value);
            self.frames.push(child);
        } else {
            self.frames.push(child);
        }
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

    // ── Agent state accessors ─────────────────────────────────────────────────

    /// Returns an immutable reference to the agent state for `node_id`.
    pub(crate) fn get_agent_state(&self, node_id: NodeId) -> Option<&AgentState> {
        self.top().and_then(|f| f.agent_states.get(&node_id))
    }

    /// Returns a mutable reference to the agent state for `node_id`.
    pub(crate) fn get_agent_state_mut(&mut self, node_id: NodeId) -> Option<&mut AgentState> {
        self.top_mut().and_then(|f| f.agent_states.get_mut(&node_id))
    }

    /// Inserts a fresh `Dispatch` agent state for `node_id` with the given session.
    pub(crate) fn init_agent_state(&mut self, node_id: NodeId, session_id: String, input: serde_json::Value) -> bool {
        match self.top_mut() {
            Some(frame) => {
                frame.agent_states.insert(node_id, AgentState {
                    session_id,
                    input,
                    cached_preamble: None,
                    continuation: AgentContinuation::Dispatch,
                });
                true
            }
            None => false,
        }
    }

    /// Removes the agent state for `node_id`.
    pub(crate) fn remove_agent_state(&mut self, node_id: NodeId) {
        if let Some(frame) = self.top_mut() {
            frame.agent_states.shift_remove(&node_id);
        }
    }

    // ── EachState accessors ───────────────────────────────────────────────────

    /// Returns `true` when the top frame has an active each-queue for `id`.
    pub(crate) fn each_queue_has(&self, id: NodeId) -> bool {
        self.top().is_some_and(|f| f.each_queues.contains_key(&id))
    }

    /// Inserts a fresh each-queue for `id`. Returns `false` when the stack is empty.
    pub(crate) fn each_queue_init(&mut self, id: NodeId, state: EachState) -> bool {
        match self.top_mut() {
            Some(frame) => { frame.each_queues.insert(id, state); true }
            None => false,
        }
    }

    /// Returns a mutable reference to the each-queue for `id`.
    pub(crate) fn each_queue_get_mut(&mut self, id: NodeId) -> Option<&mut EachState> {
        self.top_mut().and_then(|f| f.each_queues.get_mut(&id))
    }

    /// Removes and returns the each-queue for `id`.
    pub(crate) fn each_queue_remove(&mut self, id: NodeId) -> Option<EachState> {
        self.top_mut().and_then(|f| f.each_queues.shift_remove(&id))
    }

    /// Returns the stable session id for `agent_id`.
    /// If `keep_alive` is true, the id is minted once and stored in `keep_alive_sessions`
    /// so that repeated loop iterations share the same LLM conversation.
    pub(crate) fn get_or_init_session_id(&mut self, agent_id: NodeId, keep_alive: bool) -> String {
        if keep_alive {
            if let Some(frame) = self.top_mut() {
                return frame
                    .keep_alive_sessions
                    .entry(agent_id)
                    .or_insert_with(|| Uuid::now_v7().to_string())
                    .clone();
            }
        }
        Uuid::now_v7().to_string()
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

    pub fn callable_index(&self) -> Option<usize> {
        self.top().map(|f| f.callable.index)
    }

    /// Returns the session id of the first active agent in the top frame,
    /// or an empty string when no agents are active.
    pub(crate) fn top_session_id(&self) -> &str {
        self.top()
            .and_then(|f| f.agent_states.values().next())
            .map_or("", |s| s.session_id.as_str())
    }

    /// Returns session ids for all active agents across all frames.
    pub fn active_session_ids(&self) -> Vec<&str> {
        self.frames
            .iter()
            .flat_map(|f| f.agent_states.values().map(|s| s.session_id.as_str()))
            .collect()
    }

    /// Returns the current call-stack depth.
    pub(crate) fn depth(&self) -> usize {
        self.frames.len()
    }

    pub(crate) fn frames_slice(&self) -> &[Frame] {
        &self.frames
    }

    /// Records that the currently-active call on `exit_id` for `agent_id` produced a
    /// non-fatal [`ToolError`]. Looks up the call_id from the active map and inserts it
    /// into `error_exits` so that `handle_pending_tools` can route it correctly.
    /// Returns `false` if the agent or its active entry is not found.
    pub(crate) fn add_tool_error_exit(&mut self, agent_id: NodeId, exit_id: NodeId) -> bool {
        if let Some(AgentState {
            continuation: AgentContinuation::PendingTool { active, error_exits, .. },
            ..
        }) = self.top_mut().and_then(|f| f.agent_states.get_mut(&agent_id))
        {
            if let Some((call_id, _)) = active.get(&exit_id) {
                error_exits.insert(call_id.clone());
                return true;
            }
        }
        false
    }
}
