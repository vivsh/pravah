use serde_json::Value;

use super::history::FlowHistory;
use super::nodes::FlowNode;
use super::runtime::FlowCall;
use super::state::{AgentContinuation, FlowState, Frame};
use crate::clients::Message;
use crate::commons::Agent;
use crate::context::Context;
use crate::flows::NodeId;
use crate::tools::ToolDefinition;

/// A named local state variable visible in a frame.
#[derive(Debug, Clone)]
pub struct LocalVar<'a> {
    /// The interned node name for this variable.
    pub name: &'a str,
    /// The raw node id corresponding to `name`.
    pub node_id: NodeId,
    /// The current serialized value.
    pub value: &'a Value,
}

/// Phase an agent is currently in within its frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseKind {
    /// Agent has not started or has already exited.
    None,
    /// Agent is at a dispatch boundary — next step will call the LLM.
    Dispatch,
    PendingTool {
        active_calls: Vec<String>,
        waiting_count: usize,
    },
    /// Agent has produced its final output and is leaving the frame.
    Exit,
}

/// Per-agent phase info stored in a frame view.
#[derive(Debug, Clone)]
pub struct AgentPhaseView<'a> {
    /// The registered node name of the agent.
    pub agent_name: &'a str,
    /// Current execution phase.
    pub phase: PhaseKind,
}

/// Snapshot of one call frame on the execution stack.
#[derive(Debug, Clone)]
pub struct FrameView<'a> {
    /// Depth of this frame; 0 is the root frame.
    pub depth: usize,
    /// Session id active in this frame.
    pub session_id: &'a str,
    /// Node name of the callable's entry point.
    pub callable_entry: &'a str,
    /// Node name of the callable's exit point.
    pub callable_exit: &'a str,
    /// Agent phase snapshots for all agents that have been entered in this frame.
    pub agent_phases: Vec<AgentPhaseView<'a>>,
    /// Live local variables held by this frame.
    pub locals: Vec<LocalVar<'a>>,
}

/// Agent configuration as visible to the LLM, intended for testing.
pub struct AgentView {
    /// Full effective system prompt sent to the LLM on the first turn.
    pub preamble: String,
    /// Tool definitions visible to this agent.
    pub tools: Vec<ToolDefinition>,
}

/// Read-only view into a running [`FlowRuntime`](crate::flows::FlowRuntime).
/// Obtained via [`FlowRuntime::inspector`](crate::flows::FlowRuntime::inspector).
pub struct FlowInspector<'a> {
    state: &'a FlowState,
    callables: &'a [FlowCall],
    history: &'a FlowHistory,
}

impl<'a> FlowInspector<'a> {
    pub(crate) fn new(
        state: &'a FlowState,
        callables: &'a [FlowCall],
        history: &'a FlowHistory,
    ) -> Self {
        Self {
            state,
            callables,
            history,
        }
    }

    /// Number of frames on the call stack; 0 means the flow has not started.
    pub fn depth(&self) -> usize {
        self.state.depth()
    }

    /// Snapshots all frames on the current call stack, innermost last.
    pub fn frames(&self) -> Vec<FrameView<'a>> {
        self.state
            .frames_slice()
            .iter()
            .enumerate()
            .map(|(depth, frame)| self.frame_view(depth, frame))
            .collect()
    }

    /// Returns the innermost active frame, or `None` if the flow is not running.
    pub fn top_frame(&self) -> Option<FrameView<'a>> {
        let depth = self.state.depth().checked_sub(1)?;
        self.state
            .frames_slice()
            .last()
            .map(|frame| self.frame_view(depth, frame))
    }

    /// Resolves a [`NodeId`] to its registered string name in the active frame.
    /// Returns `None` if no frame is active.
    pub fn name_of(&self, id: NodeId) -> Option<&'a str> {
        let frame = self.state.frames_slice().last()?;
        Some(self.name_in_frame(frame, id))
    }

    /// Returns the full history, including entries from all sessions.
    pub fn history(&self) -> &'a FlowHistory {
        self.history
    }

    pub fn is_suspended(&self) -> bool {
        self.state.suspension().is_some()
    }

    pub fn suspension_type(&self) -> Option<&'a str> {
        self.state.suspension().map(|s| s.output_type.as_str())
    }

    /// Returns an iterator over the live messages for the current active session, oldest first.
    pub fn messages(&self) -> impl Iterator<Item = &'a Message> + 'a {
        let session_id = self.state.top_session_id().to_owned();
        self.history
            .entries()
            .iter()
            .filter(move |e| !e.evicted && e.session_id == session_id)
            .map(|e| &e.message)
    }

    /// Returns `true` when any agent in the active frame is at a dispatch boundary —
    /// the next engine step will call the LLM. It is safe to call
    /// [`FlowRuntime::inject_message`](crate::flows::FlowRuntime::inject_message) only when this returns `true`.
    pub fn is_agent_dispatch_ready(&self) -> bool {
        self.top_frame().map_or(false, |f| {
            f.agent_phases
                .iter()
                .any(|ap| matches!(ap.phase, PhaseKind::Dispatch))
        })
    }

    /// Returns the effective preamble and tool definitions for agent `T`.
    ///
    /// The preamble reflects the fully assembled system prompt — static base,
    /// runtime environment from `ctx`, and input-schema hint — exactly as it
    /// would be sent to the LLM. Returns `None` when `T` is not registered.
    pub fn agent_view<T: Agent>(&self, ctx: &Context) -> Option<AgentView> {
        let key = T::node_id();
        for callable in self.callables {
            let graph = &callable.0;
            if let Some(node_id) = graph.interner.intern_get(&key) {
                if let Some(FlowNode::Agent(info)) = graph.nodes.get(&node_id) {
                    return Some(AgentView {
                        preamble: info.effective_preamble(&serde_json::Value::Null, ctx),
                        tools: info.tools.iter().map(|t| t.definition.clone()).collect(),
                    });
                }
            }
        }
        None
    }

    fn frame_view(&self, depth: usize, frame: &'a Frame) -> FrameView<'a> {
        let locals = frame
            .states
            .iter()
            .map(|(&node_id, value)| LocalVar {
                name: self.name_in_frame(frame, node_id),
                node_id,
                value,
            })
            .collect();

        let agent_phases = frame
            .agent_states
            .iter()
            .map(|(&agent_id, agent_state)| {
                let agent_name = self.name_in_frame(frame, agent_id);
                let phase = match &agent_state.continuation {
                    AgentContinuation::Dispatch => PhaseKind::Dispatch,
                    AgentContinuation::PendingTool { active, waiting, .. } => {
                        let mut active_calls = active
                            .values()
                            .map(|(_, call_name)| call_name.clone())
                            .collect::<Vec<_>>();
                        active_calls.sort();
                        let waiting_count =
                            waiting.values().map(|queue| queue.len()).sum();
                        PhaseKind::PendingTool {
                            active_calls,
                            waiting_count,
                        }
                    }
                };
                AgentPhaseView { agent_name, phase }
            })
            .collect();

        FrameView {
            depth,
            session_id: frame.session_id.as_str(),
            callable_entry: self.name_in_frame(frame, frame.callable.entry),
            callable_exit: self.name_in_frame(frame, frame.callable.exit),
            agent_phases,
            locals,
        }
    }

    fn name_in_frame(&self, frame: &'a Frame, id: NodeId) -> &'a str {
        self.callables
            .get(frame.callable.index)
            .map(|call| call.0.interner.name_of(id))
            .unwrap_or("<unknown>")
    }
}

