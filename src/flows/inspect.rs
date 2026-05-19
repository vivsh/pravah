use serde_json::Value;

use super::history::FlowHistory;
use super::runtime::FlowCall;
use super::state::{AgentContinuation, FlowState, Frame};
use crate::clients::Message;
use crate::flows::NodeId;

#[derive(Debug, Clone)]
pub struct LocalVar<'a> {
    pub name: &'a str,
    pub node_id: NodeId,
    pub value: &'a Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseKind {
    None,
    Dispatch,
    PendingTool {
        active_calls: Vec<String>,
        waiting_count: usize,
    },
    Exit,
}

/// Per-agent phase info stored in a frame view.
#[derive(Debug, Clone)]
pub struct AgentPhaseView<'a> {
    pub agent_name: &'a str,
    pub phase: PhaseKind,
}

#[derive(Debug, Clone)]
pub struct FrameView<'a> {
    pub depth: usize,
    pub session_id: &'a str,
    pub callable_entry: &'a str,
    pub callable_exit: &'a str,
    pub agent_phases: Vec<AgentPhaseView<'a>>,
    pub locals: Vec<LocalVar<'a>>,
}

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

    pub fn depth(&self) -> usize {
        self.state.depth()
    }

    pub fn frames(&self) -> Vec<FrameView<'a>> {
        self.state
            .frames_slice()
            .iter()
            .enumerate()
            .map(|(depth, frame)| self.frame_view(depth, frame))
            .collect()
    }

    pub fn top_frame(&self) -> Option<FrameView<'a>> {
        let depth = self.state.depth().checked_sub(1)?;
        self.state
            .frames_slice()
            .last()
            .map(|frame| self.frame_view(depth, frame))
    }

    pub fn name_of(&self, id: NodeId) -> Option<&'a str> {
        let frame = self.state.frames_slice().last()?;
        Some(self.name_in_frame(frame, id))
    }

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
    /// [`FlowRuntime::push_message`] only when this returns `true`.
    pub fn is_agent_dispatch_ready(&self) -> bool {
        self.top_frame().map_or(false, |f| {
            f.agent_phases
                .iter()
                .any(|ap| matches!(ap.phase, PhaseKind::Dispatch))
        })
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
                    AgentContinuation::Exit(_) => PhaseKind::Exit,
                    AgentContinuation::PendingTool { active, waiting } => {
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

