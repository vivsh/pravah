use serde_json::Value;

use super::flows::AgentContinuation;
use super::history::FlowHistory;
use super::phase::Phase;
use super::runtime::FlowCall;
use super::state::{FlowState, Frame};
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
    Entry,
    Dispatch,
    PendingTool {
        active_calls: Vec<String>,
        waiting_count: usize,
    },
    Exit,
}

#[derive(Debug, Clone)]
pub struct FrameView<'a> {
    pub depth: usize,
    pub session_id: &'a str,
    pub callable_entry: &'a str,
    pub callable_exit: &'a str,
    pub phase: PhaseKind,
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
        FrameView {
            depth,
            session_id: frame.session_id.as_str(),
            callable_entry: self.name_in_frame(frame, frame.callable.entry),
            callable_exit: self.name_in_frame(frame, frame.callable.exit),
            phase: Self::decode_phase(frame.phase.as_ref()),
            locals,
        }
    }

    fn name_in_frame(&self, frame: &'a Frame, id: NodeId) -> &'a str {
        self.callables
            .get(frame.callable.index)
            .map(|call| call.0.interner.name_of(id))
            .unwrap_or("<unknown>")
    }

    fn decode_phase(phase: Option<&Phase>) -> PhaseKind {
        match phase {
            None => PhaseKind::None,
            Some(Phase::Entry) => PhaseKind::Entry,
            Some(Phase::Continue(None)) => PhaseKind::Exit,
            Some(Phase::Continue(Some(value))) => {
                match serde_json::from_value::<AgentContinuation>(value.clone()) {
                    Ok(AgentContinuation::Dispatch) => PhaseKind::Dispatch,
                    Ok(AgentContinuation::PendingTool { active, waiting }) => {
                        let mut active_calls = active
                            .into_values()
                            .map(|(_, call_name)| call_name)
                            .collect::<Vec<_>>();
                        active_calls.sort();
                        let waiting_count = waiting.values().map(|queue| queue.len()).sum();
                        PhaseKind::PendingTool {
                            active_calls,
                            waiting_count,
                        }
                    }
                    Ok(AgentContinuation::Exit(_)) => PhaseKind::Exit,
                    Err(_) => PhaseKind::None,
                }
            }
        }
    }
}