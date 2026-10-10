use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::agent_request::{AgentRequest, AgentResponse};
use super::ids::{EdgeId, NodeId};
use super::model::TypeSpec;
use super::registry::ContinuationChildCall;
use super::value::Value;
use crate::history::HistoryPolicy;
use std::collections::BTreeSet;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
/// Where a child frame should deliver its exit value.
///
/// Snapshots preserve this relationship using authored graph identities.
pub(crate) enum ReturnTarget {
    /// Return directly into a parent edge.
    Edge { parent_edge: EdgeId },
    /// Return into the parent either-node output.
    Either { parent_node: NodeId },
    /// Return into the active sequential each-node accumulator.
    Each { parent_node: NodeId },
    /// Return to a continuation node as a child-result event.
    Continuation {
        parent_node: NodeId,
        call_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// Accepted input retained until its complete continuation transition succeeds.
#[expect(
    clippy::large_enum_variant,
    reason = "the existing inbox owns accepted completions directly, without another allocation"
)]
pub(crate) enum ContinuationInput {
    Child {
        call_id: String,
        output: Value,
    },
    Resume {
        input: Value,
    },
    Agent {
        request: AgentRequest,
        response: AgentResponse,
    },
}

#[derive(Debug, Clone)]
/// One active VM frame.
///
/// Frames hold edge values, variables, checkpoints, and return metadata for one
/// callable graph. The runtime stack is just a `Vec<Frame>`.
pub(crate) struct Frame {
    /// Index into the runtime's compiled callable table.
    pub(crate) graph_index: usize,
    /// Edge value storage for this frame.
    pub(crate) values: Vec<Option<Value>>,
    /// Last epoch assigned to a successful value write in this frame.
    pub(crate) write_epoch: u64,
    /// Last write epoch for each edge.
    pub(crate) edge_epochs: Vec<u64>,
    /// Variable value storage for this frame.
    pub(crate) variables: Vec<Option<Value>>,
    /// Last write epoch for each variable.
    pub(crate) variable_epochs: Vec<u64>,
    /// Serialized checkpoints for active continuation-capable nodes.
    pub(crate) checkpoints: Vec<Option<Value>>,
    /// Opaque state slots for continuation nodes.
    pub(crate) continuation_states: Vec<Option<Value>>,
    /// Pending child results for continuation nodes.
    pub(crate) continuation_inboxes: Vec<Vec<ContinuationInput>>,
    /// Queued child calls for continuation nodes.
    pub(crate) continuation_child_queues: Vec<Vec<ContinuationChildCall>>,
    /// Last input activation epoch consumed by each node; zero means never run.
    pub(crate) node_epochs: Vec<u64>,
    /// Runtime-only reader counts for reclaimable multi-reader values.
    pub(crate) reader_counts: Vec<u32>,
    /// Sorted unkeyed sessions retained by this exact frame until it exits.
    pub(crate) unkeyed_conversations: Vec<String>,
    /// Parent delivery target when this frame exits.
    pub(crate) return_target: Option<ReturnTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SuspensionTarget {
    /// Resume writes to a first-class suspend node output.
    Node,
    /// Resume is delivered to the owning continuation handler.
    Continuation,
}

#[derive(Debug, Clone)]
/// Active external suspension recorded by the VM.
pub struct Suspension {
    /// One-based frame depth where suspension occurred.
    pub(crate) frame_depth: usize,
    /// Compiled graph index of the suspended frame.
    pub(crate) graph_index: usize,
    /// Suspend node waiting for resume.
    pub(crate) node: NodeId,
    /// Runtime owner that receives the resume value.
    pub(crate) target: SuspensionTarget,
    /// Expected resume type and JSON Schema.
    pub(crate) resume_type: Arc<TypeSpec>,
    /// Payload returned to the caller on suspend.
    pub(crate) payload: Value,
}

impl Suspension {
    /// Returns the one-based frame depth where execution is suspended.
    pub fn frame_depth(&self) -> usize {
        self.frame_depth
    }

    /// Returns the expected resume type name.
    pub fn resume_type(&self) -> &str {
        &self.resume_type.name
    }

    /// Returns the payload supplied to the external caller.
    pub fn payload(&self) -> &Value {
        &self.payload
    }
}

#[derive(Debug, Clone, Default)]
/// In-memory VM stack state for the graph runtime.
///
/// History is stored separately on `Snapshot`; this struct is only graph
/// execution state.
pub struct State {
    pub(crate) execution_id: Uuid,
    /// Application-only value and its fixed type, independent of frame lifetime.
    pub(crate) application_state: Option<(TypeSpec, Value)>,
    pub(crate) next_agent_sequence: u64,
    pub(crate) history_policy: HistoryPolicy,
    pub(crate) persisted_history_position: u64,
    pub(crate) loaded_conversation_keys: BTreeSet<String>,
    /// Active VM frame stack.
    pub(crate) frames: Vec<Frame>,
    /// Active node- or continuation-owned state when the VM is externally paused.
    pub(crate) waiting: Option<Waiting>,
}

#[derive(Debug, Clone)]
pub(crate) enum Waiting {
    Suspend(Suspension),
    Agent {
        frame_depth: usize,
        node: NodeId,
        request: AgentRequest,
    },
}

impl State {
    /// Returns the active VM stack depth.
    pub fn frame_depth(&self) -> usize {
        self.frames.len()
    }

    /// Returns whether the VM is waiting for external input.
    pub fn is_suspended(&self) -> bool {
        self.waiting.is_some()
    }

    /// Returns the host-supplied execution identity retained by snapshots.
    pub fn execution_id(&self) -> Uuid {
        self.execution_id
    }

    pub(crate) fn suspension(&self) -> Option<&Suspension> {
        match &self.waiting {
            Some(Waiting::Suspend(value)) => Some(value),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn values_for_test(&self) -> Vec<&Value> {
        self.frames
            .iter()
            .flat_map(|frame| frame.values.iter().filter_map(Option::as_ref))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Result of advancing the edge VM by one public step.
pub enum Step {
    /// The VM is waiting for the host to execute this exact request.
    Agent(AgentRequest),
    /// A node ran or a frame exited; call `next()` again.
    Continue,
    /// The root frame exited with this final output value.
    Done(Value),
    /// The VM paused at a node- or continuation-owned suspension payload.
    Suspend(Value),
}
