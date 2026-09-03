use serde::{Deserialize, Serialize};

use super::ids::{EdgeId, NodeId};
use super::model::TypeSpec;
use super::registry::ContinuationChildCall;
use super::value::Value;

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
/// Completed child-call result waiting for a continuation node.
pub(crate) struct ContinuationChildResult {
    /// Child call id originally requested by the continuation.
    pub(crate) call_id: String,
    /// Child frame output value.
    pub(crate) output: Value,
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
    pub(crate) continuation_inboxes: Vec<Vec<ContinuationChildResult>>,
    /// Queued child calls for continuation nodes.
    pub(crate) continuation_child_queues: Vec<Vec<ContinuationChildCall>>,
    /// Last input activation epoch consumed by each node; zero means never run.
    pub(crate) node_epochs: Vec<u64>,
    /// Runtime-only reader counts for reclaimable multi-reader values.
    pub(crate) reader_counts: Vec<u32>,
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
    pub(crate) resume_type: TypeSpec,
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
    /// Active VM frame stack.
    pub(crate) frames: Vec<Frame>,
    /// Active node- or continuation-owned state when the VM is externally paused.
    pub(crate) suspension: Option<Suspension>,
}

impl State {
    /// Returns the active VM stack depth.
    pub fn frame_depth(&self) -> usize {
        self.frames.len()
    }

    /// Returns whether the VM is waiting for external input.
    pub fn is_suspended(&self) -> bool {
        self.suspension.is_some()
    }

    #[cfg(test)]
    pub(crate) fn values_for_test(&self) -> Vec<&Value> {
        self.frames
            .iter()
            .flat_map(|frame| frame.values.iter().filter_map(Option::as_ref))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Result of advancing the edge VM by one public step.
pub enum Step {
    /// A node ran or a frame exited; call `next()` again.
    Continue,
    /// The root frame exited with this final output value.
    Done(Value),
    /// The VM paused at a node- or continuation-owned suspension payload.
    Suspend(Value),
}
