use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::history::MessageHistory;

use super::agent::support::{validate_agent_snapshot_state, validate_agent_suspension};
use super::agent_request::{AgentRequest, AgentResponse};
use super::error::GraphError;
use super::ids::{EdgeId, HandlerKey, NodeId, VarId};
use super::model::{BuiltinNode, NodeKind, UntypedGraph, VarInit, VarKey, VarScope, Variable};
use super::registry::{
    ContinuationChildCall, ContinuationContext, ContinuationEvent, ContinuationTransition,
    HandlerRegistry,
};
use super::schema::validate_value;
use super::state::{
    ContinuationInput, Frame, ReturnTarget, State, Step, Suspension, SuspensionTarget, Waiting,
};
use super::validation::{validate_graph_shape, validate_registry_keys};
use super::value::{Value, to_value};

mod agent;
mod agent_history;
mod chat;
mod compile;
mod continuation;
mod conversations;
mod dce;
mod execution;
mod fingerprint;
mod helpers;
mod history;
mod liveness;
mod maintenance;
mod path;
mod reclaim;
mod snapshot;
mod sparse;
#[cfg(test)]
mod tests;

use compile::*;
use dce::DcePlan;
pub use fingerprint::GraphFingerprint;
use helpers::*;
use liveness::{LivenessPlan, ReleaseAction};
use maintenance::validate_history_progress;
use path::{CallSite, GraphPath};
use reclaim::rebuild_reader_counts;
use snapshot::validate_snapshot_state;
use sparse::{SparseState, expand_state, sparse_state};

/// Current serialized runtime snapshot version.
pub const SNAPSHOT_VERSION: u32 = 15;

#[derive(Clone)]
struct CompiledGraph {
    path: GraphPath,
    graph: Arc<UntypedGraph>,
    nodes: Vec<CompiledNode>,
    instructions: Arc<[NodeId]>,
    child_indices: Vec<CompiledChildren>,
    inheritable_by_key: HashMap<VarKey, VarId>,
    liveness: LivenessPlan,
}

#[derive(Debug, Clone, Default)]
struct CompiledChildren {
    primary: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    continuation: Vec<usize>,
}

#[derive(Clone)]
struct CompiledNode {
    id: NodeId,
    name: Arc<str>,
    inputs: Arc<[EdgeId]>,
    outputs: Arc<[EdgeId]>,
    kind: CompiledNodeKind,
    can_continue: bool,
    can_suspend: bool,
    release_actions: Arc<[ReleaseAction]>,
}

#[derive(Clone)]
enum CompiledNodeKind {
    Builtin {
        op: BuiltinNode,
    },
    PureHandler {
        key: HandlerKey,
    },
    Continuation {
        key: HandlerKey,
        payload: Arc<Value>,
        children: Arc<[usize]>,
        output_validator: Option<Arc<jsonschema::Validator>>,
    },
    Suspend {
        payload: Arc<Value>,
        resume_type: Arc<super::model::TypeSpec>,
    },
    Load {
        var: VarId,
        key: HandlerKey,
    },
    Store {
        var: VarId,
        key: HandlerKey,
    },
    Subflow {
        child_index: usize,
    },
    Either {
        key: HandlerKey,
        left_index: usize,
        right_index: usize,
    },
    Each {
        child_index: usize,
    },
    Goto {
        target: EdgeId,
    },
}

struct PreparedContinuationChild {
    frame: Frame,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Serializable snapshot of a graph runtime.
///
/// It contains only graph identity, VM state, and runtime-owned history. The
/// graph, handlers, and services are supplied separately when restoring.
pub struct Snapshot {
    /// Snapshot format version.
    pub(crate) snapshot_version: u32,
    /// Identity of the separately prepared graph being executed.
    pub(crate) graph_fingerprint: GraphFingerprint,
    /// Serializable VM frame stack and suspension state.
    pub(crate) state: SparseState,
    /// Runtime-owned conversation/history state.
    pub(crate) history: MessageHistory,
}

impl Snapshot {
    /// Returns the snapshot wire-format version.
    pub fn version(&self) -> u32 {
        self.snapshot_version
    }

    /// Returns the identity of the graph required to restore this continuation.
    pub fn graph_fingerprint(&self) -> GraphFingerprint {
        self.graph_fingerprint
    }

    /// Returns the runtime-owned conversation history captured by this snapshot.
    pub fn history(&self) -> &MessageHistory {
        &self.history
    }
}

/// Validated, compiled graph and handlers reusable across runtime executions.
#[derive(Clone)]
pub struct PreparedGraph {
    graph: Arc<UntypedGraph>,
    callables: Arc<[CompiledGraph]>,
    root_index: usize,
    registry: Arc<HandlerRegistry>,
    fingerprint: GraphFingerprint,
}

/// Isolated edge-graph VM. It preserves Pravah's stack-machine shape: each
/// frame executes one graph, subflows push frames, and frame exits cascade.
pub struct Runtime {
    callables: Arc<[CompiledGraph]>,
    registry: Arc<HandlerRegistry>,
    graph_fingerprint: GraphFingerprint,
    state: State,
    history: MessageHistory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BranchSide {
    Left,
    Right,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BranchChoice {
    side: BranchSide,
    value: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EachVmCheckpoint {
    items: Vec<Value>,
    outputs: Vec<Value>,
    index: usize,
}

impl PreparedGraph {
    /// Builds an external executor sharing this graph's immutable registry.
    pub fn executor(&self, context: crate::Context) -> super::agent_request::AgentExecutor {
        super::agent_request::AgentExecutor::from_registry(context, Arc::clone(&self.registry))
    }
    /// Validates and compiles a graph and its handler registry once.
    ///
    /// The graph and registry are validated up front so missing handlers and
    /// malformed wiring fail before execution starts.
    pub fn new(graph: UntypedGraph, registry: HandlerRegistry) -> Result<Self, GraphError> {
        validate_graph_shape(&graph)?;
        let has_value = |key: &str| registry.has_value(key);
        let has_continuation = |key: &str| registry.has_continuation(key);
        validate_registry_keys(&graph, &has_value, &has_continuation)?;
        validate_continuation_payloads(&graph, &registry)?;

        let fingerprint = GraphFingerprint::calculate(&graph)?;
        let mut callables = Vec::new();
        let root_index = compile_graph(graph, &mut callables)?;
        let graph = callables
            .get(root_index)
            .ok_or_else(|| GraphError::Invalid("compiled root graph is missing".into()))?
            .graph
            .clone();
        Ok(Self {
            graph,
            callables: Arc::from(callables.into_boxed_slice()),
            root_index,
            registry: Arc::new(registry),
            fingerprint,
        })
    }

    /// Returns the validated graph represented by this prepared executable.
    pub fn graph(&self) -> &UntypedGraph {
        self.graph.as_ref()
    }

    /// Returns the handlers bound to this prepared graph.
    pub fn registry(&self) -> &HandlerRegistry {
        self.registry.as_ref()
    }

    /// Returns the stable fingerprint required by compatible snapshots.
    pub fn fingerprint(&self) -> GraphFingerprint {
        self.fingerprint
    }

    /// Starts an isolated runtime under a host-supplied execution identity.
    ///
    /// Fails before returning a runtime when the entry value is incompatible
    /// with the prepared graph.
    pub fn start(&self, input: Value, execution_id: Uuid) -> Result<Runtime, GraphError> {
        self.start_runtime(input, execution_id, MessageHistory::new())
    }

    /// Starts a fresh execution taking ownership of validated, completed working history.
    /// Agents select imported conversations through their configuration keys. Invalid row
    /// identities, positions or incomplete message groups fail before execution starts.
    /// Supply a fresh execution UUID for this independent execution.
    /// Unkeyed imported sessions belong to the root frame and are removed when it exits;
    /// keyed sessions remain available until explicitly dropped. Supplied usage totals remain intact.
    pub fn start_with_history(
        &self,
        input: Value,
        execution_id: Uuid,
        history: MessageHistory,
    ) -> Result<Runtime, GraphError> {
        history.validate_import()?;
        self.start_runtime(input, execution_id, history)
    }

    /// Initializes checked entry state; supplied history has already passed its public boundary.
    fn start_runtime(
        &self,
        input: Value,
        execution_id: Uuid,
        history: MessageHistory,
    ) -> Result<Runtime, GraphError> {
        let mut state = State {
            execution_id,
            ..State::default()
        };
        let mut frame = new_frame(&self.callables, &[], self.root_index, None)?;
        let root_graph = self
            .callables
            .get(self.root_index)
            .ok_or_else(|| GraphError::Invalid("compiled root graph is missing".into()))?;
        let entry = root_graph.graph.entry;
        validate_edge_value(&root_graph.graph, entry, &input, "entry input")?;
        write_edge(&mut frame, entry, input)?;
        conversations::register_sessions(&mut frame, history.entries());
        state.frames.push(frame);

        Ok(Runtime {
            callables: Arc::clone(&self.callables),
            registry: Arc::clone(&self.registry),
            graph_fingerprint: self.fingerprint,
            state,
            history,
        })
    }

    /// Restores an isolated runtime after checking version, graph, and VM state.
    ///
    /// Runtime dependencies are intentionally not serialized. Supply fresh
    /// context and services to the external AgentExecutor used after restoration.
    ///
    /// Version, fingerprint, sparse state, frame, continuation, and suspension
    /// validation complete before a runtime is returned.
    pub fn restore(&self, snapshot: Snapshot) -> Result<Runtime, GraphError> {
        if snapshot.snapshot_version != SNAPSHOT_VERSION {
            return Err(GraphError::SnapshotVersion {
                got: snapshot.snapshot_version,
                expected: SNAPSHOT_VERSION,
            });
        }
        if snapshot.graph_fingerprint != self.fingerprint {
            return Err(GraphError::GraphMismatch {
                expected: self.fingerprint.to_string(),
                got: snapshot.graph_fingerprint.to_string(),
            });
        }
        let mut state =
            expand_state(&self.callables, snapshot.state).map_err(as_snapshot_validation_error)?;
        validate_history_progress(&state, &snapshot.history)?;
        agent::validate_agent_state(&self.callables, &state, &snapshot.history)
            .map_err(as_snapshot_validation_error)?;
        validate_snapshot_state(&self.callables, self.root_index, &state)
            .map_err(as_snapshot_validation_error)?;
        conversations::validate_conversation_owners(&self.callables, &state, &snapshot.history)?;
        rebuild_reader_counts(&self.callables, &mut state)?;

        Ok(Runtime {
            callables: Arc::clone(&self.callables),
            registry: Arc::clone(&self.registry),
            graph_fingerprint: self.fingerprint,
            state,
            history: snapshot.history,
        })
    }
}

fn validate_continuation_payloads(
    graph: &UntypedGraph,
    registry: &HandlerRegistry,
) -> Result<(), GraphError> {
    for node in &graph.nodes {
        match &node.kind {
            NodeKind::Continuation {
                key,
                payload,
                children,
            } => {
                let handler = registry
                    .continuation(key)
                    .ok_or_else(|| GraphError::MissingHandler(key.as_str().into()))?;
                handler.validate_payload(payload)?;
                for child in children {
                    validate_continuation_payloads(child, registry)?;
                }
            }
            NodeKind::Subflow { graph } | NodeKind::Each { graph } => {
                validate_continuation_payloads(graph, registry)?;
            }
            NodeKind::Either { left, right, .. } => {
                validate_continuation_payloads(left, registry)?;
                validate_continuation_payloads(right, registry)?;
            }
            NodeKind::Builtin { .. }
            | NodeKind::PureHandler { .. }
            | NodeKind::Suspend { .. }
            | NodeKind::Load { .. }
            | NodeKind::Store { .. }
            | NodeKind::Goto { .. } => {}
        }
    }
    Ok(())
}

impl Runtime {
    fn continuation_context<'a>(&'a self, node: &'a CompiledNode) -> ContinuationContext<'a> {
        let validator = match &node.kind {
            CompiledNodeKind::Continuation {
                output_validator, ..
            } => output_validator.as_deref(),
            _ => None,
        };
        ContinuationContext::new(
            self.state.execution_id,
            &self.history,
            validator,
            &self.state.history_policy,
            &self.state.loaded_conversation_keys,
        )
    }

    /// Borrows the sole execution-history owner without taking a lock.
    pub fn history(&self) -> &MessageHistory {
        &self.history
    }

    /// Returns the current VM state for inspection.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Returns the active external suspension, when the VM is waiting for input.
    pub fn suspension(&self) -> Option<&Suspension> {
        self.state.suspension()
    }

    pub(crate) fn suspension_type_spec(&self) -> Result<&super::model::TypeSpec, GraphError> {
        let suspension = self
            .state
            .suspension()
            .ok_or_else(|| GraphError::SnapshotValidation("runtime is not suspended".into()))?;
        Ok(&suspension.resume_type)
    }

    /// Captures versioned VM state and history tied to this graph's fingerprint.
    pub fn snapshot(&self) -> Result<Snapshot, GraphError> {
        let history = self.history.clone();
        Ok(Snapshot {
            snapshot_version: SNAPSHOT_VERSION,
            graph_fingerprint: self.graph_fingerprint,
            state: sparse_state(&self.callables, &self.state)?,
            history,
        })
    }

    /// Advances the synchronous VM by at most one prepared instruction.
    ///
    /// Returns `ResumeRequired` if the VM is suspended; use `resume()` for a
    /// suspend node or continuation-owned suspension.
    #[expect(
        clippy::should_implement_trait,
        reason = "explicit fallible VM stepping is not iteration"
    )]
    pub fn next(&mut self) -> Result<Step, GraphError> {
        if self.state.waiting.is_some() {
            return Err(GraphError::ResumeRequired);
        }
        let step = self.step_inner()?;
        match step {
            Step::Continue => self.try_exit_frames(),
            other => Ok(other),
        }
    }

    /// Accepts a typed value at the active suspension without executing another instruction.
    ///
    /// Use [`Runtime::resume_value`] when the value is already in Pravah's
    /// runtime domain so its shared representation is preserved.
    /// Conversion or schema failure leaves the suspension unchanged.
    pub fn resume<T>(&mut self, value: T) -> Result<(), GraphError>
    where
        T: Serialize,
    {
        let value = to_value(value).map_err(|err| GraphError::ValueConversion {
            target: "resume input".into(),
            reason: err.to_string(),
        })?;
        self.resume_value(value)
    }

    /// Accepts a raw graph value at the active suspension; call `next` to continue.
    ///
    /// Validation failure leaves the suspension and continuation state unchanged.
    pub fn resume_value(&mut self, value: Value) -> Result<(), GraphError> {
        let suspension = self
            .state
            .suspension()
            .ok_or(GraphError::UnexpectedResume)?;
        validate_value(&suspension.resume_type, &value, "resume value")?;
        let (frame_index, node) = self.validate_suspension_target(suspension)?;
        match suspension.target {
            SuspensionTarget::Node => self.resume_suspend_node(frame_index, &node, value),
            SuspensionTarget::Continuation => self.resume_continuation(frame_index, node, value),
        }
    }

    /// Resolves the active suspension only after validating its frame and node.
    fn validate_suspension_target(
        &self,
        suspension: &Suspension,
    ) -> Result<(usize, CompiledNode), GraphError> {
        if suspension.frame_depth == 0 || suspension.frame_depth > self.state.frames.len() {
            return Err(GraphError::Invalid(format!(
                "suspension frame depth {} is invalid for stack depth {}",
                suspension.frame_depth,
                self.state.frames.len()
            )));
        }
        if suspension.frame_depth != self.state.frames.len() {
            return Err(GraphError::Invalid(format!(
                "suspension frame depth {} is not the active frame depth {}",
                suspension.frame_depth,
                self.state.frames.len()
            )));
        }
        let frame_index = suspension.frame_depth - 1;
        let suspended_frame = self
            .state
            .frames
            .get(frame_index)
            .ok_or_else(|| GraphError::Invalid("suspension frame is missing".into()))?;
        if suspended_frame.graph_index != suspension.graph_index {
            return Err(GraphError::Invalid(format!(
                "suspension graph index {} does not match frame graph index {}",
                suspension.graph_index, suspended_frame.graph_index
            )));
        }
        let graph = self
            .callables
            .get(suspension.graph_index)
            .ok_or_else(|| GraphError::Invalid("suspension graph index is invalid".into()))?;
        let node = graph
            .nodes
            .get(suspension.node.0)
            .filter(|node| node.id == suspension.node)
            .ok_or(GraphError::MissingNode(suspension.node))?
            .clone();
        if !node.can_suspend {
            return Err(GraphError::Invalid(format!(
                "suspended node '{}' cannot suspend",
                node.name
            )));
        }
        Ok((frame_index, node))
    }
}

fn as_snapshot_validation_error(error: GraphError) -> GraphError {
    match error {
        GraphError::SnapshotValidation(_) | GraphError::UnsupportedVersion { .. } => error,
        other => GraphError::SnapshotValidation(other.to_string()),
    }
}

#[cfg(test)]
#[path = "tests/runtime.rs"]
mod preparation_tests;
