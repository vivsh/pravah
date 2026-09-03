use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::Context;
use crate::legacy::FlowHistory;
use crate::legacy::{HistoryCompactor, HistoryStore};

use super::agent::support::{validate_agent_snapshot_state, validate_agent_suspension};
use super::error::GraphError;
use super::ids::{EdgeId, HandlerKey, NodeId, VarId};
use super::model::{BuiltinNode, NodeKind, UntypedGraph, VarInit, VarKey, VarScope, Variable};
use super::registry::{
    ContinuationChildCall, ContinuationContext, ContinuationEvent, ContinuationTransition,
    HandlerRegistry, RuntimeServices,
};
use super::schema::validate_value;
use super::state::{
    ContinuationChildResult, Frame, ReturnTarget, State, Step, Suspension, SuspensionTarget,
};
use super::validation::{validate_graph_shape, validate_registry_keys};
use super::value::Value;

mod compile;
mod continuation;
mod dce;
mod execution;
mod fingerprint;
mod helpers;
mod liveness;
mod path;
mod reclaim;
mod snapshot;
mod sparse;

use compile::*;
use dce::DcePlan;
pub use fingerprint::GraphFingerprint;
use helpers::*;
use liveness::{LivenessPlan, ReleaseAction};
use path::{CallSite, GraphPath};
use reclaim::rebuild_reader_counts;
use snapshot::validate_snapshot_state;
use sparse::{SparseState, expand_state, sparse_state};

/// Current serialized runtime snapshot version.
pub const SNAPSHOT_VERSION: u32 = 8;

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
    WorkHandler {
        key: HandlerKey,
    },
    Continuation {
        key: HandlerKey,
        payload: Arc<Value>,
        children: Arc<[usize]>,
    },
    Suspend {
        payload: Arc<Value>,
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
    pub(crate) history: FlowHistory,
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
    pub fn history(&self) -> &FlowHistory {
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
    history: Arc<Mutex<FlowHistory>>,
    runtime_context: Arc<RuntimeContext>,
}

#[derive(Clone)]
struct RuntimeContext {
    services: Arc<RuntimeServices>,
}

impl RuntimeContext {
    fn new() -> Self {
        Self {
            services: Arc::new(RuntimeServices::default()),
        }
    }

    fn with_services(&self, services: RuntimeServices) -> Self {
        Self {
            services: Arc::new(services),
        }
    }
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
    /// Validates and compiles a graph and its handler registry once.
    ///
    /// The graph and registry are validated up front so missing handlers and
    /// malformed wiring fail before execution starts.
    pub fn new(graph: UntypedGraph, registry: HandlerRegistry) -> Result<Self, GraphError> {
        validate_graph_shape(&graph)?;
        let has_value = |key: &str| registry.has_value(key);
        let has_work = |key: &str| registry.has_work(key);
        let has_continuation = |key: &str| registry.has_continuation(key);
        validate_registry_keys(&graph, &has_value, &has_work, &has_continuation)?;
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

    /// Starts an isolated runtime using the already compiled graph.
    pub fn start(&self, input: Value) -> Result<Runtime, GraphError> {
        let mut state = State::default();
        let mut frame = new_frame(&self.callables, &[], self.root_index, None)?;
        let root_graph = self
            .callables
            .get(self.root_index)
            .ok_or_else(|| GraphError::Invalid("compiled root graph is missing".into()))?;
        let entry = root_graph.graph.entry;
        validate_edge_value(&root_graph.graph, entry, &input, "entry input")?;
        write_edge(&mut frame, entry, input)?;
        state.frames.push(frame);

        Ok(Runtime {
            callables: Arc::clone(&self.callables),
            registry: Arc::clone(&self.registry),
            graph_fingerprint: self.fingerprint,
            state,
            history: Arc::new(Mutex::new(FlowHistory::new())),
            runtime_context: Arc::new(RuntimeContext::new()),
        })
    }

    /// Restores an isolated runtime after checking version, graph, and VM state.
    ///
    /// Runtime services are intentionally not serialized; configure them again
    /// with the `with_*` methods after restore.
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
        validate_snapshot_state(&self.callables, self.root_index, &state)
            .map_err(as_snapshot_validation_error)?;
        rebuild_reader_counts(&self.callables, &mut state)?;

        Ok(Runtime {
            callables: Arc::clone(&self.callables),
            registry: Arc::clone(&self.registry),
            graph_fingerprint: self.fingerprint,
            state,
            history: Arc::new(Mutex::new(snapshot.history)),
            runtime_context: Arc::new(RuntimeContext::new()),
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
            | NodeKind::WorkHandler { .. }
            | NodeKind::Suspend { .. }
            | NodeKind::Load { .. }
            | NodeKind::Store { .. }
            | NodeKind::Goto { .. } => {}
        }
    }
    Ok(())
}

impl Runtime {
    /// Sets the history compactor used by runtime-owned history.
    pub fn with_compactor(mut self, compactor: impl HistoryCompactor + 'static) -> Self {
        let services = self
            .runtime_context
            .services
            .as_ref()
            .clone()
            .with_compactor(compactor);
        self.runtime_context = Arc::new(self.runtime_context.with_services(services));
        self
    }

    /// Sets the history store used when history is flushed.
    pub fn with_store(mut self, store: impl HistoryStore + 'static) -> Self {
        let services = self
            .runtime_context
            .services
            .as_ref()
            .clone()
            .with_store(store);
        self.runtime_context = Arc::new(self.runtime_context.with_services(services));
        self
    }

    pub(crate) fn with_runtime_services(mut self, services: RuntimeServices) -> Self {
        self.runtime_context = Arc::new(self.runtime_context.with_services(services));
        self
    }

    fn continuation_context(&self, ctx: Context) -> ContinuationContext {
        ContinuationContext::new(
            ctx,
            Arc::clone(&self.runtime_context.services),
            Arc::clone(&self.history),
        )
    }

    /// Returns the current VM state for inspection.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Returns the active external suspension, when the VM is waiting for input.
    pub fn suspension(&self) -> Option<&Suspension> {
        self.state.suspension.as_ref()
    }

    pub(crate) fn suspension_type_spec(&self) -> Result<&super::model::TypeSpec, GraphError> {
        let suspension = self
            .state
            .suspension
            .as_ref()
            .ok_or_else(|| GraphError::SnapshotValidation("runtime is not suspended".into()))?;
        Ok(&suspension.resume_type)
    }

    /// Captures versioned VM state and history tied to this graph's fingerprint.
    pub fn snapshot(&self) -> Result<Snapshot, GraphError> {
        let history = self
            .history
            .try_lock()
            .map_err(|_| GraphError::Invalid("runtime history is currently locked".into()))?
            .clone();
        Ok(Snapshot {
            snapshot_version: SNAPSHOT_VERSION,
            graph_fingerprint: self.graph_fingerprint,
            state: sparse_state(&self.callables, &self.state)?,
            history,
        })
    }

    /// Advances the VM by at most one dispatchable operation.
    ///
    /// Returns `ResumeRequired` if the VM is suspended; use `resume()` for a
    /// suspend node or continuation-owned suspension.
    pub async fn next(&mut self, ctx: Context) -> Result<Step, GraphError> {
        if self.state.suspension.is_some() {
            return Err(GraphError::ResumeRequired);
        }
        let step = self.step_inner(ctx).await?;
        match step {
            Step::Continue => self.try_exit_frames(),
            other => Ok(other),
        }
    }

    /// Supplies a value to the active node- or continuation-owned suspension.
    pub async fn resume(&mut self, value: Value, ctx: Context) -> Result<Step, GraphError> {
        let suspension = self
            .state
            .suspension
            .clone()
            .ok_or(GraphError::UnexpectedResume)?;
        validate_value(&suspension.resume_type, &value, "resume value")?;
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
        match suspension.target {
            SuspensionTarget::Node => self.resume_suspend_node(frame_index, &node, value),
            SuspensionTarget::Continuation => {
                self.resume_continuation(frame_index, node, value, ctx)
                    .await
            }
        }
    }
}

fn as_snapshot_validation_error(error: GraphError) -> GraphError {
    match error {
        GraphError::SnapshotValidation(_) | GraphError::UnsupportedVersion { .. } => error,
        other => GraphError::SnapshotValidation(other.to_string()),
    }
}

#[cfg(test)]
mod preparation_tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::graph::{NodeKind, TypeSpec, UntypedGraphBuilder};

    fn identity_graph(name: &str) -> UntypedGraph {
        let mut builder = UntypedGraphBuilder::new(name);
        let input = builder.edge("input", TypeSpec::new("Number", json!({"type": "number"})));
        let output = builder.edge("output", TypeSpec::new("Number", json!({"type": "number"})));
        builder.set_entry(input).set_exit(output);
        builder.node(
            "identity",
            NodeKind::Builtin {
                op: BuiltinNode::Identity,
            },
            vec![input],
            vec![output],
        );
        builder.build().expect("identity graph should build")
    }

    /// Builds a two-instruction graph that can be snapshotted between nodes.
    fn two_step_graph(name: &str) -> UntypedGraph {
        let mut builder = UntypedGraphBuilder::new(name);
        let number = TypeSpec::new("Number", json!({"type": "number"}));
        let input = builder.edge("input", number.clone());
        let middle = builder.edge("middle", number.clone());
        let output = builder.edge("output", number);
        builder.set_entry(input).set_exit(output);
        for (node_name, source, target) in [("first", input, middle), ("second", middle, output)] {
            builder.node(
                node_name,
                NodeKind::Builtin {
                    op: BuiltinNode::Identity,
                },
                vec![source],
                vec![target],
            );
        }
        builder.build().expect("two-step graph should build")
    }

    /// Verifies starts and restores reuse one immutable compilation.
    #[test]
    fn prepared_graph_shares_compilation_across_runtimes() {
        let prepared = PreparedGraph::new(identity_graph("shared"), HandlerRegistry::new())
            .expect("graph should prepare");
        let first = prepared.start(Value::from(1_i64)).expect("first runtime");
        let second = prepared.start(Value::from(2_i64)).expect("second runtime");
        assert!(Arc::ptr_eq(&first.callables, &second.callables));

        let snapshot = first.snapshot().expect("snapshot should build");
        let restored = prepared.restore(snapshot).expect("snapshot should restore");
        assert!(Arc::ptr_eq(&first.callables, &restored.callables));
        assert_ne!(
            first.state.values_for_test(),
            second.state.values_for_test()
        );
    }

    /// Verifies repeated preparation yields identical paths and release tables.
    #[test]
    fn prepared_metadata_is_deterministic() {
        let first = PreparedGraph::new(two_step_graph("plan"), HandlerRegistry::new())
            .expect("first graph should prepare");
        let second = PreparedGraph::new(two_step_graph("plan"), HandlerRegistry::new())
            .expect("second graph should prepare");
        assert_eq!(first.callables.len(), second.callables.len());
        for (left, right) in first.callables.iter().zip(second.callables.iter()) {
            assert_eq!(left.path, right.path);
            assert_eq!(left.liveness, right.liveness);
            assert_eq!(left.instructions, right.instructions);
            assert_eq!(left.nodes.len(), right.nodes.len());
            for (left_node, right_node) in left.nodes.iter().zip(right.nodes.iter()) {
                assert_eq!(left_node.id, right_node.id);
                assert_eq!(left_node.release_actions, right_node.release_actions);
            }
        }
    }

    /// Verifies snapshots contain graph identity but never embed graph structure.
    #[test]
    fn snapshot_is_state_only_and_round_trips_through_cbor() {
        let prepared = PreparedGraph::new(identity_graph("state_only"), HandlerRegistry::new())
            .expect("graph should prepare");
        let snapshot = prepared
            .start(Value::from(3_i64))
            .expect("runtime should start")
            .snapshot()
            .expect("snapshot should build");
        let json = serde_json::to_value(&snapshot).expect("snapshot should encode");
        assert!(json.get("graph").is_none());
        assert_eq!(
            json["graph_fingerprint"],
            prepared.fingerprint().to_string()
        );
        let json_snapshot: Snapshot =
            serde_json::from_value(json).expect("snapshot JSON should decode");
        prepared
            .restore(json_snapshot)
            .expect("JSON snapshot should restore");

        let mut encoded = Vec::new();
        ciborium::into_writer(&snapshot, &mut encoded).expect("snapshot CBOR should encode");
        let decoded: Snapshot =
            ciborium::from_reader(encoded.as_slice()).expect("snapshot CBOR should decode");
        prepared
            .restore(decoded)
            .expect("CBOR snapshot should restore");
    }

    /// Verifies a snapshot cannot be restored against a different trusted graph.
    #[test]
    fn restore_rejects_graph_fingerprint_mismatch() {
        let first = PreparedGraph::new(identity_graph("first"), HandlerRegistry::new())
            .expect("first graph should prepare");
        let second = PreparedGraph::new(identity_graph("second"), HandlerRegistry::new())
            .expect("second graph should prepare");
        let snapshot = first
            .start(Value::from(1_i64))
            .expect("runtime should start")
            .snapshot()
            .expect("snapshot should build");
        assert!(matches!(
            second.restore(snapshot),
            Err(GraphError::GraphMismatch { .. })
        ));
    }

    /// Verifies unsupported snapshot versions fail before restoration begins.
    #[test]
    fn restore_rejects_unsupported_snapshot_version() {
        let prepared = PreparedGraph::new(identity_graph("versioned"), HandlerRegistry::new())
            .expect("graph should prepare");
        let mut snapshot = prepared
            .start(Value::from(1_i64))
            .expect("runtime should start")
            .snapshot()
            .expect("snapshot should build");
        snapshot.snapshot_version = 999;

        assert!(matches!(
            prepared.restore(snapshot),
            Err(GraphError::SnapshotVersion { got: 999, .. })
        ));
    }

    /// Verifies version seven snapshots are rejected rather than migrated.
    #[test]
    fn restore_rejects_previous_snapshot_format() {
        let prepared = PreparedGraph::new(identity_graph("old_version"), HandlerRegistry::new())
            .expect("graph should prepare");
        let mut snapshot = prepared
            .start(Value::from(1_i64))
            .expect("runtime should start")
            .snapshot()
            .expect("snapshot should build");
        snapshot.snapshot_version = 7;

        assert!(matches!(
            prepared.restore(snapshot),
            Err(GraphError::SnapshotVersion {
                got: 7,
                expected: SNAPSHOT_VERSION
            })
        ));
    }

    /// Verifies one in-memory state has deterministic JSON and CBOR encodings.
    #[test]
    fn snapshot_encoding_is_deterministic() {
        let prepared = PreparedGraph::new(identity_graph("deterministic"), HandlerRegistry::new())
            .expect("graph should prepare");
        let runtime = prepared
            .start(Value::from(1_i64))
            .expect("runtime should start");
        let first = runtime.snapshot().expect("first snapshot should build");
        let second = runtime.snapshot().expect("second snapshot should build");
        assert_eq!(
            serde_json::to_vec(&first).expect("first JSON"),
            serde_json::to_vec(&second).expect("second JSON")
        );
        assert_eq!(encode_cbor(&first), encode_cbor(&second));
    }

    /// Verifies malformed sparse stable identities and epochs fail restoration.
    #[tokio::test]
    async fn restore_rejects_malformed_sparse_entries() {
        let prepared = PreparedGraph::new(two_step_graph("sparse_errors"), HandlerRegistry::new())
            .expect("graph should prepare");
        let mut runtime = prepared
            .start(Value::from(1_i64))
            .expect("runtime should start");
        runtime
            .next(Context::default())
            .await
            .expect("identity should run");
        let snapshot = runtime.snapshot().expect("snapshot should build");
        for (label, corruption) in malformed_snapshots(snapshot) {
            assert!(
                matches!(
                    prepared.restore(corruption),
                    Err(GraphError::SnapshotValidation(_))
                ),
                "{label} corruption should be rejected"
            );
        }
    }

    /// Builds a concrete table covering sparse IDs, ordering, epochs, and paths.
    fn malformed_snapshots(snapshot: Snapshot) -> Vec<(&'static str, Snapshot)> {
        let mut corruptions = malformed_edge_snapshots(&snapshot);
        corruptions.extend(malformed_node_snapshots(&snapshot));

        let mut bad_variable = snapshot.clone();
        bad_variable.state.frame_mut(0).expect("frame").variables =
            Arc::from([sparse::SparseVariable {
                variable: VarId(0),
                epoch: 1,
                value: Value::from(1_i64),
            }]);
        corruptions.push(("variable range", bad_variable));

        let mut bad_path = snapshot;
        bad_path.state.frame_mut(0).expect("frame").graph_path =
            GraphPath::root().child(CallSite::Subflow { node: NodeId(99) });
        corruptions.push(("graph path", bad_path));
        corruptions
    }

    /// Builds corruptions for duplicate, unordered, invalid, and epoch edge data.
    fn malformed_edge_snapshots(snapshot: &Snapshot) -> Vec<(&'static str, Snapshot)> {
        let mut corruptions = Vec::new();
        let mut duplicate_edge = snapshot.clone();
        let edge = duplicate_edge.state.frames[0].edges[0].clone();
        let mut edges = duplicate_edge.state.frames[0].edges.to_vec();
        edges.push(edge);
        duplicate_edge.state.frame_mut(0).expect("frame").edges = Arc::from(edges);
        corruptions.push(("duplicate edge", duplicate_edge));

        let mut unordered_edge = snapshot.clone();
        let mut entries = unordered_edge.state.frames[0].edges.to_vec();
        let mut earlier = entries[0].clone();
        earlier.edge = EdgeId(0);
        entries.push(earlier);
        unordered_edge.state.frame_mut(0).expect("frame").edges = Arc::from(entries);
        corruptions.push(("unordered edge", unordered_edge));

        let mut bad_edge = snapshot.clone();
        let frame = bad_edge.state.frame_mut(0).expect("frame");
        Arc::make_mut(&mut frame.edges)[0].edge = EdgeId(99);
        corruptions.push(("edge range", bad_edge));
        let mut zero_epoch = snapshot.clone();
        let frame = zero_epoch.state.frame_mut(0).expect("frame");
        Arc::make_mut(&mut frame.edges)[0].epoch = 0;
        corruptions.push(("zero edge epoch", zero_epoch));
        corruptions
    }

    /// Builds corruptions for duplicate and out-of-range node activation data.
    fn malformed_node_snapshots(snapshot: &Snapshot) -> Vec<(&'static str, Snapshot)> {
        let mut corruptions = Vec::new();
        let mut duplicate_node = snapshot.clone();
        let activation = duplicate_node.state.frames[0].node_epochs[0].clone();
        let mut node_epochs = duplicate_node.state.frames[0].node_epochs.to_vec();
        node_epochs.push(activation);
        duplicate_node
            .state
            .frame_mut(0)
            .expect("frame")
            .node_epochs = Arc::from(node_epochs);
        corruptions.push(("duplicate node", duplicate_node));

        let mut bad_node = snapshot.clone();
        let frame = bad_node.state.frame_mut(0).expect("frame");
        Arc::make_mut(&mut frame.node_epochs)[0].node = NodeId(99);
        corruptions.push(("node range", bad_node));
        corruptions
    }

    /// Verifies graphs preserve their complete model through CBOR.
    #[test]
    fn graph_round_trips_through_cbor() {
        let graph = identity_graph("cbor_graph");
        let mut encoded = Vec::new();
        ciborium::into_writer(&graph, &mut encoded).expect("graph CBOR should encode");
        let decoded: UntypedGraph =
            ciborium::from_reader(encoded.as_slice()).expect("graph CBOR should decode");
        assert_eq!(decoded, graph);
    }

    fn encode_cbor(snapshot: &Snapshot) -> Vec<u8> {
        let mut encoded = Vec::new();
        ciborium::into_writer(snapshot, &mut encoded).expect("snapshot CBOR should encode");
        encoded
    }
}
