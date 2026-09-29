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
    let first = prepared
        .start(Value::from(1_i64), uuid::Uuid::nil())
        .expect("first runtime");
    let second = prepared
        .start(Value::from(2_i64), uuid::Uuid::nil())
        .expect("second runtime");
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
        .start(Value::from(3_i64), uuid::Uuid::nil())
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
        .start(Value::from(1_i64), uuid::Uuid::nil())
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
        .start(Value::from(1_i64), uuid::Uuid::nil())
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
        .start(Value::from(1_i64), uuid::Uuid::nil())
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
        .start(Value::from(1_i64), uuid::Uuid::nil())
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
        .start(Value::from(1_i64), uuid::Uuid::nil())
        .expect("runtime should start");
    runtime.next().expect("identity should run");
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
