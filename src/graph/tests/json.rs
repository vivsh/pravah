use serde_json::json;

use super::*;
use crate::graph::{BuiltinNode, HandlerKey, NodeKind, TypeSpec, UntypedGraphBuilder};

fn suspended_graph() -> UntypedGraph {
    let mut builder = UntypedGraphBuilder::new("json_suspend");
    let input = builder.edge("input", TypeSpec::new("Number", json!({"type": "number"})));
    let waiting = builder.edge(
        "waiting",
        TypeSpec::new("Number", json!({"type": "number"})),
    );
    let output = builder.edge("output", TypeSpec::new("Number", json!({"type": "number"})));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "prepare",
        NodeKind::Builtin {
            op: BuiltinNode::Identity,
        },
        vec![input],
        vec![waiting],
    );
    builder.node(
        "approve",
        NodeKind::Suspend {
            resume_type: "Number".into(),
            payload: to_value(json!({"prompt": "replacement number"}))
                .expect("payload should enter runtime domain"),
        },
        vec![waiting],
        vec![output],
    );
    builder.build().expect("JSON test graph should build")
}

/// Verifies stateless JSON requests preserve one-step execution and resume state.
#[tokio::test]
async fn json_invocation_runs_start_next_resume_done() {
    let invoker =
        JsonInvoker::new(suspended_graph(), HandlerRegistry::new()).expect("invoker should build");
    let first = invoker
        .invoke(JsonRequest::Start {
            execution_id: uuid::Uuid::nil(),
            version: JSON_WIRE_VERSION,
            input: json!(1),
        })
        .expect("start should advance");
    let JsonResponse::Continue { snapshot, .. } = first else {
        panic!("start should execute only the prepare node");
    };
    let second = invoker
        .invoke(JsonRequest::Next {
            version: JSON_WIRE_VERSION,
            snapshot,
        })
        .expect("next should suspend");
    let JsonResponse::Suspend {
        snapshot,
        resume_type,
        payload,
        ..
    } = second
    else {
        panic!("next should reach suspension");
    };
    assert_eq!(resume_type, "Number");
    assert_eq!(payload, json!({"prompt": "replacement number"}));

    let done = invoker
        .invoke(JsonRequest::Resume {
            version: JSON_WIRE_VERSION,
            snapshot,
            input: json!(7),
        })
        .expect("resume should finish");
    let JsonResponse::Continue { snapshot, .. } = done else {
        panic!("resume accepts input without execution");
    };
    let done = invoker
        .invoke(JsonRequest::Next {
            version: JSON_WIRE_VERSION,
            snapshot,
        })
        .unwrap();
    assert!(matches!(done, JsonResponse::Done { output, .. } if output == json!(7)));
}

/// Verifies callers cannot substitute a different graph through a snapshot.
#[tokio::test]
async fn json_invocation_rejects_snapshot_graph_substitution() {
    let invoker =
        JsonInvoker::new(suspended_graph(), HandlerRegistry::new()).expect("invoker should build");
    let JsonResponse::Continue { mut snapshot, .. } = invoker
        .invoke(JsonRequest::Start {
            execution_id: uuid::Uuid::nil(),
            version: JSON_WIRE_VERSION,
            input: json!(1),
        })
        .expect("start should advance")
    else {
        panic!("start should continue");
    };
    let mut substituted = suspended_graph();
    substituted.name = "substituted".into();
    snapshot.graph_fingerprint = PreparedGraph::new(substituted, HandlerRegistry::new())
        .expect("substituted graph should prepare")
        .fingerprint();

    let err = invoker
        .invoke(JsonRequest::Next {
            version: JSON_WIRE_VERSION,
            snapshot,
        })
        .expect_err("substituted graph should fail");
    assert!(matches!(err, GraphError::GraphMismatch { .. }));
}

/// Verifies the JSON boundary performs full schema validation before execution.
#[tokio::test]
async fn json_invocation_rejects_invalid_external_value() {
    let invoker =
        JsonInvoker::new(suspended_graph(), HandlerRegistry::new()).expect("invoker should build");
    let err = invoker
        .invoke(JsonRequest::Start {
            execution_id: uuid::Uuid::nil(),
            version: JSON_WIRE_VERSION,
            input: json!("not a number"),
        })
        .expect_err("invalid input should fail");
    assert!(matches!(err, GraphError::Schema { .. }));
}

/// Verifies malformed JSON and unsupported wire versions fail explicitly.
#[tokio::test]
async fn json_invocation_rejects_bad_wire_input() {
    let invoker =
        JsonInvoker::new(suspended_graph(), HandlerRegistry::new()).expect("invoker should build");
    let malformed = invoker
        .invoke_str("{")
        .expect_err("malformed JSON should fail");
    assert!(matches!(malformed, GraphError::JsonDecode { .. }));

    let version = invoker
        .invoke(JsonRequest::Start {
            execution_id: uuid::Uuid::nil(),
            version: JSON_WIRE_VERSION + 1,
            input: json!(1),
        })
        .expect_err("unsupported version should fail");
    assert!(matches!(version, GraphError::UnsupportedVersion { .. }));

    let obsolete = invoker
        .invoke(JsonRequest::Start {
            execution_id: uuid::Uuid::nil(),
            version: 5,
            input: json!(1),
        })
        .expect_err("obsolete wire version should fail");
    assert!(matches!(obsolete, GraphError::UnsupportedVersion { .. }));
}

/// Verifies trusted graphs cannot reference handlers absent from the host registry.
#[test]
fn json_invoker_rejects_missing_handler() {
    let mut builder = UntypedGraphBuilder::new("missing_handler");
    let input = builder.edge("input", TypeSpec::new("Number", json!({"type": "number"})));
    let output = builder.edge("output", TypeSpec::new("Number", json!({"type": "number"})));
    builder.set_entry(input).set_exit(output);
    builder.node(
        "missing",
        NodeKind::PureHandler {
            key: HandlerKey::new("missing"),
        },
        vec![input],
        vec![output],
    );
    let graph = builder.build().expect("graph shape should be valid");

    assert!(JsonInvoker::new(graph, HandlerRegistry::new()).is_err());
}
