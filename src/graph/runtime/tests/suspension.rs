use super::super::*;
use crate::graph::{TypeSpec, UntypedGraphBuilder};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    CborWrite(#[from] ciborium::ser::Error<std::io::Error>),
    #[error(transparent)]
    CborRead(#[from] ciborium::de::Error<std::io::Error>),
    #[error("missing {0}")]
    Missing(&'static str),
}

/// Builds a suspend node with enough immutable schema text to expose accidental deep copies.
fn prepared() -> Result<PreparedGraph, GraphError> {
    let mut graph = UntypedGraphBuilder::new("shared-resume");
    let input = graph.edge("input", TypeSpec::new("Null", json!({"type":"null"})));
    let output = graph.edge(
        "output",
        TypeSpec::new(
            "String",
            json!({
                "type":"string", "description":"schema metadata ".repeat(20_000)
            }),
        ),
    );
    graph.set_entry(input).set_exit(output);
    graph.node(
        "pause",
        NodeKind::Suspend {
            payload: Value::NULL,
            resume_type: "String".into(),
        },
        vec![input],
        vec![output],
    );
    PreparedGraph::new(graph.build()?, HandlerRegistry::new())
}

fn suspended(runtime: &Runtime) -> Result<&Suspension, TestError> {
    runtime.suspension().ok_or(TestError::Missing("suspension"))
}

/// Independent runtimes and snapshots share prepared resume metadata without cloning its tree.
#[tokio::test]
async fn suspension_shares_prepared_metadata() -> Result<(), TestError> {
    let prepared = prepared()?;
    let mut first = prepared.start(Value::NULL, uuid::Uuid::nil())?;
    let mut second = prepared.start(Value::NULL, uuid::Uuid::nil())?;
    assert!(matches!(first.next()?, Step::Suspend(_)));
    assert!(matches!(second.next()?, Step::Suspend(_)));
    let original = suspended(&first)?;
    assert!(Arc::ptr_eq(
        &original.resume_type,
        &suspended(&second)?.resume_type
    ));
    let CompiledNodeKind::Suspend { resume_type, .. } = &prepared
        .callables
        .get(prepared.root_index)
        .and_then(|graph| graph.nodes.first())
        .ok_or(TestError::Missing("compiled node"))?
        .kind
    else {
        return Err(TestError::Missing("compiled suspend instruction"));
    };
    assert!(Arc::ptr_eq(&original.resume_type, resume_type));
    let allocations = allocation_counter::measure(|| {
        std::hint::black_box(original.clone());
    });
    assert_eq!(allocations.count_total, 0);
    let snapshot = first.snapshot()?;
    let Some(sparse::SparseWaiting::Suspend { suspension }) = &snapshot.state.waiting else {
        return Err(TestError::Missing("snapshot suspension"));
    };
    assert!(Arc::ptr_eq(&suspension.resume_type, resume_type));
    Ok(())
}

/// Arc ownership preserves the exact schema encoding and both snapshot codec restore paths.
#[tokio::test]
async fn shared_schema_preserves_wire_and_restore() -> Result<(), TestError> {
    let prepared = prepared()?;
    let mut runtime = prepared.start(Value::NULL, uuid::Uuid::nil())?;
    runtime.next()?;
    let schema = &suspended(&runtime)?.resume_type;
    assert_eq!(
        serde_json::to_vec(schema)?,
        serde_json::to_vec(schema.as_ref())?
    );
    let mut shared = Vec::new();
    let mut owned = Vec::new();
    ciborium::into_writer(schema, &mut shared)?;
    ciborium::into_writer(schema.as_ref(), &mut owned)?;
    assert_eq!(shared, owned);
    let snapshot = runtime.snapshot()?;
    let json = serde_json::from_slice(&serde_json::to_vec(&snapshot)?)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor)?;
    for snapshot in [json, ciborium::from_reader(cbor.as_slice())?] {
        let mut restored = prepared.restore(snapshot)?;
        assert_eq!(suspended(&restored)?.resume_type.as_ref(), schema.as_ref());
        restored.resume("accepted")?;
        assert!(matches!(restored.next()?, Step::Done(_)));
    }
    Ok(())
}

/// Failed resume validation and corrupted snapshot schemas leave the original suspension intact.
#[tokio::test]
async fn invalid_resume_and_schema_are_atomic() -> Result<(), TestError> {
    let prepared = prepared()?;
    let mut runtime = prepared.start(Value::NULL, uuid::Uuid::nil())?;
    runtime.next()?;
    let snapshot = runtime.snapshot()?;
    let before = serde_json::to_vec(&snapshot)?;
    assert!(runtime.resume(42u32).is_err());
    assert_eq!(serde_json::to_vec(&runtime.snapshot()?)?, before);
    let mut corrupt = snapshot.clone();
    let Some(sparse::SparseWaiting::Suspend { suspension }) = &mut corrupt.state.waiting else {
        return Err(TestError::Missing("snapshot suspension"));
    };
    let schema = &mut suspension.resume_type;
    Arc::make_mut(schema).name = "different".into();
    assert!(matches!(
        prepared.restore(corrupt),
        Err(GraphError::SnapshotValidation(_))
    ));
    assert_eq!(serde_json::to_vec(&snapshot)?, before);
    runtime.resume("accepted")?;
    assert!(matches!(runtime.next()?, Step::Done(_)));
    Ok(())
}
