use super::*;
use pravah::graph::{JSON_WIRE_VERSION, JsonInvoker, JsonRequest, SNAPSHOT_VERSION};

/// JSON and CBOR preserve resolved limits and restore using fresh clients and preparation services.
#[tokio::test]
async fn snapshots_preserve_caps_with_fresh_dependencies() -> Result<(), TestError> {
    let flow = compile(workflow)?;
    let executor = flow.prepared().executor(Context::default());
    let mut original = flow.start(Request::capped(), uuid::Uuid::nil())?;
    let snapshot = activated(&mut original, &executor).await?;
    let json = serde_json::to_vec(&snapshot)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor)?;
    let copies: [Snapshot; 2] = [
        serde_json::from_slice(&json)?,
        ciborium::from_reader(cbor.as_slice())?,
    ];
    for copy in copies {
        let script = ScriptedFactory::new().then_output(json!("restored"));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut restored = flow.restore(copy)?;
        let executor = flow
            .prepared()
            .executor(context(script.clone(), Some(2048))?)
            .with_compactor(ObserveCap {
                calls: calls.clone(),
                cap: Some(2048),
            });
        finish(&mut restored, &executor).await?;
        assert_eq!(script.calls().len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

/// Corrupted cap values and obsolete checkpoints are rejected before model dispatch.
#[tokio::test]
async fn malformed_resolved_caps_and_old_checkpoints_are_rejected() -> Result<(), TestError> {
    let flow = compile(workflow)?;
    let executor = flow.prepared().executor(Context::default());
    let mut runtime = flow.start(Request::capped(), uuid::Uuid::nil())?;
    let snapshot = serde_json::to_value(activated(&mut runtime, &executor).await?)?;
    for cap in [
        json!(0),
        json!(-1),
        json!(4294967296_u64),
        json!(1.5),
        json!("2048"),
    ] {
        let mut corrupted = snapshot.clone();
        let cap_field = checkpoint_mut(&mut corrupted)?
            .pointer_mut("/resolved/max_output_tokens")
            .ok_or(TestError::Missing("resolved cap"))?;
        *cap_field = cap;
        assert!(matches!(
            flow.restore(serde_json::from_value(corrupted)?),
            Err(GraphError::SnapshotValidation(_))
        ));
    }
    let mut obsolete = snapshot.clone();
    *checkpoint_mut(&mut obsolete)?
        .get_mut("version")
        .ok_or(TestError::Missing("checkpoint version"))? = json!(4);
    assert!(matches!(
        flow.restore(serde_json::from_value(obsolete)?),
        Err(GraphError::UnsupportedVersion {
            format: "agent checkpoint",
            ..
        })
    ));
    assert_eq!(snapshot, serde_json::to_value(runtime.snapshot()?)?);
    Ok(())
}

/// Snapshot and wire version gates prevent older continuations from silently losing cap semantics.
#[tokio::test]
async fn old_snapshot_and_wire_formats_are_rejected() -> Result<(), TestError> {
    assert_eq!(SNAPSHOT_VERSION, 10);
    assert_eq!(JSON_WIRE_VERSION, 8);
    let flow = compile(workflow)?;
    let runtime = flow.start(Request::capped(), uuid::Uuid::nil())?;
    let mut snapshot = serde_json::to_value(runtime.snapshot()?)?;
    *snapshot
        .get_mut("snapshot_version")
        .ok_or(TestError::Missing("snapshot version"))? = json!(8);
    assert!(matches!(
        flow.restore(serde_json::from_value(snapshot)?),
        Err(GraphError::SnapshotVersion { got: 8, .. })
    ));
    let (graph, registry) = flow.into_parts();
    let invoker = JsonInvoker::new(graph, registry)?;
    assert!(matches!(
        invoker.invoke(JsonRequest::Start {
            execution_id: uuid::Uuid::nil(),
            version: 6,
            input: json!({})
        },),
        Err(GraphError::UnsupportedVersion { .. })
    ));
    Ok(())
}
