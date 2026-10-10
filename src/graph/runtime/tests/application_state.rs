use super::*;
use crate::{Flow, compile};
use serde::Deserialize;

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Application {
    owner: String,
    count: u64,
}

fn flow(root: Flow<u32>) -> Flow<u32> {
    root.map(|value| value + 1)
}

/// State survives root-frame exit and both snapshot codecs without a typed cache.
#[test]
fn completed_state_roundtrips() -> Result<(), GraphError> {
    let flow = compile(flow)?;
    let mut runtime = flow.start_with_state(
        1,
        Application {
            owner: "host".into(),
            count: 0,
        },
        Uuid::nil(),
    )?;
    runtime.set_state(Application {
        owner: "host".into(),
        count: 2,
    })?;
    assert!(matches!(runtime.next()?, Step::Done(_)));
    assert_eq!(runtime.state.frame_depth(), 0);
    let snapshot = runtime.snapshot()?;
    let json = serde_json::to_vec(&snapshot).map_err(codec_error)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor).map_err(codec_error)?;
    for snapshot in [
        serde_json::from_slice(&json).map_err(codec_error)?,
        ciborium::from_reader(cbor.as_slice()).map_err(codec_error)?,
    ] {
        let mut restored = flow.restore(snapshot)?;
        assert_eq!(restored.get_state::<Application>()?.count, 2);
        restored.set_state(Application {
            owner: "restored".into(),
            count: 3,
        })?;
        assert_eq!(restored.get_state::<Application>()?.owner, "restored");
    }
    Ok(())
}

/// Restoring completed snapshots still rejects invalid application values and schemas.
#[test]
fn corrupted_completed_state_is_rejected() -> Result<(), GraphError> {
    let flow = compile(flow)?;
    let mut runtime = flow.start_with_state(1, 7u32, Uuid::nil())?;
    runtime.next()?;
    let snapshot = runtime.snapshot()?;
    let mut corrupt = snapshot.clone();
    corrupt.state.application_state = Some((type_spec::<u32>(), Value::from("wrong")));
    assert!(matches!(
        flow.restore(corrupt),
        Err(GraphError::SnapshotValidation(_))
    ));
    let mut corrupt = snapshot;
    let (spec, _) = corrupt
        .state
        .application_state
        .as_mut()
        .ok_or_else(missing_state)?;
    spec.schema = serde_json::json!({"type": 7});
    assert!(matches!(
        flow.restore(corrupt),
        Err(GraphError::SnapshotValidation(_))
    ));
    Ok(())
}

fn codec_error(error: impl std::fmt::Display) -> GraphError {
    GraphError::SnapshotValidation(error.to_string())
}
