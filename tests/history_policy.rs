//! Initialization and restore invariants for execution-owned history intent.

use pravah::{Flow, GraphError, HistoryPolicy, compile};
use uuid::Uuid;

fn identity(root: Flow<String>) -> Flow<String> {
    root.map(|input| input)
}

/// History intent survives JSON and CBOR restoration without retaining services.
#[test]
fn policy_round_trips_without_worker_dependencies() -> Result<(), GraphError> {
    let flow = compile(identity)?;
    let policy = HistoryPolicy {
        persist: true,
        load: true,
        compact: true,
    };
    let runtime = flow
        .start("input".into(), Uuid::nil())?
        .with_history(policy)?;
    let snapshot = runtime.snapshot()?;
    let json = serde_json::to_vec(&snapshot).map_err(codec)?;
    let restored = flow.restore(serde_json::from_slice(&json).map_err(codec)?)?;
    assert_eq!(*restored.history_policy(), policy);
    let mut bytes = Vec::new();
    ciborium::into_writer(&snapshot, &mut bytes).map_err(codec)?;
    let restored = flow.restore(ciborium::from_reader(bytes.as_slice()).map_err(codec)?)?;
    assert_eq!(*restored.history_policy(), policy);
    Ok(())
}

/// Sparse snapshots omit zero progress and disabled intent without weakening non-default validation.
#[test]
fn default_progress_is_sparse_and_restores_exactly() -> Result<(), GraphError> {
    let flow = compile(identity)?;
    let snapshot = flow.start("input".into(), Uuid::nil())?.snapshot()?;
    let wire = serde_json::to_value(&snapshot).map_err(codec)?;
    for key in [
        "next_agent_sequence",
        "history_policy",
        "persisted_history_position",
        "loaded_conversation_keys",
    ] {
        assert!(wire["state"].get(key).is_none());
    }
    let restored = flow.restore(serde_json::from_value(wire.clone()).map_err(codec)?)?;
    assert_eq!(*restored.history_policy(), HistoryPolicy::default());
    assert_eq!(
        serde_json::to_value(restored.snapshot()?).map_err(codec)?,
        wire
    );
    let mut bytes = Vec::new();
    ciborium::into_writer(&snapshot, &mut bytes).map_err(codec)?;
    let restored = flow.restore(ciborium::from_reader(bytes.as_slice()).map_err(codec)?)?;
    assert_eq!(*restored.history_policy(), HistoryPolicy::default());
    Ok(())
}

/// An initialized execution cannot silently change its maintenance intent after stepping.
#[test]
fn policy_is_initialization_only() -> Result<(), GraphError> {
    let flow = compile(identity)?;
    let mut runtime = flow.start("input".into(), Uuid::nil())?;
    runtime.next()?;
    assert!(matches!(
        runtime.with_history(HistoryPolicy::default()),
        Err(GraphError::HistoryValidation(_))
    ));
    Ok(())
}

/// Unsupported old snapshots are rejected before their missing progress fields are expanded.
#[test]
fn prior_snapshot_version_is_rejected() -> Result<(), GraphError> {
    let flow = compile(identity)?;
    let mut snapshot = serde_json::to_value(flow.start("input".into(), Uuid::nil())?.snapshot()?)
        .map_err(codec)?;
    snapshot["snapshot_version"] = 12.into();
    assert!(matches!(
        flow.restore(serde_json::from_value(snapshot).map_err(codec)?),
        Err(GraphError::SnapshotVersion { .. })
    ));
    Ok(())
}

/// Restore rejects acknowledgements beyond history and malformed key ledgers without repair.
#[test]
fn corrupt_progress_is_rejected() -> Result<(), GraphError> {
    let flow = compile(identity)?;
    let snapshot = serde_json::to_value(flow.start("input".into(), Uuid::nil())?.snapshot()?)
        .map_err(codec)?;
    let mut invalid = snapshot.clone();
    invalid["state"]["persisted_history_position"] = 1.into();
    assert!(matches!(
        flow.restore(serde_json::from_value(invalid).map_err(codec)?),
        Err(GraphError::SnapshotValidation(_))
    ));
    let mut invalid = snapshot.clone();
    invalid["state"]["loaded_conversation_keys"] = serde_json::json!([""]);
    assert!(matches!(
        flow.restore(serde_json::from_value(invalid).map_err(codec)?),
        Err(GraphError::SnapshotValidation(_))
    ));
    for keys in [serde_json::json!(["a", "a"]), serde_json::json!(["b", "a"])] {
        let mut invalid = snapshot.clone();
        invalid["state"]["loaded_conversation_keys"] = keys;
        assert!(serde_json::from_value::<pravah::Snapshot>(invalid).is_err());
    }
    Ok(())
}

fn codec(error: impl std::fmt::Display) -> GraphError {
    GraphError::Invalid(error.to_string())
}
