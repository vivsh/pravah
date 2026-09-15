use super::super::*;
use crate::testing::ScriptedFactory;

fn builder() -> ChatBuilder<String> {
    Chat::builder()
        .model("openai:///test")
        .instructions("Answer briefly")
}

/// Round-trips an unfinished continuation through each supported test codec.
fn snapshots(snapshot: &Snapshot) -> Result<[Snapshot; 2], GraphError> {
    let json = serde_json::from_value(
        serde_json::to_value(snapshot)
            .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?,
    )
    .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
    let mut cbor = Vec::new();
    ciborium::into_writer(snapshot, &mut cbor)
        .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
    let cbor = ciborium::from_reader(cbor.as_slice())
        .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
    Ok([json, cbor])
}

/// Pre-activation submissions and committed configuration both survive restoration exactly once.
#[tokio::test]
async fn restores_each_configuration_boundary() -> Result<(), GraphError> {
    for committed in [false, true] {
        let original = ScriptedFactory::new();
        let mut chat = builder()
            .build(Context::default().with_client_factory(original.clone()))
            .await?;
        chat.runtime
            .resume(ChatSubmission {
                input: ChatRequest::from("question").memory("memory"),
                key: Some("durable".into()),
            })
            .await?;
        if committed {
            chat.runtime.next().await?;
            assert_eq!(chat.snapshot()?.history().entries().len(), 1);
        }
        assert!(original.calls().is_empty());
        for snapshot in snapshots(&chat.snapshot()?)? {
            complete_restored(snapshot).await?;
        }
    }
    Ok(())
}

/// Drives the preserved continuation privately without broadening the public Chat lifecycle API.
async fn complete_restored(snapshot: Snapshot) -> Result<(), GraphError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let mut restored = builder().restore::<()>(
        snapshot,
        Context::default().with_client_factory(factory.clone()),
    )?;
    assert!(matches!(
        restored.send("another").await,
        Err(GraphError::ChatNotReady { .. })
    ));
    let mut complete = false;
    for _ in 0..20 {
        if let Step::Suspend(_) = restored.runtime.next().await? {
            complete = true;
            break;
        }
    }
    assert!(complete);
    assert_eq!(factory.calls().len(), 1);
    let snapshot = restored.snapshot()?;
    assert_eq!(snapshot.history().entries().len(), 2);
    assert_eq!(
        snapshot.history().entries()[0].message.key.as_deref(),
        Some("durable")
    );
    Ok(())
}

/// Semantic corruption in either a retained input edge or active checkpoint is rejected on restore.
#[tokio::test]
async fn rejects_corrupt_persisted_requests() -> Result<(), GraphError> {
    for committed in [false, true] {
        let mut chat = builder().build(Context::default()).await?;
        chat.runtime
            .resume(ChatSubmission {
                input: ChatRequest::from("question"),
                key: None,
            })
            .await?;
        if committed {
            chat.runtime.next().await?;
        }
        let mut json = serde_json::to_value(chat.snapshot()?)
            .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
        assert!(corrupt_message(&mut json));
        let snapshot = serde_json::from_value(json)
            .map_err(|e| GraphError::SnapshotValidation(e.to_string()))?;
        assert!(
            builder()
                .restore::<()>(snapshot, Context::default())
                .is_err()
        );
    }
    Ok(())
}

/// Finds serialized request values by their schema fields, independent of numeric graph IDs.
fn corrupt_message(value: &mut serde_json::Value) -> bool {
    if let Some(message) = value.get_mut("message") {
        message["role"]["role"] = serde_json::json!("assistant");
        return true;
    }
    let mut found = false;
    match value {
        serde_json::Value::Object(fields) => {
            for value in fields.values_mut() {
                found |= corrupt_message(value);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                found |= corrupt_message(value);
            }
        }
        _ => {}
    }
    found
}
