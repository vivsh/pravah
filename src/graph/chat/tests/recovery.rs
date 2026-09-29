use super::super::*;
use crate::testing::ScriptedFactory;

fn builder() -> ChatBuilder<String, String> {
    Chat::builder()
        .model("test:///test")
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

/// New prompts apply before configuration commits, never to an already-configured invocation.
#[tokio::test]
async fn restores_each_configuration_boundary() -> Result<(), GraphError> {
    for committed in [false, true] {
        let original = ScriptedFactory::new();
        let mut chat = builder().build(
            Context::default().with_providers(crate::testing::providers(original.clone())?),
        )?;
        let mut request = ChatRequest::<String>::from("question").memory("memory");
        request.key = Some("durable".into());
        chat.runtime.resume(request)?;
        if committed {
            while chat.snapshot()?.history().entries().is_empty() {
                crate::graph::tests::host::step(&mut chat.runtime, &chat.executor).await?;
            }
            assert_eq!(chat.snapshot()?.history().entries().len(), 1);
        }
        assert!(original.calls().is_empty());
        for snapshot in snapshots(&chat.snapshot()?)? {
            let expected = if committed {
                "Answer briefly"
            } else {
                "Updated instructions"
            };
            complete_restored(snapshot, expected).await?;
        }
    }
    Ok(())
}

/// Drives the preserved continuation privately without broadening the public Chat lifecycle API.
async fn complete_restored(snapshot: Snapshot, expected: &'static str) -> Result<(), GraphError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!("answer"))
        .then_output(serde_json::json!("next answer"));
    let mut restored = builder()
        .instructions("Updated instructions")
        .compactor(ExpectInstructions(expected))
        .restore::<()>(
            snapshot,
            Context::default().with_providers(crate::testing::providers(factory.clone())?),
        )?;
    assert!(matches!(
        restored.send("another").await,
        Err(GraphError::ChatNotReady { .. })
    ));
    let mut complete = false;
    for _ in 0..20 {
        if let Step::Suspend(_) =
            crate::graph::tests::host::step(&mut restored.runtime, &restored.executor).await?
        {
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
    let mut restored = restored.with_compactor(ExpectInstructions("Updated instructions"));
    restored.send("next question").await?;
    assert_eq!(factory.calls().len(), 2);
    assert_eq!(restored.snapshot()?.history().entries().len(), 4);
    Ok(())
}

struct ExpectInstructions(&'static str);

impl crate::Compactor for ExpectInstructions {
    type Error = std::convert::Infallible;

    /// Checks the effective client instructions after restore, before the model is dispatched.
    async fn compact(
        &self,
        request: crate::CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<crate::CompactionResult, Self::Error> {
        assert!(
            request
                .options()
                .preamble
                .as_deref()
                .unwrap_or_default()
                .contains(self.0)
        );
        Ok(crate::CompactionResult::default())
    }
}

/// Semantic corruption in either a retained input edge or active checkpoint is rejected on restore.
#[tokio::test]
async fn rejects_corrupt_persisted_requests() -> Result<(), GraphError> {
    for committed in [false, true] {
        let mut chat = builder().build(Context::default())?;
        chat.runtime
            .resume(ChatRequest::<String>::from("question"))?;
        if committed {
            while chat.snapshot()?.history().entries().is_empty() {
                crate::graph::tests::host::step(&mut chat.runtime, &chat.executor).await?;
            }
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
    if value.get("input").is_some_and(serde_json::Value::is_string) {
        value["tools"] = serde_json::json!(["undeclared"]);
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
