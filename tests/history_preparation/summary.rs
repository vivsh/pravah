use super::*;
use pravah::Snapshot;

const MEMORY: &str = " \n京都: preserve this memory.\n ";

struct InspectSummary {
    expected: Option<&'static str>,
}

impl Compactor for InspectSummary {
    type Error = std::convert::Infallible;

    /// Treats previous memory as context while inspecting only completed conversation messages.
    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        assert_eq!(request.summary(), self.expected);
        if request.committed().is_empty() {
            return Ok(CompactionResult::default());
        }
        let offset = usize::from(self.expected.is_some());
        let messages: Vec<_> = request.enum_messages(0).collect();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].0, offset);
        assert_eq!(messages[1].0, offset + 1);
        assert!(matches!(messages[0].1.role, Role::User));
        assert!(matches!(messages[1].1.role, Role::Assistant));
        assert_eq!(request.enum_messages(1).count(), 1);
        assert_eq!(request.turn_count(), 1);
        Ok(CompactionResult {
            evict_indices: (0..request.committed().len()).collect(),
            summary: Some(MEMORY.into()),
        })
    }
}

/// Real compaction writes remain visible to the client and readable after either snapshot codec.
#[tokio::test]
async fn summary_is_separate_from_messages_across_restoration() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!({"text":"first"}))
        .then_output(serde_json::json!({"text":"second"}));
    let mut chat = Chat::new(
        tutor,
        Context::default().with_providers(pravah::testing::providers(factory)?),
    )?
    .with_compactor(InspectSummary { expected: None });
    for text in ["first", "second"] {
        chat.send(Question { text: text.into() }).await?;
    }
    let snapshot = chat.snapshot()?;
    let json = serde_json::to_vec(&snapshot).map_err(|e| GraphError::Invalid(e.to_string()))?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor).map_err(|e| GraphError::Invalid(e.to_string()))?;
    let copies: [Snapshot; 2] = [
        serde_json::from_slice(&json).map_err(|e| GraphError::Invalid(e.to_string()))?,
        ciborium::from_reader(cbor.as_slice()).map_err(|e| GraphError::Invalid(e.to_string()))?,
    ];
    for snapshot in copies {
        check_restored(snapshot).await?;
    }
    Ok(())
}

/// Snapshot history retains its summary but neither enumerator exposes it as conversation evidence.
async fn check_restored(snapshot: Snapshot) -> Result<(), GraphError> {
    let history = snapshot.history();
    let session = &history.entries()[0].session_id;
    assert_eq!(history.entries().len(), 3);
    assert_eq!(
        history
            .enum_messages(session, 0)
            .map(|(i, _)| i)
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let factory = ScriptedFactory::new().then_output(serde_json::json!({"text":"third"}));
    let mut chat = Chat::<Question, Answer>::from_snapshot(
        tutor,
        snapshot,
        Context::default().with_providers(pravah::testing::providers(factory.clone())?),
    )?
    .with_compactor(InspectSummary {
        expected: Some(MEMORY),
    });
    assert!(factory.calls().is_empty());
    chat.send(Question {
        text: "third".into(),
    })
    .await?;
    let calls = factory.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].1.len(), 2);
    assert!(matches!(calls[0].1[0].role, Role::System));
    assert_eq!(
        calls[0].1[0].content,
        format!("<pravah_working_memory>\n{MEMORY}\n</pravah_working_memory>")
    );
    assert_eq!(chat.snapshot()?.history().entries().len(), 3);
    Ok(())
}
