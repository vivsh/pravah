use super::*;
use pravah::MessageHistory;

struct InspectMessages;

impl Compactor for InspectMessages {
    type Error = serde_json::Error;

    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        if request.turn_count() == 0 {
            return Ok(CompactionResult::default());
        }
        assert_eq!(request.turn_count(), 1);
        let messages: Vec<_> = request.enum_messages(0).collect();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].1.key.as_deref(), Some("database:1"));
        assert!(request.byte_size()? > messages[0].1.content.len());
        Ok(CompactionResult {
            evict_indices: (0..request.committed().len()).collect(),
            summary: Some("Extracted memory".into()),
        })
    }
}

/// The renamed public API exposes keyed messages to compaction and preserves snapshot restoration.
#[tokio::test]
async fn chat_compactor_observes_messages_before_replacement() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!({"text":"first"}))
        .then_output(serde_json::json!({"text":"second"}));
    let mut chat = Chat::new(
        tutor,
        Context::default().with_providers(pravah::testing::providers(factory)?),
    )?
    .with_compactor(InspectMessages);
    chat.send_with_key(
        Question {
            text: "old question".into(),
        },
        "database:1",
    )
    .await?;
    let snapshot = chat.snapshot()?;
    let history: &MessageHistory = snapshot.history();
    let session = &history.entries()[0].session_id;
    assert_eq!(history.turn_count(session), 1);
    assert_eq!(history.enum_messages(session, 1).count(), 1);
    let factory = ScriptedFactory::new().then_output(serde_json::json!({"text":"second"}));
    let mut chat = Chat::<Question, Answer>::from_snapshot(
        tutor,
        snapshot,
        Context::default().with_providers(pravah::testing::providers(factory.clone())?),
    )?
    .with_compactor(InspectMessages);
    chat.send(Question {
        text: "new question".into(),
    })
    .await?;
    assert!(factory.calls()[0].1[0].content.contains("Extracted memory"));
    assert_eq!(chat.snapshot()?.history().entries().len(), 3);
    Ok(())
}
