use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// A snapshot written by 0.4.25 restores with a new model and the same conversation and state.
#[tokio::test]
async fn prior_release_snapshot_keeps_conversation_and_state() -> Result<(), TestError> {
    let snapshot: Snapshot =
        serde_json::from_str(include_str!("fixtures/0.4.25-model-switch.json"))?;
    let original = serde_json::to_value(&snapshot)?;
    for snapshot in copies(&snapshot)? {
        let factory = ScriptedFactory::new().then_output(serde_json::json!("second"));
        let ctx = context(&factory)?.with_model_resolver(|declared| {
            assert_eq!(declared, "test:///test");
            Some("test:///replacement".into())
        });
        let mut chat = builder()
            .key("model-switch")
            .restore::<u32>(snapshot, ctx)?;
        assert_eq!(serde_json::to_value(chat.snapshot()?)?, original);
        assert_eq!(chat.get()?, 8);
        assert!(factory.calls().is_empty());
        assert_eq!(chat.send("next question").await?.output, "second");
        assert_eq!(chat.get()?, 8);
        let calls = factory.calls();
        let (model, messages) = calls.first().ok_or(TestError::Missing("generation call"))?;
        assert_eq!(model, "test:///replacement");
        assert!(
            messages
                .iter()
                .any(|msg| msg.key.as_deref() == Some("message-1"))
        );
        let saved = chat.snapshot()?;
        let entries = saved.history().entries();
        assert_eq!(entries.len(), 4);
        assert!(
            entries
                .iter()
                .all(|entry| entry.session_id == "key:model-switch")
        );
    }
    Ok(())
}

/// A live builder Chat can switch models on its next send without rebuilding or restoring.
#[tokio::test]
async fn next_send_reads_live_selection() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!("first"))
        .then_output(serde_json::json!("second"));
    let switched = Arc::new(AtomicBool::new(false));
    let selection = Arc::clone(&switched);
    let ctx = context(&factory)?.with_model_resolver(move |_| {
        selection
            .load(Ordering::Relaxed)
            .then(|| "test:///replacement".into())
    });
    let mut chat = builder().build(ctx)?;
    chat.send("first question").await?;
    switched.store(true, Ordering::Relaxed);
    chat.send("second question").await?;
    let models: Vec<_> = factory
        .calls()
        .into_iter()
        .map(|(model, _)| model)
        .collect();
    assert_eq!(models, ["test:///test", "test:///replacement"]);
    assert_eq!(chat.snapshot()?.history().entries().len(), 4);
    Ok(())
}
