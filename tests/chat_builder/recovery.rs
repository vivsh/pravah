use super::*;
use pravah::{Agent, AgentConfig};

fn configured(root: Agent<String>) -> Agent<String> {
    root.configure(configure)
}

async fn configure(input: String, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "openai:///test",
        "Answer briefly.",
        Message::user(input).with_key("configured"),
    )
    .keep_alive())
}

/// Explicit keys override function configuration for one submission, not later turns.
#[tokio::test]
async fn function_chat_keys_do_not_leak() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!("one"))
        .then_output(serde_json::json!("two"));
    let mut chat = Chat::new(configured, context(&factory)).await?;
    chat.send_with_key("first", "explicit").await?;
    let snapshot = chat.snapshot()?;
    let mut chat = Chat::<String, String>::from_snapshot(configured, snapshot, context(&factory))?;
    chat.send("second").await?;
    let snapshot = chat.snapshot()?;
    let keys: Vec<_> = snapshot
        .history()
        .entries()
        .iter()
        .filter(|entry| matches!(entry.message.role, Role::User))
        .map(|entry| entry.message.key.as_deref())
        .collect();
    assert_eq!(keys, [Some("explicit"), Some("configured")]);
    Ok(())
}

/// Definition settings affect fingerprints, while application state does not.
#[tokio::test]
async fn restore_requires_same_definition() -> Result<(), TestError> {
    let chat = builder().state(1u32).build(Context::default()).await?;
    let other = builder().state(2u32).build(Context::default()).await?;
    let one = serde_json::to_value(chat.snapshot()?)?;
    let two = serde_json::to_value(other.snapshot()?)?;
    assert_eq!(one["graph_fingerprint"], two["graph_fingerprint"]);
    for snapshot in copies(&chat.snapshot()?)? {
        assert!(
            builder()
                .instructions("different")
                .restore::<u32>(snapshot, Context::default())
                .is_err()
        );
    }
    Ok(())
}
