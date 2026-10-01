use super::*;
use pravah::clients::Message;
use pravah::{Agent, AgentConfig};

fn configured(root: Agent<String>) -> Agent<String> {
    root.configure(configure)
}

async fn configure(input: String, _ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(AgentConfig::new(
        "test:///test",
        "Answer briefly.",
        Message::user(input).with_key("configured"),
    )
    .key("conversation"))
}

/// Explicit keys override function configuration for one submission, not later turns.
#[tokio::test]
async fn function_chat_keys_do_not_leak() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!("one"))
        .then_output(serde_json::json!("two"));
    let mut chat = Chat::new(configured, context(&factory)?)?;
    chat.send_with_key("first", "explicit").await?;
    let snapshot = chat.snapshot()?;
    let mut chat = Chat::<String, String>::from_snapshot(configured, snapshot, context(&factory)?)?;
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

/// Structural settings affect fingerprints, while instructions and initial state do not.
#[tokio::test]
async fn restore_requires_same_definition() -> Result<(), TestError> {
    let chat = builder().state(1u32).build(Context::default())?;
    let other = builder()
        .instructions("different")
        .state(2u32)
        .build(Context::default())?;
    let one = serde_json::to_value(chat.snapshot()?)?;
    let two = serde_json::to_value(other.snapshot()?)?;
    assert_eq!(one["graph_fingerprint"], two["graph_fingerprint"]);
    let changed = [
        builder().model("openai:///different"),
        builder().max_output_tokens(512),
        builder().turn_budget(2),
        builder().provider_config(serde_json::json!({"temperature": 0.5})),
        builder().resources([pravah::McpResourceRef::new("docs", "docs://guide")]),
        builder().tools(super::tools::toolset),
        builder().control(control),
    ];
    for definition in changed {
        assert!(matches!(
            definition.restore::<u32>(chat.snapshot()?, Context::default()),
            Err(GraphError::GraphMismatch { .. })
        ));
    }
    Ok(())
}

/// Tool budgets remain part of the definition even when instruction text is replaceable.
#[tokio::test]
async fn tool_budget_changes_remain_incompatible() -> Result<(), TestError> {
    let chat = builder()
        .tools(super::tools::toolset)
        .tool_budget::<super::tools::Search>(1)
        .build(Context::default())?;
    let result = builder()
        .tools(super::tools::toolset)
        .tool_budget::<super::tools::Search>(2)
        .restore::<()>(chat.snapshot()?, Context::default());
    assert!(matches!(result, Err(GraphError::GraphMismatch { .. })));
    Ok(())
}

async fn control(
    _: pravah::AgentLoop<String>,
    _: Context,
) -> Result<pravah::AgentDecision, GraphError> {
    Ok(pravah::AgentDecision::continue_())
}

/// Fresh instructions reach later turns without replacing conversation or application state.
#[tokio::test]
async fn instructions_can_change_between_turns() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("one"));
    let mut chat = builder().state(42u32).build(context(&factory)?)?;
    chat.send_with_key("first", "user-1").await?;
    for snapshot in copies(&chat.snapshot()?)? {
        let factory = ScriptedFactory::new().then_output(serde_json::json!("two"));
        let mut restored = builder()
            .instructions("New instructions")
            .compactor(ExpectInstructions)
            .restore::<u32>(snapshot, context(&factory)?)?;
        assert!(factory.calls().is_empty());
        assert_eq!(restored.get()?, 42);
        restored.send("second").await?;
        let saved = restored.snapshot()?;
        assert_eq!(saved.history().entries().len(), 4);
        assert_eq!(
            saved.history().entries()[0].message.key.as_deref(),
            Some("user-1")
        );
        assert_eq!(factory.calls()[0].1.len(), 3);
    }
    Ok(())
}

struct ExpectInstructions;

impl pravah::Compactor for ExpectInstructions {
    type Error = std::convert::Infallible;

    /// Checks that the next turn sees only the replacement instructions.
    async fn compact(
        &self,
        request: pravah::CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<pravah::CompactionResult, Self::Error> {
        let preamble = request.options().preamble.as_deref().unwrap_or_default();
        assert!(preamble.contains("New instructions"));
        assert!(!preamble.contains("Answer briefly."));
        Ok(pravah::CompactionResult::default())
    }
}
