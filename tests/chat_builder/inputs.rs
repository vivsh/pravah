use super::*;
use pravah::McpResourceRef;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountedInput<'a>(&'a AtomicUsize);

impl From<CountedInput<'_>> for ChatRequest {
    fn from(input: CountedInput<'_>) -> Self {
        input.0.fetch_add(1, Ordering::Relaxed);
        ChatRequest::from("question")
    }
}

/// Conversion happens once for accepted input and never for an unfinished turn's rejection.
#[tokio::test]
async fn conversion_follows_readiness_check() -> Result<(), TestError> {
    let conversions = AtomicUsize::new(0);
    let factory = ScriptedFactory::new();
    let mut chat = builder().build(context(&factory)).await?;
    assert!(chat.send(CountedInput(&conversions)).await.is_err());
    assert_eq!(conversions.load(Ordering::Relaxed), 1);
    let before = serde_json::to_value(chat.snapshot()?)?;
    assert!(matches!(
        chat.send_with_key(CountedInput(&conversions), "key").await,
        Err(GraphError::ChatNotReady { .. })
    ));
    assert_eq!(conversions.load(Ordering::Relaxed), 1);
    assert_eq!(serde_json::to_value(chat.snapshot()?)?, before);
    Ok(())
}

/// Both submission methods accept borrowed/owned text, raw messages and enriched requests.
#[tokio::test]
async fn submissions_accept_convertible_inputs() -> Result<(), TestError> {
    let mut factory = ScriptedFactory::new();
    for _ in 0..8 {
        factory = factory.then_output(serde_json::json!("ok"));
    }
    let mut chat = builder().build(context(&factory)).await?;
    chat.send("borrowed").await?;
    chat.send(String::from("owned")).await?;
    chat.send(Message::user("raw").with_key("configured"))
        .await?;
    chat.send(ChatRequest::from("request").memory("context"))
        .await?;
    chat.send_with_key("borrowed", "one").await?;
    chat.send_with_key(String::from("owned"), "two").await?;
    chat.send_with_key(Message::user("raw"), "three").await?;
    chat.send_with_key(ChatRequest::from("request"), "four")
        .await?;
    assert_eq!(factory.calls().len(), 8);
    let snapshot = chat.snapshot()?;
    let keys: Vec<_> = snapshot
        .history()
        .entries()
        .iter()
        .filter(|entry| matches!(entry.message.role, Role::User))
        .map(|entry| entry.message.key.as_deref())
        .collect();
    assert_eq!(
        keys,
        [
            None,
            None,
            Some("configured"),
            None,
            Some("one"),
            Some("two"),
            Some("three"),
            Some("four")
        ]
    );
    let before = serde_json::to_value(chat.snapshot()?)?;
    assert!(matches!(
        chat.send(Message::assistant("invalid")).await,
        Err(GraphError::ChatRequestValidation { .. })
    ));
    assert_eq!(serde_json::to_value(chat.snapshot()?)?, before);
    assert_eq!(factory.calls().len(), 8);
    Ok(())
}

/// Every supported attachment and optional message field agrees with Rath serialization.
#[test]
fn message_schema_matches_serde() -> Result<(), TestError> {
    let schema = serde_json::to_value(schemars::schema_for!(ChatRequest))?;
    let validator = jsonschema::validator_for(&schema)
        .map_err(|_| TestError::Missing("valid request schema"))?;
    for (kind, field) in [("inline", "data"), ("file", "path"), ("url", "url")] {
        let message: Message = serde_json::from_value(serde_json::json!({
            "role":{"role":"user"}, "content":"question", "key":"db:42",
            "attachments":[{"type":kind,"mime_type":"text/plain",(field):"example"}],
            "usage":{"input":u32::MAX,"output":null}
        }))?;
        let request = ChatRequest::new(message);
        let json = serde_json::to_value(&request)?;
        assert!(validator.is_valid(&json));
        let mut cbor = Vec::new();
        ciborium::into_writer(&request, &mut cbor)?;
        let restored: ChatRequest = ciborium::from_reader(cbor.as_slice())?;
        assert_eq!(serde_json::to_value(restored)?, json);
        let mut invalid = json;
        invalid["message"]["attachments"][0][field] = serde_json::json!(42);
        assert!(!validator.is_valid(&invalid));
    }
    Ok(())
}

/// Tool and resource preflight failures leave the exact ready snapshot unchanged.
#[tokio::test]
async fn invalid_selections_are_atomic() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("ok"));
    let mut chat = builder()
        .tools(super::tools::toolset)
        .build(context(&factory))
        .await?;
    let before = serde_json::to_value(chat.snapshot()?)?;
    let resource = McpResourceRef::new("docs", "docs://guide");
    for request in [
        ChatRequest::from("q").tools(["unknown"]),
        ChatRequest::from("q").tools(["search", "search"]),
        ChatRequest::from("q").resources([resource.clone(), resource]),
        ChatRequest::from("q").resources([McpResourceRef::new("", "invalid")]),
    ] {
        assert!(matches!(
            chat.send(request).await,
            Err(GraphError::ChatRequestValidation { .. })
        ));
        assert_eq!(serde_json::to_value(chat.snapshot()?)?, before);
    }
    assert!(factory.calls().is_empty());
    assert_eq!(chat.send("corrected").await?.output, "ok");
    Ok(())
}

/// Scalar caps replace prior values, whereas invalid or repeated budgets fail at construction.
#[tokio::test]
async fn builder_settings_validate_at_build() -> Result<(), TestError> {
    builder()
        .max_output_tokens(0)
        .max_output_tokens(256)
        .build(Context::default())
        .await?;
    for definition in [
        Chat::builder::<String>(),
        builder().max_output_tokens(0),
        builder().turn_budget(0),
        builder().turn_budget(1).turn_budget(2),
        builder().tool_budget::<super::tools::Search>(1),
        builder()
            .tools(super::tools::toolset)
            .tool_budget::<super::tools::Search>(1)
            .tool_budget::<super::tools::Search>(2),
        builder()
            .tools(super::tools::toolset)
            .tools(super::tools::toolset),
    ] {
        assert!(definition.build(Context::default()).await.is_err());
    }
    Ok(())
}

/// An explicit empty resource selection replaces defaults without touching the network.
#[tokio::test]
async fn resource_override_replaces_defaults() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let mut chat = builder()
        .resources([McpResourceRef::new("unregistered", "docs://guide")])
        .build(context(&factory))
        .await?;
    assert_eq!(
        chat.send(ChatRequest::from("q").resources([]))
            .await?
            .output,
        "answer"
    );
    let factory = ScriptedFactory::new();
    let mut chat = builder()
        .resources([McpResourceRef::new("unregistered", "docs://guide")])
        .build(context(&factory))
        .await?;
    assert!(chat.send(ChatRequest::from("q")).await.is_err());
    assert!(factory.calls().is_empty());
    assert!(chat.snapshot()?.history().entries().is_empty());
    Ok(())
}
