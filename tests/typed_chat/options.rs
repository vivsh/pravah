use super::*;
use pravah::clients::Message;
use pravah::tools::ToolError;
use pravah::{Agent, AgentConfig, AgentDecision, AgentLoop, McpResourceRef, ToolFilter, Toolset};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Search {
    query: String,
}

async fn search(input: Search, _: Context) -> Result<String, ToolError> {
    Ok(input.query)
}

fn tools(root: Toolset) -> Toolset {
    root.tool(search)
}

fn agent(root: Agent<Question>) -> Agent<String> {
    root.tools(tools).control(control).configure(configure)
}

async fn control(loop_: AgentLoop<Question>, _: Context) -> Result<AgentDecision, GraphError> {
    assert_eq!(loop_.input().depth, 2);
    assert_eq!(loop_.input().topic, "question");
    Ok(AgentDecision::continue_())
}

async fn configure(input: Question, _: Context) -> Result<AgentConfig, GraphError> {
    assert_eq!(input.depth, 2);
    Ok(AgentConfig::new(
        "test:///test",
        "Answer",
        Message::user(input.topic).with_key("configured"),
    )
    .memory("configured-memory")
    .tool_filter(ToolFilter::only(Vec::<String>::new()))
    .key("conversation"))
}

fn question() -> Question {
    Question {
        topic: "question".into(),
        depth: 2,
    }
}

fn resource_agent(root: Agent<Question>) -> Agent<String> {
    root.configure(configure_resources)
}

async fn configure_resources(input: Question, ctx: Context) -> Result<AgentConfig, GraphError> {
    Ok(configure(input, ctx)
        .await?
        .resources([McpResourceRef::new("unregistered", "docs://guide")]))
}

/// Function-defined resource defaults can be explicitly cleared without any network request.
#[tokio::test]
async fn function_resource_override() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("ok"));
    let mut chat = Chat::new(
        resource_agent,
        Context::default().with_providers(pravah::testing::providers(factory.clone())?),
    )?;
    chat.send(ChatRequest::from(question()).resources([]))
        .await?;
    assert_eq!(factory.calls().len(), 1);
    assert!(chat.send(question()).await.is_err());
    assert_eq!(factory.calls().len(), 1);
    assert_eq!(chat.snapshot()?.history().entries().len(), 2);
    Ok(())
}

struct VerifyOptions;

impl pravah::Compactor for VerifyOptions {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        request: pravah::CompactionRequest<'_>,
        _: Context,
    ) -> Result<pravah::CompactionResult, Self::Error> {
        let explicit = request
            .protected()
            .iter()
            .any(|entry| entry.message.key.as_deref() == Some("explicit"));
        let preamble = request.options().preamble.as_deref().unwrap_or_default();
        assert_eq!(preamble.contains("override-memory"), explicit);
        assert_eq!(preamble.contains("configured-memory"), !explicit);
        assert_eq!(request.options().tools.len(), usize::from(explicit));
        Ok(pravah::CompactionResult::default())
    }
}

/// Function callbacks receive domain input; request overrides are invocation-local and keyed.
#[tokio::test]
async fn function_overrides_and_omissions() -> Result<(), TestError> {
    let factory = ScriptedFactory::new()
        .then_output(serde_json::json!("one"))
        .then_output(serde_json::json!("two"));
    let mut chat = Chat::new(
        agent,
        Context::default().with_providers(pravah::testing::providers(factory.clone())?),
    )?
    .with_compactor(VerifyOptions);
    chat.send_with_key(
        ChatRequest::from(question())
            .memory("override-memory")
            .tools(["search"]),
        "explicit",
    )
    .await?;
    chat.send(question()).await?;
    let snapshot = chat.snapshot()?;
    assert_eq!(
        snapshot.history().entries()[0].message.key.as_deref(),
        Some("explicit")
    );
    assert_eq!(
        snapshot.history().entries()[2].message.key.as_deref(),
        Some("configured")
    );
    Ok(())
}

/// Both construction paths expose the unwrapped domain input to the same controller.
#[tokio::test]
async fn builder_typed_controller() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("ok"));
    let mut chat = Chat::builder::<Question, String>()
        .model("test:///test")
        .control(control)
        .build(Context::default().with_providers(pravah::testing::providers(factory)?))?;
    chat.send(ChatRequest::from(question()).memory("context"))
        .await?;
    Ok(())
}

/// Function chats perform the same preflight checks as builders without accepting invalid input.
#[tokio::test]
async fn function_preflight_is_atomic() -> Result<(), TestError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("ok"));
    let mut chat = Chat::new(
        agent,
        Context::default().with_providers(pravah::testing::providers(factory.clone())?),
    )?;
    let before = serde_json::to_value(chat.snapshot()?)?;
    for request in [
        ChatRequest::from(question()).tools(["unknown"]),
        ChatRequest::from(question()).tools(["search", "search"]),
        ChatRequest::from(question()).resources([McpResourceRef::new("", "invalid")]),
    ] {
        assert!(matches!(
            chat.send(request).await,
            Err(GraphError::ChatRequestValidation { .. })
        ));
        assert_eq!(serde_json::to_value(chat.snapshot()?)?, before);
    }
    assert!(factory.calls().is_empty());
    chat.send(question()).await?;
    Ok(())
}
