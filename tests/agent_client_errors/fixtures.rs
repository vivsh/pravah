use super::*;
use pravah::clients::{Client, ClientOptions, Message, ModelUrl, ProviderFactory};

pub(super) const PRIVATE_BODY: &str = r#"{"error":{"message":"private prompt and generated content","code":"test_code"},"credential":"test-secret","output":"private answer"}"#;

#[derive(Debug, thiserror::Error)]
pub(super) enum TestError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Client(#[from] ClientError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Timeout(#[from] tokio::time::error::Elapsed),
    #[error("missing {0}")]
    Missing(&'static str),
}

struct CreationFailure(ClientError);

impl ProviderFactory for CreationFailure {
    async fn llm(&self, _: &ModelUrl, _: ClientOptions) -> Result<Client, ClientError> {
        Err(self.0.clone())
    }
}

pub(super) fn client_context(
    operation: AgentClientOperation,
    error: ClientError,
) -> Result<Context, pravah::GraphError> {
    Ok(match operation {
        AgentClientOperation::Create => {
            Context::default().with_providers(pravah::testing::providers(CreationFailure(error))?)
        }
        AgentClientOperation::Execute => Context::default().with_providers(
            pravah::testing::providers(ScriptedFactory::new().then_err(error))?,
        ),
    })
}

pub(super) fn flow(root: pravah::Flow<String>) -> pravah::Flow<String> {
    root.agent(agent)
}

fn agent(root: pravah::Agent<String>) -> pravah::Agent<String> {
    root.configure(configure)
}

async fn configure(input: String, _: Context) -> Result<pravah::AgentConfig, GraphError> {
    Ok(pravah::AgentConfig::new(
        "test:///test",
        "Answer.",
        Message::user(input),
    ))
}

/// Exercises the public keyed submission boundary without changing unfinished-turn behavior.
pub(super) async fn keyed_error(context: Context) -> Result<GraphError, GraphError> {
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .instructions("Answer.")
        .build(context)?;
    let error = chat
        .send_with_key("question", "message-42")
        .await
        .err()
        .ok_or_else(|| GraphError::Invalid("expected client failure".into()))?;
    let snapshot = chat.snapshot()?;
    assert_eq!(snapshot.history().entries().len(), 1);
    assert_eq!(
        snapshot
            .history()
            .entries()
            .first()
            .and_then(|entry| entry.message.key.as_deref()),
        Some("message-42")
    );
    assert!(matches!(
        chat.send_with_key("another", "message-43").await,
        Err(GraphError::ChatNotReady { .. })
    ));
    Ok(error)
}

/// Obtains private HTTP fields through Rath's public client API against a loopback fixture.
pub(super) async fn http_diagnostic() -> Result<ClientError, TestError> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let model = format!(
        "ollama:///fixture?base_url=http://{}",
        listener.local_addr()?
    );
    let client = ClientOptions::default().create(&model).await?;
    let app = axum::Router::new().fallback(|| async {
        (
            axum::http::StatusCode::BAD_REQUEST,
            [("x-request-id", "request-42"), ("retry-after", "17")],
            PRIVATE_BODY,
        )
    });
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client.execute(&[Message::user("test question")]),
    )
    .await;
    server.abort();
    let _ = server.await;
    let error = result?.err().ok_or(TestError::Missing("HTTP error"))?;
    Ok(error.with_source(
        ClientError::new(ErrorKind::Transport, "private cause")
            .with_source(ClientError::new(ErrorKind::Timeout, "private nested cause")),
    ))
}
