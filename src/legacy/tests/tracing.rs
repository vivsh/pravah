use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::clients::{ClientOutput, Message, Provider};
use crate::legacy::{RateLimit, RateLimitLayer, RetryConfig, RetryLayer};
use tokio::time::Duration;

#[derive(Clone)]
struct FlakyFactory {
    failures_left: Arc<AtomicUsize>,
    attempts: Arc<AtomicUsize>,
}

struct FlakyClient {
    url: crate::clients::ModelUrl,
    failures_left: Arc<AtomicUsize>,
    attempts: Arc<AtomicUsize>,
}

impl FlakyFactory {
    fn new(failures: usize) -> Self {
        Self {
            failures_left: Arc::new(AtomicUsize::new(failures)),
            attempts: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl LlmBackend for FlakyClient {
    fn model_url(&self) -> &crate::clients::ModelUrl {
        &self.url
    }

    fn options(&self) -> &crate::clients::ClientOptions {
        static OPTS: std::sync::OnceLock<crate::clients::ClientOptions> =
            std::sync::OnceLock::new();
        OPTS.get_or_init(crate::clients::ClientOptions::default)
    }

    async fn execute(&self, _messages: &[Message]) -> Result<ClientResponse, ClientError> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let remaining = self.failures_left.load(Ordering::SeqCst);
        if remaining > 0 {
            self.failures_left.fetch_sub(1, Ordering::SeqCst);
            return Err(ClientError::new(ErrorKind::Provider, "transient failure"));
        }
        Ok(ClientResponse::new(
            Provider::OpenAi,
            ClientOutput::Output(serde_json::json!({ "ok": true })),
        ))
    }
}

impl ProviderFactory for FlakyFactory {
    async fn llm(
        &self,
        model_url: &ModelUrl,
        _options: ClientOptions,
    ) -> Result<Client, ClientError> {
        let url = model_url.clone();
        Ok(Client::from_backend(FlakyClient {
            url,
            failures_left: Arc::clone(&self.failures_left),
            attempts: Arc::clone(&self.attempts),
        }))
    }
}

/// Layers compose around one factory and retries recover transient failures.
#[tokio::test]
async fn layers_compose() -> Result<(), ClientError> {
    let base = FlakyFactory::new(1);
    let attempts = Arc::clone(&base.attempts);
    let factory = TracingLayer.layer(base);
    let factory = RetryLayer::new(RetryConfig::new(1, Duration::from_millis(1))).layer(factory);
    let factory = RateLimitLayer::new()
        .with_limit(Provider::External("test".into()), RateLimit::new(60_000, 4))
        .layer(factory);

    let client = crate::clients::ProviderRegistry::new()
        .register("test", factory)?
        .llm("test:///test-model", ClientOptions::default())
        .await?;
    let response = client
        .execute(&[Message::user("hi")])
        .await
        .expect("retry layer should recover the transient failure");

    assert!(matches!(response.output, ClientOutput::Output(_)));
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    Ok(())
}
