use async_trait::async_trait;

use crate::clients::{
    Client, ClientError, ClientFactory, ClientFactoryLayer, ClientOptions, ClientResponse,
    EmbedRequest, EmbedResponse, Message, Provider,
};

struct TracingClient {
    inner: Box<dyn Client>,
}

#[async_trait]
impl Client for TracingClient {
    fn provider(&self) -> Provider {
        self.inner.provider()
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        let provider = self.inner.provider();
        tracing::debug!(
            provider = %provider.as_str(),
            message_count = messages.len(),
            "client request"
        );
        tracing::trace!(
            provider = %provider.as_str(),
            messages = ?messages,
            last_message = ?messages.last(),
            "client request payload"
        );
        let result = self.inner.execute(messages).await;
        match &result {
            Ok(response) => {
                tracing::debug!(provider = %provider.as_str(), "client response");
                tracing::trace!(provider = %provider.as_str(), response = ?response, "client response payload");
            }
            Err(error) => {
                tracing::debug!(provider = %provider.as_str(), error = %error, "client error");
            }
        }
        result
    }

    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, ClientError> {
        let provider = self.inner.provider();
        tracing::debug!(
            provider = %provider.as_str(),
            input_len = request.input.len(),
            "embedding request"
        );
        let result = self.inner.embed(request).await;
        match &result {
            Ok(response) => {
                tracing::debug!(
                    provider = %provider.as_str(),
                    dimensions = response.values.len(),
                    "embedding response"
                );
            }
            Err(error) => {
                tracing::debug!(provider = %provider.as_str(), error = %error, "embedding error");
            }
        }
        result
    }
}

/// Client-factory wrapper that logs requests and responses at the client boundary.
pub struct TracingFactory<F: ClientFactory> {
    inner: F,
}

impl<F: ClientFactory> TracingFactory<F> {
    /// Wraps `inner` with request/response logging.
    pub fn new(inner: F) -> Self {
        Self { inner }
    }
}

impl<F: ClientFactory> ClientFactory for TracingFactory<F> {
    fn create(
        &self,
        model_url: &str,
        options: ClientOptions,
    ) -> Result<Box<dyn Client>, ClientError> {
        let inner = self.inner.create(model_url, options)?;
        Ok(Box::new(TracingClient { inner }))
    }
}

/// Layer that logs client requests and responses.
#[derive(Debug, Clone, Copy, Default)]
pub struct TracingLayer;

impl<F: ClientFactory> ClientFactoryLayer<F> for TracingLayer {
    type Factory = TracingFactory<F>;

    fn layer(self, inner: F) -> Self::Factory {
        TracingFactory::new(inner)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::clients::{ClientOutput, Message};
    use crate::flows::{RateLimit, RateLimitLayer, RetryConfig, RetryLayer};
    use tokio::time::Duration;

    #[derive(Clone)]
    struct FlakyFactory {
        failures_left: Arc<AtomicUsize>,
        attempts: Arc<AtomicUsize>,
    }

    struct FlakyClient {
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

    #[async_trait]
    impl Client for FlakyClient {
        fn provider(&self) -> Provider {
            Provider::OpenAi
        }

        async fn execute(&self, _messages: &[Message]) -> Result<ClientResponse, ClientError> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            let remaining = self.failures_left.load(Ordering::SeqCst);
            if remaining > 0 {
                self.failures_left.fetch_sub(1, Ordering::SeqCst);
                return Err(ClientError::Llm("transient failure".into()));
            }
            Ok(ClientResponse::new(
                Provider::OpenAi,
                ClientOutput::Output(serde_json::json!({ "ok": true })),
            ))
        }
    }

    impl ClientFactory for FlakyFactory {
        fn create(
            &self,
            _model_url: &str,
            _options: ClientOptions,
        ) -> Result<Box<dyn Client>, ClientError> {
            Ok(Box::new(FlakyClient {
                failures_left: Arc::clone(&self.failures_left),
                attempts: Arc::clone(&self.attempts),
            }))
        }
    }

    /// Layers compose around one factory and retries recover transient failures.
    #[tokio::test]
    async fn layers_compose() {
        let base = FlakyFactory::new(1);
        let attempts = Arc::clone(&base.attempts);
        let factory = base
            .layer(TracingLayer)
            .layer(RetryLayer::new(RetryConfig::new(1, Duration::from_millis(1))))
            .layer(RateLimitLayer::new().with_limit(Provider::OpenAi, RateLimit::new(60_000, 4)));

        let client = factory
            .create("openai:///test-model", ClientOptions::default())
            .expect("layered factory should build a client");
        let response = client
            .execute(&[Message::user("hi")])
            .await
            .expect("retry layer should recover the transient failure");

        assert!(matches!(response.output, ClientOutput::Output(_)));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }
}