#[cfg(test)]
use crate::clients::ErrorKind;

use crate::clients::{
    Client, ClientError, ClientOptions, ClientResponse, LlmBackend, Message, ModelUrl,
    ProviderFactory,
};
struct TracingClient {
    inner: Client,
}

impl LlmBackend for TracingClient {
    fn model_url(&self) -> &ModelUrl {
        self.inner.model_url()
    }

    fn options(&self) -> &crate::clients::ClientOptions {
        self.inner.options()
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
}

/// Client-factory wrapper that logs requests and responses at the client boundary.
pub struct TracingFactory<F: ProviderFactory> {
    inner: F,
}

impl<F: ProviderFactory> TracingFactory<F> {
    /// Wraps `inner` with request/response logging.
    pub fn new(inner: F) -> Self {
        Self { inner }
    }
}

impl<F: ProviderFactory> ProviderFactory for TracingFactory<F> {
    async fn llm(
        &self,
        model_url: &ModelUrl,
        options: ClientOptions,
    ) -> Result<Client, ClientError> {
        let inner = self.inner.llm(model_url, options).await?;
        Ok(Client::from_backend(TracingClient { inner }))
    }
}

/// Layer that logs client requests and responses.
#[derive(Debug, Clone, Copy, Default)]
pub struct TracingLayer;

impl TracingLayer {
    /// Wraps a provider factory with this execution policy.
    pub fn layer<F: ProviderFactory>(self, inner: F) -> TracingFactory<F> {
        TracingFactory::new(inner)
    }
}

#[cfg(test)]
#[path = "tests/tracing.rs"]
mod tests;
