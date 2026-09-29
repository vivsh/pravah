use tokio::time::Duration;

use crate::clients::{
    Client, ClientError, ClientOptions, ClientResponse, ErrorKind, LlmBackend, Message, ModelUrl,
    ProviderFactory,
};

/// Retry settings for transient client failures.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Number of retries after the first failure.
    pub max_retries: u32,
    /// Delay before the first retry.
    pub initial_delay: Duration,
    /// Multiplier applied after each retry.
    pub backoff_factor: f64,
    /// Maximum retry delay.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            backoff_factor: 2.0,
            max_delay: Duration::from_secs(30),
        }
    }
}

impl RetryConfig {
    /// Creates a config with the given retry count and initial delay.
    /// `backoff_factor` and `max_delay` are set to their defaults.
    pub fn new(max_retries: u32, initial_delay: Duration) -> Self {
        Self {
            max_retries,
            initial_delay,
            ..Default::default()
        }
    }

    /// Sets the exponential backoff multiplier applied after each failed attempt.
    pub fn with_backoff_factor(mut self, factor: f64) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Caps the delay between retries regardless of backoff growth.
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }
}

/// Preserves the legacy broad provider/transport retry policy using Rath classifications.
fn is_retryable(err: &ClientError) -> bool {
    matches!(
        err.kind(),
        ErrorKind::Provider
            | ErrorKind::Http
            | ErrorKind::Transport
            | ErrorKind::Timeout
            | ErrorKind::InvalidResponse
            | ErrorKind::Other
    )
}

fn backoff_delay(config: &RetryConfig, attempt: u32) -> Duration {
    let secs = config.initial_delay.as_secs_f64() * config.backoff_factor.powi(attempt as i32);
    Duration::from_secs_f64(secs.min(config.max_delay.as_secs_f64()))
}

struct RetryingClient {
    inner: Client,
    config: RetryConfig,
}

impl LlmBackend for RetryingClient {
    fn model_url(&self) -> &ModelUrl {
        self.inner.model_url()
    }

    fn options(&self) -> &crate::clients::ClientOptions {
        self.inner.options()
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        let mut attempt = 0u32;
        loop {
            match self.inner.execute(messages).await {
                Ok(response) => return Ok(response),
                Err(err) if attempt < self.config.max_retries && is_retryable(&err) => {
                    let delay = backoff_delay(&self.config, attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max = self.config.max_retries,
                        error = %err,
                        delay_ms = delay.as_millis(),
                        "retrying LLM call"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                Err(err) => return Err(err),
            }
        }
    }
}

/// Client-factory wrapper that retries transient LLM failures with exponential backoff.
/// Retries apply only to `execute()`.
pub struct RetryingFactory<F: ProviderFactory> {
    inner: F,
    config: RetryConfig,
}

/// Layer that wraps clients with retry behavior.
#[derive(Debug, Clone, Default)]
pub struct RetryLayer {
    config: RetryConfig,
}

impl RetryLayer {
    /// Creates a layer with the given retry configuration.
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }
}

impl<F: ProviderFactory> RetryingFactory<F> {
    /// Wraps `inner` with the default retry policy.
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            config: RetryConfig::default(),
        }
    }

    /// Replaces the retry policy.
    pub fn with_config(mut self, config: RetryConfig) -> Self {
        self.config = config;
        self
    }
}

impl<F: ProviderFactory> ProviderFactory for RetryingFactory<F> {
    async fn llm(
        &self,
        model_url: &ModelUrl,
        options: ClientOptions,
    ) -> Result<Client, ClientError> {
        let inner = self.inner.llm(model_url, options).await?;
        Ok(Client::from_backend(RetryingClient {
            inner,
            config: self.config.clone(),
        }))
    }
}

impl RetryLayer {
    /// Wraps a provider factory with this execution policy.
    pub fn layer<F: ProviderFactory>(self, inner: F) -> RetryingFactory<F> {
        RetryingFactory::new(inner).with_config(self.config)
    }
}

#[cfg(test)]
mod tests;
