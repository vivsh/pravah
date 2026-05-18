use async_trait::async_trait;
use tokio::time::Duration;

use crate::clients::{
    Client, ClientError, ClientFactory, ClientFactoryLayer, ClientOptions, ClientResponse,
    EmbedRequest, EmbedResponse, Message, Provider,
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
    pub fn new(max_retries: u32, initial_delay: Duration) -> Self {
        Self {
            max_retries,
            initial_delay,
            ..Default::default()
        }
    }

    pub fn with_backoff_factor(mut self, factor: f64) -> Self {
        self.backoff_factor = factor;
        self
    }

    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }
}

/// Returns `true` when the failure should be retried.
fn is_retryable(err: &ClientError) -> bool {
    matches!(
        err,
        ClientError::Llm(_) | ClientError::EmptyResponse | ClientError::Other(_)
    )
}

fn backoff_delay(config: &RetryConfig, attempt: u32) -> Duration {
    let secs = config.initial_delay.as_secs_f64()
        * config.backoff_factor.powi(attempt as i32);
    Duration::from_secs_f64(secs.min(config.max_delay.as_secs_f64()))
}

struct RetryingClient {
    inner: Box<dyn Client>,
    config: RetryConfig,
}

#[async_trait]
impl Client for RetryingClient {
    fn provider(&self) -> Provider {
        self.inner.provider()
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

    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, ClientError> {
        self.inner.embed(request).await
    }
}

/// Client-factory wrapper that retries transient LLM failures with exponential backoff.
/// Retries apply only to `execute()`.
pub struct RetryingFactory<F: ClientFactory> {
    inner: F,
    config: RetryConfig,
}

/// Layer that wraps clients with retry behavior.
#[derive(Debug, Clone, Default)]
pub struct RetryLayer {
    config: RetryConfig,
}

impl RetryLayer {
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }
}

impl<F: ClientFactory> RetryingFactory<F> {
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

impl<F: ClientFactory> ClientFactory for RetryingFactory<F> {
    fn create(
        &self,
        model_url: &str,
        options: ClientOptions,
    ) -> Result<Box<dyn Client>, ClientError> {
        let inner = self.inner.create(model_url, options)?;
        Ok(Box::new(RetryingClient {
            inner,
            config: self.config.clone(),
        }))
    }
}

impl<F: ClientFactory> ClientFactoryLayer<F> for RetryLayer {
    type Factory = RetryingFactory<F>;

    fn layer(self, inner: F) -> Self::Factory {
        RetryingFactory::new(inner).with_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `is_retryable` only accepts transient errors.
    #[test]
    fn test_retryable_variants() {
        assert!(is_retryable(&ClientError::Llm("rate limited".into())));
        assert!(is_retryable(&ClientError::EmptyResponse));
        assert!(!is_retryable(&ClientError::Validation("bad".into())));
        assert!(!is_retryable(&ClientError::InvalidUrl("x".into())));
        assert!(!is_retryable(&ClientError::Deserialize {
            source: serde_json::from_str::<()>("!").unwrap_err(),
            raw: "!".into(),
        }));
    }

    /// Backoff grows and then caps at `max_delay`.
    #[test]
    fn test_backoff_delay_growth() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_secs(1),
            backoff_factor: 2.0,
            max_delay: Duration::from_secs(10),
        };
        assert_eq!(backoff_delay(&config, 0), Duration::from_secs(1));
        assert_eq!(backoff_delay(&config, 1), Duration::from_secs(2));
        assert_eq!(backoff_delay(&config, 2), Duration::from_secs(4));
        assert_eq!(backoff_delay(&config, 3), Duration::from_secs(8));
        assert_eq!(backoff_delay(&config, 4), Duration::from_secs(10));
    }

    /// The default config matches the documented values.
    #[test]
    fn test_retry_config_defaults() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.initial_delay, Duration::from_secs(1));
        assert_eq!(cfg.backoff_factor, 2.0_f64);
        assert_eq!(cfg.max_delay, Duration::from_secs(30));
    }
}
