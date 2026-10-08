use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

use crate::clients::{
    Client, ClientError, ClientOptions, ClientResponse, ErrorKind, LlmBackend, Message, ModelUrl,
    Provider, ProviderFactory,
};

/// Per-provider rate-limit settings.
/// `rpm` sets the sustained rate and `burst` sets the immediate bucket depth.
#[derive(Debug, Clone, Copy)]
pub struct RateLimit {
    /// Sustained requests per minute; must be non-zero when creating a client.
    pub rpm: u32,
    /// Maximum immediate requests when the bucket is full; must be non-zero.
    pub burst: u32,
}

impl RateLimit {
    /// Builds a rate limit, validated when creating a client for the configured provider.
    /// Zero RPM or burst values produce a validation error during client creation.
    pub fn new(rpm: u32, burst: u32) -> Self {
        Self { rpm, burst }
    }
}

struct BucketState {
    tokens: f64,
    last_refill: Instant,
}

/// Token bucket for one provider.
struct TokenBucket {
    state: Mutex<BucketState>,
    label: String,
    /// Tokens added each second.
    refill_rate: f64,
    /// Bucket capacity.
    capacity: f64,
}

impl TokenBucket {
    /// Derives immutable rate and capacity while initializing the existing token state.
    fn new(label: String, limit: RateLimit) -> Self {
        let capacity = limit.burst as f64;
        Self {
            state: Mutex::new(BucketState {
                tokens: capacity,
                last_refill: Instant::now(),
            }),
            label,
            refill_rate: limit.rpm as f64 / 60.0,
            capacity,
        }
    }

    /// Rejects zero rate or capacity without accessing mutable token state.
    fn validate(&self) -> Result<(), ClientError> {
        let invalid = if self.refill_rate <= 0.0 {
            Some("rpm must be non-zero")
        } else if self.capacity < 1.0 {
            Some("burst must be non-zero")
        } else {
            None
        };
        if let Some(reason) = invalid {
            return Err(ClientError::new(
                ErrorKind::Validation,
                format!("rate limit for provider '{}': {reason}", self.label),
            ));
        }
        Ok(())
    }

    /// Waits until one token is available and then consumes it.
    /// The mutex is never held across an `.await` point.
    async fn acquire(&self) {
        loop {
            let wait = {
                let mut state = self.state.lock().await;
                let elapsed = state.last_refill.elapsed().as_secs_f64();
                state.tokens = (state.tokens + elapsed * self.refill_rate).min(self.capacity);
                state.last_refill = Instant::now();

                if state.tokens >= 1.0 {
                    state.tokens -= 1.0;
                    None
                } else {
                    Some(Duration::from_secs_f64(
                        (1.0 - state.tokens) / self.refill_rate,
                    ))
                }
            };
            match wait {
                None => return,
                Some(d) => {
                    tracing::debug!(
                        provider = %self.label,
                        wait_ms = d.as_millis(),
                        "rate limit: sleeping"
                    );
                    tokio::time::sleep(d).await;
                }
            }
        }
    }
}

struct RateLimitingClient {
    inner: Client,
    bucket: Arc<TokenBucket>,
}

impl LlmBackend for RateLimitingClient {
    fn model_url(&self) -> &ModelUrl {
        self.inner.model_url()
    }

    fn options(&self) -> &crate::clients::ClientOptions {
        self.inner.options()
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        self.bucket.acquire().await;
        self.inner.execute(messages).await
    }
}

/// Client-factory wrapper that applies per-provider async rate limits.
/// Providers without a configured limit pass through unchanged.
/// Client creation rejects zero rate or burst before calling the inner factory.
pub struct RateLimitingFactory<F: ProviderFactory> {
    inner: F,
    buckets: HashMap<String, Arc<TokenBucket>>,
}

/// Layer that applies per-provider request limits.
#[derive(Debug, Clone, Default)]
pub struct RateLimitLayer {
    limits: Vec<(Provider, RateLimit)>,
}

impl RateLimitLayer {
    /// Creates a layer with no limits configured.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a per-provider rate limit. Call once per provider that needs limiting.
    /// Zero rate or burst is rejected during client creation for this provider.
    pub fn with_limit(mut self, provider: Provider, limit: RateLimit) -> Self {
        self.limits.push((provider, limit));
        self
    }
}

impl<F: ProviderFactory> RateLimitingFactory<F> {
    /// Wraps `inner` with no limits configured.
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            buckets: HashMap::new(),
        }
    }

    /// Sets the rate limit for one provider.
    /// A second call for the same provider replaces the previous value.
    /// Zero rate or burst is rejected during client creation for this provider.
    pub fn with_limit(mut self, provider: Provider, limit: RateLimit) -> Self {
        let label = provider.as_str().to_owned();
        self.buckets
            .insert(label.clone(), Arc::new(TokenBucket::new(label, limit)));
        self
    }
}

impl<F: ProviderFactory> ProviderFactory for RateLimitingFactory<F> {
    async fn llm(
        &self,
        model_url: &ModelUrl,
        options: ClientOptions,
    ) -> Result<Client, ClientError> {
        let bucket = self.buckets.get(model_url.provider().as_str());
        if let Some(bucket) = bucket {
            bucket.validate()?;
        }
        let inner = self.inner.llm(model_url, options).await?;
        match bucket {
            Some(bucket) => Ok(Client::from_backend(RateLimitingClient {
                inner,
                bucket: Arc::clone(bucket),
            })),
            None => Ok(inner),
        }
    }
}

impl RateLimitLayer {
    /// Wraps a provider factory with this execution policy.
    pub fn layer<F: ProviderFactory>(self, inner: F) -> RateLimitingFactory<F> {
        self.limits.into_iter().fold(
            RateLimitingFactory::new(inner),
            |factory, (provider, limit)| factory.with_limit(provider, limit),
        )
    }
}

#[cfg(test)]
#[path = "tests/limiter.rs"]
mod tests;
