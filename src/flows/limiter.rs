use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

use crate::clients::{
    Client, ClientError, ClientFactory, ClientFactoryLayer, ClientOptions, ClientResponse,
    EmbedRequest, EmbedResponse, LlmUrl, Message, Provider,
};

/// Per-provider rate-limit settings.
/// `rpm` sets the sustained rate and `burst` sets the immediate bucket depth.
#[derive(Debug, Clone, Copy)]
pub struct RateLimit {
    /// Sustained requests per minute.
    pub rpm: u32,
    /// Maximum number of immediate requests when the bucket is full.
    pub burst: u32,
}

impl RateLimit {
    /// Builds a rate limit.
    /// Debug builds assert that both values are non-zero.
    pub fn new(rpm: u32, burst: u32) -> Self {
        debug_assert!(rpm > 0, "rpm must be > 0");
        debug_assert!(burst > 0, "burst must be >= 1");
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
    inner: Box<dyn Client>,
    bucket: Arc<TokenBucket>,
}

#[async_trait]
impl Client for RateLimitingClient {
    fn model_url(&self) -> &LlmUrl {
        self.inner.model_url()
    }

    fn options(&self) -> &crate::clients::ClientOptions {
        self.inner.options()
    }

    async fn execute(&self, messages: &[Message]) -> Result<ClientResponse, ClientError> {
        self.bucket.acquire().await;
        self.inner.execute(messages).await
    }

    async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, ClientError> {
        self.inner.embed(request).await
    }
}

/// Client-factory wrapper that applies per-provider async rate limits.
/// Providers without a configured limit pass through unchanged.
pub struct RateLimitingFactory<F: ClientFactory> {
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
    pub fn with_limit(mut self, provider: Provider, limit: RateLimit) -> Self {
        self.limits.push((provider, limit));
        self
    }
}

impl<F: ClientFactory> RateLimitingFactory<F> {
    /// Wraps `inner` with no limits configured.
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            buckets: HashMap::new(),
        }
    }

    /// Sets the rate limit for one provider.
    /// A second call for the same provider replaces the previous value.
    pub fn with_limit(mut self, provider: Provider, limit: RateLimit) -> Self {
        let label = provider.as_str().to_owned();
        self.buckets
            .insert(label.clone(), Arc::new(TokenBucket::new(label, limit)));
        self
    }
}

impl<F: ClientFactory> ClientFactory for RateLimitingFactory<F> {
    fn create(
        &self,
        model_url: &str,
        options: ClientOptions,
    ) -> Result<Box<dyn Client>, ClientError> {
        let inner = self.inner.create(model_url, options)?;
        let url = LlmUrl::parse(model_url)?;
        match self.buckets.get(url.provider.as_str()) {
            Some(bucket) => Ok(Box::new(RateLimitingClient {
                inner,
                bucket: Arc::clone(bucket),
            })),
            None => Ok(inner),
        }
    }
}

impl<F: ClientFactory> ClientFactoryLayer<F> for RateLimitLayer {
    type Factory = RateLimitingFactory<F>;

    fn layer(self, inner: F) -> Self::Factory {
        self.limits.into_iter().fold(
            RateLimitingFactory::new(inner),
            |factory, (provider, limit)| factory.with_limit(provider, limit),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full bucket allows an immediate burst.
    #[tokio::test]
    async fn test_burst_fires_immediately() {
        let bucket = TokenBucket::new("test".to_owned(), RateLimit::new(60, 3));
        let start = std::time::Instant::now();
        bucket.acquire().await;
        bucket.acquire().await;
        bucket.acquire().await;
        assert!(start.elapsed().as_millis() < 100, "burst should not sleep");
    }

    /// After the burst is spent, the next acquire must wait for refill.
    #[tokio::test]
    async fn test_throttle_after_burst() {
        let bucket = Arc::new(TokenBucket::new("test".to_owned(), RateLimit::new(60, 1)));
        bucket.acquire().await;
        let start = std::time::Instant::now();
        bucket.acquire().await;
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() >= 900,
            "expected ~1 s wait, got {elapsed:?}"
        );
    }

    /// `RateLimit::new` stores the provided values.
    #[test]
    fn test_rate_limit_new() {
        let limit = RateLimit::new(120, 10);
        assert_eq!(limit.rpm, 120);
        assert_eq!(limit.burst, 10);
    }
}
