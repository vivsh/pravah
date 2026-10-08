use super::*;
use crate::legacy::client_factory::CountingFactory;
use crate::testing::ScriptedFactory;
use std::sync::atomic::Ordering;

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

/// Invalid constructors and literals are rejected before construction through factories and layers.
#[tokio::test]
async fn invalid_limits_fail_before_construction() -> Result<(), ClientError> {
    let url = ModelUrl::parse("openai:///fixture")?;
    for (rpm, burst, field) in [(0, 1, "rpm"), (60, 0, "burst"), (0, 0, "rpm")] {
        for literal in [false, true] {
            for layered in [false, true] {
                let script = ScriptedFactory::new();
                let base = CountingFactory::new(script.clone());
                let creations = Arc::clone(&base.creations);
                let limit = if literal {
                    RateLimit { rpm, burst }
                } else {
                    RateLimit::new(rpm, burst)
                };
                let factory = if layered {
                    RateLimitLayer::new()
                        .with_limit(Provider::OpenAi, limit)
                        .layer(base)
                } else {
                    RateLimitingFactory::new(base).with_limit(Provider::OpenAi, limit)
                };
                let error = factory
                    .llm(&url, ClientOptions::default())
                    .await
                    .err()
                    .expect("invalid configuration must fail");
                assert_eq!(error.kind(), ErrorKind::Validation);
                assert!(error.to_string().contains(field));
                assert!(error.to_string().contains("openai"));
                assert_eq!(creations.load(Ordering::SeqCst), 0);
                assert!(script.calls().is_empty());
            }
        }
    }
    Ok(())
}

/// A malformed limit for another provider does not prevent unrelated client requests.
#[tokio::test]
async fn unconfigured_provider_passes_through() -> Result<(), ClientError> {
    let script = ScriptedFactory::new().then_output(serde_json::json!(42));
    let base = CountingFactory::new(script.clone());
    let creations = Arc::clone(&base.creations);
    let factory = RateLimitingFactory::new(base).with_limit(Provider::OpenAi, RateLimit::new(0, 0));
    let client = factory
        .llm(
            &ModelUrl::parse("ollama:///fixture")?,
            ClientOptions::default(),
        )
        .await?;
    client.execute(&[]).await?;
    assert_eq!(creations.load(Ordering::SeqCst), 1);
    assert_eq!(script.calls().len(), 1);
    Ok(())
}

/// Valid factory and layer policies allow requests and preserve the configured initial burst.
#[tokio::test]
async fn valid_limits_execute() -> Result<(), ClientError> {
    let url = ModelUrl::parse("openai:///fixture")?;
    for layered in [false, true] {
        let script = ScriptedFactory::new().then_output(serde_json::json!(42));
        let base = CountingFactory::new(script.clone());
        let creations = Arc::clone(&base.creations);
        let limit = RateLimit::new(1, 1);
        let factory = if layered {
            RateLimitLayer::new()
                .with_limit(Provider::OpenAi, limit)
                .layer(base)
        } else {
            RateLimitingFactory::new(base).with_limit(Provider::OpenAi, limit)
        };
        let client = factory.llm(&url, ClientOptions::default()).await?;
        let response = client.execute(&[]).await?;
        assert!(
            matches!(response.output, crate::clients::ClientOutput::Output(value)
            if value == serde_json::json!(42))
        );
        assert_eq!(creations.load(Ordering::SeqCst), 1);
        assert_eq!(script.calls().len(), 1);
    }
    Ok(())
}
