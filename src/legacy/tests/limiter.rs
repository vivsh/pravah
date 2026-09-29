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
