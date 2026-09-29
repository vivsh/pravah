use super::super::*;

/// Legacy retries keep broad provider/transport coverage without retrying caller or output errors.
#[test]
fn retry_classification() {
    for kind in [
        ErrorKind::Provider,
        ErrorKind::Http,
        ErrorKind::Transport,
        ErrorKind::Timeout,
        ErrorKind::InvalidResponse,
        ErrorKind::Other,
    ] {
        assert!(is_retryable(&ClientError::new(kind, "transient failure")));
    }
    for kind in [
        ErrorKind::Validation,
        ErrorKind::InvalidUrl,
        ErrorKind::UnsupportedCapability,
        ErrorKind::Serialize,
        ErrorKind::Deserialize,
        ErrorKind::OutputLimitReached,
        ErrorKind::TokenCounting,
    ] {
        assert!(!is_retryable(&ClientError::new(kind, "terminal failure")));
    }
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
