use super::*;
use crate::legacy::client_factory::CountingFactory;
use crate::testing::ScriptedFactory;
use std::sync::{Arc, atomic::Ordering};

/// Legacy retries cover transient failures without retrying caller or output errors.
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

/// Backoff grows and then caps at the configured maximum.
#[test]
fn backoff_delay_growth() -> Result<(), ClientError> {
    let config =
        RetryConfig::new(5, Duration::from_secs(1)).with_max_delay(Duration::from_secs(10));
    for (attempt, seconds) in [1, 2, 4, 8, 10].into_iter().enumerate() {
        assert_eq!(
            backoff_delay(&config, attempt as u32)?,
            Duration::from_secs(seconds)
        );
    }
    Ok(())
}

/// The default retry policy preserves its documented values.
#[test]
fn retry_config_defaults() {
    let config = RetryConfig::default();
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.initial_delay, Duration::from_secs(1));
    assert_eq!(config.backoff_factor, 2.0);
    assert_eq!(config.max_delay, Duration::from_secs(30));
}

/// Builder and literal policies fail before inner construction through factories and layers.
#[tokio::test]
async fn invalid_retry_policies_fail_before_construction() -> Result<(), ClientError> {
    let url = ModelUrl::parse("openai:///fixture")?;
    for factor in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for literal in [false, true] {
            for layered in [false, true] {
                let script = ScriptedFactory::new();
                let base = CountingFactory::new(script.clone());
                let creations = Arc::clone(&base.creations);
                let config = if literal {
                    RetryConfig {
                        backoff_factor: factor,
                        ..RetryConfig::default()
                    }
                } else {
                    RetryConfig::default().with_backoff_factor(factor)
                };
                let factory = if layered {
                    RetryLayer::new(config).layer(base)
                } else {
                    RetryingFactory::new(base).with_config(config)
                };
                let error = factory
                    .llm(&url, ClientOptions::default())
                    .await
                    .err()
                    .expect("invalid configuration must fail");
                assert_eq!(error.kind(), ErrorKind::Validation);
                assert_eq!(creations.load(Ordering::SeqCst), 0);
                assert!(script.calls().is_empty());
            }
        }
    }
    Ok(())
}

/// Zero and fractional multipliers still complete a real scripted retry sequence.
#[tokio::test]
async fn valid_retry_policies_execute() -> Result<(), ClientError> {
    let url = ModelUrl::parse("openai:///fixture")?;
    for factor in [0.0, 0.5] {
        let script = ScriptedFactory::new()
            .then_err(ClientError::new(ErrorKind::Timeout, "first failure"))
            .then_err(ClientError::new(ErrorKind::Timeout, "second failure"))
            .then_output(serde_json::json!(42));
        let factory = RetryingFactory::new(script.clone())
            .with_config(RetryConfig::new(2, Duration::from_micros(1)).with_backoff_factor(factor));
        let client = factory.llm(&url, ClientOptions::default()).await?;
        let response = client.execute(&[]).await?;
        assert!(
            matches!(response.output, crate::clients::ClientOutput::Output(value)
            if value == serde_json::json!(42))
        );
        assert_eq!(script.calls().len(), 3);
    }
    Ok(())
}

/// A zero-retry policy returns the first failure without another request.
#[tokio::test]
async fn zero_retries_preserve_first_failure() -> Result<(), ClientError> {
    let script = ScriptedFactory::new().then_err(ClientError::new(ErrorKind::Timeout, "failure"));
    let factory = RetryLayer::new(RetryConfig::new(0, Duration::ZERO)).layer(script.clone());
    let client = factory
        .llm(
            &ModelUrl::parse("openai:///fixture")?,
            ClientOptions::default(),
        )
        .await?;
    assert_eq!(
        client
            .execute(&[])
            .await
            .expect_err("expected failure")
            .kind(),
        ErrorKind::Timeout
    );
    assert_eq!(script.calls().len(), 1);
    Ok(())
}

/// Zero delays avoid zero-times-infinity arithmetic, and valid multipliers keep their behavior.
#[test]
fn zero_and_fractional_delays() -> Result<(), ClientError> {
    let zero = RetryConfig::new(3, Duration::ZERO).with_backoff_factor(f64::MAX);
    assert_eq!(backoff_delay(&zero, 2)?, Duration::ZERO);
    let capped = RetryConfig::new(3, Duration::from_secs(1)).with_max_delay(Duration::ZERO);
    assert_eq!(backoff_delay(&capped, 2)?, Duration::ZERO);
    let fractional = RetryConfig::new(3, Duration::from_secs(1)).with_backoff_factor(0.5);
    assert_eq!(backoff_delay(&fractional, 2)?, Duration::from_millis(250));
    let zero_factor = fractional.with_backoff_factor(0.0);
    assert_eq!(backoff_delay(&zero_factor, 0)?, Duration::from_secs(1));
    assert_eq!(backoff_delay(&zero_factor, 1)?, Duration::ZERO);
    Ok(())
}

/// Overflow and maximum-duration rounding return the exact cap without long sleeps.
#[test]
fn overflow_preserves_exact_cap() -> Result<(), ClientError> {
    let config = RetryConfig::new(3, Duration::from_secs(1)).with_backoff_factor(f64::MAX);
    assert_eq!(backoff_delay(&config, 2)?, config.max_delay);
    let config = config.with_max_delay(Duration::MAX);
    assert_eq!(backoff_delay(&config, 2)?, Duration::MAX);
    let config = RetryConfig::new(3, Duration::MAX).with_max_delay(Duration::MAX);
    assert_eq!(backoff_delay(&config, 0)?, Duration::MAX);
    Ok(())
}
