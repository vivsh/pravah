use super::*;
use crate::clients::{
    Client, ClientOutput, ClientResponse, ErrorKind, LlmBackend, Message, ModelUrl, ProviderFactory,
};
use crate::clients::{ClientError, ClientOptions};
use rath::llm::{TokenCount, TokenCountSource};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Default contexts construct built-in clients without provider calls.
#[tokio::test]
async fn context_installs_default_provider_registry() -> Result<(), ClientError> {
    let context = Context::default();
    let client = context
        .providers()
        .llm("ollama:///qwen3:8b", ClientOptions::default())
        .await?;
    assert_eq!(client.model_url().model(), "qwen3:8b");
    Ok(())
}

struct CountingFactory(Arc<AtomicUsize>);

struct CountingBackend {
    calls: Arc<AtomicUsize>,
    url: ModelUrl,
    options: ClientOptions,
}

impl ProviderFactory for CountingFactory {
    async fn llm(&self, url: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Client::from_backend(CountingBackend {
            calls: self.0.clone(),
            url: url.clone(),
            options,
        }))
    }
}

impl LlmBackend for CountingBackend {
    fn model_url(&self) -> &ModelUrl {
        &self.url
    }
    fn options(&self) -> &ClientOptions {
        &self.options
    }
    fn estimate_tokens(&self, _: &[Message]) -> Result<TokenCount, ClientError> {
        Ok(TokenCount {
            input_tokens: 11,
            source: TokenCountSource::Estimated,
        })
    }
    async fn count_tokens(&self, _: &[Message]) -> Result<TokenCount, ClientError> {
        Ok(TokenCount {
            input_tokens: 9,
            source: TokenCountSource::ProviderReported,
        })
    }
    async fn execute(&self, _: &[Message]) -> Result<ClientResponse, ClientError> {
        Ok(ClientResponse::new(
            self.url.provider().clone(),
            ClientOutput::Output(serde_json::json!(self.calls.load(Ordering::SeqCst))),
        ))
    }
}

/// URL controls override typed options before the injected factory constructs its client.
#[tokio::test]
async fn builtin_factory_preserves_url_option_precedence() -> Result<(), ClientError> {
    use crate::clients::{CacheControl, ThinkingLevel};
    let calls = Arc::new(AtomicUsize::new(0));
    let context = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
        CountingFactory(calls.clone()),
    ));
    let options = ClientOptions::default()
        .with_temperature(0.1)
        .with_thinking(Some(ThinkingLevel::Low))
        .with_cache(CacheControl::Ephemeral5m);
    let client = context
        .providers()
        .llm(
            "anthropic:///recorded?temperature=0.8&thinking=high&cache=1h",
            options,
        )
        .await?;
    assert_eq!(client.options().temperature, Some(0.8));
    assert_eq!(client.options().thinking, Some(ThinkingLevel::High));
    assert_eq!(client.options().cache, Some(CacheControl::Ephemeral1h));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Returned clients retain their own dependencies and counting behavior after Context is dropped.
#[tokio::test]
async fn injected_client_outlives_context_and_retains_counting() -> Result<(), ClientError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let weak = Arc::downgrade(&calls);
    let context = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
        CountingFactory(calls),
    ));
    let client = context
        .providers()
        .llm("openai:///recorded", ClientOptions::default())
        .await?;
    drop(context);
    assert!(weak.upgrade().is_some());
    let messages = [Message::user("recorded")];
    assert_eq!(
        client.estimate_tokens(&messages)?,
        TokenCount {
            input_tokens: 11,
            source: TokenCountSource::Estimated
        }
    );
    assert_eq!(
        client.count_tokens(&messages).await?,
        TokenCount {
            input_tokens: 9,
            source: TokenCountSource::ProviderReported
        }
    );
    assert!(
        matches!(client.execute(&messages).await?.output, ClientOutput::Output(value) if value == serde_json::json!(1))
    );
    let error = client
        .count_content_tokens("unsupported")
        .await
        .err()
        .ok_or_else(|| ClientError::new(ErrorKind::Validation, "expected unsupported counting"))?;
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    drop(client);
    assert!(weak.upgrade().is_none());
    Ok(())
}

struct UnsupportedFactory;
impl ProviderFactory for UnsupportedFactory {}

/// A factory without LLM support fails explicitly rather than selecting a built-in client.
#[tokio::test]
async fn unsupported_injected_capability_has_no_fallback() -> Result<(), ClientError> {
    let context = Context::default()
        .with_providers(ProviderRegistry::with_builtin_factory(UnsupportedFactory));
    let error = context
        .providers()
        .llm("openai:///recorded", ClientOptions::default())
        .await
        .err()
        .ok_or_else(|| ClientError::new(ErrorKind::Validation, "expected unsupported LLM"))?;
    assert_eq!(error.kind(), ErrorKind::UnsupportedCapability);
    Ok(())
}

/// Replacing injection with the standard registry restores ordinary built-in construction.
#[tokio::test]
async fn builtin_registry_replaces_injected_factory() -> Result<(), ClientError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let context = Context::default()
        .with_providers(ProviderRegistry::with_builtin_factory(CountingFactory(
            calls.clone(),
        )))
        .with_providers(ProviderRegistry::with_builtins());
    let client = context
        .providers()
        .llm("ollama:///recorded", ClientOptions::default())
        .await?;
    assert_eq!(client.model_url().model(), "recorded");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}
