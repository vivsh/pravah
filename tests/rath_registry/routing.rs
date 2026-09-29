use super::*;
use pravah::clients::{
    CacheControl, ErrorBody, ErrorKind, Provider, ResponseFormat, ThinkingLevel,
};
use pravah::{AgentClientOperation, ChatBuilder};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const MODEL: &str = "openai:///recorded-model";

fn definition() -> ChatBuilder<String, String> {
    Chat::builder::<String, String>().model(MODEL)
}

struct CountingFactory {
    calls: Arc<AtomicUsize>,
    script: ScriptedFactory,
}

impl ProviderFactory for CountingFactory {
    async fn llm(&self, url: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.script.llm(url, options).await
    }
}

/// Context clones share one factory, but Chat executions retain independent histories.
#[tokio::test]
async fn clones_share_factory_until_last_execution_drops() -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let weak = Arc::downgrade(&calls);
    let script = ScriptedFactory::new()
        .then_output(json!("one"))
        .then_output(json!("two"))
        .then_output(json!("three"));
    let ctx = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
        CountingFactory { calls, script },
    ));
    let mut first = definition().build(ctx.clone())?;
    let mut second = definition().build(ctx.clone())?;
    drop(ctx);
    assert_eq!(first.send("one").await?.output, "one");
    assert_eq!(second.send("two").await?.output, "two");
    assert_eq!(second.send("three").await?.output, "three");
    assert_eq!(first.snapshot()?.history().entries().len(), 2);
    assert_eq!(second.snapshot()?.history().entries().len(), 4);
    assert_eq!(
        weak.upgrade()
            .ok_or(TestError::Missing("factory dependency"))?
            .load(Ordering::SeqCst),
        3
    );
    drop(first);
    assert!(weak.upgrade().is_some());
    drop(second);
    assert!(weak.upgrade().is_none());
    Ok(())
}

/// Installing a registry replaces routing; existing Context clones retain their prior choice.
#[tokio::test]
async fn registry_replacement_has_no_override_chain() -> Result<(), TestError> {
    let original = ScriptedFactory::new().then_output(json!("original"));
    let replacement = ScriptedFactory::new().then_output(json!("replacement"));
    let ctx =
        Context::default().with_providers(ProviderRegistry::with_builtin_factory(original.clone()));
    let retained = ctx.clone();
    let ctx = ctx.with_providers(ProviderRegistry::with_builtin_factory(replacement.clone()));
    let mut chat = definition().build(ctx)?;
    assert_eq!(chat.send("question").await?.output, "replacement");
    assert!(original.calls().is_empty());
    let mut chat = definition().build(retained)?;
    assert_eq!(chat.send("question").await?.output, "original");
    assert_eq!(replacement.calls().len(), 1);
    Ok(())
}

/// An empty registry disables earlier injection, and later injection replaces an empty registry.
#[tokio::test]
async fn last_registry_wins_in_both_orders() -> Result<(), TestError> {
    for inject_last in [false, true] {
        let script = ScriptedFactory::new().then_output(json!("recorded"));
        let injected = ProviderRegistry::with_builtin_factory(script.clone());
        let empty = ProviderRegistry::new();
        let (first, last) = if inject_last {
            (empty, injected)
        } else {
            (injected, empty)
        };
        let ctx = Context::default()
            .with_providers(first)
            .with_providers(last);
        let mut chat = definition().build(ctx)?;
        match chat.send("question").await {
            Ok(reply) => {
                assert!(inject_last);
                assert_eq!(reply.output, "recorded");
            }
            Err(error) => {
                assert!(!inject_last);
                assert_eq!(
                    error.client_error().map(ClientError::kind),
                    Some(ErrorKind::UnsupportedCapability)
                );
            }
        }
        assert_eq!(script.calls().len(), usize::from(inject_last));
    }
    Ok(())
}

struct CheckOptions(ScriptedFactory);

impl ProviderFactory for CheckOptions {
    /// Checks the effective generation settings at the real Chat construction boundary.
    async fn llm(&self, url: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        assert_eq!(url.provider(), &Provider::Anthropic);
        assert_eq!(url.model(), "recorded-model");
        assert_eq!(url.temperature(), Some(0.7));
        assert_eq!(options.temperature, Some(0.7));
        assert_eq!(options.thinking, Some(ThinkingLevel::High));
        assert_eq!(options.cache, Some(CacheControl::Ephemeral1h));
        assert_eq!(options.max_output_tokens, Some(128));
        assert_eq!(options.provider_config, Some(json!({"top_p": 0.8})));
        assert!(
            options
                .preamble
                .as_deref()
                .is_some_and(|p| p.contains("Preserve these instructions."))
        );
        assert!(
            matches!(&options.response_format, ResponseFormat::JsonSchema {schema} if schema.get("type") == Some(&json!("string")))
        );
        self.0.llm(url, options).await
    }
}

/// Built-in factory injection receives the original identity and resolved Chat generation settings.
#[tokio::test]
async fn injected_factory_receives_effective_chat_options() -> Result<(), TestError> {
    let script = ScriptedFactory::new().then_output(json!("answer"));
    let ctx = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
        CheckOptions(script.clone()),
    ));
    let mut chat = definition()
        .model("anthropic:///recorded-model?temperature=0.7&thinking=high&cache=1h")
        .instructions("Preserve these instructions.")
        .provider_config(json!({"top_p": 0.8}))
        .max_output_tokens(128)
        .build(ctx)?;
    assert_eq!(chat.send("question").await?.output, "answer");
    assert_eq!(script.calls().len(), 1);
    Ok(())
}

struct BrokenFactory(Arc<AtomicUsize>);

impl ProviderFactory for BrokenFactory {
    /// Echoes a locator-resolved synthetic credential so registry redaction is exercised.
    async fn llm(&self, url: &ModelUrl, _: ClientOptions) -> Result<Client, ClientError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let secret = url.api_key().unwrap_or("no credential");
        Err(ClientError::new(
            ErrorKind::UnsupportedCapability,
            format!("recorded-only failure: {secret}"),
        )
        .with_body(ErrorBody::Complete(secret.as_bytes().to_vec()), &[]))
    }
}

/// Creation failure is propagated exactly once, redacted, and cannot fall back to a real provider.
#[tokio::test]
async fn creation_failure_has_no_fallback_and_redacts_credentials() -> Result<(), TestError> {
    let secret = std::env::var("PATH").map_err(|_| TestError::Missing("PATH fixture"))?;
    assert!(!secret.is_empty());
    let calls = Arc::new(AtomicUsize::new(0));
    let ctx = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
        BrokenFactory(calls.clone()),
    ));
    let mut chat = definition()
        .model("openai:///recorded-model?api_key_env=PATH")
        .build(ctx)?;
    let error = chat
        .send_with_key("question", "key")
        .await
        .err()
        .ok_or(TestError::Missing("creation failure"))?;
    assert!(matches!(
        &error,
        GraphError::AgentClient {
            operation: AgentClientOperation::Create,
            ..
        }
    ));
    let source = error
        .client_error()
        .ok_or(TestError::Missing("typed cause"))?;
    assert_eq!(source.kind(), ErrorKind::UnsupportedCapability);
    assert_eq!(source.provider(), Some(&Provider::OpenAi));
    assert!(source.message().contains("recorded-only failure"));
    assert!(!format!("{error} {error:?} {source}").contains(&secret));
    let body = source
        .response_body()
        .ok_or(TestError::Missing("redacted body"))?;
    assert!(!String::from_utf8_lossy(body.bytes()).contains(&secret));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let snapshot = chat.snapshot()?;
    let before = serde_json::to_value(&snapshot)?;
    let restored = definition()
        .model("openai:///recorded-model?api_key_env=PATH")
        .restore::<()>(snapshot, Context::default())?;
    assert_eq!(before, serde_json::to_value(restored.snapshot()?)?);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

/// A cancelled, configured invocation restores against its production definition without dispatch.
#[tokio::test]
async fn cancelled_builtin_dispatch_restores_unfinished_checkpoint() -> Result<(), TestError> {
    let ctx =
        Context::default().with_providers(ProviderRegistry::with_builtin_factory(PendingFactory));
    let mut chat = definition().build(ctx)?;
    {
        let send = chat.send_with_key("question", "pending-key");
        futures::pin_mut!(send);
        assert!(futures::poll!(send).is_pending());
    }
    let snapshot = chat.snapshot()?;
    assert_eq!(snapshot.history().entries().len(), 1);
    assert_eq!(
        snapshot
            .history()
            .entries()
            .first()
            .and_then(|e| e.message.key.as_deref()),
        Some("pending-key")
    );
    let before = serde_json::to_value(&snapshot)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(&snapshot, &mut cbor)?;
    for snapshot in [
        serde_json::from_value(before.clone())?,
        ciborium::from_reader(cbor.as_slice())?,
    ] {
        let restored = definition().restore::<()>(snapshot, Context::default())?;
        assert_eq!(before, serde_json::to_value(restored.snapshot()?)?);
    }
    assert!(matches!(
        chat.send("another").await,
        Err(GraphError::ChatNotReady { .. })
    ));
    Ok(())
}

/// Execution failure remains distinct from construction and consumes no implicit retry response.
#[tokio::test]
async fn execution_failure_does_not_retry() -> Result<(), TestError> {
    let script = ScriptedFactory::new()
        .then_err(ClientError::new(ErrorKind::Timeout, "recorded timeout"))
        .then_output(json!("must not run"));
    let calls = Arc::new(AtomicUsize::new(0));
    let ctx = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
        CountingFactory {
            calls: calls.clone(),
            script: script.clone(),
        },
    ));
    let mut chat = definition().build(ctx)?;
    let error = chat
        .send("question")
        .await
        .err()
        .ok_or(TestError::Missing("execution error"))?;
    assert!(matches!(
        &error,
        GraphError::AgentClient {
            operation: AgentClientOperation::Execute,
            ..
        }
    ));
    assert_eq!(
        error.client_error().map(ClientError::kind),
        Some(ErrorKind::Timeout)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(script.calls().len(), 1);
    assert_eq!(script.remaining(), 1);
    Ok(())
}

/// Rath validates malformed URLs and conflicting provider settings before the injected factory runs.
#[tokio::test]
async fn invalid_settings_never_reach_injected_factory() -> Result<(), TestError> {
    for (model, config) in [
        ("openai:///recorded?temperature=2", json!({})),
        ("openai:///recorded?unknown=1", json!({})),
        (
            "openai:///recorded?base_url=https://user:password@example.invalid",
            json!({}),
        ),
        (MODEL, json!({"messages": []})),
        (MODEL, json!({"max_tokens": 12})),
        (MODEL, json!([])),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let ctx = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
            BrokenFactory(calls.clone()),
        ));
        let mut chat = definition()
            .model(model)
            .provider_config(config)
            .build(ctx)?;
        let error = chat
            .send("question")
            .await
            .err()
            .ok_or(TestError::Missing("validation error"))?;
        assert!(matches!(
            &error,
            GraphError::AgentClient {
                operation: AgentClientOperation::Create,
                ..
            }
        ));
        assert!(matches!(
            error.client_error().map(ClientError::kind),
            Some(ErrorKind::Validation | ErrorKind::InvalidUrl)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

/// Built-in injection leaves explicitly registered external providers independently routed.
#[tokio::test]
async fn external_registration_is_not_intercepted() -> Result<(), TestError> {
    let builtin = ScriptedFactory::new().then_output(json!("builtin"));
    let external = ScriptedFactory::new().then_output(json!("external"));
    let providers = ProviderRegistry::with_builtin_factory(builtin.clone())
        .register("recorded", external.clone())?;
    let mut chat = definition()
        .model("recorded:///model")
        .build(Context::default().with_providers(providers))?;
    assert_eq!(chat.send("question").await?.output, "external");
    assert!(builtin.calls().is_empty());
    assert_eq!(external.calls().len(), 1);
    Ok(())
}
