//! Request-based triggers run once per dispatch without constructing extra clients.

use pravah::clients::{Client, ClientError, ClientOptions, ModelUrl, ProviderFactory};
use pravah::testing::ScriptedFactory;
use pravah::{Chat, CompactionRequest, CompactionResult, Compactor, Context, GraphError};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct CountFactory {
    script: ScriptedFactory,
    creates: Arc<AtomicUsize>,
}

impl ProviderFactory for CountFactory {
    async fn llm(&self, model: &ModelUrl, options: ClientOptions) -> Result<Client, ClientError> {
        self.creates.fetch_add(1, Ordering::SeqCst);
        self.script.llm(model, options).await
    }
}

struct SkipCompaction(Arc<AtomicUsize>);

impl Compactor for SkipCompaction {
    type Error = std::convert::Infallible;

    fn needs_compaction(&self, request: &CompactionRequest<'_>) -> bool {
        let turn = self.0.fetch_add(1, Ordering::SeqCst);
        assert_eq!(request.model(), "test:///test");
        assert_eq!(request.turn_count(), turn);
        assert_eq!(request.protected().len(), 1);
        assert_eq!(request.total_input(), None);
        assert!(
            request
                .options()
                .preamble
                .as_deref()
                .is_some_and(|p| p.contains("Answer"))
        );
        false
    }

    async fn compact(
        &self,
        _: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        self.0.fetch_add(100, Ordering::SeqCst);
        Ok(CompactionResult::default())
    }
}

/// A false trigger preserves requests and avoids extra client construction and post-response calls.
#[tokio::test]
async fn request_trigger_skips_client_preparation_and_runs_once_per_dispatch()
-> Result<(), GraphError> {
    let script = ScriptedFactory::new()
        .then_output(serde_json::json!("one"))
        .then_output(serde_json::json!("two"));
    let creates = Arc::new(AtomicUsize::new(0));
    let triggers = Arc::new(AtomicUsize::new(0));
    let ctx = Context::default().with_providers(pravah::testing::providers(CountFactory {
        script: script.clone(),
        creates: creates.clone(),
    })?);
    let mut chat = Chat::builder::<String, String>()
        .model("test:///test")
        .instructions("Answer briefly.")
        .compactor(SkipCompaction(triggers.clone()))
        .build(ctx)?;
    assert_eq!(creates.load(Ordering::SeqCst), 0);
    assert_eq!(chat.send("first").await?.output, "one");
    assert_eq!(chat.send("second").await?.output, "two");
    assert_eq!(triggers.load(Ordering::SeqCst), 2);
    assert_eq!(creates.load(Ordering::SeqCst), 2);
    assert_eq!(script.calls().len(), 2);
    assert_eq!(chat.snapshot()?.history().entries().len(), 4);
    Ok(())
}
