use super::*;
use pravah::testing::{ScriptedFactory, mock_tool_call};
use pravah::{CompactionRequest, CompactionResult, Compactor};

struct CountingFactory {
    builds: Arc<AtomicUsize>,
    scripted: ScriptedFactory,
}

impl ProviderFactory for CountingFactory {
    async fn llm(
        &self,
        model: &ModelUrl,
        mut options: ClientOptions,
    ) -> Result<Client, ClientError> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        options.temperature = Some(0.25);
        self.scripted.llm(model, options).await
    }
}

struct CheckOptions;
impl Compactor for CheckOptions {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        assert_eq!(request.options().temperature, Some(0.25));
        Ok(CompactionResult::default())
    }
}

fn counted_context(builds: &Arc<AtomicUsize>, scripted: &ScriptedFactory) -> Context {
    Context::default().with_providers(ProviderRegistry::with_builtin_factory(CountingFactory {
        builds: builds.clone(),
        scripted: scripted.clone(),
    }))
}

/// Preparation builds no unused client, but a compactor still sees effective provider options.
#[tokio::test]
async fn preparation_constructs_clients_only_when_needed() -> Result<(), GraphError> {
    for policy in [false, true] {
        let builds = Arc::new(AtomicUsize::new(0));
        let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
        let mut definition = builder();
        if policy {
            definition = definition.compactor(CheckOptions);
        }
        let mut chat = definition.build(counted_context(&builds, &factory))?;
        assert_eq!(chat.send("question").await?.output, "answer");
        assert_eq!(builds.load(Ordering::SeqCst), if policy { 2 } else { 1 });
        assert_eq!(factory.calls().len(), 1);
    }
    Ok(())
}

/// Budget-driven conclusion still constructs a preparation client for provider-aware guidance.
#[tokio::test]
async fn budget_guidance_retains_provider_preparation() -> Result<(), GraphError> {
    let builds = Arc::new(AtomicUsize::new(0));
    let factory = ScriptedFactory::new()
        .then_tool_calls(vec![mock_tool_call(
            "call",
            "unknown",
            serde_json::json!({}),
        )])
        .then_output(serde_json::json!("answer"));
    let mut chat = builder()
        .turn_budget(1)
        .build(counted_context(&builds, &factory))?;
    assert_eq!(chat.send("question").await?.output, "answer");
    assert_eq!(builds.load(Ordering::SeqCst), 3);
    assert_eq!(factory.calls().len(), 2);
    Ok(())
}

struct FailingFactory(Arc<AtomicUsize>);

impl ProviderFactory for FailingFactory {
    async fn llm(&self, _: &ModelUrl, _: ClientOptions) -> Result<Client, ClientError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(ClientError::new(
            pravah::clients::ErrorKind::Validation,
            "intentional construction failure",
        ))
    }
}

/// With no preparation policy, client failure belongs to generation and leaves its request pending.
#[tokio::test]
async fn unused_client_removal_preserves_creation_errors() -> Result<(), GraphError> {
    let builds = Arc::new(AtomicUsize::new(0));
    let context = Context::default().with_providers(ProviderRegistry::with_builtin_factory(
        FailingFactory(builds.clone()),
    ));
    let mut chat = builder().build(context)?;
    chat.submit_with_key("question", "key")?;
    loop {
        let fetch = match chat.next()? {
            ChatStep::Continue => continue,
            ChatStep::Fetch(fetch) => fetch,
            _ => return Err(GraphError::Invalid("expected external boundary".into())),
        };
        if fetch.request().url() != "rath://generate" {
            let response = chat.executor().execute(&fetch).await?;
            chat.resume_fetch(fetch.id(), Ok(response))?;
            assert_eq!(builds.load(Ordering::SeqCst), 0);
            continue;
        }
        let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
        let error = chat.executor().execute(&fetch).await;
        assert!(matches!(
            error,
            Err(GraphError::AgentClient {
                operation: pravah::AgentClientOperation::Create,
                ..
            })
        ));
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(
            before,
            serde_json::to_value(chat.snapshot()?).map_err(codec)?
        );
        assert_eq!(chat.pending_fetch().map(Fetch::id), Some(fetch.id()));
        return Ok(());
    }
}
