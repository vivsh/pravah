use super::*;
use pravah::Snapshot;
use pravah::deps::{Deps, DepsError};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    CborWrite(#[from] ciborium::ser::Error<std::io::Error>),
    #[error(transparent)]
    CborRead(#[from] ciborium::de::Error<std::io::Error>),
}

struct MemoryService {
    calls: AtomicUsize,
    text: &'static str,
}

impl MemoryService {
    fn new(text: &'static str) -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            text,
        })
    }
}

struct ReadContext;

impl Compactor for ReadContext {
    type Error = DepsError;

    /// Reads the current execution's service and uses it to replace completed history.
    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        let service = ctx.deps().require::<MemoryService>()?;
        service.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CompactionResult {
            evict_indices: (0..request.committed().len()).collect(),
            summary: (!request.committed().is_empty()).then(|| service.text.to_owned()),
        })
    }
}

fn context(
    service: Arc<MemoryService>,
    factory: ScriptedFactory,
) -> Result<Context, pravah::GraphError> {
    let mut deps = Deps::default();
    deps.insert(service);
    Ok(Context::default()
        .with_deps(deps)
        .with_providers(pravah::testing::providers(factory)?))
}

fn factory() -> ScriptedFactory {
    ScriptedFactory::new()
        .then_output(serde_json::json!({"text":"one"}))
        .then_output(serde_json::json!({"text":"two"}))
}

fn copies(snapshot: &Snapshot) -> Result<[Snapshot; 2], TestError> {
    let json = serde_json::to_vec(snapshot)?;
    let mut cbor = Vec::new();
    ciborium::into_writer(snapshot, &mut cbor)?;
    Ok([
        serde_json::from_slice(&json)?,
        ciborium::from_reader(cbor.as_slice())?,
    ])
}

/// Policies receive their execution's dependencies on every dispatch without cross-chat leakage.
#[tokio::test]
async fn each_chat_uses_its_bound_context() -> Result<(), GraphError> {
    let first = MemoryService::new("first account facts");
    let second = MemoryService::new("second account facts");
    let mut first_chat =
        Chat::new(tutor, context(first.clone(), factory())?)?.with_compactor(ReadContext);
    let mut second_chat =
        Chat::new(tutor, context(second.clone(), factory())?)?.with_compactor(ReadContext);
    for chat in [&mut first_chat, &mut second_chat] {
        chat.send(Question {
            text: "hello".into(),
        })
        .await?;
    }
    first_chat
        .send(Question {
            text: "again".into(),
        })
        .await?;
    assert_eq!(first.calls.load(Ordering::SeqCst), 2);
    assert_eq!(second.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Both snapshot codecs restore policies against fresh context rather than original dependencies.
#[tokio::test]
async fn restore_uses_fresh_context_for_preparation() -> Result<(), TestError> {
    let original = MemoryService::new("original account facts");
    let mut chat =
        Chat::new(tutor, context(original.clone(), factory())?)?.with_compactor(ReadContext);
    chat.send(Question {
        text: "first".into(),
    })
    .await?;
    let snapshot = chat.snapshot()?;
    for copy in copies(&snapshot)? {
        let fresh = MemoryService::new("restored account facts");
        let client = factory();
        let mut restored =
            Chat::<_, _>::from_snapshot(tutor, copy, context(fresh.clone(), client.clone())?)?
                .with_compactor(ReadContext);
        assert_eq!(fresh.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            serde_json::to_value(restored.snapshot()?)?,
            serde_json::to_value(&snapshot)?
        );
        restored
            .send(Question {
                text: "next".into(),
            })
            .await?;
        assert_eq!(fresh.calls.load(Ordering::SeqCst), 1);
        assert!(client.calls().iter().any(|(_, messages)| {
            messages.iter().any(|message| {
                matches!(message.role, Role::System) && message.content.contains(fresh.text)
            })
        }));
    }
    assert_eq!(original.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

/// Missing context dependencies fail preparation without executing a model or mutating the snapshot.
#[tokio::test]
async fn missing_dependency_preserves_history_and_checkpoint() -> Result<(), TestError> {
    let client = factory();
    let (mut runtime, executor) = failures::before_dispatch(ReadContext, &client).await?;
    let before = serde_json::to_value(runtime.snapshot()?)?;
    let result = host::step(&mut runtime, &executor).await;
    assert!(
        matches!(result, Err(GraphError::HistoryCompaction { source, .. })
        if source.is::<DepsError>())
    );
    assert!(client.calls().is_empty());
    assert_eq!(serde_json::to_value(runtime.snapshot()?)?, before);
    Ok(())
}

/// An unfinished dispatch restores with new policy dependencies and preserves its committed checkpoint.
#[tokio::test]
async fn unfinished_dispatch_uses_restored_context() -> Result<(), TestError> {
    let old_client = factory();
    let (runtime, _) = failures::before_dispatch(ReadContext, &old_client).await?;
    let snapshot = runtime.snapshot()?;
    let flow = pravah::compile(|root: pravah::Flow<Question>| root.agent(tutor))?;
    for copy in copies(&snapshot)? {
        let service = MemoryService::new("restored service");
        let client = factory();
        let mut restored = flow.restore(copy)?;
        let executor = flow
            .prepared()
            .executor(context(service.clone(), client.clone())?)
            .with_compactor(ReadContext);
        assert_eq!(
            serde_json::to_value(restored.snapshot()?)?,
            serde_json::to_value(&snapshot)?
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 0);
        host::finish(&mut restored, &executor).await?;
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
        assert_eq!(client.calls().len(), 1);
    }
    assert!(old_client.calls().is_empty());
    Ok(())
}
