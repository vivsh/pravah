use super::*;
use pravah::{Flow, Runtime, Step, compile};

#[derive(Debug, thiserror::Error)]
#[error("memory service unavailable")]
struct PreparationFailure;

struct Failing;

impl Compactor for Failing {
    type Error = PreparationFailure;

    async fn compact(
        &self,
        _request: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        Err(PreparationFailure)
    }
}

struct Invalid {
    indices: Vec<usize>,
}

impl Compactor for Invalid {
    type Error = std::convert::Infallible;

    async fn compact(
        &self,
        _request: CompactionRequest<'_>,
        _ctx: Context,
    ) -> Result<CompactionResult, Self::Error> {
        Ok(CompactionResult {
            evict_indices: self.indices.clone(),
            summary: None,
        })
    }
}

fn workflow(root: Flow<Question>) -> Flow<Answer> {
    root.agent(tutor)
}

/// Stops after the user message is recorded, before the first model execution.
pub(super) async fn before_dispatch(
    policy: impl Compactor + 'static,
    factory: &ScriptedFactory,
) -> Result<(Runtime, FetchExecutor, pravah::HistoryManager), GraphError> {
    let flow = compile(workflow)?;
    let executor = flow
        .prepared()
        .executor(Context::default().with_providers(pravah::testing::providers(factory.clone())?));
    let manager = pravah::HistoryManager::new().with_compactor(policy);
    let mut execution = flow.start(
        Question {
            text: "protected".into(),
        },
        uuid::Uuid::nil(),
    )?;
    for _ in 0..10 {
        if !execution.snapshot()?.history().is_empty() {
            return Ok((execution, executor, manager));
        }
        host::step(&mut execution, &executor).await?;
    }
    Err(GraphError::Invalid("did not reach dispatch".into()))
}

/// A typed preparation error survives in GraphError and leaves the whole checkpoint retryable.
#[tokio::test]
async fn policy_failure_makes_zero_calls_and_preserves_snapshot() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!({"text":"ok"}));
    let (mut execution, executor, mut manager) = before_dispatch(Failing, &factory).await?;
    let before = execution.snapshot()?;
    match host::step_with_manager(&mut execution, &executor, &mut manager).await {
        Err(GraphError::HistoryCompaction { source, .. }) => {
            assert!(source.is::<PreparationFailure>())
        }
        other => panic!("unexpected preparation outcome: {other:?}"),
    }
    assert!(factory.calls().is_empty());
    assert_eq!(
        serde_json::to_value(&before).expect("snapshot"),
        serde_json::to_value(execution.snapshot()?).expect("snapshot")
    );
    let flow = compile(workflow)?;
    let mut restored = flow.restore(before)?;
    let executor = flow
        .prepared()
        .executor(Context::default().with_providers(pravah::testing::providers(factory.clone())?));
    let mut manager = pravah::HistoryManager::new().with_compactor(Summarize);
    assert!(matches!(
        host::finish_with_manager(&mut restored, &executor, &mut manager).await?,
        Step::Done(_)
    ));
    assert_eq!(factory.calls().len(), 1);
    Ok(())
}

/// Invalid or protected indices do not change history or permit a model call.
#[tokio::test]
async fn runtime_rejects_protected_and_out_of_range_eviction() -> Result<(), GraphError> {
    for indices in [vec![0], vec![99], vec![0, 0]] {
        let factory = ScriptedFactory::new();
        let (mut execution, executor, mut manager) =
            before_dispatch(Invalid { indices }, &factory).await?;
        let before = serde_json::to_value(execution.snapshot()?).expect("snapshot");
        assert!(matches!(
            host::step_with_manager(&mut execution, &executor, &mut manager).await,
            Err(GraphError::HistoryCompactionValidation { .. })
        ));
        assert_eq!(
            serde_json::to_value(execution.snapshot()?).expect("snapshot"),
            before
        );
        assert!(factory.calls().is_empty());
    }
    Ok(())
}
