use super::*;
use pravah::{CompactionRequest, CompactionResult, Compactor};

fn revised(root: Agent<Question>) -> Agent<Notes> {
    root.model("test:///model")
        .instructions("Revised.")
        .key("research")
        .turn_budget(3)
        .max_output_tokens(500)
        .build()
}

fn revised_flow(root: Flow<Question>) -> Flow<Notes> {
    root.agent(revised)
}

struct ExpectInstructions(&'static str);
impl Compactor for ExpectInstructions {
    type Error = std::convert::Infallible;
    async fn compact(
        &self,
        request: CompactionRequest<'_>,
        _: Context,
    ) -> Result<CompactionResult, Self::Error> {
        assert!(
            request
                .options()
                .preamble
                .as_deref()
                .is_some_and(|text| text.contains(self.0))
        );
        Ok(CompactionResult::default())
    }
}

/// Instruction-only restores update future activation but preserve already committed configuration.
#[tokio::test]
async fn instructions_observe_configuration_boundary() -> Result<(), GraphError> {
    for committed in [false, true] {
        check_instructions(committed).await?;
    }
    Ok(())
}

/// Drives one restore boundary with a preparer observing the effective model instructions.
async fn check_instructions(committed: bool) -> Result<(), GraphError> {
    let original = compile(research)?;
    let mut runtime = original
        .start(
            Question {
                topic: "durability".into(),
            },
            Uuid::nil(),
        )?
        .with_history(pravah::HistoryPolicy {
            compact: true,
            ..Default::default()
        })?;
    let local = original.prepared().executor(Context::default());
    if committed {
        pending_generation(&mut runtime, &local).await?;
    }
    let revised = compile(revised_flow)?;
    let mut runtime = revised.restore(runtime.snapshot()?)?;
    let expected = if committed {
        "Research carefully."
    } else {
        "Revised."
    };
    let script = ScriptedFactory::new().then_output(serde_json::json!({"finding":"verified"}));
    let executor = revised
        .prepared()
        .executor(Context::default().with_providers(providers(script.clone())?))
        .with_compactor(ExpectInstructions(expected));
    assert!(matches!(
        host::finish(&mut runtime, &executor).await?,
        Step::Done(_)
    ));
    assert_eq!(script.calls().len(), 1);
    Ok(())
}

/// A pending declarative generation retains its request and history through CBOR and accepted-response restore.
#[tokio::test]
async fn cbor_restore_preserves_pending_and_accepted_generation() -> Result<(), GraphError> {
    let script = ScriptedFactory::new().then_output(serde_json::json!({"finding":"verified"}));
    let flow = compile(research)?;
    let executor = flow
        .prepared()
        .executor(Context::default().with_providers(providers(script.clone())?));
    let mut runtime = flow.start(
        Question {
            topic: "durability".into(),
        },
        Uuid::nil(),
    )?;
    let request = pending_generation(&mut runtime, &executor).await?;
    let mut encoded = Vec::new();
    ciborium::into_writer(&runtime.snapshot()?, &mut encoded).map_err(codec)?;
    let mut restored = flow.restore(ciborium::from_reader(encoded.as_slice()).map_err(codec)?)?;
    assert_eq!(
        restored.pending_agent().map(|request| request.id()),
        Some(request.id())
    );
    restored.resume_agent(executor.execute(&request).await)?;
    let mut restored = flow.restore(restored.snapshot()?)?;
    assert!(matches!(
        host::finish(&mut restored, &executor).await?,
        Step::Done(_)
    ));
    assert_eq!(script.calls().len(), 1);
    assert_eq!(restored.history().entries().len(), 2);
    Ok(())
}
