use super::*;
use pravah::testing::{ScriptedFactory, mock_tool_call};
use pravah::{AgentDecision, AgentLoop, AgentResume};

async fn control(input: AgentLoop<String>, _: Context) -> Result<AgentDecision, GraphError> {
    match input.input().as_str() {
        "suspend" => Ok(AgentDecision::suspend(Value::from("approval"))
            .with_state(Value::array([Value::from("retained")]))),
        "abort" => {
            Ok(AgentDecision::abort("policy rejected").with_state(Value::from("must not commit")))
        }
        "conclude" => Ok(AgentDecision::conclude("Answer now.")),
        _ => Ok(AgentDecision::continue_()),
    }
}

fn definition() -> pravah::ChatBuilder<String, String> {
    builder().control(control)
}

fn scripted_context(factory: &ScriptedFactory) -> Context {
    Context::default().with_providers(ProviderRegistry::with_builtin_factory(factory.clone()))
}

/// Drives external operations normally, stopping at the next visible turn boundary.
async fn drive(chat: &mut Chat<String, String>) -> Result<ChatStep<String>, GraphError> {
    loop {
        match chat.next()? {
            ChatStep::Continue => {}
            ChatStep::Agent(fetch) => {
                let response = chat.executor().execute(&fetch).await;
                chat.resume_agent(response)?;
            }
            step => return Ok(step),
        }
    }
}

/// Suspension retains the moved checkpoint and explicit resume advances without another controller.
#[tokio::test]
async fn controller_suspension_restores_and_resumes() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new().then_output(serde_json::json!("answer"));
    let mut chat = definition().build(scripted_context(&factory))?;
    chat.submit_with_key("suspend", "original-key")?;
    assert!(matches!(drive(&mut chat).await?, ChatStep::Suspend(_)));
    assert!(factory.calls().is_empty());
    chat = definition().restore(
        cbor_roundtrip(json_roundtrip(chat.snapshot()?)?)?,
        scripted_context(&factory),
    )?;
    chat.resume(AgentResume::Continue)?;
    let ChatStep::Done(turn) = drive(&mut chat).await? else {
        return Err(GraphError::Invalid("expected completed turn".into()));
    };
    assert_eq!(turn.output, "answer");
    assert_eq!(factory.calls().len(), 1);
    assert_eq!(
        chat.snapshot()?
            .history()
            .entries()
            .first()
            .and_then(|entry| entry.message.key.as_deref()),
        Some("original-key")
    );
    Ok(())
}

/// Failures after delivery leave the accepted outcome and its preceding checkpoint retryable.
#[tokio::test]
async fn failed_decisions_and_outputs_preserve_accepted_state() -> Result<(), GraphError> {
    for (input, output) in [
        ("abort", serde_json::json!("unused")),
        ("invalid", serde_json::json!(42)),
    ] {
        let factory = ScriptedFactory::new().then_output(output);
        let mut chat = definition().build(scripted_context(&factory))?;
        chat.submit(input)?;
        loop {
            let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
            match chat.next() {
                Ok(ChatStep::Continue) => {}
                Ok(ChatStep::Agent(fetch)) => {
                    let response = chat.executor().execute(&fetch).await;
                    chat.resume_agent(response)?;
                }
                Err(error) => {
                    assert!(matches!(
                        error,
                        GraphError::AgentPolicyAbort { .. } | GraphError::Schema { .. }
                    ));
                    assert_eq!(
                        before,
                        serde_json::to_value(chat.snapshot()?).map_err(codec)?
                    );
                    check_failed_restore(chat.snapshot()?, &factory)?;
                    break;
                }
                _ => return Err(GraphError::Invalid("unexpected turn boundary".into())),
            }
        }
        assert_eq!(factory.calls().len(), usize::from(input != "abort"));
    }
    Ok(())
}

/// Accepted failures remain local processing failures when runtime dependencies are replaced.
fn check_failed_restore(snapshot: Snapshot, factory: &ScriptedFactory) -> Result<(), GraphError> {
    let before = serde_json::to_value(&snapshot).map_err(codec)?;
    let mut chat =
        definition().restore::<()>(cbor_roundtrip(snapshot)?, scripted_context(factory))?;
    assert!(chat.pending_agent().is_none());
    assert!(chat.next().is_err());
    assert_eq!(
        before,
        serde_json::to_value(chat.snapshot()?).map_err(codec)?
    );
    Ok(())
}

/// Forced conclusion still rejects tool proposals, retaining the completed model operation.
#[tokio::test]
async fn conclusion_tool_failure_does_not_redispatch() -> Result<(), GraphError> {
    let factory = ScriptedFactory::new().then_tool_calls(vec![mock_tool_call(
        "call",
        "unknown",
        serde_json::json!({}),
    )]);
    let mut chat = definition().build(scripted_context(&factory))?;
    chat.submit("conclude")?;
    assert!(matches!(
        drive(&mut chat).await,
        Err(GraphError::AgentConclusion { .. })
    ));
    let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    chat = definition().restore(
        json_roundtrip(chat.snapshot()?)?,
        scripted_context(&factory),
    )?;
    assert!(matches!(
        chat.next(),
        Err(GraphError::AgentConclusion { .. })
    ));
    assert_eq!(
        before,
        serde_json::to_value(chat.snapshot()?).map_err(codec)?
    );
    assert_eq!(factory.calls().len(), 1);
    Ok(())
}
