use super::*;

/// Stale IDs, invalid protocol versions and malformed generation values never mutate the wait.
#[tokio::test]
async fn response_validation_and_pending_restore_remain_strict() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().build(context(&calls))?;
    chat.submit("question")?;
    let request = preparation::pending(&mut chat).await?;
    let before = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    for response in [
        AgentResponse::new(Uuid::nil(), Ok(Value::NULL)),
        AgentResponse::new(request.id(), Ok(Value::NULL)),
    ] {
        assert!(chat.resume_agent(response).is_err());
        assert_eq!(
            before,
            serde_json::to_value(chat.snapshot()?).map_err(codec)?
        );
    }
    let mut wire = serde_json::to_value(AgentResponse::new(
        request.id(),
        Err(AgentError::new("test", "offline")),
    ))
    .map_err(codec)?;
    wire["version"] = 0.into();
    assert!(
        chat.resume_agent(serde_json::from_value(wire).map_err(codec)?)
            .is_err()
    );
    assert_eq!(
        before,
        serde_json::to_value(chat.snapshot()?).map_err(codec)?
    );
    Ok(())
}

/// Delivery validates response structure without decoding or copying the pending request.
#[tokio::test]
async fn response_delivery_does_not_reconstruct_pending_request() -> Result<(), GraphError> {
    let mut counts = Vec::new();
    for size in [1, 100_000] {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut chat = builder().build(context(&calls))?;
        chat.submit("m".repeat(size))?;
        let request = preparation::pending(&mut chat).await?;
        let response = chat.executor().execute(&request).await;
        let mut result = Ok(());
        let allocations = allocation_counter::measure(|| result = chat.resume_agent(response));
        result?;
        counts.push((allocations.count_total, allocations.bytes_total));
    }
    assert_eq!(counts[0], counts[1]);
    Ok(())
}

/// Corrupt deterministic IDs and frame ownership fail restoration instead of repairing state.
#[test]
fn restore_rejects_malformed_pending_request() -> Result<(), GraphError> {
    let flow = compile(agent_flow)?;
    let mut runtime = flow.start("question".into(), Uuid::from_u128(7))?;
    next_agent(&mut runtime)?;
    let snapshot = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    for (key, value) in [
        ("frame_depth", serde_json::json!(0)),
        ("node", serde_json::json!(999)),
    ] {
        let mut invalid = snapshot.clone();
        invalid["state"]["waiting"][key] = value;
        assert!(
            flow.restore(serde_json::from_value(invalid).map_err(codec)?)
                .is_err()
        );
    }
    Ok(())
}

/// A request cannot substitute a different handler, definition, execution identity or invocation input on restore.
#[test]
fn restore_rejects_work_checkpoint_substitution() -> Result<(), GraphError> {
    let flow = compile(agent_flow)?;
    let mut runtime = flow.start("question".into(), Uuid::from_u128(7))?;
    next_agent(&mut runtime)?;
    let snapshot = serde_json::to_value(runtime.snapshot()?).map_err(codec)?;
    for (field, replacement) in [
        ("handler", serde_json::json!("another")),
        ("input", serde_json::json!("another")),
        ("execution_id", serde_json::json!(Uuid::nil())),
        ("definition", serde_json::Value::Null),
    ] {
        let mut invalid = snapshot.clone();
        invalid["state"]["waiting"]["request"][field] = replacement;
        assert!(
            flow.restore(serde_json::from_value(invalid).map_err(codec)?)
                .is_err()
        );
    }
    Ok(())
}

/// Generation cannot replace accepted history or alter frozen model options in a saved request.
#[tokio::test]
async fn restore_rejects_generation_input_substitution() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().build(context(&calls))?;
    chat.submit("question")?;
    preparation::pending(&mut chat).await?;
    let snapshot = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    for mutation in 0..3 {
        let mut invalid = snapshot.clone();
        let request = &mut invalid["state"]["waiting"]["request"];
        match mutation {
            0 => request["entries"][0]["message"]["content"] = serde_json::json!("substituted"),
            1 => request["model"] = serde_json::json!("test:///different"),
            _ => request["options"]["tool_choice"] = serde_json::json!("required"),
        }
        assert!(
            builder()
                .restore::<()>(
                    serde_json::from_value(invalid).map_err(codec)?,
                    context(&calls)
                )
                .is_err()
        );
    }
    Ok(())
}

/// Final persistence checkpoints cannot smuggle effects or replace the typed completion on restore.
#[tokio::test]
async fn restore_rejects_corrupted_final_flush() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder()
        .build(context(&calls))?
        .with_store(RejectAssistant);
    chat.submit("question")?;
    let snapshot = loop {
        match chat.next()? {
            ChatStep::Continue => {}
            ChatStep::Agent(request) if request.kind() == "persist_history" => {
                break chat.snapshot()?;
            }
            ChatStep::Agent(request) => {
                chat.resume_agent(chat.executor().execute(&request).await)?
            }
            _ => return Err(GraphError::Invalid("expected final persistence".into())),
        }
    };
    builder().restore::<()>(json_roundtrip(snapshot.clone())?, context(&calls))?;
    let wire = serde_json::to_value(snapshot).map_err(codec)?;
    for (field, replacement) in [
        ("outputs", serde_json::json!([42])),
        ("outputs", serde_json::json!([])),
        ("checkpoint", serde_json::json!(true)),
        ("state", serde_json::json!({})),
        ("history", serde_json::json!([{"Append": []}])),
    ] {
        let mut invalid = wire.clone();
        invalid["state"]["frames"][0]["checkpoints"][0]["value"]["transition"][field] = replacement;
        assert!(
            builder()
                .restore::<()>(
                    serde_json::from_value(invalid).map_err(codec)?,
                    context(&calls)
                )
                .is_err()
        );
    }
    Ok(())
}
