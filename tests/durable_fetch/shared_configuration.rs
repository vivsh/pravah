use super::*;
use pravah::{AgentDecision, AgentLoop, ChatRequest};

async fn control(_: AgentLoop<String>, _: Context) -> Result<AgentDecision, GraphError> {
    Ok(AgentDecision::continue_())
}

/// Stops immediately before the first controller boundary, after configuration and history commit.
async fn configured(memory: String) -> Result<Chat<String, String>, GraphError> {
    let mut chat = builder().control(control).build(Context::default())?;
    chat.submit(ChatRequest::from("question").memory(memory))?;
    for _ in 0..1 {
        loop {
            match chat.next()? {
                ChatStep::Continue => {}
                ChatStep::Fetch(fetch) => {
                    let response = chat.executor().execute(&fetch).await?;
                    chat.resume_fetch(fetch.id(), Ok(response))?;
                    break;
                }
                _ => return Err(GraphError::Invalid("expected setup hook".into())),
            }
        }
    }
    assert!(matches!(chat.next()?, ChatStep::Continue));
    Ok(chat)
}

/// Controller transitions do not allocate memory-sized copies of the resolved configuration.
#[tokio::test]
async fn controller_transition_shares_large_configuration() -> Result<(), GraphError> {
    let mut measured = Vec::new();
    for size in [1, 100_000] {
        let mut chat = configured("m".repeat(size)).await?;
        let mut result = None;
        let counts = allocation_counter::measure(|| result = Some(chat.next()));
        assert!(matches!(result, Some(Ok(ChatStep::Fetch(_)))));
        measured.push((counts.count_total, counts.bytes_total));
    }
    assert_eq!(measured.first(), measured.last());
    Ok(())
}

/// Finds the checkpoint's resolved configuration, not authored settings or history text.
fn corrupt_memory(value: &mut serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(fields) => {
            if let Some(config) = fields
                .get_mut("resolved")
                .and_then(serde_json::Value::as_object_mut)
            {
                config.insert("memory".into(), serde_json::json!(false));
                return true;
            }
            fields.values_mut().any(corrupt_memory)
        }
        serde_json::Value::Array(values) => values.iter_mut().any(corrupt_memory),
        _ => false,
    }
}

/// Skipping repeated internal decoding never skips complete configuration validation on restore.
#[tokio::test]
async fn restore_rejects_malformed_shared_configuration() -> Result<(), GraphError> {
    let chat = configured("memory".into()).await?;
    let mut snapshot = serde_json::to_value(chat.snapshot()?).map_err(codec)?;
    assert!(corrupt_memory(&mut snapshot));
    let snapshot = serde_json::from_value(snapshot).map_err(codec)?;
    assert!(
        builder()
            .control(control)
            .restore::<()>(snapshot, Context::default())
            .is_err()
    );
    Ok(())
}
