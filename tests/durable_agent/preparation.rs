use super::*;
use pravah::clients::Attachment;
use pravah::{Agent, AgentConfig, ChatRequest};

/// Executes only configuration, stopping before the coalesced generation worker.
pub(super) async fn pending(chat: &mut Chat<String, String>) -> Result<AgentRequest, GraphError> {
    loop {
        match chat.next()? {
            ChatStep::Continue => {}
            ChatStep::Agent(request) if request.kind() == "generate" => return Ok(request),
            ChatStep::Agent(request) => {
                chat.resume_agent(chat.executor().execute(&request).await)?
            }
            _ => return Err(codec("expected generation")),
        }
    }
}

/// The request carries resolved memory and exact keyed messages, with no extra preparation request.
#[tokio::test]
async fn preparation_preserves_messages_and_options() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().build(context(&calls))?;
    chat.send_with_key("prior", "old-key").await?;
    chat.submit_with_key(
        ChatRequest::from("pending").memory("m".repeat(100_000)),
        "new-key",
    )?;
    let request = pending(&mut chat).await?;
    let wire = serde_json::to_value(&request).map_err(codec)?;
    assert_eq!(wire["entries"].as_array().map(Vec::len), Some(3));
    assert_eq!(wire["entries"][2]["message"]["key"], "new-key");
    assert!(
        wire["options"]["preamble"]
            .as_str()
            .is_some_and(|text| text.contains(&"m".repeat(100_000)))
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

fn attachments(root: Agent<String>) -> Agent<String> {
    root.configure(configure_attachments)
}
async fn configure_attachments(path: String, _: Context) -> Result<AgentConfig, GraphError> {
    let mut message = Message::user("attachments");
    message.attachments = vec![
        Attachment::Inline {
            mime_type: "text/plain".into(),
            data: "aW5saW5l".into(),
        },
        Attachment::Url {
            mime_type: "image/png".into(),
            url: "https://example.invalid/image.png".into(),
        },
        Attachment::File {
            mime_type: "text/plain".into(),
            path,
        },
    ];
    Ok(AgentConfig::new("openai:///test", "test", message).key("conversation"))
}

/// Local attachments are frozen by activation before the pending generation is snapshotted.
#[tokio::test]
async fn file_materialization_survives_restore_without_rereading() -> Result<(), GraphError> {
    let directory = tempfile::tempdir().map_err(codec)?;
    let file = directory.path().join("message.txt");
    tokio::fs::write(&file, b"original").await.map_err(codec)?;
    let make_context = || {
        Context::new(pravah::FlowConf {
            working_dir: Some(directory.path().to_owned()),
            ..Default::default()
        })
    };
    let mut chat = Chat::new(attachments, make_context())?;
    chat.submit_with_key("message.txt", "attachment-key")?;
    let request = pending(&mut chat).await?;
    let snapshot = cbor_roundtrip(json_roundtrip(chat.snapshot()?)?)?;
    tokio::fs::write(&file, b"changed").await.map_err(codec)?;
    let restored = Chat::<String, String>::from_snapshot(attachments, snapshot, make_context())?;
    assert_eq!(
        restored.pending_agent().map(AgentRequest::id),
        Some(request.id())
    );
    let wire = serde_json::to_value(restored.pending_agent()).map_err(codec)?;
    let message: Message =
        serde_json::from_value(wire["entries"][0]["message"].clone()).map_err(codec)?;
    assert_eq!(message.key.as_deref(), Some("attachment-key"));
    assert!(
        matches!(message.attachments.last(), Some(Attachment::Inline { data, .. }) if data == "b3JpZ2luYWw=")
    );
    Ok(())
}
