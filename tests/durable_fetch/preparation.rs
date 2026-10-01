use super::*;
use pravah::clients::Attachment;
use pravah::graph::{FetchBody, fetch::rath::RathRequest};
use pravah::{Agent, AgentConfig, ChatRequest};

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, GraphError> {
    value
        .get(name)
        .ok_or_else(|| GraphError::Invalid(format!("missing {name}")))
}

fn body(value: Option<&FetchBody>) -> Result<&Value, GraphError> {
    match value {
        Some(FetchBody::Value(value)) => Ok(value),
        _ => Err(GraphError::Invalid("missing structured body".into())),
    }
}

/// Advances only setup hooks, stopping before preparation executes or a model is called.
pub(super) async fn pending(chat: &mut Chat<String, String>) -> Result<Fetch, GraphError> {
    loop {
        match chat.next()? {
            ChatStep::Continue => {}
            ChatStep::Fetch(fetch)
                if matches!(
                    fetch.request().url(),
                    "pravah://agent-prepare" | "rath://generate"
                ) =>
            {
                return Ok(fetch);
            }
            ChatStep::Fetch(fetch) => {
                let response = chat.executor().execute(&fetch).await?;
                chat.resume_fetch(fetch.id(), Ok(response))?;
            }
            _ => return Err(GraphError::Invalid("expected preparation".into())),
        }
    }
}

/// Preparation shares options and retained messages while preserving the ordinary Rath wire shape.
#[tokio::test]
async fn preparation_shares_validated_messages_and_options() -> Result<(), GraphError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut chat = builder().build(context(&calls))?;
    chat.send_with_key("prior", "old-key").await?;
    chat.submit_with_key(
        ChatRequest::from("pending").memory("m".repeat(100_000)),
        "new-key",
    )?;
    let fetch = pending(&mut chat).await?;
    let source = body(fetch.request().body_ref())?;
    let generation = source;
    let old_options = field(source, "options")?;
    assert!(std::ptr::eq(
        field(old_options, "preamble")?,
        field(field(generation, "options")?, "preamble")?
    ));
    let entries = field(source, "messages")?
        .as_array()
        .ok_or_else(|| codec("entries"))?;
    let messages = field(generation, "messages")?
        .as_array()
        .ok_or_else(|| codec("messages"))?;
    assert_eq!(entries.len(), messages.len());
    for (entry, message) in entries.iter().zip(messages) {
        assert!(std::ptr::eq(
            field(entry, "content")?,
            field(message, "content")?
        ));
    }
    let request = fetch.request().clone();
    let decoded = RathRequest::from_fetch_request(&request)?;
    let normal = RathRequest::new(
        decoded.model(),
        decoded.options().clone(),
        decoded.messages().to_vec(),
    )
    .into_fetch_request()?;
    assert_eq!(request, normal);
    assert_eq!(
        decoded
            .messages()
            .last()
            .and_then(|message| message.key.as_deref()),
        Some("new-key")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

fn attachments(root: Agent<String>) -> Agent<String> {
    root.configure(configure_attachments)
}

/// Supplies each attachment kind without reading files during configuration.
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

/// Files materialize once; restored generation retains frozen bytes and original history metadata.
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
    let fetch = pending(&mut chat).await?;
    let reply = chat.executor().execute(&fetch).await?;
    chat.resume_fetch(fetch.id(), Ok(reply))?;
    let ChatStep::Fetch(fetch) = chat.next()? else {
        return Err(codec("generation missing"));
    };
    assert_eq!(fetch.request().url(), "rath://generate");
    let snapshot = cbor_roundtrip(json_roundtrip(chat.snapshot()?)?)?;
    tokio::fs::write(&file, b"changed").await.map_err(codec)?;
    let restored = Chat::<String, String>::from_snapshot(attachments, snapshot, make_context())?;
    let request = restored
        .pending_fetch()
        .ok_or_else(|| codec("pending request"))?;
    let decoded = RathRequest::from_fetch_request(request.request())?;
    let message = decoded.messages().last().ok_or_else(|| codec("message"))?;
    assert_eq!(message.key.as_deref(), Some("attachment-key"));
    assert!(
        matches!(message.attachments.last(), Some(Attachment::Inline { data, .. }) if data == "b3JpZ2luYWw=")
    );
    let snapshot = restored.snapshot()?;
    let history = snapshot
        .history()
        .entries()
        .last()
        .ok_or_else(|| codec("history"))?;
    assert!(
        matches!(history.message.attachments.last(), Some(Attachment::File { path, .. }) if path == "message.txt")
    );
    Ok(())
}
