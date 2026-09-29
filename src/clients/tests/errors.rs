use crate::Context;
use crate::clients::{Attachment, ClientError, ErrorKind, Message, Provider, materialize_messages};

/// Rath diagnostics remain inspectable through Pravah's client namespace without a wrapper.
#[test]
fn structured_error_metadata_and_causes_are_accessible() {
    let error = ClientError::new(ErrorKind::Transport, "request failed")
        .with_context(Provider::OpenAi, "llm.execute")
        .with_source(ClientError::new(ErrorKind::Timeout, "deadline exceeded"));
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(error.provider(), Some(&Provider::OpenAi));
    assert_eq!(error.operation(), Some("llm.execute"));
    assert_eq!(
        error.source().map(ClientError::kind),
        Some(ErrorKind::Timeout)
    );
    assert!(error.response_body().is_none());
}

/// Local attachment failures use structured validation errors before any provider call.
#[tokio::test]
async fn invalid_attachment_paths_are_validation_errors() {
    let mut message = Message::user("question");
    message.attachments.push(Attachment::File {
        mime_type: "text/plain".into(),
        path: "../outside-context.txt".into(),
    });
    let result = materialize_messages(&[message], &Context::default()).await;
    assert!(matches!(result, Err(error) if error.kind() == ErrorKind::Validation));
}
