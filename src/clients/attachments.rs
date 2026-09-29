use base64::Engine;

use super::{Attachment, ClientError, ErrorKind, Message};
use crate::context::Context;

/// Resolves file attachments through the execution context before provider dispatch.
async fn materialize_attachment(
    attachment: Attachment,
    ctx: &Context,
) -> Result<Attachment, ClientError> {
    match attachment {
        attachment @ (Attachment::Inline { .. } | Attachment::Url { .. }) => Ok(attachment),
        Attachment::File { mime_type, path } => {
            let resolved = ctx.resolve(&path).map_err(|e| {
                ClientError::new(
                    ErrorKind::Validation,
                    format!("attachment path '{path}' is invalid: {e}"),
                )
            })?;
            let bytes = tokio::fs::read(&resolved).await.map_err(|e| {
                ClientError::new(
                    ErrorKind::Validation,
                    format!(
                        "failed to read attachment file '{}': {e}",
                        resolved.display()
                    ),
                )
            })?;
            Ok(Attachment::Inline {
                mime_type,
                data: base64::engine::general_purpose::STANDARD.encode(bytes),
            })
        }
        _ => Err(ClientError::new(
            ErrorKind::UnsupportedCapability,
            "unsupported attachment kind",
        )),
    }
}

/// Materializes each message's attachments without changing the original history.
pub(crate) async fn materialize_messages(
    messages: &[Message],
    ctx: &Context,
) -> Result<Vec<Message>, ClientError> {
    materialize_owned_messages(messages.to_vec(), ctx).await
}

/// Moves an owned request's text and buffered attachments; only file contents require new data.
pub(crate) async fn materialize_owned_messages(
    mut messages: Vec<Message>,
    ctx: &Context,
) -> Result<Vec<Message>, ClientError> {
    for message in &mut messages {
        let source = std::mem::take(&mut message.attachments);
        let mut attachments = Vec::with_capacity(source.len());
        for attachment in source {
            attachments.push(materialize_attachment(attachment, ctx).await?);
        }
        message.attachments = attachments;
    }
    Ok(messages)
}
